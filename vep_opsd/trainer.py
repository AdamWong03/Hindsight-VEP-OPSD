from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch
from transformers import Trainer

from .artifacts import RolloutWriter
from .config import VEPPrivilegedOPSDConfig
from .objective import (
    generalized_jsd_from_log_probs,
    generated_token_logits,
    sampled_token_log_probs,
    student_distribution_on_teacher_support,
    teacher_topk_distribution,
    truncate_generated_ids,
)
from .prompts import extract_final_answer, privileged_teacher_messages, student_messages


def _model_device(model):
    return next(model.parameters()).device


def _to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _model_inputs(processor, messages, images, device):
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = processor(
        text=[rendered], images=list(images), padding=True, return_tensors="pt"
    )
    return _to_device(dict(encoded), device)


def _append_trajectory(prompt: dict[str, Any], token_ids: list[int], device):
    generated = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
    result = dict(prompt)
    result["input_ids"] = torch.cat([prompt["input_ids"], generated], dim=1)
    result["attention_mask"] = torch.cat(
        [
            prompt["attention_mask"],
            torch.ones_like(generated, dtype=prompt["attention_mask"].dtype),
        ],
        dim=1,
    )
    if "token_type_ids" in result:
        suffix = result["token_type_ids"][:, -1:].expand(-1, generated.shape[1])
        result["token_type_ids"] = torch.cat(
            [result["token_type_ids"], suffix], dim=1
        )
    # The final G+1 positions begin at the prompt token that predicts y_1.
    result["logits_to_keep"] = generated.shape[1] + 1
    return result, generated.squeeze(0)


@contextmanager
def _inference_mode(model):
    was_training = model.training
    old_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_cache is not None:
        model.config.use_cache = True
    try:
        with torch.no_grad():
            yield
    finally:
        if old_cache is not None:
            model.config.use_cache = old_cache
        model.train(was_training)


@contextmanager
def _fixed_base_teacher(model):
    """Run the frozen initialization by temporarily disabling Student LoRA."""

    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        raise RuntimeError(
            "Privileged OPSD requires PEFT: the fixed Teacher is the base model "
            "with the trainable Student adapter disabled"
        )
    with disable_adapter():
        with _inference_mode(model):
            yield


class VEPPrivilegedOPSDTrainer(Trainer):
    """On-policy token distillation from a future-and-Gold privileged Teacher."""

    def __init__(
        self, *args, processor, opsd_config: VEPPrivilegedOPSDConfig, **kwargs
    ):
        super().__init__(*args, processing_class=processor, **kwargs)
        opsd_config.validate()
        self.processor = processor
        self.opsd_config = opsd_config
        self.rollout_writer = RolloutWriter(self.args.output_dir)

    def _base_model(self, model):
        return self.accelerator.unwrap_model(model)

    def _generate(self, base_model, examples):
        """Sample one current-policy trajectory per example from past frames only."""

        device = _model_device(base_model)
        prompts, images = [], []
        for example in examples:
            messages = student_messages(example["observed_frames"], example["question"])
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            images.extend(example["observed_frames"])

        self.processor.tokenizer.padding_side = "left"
        try:
            encoded = self.processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            )
        finally:
            self.processor.tokenizer.padding_side = "right"
        prompt = _to_device(dict(encoded), device)
        prompt_length = int(prompt["input_ids"].shape[1])

        eos = base_model.generation_config.eos_token_id
        eos_ids = {
            int(value)
            for value in (eos if isinstance(eos, (list, tuple)) else [eos])
            if value is not None
        }
        pad_id = self.processor.tokenizer.pad_token_id
        sequences = base_model.generate(
            **prompt,
            do_sample=True,
            temperature=self.opsd_config.sampling_temperature,
            top_p=self.opsd_config.sampling_top_p,
            num_return_sequences=1,
            max_new_tokens=self.opsd_config.max_new_tokens,
            use_cache=True,
            pad_token_id=pad_id,
        )
        trajectories = []
        for sequence in sequences:
            ids = truncate_generated_ids(
                sequence[prompt_length:].tolist(), eos_ids, pad_id
            )
            text = self.processor.tokenizer.decode(
                ids, skip_special_tokens=False
            ).strip()
            trajectories.append(
                {
                    "token_ids": ids,
                    "length": len(ids),
                    "text": text,
                    "answer": extract_final_answer(text),
                }
            )
        return trajectories

    def _student_prompt(self, model, example):
        return _model_inputs(
            self.processor,
            student_messages(example["observed_frames"], example["question"]),
            example["observed_frames"],
            _model_device(model),
        )

    def _teacher_prompt(self, model, example):
        images = list(example["observed_frames"]) + list(example["future_frames"])
        return _model_inputs(
            self.processor,
            privileged_teacher_messages(
                example["observed_frames"],
                example["future_frames"],
                example["question"],
                example["reference_answer"],
            ),
            images,
            _model_device(model),
        )

    def _teacher_target(self, base_model, example, trajectory):
        prompt = self._teacher_prompt(base_model, example)
        inputs, generated = _append_trajectory(
            prompt, trajectory["token_ids"], _model_device(base_model)
        )
        with _fixed_base_teacher(base_model):
            outputs = base_model(**inputs, use_cache=False, return_dict=True)
        logits = generated_token_logits(outputs.logits, generated.numel())
        target = teacher_topk_distribution(
            logits,
            top_k=self.opsd_config.distillation_top_k,
            temperature=self.opsd_config.distillation_temperature,
            add_tail_bucket=self.opsd_config.add_tail_bucket,
        )
        return target, sampled_token_log_probs(logits, generated)

    def _student_loss(self, model, example, trajectory, teacher_target):
        prompt = self._student_prompt(model, example)
        inputs, generated = _append_trajectory(
            prompt, trajectory["token_ids"], _model_device(model)
        )
        outputs = model(**inputs, use_cache=False, return_dict=True)
        logits = generated_token_logits(outputs.logits, generated.numel())
        student_log_probs = student_distribution_on_teacher_support(
            logits,
            teacher_target,
            temperature=self.opsd_config.distillation_temperature,
            add_tail_bucket=self.opsd_config.add_tail_bucket,
        )
        return generalized_jsd_from_log_probs(
            teacher_target.log_probs,
            student_log_probs,
            beta=self.opsd_config.jsd_beta,
            temperature=self.opsd_config.distillation_temperature,
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        examples = inputs.get("examples")
        if not examples:
            raise ValueError("VEP-OPSD batch contains no examples")
        base_model = self._base_model(model)

        started = time.perf_counter()
        with _inference_mode(base_model):
            trajectories = self._generate(base_model, examples)
        rollout_seconds = (time.perf_counter() - started) / len(examples)

        losses, records = [], []
        for example, trajectory in zip(examples, trajectories):
            started = time.perf_counter()
            teacher_target, teacher_sampled_logp = self._teacher_target(
                base_model, example, trajectory
            )
            teacher_seconds = time.perf_counter() - started

            started = time.perf_counter()
            loss, per_token_loss = self._student_loss(
                model, example, trajectory, teacher_target
            )
            student_seconds = time.perf_counter() - started
            losses.append(loss)
            records.append(
                {
                    "global_step": int(self.state.global_step),
                    "sample_id": example["sample_id"],
                    "question": example["question"],
                    "observed_indices": example["observed_indices"],
                    "future_indices": example["future_indices"],
                    "reference_answer": example["reference_answer"],
                    "trajectory": trajectory,
                    "teacher_sampled_token_logp_mean": float(
                        teacher_sampled_logp.mean().item()
                    ),
                    "token_jsd_mean": float(per_token_loss.detach().mean().item()),
                    "token_jsd_max": float(per_token_loss.detach().max().item()),
                    "stage_seconds": {
                        "rollout": round(rollout_seconds, 3),
                        "teacher": round(teacher_seconds, 3),
                        "student": round(student_seconds, 3),
                    },
                }
            )

        loss = torch.stack(losses).mean()
        for record in records:
            record["loss"] = float(loss.detach().item())
            self.rollout_writer.append(record)
        outputs: dict[str, Any] = {"loss": loss, "records": records}
        return (loss, outputs) if return_outputs else loss

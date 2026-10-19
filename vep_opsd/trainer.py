from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from transformers import Trainer

from .artifacts import RolloutWriter
from .config import HindsightOPSDConfig
from .objective import (
    BinaryTokenSpec,
    binary_yes_probability,
    length_normalized_log_probability,
    listwise_cross_entropy,
    resolve_yes_no_tokens,
    teacher_preference,
    truncate_generated_ids,
)
from .prompts import extract_final_answer, student_messages, verifier_messages


def _model_device(model):
    return next(model.parameters()).device


def _to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _model_inputs(processor, messages, images, device):
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[rendered], images=list(images), padding=True, return_tensors="pt")
    return rendered, _to_device(dict(encoded), device)


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


class HindsightOPSDTrainer(Trainer):
    """Generate -> future-grounded verify -> listwise distill."""

    def __init__(self, *args, processor, opsd_config: HindsightOPSDConfig, **kwargs):
        super().__init__(*args, processing_class=processor, **kwargs)
        opsd_config.validate()
        self.processor = processor
        self.opsd_config = opsd_config
        self.rollout_writer = RolloutWriter(self.args.output_dir)
        self.binary_spec: BinaryTokenSpec | None = None

    def _base_model(self, model):
        return self.accelerator.unwrap_model(model)

    def _sample_candidates(self, base_model, example):
        device = _model_device(base_model)
        messages = student_messages(example["observed_frames"], example["question"])
        _, prompt = _model_inputs(self.processor, messages, example["observed_frames"], device)
        prompt_len = int(prompt["input_ids"].shape[1])
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
            num_return_sequences=self.opsd_config.candidates_per_example,
            max_new_tokens=self.opsd_config.max_new_tokens,
            use_cache=True,
        )
        candidates = []
        for sequence in sequences:
            ids = truncate_generated_ids(sequence[prompt_len:].tolist(), eos_ids, pad_id)
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=False).strip()
            candidates.append({"token_ids": ids, "text": text, "answer": extract_final_answer(text)})
        return candidates

    def _verify_candidate(self, base_model, example, candidate):
        device = _model_device(base_model)
        messages = verifier_messages(
            example["observed_frames"],
            example["future_frames"],
            example["question"],
            candidate["text"],
        )
        images = list(example["observed_frames"]) + list(example["future_frames"])
        rendered, inputs = _model_inputs(self.processor, messages, images, device)
        if self.binary_spec is None:
            self.binary_spec = resolve_yes_no_tokens(self.processor.tokenizer, rendered)
        outputs = base_model(**inputs, use_cache=False, return_dict=True)
        last = int(inputs["attention_mask"][0].sum().item()) - 1
        score = binary_yes_probability(outputs.logits[:, last, :], self.binary_spec)[0]
        return float(score.item())

    def _student_score(self, model, example, candidate):
        device = _model_device(model)
        messages = student_messages(example["observed_frames"], example["question"])
        _, prompt = _model_inputs(self.processor, messages, example["observed_frames"], device)
        prompt_len = int(prompt["input_ids"].shape[1])
        candidate_ids = torch.tensor(candidate["token_ids"], device=device, dtype=torch.long)
        input_ids = torch.cat([prompt["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
        attention = torch.cat(
            [
                prompt["attention_mask"],
                torch.ones(
                    (1, candidate_ids.numel()),
                    device=device,
                    dtype=prompt["attention_mask"].dtype,
                ),
            ],
            dim=1,
        )
        forward_inputs = dict(prompt)
        forward_inputs["input_ids"] = input_ids
        forward_inputs["attention_mask"] = attention
        if "token_type_ids" in forward_inputs:
            suffix = forward_inputs["token_type_ids"][:, -1:].expand(-1, candidate_ids.numel())
            forward_inputs["token_type_ids"] = torch.cat(
                [forward_inputs["token_type_ids"], suffix], dim=1
            )
        outputs = model(**forward_inputs, use_cache=False, return_dict=True)
        return length_normalized_log_probability(outputs.logits, prompt_len, candidate_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        examples = inputs.get("examples")
        if not examples:
            raise ValueError("VEP-OPSD batch contains no examples")
        device = _model_device(model)
        base_model = self._base_model(model)
        group_losses = []
        records = []
        for example in examples:
            with _inference_mode(base_model):
                candidates = self._sample_candidates(base_model, example)
                consistency = [
                    self._verify_candidate(base_model, example, candidate)
                    for candidate in candidates
                ]
            teacher_probs, skipped = teacher_preference(
                torch.tensor(consistency, device=device),
                self.opsd_config.teacher_temperature,
                self.opsd_config.minimum_consistency,
            )
            student_scores = torch.stack(
                [self._student_score(model, example, candidate) for candidate in candidates]
            )
            group_loss = (
                student_scores.sum() * 0.0
                if skipped
                else listwise_cross_entropy(
                    teacher_probs,
                    student_scores,
                    self.opsd_config.student_temperature,
                )
            )
            group_losses.append(group_loss)
            records.append(
                {
                    "global_step": int(self.state.global_step),
                    "sample_id": example["sample_id"],
                    "observed_indices": example["observed_indices"],
                    "future_indices": example["future_indices"],
                    "reference_answer": example["reference_answer"],
                    "skipped": skipped,
                    "teacher_scores": consistency,
                    "teacher_preference": teacher_probs.detach().cpu().tolist(),
                    "student_scores": student_scores.detach().float().cpu().tolist(),
                    "yes_no_tokens": self.binary_spec.to_dict() if self.binary_spec else None,
                    "candidates": candidates,
                }
            )
        loss = torch.stack(group_losses).mean()
        for record in records:
            record["loss"] = float(loss.detach().item())
            self.rollout_writer.append(record)
        outputs: dict[str, Any] = {"loss": loss, "records": records}
        return (loss, outputs) if return_outputs else loss


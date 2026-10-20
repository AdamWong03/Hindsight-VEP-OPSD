from __future__ import annotations

import time
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
from .prompts import (
    extract_final_answer,
    is_degenerate_answer,
    student_messages,
    verifier_messages,
)


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

    def _generate(self, base_model, examples, num_return_sequences):
        """One generate() call for the given examples, `num_return_sequences` each.

        Batching the samples together means the 8B weights are read once per decode
        step for all of them, which is what the rollout cost is bound by.
        """
        device = _model_device(base_model)
        prompts, images_per_row = [], []
        for example in examples:
            messages = student_messages(example["observed_frames"], example["question"])
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            images_per_row.append(list(example["observed_frames"]))
        images = [image for row in images_per_row for image in row]

        # Generated tokens are appended on the right, so unequal prompts have to
        # be left padded.
        self.processor.tokenizer.padding_side = "left"
        try:
            encoded = self.processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            )
        finally:
            self.processor.tokenizer.padding_side = "right"
        prompt = _to_device(dict(encoded), device)
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
            num_return_sequences=num_return_sequences,
            max_new_tokens=self.opsd_config.max_new_tokens,
            use_cache=True,
            pad_token_id=pad_id,
        )
        grouped = []
        for index in range(len(examples)):
            candidates = []
            for sequence in sequences[index * num_return_sequences : (index + 1) * num_return_sequences]:
                ids = truncate_generated_ids(sequence[prompt_len:].tolist(), eos_ids, pad_id)
                text = self.processor.tokenizer.decode(ids, skip_special_tokens=False).strip()
                candidates.append(
                    {"token_ids": ids, "text": text, "answer": extract_final_answer(text)}
                )
            grouped.append(candidates)
        return grouped

    def _sample_candidates(self, base_model, examples):
        """Sample M candidates per example, replacing candidates that refuse to predict.

        A non-answer is vacuously consistent with any future, so leaving one in the
        group rewards abstention; the collapsed run in this repo ended with 99.6%
        of candidates being "cannot be determined". Degenerate candidates are
        discarded and resampled, and an example left with fewer than M usable
        candidates is reported so the group can be skipped.
        """
        wanted = self.opsd_config.candidates_per_example
        groups = self._generate(base_model, examples, wanted)
        info = [
            {"degenerate": 0, "resampled": False, "usable": wanted} for _ in examples
        ]
        if not self.opsd_config.reject_degenerate_candidates:
            return groups, info

        for slot, group in enumerate(groups):
            info[slot]["degenerate"] = sum(
                1 for candidate in group if is_degenerate_answer(candidate["answer"])
            )

        for _ in range(self.opsd_config.max_resample_rounds):
            short = [
                index
                for index, group in enumerate(groups)
                if sum(1 for candidate in group if not is_degenerate_answer(candidate["answer"]))
                < wanted
            ]
            if not short:
                break
            needed = max(
                wanted
                - sum(1 for candidate in groups[index] if not is_degenerate_answer(candidate["answer"]))
                for index in short
            )
            fresh = self._generate(base_model, [examples[index] for index in short], needed)
            for position, index in enumerate(short):
                keep = [c for c in groups[index] if not is_degenerate_answer(c["answer"])]
                for candidate in fresh[position]:
                    if len(keep) >= wanted:
                        break
                    if not is_degenerate_answer(candidate["answer"]):
                        keep.append(candidate)
                    else:
                        info[index]["degenerate"] += 1
                groups[index] = keep
                info[index]["resampled"] = True

        for slot, group in enumerate(groups):
            info[slot]["usable"] = len(group)
        return groups, info

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
        # Only the final position is scored; without this the language-model head
        # projects the whole prompt (~590 positions x 151936) and throws it away.
        inputs["logits_to_keep"] = 1
        outputs = base_model(**inputs, use_cache=False, return_dict=True)
        score = binary_yes_probability(outputs.logits[:, -1, :], self.binary_spec)[0]
        return float(score.item())

    def _student_prompt(self, model, example):
        """Encode the student prompt once per example; every candidate shares it."""
        device = _model_device(model)
        messages = student_messages(example["observed_frames"], example["question"])
        _, prompt = _model_inputs(self.processor, messages, example["observed_frames"], device)
        return prompt, int(prompt["input_ids"].shape[1])

    def _student_score(self, model, prompt, prompt_len, candidate):
        device = _model_device(model)
        candidate_ids = torch.tensor(candidate["token_ids"], device=device, dtype=torch.long)
        input_ids = torch.cat([prompt["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
        if int(prompt["input_ids"].shape[1]) != prompt_len:
            raise ValueError("student prompt length drifted between encoding and scoring")
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
        # Only the generated tokens are scored, so the language-model head does
        # not need to project the prompt positions. ``G + 1`` rows are kept
        # because row ``P - 1`` predicts the first generated token.
        forward_inputs["logits_to_keep"] = int(candidate_ids.numel()) + 1
        outputs = model(**forward_inputs, use_cache=False, return_dict=True)
        # The kept window is [P-1, P+G-1]; the first G rows are the predictions.
        return length_normalized_log_probability(outputs.logits, 1, candidate_ids)

    def _empty_group_zero(self, model, example):
        """Zero loss built from an ordinary student forward.

        A micro batch whose groups were all skipped has no gradient of its own.
        The zero term must not be built from a *different* kind of graph (a sum
        over parameters, for instance): ZeRO-2 reduce-scatters a parameter bucket
        as soon as its gradient is ready, so a different graph changes the order
        of those collectives and the ranks deadlock at the first boundary that
        mixes them (observed twice: ALLREDUCE timeout on the first optimizer
        step). Scoring one sentinel token reuses the exact code path of a scored
        candidate, so the graph, and therefore the collective order, is identical
        to what the other ranks produce.
        """
        prompt, prompt_len = self._student_prompt(model, example)
        sentinel = {"token_ids": [int(self.processor.tokenizer.eos_token_id)]}
        return self._student_score(model, prompt, prompt_len, sentinel) * 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        examples = inputs.get("examples")
        if not examples:
            raise ValueError("VEP-OPSD batch contains no examples")
        device = _model_device(model)
        base_model = self._base_model(model)
        group_losses = []
        records = []
        started = time.perf_counter()
        with _inference_mode(base_model):
            candidates_per_example, sample_info = self._sample_candidates(base_model, examples)
        rollout_seconds = (time.perf_counter() - started) / max(len(examples), 1)
        for example, candidates, info in zip(examples, candidates_per_example, sample_info):
            stage_times = {"rollout": rollout_seconds}
            wanted = self.opsd_config.candidates_per_example
            skip_reason = None
            consistency, teacher_probs, student_scores = [], None, None

            if len(candidates) < wanted:
                # Not enough committed predictions to form a listwise target.
                skip_reason = "degenerate_candidates"
            else:
                started = time.perf_counter()
                with _inference_mode(base_model):
                    consistency = [
                        self._verify_candidate(base_model, example, candidate)
                        for candidate in candidates
                    ]
                stage_times["verify"] = time.perf_counter() - started

            spread = max(consistency) - min(consistency) if consistency else 0.0
            if skip_reason is None:
                scores = torch.tensor(consistency, device=device)
                teacher_probs, all_low = teacher_preference(
                    scores,
                    self.opsd_config.teacher_temperature,
                    self.opsd_config.minimum_consistency,
                )
                if all_low:
                    skip_reason = "low_consistency"
                elif spread < self.opsd_config.minimum_teacher_spread:
                    # A uniform teacher ranks nothing: the objective is already at
                    # its minimum and the gradient is noise.
                    skip_reason = "flat_teacher"

            started = time.perf_counter()
            if skip_reason is not None:
                # A skipped group contributes exactly zero, so its student passes
                # are pure cost.
                student_scores = torch.zeros(wanted, device=device, dtype=torch.float32)
                group_loss = student_scores.sum() * 0.0
            else:
                # The student prompt is identical for all four candidates, so it
                # is encoded once instead of once per candidate.
                prompt, prompt_len = self._student_prompt(model, example)
                student_scores = torch.stack(
                    [
                        self._student_score(model, prompt, prompt_len, candidate)
                        for candidate in candidates
                    ]
                )
                group_loss = listwise_cross_entropy(
                    teacher_probs,
                    student_scores,
                    self.opsd_config.student_temperature,
                )
            stage_times["student"] = time.perf_counter() - started
            group_losses.append(group_loss)
            records.append(
                {
                    "global_step": int(self.state.global_step),
                    "sample_id": example["sample_id"],
                    "observed_indices": example["observed_indices"],
                    "future_indices": example["future_indices"],
                    "reference_answer": example["reference_answer"],
                    "skipped": skip_reason is not None,
                    "skip_reason": skip_reason,
                    "teacher_spread": round(spread, 4),
                    "degenerate_candidates": info["degenerate"],
                    "resampled": info["resampled"],
                    "teacher_scores": consistency,
                    "teacher_preference": (
                        teacher_probs.detach().cpu().tolist() if teacher_probs is not None else None
                    ),
                    "student_scores": (
                        student_scores.detach().float().cpu().tolist()
                        if student_scores is not None
                        else [0.0] * wanted
                    ),
                    "yes_no_tokens": self.binary_spec.to_dict() if self.binary_spec else None,
                    "stage_seconds": {key: round(value, 3) for key, value in stage_times.items()},
                    "candidates": candidates,
                }
            )
        has_signal = any(record["skip_reason"] is None for record in records)
        for record in records:
            record["micro_batch_has_signal"] = has_signal
        loss = torch.stack(group_losses).mean()
        if not loss.requires_grad:
            loss = loss + self._empty_group_zero(model, examples[0])
        for record in records:
            record["loss"] = float(loss.detach().item())
            self.rollout_writer.append(record)
        outputs: dict[str, Any] = {"loss": loss, "records": records}
        return (loss, outputs) if return_outputs else loss

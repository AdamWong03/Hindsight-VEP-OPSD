"""Train text-token EIPD from the original Qwen3-VL checkpoint.

The trainable Student sees Past frames only. A frozen copy of the same base
checkpoint is replayed under Past and Past+Bridge conditions. In LoRA mode the
copy is implemented by disabling the Student adapter. In full-finetuning mode
it is a separate frozen model, so the Teacher remains the original checkpoint
while all language-model Student parameters are updated. All three branches
are teacher-forced on the Student's sampled reasoning prefix followed by the
dataset Gold Answer. No Future-L1 checkpoint or continuous latent token is
used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .eipd_objective import eipd_loss
from .trainer import drop_empty_param_groups, sync_scheduler_lambdas


def log_stage(message: str):
    """Emit one unbuffered startup marker without duplicating it on every rank."""
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[EIPD startup] {message}", flush=True)


class ManifestDataset(Dataset):
    """Read portable PNG manifests produced by eipd_final_data."""

    def __init__(self, manifest: str):
        path = Path(manifest).expanduser().resolve()
        self.root = path.parent
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError("Empty EIPD manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = dict(self.rows[index])
        if not all(key in row for key in ("past_files", "bridge_file", "future_files")):
            raise ValueError("Requires a finalized EIPD cached-frame manifest")
        if not max(row["past_native_indices"]) < row["bridge_native_index"] < min(
            row["future_native_indices"]
        ):
            raise ValueError(f"Invalid Bridge chronology: {row['sample_id']}")

        def checked_path(relative):
            target = (self.root / relative).resolve()
            if not target.is_relative_to(self.root):
                raise ValueError(f"Unsafe cached frame path: {relative}")
            if not target.is_file():
                raise FileNotFoundError(target)
            return target

        def rgb(relative):
            with Image.open(checked_path(relative)) as image:
                return image.convert("RGB").copy()

        return {
            "row": row,
            "observed": [rgb(value) for value in row["past_files"]],
            "bridge": rgb(row["bridge_file"]),
        }


def collate(items):
    if len(items) != 1:
        raise ValueError("EIPD currently requires per-device batch size 1")
    return items[0]


def messages(observed, question, bridge=None):
    content = [{"type": "text", "text": "Observed frames in temporal order:"}]
    content.extend({"type": "image", "image": image} for image in observed)
    if bridge is not None:
        content.extend(
            [
                {
                    "type": "text",
                    "text": "Additional intermediate future observation available only to the training teacher:",
                },
                {"type": "image", "image": bridge},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Question: "
                + question
                + "\nPredict the subsequent event. Reason briefly, then write exactly one concise "
                "prediction inside <answer>...</answer>. Never refuse to predict."
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You predict future events from video. Do not claim to have observed future "
                "frames, and do not answer with a yes/no judgment."
            ),
        },
        {"role": "user", "content": content},
    ]


def encode_prompt(processor, observed, question, device, bridge=None):
    conversation = messages(observed, question, bridge)
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    images = [*observed] + ([bridge] if bridge is not None else [])
    batch = processor(text=[prompt], images=images, padding=False, return_tensors="pt")
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def answer_prefix_ids(tokenizer, sampled_ids: list[int]) -> list[int]:
    """Return the on-policy prefix through the first complete <answer> tag."""
    text = tokenizer.decode(sampled_ids, skip_special_tokens=False)
    match = re.search(r"<answer\s*>", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("Student rollout did not reach an <answer> boundary")
    # Decode-prefix matching avoids assuming that <answer> is one tokenizer ID.
    for end in range(1, len(sampled_ids) + 1):
        if len(tokenizer.decode(sampled_ids[:end], skip_special_tokens=False)) >= match.end():
            return sampled_ids[:end]
    raise ValueError("Unable to align sampled <answer> boundary to token IDs")


def _append_continuation(prompt, continuation):
    result = dict(prompt)
    result["input_ids"] = torch.cat([prompt["input_ids"], continuation], dim=1)
    result["attention_mask"] = torch.cat(
        [
            prompt["attention_mask"],
            torch.ones_like(continuation, dtype=prompt["attention_mask"].dtype),
        ],
        dim=1,
    )
    if "token_type_ids" in result:
        suffix = result["token_type_ids"][:, -1:].expand(-1, continuation.shape[1])
        result["token_type_ids"] = torch.cat([result["token_type_ids"], suffix], dim=1)
    return result


def _answer_logits(model, prompt, continuation, answer_tokens):
    """Return logits that predict the final ``answer_tokens`` continuation IDs."""
    if answer_tokens <= 0 or continuation.shape[1] <= answer_tokens:
        raise ValueError("Answer span requires a nonempty prefix and target")
    batch = _append_continuation(prompt, continuation)
    # Avoid materializing vocabulary logits for the whole multimodal prompt.
    batch["logits_to_keep"] = answer_tokens + 1
    output = model(**batch, use_cache=False, return_dict=True)
    logits = output.logits[0]
    if logits.shape[0] != answer_tokens + 1:
        raise RuntimeError(
            f"Unexpected logits_to_keep result: {logits.shape[0]} rows for {answer_tokens} answer tokens"
        )
    return logits[:-1]


@contextmanager
def frozen_teacher(model, *, disable_adapter: bool):
    """Run an immutable Teacher, optionally disabling the Student LoRA."""
    was_training = model.training
    old_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = False
    adapter_context = model.disable_adapter() if disable_adapter else nullcontext()
    with adapter_context:
        with torch.no_grad():
            try:
                yield
            finally:
                model.config.use_cache = old_cache
                model.train(was_training)


class EIPDTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor,
        alpha: float,
        eipd_lambda: float,
        max_new_tokens: int,
        max_answer_tokens: int = 128,
        teacher_model=None,
        teacher_device: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, processing_class=processor, **kwargs)
        self.processor = processor
        self.alpha = alpha
        self.eipd_lambda = eipd_lambda
        self.max_new_tokens = max_new_tokens
        self.max_answer_tokens = max_answer_tokens
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            target = torch.device(teacher_device) if teacher_device else self.accelerator.device
            self.teacher_model.to(target)
            self.teacher_model.eval()

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        drop_empty_param_groups(optimizer)
        return optimizer

    def _sync_scheduler(self):
        scheduler = getattr(self, "lr_scheduler", None)
        optimizer = getattr(self, "optimizer", None)
        if scheduler is not None and optimizer is not None:
            sync_scheduler_lambdas(scheduler, optimizer)

    def _prompt(self, model, example, bridge=None):
        return encode_prompt(
            self.processor,
            example["observed"],
            example["row"]["question"],
            next(model.parameters()).device,
            bridge=bridge,
        )

    def _gold_ids(self, row, device):
        ids = self.processor.tokenizer.encode(
            row["answer"].strip() + "</answer>", add_special_tokens=False
        )
        if not 0 < len(ids) <= self.max_answer_tokens:
            raise ValueError("Missing or oversized Gold Answer")
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def _gold_only_loss(self, model, prompt, gold):
        prefix = self.processor.tokenizer.encode("<answer>", add_special_tokens=False)
        prefix = torch.tensor(prefix, dtype=torch.long, device=gold.device).unsqueeze(0)
        continuation = torch.cat([prefix, gold], dim=1)
        logits = _answer_logits(model, prompt, continuation, gold.shape[1])
        return F.cross_entropy(logits.float(), gold[0])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        self._sync_scheduler()
        row = inputs["row"]
        device = next(model.parameters()).device
        student_prompt = self._prompt(model, inputs)
        tokenizer = self.processor.tokenizer
        base_model = self.accelerator.unwrap_model(model)

        was_training = base_model.training
        old_cache = base_model.config.use_cache
        base_model.eval()
        base_model.config.use_cache = True
        try:
            with torch.no_grad():
                generated = base_model.generate(
                    **student_prompt,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    synced_gpus=self.args.world_size > 1,
                )
        finally:
            base_model.config.use_cache = old_cache
            base_model.train(was_training)

        sampled_ids = generated[0, student_prompt["input_ids"].shape[1] :].tolist()
        gold = self._gold_ids(row, device)
        try:
            reasoning_ids = answer_prefix_ids(tokenizer, sampled_ids)
        except ValueError:
            # Malformed on-policy samples receive direct Gold supervision, but
            # are not silently converted into an off-policy EIPD example.
            supervised = self._gold_only_loss(model, student_prompt, gold)
            self.log(
                {"invalid_eipd_rollout": 1.0, "gold_ce": float(supervised.detach())}
            )
            return (supervised, {"loss": supervised}) if return_outputs else supervised

        reasoning = torch.tensor(reasoning_ids, dtype=torch.long, device=device).unsqueeze(0)
        continuation = torch.cat([reasoning, gold], dim=1)

        teacher_model = self.teacher_model or base_model
        teacher_device = next(teacher_model.parameters()).device
        teacher_continuation = continuation.to(teacher_device)
        with frozen_teacher(
            teacher_model, disable_adapter=self.teacher_model is None
        ):
            no_bridge = _answer_logits(
                teacher_model,
                self._prompt(teacher_model, inputs),
                teacher_continuation,
                gold.shape[1],
            )
            with_bridge = _answer_logits(
                teacher_model,
                self._prompt(teacher_model, inputs, bridge=inputs["bridge"]),
                teacher_continuation,
                gold.shape[1],
            )
        no_bridge = no_bridge.to(device)
        with_bridge = with_bridge.to(device)

        student_logits = _answer_logits(model, student_prompt, continuation, gold.shape[1])
        supervised = F.cross_entropy(student_logits.float(), gold[0])
        distill = eipd_loss(student_logits, no_bridge, with_bridge, alpha=self.alpha)
        total = supervised + self.eipd_lambda * distill.loss
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite EIPD objective")
        self.log(
            {
                "gold_ce": float(supervised.detach()),
                "eipd_loss": float(distill.loss.detach()),
                "bridge_js": float(distill.mean_js),
                "bridge_active_tokens": distill.active_tokens,
            }
        )
        return (total, {"loss": total}) if return_outputs else total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Original Qwen3-VL checkpoint")
    parser.add_argument(
        "--processor",
        default=None,
        help="Optional complete Qwen3-VL processor directory when the weight copy lacks processor files",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--training-mode",
        choices=("lora", "full"),
        default="lora",
        help="Use LoRA or full language-model finetuning; the visual tower is frozen in both modes",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument(
        "--teacher-device",
        default=None,
        help="Optional device for the independent full-FT Teacher (for example cuda:1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-d", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--image-max-pixels", type=int, default=2048 * 32 * 32)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="steps")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    if args.lambda_d < 0 or not 0 < args.alpha <= 1:
        parser.error("Set valid EIPD weights")

    set_seed(args.seed)
    log_stage("loading Student checkpoint")
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != "qwen3_vl":
        raise ValueError("EIPD supports Qwen3-VL only")
    if hasattr(config, "latent_id") or hasattr(config, "max_latent_token"):
        raise ValueError(
            "This entry requires the original Qwen3-VL checkpoint, not a Future-L1 checkpoint"
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    log_stage("loading processor")
    processor_source = args.processor or args.model
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=2 * 32 * 32,
        max_pixels=args.image_max_pixels,
    )
    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Processor vocabulary is larger than the original model embeddings")
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.config.use_cache = False
    for parameter in model.model.visual.parameters():
        parameter.requires_grad_(False)
    teacher_model = None
    if args.training_mode == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        for name, parameter in model.named_parameters():
            if "visual" in name:
                parameter.requires_grad_(False)
    else:
        # Full Student updates cannot share parameters with an immutable Teacher.
        # Keep one independent original checkpoint per distributed process.
        log_stage("loading independent frozen Teacher checkpoint")
        teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        teacher_model.requires_grad_(False)
        teacher_model.config.use_cache = False
        teacher_model.eval()
        log_stage("Student and Teacher checkpoints loaded")

    gradient_checkpointing = (
        args.training_mode == "full"
        if args.gradient_checkpointing is None
        else args.gradient_checkpointing
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        deepspeed=args.deepspeed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    record = dict(
        vars(args),
        method="EIPD-text-Qwen3VL",
        initialization="original-qwen3-vl",
        teacher="independent-frozen-original" if teacher_model is not None else "adapter-disabled-original",
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        eipd_manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        pytorch_version=torch.__version__,
        model_type=config.model_type,
    )
    (output / "eipd_config.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    log_stage("loading 50K manifest")
    train_dataset = ManifestDataset(args.manifest)
    log_stage(f"manifest loaded: {len(train_dataset)} examples")
    trainer = EIPDTrainer(
        model=model,
        processor=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        alpha=args.alpha,
        eipd_lambda=args.lambda_d,
        max_new_tokens=args.max_new_tokens,
        max_answer_tokens=args.max_answer_tokens,
        teacher_model=teacher_model,
        teacher_device=args.teacher_device,
    )
    log_stage("Trainer initialized; preparing DeepSpeed engine")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))


if __name__ == "__main__":
    main()

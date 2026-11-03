"""Single-stage Cross-Depth Latent Self-Distillation training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .cdlsd_data import CDLSDDataset, LATENT_TOKEN, collate_single
from .cdlsd_latent import (
    answer_logits,
    decode_answer,
    gold_token_ids,
    latent_rollout,
    sample_answer,
)
from .cdlsd_objective import forward_kl, response_cross_entropy, route_kd_through_states
from .eipd_latent import LatentProjector, without_gradient_checkpointing
from .trainer import drop_empty_param_groups, sync_scheduler_lambdas


def log(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[cdlsd] {message}", flush=True)


def save_auxiliary(output_dir: str | Path, model, projector, token_id: int) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    embedding = module.get_input_embeddings().weight.detach()
    torch.save(
        {
            "projector": projector.state_dict(),
            "latent_token_id": int(token_id),
            "latent_token_embedding": embedding[token_id].float().cpu(),
        },
        output / "cdlsd_modules.pt",
    )


class CDLSDTrainer(Trainer):
    def __init__(self, *args, processor, projector, short_steps: int, long_steps: int,
                 lambda_kd: float, projector_lr: float, max_new_tokens: int,
                 temperature: float, top_p: float, token_clip: float | None,
                 checkpoint_prefill: bool, latent_token_id: int,
                 rollout_log_every: int, **kwargs):
        super().__init__(*args, processing_class=processor, **kwargs)
        self.processor = processor
        self.projector = projector
        self.short_steps = short_steps
        self.long_steps = long_steps
        self.lambda_kd = lambda_kd
        self.projector_lr = projector_lr
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.token_clip = token_clip
        self.checkpoint_prefill = checkpoint_prefill
        self.latent_token_id = latent_token_id
        self.rollout_log_every = rollout_log_every
        self.micro_step = 0
        self.nonfinite_count = 0
        rank = int(os.environ.get("RANK", "0"))
        self.rollout_path = Path(self.args.output_dir) / "rollouts" / f"rank-{rank:03d}.jsonl"

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        latent = [
            parameter for name, parameter in self.model.named_parameters()
            if "cdlsd_latent_projector" in name and parameter.requires_grad
        ]
        if latent:
            ids = {id(parameter) for parameter in latent}
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in ids]
            optimizer.add_param_group({
                "params": latent,
                "lr": self.projector_lr,
                "initial_lr": self.projector_lr,
                "weight_decay": 0.0,
            })
        dropped = drop_empty_param_groups(optimizer)
        log(f"optimizer: projector lr={self.projector_lr:g}, "
            f"groups={len(optimizer.param_groups)}, dropped_empty={dropped}")
        return optimizer

    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        super()._save(output_dir, state_dict=state_dict)
        save_auxiliary(output_dir, self.model, self.projector, self.latent_token_id)

    def _write_rollout(self, row: dict, sampled: torch.Tensor, metrics: dict) -> None:
        if self.rollout_log_every <= 0 or self.micro_step % self.rollout_log_every:
            return
        self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "micro_step": self.micro_step,
            "sample_id": row.get("sample_id"),
            "question": row["question"],
            "gold": row["answer"],
            "sampled": decode_answer(self.processor.tokenizer, sampled.detach().cpu()),
            "sampled_token_ids": sampled.detach().cpu().tolist(),
            **{key: float(value) for key, value in metrics.items()},
        }
        with self.rollout_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        scheduler = getattr(self, "lr_scheduler", None)
        optimizer = getattr(self, "optimizer", None)
        if scheduler is not None and optimizer is not None:
            sync_scheduler_lambdas(scheduler, optimizer)

        row = inputs["row"]
        states = latent_rollout(
            model, self.projector, self.processor, inputs["observed"], row["question"],
            short_steps=self.short_steps, long_steps=self.long_steps,
            checkpoint_prefill=self.checkpoint_prefill,
        )
        device = states.short.device
        gold_ids = gold_token_ids(self.processor.tokenizer, row["answer"], device)
        logits4_gold = answer_logits(
            model, self.processor.tokenizer, row["question"], states.short, gold_ids)
        ce4 = response_cross_entropy(logits4_gold, gold_ids)
        del logits4_gold
        logits8_gold = answer_logits(
            model, self.processor.tokenizer, row["question"], states.long, gold_ids)
        ce8 = response_cross_entropy(logits8_gold, gold_ids)
        del logits8_gold
        supervised = 0.5 * (ce4 + ce8)
        advantage = (ce8.detach() < ce4.detach()).to(dtype=torch.float32)

        # Every rank executes generation and both replays.  The detached mask,
        # rather than a Python branch, decides whether the KD gradient is active.
        # HF checkpointed decoder layers forcibly disable use_cache even under
        # no_grad.  Sampling is incremental and therefore temporarily suspends
        # their checkpoint wrappers; the differentiable replay below still uses
        # normal activation checkpointing.
        with without_gradient_checkpointing(model):
            sampled = sample_answer(
                model, self.processor.tokenizer, row["question"], states.short,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_p=self.top_p,
            )
        student_logits = answer_logits(
            model, self.processor.tokenizer, row["question"], states.short, sampled)
        with torch.no_grad():
            teacher_logits = answer_logits(
                model, self.processor.tokenizer, row["question"], states.long.detach(), sampled)
        kd, per_token = forward_kl(
            teacher_logits, student_logits, token_clip=self.token_clip)
        del teacher_logits, student_logits
        routed = route_kd_through_states(kd * advantage, states.short)
        total = supervised + self.lambda_kd * routed.routed

        quantiles = torch.quantile(
            per_token.float(),
            torch.tensor([0.5, 0.95, 0.99], device=per_token.device),
        )
        metrics = {
            "ce_short": ce4.detach(),
            "ce_long": ce8.detach(),
            "ce_gap": (ce4 - ce8).detach(),
            "teacher_advantage": advantage.detach(),
            "kd_value": kd.detach(),
            "kd_active_value": (kd.detach() * advantage),
            "kd_state_grad_norm": routed.state_grad_norm,
            "kd_token_p50": quantiles[0],
            "kd_token_p95": quantiles[1],
            "kd_token_p99": quantiles[2],
            "completion_tokens": torch.tensor(float(sampled.numel()), device=device),
        }
        self._write_rollout(row, sampled, metrics)
        self.micro_step += 1

        if not torch.isfinite(total):
            self.nonfinite_count += 1
            raise FloatingPointError(
                f"non-finite CDLSD loss at micro step {self.micro_step}; "
                f"count={self.nonfinite_count}"
            )
        self.log({key: float(value) for key, value in metrics.items()})
        if return_outputs:
            return total, {"loss": total.detach(), **metrics}
        return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--short-steps", type=int, default=4)
    parser.add_argument("--long-steps", type=int, default=8)
    parser.add_argument("--lambda-kd", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--kd-token-clip", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--projector-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--checkpoint-prefill", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--image-max-pixels", type=int, default=2048 * 32 * 32)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--rollout-log-every", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top-p must be in [0,1]")
    if args.lambda_kd < 0:
        parser.error("--lambda-kd must be non-negative")
    set_seed(args.seed)
    root = Path(args.data_root).resolve()
    manifest = Path(args.manifest or root / "train.jsonl")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log(f"loading original model {args.model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(
        args.processor or args.model,
        min_pixels=2 * 32 * 32,
        max_pixels=args.image_max_pixels,
    )
    tokenizer = processor.tokenizer
    added = tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_TOKEN]})
    if added:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            weight = model.get_input_embeddings().weight
            weight[-added:] = weight[:-added].mean(dim=0, keepdim=True).to(weight.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

    from peft import LoraConfig, TaskType, get_peft_model
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        trainable_token_indices=[latent_token_id],
    ))
    model.config.use_cache = False
    hidden = model.config.text_config.hidden_size
    projector = LatentProjector(hidden)
    projector_rms = projector.init_gamma_from_embeddings(
        model.get_input_embeddings().weight.detach())
    model.add_module("cdlsd_latent_projector", projector)

    if args.resume_from_checkpoint:
        auxiliary = Path(args.resume_from_checkpoint) / "cdlsd_modules.pt"
        if auxiliary.is_file():
            payload = torch.load(auxiliary, map_location="cpu", weights_only=False)
            projector.load_state_dict(payload["projector"])

    dataset = CDLSDDataset(str(root), str(manifest))
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    log(f"dataset={len(dataset)} | K={args.short_steps}/{args.long_steps} | "
        f"trainable={trainable / 1e6:.1f}M | projector gamma={projector_rms:.5f}")

    config = dict(
        vars(args),
        method="CDLSD-VEP-v1.0",
        base_model=str(args.model),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        latent_token=LATENT_TOKEN,
        latent_token_id=latent_token_id,
        trainable_parameters=trainable,
        projector_gamma_init=projector_rms,
        teacher="current shared long-depth path with stop-gradient",
        kd_direction="KL(long || short)",
        kd_gradient_path="short latent states only",
    )
    (output / "cdlsd_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
    )
    trainer = CDLSDTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collate_single,
        processor=processor,
        projector=projector,
        short_steps=args.short_steps,
        long_steps=args.long_steps,
        lambda_kd=args.lambda_kd,
        projector_lr=args.projector_lr,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        token_clip=args.kd_token_clip if args.kd_token_clip > 0 else None,
        checkpoint_prefill=args.checkpoint_prefill,
        latent_token_id=latent_token_id,
        rollout_log_every=args.rollout_log_every,
    )
    log("trainer ready")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    save_auxiliary(output, trainer.model, projector, latent_token_id)
    log(f"finished: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

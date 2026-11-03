"""Stage 2: evidence-increment distillation from a frozen dual-condition teacher.

    L2 = CE_past + lambda_v * (state + eta * temporal) + lambda_D * EIPD

Student:  Past + Q            -> Z_S -> answer logits     (LoRA only)
Teacher0: Past + Q            -> Z_0 -> answer logits     (frozen)
Teacher+: Past + Bridge + Q   -> Z_+ -> answer logits     (frozen)

Teacher and Student are the *same* Stage-1 weights; only the Student LoRA
adapter is trainable, so `disable_adapter()` restores the Stage-1 parameters
exactly. The latent projector and value projection are frozen for the same
reason: they live outside the adapter, so updating them would silently turn the
"frozen" teacher into a moving target. `teacher_fingerprint` proves the point
with a real logits comparison instead of trusting `requires_grad`.

Stage 2 is LoRA-only by construction: full finetuning would share parameters
with the teacher and destroy the frozen reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                          Trainer, TrainingArguments, set_seed)

from .eipd_future_targets import future_prediction_loss
from .eipd_latent import LatentProjector, rollout_latents
from .eipd_objective import EIPDResult, eipd_loss
from .eipd_stage1 import (LATENT_TOKENS, Stage1Dataset, answer_context,
                          answer_scores, collate, encode_prompt, sync_module_device)
from .trainer import drop_empty_param_groups, sync_scheduler_lambdas


def log_stage2(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[stage2] {message}", flush=True)


@contextmanager
def frozen_teacher(model):
    """Adapter disabled and no gradients: the Stage-1 model, bit for bit."""
    was_training = model.training
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        try:
            yield
        finally:
            model.train(was_training)


def condition_logits(model, projector, processor, sample, *, steps: int, device,
                     use_bridge: bool):
    """Z and answer logits for one teacher/student condition."""
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(backbone, "model", backbone)
    row = sample["row"]
    batch, start = encode_prompt(backbone, processor, sample["observed"], row["question"],
                                 sample["bridge"] if use_bridge else None, device)
    prefill = {key: batch[key] for key in
               ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
               if batch.get(key) is not None}
    traj = rollout_latents(model, projector, steps=steps, prefill_kwargs=prefill,
                           start_position=start)
    logits, ce = answer_scores(model, processor.tokenizer, row["question"],
                               row["answer"], traj, device)
    return traj, logits, ce


def stage2_loss(model, projector, value_projection, processor, sample, *,
                steps: int, eta: float, lam_v: float, lam_d: float,
                alpha: float, kd_mode: str = "js") -> dict:
    device = next(model.parameters()).device
    sync_module_device(projector, value_projection, device=device)

    student_traj, student_logits, ce_past = condition_logits(
        model, projector, processor, sample, steps=steps, device=device, use_bridge=False)

    with frozen_teacher(model):
        _, p0, _ = condition_logits(model, projector, processor, sample,
                                    steps=steps, device=device, use_bridge=False)
        _, p_plus, _ = condition_logits(model, projector, processor, sample,
                                        steps=steps, device=device, use_bridge=True)

    gain = _gold_gain(p0, p_plus, sample, processor.tokenizer,
                      model.get_input_embeddings().weight.device)
    if kd_mode == "gain":
        # Only distil when the Bridge actually improved the teacher's likelihood of
        # the gold answer. A large JS says the distribution moved, not that it moved
        # towards the truth.
        if float(gain) > 0:
            distill = eipd_loss(student_logits, p0, p_plus, alpha=alpha, uniform=True)
        else:
            zero = torch.zeros((), device=student_logits.device,
                               dtype=student_logits.dtype)
            distill = EIPDResult(loss=zero, mean_js=zero, active_tokens=0,
                                 normalized_weights=zero)
    else:
        distill = eipd_loss(student_logits, p0, p_plus, alpha=alpha)

    features = sample["features"].to(device)
    anchors = sample["anchors"].anchors
    picked = torch.stack([student_traj.latents[k - 1] for k in anchors])
    future = future_prediction_loss(value_projection(picked), features, eta=eta)

    total = ce_past + lam_v * future["loss"] + lam_d * distill.loss
    return {
        "loss": total,
        "ce_past": ce_past.detach(),
        "eipd_loss": distill.loss.detach(),
        "bridge_js": distill.mean_js,
        "bridge_active_tokens": torch.tensor(float(distill.active_tokens)),
        "state_loss": future["state_loss"].detach(),
        "temporal_loss": future["temporal_loss"].detach(),
        "gold_logprob_gain": gain,
        "kd_active": torch.tensor(float(distill.active_tokens)),
    }


def _gold_gain(p0: torch.Tensor, p_plus: torch.Tensor, sample, tokenizer,
               device) -> torch.Tensor:
    """log p+(A*) - log p0(A*) — the scientific check JS alone cannot give."""
    ids, labels = answer_ids(tokenizer, sample["row"]["question"], sample["row"]["answer"])
    ids, labels = ids.to(device), labels.to(device)
    answer = labels[labels != -100]

    def logprob(logits):
        chosen = logits.gather(-1, answer.unsqueeze(-1)).squeeze(-1)
        return chosen.sum()

    return (logprob(torch.log_softmax(p_plus.float(), -1))
            - logprob(torch.log_softmax(p0.float(), -1))).detach()


def answer_ids(tokenizer, question: str, answer: str):
    return answer_context(tokenizer, question, answer)


class Stage2Trainer(Trainer):
    def __init__(self, *args, processor, projector, value_projection, steps, eta,
                 lam_v, lam_d, alpha, teacher_check_every: int = 50,
                 nonfinite: str = "skip", kd_mode: str = "js", **kwargs):
        super().__init__(*args, processing_class=processor, **kwargs)
        self.processor = processor
        self.projector = projector
        self.value_projection = value_projection
        self.steps = steps
        self.eta = eta
        self.lam_v = lam_v
        self.lam_d = lam_d
        self.alpha = alpha
        self.teacher_check_every = max(1, teacher_check_every)
        self.nonfinite = nonfinite
        self.kd_mode = kd_mode
        self._micro = 0
        self._teacher_reference: torch.Tensor | None = None
        self._probe_inputs: dict | None = None

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        drop_empty_param_groups(optimizer)
        return optimizer

    def _teacher_probe(self, model, inputs):
        """Logits of the frozen teacher must not move as the Student trains.

        The probe must run on the *same* sample every time: comparing logits from
        different samples measures the data, not the teacher.
        """
        device = next(model.parameters()).device
        if self._probe_inputs is None:
            self._probe_inputs = {key: (value.detach().to("cpu") if torch.is_tensor(value)
                                        else value)
                                  for key, value in inputs.items()}
        probe = {key: (value.to(device) if torch.is_tensor(value) else value)
                 for key, value in self._probe_inputs.items()}
        with frozen_teacher(model):
            _, logits, _ = condition_logits(model, self.projector, self.processor, probe,
                                            steps=self.steps, device=device, use_bridge=True)
        logits = logits.detach().float().cpu()
        if self._teacher_reference is None:
            self._teacher_reference = logits
            return None
        drift = float((logits - self._teacher_reference).abs().max())
        self.log({"teacher_logit_drift": drift})
        if drift > 1e-2:
            raise RuntimeError(
                f"frozen teacher changed during Stage 2 (max logit drift {drift:.3e}); "
                "a parameter outside the LoRA adapter was updated")
        return drift

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        scheduler = getattr(self, "lr_scheduler", None)
        if scheduler is not None and getattr(self, "optimizer", None) is not None:
            sync_scheduler_lambdas(scheduler, self.optimizer)
        if self._micro % self.teacher_check_every == 0:
            self._teacher_probe(model, inputs)
        self._micro += 1
        parts = stage2_loss(model, self.projector, self.value_projection, self.processor,
                            inputs, steps=self.steps, eta=self.eta, lam_v=self.lam_v,
                            lam_d=self.lam_d, alpha=self.alpha,
                            kd_mode=self.kd_mode)
        if not torch.isfinite(parts["loss"]):
            self._skipped = getattr(self, "_skipped", 0) + 1
            self.log({"nonfinite_batch": 1.0, "nonfinite_total": float(self._skipped)})
            if getattr(self, "nonfinite", "skip") == "raise":
                raise RuntimeError("non-finite Stage 2 objective")
            return (parts["loss"] * 0.0, {"loss": parts["loss"]}) if return_outputs \
                else parts["loss"] * 0.0
        self.log({k: (float(v) if v.ndim == 0 else v) for k, v in parts.items()
                  if k != "loss"})
        return (parts["loss"], {"loss": parts["loss"]}) if return_outputs else parts["loss"]


def load_stage1_modules(stage1_dir: str, hidden: int, device) -> tuple:
    payload = torch.load(Path(stage1_dir) / "latent_modules.pt", map_location="cpu",
                         weights_only=False)
    return payload


def restore_added_tokens(model, payload) -> int:
    """Copy the Stage-1 latent-marker embeddings back into the base embedding matrix."""
    ids = payload.get("added_token_ids")
    rows = payload.get("added_token_embeddings")
    if not ids or rows is None:
        raise RuntimeError("Stage 1 checkpoint lacks the latent token embeddings; "
                           "re-run Stage 1 with this version of eipd_stage1.py")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    weight = base.get_input_embeddings().weight
    if max(ids) >= weight.shape[0]:
        base.resize_token_embeddings(max(ids) + 1)
        weight = base.get_input_embeddings().weight
    with torch.no_grad():
        weight[ids] = rows.to(weight.device, weight.dtype)
    return len(ids)
    projector = LatentProjector(hidden)
    projector.load_state_dict(payload["projector"])
    value_projection = torch.nn.Linear(hidden, hidden, bias=False)
    value_projection.load_state_dict(payload["value_projection"])
    projector.to(device).eval()
    value_projection.to(device).eval()
    for module in (projector, value_projection):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return projector, value_projection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Stage 1 output directory")
    parser.add_argument("--base-model", default=None,
                        help="original Qwen3-VL dir when the Stage-1 copy lacks a full "
                             "checkpoint (LoRA mode stores the adapter only)")
    parser.add_argument("--eipd-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-steps", type=int, default=16)
    parser.add_argument("--lambda-v", type=float, default=0.2)
    parser.add_argument("--lambda-d", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--image-max-pixels", type=int, default=2048 * 32 * 32)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--teacher-check-every", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--nonfinite", choices=("skip", "raise"), default="skip")
    parser.add_argument("--kd-mode", choices=("js", "gain"), default="js",
                        help="js = weight the KL by teacher disagreement; gain = only "
                             "distil samples whose Bridge improved the gold likelihood")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    set_seed(args.seed)
    root = Path(args.eipd_root).resolve()
    manifest = Path(args.manifest or root / "train.jsonl")
    features = Path(args.features or root / "future_features" / "train")

    stage1 = Path(args.stage1)
    # Stage 1 weights win: they are the teacher. --base-model is only the fallback
    # for a LoRA Stage 1, which saves an adapter instead of a full checkpoint.
    has_weights = (stage1 / "config.json").is_file() and any(
        (stage1 / name).is_file() for name in
        ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))
    source = str(stage1) if has_weights else args.base_model
    if source is None:
        parser.error("Stage 1 wrote an adapter only; pass --base-model")

    log_stage2("loading Stage 1 checkpoint")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(
        str(stage1) if (stage1 / "preprocessor_config.json").is_file() else source,
        min_pixels=2 * 32 * 32, max_pixels=args.image_max_pixels)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    adapter_file = stage1 / "adapter_model.safetensors"
    if adapter_file.is_file():
        model = PeftModel.from_pretrained(model, str(stage1), is_trainable=True)
        log_stage2("loaded Stage 1 LoRA adapter as trainable Student")
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                            "up_proj", "down_proj"]))
    for name, parameter in model.named_parameters():
        if "visual" in name or "lora_" not in name:
            parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters; Stage 2 is LoRA-only")
    log_stage2(f"trainable tensors: {len(trainable)} "
               f"({sum(p.numel() for p in trainable)/1e6:.1f}M params)")

    device = next(model.parameters()).device
    payload = load_stage1_modules(str(stage1), model.config.text_config.hidden_size, device)
    projector = LatentProjector(model.config.text_config.hidden_size)
    projector.load_state_dict(payload["projector"])
    value_projection = torch.nn.Linear(model.config.text_config.hidden_size,
                                      model.config.text_config.hidden_size, bias=False)
    value_projection.load_state_dict(payload["value_projection"])
    for module in (projector, value_projection):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    # Same registration as Stage 1 so they follow the model onto the rank's GPU;
    # both stay frozen, which is what keeps `disable_adapter()` reproducing the
    # Stage-1 teacher exactly.
    model.add_module("latent_projector", projector)
    model.add_module("latent_value_projection", value_projection)
    restored = restore_added_tokens(model, payload)
    log_stage2(f"frozen projector + value projection loaded; {restored} latent token embedding(s) restored")

    dataset = Stage1Dataset(str(root), str(manifest), str(features),
                            steps=args.latent_steps)
    train_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs, max_steps=args.max_steps,
        deepspeed=args.deepspeed, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine", bf16=True, remove_unused_columns=False,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, save_only_model=args.save_only_model,
        report_to="none", dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "eipd_stage2_config.json").write_text(json.dumps(dict(
        vars(args), method="EIPD-stage2-evidence-increment",
        eipd_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        teacher="stage1-weights-adapter-disabled",
        frozen=["projector", "value_projection", "visual", "base-weights"]), indent=2),
        encoding="utf-8")

    trainer = Stage2Trainer(
        model=model, processor=processor, args=train_args, train_dataset=dataset,
        data_collator=collate, projector=projector, value_projection=value_projection,
        steps=args.latent_steps, eta=args.eta, lam_v=args.lambda_v, lam_d=args.lambda_d,
        alpha=args.alpha, teacher_check_every=args.teacher_check_every,
        nonfinite=args.nonfinite, kd_mode=args.kd_mode)
    log_stage2("trainer ready")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

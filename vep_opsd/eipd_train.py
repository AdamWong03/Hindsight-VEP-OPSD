"""Training entry for EIPD on an installed checkout of OpenGVLab/Future-L1.

This is deliberately separate from the repository's legacy Gold+Future OPSD.
Requires FUTURE_L1_ROOT pointing at the exact upstream revision used by training.
Run a single-GPU gradient/latent parity smoke test before distributed training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoProcessor, Trainer, TrainingArguments, set_seed

from .trainer import drop_empty_param_groups, sync_scheduler_lambdas
from .eipd_objective import eipd_loss
from .eipd_replay import replay_with_latents, answer_prediction_rows

class ManifestDataset(Dataset):
    def __init__(self, manifest: str):
        self.rows = [json.loads(x) for x in Path(manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
        if not self.rows:
            raise ValueError("Empty EIPD manifest")
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        import decord
        from PIL import Image
        row = self.rows[i]
        reader = decord.VideoReader(row["video_path"], num_threads=1)
        observed_indices = row["observed_raw_frames"]
        bridge_index = row["bridge_raw_frame"]
        target_start = min(row["future_raw_frames"])
        if not max(observed_indices) < bridge_index < target_start:
            raise ValueError(f"Sample {row['id']} has invalid bridge temporal order")
        pixels = reader.get_batch([*observed_indices, bridge_index]).asnumpy()
        obs = [Image.fromarray(pixels[k]) for k in range(len(observed_indices))]
        bridge = Image.fromarray(pixels[-1])
        return {"row": row, "observed": obs, "bridge": bridge}

def collate(items):
    if len(items) != 1:
        raise ValueError("EIPD reference implementation requires per-device batch size 1")
    return items[0]

def messages(obs, question, bridge=None):
    """Identical base prompt except the explicitly added training-only bridge image."""
    content = [{"type": "text", "text": "Observed frames in temporal order:"}]
    content.extend({"type": "image", "image": image} for image in obs)
    if bridge is not None:
        content.extend([{"type": "text", "text": "Additional intermediate future observation:"},
                        {"type": "image", "image": bridge}])
    content.append({"type": "text", "text": "Question: " + question +
                    "\nPredict the subsequent event. Generate your normal continuous latent reasoning "
                    "when helpful, then write <answer>one concise prediction</answer>."})
    return [{"role": "system", "content": "You predict future events from video. Do not state that future frames were observed."},
            {"role": "user", "content": content}]

def encode_prompt(processor, obs, question, device, bridge=None):
    items = messages(obs, question, bridge)
    prompt = processor.apply_chat_template(items, tokenize=False, add_generation_prompt=True)
    frames = [*obs] + ([bridge] if bridge is not None else [])
    batch = processor(text=[prompt], images=frames, padding=False, return_tensors="pt")
    return {key: tensor.to(device) if torch.is_tensor(tensor) else tensor
            for key, tensor in batch.items()}

def answer_prefix_ids(tokenizer, sampled_ids: list[int]) -> list[int]:
    """Use student's own sampled discrete reasoning up to and including <answer>."""
    for k in range(1, len(sampled_ids) + 1):
        text = tokenizer.decode(sampled_ids[:k], skip_special_tokens=False)
        if "<answer>" in text:
            return sampled_ids[:k]
    raise ValueError("Student rollout did not reach an <answer> boundary")

@contextmanager
def frozen_base_teacher(model):
    """Same initial checkpoint, adapter disabled, no EMA or answer conditioning."""
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        with torch.no_grad():
            try:
                yield
            finally:
                model.train(was_training)

class EIPDTrainer(Trainer):
    def __init__(self, *args, processor, alpha: float, eipd_lambda: float,
                 max_new_tokens: int, sft_collator, max_answer_tokens: int = 128, **kwargs):
        super().__init__(*args, processing_class=processor, **kwargs)
        self.processor = processor
        self.alpha = alpha
        self.eipd_lambda = eipd_lambda
        self.max_new_tokens = max_new_tokens
        self.sft_collator = sft_collator
        self.max_answer_tokens = max_answer_tokens

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        drop_empty_param_groups(optimizer)
        return optimizer

    def _sync_scheduler(self):
        scheduler = getattr(self, "lr_scheduler", None)
        optimizer = getattr(self, "optimizer", None)
        if scheduler is not None and optimizer is not None:
            sync_scheduler_lambdas(scheduler, optimizer)

    def _teacher_replay(self, model, example, continuation, answer_length, bridge):
        p = encode_prompt(self.processor, example["observed"], example["row"]["question"],
                          next(model.parameters()).device, bridge=bridge)
        logits, states = replay_with_latents(
            model, p, continuation, model.config.latent_id, require_latents=True,
            answer_tokens=answer_length)
        return answer_prediction_rows(logits, p["input_ids"].shape[1],
                                      continuation.shape[1] - answer_length, answer_length)

    def _sft_loss(self, model, row):
        # Reuse the official Future-L1 SFT collator and its latent-image supervision.
        if not row.get("sft_response"):
            raise ValueError("EIPD manifest missing original SFT reasoning response")
        item = {"video": row["video_path"], "image": row["image"],
                "reasoning_image": row["reasoning_image"], "answer": row["answer"],
                "conversations": [
                    {"from": "human", "value": " ".join("frame_" + str(i) + " <image>" for i in row["image"]) +
                     "\n" + row["question"]},
                    {"from": "gpt", "value": row["sft_response"]},
                ]}
        batch = self.sft_collator([item])
        if not batch:
            raise ValueError("Upstream Future-L1 SFT collator rejected EIPD sample")
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**batch, return_dict=True, use_cache=False)
        if out.loss is None or not torch.isfinite(out.loss):
            raise RuntimeError("Future-L1 SFT loss is missing or non-finite")
        return out.loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        self._sync_scheduler()
        row = inputs["row"]
        device = next(model.parameters()).device
        prompt = encode_prompt(self.processor, inputs["observed"], row["question"], device)
        token = self.processor.tokenizer
        bare_model = self.accelerator.unwrap_model(model)
        was_training = bare_model.training
        bare_model.eval()
        try:
            with torch.no_grad():
                generated = bare_model.generate(
                    **prompt, do_sample=True, temperature=0.8, top_p=0.95,
                    max_new_tokens=self.max_new_tokens, use_cache=True,
                    pad_token_id=token.pad_token_id)
        finally:
            bare_model.train(was_training)
        sampled_ids = generated[0, prompt["input_ids"].shape[1]:].tolist()
        # Invalid on-policy output contributes *only* original SFT; do not sneak
        # ground-truth reasoning into EIPD or silently drop the training row.
        try:
            reasoning_ids = answer_prefix_ids(token, sampled_ids)
            if bare_model.config.latent_id not in reasoning_ids:
                raise ValueError("No sampled continuous latent positions")
            gold_ids = token.encode(row["answer"] + "</answer>", add_special_tokens=False)
            if not 0 < len(gold_ids) <= self.max_answer_tokens:
                raise ValueError("Missing/oversized supervised answer")
        except ValueError:
            sft_loss = self._sft_loss(model, row)
            self.log({"invalid_eipd_rollout": 1.0, "sft_loss": float(sft_loss.detach())})
            return (sft_loss, {"loss": sft_loss}) if return_outputs else sft_loss
        reason_tensor = torch.tensor(reasoning_ids, dtype=torch.long, device=device).unsqueeze(0)
        answer_tensor = torch.tensor(gold_ids, dtype=torch.long, device=device).unsqueeze(0)
        cont = torch.cat([reason_tensor, answer_tensor], dim=1)
        with frozen_base_teacher(bare_model):
            p0 = self._teacher_replay(bare_model, inputs, cont, len(gold_ids), bridge=None)
            p1 = self._teacher_replay(bare_model, inputs, cont, len(gold_ids), bridge=inputs["bridge"])
        student_logits, states = replay_with_latents(
            model, prompt, cont, bare_model.config.latent_id, require_latents=True,
            answer_tokens=len(gold_ids))
        student_logits = answer_prediction_rows(student_logits, prompt["input_ids"].shape[1],
                                                 len(reasoning_ids), len(gold_ids))
        distill = eipd_loss(student_logits, p0, p1, alpha=self.alpha)
        sft_loss = self._sft_loss(model, row)
        total = sft_loss + self.eipd_lambda * distill.loss
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite EIPD training objective")
        if not all(x.requires_grad for x in states.values()):
            raise RuntimeError("Latent states are detached")
        if distill.active_tokens:
            checks = torch.autograd.grad(distill.loss, tuple(states.values()),
                                         retain_graph=True, allow_unused=True)
            if all(g is None for g in checks):
                raise RuntimeError("EIPD answer loss has no autograd path into latent states")
        self.log({"sft_loss": float(sft_loss.detach()), "eipd_loss": float(distill.loss.detach()),
                  "bridge_js": float(distill.mean_js), "latent_steps": len(states),
                  "bridge_active_tokens": distill.active_tokens})
        return (total, {"loss": total}) if return_outputs else total

def import_upstream(root: str):
    root = str(Path(root).resolve())
    if not (Path(root) / "src/model/future_l1.py").exists():
        raise FileNotFoundError("FUTURE_L1_ROOT must be a checkout of OpenGVLab/Future-L1")
    sys.path.insert(0, root)
    from src.model.future_l1 import FutureL1_Qwen3VL
    from src.train.monkey_patch_forward import (
        replace_qwen3_with_mixed_modality_forward, replace_qwen3_vl_generation_forward)
    from src.dataset.twiff_sft_dataset import TwiFFDataCollator
    replace_qwen3_with_mixed_modality_forward()
    replace_qwen3_vl_generation_forward()
    return FutureL1_Qwen3VL, TwiFFDataCollator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--future-l1-root", default=os.environ.get("FUTURE_L1_ROOT", ""))
    ap.add_argument("--model", required=True, help="Future-L1-SFT Qwen3-VL checkpoint")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="Use 1 for a checkpoint-backed GPU smoke test")
    ap.add_argument("--deepspeed", default=None, help="Optional ZeRO-2 configuration JSON for distributed training")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-answer-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-d", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-latent-token", type=int, default=None,
                    help="Optional assertion of the SFT checkpoint's latent budget; does not override it")
    ap.add_argument("--image-max-pixels", type=int, default=128 * 32 * 32)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    args = ap.parse_args()
    if not args.future_l1_root or args.lambda_d < 0 or not 0 < args.alpha <= 1:
        ap.error("Set --future-l1-root and valid EIPD weights")
    set_seed(args.seed)
    FutureL1Model, UpstreamSFTCollator = import_upstream(args.future_l1_root)
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != "qwen3_vl":
        raise ValueError("EIPD presently supports Future-L1 Qwen3-VL only")
    model = FutureL1Model.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16,
                                           attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.add_tokens(["<|latent|>", "<|latent_start|>", "<|latent_end|>"],
                                   special_tokens=False)
    for key, literal in [("latent_id", "<|latent|>"), ("latent_start_id", "<|latent_start|>"),
                         ("latent_end_id", "<|latent_end|>")]:
        actual_id = processor.tokenizer.convert_tokens_to_ids(literal)
        old_id = getattr(model.config, key, actual_id)
        if old_id != actual_id:
            raise ValueError(f"{key} mismatch: checkpoint={old_id}, tokenizer={actual_id}")
        setattr(model.config, key, actual_id)
    if model.get_input_embeddings().weight.shape[0] < len(processor.tokenizer):
        raise ValueError("Model embeddings do not include checkpoint's latent tokens")
    trained_budget = getattr(model.config, "max_latent_token", None)
    if trained_budget is None:
        raise ValueError("Checkpoint lacks max_latent_token; use a trained Future-L1 checkpoint")
    if args.max_latent_token is not None and args.max_latent_token != int(trained_budget):
        raise ValueError("Do not override checkpoint latent budget when comparing EIPD")
    trained_budget = int(trained_budget)
    model.config.latent_loss = "mse"
    model.config.latent_lambda = float(getattr(model.config, "latent_lambda", 0.1) or 0.1)
    model.config.use_cache = False
    for param in model.model.visual.parameters():
        param.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                           lora_alpha=args.lora_alpha, lora_dropout=0,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad_(False)
    data_options = SimpleNamespace(max_latent_token=trained_budget,
                       image_max_pixels=args.image_max_pixels, image_min_pixels=2 * 32 * 32,
                       shuffle_latent_images=False, random_seed=args.seed,
                       merge_latent_segments=False, fixed_latent_budget=None,
                       decoder_recon_max_text_len=0)
    sft_collator = UpstreamSFTCollator(processor, data_options)
    train_args = TrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
                      max_steps=args.max_steps, deepspeed=args.deepspeed,
                      per_device_train_batch_size=1,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      learning_rate=args.lr, bf16=True, remove_unused_columns=False,
                      logging_steps=10, save_steps=500, save_total_limit=2,
                      report_to="none", dataloader_num_workers=0, seed=args.seed,
                      gradient_checkpointing=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    def commit_of(path):
        try:
            return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    manifest_sha256 = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    config_record = dict(vars(args), source_git_commit=commit_of(args.future_l1_root),
                         eipd_manifest_sha256=manifest_sha256,
                         pytorch_version=torch.__version__,
                         model_type=config.model_type)
    (out / "eipd_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    trainer = EIPDTrainer(model=model, processor=processor, args=train_args,
                          train_dataset=ManifestDataset(args.manifest), data_collator=collate,
                          alpha=args.alpha, eipd_lambda=args.lambda_d,
                          max_new_tokens=args.max_new_tokens,
                          max_answer_tokens=args.max_answer_tokens, sft_collator=sft_collator)
    trainer.train()
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))

if __name__ == "__main__":
    main()

"""Stage 1: pure continuous-latent future prediction from original Qwen3-VL.

    L1 = (CE_past + CE_past_bridge) / 2 + lambda_v * (state + eta * temporal)

Two views of the same sample train the same weights: Past-only (the inference
path) and Past+Bridge (the condition Stage 2's teacher will use). The future
targets are cached frozen vision features, so training never touches a future
PNG and never re-runs the vision tower.

Run a one-step smoke before any long run:

    python -m vep_opsd.eipd_stage1 --model /models/Qwen3-VL-8B-Instruct \
        --eipd-root /data/EIPD --output /runs/stage1-smoke \
        --max-steps 1 --gradient-accumulation-steps 1
"""
from __future__ import annotations

import argparse
import hashlib
import random
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                          Trainer, TrainingArguments, set_seed)

from .eipd_future_targets import (assign_anchors, future_prediction_loss,
                                  select_future_frames)
from .eipd_latent import (LatentProjector, LatentTrajectory, bottleneck_answer_logits,
                          prefill_only, rollout_latents_shared)
from .trainer import drop_empty_param_groups, sync_scheduler_lambdas

LATENT_TOKENS = ["<|latent_start|>"]
SYSTEM = ("You predict future events from video. Reason in continuous latent states, "
          "not in words.")
ANSWER_INSTRUCTION = "Predict the subsequent event."


def log_stage(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[stage1] {message}", flush=True)


# --------------------------------------------------------------------------- data
class ShardedFeatureStore:
    """sample_id -> {"features": (m,d) float32, "native": [...]} across .pt shards."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        shards = sorted(self.directory.glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(f"no feature shards under {self.directory}")
        self.index: dict[str, Path] = {}
        for path in shards:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for sample_id in payload["sample_ids"]:
                self.index[sample_id] = path
        self._cache_path: Path | None = None
        self._cache: dict = {}

    def get(self, sample_id: str) -> dict:
        path = self.index[sample_id]
        if path != self._cache_path:
            self._cache = torch.load(path, map_location="cpu", weights_only=False)["features"]
            self._cache_path = path
        return self._cache[sample_id]


class Stage1Dataset(Dataset):
    def __init__(self, eipd_root: str, manifest: str, features: str, *, steps: int = 16):
        self.root = Path(eipd_root).resolve()
        self.rows = [json.loads(line) for line in
                     Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("empty manifest")
        self.store = ShardedFeatureStore(Path(features))
        self.steps = steps
        missing = [r["sample_id"] for r in self.rows if r["sample_id"] not in self.store.index]
        if missing:
            raise KeyError(f"{len(missing)}/{len(self.rows)} samples lack cached future "
                           f"features (e.g. {missing[0]}); run "
                           f"scripts/precompute_future_features.py first")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image
        row = dict(self.rows[index])

        def load(relative: str):
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                return image.convert("RGB").copy()

        cache = self.store.get(row["sample_id"])
        features = cache["features"]                     # (m, d) fp32, L2-normalised
        native = list(row["future_native_indices"])
        past_end = max(row["past_native_indices"])
        kept = select_future_frames(native, self.steps)
        if len(kept) != len(native):
            features = features[[native.index(t) for t in kept]]
        anchors = assign_anchors(past_end, kept, steps=self.steps)
        return {
            "row": row,
            "observed": [load(x) for x in row["past_files"]],
            "bridge": load(row["bridge_file"]),
            "features": features,
            "anchors": anchors,
        }


def sync_module_device(*modules, device):
    """Move auxiliary modules onto the device the trainer actually uses.

    The latent projector and value projection are not children of the model, so
    neither Trainer nor DeepSpeed relocates them; at construction time the model
    is still on CPU, which is why they must follow the model lazily.
    """
    for module in modules:
        if module is None:
            continue
        current = next(module.parameters()).device
        if current != device:
            module.to(device)

def collate(items):
    if len(items) != 1:
        raise ValueError("Stage 1 requires per-device batch size 1")
    return items[0]


# --------------------------------------------------------------------------- prompts
def encode_prompt(backbone, processor, images, question: str, bridge_image, device):
    """Tokenised multimodal prompt ending with the latent-start marker.

    The image list order is [observed..., bridge?]; the text mentions the bridge
    block only in the teacher condition, so both views share the same template.
    """
    content = [{"type": "text", "text": "Observed frames in temporal order:"}]
    content += [{"type": "image", "image": image} for image in images]
    if bridge_image is not None:
        content += [{"type": "text",
                     "text": "Additional intermediate future observation available only to "
                             "the training teacher:"},
                    {"type": "image", "image": bridge_image}]
    content += [{"type": "text", "text": "Question: " + question + "\n" + ANSWER_INSTRUCTION}]
    text = processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True) + LATENT_TOKENS[0]
    frames = [*images] + ([bridge_image] if bridge_image is not None else [])
    batch = processor(text=[text], images=frames, return_tensors="pt")
    batch = {key: (tensor.to(device) if torch.is_tensor(tensor) else tensor)
             for key, tensor in batch.items()}
    with torch.no_grad():
        positions, _deltas = backbone.get_rope_index(
            batch["input_ids"], image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch.get("attention_mask"))
    next_position = int(positions[:, :, -1].max().item()) + 1
    return batch, next_position


def answer_context(tokenizer, question: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[Q ; <answer>A</answer>] token ids; labels supervise the answer span only."""
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer("<answer>" + answer + "</answer>",
                           add_special_tokens=False)["input_ids"]
    ids = torch.tensor([question_ids + answer_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(question_ids) + answer_ids], dtype=torch.long)
    return ids, labels


def answer_scores(model, tokenizer, question, answer, latents, device):
    """(logits [T, V] float32 predicting the answer span, CE scalar).

    `latents` may be a LatentTrajectory or a plain (K, d) tensor.
    """
    latents = getattr(latents, "latents", latents)
    ids, labels = answer_context(tokenizer, question, answer)
    ids, labels = ids.to(device), labels.to(device)
    answer_length = int((labels != -100).sum())
    embeddings = model.get_input_embeddings()(ids)
    question_length = ids.shape[1] - answer_length
    logits = bottleneck_answer_logits(
        model, embeddings[:, :question_length], latents, embeddings[:, question_length:])
    # (1, T+1, V) -> (T, V): eipd_objective expects one row per answer token.
    logits = logits[0, :answer_length].float()
    ce = F.cross_entropy(logits, labels[labels != -100].reshape(-1))
    return logits, ce


def answer_ce(model, tokenizer, question, answer, latents, device) -> torch.Tensor:
    return answer_scores(model, tokenizer, question, answer, latents, device)[1]


# --------------------------------------------------------------------------- step
def stage1_loss(model, projector, value_projection, processor, tokenizer, sample,
                *, steps: int, eta: float, lam_v: float, batch_views: bool = True,
                prefill_checkpoint: bool = True, single_view: bool = False) -> dict:
    device = next(model.parameters()).device
    sync_module_device(projector, value_projection, device=device)
    row = sample["row"]
    observed, bridge = sample["observed"], sample["bridge"]

    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(backbone, "model", backbone)

    # Which views to run. Running both costs two prefills and two answer passes per
    # sample; sampling one at random has the same expectation (and halves the work),
    # at the price of a noisier gradient. The future loss only applies to past-only
    # views, so it is scaled by two to keep E[L] == (CE_p + CE_+)/2 + lam_v * future.
    if single_view:
        use_bridge = random.random() < 0.5
        views = [bridge] if use_bridge else [None]
    else:
        use_bridge = True
        views = [None, bridge]

    prefills = []
    for bridge_image in views:
        batch, start = encode_prompt(backbone, processor, observed, row["question"],
                                     bridge_image, device)
        prefills.append({
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch.get("pixel_values"),
            "image_grid_thw": batch.get("image_grid_thw"),
            "start": start,
        })

    def seed(item):
        prefill = {key: item[key] for key in
                   ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
                   if item.get(key) is not None}
        return prefill_only(model, projector, prefill, item["start"],
                            checkpoint=prefill_checkpoint)

    if batch_views and len(prefills) > 1:
        latents = rollout_latents_shared(model, projector, [seed(i) for i in prefills],
                                        steps=steps)
    else:
        latents = torch.stack([
            rollout_latents_shared(model, projector, [seed(item)], steps=steps)[0]
            for item in prefills
        ])

    ces = [answer_ce(model, tokenizer, row["question"], row["answer"], latent, device)
           for latent in latents]
    if single_view:
        ce_loss = ces[0]
        future_weight = 0.0 if use_bridge else 2.0 * lam_v
        traj_past = LatentTrajectory(latents[0], [])
    else:
        ce_loss = 0.5 * (ces[0] + ces[1])
        future_weight = lam_v
        traj_past = LatentTrajectory(latents[0], [])

    features = sample["features"].to(device)
    anchors = sample["anchors"].anchors
    if len(anchors) != features.shape[0]:
        raise RuntimeError("anchor/target mismatch")
    picked = torch.stack([traj_past.latents[k - 1] for k in anchors])
    predicted = value_projection(picked)
    if future_weight > 0:
        future = future_prediction_loss(predicted, features, eta=eta)
    else:
        zero = torch.zeros((), device=device, dtype=predicted.dtype)
        future = {"loss": zero, "state_loss": zero, "temporal_loss": zero}

    total = ce_loss + future_weight * future["loss"]
    return {"loss": total,
            "ce_past": ces[0].detach(),
            "ce_bridge": ces[-1].detach() if use_bridge and len(ces) > 1 else torch.zeros_like(ces[0].detach()),
            "view_bridge": torch.tensor(float(use_bridge)),
            "state_loss": future["state_loss"].detach(),
            "temporal_loss": future["temporal_loss"].detach(),
            "anchors": torch.tensor(float(len(anchors))),
            "forced_anchors": torch.tensor(float(sample["anchors"].forced))}


class Stage1Trainer(Trainer):
    def __init__(self, *args, processor, projector, value_projection, steps, eta, lam_v,
                 latent_lr: float = 0.0, nonfinite: str = "skip",
                 batch_views: bool = True, prefill_checkpoint: bool = True,
                 single_view: bool = False, **kwargs):
        super().__init__(*args, processing_class=processor, **kwargs)
        self.processor = processor
        self.projector = projector
        self.value_projection = value_projection
        self.steps = steps
        self.eta = eta
        self.lam_v = lam_v
        self.latent_lr = float(latent_lr or 0.0)
        self.batch_views = bool(batch_views)
        self.prefill_checkpoint = bool(prefill_checkpoint)
        self.single_view = bool(single_view)
        self.nonfinite = nonfinite
        self._skipped = 0

    def create_optimizer(self):
        init_lr = getattr(self.args, "learning_rate", None)
        if getattr(self, "optimizer", None) is None and init_lr is not None:
            self.args.learning_rate = init_lr
        optimizer = super().create_optimizer()
        if (self.latent_lr > 0
                and abs(self.latent_lr - float(self.args.learning_rate)) > 1e-12):
            latent = [p for name, p in self.model.named_parameters()
                      if "latent_projector" in name or "latent_value_projection" in name]
            if latent:
                ids = {id(p) for p in latent}
                for group in optimizer.param_groups:
                    group["params"] = [p for p in group["params"] if id(p) not in ids]
                optimizer.add_param_group({"params": latent,
                                           "lr": self.latent_lr,
                                           "initial_lr": self.latent_lr,
                                           "weight_decay": 0.0})
                log_stage(f"latent module lr={self.latent_lr} ({len(latent)} tensors)")
        drop_empty_param_groups(optimizer)
        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        scheduler = getattr(self, "lr_scheduler", None)
        if scheduler is not None and getattr(self, "optimizer", None) is not None:
            sync_scheduler_lambdas(scheduler, self.optimizer)
        parts = stage1_loss(model, self.projector, self.value_projection,
                            self.processor, self.processor.tokenizer, inputs,
                            steps=self.steps, eta=self.eta, lam_v=self.lam_v,
                            batch_views=self.batch_views,
                            prefill_checkpoint=self.prefill_checkpoint,
                            single_view=self.single_view)
        if not torch.isfinite(parts["loss"]):
            self._skipped += 1
            self.log({"nonfinite_batch": 1.0, "nonfinite_total": float(self._skipped)})
            if self.nonfinite == "raise":
                raise RuntimeError("non-finite Stage 1 objective")
            # A zero-loss micro-batch contributes no gradient and lets the run
            # continue; raising here would destroy a multi-hour job on one spike.
            return (parts["loss"] * 0.0, {"loss": parts["loss"]}) if return_outputs \
                else parts["loss"] * 0.0
        self.log({k: float(v) for k, v in parts.items() if k != "loss"})
        return (parts["loss"], {"loss": parts["loss"]}) if return_outputs else parts["loss"]


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="original Qwen3-VL checkpoint")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--eipd-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-steps", type=int, default=16)
    parser.add_argument("--lambda-v", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--training-mode", choices=("lora", "full"), default="full")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--latent-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=3.0)
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
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--nonfinite", choices=("skip", "raise"), default="skip")
    parser.add_argument("--single-view", type=int, default=0,
                        help="1 = run Past or Past+Bridge at random (same expectation, "
                             "half the work, noisier gradients)")
    parser.add_argument("--prefill-checkpoint", type=int, default=1,
                        help="1 = recompute each multimodal prefill during backward "
                             "(frees ~14 GiB per view, costs one extra prefill forward)")
    parser.add_argument("--batch-views", type=int, default=1,
                        help="1 = share one decode chain between the two views. 0 = separate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    set_seed(args.seed)
    root = Path(args.eipd_root).resolve()
    manifest = Path(args.manifest or root / "train.jsonl")
    features = Path(args.features or root / "future_features" / "train")

    log_stage("loading checkpoint")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(
        args.processor or args.model, min_pixels=2 * 32 * 32, max_pixels=args.image_max_pixels)
    tokenizer = processor.tokenizer
    added = tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    if added:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            embedding = model.get_input_embeddings().weight
            fill = embedding[:-added].mean(dim=0)
            embedding[-added:] = fill.to(embedding.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False
    for parameter in model.model.visual.parameters():
        parameter.requires_grad_(False)
    if args.training_mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                            "up_proj", "down_proj"]))
        for name, parameter in model.named_parameters():
            if "visual" in name:
                parameter.requires_grad_(False)

    hidden = model.config.text_config.hidden_size
    projector = LatentProjector(hidden)
    rms = projector.init_gamma_from_embeddings(
        model.get_input_embeddings().weight.detach())
    value_projection = torch.nn.Linear(hidden, hidden, bias=False)
    torch.nn.init.eye_(value_projection.weight)
    # Register as submodules: Trainer builds the optimizer from model.parameters(),
    # so a detached projector would silently never train (and would never follow
    # the model onto the rank's GPU either).
    model.add_module("latent_projector", projector)
    model.add_module("latent_value_projection", value_projection)
    log_stage(f"latent projector γ={rms:.4f}, K={args.latent_steps}, mode={args.training_mode}")

    dataset = Stage1Dataset(str(root), str(manifest), str(features),
                            steps=args.latent_steps)
    log_stage(f"{len(dataset)} samples with cached future features")

    train_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs, max_steps=args.max_steps,
        deepspeed=args.deepspeed, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine", bf16=True,
        remove_unused_columns=False, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model, report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "eipd_stage1_config.json").write_text(json.dumps(
        dict(vars(args),
             method="EIPD-stage1-pure-latent",
             latent_tokens=LATENT_TOKENS,
             eipd_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
             feature_config=json.loads(
                 (features.parent / "feature_config.json").read_text(encoding="utf-8")),
             projector_gamma_init=rms), indent=2), encoding="utf-8")

    trainer = Stage1Trainer(
        model=model, processor=processor, args=train_args, train_dataset=dataset,
        data_collator=collate, projector=projector, value_projection=value_projection,
        steps=args.latent_steps, eta=args.eta, lam_v=args.lambda_v,
        latent_lr=args.latent_lr, nonfinite=args.nonfinite,
        batch_views=bool(args.batch_views),
        prefill_checkpoint=bool(args.prefill_checkpoint),
        single_view=bool(args.single_view))
    log_stage("trainer ready")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output))
    # The latent marker embedding lives in the *base* embedding matrix, which a
    # LoRA checkpoint does not store. Save those rows explicitly so Stage 2 can
    # restore the trained vector instead of re-initialising it at random.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    embedding = base.get_input_embeddings().weight.detach()
    added_ids = [tokenizer.convert_tokens_to_ids(t) for t in LATENT_TOKENS]
    torch.save({"projector": projector.state_dict(),
                "value_projection": value_projection.state_dict(),
                "added_token_ids": added_ids,
                "added_token_embeddings": embedding[added_ids].float().cpu()},
               output / "latent_modules.pt")
    processor.save_pretrained(str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

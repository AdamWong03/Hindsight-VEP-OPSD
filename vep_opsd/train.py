from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    HfArgumentParser,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    set_seed,
)

from .config import HindsightOPSDConfig
from .data import VEP50KDataset, collate_examples
from .trainer import HindsightOPSDTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str
    attn_implementation: str = "sdpa"
    freeze_vision_tower: bool = True


@dataclass
class DataArguments:
    manifest: str
    video_root: str | None = None
    image_min_pixels: int = 2 * 32 * 32
    image_max_pixels: int = 16384 * 32 * 32


@dataclass
class OPSDArguments:
    candidates_per_example: int = 4
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    teacher_temperature: float = 0.2
    student_temperature: float = 1.0
    minimum_consistency: float = 0.6
    max_new_tokens: int = 512


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, OPSDArguments, TrainingArguments))
    model_args, data_args, opsd_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if config.model_type != "qwen3_vl":
        raise ValueError(f"Hindsight VEP-OPSD v1 requires qwen3_vl, got {config.model_type}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=model_args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    if model_args.freeze_vision_tower:
        for parameter in model.model.visual.parameters():
            parameter.requires_grad_(False)
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    opsd_config = HindsightOPSDConfig(**asdict(opsd_args))
    opsd_config.validate()
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vep_opsd_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "Hindsight VEP-OPSD v1",
                "model": asdict(model_args),
                "data": asdict(data_args),
                "algorithm": opsd_config.to_dict(),
                "training": training_args.to_dict(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    dataset = VEP50KDataset(data_args.manifest, data_args.video_root)
    trainer = HindsightOPSDTrainer(
        model=model,
        processor=processor,
        opsd_config=opsd_config,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_examples,
    )
    checkpoints = list(output_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(checkpoints))
    trainer.save_state()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()


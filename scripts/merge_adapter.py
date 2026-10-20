#!/usr/bin/env python
"""Merge a trained Student LoRA into Qwen3-VL for the existing eval scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="completed OPSD output directory")
    parser.add_argument("--output", required=True, help="standalone merged checkpoint")
    parser.add_argument(
        "--base",
        default=None,
        help="base checkpoint; defaults to adapter_config.json",
    )
    args = parser.parse_args()

    adapter = Path(args.adapter)
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    base = args.base or adapter_config["base_model_name_or_path"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.merge_and_unload().save_pretrained(output, safe_serialization=True)
    AutoProcessor.from_pretrained(str(adapter)).save_pretrained(output)


if __name__ == "__main__":
    main()

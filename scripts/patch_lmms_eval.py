#!/usr/bin/env python
"""Relax the upstream version pins of a local lmms-eval checkout.

lmms-eval pins its dependencies exactly (``datasets==2.16.1``, ``numpy==1.26.4``,
``protobuf==3.20``, ``httpx==0.23.3``, ...). Installed as-is next to a Qwen3-VL
training stack it would downgrade numpy and datasets and break the environment,
so the pins are widened to ``>=`` here. The modules themselves are untouched.

Usage:
    python scripts/patch_lmms_eval.py lmms-eval
    python scripts/patch_lmms_eval.py lmms-eval --check   # exit 1 if still pinned

Safe to run twice: already-relaxed pins are left alone.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (pinned, relaxed) pairs, applied to the [project] dependencies table.
PIN_REWRITES = [
    ("black", "24.1.0"),
    ("isort", "5.13.2"),
    ("datasets", "2.16.1"),
    ("httpx", "0.23.3"),
    ("numpy", "1.26.4"),
    ("sqlitedict", "2.1.0"),
    ("sentencepiece", "0.1.99"),
    ("tenacity", "8.3.0"),
    ("protobuf", "3.20"),
]


def relax(text: str) -> tuple[str, list[str]]:
    changes = []
    for name, version in PIN_REWRITES:
        pinned = f'"{name}=={version}"'
        relaxed = f'"{name}>={version}"'
        if pinned in text:
            text = text.replace(pinned, relaxed)
            changes.append(f"{name}: =={version} -> >={version}")
    return text, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkout", help="path to the lmms-eval checkout")
    parser.add_argument("--check", action="store_true", help="only report, exit 1 when pins remain")
    args = parser.parse_args()

    pyproject = Path(args.checkout) / "pyproject.toml"
    if not pyproject.is_file():
        raise SystemExit(f"no lmms-eval checkout at {args.checkout} (missing pyproject.toml)")

    original = pyproject.read_text(encoding="utf-8")
    updated, changes = relax(original)

    if args.check:
        if changes:
            print("still pinned:\n  " + "\n  ".join(changes))
            sys.exit(1)
        print("lmms-eval pins already relaxed")
        return

    if not changes:
        print("lmms-eval pins already relaxed, nothing to do")
        return
    pyproject.write_text(updated, encoding="utf-8")
    print("relaxed lmms-eval pins:\n  " + "\n  ".join(changes))


if __name__ == "__main__":
    main()

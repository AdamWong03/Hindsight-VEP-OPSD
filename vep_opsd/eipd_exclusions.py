"""Extract source-video IDs from JSON/JSONL manifests into overlap exclusions.

Run once per benchmark and Future-L1-50K source manifest. This is *not*
perceptual near-duplicate detection; the latter requires a separate audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .eipd_data import iter_input

def video_ids(source: str) -> set[str]:
    result = set()
    for row in iter_input(source, None):
        video = row.get("video") or row.get("video_path") or row.get("video_name")
        if not video and isinstance(row.get("meta_data"), dict):
            video = row["meta_data"].get("video")
        if not video:
            continue
        base = Path(str(video)).name
        if base:
            result.add(base)
            result.add(Path(base).stem)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    all_ids = set()
    for src in args.inputs:
        part = video_ids(src)
        if not part:
            raise RuntimeError(f"No video IDs found in {src}; refusing silent empty exclusions")
        all_ids.update(part)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sorted(all_ids)) + "\n", encoding="utf-8")
    print(f"{len(all_ids)} excluded full names/stems saved: {output}")

if __name__ == "__main__":
    main()

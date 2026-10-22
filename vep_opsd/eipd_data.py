"""TwiFF -> audited EIPD manifest. No model-based gain selection or answer generation."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

def stable_score(key: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")

def uniform_pool(vlen: int, n: int = 8) -> list[int]:
    """Exact TwiFF middle-sampling convention (1-based pool references)."""
    if vlen < 2:
        raise ValueError("video_too_short")
    import numpy as np
    start = 0 if 2 >= vlen else 1
    end = vlen - 1 if vlen - 2 <= start else vlen - 2
    count = min(n - 2, end - start)
    steps = np.linspace(start + 1, end, count + 1).astype(int)
    middle = [(steps[i] + steps[i + 1] - 1) // 2 for i in range(len(steps) - 1)]
    result = [start, *middle, end]
    return result + [result[-1]] * (n - len(result))

def parse_answer(raw: str) -> str:
    raw = str(raw or "")
    for tag in ("ans", "answer"):
        found = re.findall(r"<" + tag + r">\s*(.*?)\s*</" + tag + r">", raw, re.I | re.S)
        if found:
            return " ".join(found[-1].split())
    return ""

def normalize_row(row: dict) -> dict:
    video = str(row.get("video") or row.get("video_path") or "").strip()
    obs = row.get("question_images_index", row.get("image", []))
    future = row.get("reasoning_images_index", row.get("reasoning_image", []))
    obs = [int(x) for x in (obs or [])]
    future = [int(x) for x in (future or [])]
    question = row.get("question")
    if question is None:
        turns = row.get("conversations") or []
        if isinstance(turns, dict):
            turns = [dict(zip(turns, vals)) for vals in zip(*turns.values())]
        question = next((x.get("value", "") for x in turns if x.get("from") == "human"), "")
    question = " ".join(re.sub(r"frame_\d+\s*", "", str(question), flags=re.I).replace("<image>", " ").split())
    raw_answer = row.get("answer") or ""
    answer = parse_answer(raw_answer)
    if not answer and "conversations" in row:
        turns = row["conversations"]
        if isinstance(turns, list):
            answer = parse_answer(next((x.get("value", "") for x in turns if x.get("from") == "gpt"), ""))
    metadata = row.get("meta_data") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    category = str(metadata.get("classification") or row.get("classification") or "Unknown").strip()
    return dict(video=video, image=obs, reasoning_image=future, question=question,
                answer=answer, category=category)

def validate_meta(rec: dict) -> str | None:
    if not rec["video"] or Path(rec["video"]).name != rec["video"]:
        return "unsafe_or_missing_video_id"
    if not rec["question"]:
        return "missing_question"
    if not rec["answer"]:
        return "missing_grounded_answer_tag"
    a, b = rec["image"], rec["reasoning_image"]
    if not a or not b:
        return "missing_temporal_indices"
    if min(a + b) < 1 or max(a + b) > 8:
        return "invalid_pool_index"
    if max(a) >= min(b):
        return "temporal_order_violation"
    return None

def iter_input(path: str, hf_dataset: str | None) -> Iterator[dict]:
    if hf_dataset:
        from datasets import load_dataset
        yield from load_dataset(hf_dataset, split="train", streaming=True)
        return
    with Path(path).open("r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            rows = json.load(handle)
            if isinstance(rows, dict):
                rows = [rows]
            yield from rows
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

def resolve_video(video_id: str, video_root: Path, mapping: dict[str, str]) -> Path:
    return Path(mapping[video_id]) if video_id in mapping else video_root / video_id

def inspect_video(rec: dict, root: Path, mapping: dict[str, str]) -> dict:
    import decord
    path = resolve_video(rec["video"], root, mapping)
    if not path.is_file():
        raise ValueError("video_missing")
    reader = decord.VideoReader(str(path), num_threads=1)
    pool = uniform_pool(len(reader))
    t0 = max(pool[i - 1] for i in rec["image"])
    t1 = min(pool[i - 1] for i in rec["reasoning_image"])
    if t0 >= t1:
        raise ValueError("raw_temporal_order_violation")
    bridge = (t0 + t1) // 2
    if not (t0 < bridge < t1):
        raise ValueError("no_strict_bridge")
    if len({pool[i - 1] for i in rec["image"]}) != len(rec["image"]):
        raise ValueError("duplicate_observation_frames")
    # Verify all frames are actually decodable; bridge must never be replaced by a target.
    try:
        reader.get_batch([*map(lambda i: pool[i - 1], rec["image"]), bridge,
                          *map(lambda i: pool[i - 1], rec["reasoning_image"])]).asnumpy()
    except Exception as exc:
        raise ValueError("decode_failure") from exc
    fps = float(reader.get_avg_fps())
    return dict(video_path=str(path.resolve()), observed_raw_frames=[pool[i - 1] for i in rec["image"]],
                future_raw_frames=[pool[i - 1] for i in rec["reasoning_image"]],
                bridge_raw_frame=bridge, video_length=len(reader), fps=fps,
                observed_end_seconds=t0 / fps if fps > 0 else None,
                bridge_seconds=bridge / fps if fps > 0 else None,
                future_start_seconds=t1 / fps if fps > 0 else None)

def build(args) -> dict:
    """Select by video ID BEFORE video decoding. Do not score by model gain."""
    root = Path(args.video_root).resolve()
    mapping = json.loads(Path(args.video_map).read_text()) if args.video_map else {}
    exclude = set()
    for file in args.exclude_video_lists:
        exclude.update(x.strip() for x in Path(file).read_text(encoding="utf-8").splitlines() if x.strip())
    rejects = Counter()
    capacity = max(1, (args.train_size + args.val_size) * args.oversample)
    # Heap entries: (-priority, video_id). Lazy entries are avoided by not replacing existing video IDs.
    heap: list[tuple[int, str]] = []
    selected: dict[str, tuple[int, dict]] = {}
    for raw in iter_input(args.input, args.hf_dataset):
        try:
            rec = normalize_row(raw)
            error = validate_meta(rec)
            if error:
                rejects[error] += 1
                continue
            if rec["video"] in exclude or Path(rec["video"]).stem in exclude:
                rejects["explicit_overlap_exclusion"] += 1
                continue
            if args.categories and rec["category"].lower() not in args.categories:
                rejects["category_excluded"] += 1
                continue
            video = rec["video"]
            if video in selected:
                rejects["duplicate_source_video"] += 1
                continue
            priority = stable_score(video, args.seed)
            if len(heap) >= capacity and priority >= -heap[0][0]:
                continue
            if len(heap) >= capacity:
                _, old_video = heapq.heappop(heap)
                del selected[old_video]
            heapq.heappush(heap, (-priority, video))
            selected[video] = (priority, rec)
        except (ValueError, TypeError, KeyError):
            rejects["malformed_metadata"] += 1
    ordered = sorted(selected.values(), key=lambda entry: entry[0])
    train, val = [], []
    audit = []
    for priority, rec in ordered:
        if len(train) >= args.train_size and len(val) >= args.val_size:
            break
        split = "val" if (stable_score("split:" + rec["video"], args.seed) % 10**6) < int(args.val_fraction * 10**6) else "train"
        target = val if split == "val" else train
        limit = args.val_size if split == "val" else args.train_size
        if len(target) >= limit:
            continue
        try:
            video = inspect_video(rec, root, mapping)
        except (ValueError, OSError, RuntimeError) as exc:
            rejects[str(exc)] += 1
            continue
        record = {**rec, **video, "id": hashlib.sha256((rec["video"] + "\0" + rec["question"]).encode()).hexdigest()[:20],
                  "source_dataset": args.hf_dataset or args.input, "split": split,
                  "selection": "deterministic_video_hash_not_teacher_gain"}
        target.append(record)
        audit.append(dict(id=record["id"], video=rec["video"], split=split, category=rec["category"],
                          bridge_gap_frames=record["future_raw_frames"][0] - record["bridge_raw_frame"]))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        with (out / (split + ".jsonl")).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = dict(train=len(train), val=len(val), requested_train=args.train_size,
                   requested_val=args.val_size, candidate_capacity=capacity, rejected=dict(rejects),
                   seed=args.seed, selection="stable hash of original video filename",
                   note="Metadata and decoder validation only; semantic correctness and benchmark decontamination require separate audits")
    (out / "audit.json").write_text(json.dumps(dict(summary=summary, samples=audit), indent=2), encoding="utf-8")
    if args.strict_counts and (len(train) != args.train_size or len(val) != args.val_size):
        raise RuntimeError(f"Insufficient valid samples: {summary}; partial manifests and audit retained")
    return summary

def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="TwiFF JSON/JSONL metadata")
    src.add_argument("--hf-dataset", help="e.g. Liu-Junhua/TwiFF-2.7M")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--video-map", help="JSON basename-to-path mapping, if clips are nested")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-video-lists", nargs="*", default=[], help="One video ID or basename stem per line; include benchmarks and Future-L1-50K")
    parser.add_argument("--categories", nargs="*", default=[], type=str.lower)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-counts", action="store_true")
    args = parser.parse_args()
    if args.train_size <= 0 or args.val_size <= 0 or not 0 < args.val_fraction < 1 or args.oversample < 1:
        parser.error("Invalid requested sizes/fraction/oversample")
    print(json.dumps(build(args), indent=2))

if __name__ == "__main__":
    cli()

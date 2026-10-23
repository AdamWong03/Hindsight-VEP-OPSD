"""Production EIPD-Data-v1: metadata planning, lossless frame materialization, verification.

No Teacher-gain filtering, no answer generation, no video redistribution.
Use a full TwiFF source scan, source-video grouping and reproducible priority.
The separate data/manifest paths are portable across servers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from .eipd_data import input_files, iter_input, normalize_row, uniform_pool, validate_meta

SCHEMA_VERSION = "eipd-data-v1.0"
CATEGORIES = ("Predictive", "Instructional", "Camera")
TRAIN_QUOTAS = {"Predictive": 30000, "Instructional": 15000, "Camera": 5000}
VAL_QUOTAS = {"Predictive": 3000, "Instructional": 1500, "Camera": 500}
VIDEO_PATTERN = re.compile(r"^([A-Za-z0-9_-]{11})_([0-9]+)\.mp4$")
YOUTUBE_URL = re.compile(r"(?:youtube\.com/(?:watch\?.*?v=|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(2 << 20), b""):
            h.update(part)
    return h.hexdigest()


def metadata_fingerprint(path: str) -> tuple[str, list[dict]]:
    """Stable fingerprint of every local metadata shard and its relative name."""

    source = Path(path).expanduser().resolve()
    files = input_files(path)
    entries = []
    digest = hashlib.sha256()
    for file in files:
        relative = str(file.relative_to(source)) if source.is_dir() else file.name
        item = {"path": relative, "size": file.stat().st_size, "sha256": sha256_file(file)}
        entries.append(item)
        digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), entries

def score(text: str, seed: int) -> str:
    return sha256_bytes(f"{seed}\t{text}".encode("utf-8"))

def source_id(filename: str) -> str:
    name = Path(filename).name
    match = VIDEO_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Cannot establish original-source-video ID: {filename}")
    return match.group(1)

def classify(value: str) -> str:
    return {c.lower(): c for c in CATEGORIES}.get(str(value).strip().lower(), "")

def root_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def db_open(root: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(root / "state.sqlite"), timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS candidates(
        source TEXT PRIMARY KEY, priority TEXT NOT NULL, split TEXT NOT NULL,
        category TEXT NOT NULL, payload TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS completed(
        source TEXT PRIMARY KEY, status TEXT NOT NULL,
        reason TEXT, payload TEXT
    )""")
    return con

def read_exclusions(paths: list[str]) -> tuple[set[str], dict]:
    if not paths:
        raise ValueError("Three reviewed exclusion lists required (Future-L1, FutureBench, TwiFF-Bench)")
    excluded = set()
    coverage = {}
    for path in paths:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(file)
        lines = {x.strip() for x in file.read_text(encoding="utf-8").splitlines()
                 if x.strip() and not x.lstrip().startswith("#")}
        if not lines:
            raise ValueError(f"Exclusion file is empty: {file}. Verify its source metadata.")
        for token in lines:
            excluded.add(token)
            try:
                excluded.add(source_id(token))
            except ValueError:
                pass
        coverage[str(file)] = {"entries": len(lines), "sha256": sha256_file(file)}
    return excluded, coverage

def _config(root: Path, configuration: dict, name: str) -> None:
    path = root / name
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise ValueError(f"{path} changed. Construct a NEW data version directory.")
    else:
        path.write_text(json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8")

def _split(source: str, seed: int) -> str:
    # Fixed 10% grouping; source video never enters both sets.
    return "val" if int(score("split:"+source, seed)[:12], 16) % 10 == 0 else "train"

def plan(args) -> dict:
    root = root_dir(args.output)
    exclusion, coverage = read_exclusions(args.exclusions)
    train_quotas, val_quotas = dict(TRAIN_QUOTAS), dict(VAL_QUOTAS)
    if args.train_quotas:
        train_quotas = json.loads(args.train_quotas)
    if args.val_quotas:
        val_quotas = json.loads(args.val_quotas)
    for quotas in (train_quotas, val_quotas):
        if set(quotas) != set(CATEGORIES) or any(type(x) is not int or x < 0 for x in quotas.values()):
            raise ValueError(f"Quotas must specify nonnegative integers for {CATEGORIES}")
    if args.reserve_multiplier < 1:
        raise ValueError("reserve-multiplier must be >=1")
    hf_revision = None
    local_sha = None
    if args.hf_dataset:
        from huggingface_hub import HfApi
        hf_revision = getattr(args, "hf_revision", None) or HfApi().dataset_info(args.hf_dataset).sha
        if not hf_revision:
            raise RuntimeError("Cannot establish immutable Hugging Face source revision")
    else:
        local_sha, local_files = metadata_fingerprint(args.input)
    cfg = {"schema": SCHEMA_VERSION, "seed": args.seed,
           "source": args.hf_dataset or str(Path(args.input).resolve()),
           "source_revision": hf_revision, "source_file_sha256": local_sha,
           "source_files": None if args.hf_dataset else local_files,
           "exclusions": coverage, "train_quotas": train_quotas,
           "val_quotas": val_quotas, "reserve_multiplier": args.reserve_multiplier}
    _config(root, cfg, "plan_config.json")
    con = db_open(root)
    scanned = 0
    errors = Counter()
    try:
        if args.hf_dataset:
            from datasets import load_dataset
            rows = load_dataset(args.hf_dataset, split="train", streaming=True,
                                revision=hf_revision)
        else:
            rows = iter_input(args.input, None)
        for raw in rows:
            scanned += 1
            try:
                rec = normalize_row(raw)
                original_question = raw.get("question", "")
                if original_question and str(original_question).count("<image>") != len(rec["image"]):
                    errors["observed_placeholder_mismatch"] += 1
                    continue
                err = validate_meta(rec, require_source_time=True)
                if err:
                    errors[err] += 1
                    continue
                src = source_id(rec["video"])
                if rec["video"] in exclusion or Path(rec["video"]).stem in exclusion or src in exclusion:
                    errors["source_excluded"] += 1
                    continue
                category = classify(rec["category"])
                if not category:
                    errors["unknown_category"] += 1
                    continue
                # Reject scripts that are neither paired to their visual anchors nor answer-groundable.
                if len(rec["answer"]) > args.max_answer_chars or len(rec["sft_response"]) > args.max_reason_chars:
                    errors["overlength_target"] += 1
                    continue
                rec["category"] = category
                rec["source_id"] = src
                split = _split(src, args.seed)
                priority = score("rank:"+src, args.seed)
                # Deterministic tie-breaking for duplicate clips of one source.
                priority = priority + ":" + score(rec["video"]+"|"+rec["question"], args.seed)
                con.execute("""INSERT INTO candidates(source,priority,split,category,payload)
                               VALUES (?,?,?,?,?)
                               ON CONFLICT(source) DO UPDATE SET
                                  priority=excluded.priority, split=excluded.split,
                                  category=excluded.category, payload=excluded.payload
                               WHERE excluded.priority < candidates.priority""",
                            (src, priority, split, category, json.dumps(rec, ensure_ascii=False)))
            except (TypeError, ValueError, KeyError):
                errors["malformed_record"] += 1
            if scanned % 10000 == 0:
                con.commit()
        con.commit()
        counts = {}
        for split, quotas in (("train",train_quotas),("val",val_quotas)):
            for category, count in quotas.items():
                available = con.execute("SELECT COUNT(*) FROM candidates WHERE split=? AND category=?",
                                        (split,category)).fetchone()[0]
                counts[f"{split}/{category}"] = available
                if available < count:
                    raise RuntimeError(f"Only {available} eligible source videos for {split}/{category}, need {count}")
        target = root / "candidates.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for split, quotas in (("train", train_quotas), ("val", val_quotas)):
                for category, count in quotas.items():
                    cursor = con.execute("""SELECT payload FROM candidates WHERE split=? AND category=?
                                           ORDER BY priority ASC LIMIT ?""",
                                         (split, category, count * args.reserve_multiplier))
                    for priority_index, (payload,) in enumerate(cursor):
                        rec = json.loads(payload)
                        rec.update(split=split, selection_rank=priority_index,
                                   sample_id=score(rec["video"]+"|"+rec["question"], args.seed)[:24],
                                   schema_version=SCHEMA_VERSION)
                        handle.write(json.dumps(rec, ensure_ascii=False)+"\n")
        selected_sources = set()
        with target.open("r",encoding="utf-8") as candidates:
            for line in candidates:
                if line.strip():
                    selected_sources.add(json.loads(line)["source_id"])
        (root/"candidate_source_ids.txt").write_text(
            "\n".join(sorted(selected_sources))+"\n",encoding="utf-8")
        report = {"scanned": scanned, "eligible_by_group": counts, "rejected": dict(errors),
                  "candidate_sha256": sha256_file(target), "quotas": {"train":train_quotas,"val":val_quotas}}
        (root/"plan_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        return report
    finally:
        con.close()

def video_location(rec: dict, video_root: Path, lookup: dict[str,str]) -> Path:
    key = rec["video"]
    value = lookup.get(key, str(video_root / key))
    return Path(value).expanduser().resolve()

def _load_lookup(path: str | None) -> dict[str,str]:
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".json":
        lookup = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(lookup, dict):
            raise ValueError("video map must be a dictionary")
        return {str(k):str(v) for k,v in lookup.items()}
    result = {}
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["video"])] = str(row["path"])
    return result

def extract_frames(rec: dict, video: Path, dest: Path, image_format: str, jpeg_quality: int) -> dict:
    import decord
    from PIL import Image, ImageChops, ImageStat
    reader = decord.VideoReader(str(video), num_threads=1)
    pool = uniform_pool(len(reader))
    past = [pool[i-1] for i in rec["image"]]
    future = [pool[i-1] for i in rec["reasoning_image"]]
    left, right = max(past), min(future)
    bridge = (left + right) // 2
    if not left < bridge < right:
        raise ValueError("no_strict_native_bridge")
    if len(set(past)) != len(past) or len(set(future)) != len(future):
        raise ValueError("duplicated_native_timepoint")
    indices = past + [bridge] + future
    frames = reader.get_batch(indices).asnumpy()
    images = [Image.fromarray(x).convert("RGB") for x in frames]
    b = images[len(past)]
    avg_diff = float(sum(ImageStat.Stat(ImageChops.difference(images[len(past)-1], b)).mean)/3.0)
    if not any(ch.getbbox() for ch in ImageChops.difference(images[len(past)-1], b).split()):
        raise ValueError("bridge_identical_to_last_past")
    ext = "png" if image_format == "png" else "jpg"
    def save(img, name):
        file = dest / f"{name}.{ext}"
        if ext == "png":
            img.save(file, "PNG", compress_level=3)
        else:
            img.save(file, "JPEG", quality=jpeg_quality, subsampling=0)
        return str(file.name)
    paths_past = [save(im, f"past_{j:02d}") for j,im in enumerate(images[:len(past)])]
    bpath = save(b,"bridge")
    paths_future = [save(im, f"future_{j:02d}") for j,im in enumerate(images[len(past)+1:])]
    fps = float(reader.get_avg_fps())
    sizes = [len((dest / x).read_bytes()) for x in paths_past + [bpath] + paths_future]
    return {"past_files": paths_past, "bridge_file": bpath, "future_files": paths_future,
            "frame_sha256": {x: sha256_file(dest / x) for x in paths_past + [bpath] + paths_future},
            "past_native_indices": past, "bridge_native_index": bridge, "future_native_indices": future,
            "fps": fps, "native_video_frames": len(reader), "image_format": ext,
            "frame_storage_bytes": sum(sizes), "bridge_mean_abs_pixel_delta": avg_diff,
            "native_time_seconds": {"past_end":left/fps, "bridge":bridge/fps,"future_start":right/fps}
            if fps > 0 else None}

def _final_manifest(root: Path, con: sqlite3.Connection, cfg: dict) -> dict:
    total, categories = {}, {}
    for split in ("train", "val"):
        wanted = cfg[f"{split}_quotas"]
        records = []
        for cat, limit in wanted.items():
            matches = []
            for (text,) in con.execute(
                    """SELECT completed.payload FROM completed JOIN candidates USING(source)
                       WHERE completed.status='done' AND candidates.split=?
                       AND candidates.category=? ORDER BY candidates.priority ASC""",
                    (split,cat)):
                matches.append(json.loads(text))
            records.extend(matches[:limit])
            if len(matches) < limit:
                raise RuntimeError(f"Dataset incomplete: {split}/{cat} {len(matches)}/{limit}")
        records.sort(key=lambda r:(r["category"],r["source_id"]))
        out = root / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as file:
            for r in records:
                file.write(json.dumps(r, ensure_ascii=False)+"\n")
        total[split] = len(records)
        categories[split] = dict(Counter(r["category"] for r in records))
    if {x["source_id"] for x in map(json.loads, (root/"train.jsonl").read_text(encoding="utf-8").splitlines())} & \
       {x["source_id"] for x in map(json.loads, (root/"val.jsonl").read_text(encoding="utf-8").splitlines())}:
        raise RuntimeError("SOURCE VIDEO LEAKAGE train/val")
    result = {"schema": SCHEMA_VERSION, "counts":total, "categories":categories,
              "manifest_sha256":{s:sha256_file(root/f"{s}.jsonl") for s in ("train","val")},
              "candidate_sha256":sha256_file(root/"candidates.jsonl"),
              "note":"Structural/video QA complete, semantic correctness still requires manual audit."}
    (root/"dataset_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def materialize(args):
    root = root_dir(args.output)
    cfg = json.loads((root/"plan_config.json").read_text(encoding="utf-8"))
    con = db_open(root)
    video_root = Path(args.video_root).resolve()
    lookup = _load_lookup(args.video_map)
    if args.image_format not in ("png", "jpeg"):
        raise ValueError("image-format must be png or jpeg")
    _config(root, {"image_format":args.image_format,"jpeg_quality":args.jpeg_quality,
                   "schema":SCHEMA_VERSION}, "materialize_config.json")
    rows = [json.loads(line) for line in (root/"candidates.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    accepted = Counter()
    failures = Counter()
    for index, rec in enumerate(rows):
        split, cat, src = rec["split"], rec["category"], rec["source_id"]
        quota = cfg[f"{split}_quotas"][cat]
        if accepted[(split,cat)] >= quota:
            continue
        previous = con.execute("SELECT status,payload FROM completed WHERE source=?", (src,)).fetchone()
        if previous:
            if previous[0] == "done":
                existing = json.loads(previous[1])
                if (root/existing["past_files"][0]).exists() and (root/existing["bridge_file"]).exists():
                    accepted[(split,cat)] += 1
                    continue
                con.execute("DELETE FROM completed WHERE source=?", (src,))
            else:
                failures[previous[0]] += 1
                continue
        video = video_location(rec, video_root, lookup)
        if not video.is_file():
            failures["video_missing"] += 1
            # Missing may become available after later download; do not record terminal state.
            continue
        d = root/"frames"/split/rec["sample_id"]
        tmp = d.with_name(d.name+".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            extra = extract_frames(rec, video, tmp, args.image_format, args.jpeg_quality)
            if d.exists():
                shutil.rmtree(d)
            tmp.replace(d)
            relative = str(d.relative_to(root))
            file_hashes = extra.pop("frame_sha256")
            data = {**rec, "video_path":None, "origin_video_sha256":sha256_file(video),
                    "frame_sha256":{f"{relative}/{filename}": digest for filename,digest in file_hashes.items()},
                    "past_files":[f"{relative}/{x}" for x in extra.pop("past_files")],
                    "bridge_file":f"{relative}/{extra.pop('bridge_file')}",
                    "future_files":[f"{relative}/{x}" for x in extra.pop("future_files")],
                    **extra}
            con.execute("INSERT OR REPLACE INTO completed VALUES(?,?,?,?)",
                        (src,"done",None,json.dumps(data,ensure_ascii=False)))
            con.commit()
            accepted[(split,cat)] += 1
        except Exception as exc:
            failures[type(exc).__name__+":"+str(exc)[:100]] += 1
            con.execute("INSERT OR REPLACE INTO completed VALUES(?,?,?,?)",
                        (src,"bad",str(exc)[:400],None))
            con.commit()
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        if (index+1)%200 == 0:
            print(json.dumps({"candidate":index+1,"accepted":{f"{a}/{b}":n for (a,b),n in accepted.items()},
                              "failed":dict(failures)},ensure_ascii=False),flush=True)
    report = {"accepted_this_pass":{f"{a}/{b}":n for (a,b),n in accepted.items()},
              "failures_this_pass":dict(failures)}
    (root/"materialize_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    try:
        return _final_manifest(root, con, cfg)
    finally:
        con.close()

def verify(args):
    from PIL import Image
    root = Path(args.output).resolve()
    config = json.loads((root/"dataset_summary.json").read_text(encoding="utf-8"))
    seen = {}
    check = Counter()
    for split in ("train","val"):
        if sha256_file(root/f"{split}.jsonl") != config["manifest_sha256"][split]:
            raise RuntimeError("Manifest checksum mismatch")
        for raw in (root/f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            src = r["source_id"]
            if src in seen:
                raise RuntimeError(f"source video appears more than once: {src}")
            seen[src] = split
            if not max(r["past_native_indices"]) < r["bridge_native_index"] < min(r["future_native_indices"]):
                raise RuntimeError(f"Bridge temporal ordering invalid: {r['sample_id']}")
            if r["sft_response"].count("<image>") != len(r["future_files"]):
                raise RuntimeError("SFT image/text alignment broken")
            for key in ("past_files","future_files"):
                for rel in r[key]:
                    p=(root/rel).resolve()
                    if not p.is_relative_to(root):
                        raise RuntimeError("Unsafe frame path")
                    with Image.open(p) as im:
                        im.verify()
                    if sha256_file(p) != r["frame_sha256"][rel]:
                        raise RuntimeError(f"Corrupt or replaced training image: {rel}")
                    check["images"]+=1
            bridge=(root/r["bridge_file"]).resolve()
            if not bridge.is_relative_to(root):
                raise RuntimeError("Unsafe bridge path")
            with Image.open(bridge) as im: im.verify()
            if sha256_file(bridge) != r["frame_sha256"][r["bridge_file"]]:
                raise RuntimeError(f"Corrupt or replaced Bridge: {bridge}")
            check["images"]+=1
            check[split]+=1
    return {"verified":dict(check),"source_unique":len(seen),"status":"structure verified; semantic audit pending"}

def panda_subset(args):
    wanted = set()
    for raw in Path(args.video_ids).read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        wanted.add(name if re.fullmatch(r"[A-Za-z0-9_-]{11}", name) else source_id(name))
    matches = set()
    with Path(args.csv).open("r",encoding="utf-8",newline="") as inp, \
         Path(args.output_csv).open("w",encoding="utf-8",newline="") as out:
        reader=csv.DictReader(inp)
        if not reader.fieldnames or "url" not in reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError("Expected official Panda-70M CSV fields url, timestamp")
        writer=csv.DictWriter(out,fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            match=YOUTUBE_URL.search(row.get("url",""))
            if match and match.group(1) in wanted:
                writer.writerow(row)
                matches.add(match.group(1))
    print(json.dumps({"matched_source_videos":len(matches), "requested":len(wanted),
                      "important":"Official video2dataset emits numeric clip paths; TwiFF clip names must be mapped and time-aligned before EIPD extraction."}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("plan",help="Stream all 2.7M metadata, source-group deduplicate, freeze candidate rankings")
    source=a.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--hf-dataset")
    a.add_argument("--hf-revision",help="Optional immutable Hugging Face dataset commit SHA")
    a.add_argument("--output",required=True)
    a.add_argument("--exclusions",nargs=3,required=True,help="Future-L1, FutureBench and TwiFF-Bench reviewed source lists")
    a.add_argument("--train-quotas",help="JSON map by category; defaults to 30k/15k/5k")
    a.add_argument("--val-quotas",help="JSON map by category; defaults to 3k/1.5k/0.5k")
    a.add_argument("--reserve-multiplier",type=int,default=4)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--max-answer-chars",type=int,default=750)
    a.add_argument("--max-reason-chars",type=int,default=8000)
    b=sub.add_parser("materialize",help="Decode local original videos and losslessly cache all EIPD views")
    b.add_argument("--output",required=True)
    b.add_argument("--video-root",required=True)
    b.add_argument("--video-map",help="JSON mapping TwiFF video basename to matching original local mp4")
    b.add_argument("--image-format",choices=("png","jpeg"),default="png")
    b.add_argument("--jpeg-quality",type=int,default=98)
    c=sub.add_parser("verify",help="Read all cached images and enforce temporal/source/leakage invariants")
    c.add_argument("--output",required=True)
    d=sub.add_parser("panda-subset",help="Find source videos in official Panda-70M CSV; does not guess clip mapping")
    d.add_argument("--csv",required=True)
    d.add_argument("--video-ids",required=True)
    d.add_argument("--output-csv",required=True)
    args=p.parse_args()
    result=plan(args) if args.mode=="plan" else (
        materialize(args) if args.mode=="materialize" else (
        verify(args) if args.mode=="verify" else panda_subset(args)))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == "__main__":
    main()

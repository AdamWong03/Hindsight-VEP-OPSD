"""Restartable exact TwiFF clip acquisition from Panda/YouTube source videos."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .eipd_data import parse_timestamp


TERMINAL = {"GONE", "PRIVATE", "MEMBERS_ONLY", "AGE_RESTRICTED", "BAD_METADATA"}
SUCCESS = {"OK", "EXISTS", "REUSED"}
FORMAT_720P = (
    "bv*[height<=720][ext=mp4]/bv*[height<=720]/"
    "b[height<=720][ext=mp4]/b[height<=720]"
)


@dataclass(frozen=True)
class Task:
    video: str
    source_id: str
    start: float
    end: float
    split: str
    category: str
    selection_rank: int


def task_shard(task: Task, count: int) -> int:
    """Stable source-level partition for independent proxy workers."""

    digest = hashlib.sha256(task.source_id.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % count


def valid_video(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        return None
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return {
            "bytes": path.stat().st_size,
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration": float(data["format"]["duration"]),
            "avg_frame_rate": stream.get("avg_frame_rate"),
        }
    except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
        return None


def classify_error(detail: str) -> str:
    text = detail.lower()
    if "private video" in text:
        return "PRIVATE"
    if "members-only" in text:
        return "MEMBERS_ONLY"
    if "confirm your age" in text or "age-restricted" in text:
        return "AGE_RESTRICTED"
    if any(x in text for x in (
        "video unavailable", "video is unavailable", "been removed", "does not exist",
        "uploader has closed", "account associated with this video has been terminated",
    )):
        return "GONE"
    if "sign in to confirm" in text or "not a bot" in text or "please sign in" in text:
        return "AUTH_OR_BOT"
    if "429" in text or "too many requests" in text or "rate-limit" in text:
        return "RATE_LIMIT"
    if "403" in text:
        return "FORBIDDEN"
    if "timed out" in text:
        return "TIMEOUT"
    return "ERROR"


def load_tasks(root: Path, factor: float) -> list[Task]:
    config = json.loads((root / "plan_config.json").read_text(encoding="utf-8"))
    quotas = {
        (split, category): count
        for split in ("train", "val")
        for category, count in config[f"{split}_quotas"].items()
    }
    limits = {key: max(value, int(value * factor + 0.999)) for key, value in quotas.items()}
    groups: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for line in (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["split"], row["category"])
        if len(groups[key]) >= limits[key]:
            continue
        try:
            start = parse_timestamp(row["source_start"])
            end = parse_timestamp(row["source_end"])
            if end <= start:
                raise ValueError
            task = Task(
                video=row["video"], source_id=row["source_id"], start=start, end=end,
                split=row["split"], category=row["category"],
                selection_rank=int(row["selection_rank"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Candidate lacks exact source time mapping: {row.get('video')}") from exc
        groups[key].append(task)
    missing = {key: limits[key] - len(groups[key]) for key in limits if len(groups[key]) < limits[key]}
    if missing:
        raise RuntimeError(f"Insufficient planned download reserve: {missing}")
    return sorted(
        (task for rows in groups.values() for task in rows),
        key=lambda task: (task.selection_rank, task.split, task.category, task.video),
    )


def state_open(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(root / "download_state.sqlite", timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS downloads(
        video TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL,
        updated REAL NOT NULL, payload TEXT NOT NULL)"""
    )
    return connection


def yt_command(args) -> list[str]:
    command = [
        args.python, "-m", "yt_dlp", "--no-playlist", "--js-runtimes", "node",
        "--socket-timeout", "30", "--retries", "5", "--fragment-retries", "5",
        "-f", FORMAT_720P,
    ]
    if args.player_client:
        command.extend(["--extractor-args", f"youtube:player_client={args.player_client}"])
    if args.sleep_requests:
        command.extend(["--sleep-requests", str(args.sleep_requests)])
    if args.proxy:
        command.extend(["--proxy", args.proxy])
    if args.cookies:
        command.extend(["--cookies", args.cookies])
    if args.remote_ejs:
        command.extend(["--remote-components", "ejs:github"])
    return command


def acquire(task: Task, args, output: Path, reuse_roots: list[Path]) -> dict:
    started = time.time()
    destination = output / task.video
    existing = valid_video(destination)
    if existing:
        return {**task.__dict__, "status": "EXISTS", **existing}
    for root in reuse_roots:
        source = root / task.video
        info = valid_video(source)
        if info:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
                mode = "hardlink"
            except OSError:
                shutil.copy2(source, destination)
                mode = "copy"
            return {**task.__dict__, "status": "REUSED", "reuse_mode": mode, **info}

    output.mkdir(parents=True, exist_ok=True)
    partial = output / ".partial"
    partial.mkdir(exist_ok=True)
    url = f"https://www.youtube.com/watch?v={task.source_id}"
    with tempfile.TemporaryDirectory(prefix=f"{task.source_id}_", dir=partial) as tmp:
        target = Path(tmp) / "clip.%(ext)s"
        command = yt_command(args)
        command.extend(
            [
                "--quiet", "--no-warnings", "--download-sections",
                f"*{task.start:.3f}-{task.end:.3f}", "--force-keyframes-at-cuts",
                "--recode-video", "mp4", "-o", str(target), url,
            ]
        )
        time.sleep(random.uniform(args.min_sleep, args.max_sleep))
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            result = None
        candidates = sorted(Path(tmp).glob("clip*.mp4"))
        info = valid_video(candidates[-1]) if candidates else None
        if result is not None and result.returncode == 0 and info:
            os.replace(candidates[-1], destination)
            return {
                **task.__dict__, "status": "OK", "url": url,
                "elapsed_seconds": round(time.time() - started, 3), **info,
            }
        detail = "subprocess timeout" if result is None else (result.stdout + result.stderr)[-3000:]
        return {
            **task.__dict__, "status": classify_error(detail), "url": url,
            "detail": detail, "elapsed_seconds": round(time.time() - started, 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="EIPD-Data-v1 root")
    parser.add_argument("--clip-root", default=None)
    parser.add_argument("--reserve-factor", type=float, default=1.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 means all scheduled")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--reuse-root", action="append", default=[])
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--remote-ejs", action="store_true")
    parser.add_argument("--retry-terminal", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--min-sleep", type=float, default=1.0)
    parser.add_argument("--max-sleep", type=float, default=4.0)
    parser.add_argument("--sleep-requests", type=float, default=0.75)
    parser.add_argument("--proxy", default=None, help="Explicit yt-dlp proxy URL")
    parser.add_argument(
        "--player-client", default=None,
        help="Optional YouTube player client override; defaults to yt-dlp auto-selection",
    )
    parser.add_argument(
        "--auth-circuit-breaker", type=int, default=16,
        help="Stop scheduling after this many auth/bot failures without a success (0 disables)",
    )
    args = parser.parse_args()
    if (args.reserve_factor < 1 or not 1 <= args.workers <= 32
            or args.min_sleep < 0 or args.max_sleep < args.min_sleep
            or args.sleep_requests < 0 or args.auth_circuit_breaker < 0
            or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count):
        parser.error("invalid reserve, worker (must be 1-32), sleep, or circuit-breaker setting")
    root = Path(args.output).expanduser().resolve()
    output = Path(args.clip_root).expanduser().resolve() if args.clip_root else root / "raw_clips"
    reuse = [Path(value).expanduser().resolve() for value in args.reuse_root]
    tasks = load_tasks(root, args.reserve_factor)
    tasks = [task for task in tasks if task_shard(task, args.shard_count) == args.shard_index]
    connection = state_open(root)
    previous = dict(connection.execute("SELECT video,status FROM downloads"))
    eligible = [
        task for task in tasks
        if not (previous.get(task.video) in TERMINAL and not args.retry_terminal)
        and not (
            previous.get(task.video) in SUCCESS
            and (output / task.video).is_file()
            and (output / task.video).stat().st_size > 0
        )
    ]
    # New candidates are the useful frontier. Historical transient failures are
    # retained, but retried only after never-attempted reserve candidates so a
    # stale block of age/auth/network errors cannot starve the whole build.
    pending = (
        [task for task in eligible if task.video not in previous]
        + [task for task in eligible if task.video in previous]
    )
    if args.max_tasks:
        pending = pending[: args.max_tasks]
    plan = (root / "download_plan.jsonl" if args.shard_count == 1 else
            root / f"download_plan.shard-{args.shard_index:02d}-of-{args.shard_count:02d}.jsonl")
    with plan.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({**task.__dict__, "url": f"https://www.youtube.com/watch?v={task.source_id}"}) + "\n")
    print(json.dumps({"planned": len(tasks), "pending": len(pending), "clip_root": str(output)}), flush=True)

    counts = Counter()
    audit = (root / "logs" / "clip-download.jsonl" if args.shard_count == 1 else
             root / "logs" / f"clip-download.shard-{args.shard_index:02d}-of-{args.shard_count:02d}.jsonl")
    audit.parent.mkdir(parents=True, exist_ok=True)
    auth_failures_without_success = 0
    circuit_open = False
    completed = 0

    def record_result(future, task: Task) -> str:
        nonlocal completed
        try:
            record = future.result()
        except Exception as exc:
            record = {**task.__dict__, "status": "WORKER_ERROR", "detail": repr(exc)}
        status = record["status"]
        counts[status] += 1
        attempts = connection.execute(
            "SELECT attempts FROM downloads WHERE video=?", (task.video,)
        ).fetchone()
        connection.execute(
            "INSERT OR REPLACE INTO downloads VALUES(?,?,?,?,?)",
            (task.video, status, (attempts[0] if attempts else 0) + 1,
             time.time(), json.dumps(record, ensure_ascii=False)),
        )
        connection.commit()
        with audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed += 1
        print(
            json.dumps({"completed": completed, "total": len(pending), "status": status,
                        "counts": dict(counts)}, ensure_ascii=False),
            flush=True,
        )
        return status

    # Keep only a small bounded number of subprocesses in flight. This matters
    # when a hosting provider starts returning authentication/rate-limit errors:
    # submitting the complete reserve at once would consume it before a caller
    # can stop the run.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(pending)
        futures = {}
        for task in iterator:
            futures[pool.submit(acquire, task, args, output, reuse)] = task
            if len(futures) >= args.workers:
                break
        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task = futures.pop(future)
                status = record_result(future, task)
                if status in SUCCESS:
                    auth_failures_without_success = 0
                    # Authentication failures return much faster than successful
                    # video cuts. A later success from the same in-flight batch
                    # proves the endpoint is usable and cancels a provisional trip.
                    circuit_open = False
                elif status == "AUTH_OR_BOT":
                    auth_failures_without_success += 1
                if (args.auth_circuit_breaker
                        and auth_failures_without_success >= args.auth_circuit_breaker):
                    circuit_open = True
                if not circuit_open:
                    next_task = next(iterator, None)
                    if next_task is not None:
                        futures[pool.submit(acquire, next_task, args, output, reuse)] = next_task
    connection.close()
    summary = {"tasks": len(pending), "completed": completed, "counts": dict(counts),
               "circuit_open": circuit_open}
    print(json.dumps(summary, indent=2))
    if circuit_open:
        print(
            "Authentication circuit breaker opened; pending candidates were preserved for retry.",
            file=sys.stderr,
        )
        raise SystemExit(75)


if __name__ == "__main__":
    main()

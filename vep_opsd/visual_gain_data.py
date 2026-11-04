"""Frozen, Past-only data and question-matched video permutation for VEP audits.

This module deliberately needs only the Python standard library. Gold answers
are never passed to the matching function.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def question_words(question: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", question.lower())


def source_id(row: dict) -> str:
    value = row.get('source_id')
    if not value:
        raise ValueError(f"sample {row.get('sample_id')} lacks original source_id")
    return str(value)


def matched_permutation(inputs: list[dict], seed: int = 42) -> tuple[list[int], dict]:
    """One-to-one donors, same category and different original source.

    Prefer exact Past frame counts, then question TF-IDF cosine. Pairing gives a
    genuine permutation rather than repeatedly choosing a popular donor. Odd
    strata leave a triple cycle; singleton frame strata are matched within their
    category with their frame-count difference explicitly recorded.
    """
    rows = [{k: row[k] for k in ('sample_id', 'source_id', 'category', 'question', 'past_files')}
            for row in inputs]
    words = [Counter(question_words(row['question'])) for row in rows]
    df = Counter(word for tokens in words for word in tokens)
    vectors = []
    for tokens in words:
        vector = {word: count * (math.log((1 + len(rows)) / (1 + df[word])) + 1)
                  for word, count in tokens.items()}
        norm = math.sqrt(sum(x * x for x in vector.values())) or 1.
        vectors.append({word: value / norm for word, value in vector.items()})

    def similarity(a, b):
        left, right = vectors[a], vectors[b]
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(word, 0.) for word, value in left.items())

    def allowed(a, b):
        return a != b and source_id(rows[a]) != source_id(rows[b])

    rng = random.Random(seed)
    donors = [-1] * len(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['category'], len(row['past_files']))].append(i)
    leftovers = defaultdict(list)

    def assign(indices, category, allow_count_mismatch=False):
        remaining = list(indices)
        rng.shuffle(remaining)
        while len(remaining) >= 2:
            if len(remaining) == 3 and all(allowed(a, b) for a in remaining for b in remaining if a != b):
                a, b, c = remaining
                donors[a], donors[b], donors[c] = b, c, a
                remaining.clear()
                break
            a = remaining.pop()
            candidates = [b for b in remaining if allowed(a, b)]
            if not candidates:
                leftovers[category].append(a)
                continue
            def key(b):
                frame_gap = abs(len(rows[a]['past_files']) - len(rows[b]['past_files']))
                return (frame_gap if allow_count_mismatch else 0, -similarity(a, b), b)
            b = min(candidates, key=key)
            remaining.remove(b)
            donors[a], donors[b] = b, a
        leftovers[category].extend(remaining)

    for (category, _), indices in sorted(groups.items()):
        assign(indices, category)
    pending = dict(leftovers)
    leftovers.clear()
    for category, indices in sorted(pending.items()):
        assign(indices, category, allow_count_mismatch=True)
    # Incorporate a singleton (or same-source leftover) into an existing pair.
    for category, indices in leftovers.items():
        for a in indices:
            candidates = [(b, donors[b]) for b, row in enumerate(rows)
                          if row['category'] == category and donors[b] >= 0
                          and donors[donors[b]] == b and b < donors[b]
                          and allowed(a, b) and allowed(donors[b], a)]
            if not candidates:
                raise ValueError(f'cannot construct source-disjoint permutation in {category}')
            b, c = max(candidates, key=lambda pair: similarity(a, pair[0]) + similarity(a, pair[1]))
            donors[a], donors[b], donors[c] = b, c, a
    if sorted(donors) != list(range(len(rows))):
        raise AssertionError('donor mapping is not a permutation')
    pairs = []
    for i, j in enumerate(donors):
        assert allowed(i, j) and rows[i]['category'] == rows[j]['category']
        pairs.append({
            'sample_id': rows[i]['sample_id'], 'donor_sample_id': rows[j]['sample_id'],
            'source_id': source_id(rows[i]), 'donor_source_id': source_id(rows[j]),
            'category': rows[i]['category'], 'frame_count': len(rows[i]['past_files']),
            'donor_frame_count': len(rows[j]['past_files']),
            'question_cosine': similarity(i, j),
        })
    return donors, {'seed': seed, 'algorithm': 'category/frame-count strata; TF-IDF greedy pairs and triple cycles',
                    'matching_inputs': ['question', 'category', 'past_files count', 'source_id'],
                    'gold_used_for_matching': False, 'pairs': pairs}


def prepare(source: Path, destination: Path, *, workers: int, seed: int, expected: int):
    source, destination = source.resolve(), destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = source / 'val.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f'expected {expected} validation samples, got {len(rows)}')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate sample IDs')
    clean = [{key: row[key] for key in ('sample_id', 'source_id', 'category', 'question', 'answer', 'past_files')}
             for row in rows]
    expected_hashes = {relative: row.get('frame_sha256', {}).get(relative)
                       for row in rows for relative in row['past_files']}
    donors, matching = matched_permutation(clean, seed)
    (destination / 'matching.json').write_text(json.dumps(matching, ensure_ascii=False, indent=2))
    print(f"[{time.strftime('%F %T')}] matching complete; copying Past frames only", flush=True)
    paths = sorted({relative for row in clean for relative in row['past_files']})

    def copy(relative):
        original = (source / relative).resolve()
        if not original.is_relative_to(source) or not original.is_file():
            raise ValueError(f'invalid Past path: {relative}')
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f'invalid destination path: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(original)
        if expected_hashes.get(relative) and checksum != expected_hashes[relative]:
            raise ValueError(f'source Past frame differs from dataset checksum: {relative}')
        if not (target.is_file() and target.stat().st_size == original.stat().st_size and digest(target) == checksum):
            temporary = target.with_suffix(target.suffix + '.partial')
            shutil.copyfile(original, temporary)
            if digest(temporary) != checksum:
                raise ValueError(f'checksum mismatch copying {relative}')
            temporary.replace(target)
        return relative, checksum, target.stat().st_size

    files = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy, path) for path in paths]
        for count, future in enumerate(as_completed(futures), 1):
            path, checksum, size = future.result()
            files[path] = {'sha256': checksum, 'bytes': size}
            if count == 1 or count % 100 == 0 or count == len(paths):
                print(f"[{time.strftime('%F %T')}] Past files {count}/{len(paths)}", flush=True)
    (destination / 'val.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in clean))
    summary = {'samples': len(clean), 'source_manifest_sha256': digest(manifest),
               'staged_manifest_sha256': digest(destination / 'val.jsonl'),
               'categories': dict(Counter(row['category'] for row in clean)),
               'unique_sources': len({source_id(row) for row in clean}),
               'past_files': len(files), 'past_bytes': sum(f['bytes'] for f in files.values()),
               'future_or_bridge_copied': False, 'seed': seed, 'files': files}
    (destination / 'data-config.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != 'files'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-count', type=int, default=5000)
    args = parser.parse_args()
    prepare(Path(args.source_root), Path(args.output_root), workers=args.workers,
            seed=args.seed, expected=args.expected_count)


if __name__ == '__main__':
    main()

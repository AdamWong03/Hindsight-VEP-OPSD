"""Aggregate frozen Visual-Gain results, source-cluster CIs and blind review packs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

MAIN = ('q_only', 'vq', 'shuffled_vq', 'cdlsd_k4')


def cluster_bootstrap(records, repeats=2000, seed=42):
    """Mean of per-original-video means; one sampled index per source cluster."""
    import numpy as np

    grouped = defaultdict(list)
    for source, values in records:
        grouped[source].append(values)
    if not grouped:
        raise ValueError('empty cluster sample')
    matrix = np.asarray([np.asarray(values).mean(axis=0) for _, values in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repeats, 64):
        indices = rng.integers(0, len(matrix), size=(min(64, repeats - start), len(matrix)))
        draws.append(matrix[indices].mean(axis=1))
    bootstrap = np.concatenate(draws, axis=0)
    return {'sources': len(grouped), 'samples': len(records),
            'mean': matrix.mean(axis=0).tolist(),
            'ci95_low': np.quantile(bootstrap, .025, axis=0).tolist(),
            'ci95_high': np.quantile(bootstrap, .975, axis=0).tolist(),
            'bootstrap_repeats': repeats, 'bootstrap_seed': seed,
            'estimand': 'mean of original-source-video means'}


def load_records(directory):
    outputs = defaultdict(dict)
    for path in sorted(Path(directory).glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            condition, key = row['condition'], row['sample_id']
            if key in outputs[condition]:
                raise ValueError(f'duplicate {condition}/{key}')
            outputs[condition][key] = row
    return outputs


def summarize_condition(rows):
    n = len(rows)
    return {'samples': n, 'sources': len({r['source_id'] for r in rows}),
            'nll_per_answer_token': sum(r['nll_mean'] for r in rows) / n,
            'token_weighted_nll': sum(r['nll_sum'] for r in rows) / sum(r['gold_tokens'] for r in rows),
            'token_f1': sum(r['token_f1'] for r in rows) / n,
            'exact_match': sum(r['exact'] for r in rows) / n,
            'truncated_rate': sum(r['truncated'] for r in rows) / n,
            'empty_predictions': sum(not r['prediction'].strip() for r in rows)}


def evaluate_groups(outputs, repeats=2000):
    ids = set(outputs['q_only'])
    for condition in MAIN:
        if set(outputs[condition]) != ids:
            raise ValueError('main conditions must contain exactly the same sample IDs')
    for key in ids:
        ref = outputs['q_only'][key]
        for condition in MAIN[1:]:
            row = outputs[condition][key]
            for field in ('source_id', 'question', 'gold', 'category', 'gold_token_ids'):
                if ref[field] != row[field]:
                    raise ValueError(f'incomparable {field} for {key}/{condition}')
    groups = {'all': sorted(ids)}
    for key in sorted(ids):
        category = outputs['q_only'][key]['category']
        groups.setdefault(category, []).append(key)
    result = {}
    names = ['delta_visual_nll', 'delta_specific_nll', 'delta_visual_token_f1', 'delta_specific_token_f1']
    for name, group_ids in groups.items():
        table = {condition: summarize_condition([outputs[condition][key] for key in group_ids])
                 for condition in MAIN}
        differences = []
        identical_q = identical_shuffled = 0
        positive_visual = positive_specific = 0
        for key in group_ids:
            q, v, shuffled = [outputs[condition][key] for condition in MAIN[:3]]
            delta_v, delta_s = q['nll_mean'] - v['nll_mean'], shuffled['nll_mean'] - v['nll_mean']
            differences.append((q['source_id'], [delta_v, delta_s,
                                                v['token_f1'] - q['token_f1'],
                                                v['token_f1'] - shuffled['token_f1']]))
            identical_q += q['prediction'] == v['prediction']
            identical_shuffled += shuffled['prediction'] == v['prediction']
            positive_visual += delta_v > 0
            positive_specific += delta_s > 0
        stats = cluster_bootstrap(differences, repeats)
        stats['metrics_in_order'] = names
        stats['positive_visual_fraction'] = positive_visual / len(group_ids)
        stats['positive_specific_fraction'] = positive_specific / len(group_ids)
        stats['identical_vq_q_only'] = identical_q
        stats['identical_vq_shuffled'] = identical_shuffled
        result[name] = {'conditions': table, 'paired_visual_dependence': stats}
    return result


def format_comparison(outputs):
    result = {}
    raw = outputs.get('raw_q_only', {})
    for category in ['all'] + sorted({row['category'] for row in raw.values()}):
        ids = [key for key, row in raw.items() if category == 'all' or row['category'] == category]
        if not ids:
            continue
        native_rows = [outputs['q_only'][key] for key in ids]
        raw_rows = [raw[key] for key in ids]
        result[category] = {'native_q_only': summarize_condition(native_rows),
                            'raw_q_only': summarize_condition(raw_rows)}
    return {'status': 'descriptive format calibration, not a replacement primary baseline',
            'native_prefix': 'official 8B chat template + <answer>',
            'cdlsd_prefix': 'raw Question embeddings; four latent vectors; <answer>',
            'raw_probe_prefix': 'raw Question embeddings; <answer>; no latent placeholders',
            'no_format_changed_in_the_frozen_cdlsd_checkpoint': True,
            'direct_base_vs_cdlsd_nll_is_confounded_by_prefix_and_training': True,
            'groups': result}


def review_pack(outputs, directory):
    rng = random.Random(6719)
    ordered = [(condition, key) for condition in MAIN for key in sorted(outputs[condition])]
    rng.shuffle(ordered)
    mapping = {}
    fields = ['review_id', 'category', 'source_id', 'past_files', 'question', 'gold', 'candidate',
              'verdict', 'error_types', 'reason']
    with (directory / 'semantic-review-blind.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, (condition, key) in enumerate(ordered):
            row = outputs[condition][key]
            review_id = hashlib.sha256(f'6719:{position}:{key}:{condition}'.encode()).hexdigest()[:16]
            mapping[review_id] = {'sample_id': key, 'condition': condition,
                                  'category': row['category'], 'source_id': row['source_id']}
            writer.writerow({'review_id': review_id, 'category': row['category'],
                             'source_id': row['source_id'], 'past_files': json.dumps(row.get('past_files', [])),
                             'question': row['question'],
                             'gold': row['gold'], 'candidate': row['prediction']})
    (directory / 'semantic-review-key.json').write_text(json.dumps(mapping, indent=2))
    # These are triage candidates, never automatic semantic-correctness labels.
    candidates = []
    direction_pairs = [('left', 'right'), ('upward', 'downward'), ('forward', 'backward')]
    import re
    for condition in MAIN:
        for key, row in outputs[condition].items():
            gold = set(re.findall(r'[a-z]+', row['gold'].lower()))
            prediction = set(re.findall(r'[a-z]+', row['prediction'].lower()))
            flags = [f'{a}/{b}' for a, b in direction_pairs
                     if (a in gold and b not in gold and b in prediction and a not in prediction)
                     or (b in gold and a not in gold and a in prediction and b not in prediction)]
            if flags:
                candidates.append({**row, 'triage_flags': flags, 'semantic_label': 'unreviewed'})
    (directory / 'semantic-error-candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    return {'status': 'pending independent semantic annotation; no accuracy inferred from Token-F1',
            'blind_rows': len(mapping), 'heuristic_candidates': len(candidates),
            'verdicts': ['correct', 'incorrect', 'uncertain'],
            'error_types': ['direction', 'subject', 'action', 'temporal_order', 'other'],
            'judge_basis': 'reference-grounded; mark plausible alternative futures as uncertain, not automatically incorrect',
            'review_warning': 'Candidate flags are lexical triage only. Reviewers must assess meaning and may consult Past frames.'}


def matching_stats(matching):
    import numpy as np
    result = {}
    for category in ['all'] + sorted({p['category'] for p in matching['pairs']}):
        pairs = [p for p in matching['pairs'] if category == 'all' or p['category'] == category]
        similarities = [p['question_cosine'] for p in pairs]
        result[category] = {'samples': len(pairs),
                            'same_frame_count_fraction': sum(p['frame_count'] == p['donor_frame_count'] for p in pairs) / len(pairs),
                            'distinct_original_source_fraction': sum(p['source_id'] != p['donor_source_id'] for p in pairs) / len(pairs),
                            'question_cosine_mean': float(np.mean(similarities)),
                            'question_cosine_p10_p50_p90': np.quantile(similarities, [.1, .5, .9]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(args.results)
    matching = json.loads((Path(args.data_root) / 'matching.json').read_text())
    expected = len(matching['pairs'])
    if not args.pilot and any(len(records[c]) != expected for c in MAIN):
        raise ValueError(f'incomplete full results: expected {expected} each, got {dict((c,len(records[c])) for c in MAIN)}')
    expected_probes = sum(min(30, count) for count in Counter(p['category'] for p in matching['pairs']).values())
    if not args.pilot and len(records.get('raw_q_only', {})) != expected_probes:
        raise ValueError('incomplete stratified input-format probes')
    report = {'status': 'pilot correctness check only' if args.pilot else 'full numeric evaluation complete; semantic review pending',
              'groups': evaluate_groups(records, args.bootstrap_repeats),
              'format_comparison': format_comparison(records),
              'matching': matching_stats(matching),
              'semantic_review': review_pack(records, output)}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    with (output / 'category-comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['category', 'condition', 'samples', 'sources', 'nll_per_answer_token',
                         'token_f1', 'exact_match', 'truncated_rate'])
        for category, group in report['groups'].items():
            for condition, stats in group['conditions'].items():
                writer.writerow([category, condition] + [stats[k] for k in
                    ['samples', 'sources', 'nll_per_answer_token', 'token_f1', 'exact_match', 'truncated_rate']])
    lines = ['# VEP Visual-Gain Audit', '', report['status'], '',
             'Native Q-only / V+Q / matched shuffled V+Q share the official 8B chat template and the same answer prefix. '
             'CDLSD retains its original raw-question/latent reader. Absolute cross-model NLL is therefore confounded.', '',
             '| Category | Condition | N | Gold NLL/token | Token-F1 ×100 | Truncated % |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for category, group in report['groups'].items():
        for condition, stats in group['conditions'].items():
            lines.append(f"| {category} | {condition} | {stats['samples']} | {stats['nll_per_answer_token']:.4f} | {100*stats['token_f1']:.2f} | {100*stats['truncated_rate']:.1f} |")
    lines += ['', 'Positive NLL deltas favor correct Past. CIs use paired bootstrap of original-source-video means.', '',
              '| Category | Δvisual [95% CI] | Δspecific [95% CI] |', '| --- | --- | --- |']
    for category, group in report['groups'].items():
        s = group['paired_visual_dependence']
        values = [f"{s['mean'][i]:.4f} [{s['ci95_low'][i]:.4f}, {s['ci95_high'][i]:.4f}]" for i in (0, 1)]
        lines.append(f'| {category} | {values[0]} | {values[1]} |')
    lines += ['', 'Semantic accuracy is pending independent annotation. The blind CSV covers all outputs; '
              'the error-candidate JSON contains unreviewed lexical flags, not verified semantic errors. '
              'Direction, subject, action and temporal ordering must be assessed beyond word overlap.', '',
              'No benchmark score or method-effectiveness claim is inferred from these diagnostics.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.md')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

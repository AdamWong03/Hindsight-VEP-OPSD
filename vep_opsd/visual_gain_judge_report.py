"""Unblind independent semantic reviews and report coverage-aware agreement CIs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .visual_gain_report import cluster_bootstrap


def summarize(root, mode):
    root = Path(root)
    key = json.loads((root / 'review-key.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    reviews = {}
    for line in (root / f'{mode}-labels.jsonl').read_text().splitlines():
        row = json.loads(line)
        review_id = row['review_id']
        if review_id in reviews or review_id not in key:
            raise ValueError('duplicate or unknown independent semantic record')
        if row['fingerprint'] != config['fingerprint']:
            raise ValueError('mixed judge protocols')
        reviews[review_id] = row
    raw_ids = {v['sample_id'] for v in key.values() if v['condition'] == 'raw_q_only'}
    expected = {k for k, v in key.items() if mode == 'full' or v['sample_id'] in raw_ids}
    if set(reviews) != expected:
        raise ValueError(f'incomplete semantic results: {len(reviews)}/{len(expected)}')
    grouped = defaultdict(list)
    by_sample = defaultdict(dict)
    for review_id, review in reviews.items():
        meta = key[review_id]
        row = {**review, **meta}
        grouped[(meta['category'], meta['condition'])].append(row)
        grouped[('all', meta['condition'])].append(row)
        by_sample[meta['sample_id']][meta['condition']] = row
    groups = {}
    for (category, condition), rows in grouped.items():
        valid = [row for row in rows if row['valid']]
        counts = Counter(row['verdict'] for row in valid)
        decided = counts['correct'] + counts['incorrect']
        groups.setdefault(category, {})[condition] = {
            'expected': len(rows), 'valid': len(valid), 'invalid': len(rows) - len(valid),
            'valid_fraction': len(valid) / len(rows),
            'reference_agreement': counts['correct'] / len(valid) if valid else None,
            'conditional_agreement_on_decided': counts['correct'] / decided if decided else None,
            'counts': dict(counts),
            'uncertain_fraction': counts['uncertain'] / len(valid) if valid else None,
            'error_type_counts': dict(Counter(error for row in valid if row['verdict'] == 'incorrect'
                                             for error in row['error_types']))}
    comparisons = {}
    contrasts = [('vq', 'q_only'), ('vq', 'shuffled_vq'), ('cdlsd_k4', 'vq'), ('raw_q_only', 'q_only')]
    for category in ['all', 'Camera', 'Instructional', 'Predictive']:
        comparisons[category] = {}
        for left, right in contrasts:
            valid_pairs, decided_pairs = [], []
            for rows in by_sample.values():
                if left not in rows or right not in rows:
                    continue
                a, b = rows[left], rows[right]
                if category != 'all' and a['category'] != category:
                    continue
                if not a['valid'] or not b['valid']:
                    continue
                value = float(a['verdict'] == 'correct') - float(b['verdict'] == 'correct')
                valid_pairs.append((a['source_id'], [value]))
                if a['verdict'] != 'uncertain' and b['verdict'] != 'uncertain':
                    decided_pairs.append((a['source_id'], [value]))
            if valid_pairs:
                comparisons[category][f'{left}_minus_{right}'] = {
                    'reference_match_indicator': cluster_bootstrap(valid_pairs),
                    'both_decided_subset': cluster_bootstrap(decided_pairs) if decided_pairs else None,
                    'uncertain_handling': 'not counted as definite reference agreement; retained in denominator; not called an error',
                    'causal_warning': 'CDLSD and base differ in training and reader; this contrast alone cannot identify KD or visual-latent effects'
                                     if left == 'cdlsd_k4' else None}
    return {'status': f'{mode} automated semantic review complete',
            'scope': 'independent local Qwen2.5-32B reference-grounded text review, NOT ground-truth future-event accuracy or a benchmark judge',
            'judge_limitations': ['No future video is available to this evaluator.',
                                  'The reference is one plausible future, not exhaustive ground truth.',
                                  'The evaluator has independent parameters but shares the Qwen family; residual style/family bias is possible.',
                                  'Coverage and uncertain labels must accompany any agreement number.'],
            'records': len(reviews), 'config_fingerprint': config['fingerprint'],
            'groups': groups, 'paired_source_bootstrap': comparisons}, reviews, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--mode', choices=('pilot', 'full'), required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    report, reviews, key = summarize(root, args.mode)
    (root / f'{args.mode}-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (root / f'{args.mode}-automatic-annotations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['review_id', 'category', 'verdict', 'error_types', 'reason'])
        writer.writeheader()
        for review_id, row in sorted(reviews.items()):
            writer.writerow({'review_id': review_id, 'category': key[review_id]['category'],
                             'verdict': row.get('verdict', '') if row['valid'] else '',
                             'error_types': ';'.join(row.get('error_types', [])),
                             'reason': row.get('reason', row.get('parse_error', ''))})
    lines = ['# Independent automated semantic review', '', report['status'], '', report['scope'], '',
             '| Category | Condition | N | Valid | Reference agreement % | Uncertain % |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for category in ['all', 'Camera', 'Instructional', 'Predictive']:
        for condition, row in sorted(report['groups'].get(category, {}).items()):
            agreement = f"{100*row['reference_agreement']:.2f}" if row['reference_agreement'] is not None else 'NA'
            uncertainty = f"{100*row['uncertain_fraction']:.2f}" if row['uncertain_fraction'] is not None else 'NA'
            lines.append(f"| {category} | {condition} | {row['expected']} | {row['valid']} | {agreement} | {uncertainty} |")
    lines += ['', 'Paired differences in definite reference-match indicator; uncertain is retained but is not an error label.', '',
              '| Category | Contrast | Paired N | Difference percentage points [95% CI] |',
              '| --- | --- | ---: | --- |']
    for category, comparisons in report['paired_source_bootstrap'].items():
        for contrast, value in comparisons.items():
            stats = value['reference_match_indicator']
            lines.append(f"| {category} | {contrast} | {stats['samples']} | {100*stats['mean'][0]:.2f} [{100*stats['ci95_low'][0]:.2f}, {100*stats['ci95_high'][0]:.2f}] |")
    lines += ['', 'Limitations:', ''] + [f'- {note}' for note in report['judge_limitations']]
    (root / f'{args.mode}-report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': report['status'], 'records': report['records'],
                      'report': str(root / f'{args.mode}-report.md')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

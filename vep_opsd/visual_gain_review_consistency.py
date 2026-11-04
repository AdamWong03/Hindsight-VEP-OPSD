"""Compare two AI reference-text reviews, explicitly not a ground-truth accuracy test."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .visual_gain_judge import load_blind_inputs
from .visual_gain_report import cluster_bootstrap

PRIOR_EXPOSURE = {
    '4b2ddac312382b72', 'b9817da034ae35fc', 'c4643850d5bed499', '172b55920fc2dc83',
    '36608718165d1648', '27f41781c1fba90b', '76cd55827b5f1b1c', '4f49766ed119d5c6',
    '7c241c74611d781a', '336860939b963fb3', '5c5d3d56c7dd99da', '140c7c1b77013402',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--judge-directory', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    inputs, key, _ = load_blind_inputs(run)
    inputs = {row['review_id']: row for row in inputs}
    selected = set(json.loads((run / 'followup/second-review-selection.json').read_text())['sample_ids'])
    decisions = {}
    for file in sorted((run / 'followup').glob('codex-second-review-part*.json')):
        data = json.loads(file.read_text())['decisions']
        if set(data).intersection(decisions):
            raise ValueError('duplicate second-review ID')
        decisions.update(data)
    expected = {review_id for review_id, meta in key.items() if meta['sample_id'] in selected}
    if set(decisions) != expected or len(selected) != 30:
        raise ValueError('second review does not exactly cover 30 selected videos and all five conditions')
    judged = {r['review_id']: r for r in
              (json.loads(line) for line in (Path(args.judge_directory) / 'pilot-labels.jsonl').read_text().splitlines())}
    groups = defaultdict(list)
    disagreements = []
    for review_id, second in decisions.items():
        if review_id in PRIOR_EXPOSURE:
            continue
        first, meta = judged[review_id], key[review_id]
        if not first['valid']:
            continue
        row = {'review_id': review_id, 'source_id': meta['source_id'],
               'second_ai': second['verdict'], 'local_judge': first['verdict']}
        groups['all'].append(row)
        groups[meta['category']].append(row)
        if second['verdict'] != first['verdict']:
            disagreements.append({**inputs[review_id], 'category': meta['category'],
                                  'second_ai': second, 'local_judge': first})
    report = {'scope': 'two AI raters, not human validation or ground-truth future accuracy',
              'selected_videos': len(selected), 'reviewed_outputs': len(decisions),
              'excluded_previously_exposed_ids': sorted(set(decisions) & PRIOR_EXPOSURE),
              'policy': 'Fixed category/source sampling before looking at first-rater labels. Disagreements are not automatically errors by either rater.',
              'groups': {}}
    for group, rows in groups.items():
        stats = cluster_bootstrap([(r['source_id'], [float(r['second_ai'] == r['local_judge'])]) for r in rows])
        report['groups'][group] = {
            'outputs': len(rows), 'source_cluster_agreement': stats,
            'second_ai_counts': dict(Counter(r['second_ai'] for r in rows)),
            'local_judge_counts': dict(Counter(r['local_judge'] for r in rows)),
            'confusion_second_vs_local': dict(Counter(r['second_ai'] + '->' + r['local_judge'] for r in rows))}
    (run / 'followup/review-consistency.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (run / 'followup/review-disagreements.json').write_text(json.dumps(disagreements, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

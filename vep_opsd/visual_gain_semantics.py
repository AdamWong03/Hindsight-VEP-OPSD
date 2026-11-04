"""Score independently completed blind reviews; never infer semantics from F1."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def score_annotations(path, key_path):
    key = json.loads(Path(key_path).read_text())
    groups, seen = defaultdict(list), set()
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            verdict = row.get('verdict', '').strip().lower()
            if not verdict:
                continue
            review_id = row['review_id']
            if review_id in seen or review_id not in key:
                raise ValueError('duplicate or unknown blind review ID')
            if verdict not in ('correct', 'incorrect', 'uncertain'):
                raise ValueError(f'invalid semantic verdict: {verdict}')
            seen.add(review_id)
            if row['category'] != key[review_id]['category']:
                raise ValueError('annotated category differs from original blind record')
            condition = key[review_id]['condition']
            groups[(row['category'], condition)].append(row)
            groups[('all', condition)].append(row)
    result = {}
    for (category, condition), rows in groups.items():
        counts = Counter(row['verdict'].strip().lower() for row in rows)
        decided = counts['correct'] + counts['incorrect']
        expected = sum(v['condition'] == condition and (category == 'all' or v['category'] == category)
                       for v in key.values())
        result.setdefault(category, {})[condition] = {
            'reviewed': len(rows), 'expected': expected, 'coverage': len(rows) / expected,
            'correct': counts['correct'], 'incorrect': counts['incorrect'],
            'uncertain': counts['uncertain'],
            'accuracy_on_decided': counts['correct'] / decided if decided else None,
            'correct_fraction_of_reviewed': counts['correct'] / len(rows),
            'error_counts': dict(Counter(error.strip() for row in rows
                                        for error in row.get('error_types', '').split(';') if error.strip()))}
    return {'status': 'independent annotations scored' if seen else 'semantic annotation pending',
            'reviewed': len(seen), 'expected': len(key), 'groups': result,
            'uncertain_handling': 'excluded from accuracy_on_decided; retained in coverage and correct_fraction_of_reviewed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = score_annotations(args.annotations, args.key)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'groups'}, indent=2))


if __name__ == '__main__':
    main()

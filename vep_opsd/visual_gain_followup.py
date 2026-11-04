"""Post-hoc interface validation and uncertainty sensitivity; frozen outputs only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

from .visual_gain_report import cluster_bootstrap, load_records


def log(message):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] [interface-followup] {message}', flush=True)


def analyze(records, matching):
    raw, native = records['raw_q_only'], records['q_only']
    formatting = {}
    bins = [('first', 0, 1), ('tokens_2_5', 1, 5), ('tokens_6_10', 5, 10),
            ('tokens_11_20', 10, 20), ('tokens_21_onward', 20, None)]
    for category in ['all', 'Camera', 'Instructional', 'Predictive']:
        ids = [key for key, row in raw.items() if category == 'all' or row['category'] == category]
        paired = []
        positions = {}
        for key in ids:
            if native[key]['gold_token_ids'] != raw[key]['gold_token_ids']:
                raise ValueError('format analysis target-token mismatch')
            paired.append((raw[key]['source_id'], [native[key]['nll_mean'] - raw[key]['nll_mean'],
                                                  raw[key]['token_f1'] - native[key]['token_f1']]))
        for name, start, end in bins:
            a = [v for key in ids for v in native[key]['gold_token_nll'][start:end]]
            b = [v for key in ids for v in raw[key]['gold_token_nll'][start:end]]
            if a:
                positions[name] = {'tokens': len(a), 'native_nll': mean(a),
                                   'raw_nll': mean(b), 'difference': mean(a) - mean(b)}
        formatting[category] = {
            'samples': len(ids), 'paired': cluster_bootstrap(paired),
            'metrics_in_order': ['native_minus_raw_nll', 'raw_minus_native_token_f1'],
            'native_nll': mean(native[key]['nll_mean'] for key in ids),
            'raw_nll': mean(raw[key]['nll_mean'] for key in ids),
            'native_mean_words': mean(len(native[key]['prediction'].split()) for key in ids),
            'raw_mean_words': mean(len(raw[key]['prediction'].split()) for key in ids),
            'raw_truncated_fraction': mean(raw[key]['truncated'] for key in ids),
            'raw_empty': sum(not raw[key]['prediction'].strip() for key in ids),
            'positive_nll_difference_count': sum(native[key]['nll_mean'] > raw[key]['nll_mean'] for key in ids),
            'token_position_descriptions': positions}

    parent = {}
    def find(a):
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]
    for pair in matching['pairs']:
        a, b = find(pair['source_id']), find(pair['donor_source_id'])
        parent[b] = a
    sizes = Counter(find(a) for a in parent)
    sensitivity = {}
    for category in ['all', 'Camera', 'Instructional', 'Predictive']:
        values = []
        for key, row in records['q_only'].items():
            if category != 'all' and row['category'] != category:
                continue
            v, s = records['vq'][key], records['shuffled_vq'][key]
            values.append((find(row['source_id']), [row['nll_mean'] - v['nll_mean'],
                                                   s['nll_mean'] - v['nll_mean']]))
        sensitivity[category] = cluster_bootstrap(values)
    return {'scope': 'post-hoc description; original primary output and primary CIs unchanged',
            'formatting': formatting,
            'donor_component_bootstrap': {'purpose': 'sensitivity to dependence induced by fixed video donor pairs',
                'component_size_counts': dict(Counter(sizes.values())),
                'metrics_in_order': ['delta_visual_nll', 'delta_specific_nll'], 'groups': sensitivity}}


def verify_input_api(records, args):
    import torch
    from .visual_gain_eval import load_base, native_prefix, native_logits, positions, response_ids

    model, _, processor = load_base(args, torch.device('cuda:0'))
    tokenizer = processor.tokenizer
    failures, raw_boundary_differences = [], []
    largest = 0.
    raw = records['raw_q_only']
    with torch.inference_mode():
        for i, (key, row) in enumerate(sorted(raw.items())):
            ids = response_ids(tokenizer, row['gold'], torch.device('cuda:0'))
            batch, text = native_prefix(processor, row['question'], [], torch.device('cuda:0'))
            native_ids = batch['input_ids'][0].tolist()
            joint = tokenizer(text, add_special_tokens=False)['input_ids']
            if native_ids != joint:
                failures.append(key)
            raw_ids = tokenizer(row['question'], add_special_tokens=False)['input_ids'] + tokenizer('<answer>', add_special_tokens=False)['input_ids']
            if raw_ids != tokenizer(row['question'] + '<answer>', add_special_tokens=False)['input_ids']:
                raw_boundary_differences.append(key)
            actual = native_logits(model, batch, ids)
            full = dict(batch)
            full['input_ids'] = torch.cat([batch['input_ids'], ids.unsqueeze(0)], dim=1)
            full['attention_mask'] = torch.ones_like(full['input_ids'])
            other = model(inputs_embeds=model.get_input_embeddings()(full['input_ids']),
                          attention_mask=full['attention_mask'], position_ids=positions(model, full),
                          cache_position=torch.arange(full['input_ids'].shape[1], device=ids.device),
                          logits_to_keep=ids.numel() + 1, use_cache=False, return_dict=True).logits[0, :ids.numel()]
            gap = float((actual.float() - other.float()).abs().max())
            largest = max(largest, gap)
            torch.testing.assert_close(actual, other, rtol=0, atol=0)
            if i == 0 or (i + 1) % 10 == 0:
                log(f'original 8B native input_ids vs identical inputs_embeds: {i+1}/{len(raw)}; max logit difference={largest}')
    if failures:
        raise ValueError(f'native chat token concatenation differs from canonical encoding in {len(failures)} samples')
    return {'samples': len(raw), 'native_prefix_canonical_tokenization_mismatches': failures,
            'input_ids_vs_equal_embeddings_max_logit_difference': largest,
            'raw_question_marker_joint_tokenization_mismatches': raw_boundary_differences,
            'raw_boundary_note': 'The existing raw probe intentionally encodes Question and <answer> separately, as the latent reader does. Any joint-tokenization differences are recorded, not silently changed.',
            'no_generation_repeated': True, 'no_image_embedding_manipulation': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-input-api', action='store_true')
    parser.add_argument('--base-model', default='/data2/wyd/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--processor', default='/data1/wyd/models/Qwen3-VL-4B-Instruct')
    parser.add_argument('--image-max-pixels', type=int, default=2097152)
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(root / 'full')
    matching = json.loads((root / 'matching.json').read_text())
    result = analyze(records, matching)
    log('frozen-output format contrasts and donor-pair sensitivity calculated')
    if args.verify_input_api:
        result['interface_validation'] = verify_input_api(records, args)
    (output / 'interface-followup.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# Frozen-output interface follow-up', '', result['scope'], '',
             '| Category | N | Native NLL | Raw NLL | Native−raw [95% CI] | Raw truncated |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    for category, row in result['formatting'].items():
        ci = row['paired']
        lines.append(f"| {category} | {row['samples']} | {row['native_nll']:.4f} | {row['raw_nll']:.4f} | {ci['mean'][0]:.4f} [{ci['ci95_low'][0]:.4f}, {ci['ci95_high'][0]:.4f}] | {100*row['raw_truncated_fraction']:.1f}% |")
    lines += ['', 'Gold-content tokenization is held fixed. Both interfaces use original 8B weights. '
              'These 90 samples were fixed before evaluating outcomes, not selected by their scores.', '',
              'Raw-probe answers are often longer and truncated; lower Gold NLL is not by itself evidence of better event prediction.', '',
              'Original-source CIs remain the primary analysis. Donor-component CIs below are a post-hoc sensitivity check that keeps mutually swapped videos together.', '',
              '| Category | Δspecific with donor-component bootstrap [95% CI] |', '| --- | --- |']
    for category, row in result['donor_component_bootstrap']['groups'].items():
        lines.append(f"| {category} | {row['mean'][1]:.4f} [{row['ci95_low'][1]:.4f}, {row['ci95_high'][1]:.4f}] |")
    if 'interface_validation' in result:
        lines += ['', 'Original-model interface validation:', '', json.dumps(result['interface_validation'], ensure_ascii=False, indent=2)]
    (output / 'interface-followup.md').write_text('\n'.join(lines)+'\n')
    log(f'complete: {output / "interface-followup.md"}')


if __name__ == '__main__':
    main()

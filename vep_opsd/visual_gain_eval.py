"""Frozen-model Visual-Gain Audit; no optimizer, trainer, or future-frame input."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] [visual-gain] {message}", flush=True)


if __name__ == '__main__':
    log('importing inference dependencies')

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .cdlsd_data import CDLSDDataset, encode_multimodal_prefill
from .cdlsd_eval import load_checkpoint
from .cdlsd_latent import answer_logits, decode_answer, sample_answer
from .eipd_latent import (_mrope_positions, inner_model, prefill_only,
                          rollout_latents_shared, without_gradient_checkpointing)
from .eval_metrics import normalise, token_f1
from .visual_gain_data import digest

SYSTEM = ('Predict the subsequent event from the observed video when provided. '
          'Answer the question directly in <answer>...</answer> tags without reasoning.')
MARKER = '<answer>'
CONDITIONS = ('q_only', 'vq', 'shuffled_vq', 'cdlsd_k4')


def response_ids(tokenizer, answer, device):
    """Identical Gold content tokens in every condition; no tags, EOS or prompt."""
    values = tokenizer(answer.strip(), add_special_tokens=False)['input_ids']
    if not values:
        raise ValueError('empty Gold Answer')
    return torch.tensor(values, dtype=torch.long, device=device)


def score_logits(logits, ids):
    if logits.shape[0] != ids.numel():
        raise ValueError('answer-token alignment failure')
    losses = torch.cat([F.cross_entropy(logits[start:start + 32].float(),
                                       ids[start:start + 32], reduction='none')
                        for start in range(0, ids.numel(), 32)])
    if not torch.isfinite(losses).all():
        raise ValueError('non-finite Gold NLL')
    return {'nll_mean': float(losses.mean()), 'nll_sum': float(losses.sum()),
            'gold_tokens': ids.numel(), 'gold_token_nll': losses.cpu().tolist()}


def native_prefix(processor, question, images, device):
    # The three native conditions differ only in the images supplied here.
    content = [{'type': 'image', 'image': image} for image in images]
    content.append({'type': 'text', 'text': 'Question: ' + question})
    text = processor.apply_chat_template([
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True)
    kwargs = {'text': [text], 'return_tensors': 'pt'}
    if images:
        kwargs['images'] = images
    encoded = processor(**kwargs)
    keep = ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')
    batch = {key: encoded[key].to(device) for key in keep if encoded.get(key) is not None}
    marker = torch.tensor([processor.tokenizer(MARKER, add_special_tokens=False)['input_ids']],
                          dtype=torch.long, device=device)
    batch['input_ids'] = torch.cat([batch['input_ids'], marker], dim=1)
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    return batch, text + MARKER


def positions(model, batch):
    return inner_model(model).get_rope_index(
        batch['input_ids'], image_grid_thw=batch.get('image_grid_thw'),
        attention_mask=batch['attention_mask'])[0]


def native_logits(model, batch, ids):
    full = dict(batch)
    full['input_ids'] = torch.cat([batch['input_ids'], ids.unsqueeze(0)], dim=1)
    full['attention_mask'] = torch.ones_like(full['input_ids'])
    length = full['input_ids'].shape[1]
    out = model(**full, position_ids=positions(model, full),
                cache_position=torch.arange(length, device=ids.device),
                use_cache=False, logits_to_keep=ids.numel() + 1, return_dict=True)
    return out.logits[0, :ids.numel()]


def native_generate(model, tokenizer, batch, max_new_tokens):
    pos = positions(model, batch)
    physical = batch['input_ids'].shape[1]
    next_position = int(pos[:, :, -1].max()) + 1
    out = model(**batch, position_ids=pos,
                cache_position=torch.arange(physical, device=pos.device),
                use_cache=True, logits_to_keep=1, return_dict=True)
    cache, logits = out.past_key_values, out.logits[:, -1]
    generated = []
    for offset in range(max_new_tokens):
        token = logits.argmax(-1, keepdim=True)
        value = int(token.item())
        generated.append(value)
        if value == tokenizer.eos_token_id or '</answer>' in tokenizer.decode(generated, skip_special_tokens=False):
            break
        if offset + 1 == max_new_tokens:
            break
        out = model(input_ids=token, past_key_values=cache,
                    cache_position=torch.tensor([physical + offset], device=pos.device),
                    position_ids=_mrope_positions(next_position + offset, 1, pos.device),
                    use_cache=True, logits_to_keep=1, return_dict=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
    return torch.tensor(generated, dtype=torch.long, device=pos.device)


def verify_native_alignment(model, batch, ids):
    """Check teacher-forcing against KV-cache continuation on the same tokens."""
    ids = ids[:8]
    full = native_logits(model, batch, ids).float().log_softmax(-1)
    reference = full.gather(1, ids[:, None]).squeeze(1)
    pos = positions(model, batch)
    physical = batch['input_ids'].shape[1]
    start = int(pos[:, :, -1].max()) + 1
    out = model(**batch, position_ids=pos,
                cache_position=torch.arange(physical, device=ids.device),
                logits_to_keep=1, use_cache=True, return_dict=True)
    actual = []
    for t in range(ids.numel()):
        actual.append(out.logits[0, -1].float().log_softmax(-1)[ids[t]])
        if t + 1 < ids.numel():
            out = model(input_ids=ids[t:t + 1].view(1, 1), past_key_values=out.past_key_values,
                        cache_position=torch.tensor([physical + t], device=ids.device),
                        position_ids=_mrope_positions(start + t, 1, ids.device),
                        logits_to_keep=1, use_cache=True, return_dict=True)
    actual = torch.stack(actual)
    # BF16 logits around magnitude 16--32 have 0.125 spacing. Different GEMM
    # shapes for prefill versus single-token decode can move one or two bins.
    # The FP32 tiny multimodal test below uses a strict numerical tolerance.
    bf16 = next(model.parameters()).dtype == torch.bfloat16
    torch.testing.assert_close(actual, reference, rtol=.01 if bf16 else 1e-5,
                               atol=.25 if bf16 else 5e-5)
    log(f'verified Gold alignment and native KV cache; max target log-probability gap={float((actual-reference).abs().max()):.6f}')


def short_states(model, projector, processor, item):
    """Exactly the first four existing recurrence steps; no fifth-to-eighth work."""
    backbone = inner_model(model)
    kwargs, start = encode_multimodal_prefill(
        backbone, processor, item['observed'], item['row']['question'], next(model.parameters()).device)
    with without_gradient_checkpointing(model):
        seed = prefill_only(model, projector, kwargs, start, checkpoint=False)
    return rollout_latents_shared(model, projector, [seed], steps=4)[0]


def format_indices(rows, per_category=30, seed=137):
    rng = random.Random(seed)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['category'], []).append(i)
    result = set()
    for indices in groups.values():
        result.update(rng.sample(indices, min(per_category, len(indices))))
    return result


def read_completed(path, fingerprint):
    """Recover only a truncated last JSON line; reject corrupt interior records."""
    completed = set()
    if not path.exists():
        return completed
    with path.open('r+b') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(offset)
                log('removed incomplete trailing record before resume')
                break
            row = json.loads(line)
            if row['config_fingerprint'] != fingerprint:
                raise ValueError('output belongs to a different protocol; use a fresh output directory')
            key = (row['sample_id'], row['condition'])
            if key in completed:
                raise ValueError(f'duplicate result: {key}')
            completed.add(key)
    return completed


def load_base(args, device):
    log(f'loading official processor: {args.processor}')
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True,
                                             min_pixels=2048, max_pixels=args.image_max_pixels)
    # The weight-only local 8B directory lacks processor metadata. Its official
    # vocabulary matches this processor, but use the original 8B chat template.
    template = Path(args.base_model) / 'chat_template.json'
    processor.chat_template = json.loads(template.read_text())['chat_template']
    processor.tokenizer.chat_template = processor.chat_template
    official_vocab = json.loads((Path(args.base_model) / 'vocab.json').read_text())
    loaded_vocab = processor.tokenizer.get_vocab()
    if any(loaded_vocab.get(token) != value for token, value in official_vocab.items()):
        raise ValueError('processor does not match original 8B vocabulary')
    log(f'loading original local 8B weights: {args.base_model}')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.to(device).eval()
    model.requires_grad_(False)
    return model, None, processor


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('base', 'cdlsd'), required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--processor', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--image-max-pixels', type=int, default=2097152)
    parser.add_argument('--limit-per-category', type=int, default=0)
    parser.add_argument('--verify-caches', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('invalid shard index/count')
    root, output = Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = CDLSDDataset(str(root), str(root / 'val.jsonl'))
    matching = json.loads((root / 'matching.json').read_text())
    lookup = {r['sample_id']: i for i, r in enumerate(dataset.rows)}
    donors = {p['sample_id']: lookup[p['donor_sample_id']] for p in matching['pairs']}
    indices = list(range(len(dataset)))
    if args.limit_per_category:
        counts = {}
        indices = []
        for i, row in enumerate(dataset.rows):
            category = row['category']
            if counts.get(category, 0) < args.limit_per_category:
                indices.append(i)
                counts[category] = counts.get(category, 0) + 1
    indices = indices[args.shard_index::args.shard_count]
    protocol = {**vars(args), 'system': SYSTEM, 'answer_prefix': MARKER,
                'target': 'all Gold content tokens only, independently tokenized; no tags/EOS/prompt/padding',
                'manifest_sha256': digest(root / 'val.jsonl'),
                'matching_sha256': digest(root / 'matching.json'),
                'eval_code_sha256': digest(Path(__file__)),
                'frozen_latent_code_sha256': digest(Path(__file__).with_name('cdlsd_latent.py')),
                'frozen_recurrence_code_sha256': digest(Path(__file__).with_name('eipd_latent.py')),
                'official_8b_chat_template_sha256': digest(Path(args.base_model) / 'chat_template.json'),
                'conditions': list(CONDITIONS), 'dtype': 'bfloat16', 'attention': 'sdpa',
                'format_probe_per_category': 30, 'format_probe_seed': 137,
                'cdlsd_latent_steps_actually_executed': 4,
                'native_vs_cdlsd_prompt_confound': True}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    label = f'{args.family}-{args.shard_index:02d}-of-{args.shard_count:02d}'
    result_path = output / f'{label}.jsonl'
    completed = read_completed(result_path, fingerprint)
    (output / f'{label}.config.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    log(f'family={args.family} shard={args.shard_index}/{args.shard_count} samples={len(indices)} resume_records={len(completed)}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('GPU required for this full-scale audit; refusing an accidental CPU run')
    model, projector, processor = (load_base(args, device) if args.family == 'base' else
                                    load_checkpoint(args.base_model, args.checkpoint, device, args.image_max_pixels))
    tokenizer = processor.tokenizer
    probes = format_indices(dataset.rows)
    if args.limit_per_category:
        probes.update(indices)
    first = True
    metadata = {'model_class': type(model).__name__,
                'tokenizer_length': len(tokenizer),
                'output_vocabulary_size': model.get_output_embeddings().weight.shape[0],
                'cuda_name': torch.cuda.get_device_name(),
                'torch_version': torch.__version__, 'protocol_fingerprint': fingerprint}
    metadata['chat_template_sha256'] = hashlib.sha256(processor.chat_template.encode()).hexdigest()
    (output / f'{label}.model.json').write_text(json.dumps(metadata, indent=2))
    log(f'model ready: {metadata}')
    with result_path.open('a', buffering=1) as stream:
        for offset, index in enumerate(indices):
            row = dataset.rows[index]
            conditions = (['q_only', 'vq', 'shuffled_vq'] + (['raw_q_only'] if index in probes else [])
                          if args.family == 'base' else ['cdlsd_k4'])
            pending = [condition for condition in conditions if (row['sample_id'], condition) not in completed]
            if not pending:
                continue
            item = None
            for condition in pending:
                started = time.monotonic()
                ids = response_ids(tokenizer, row['answer'], device)
                states, donor_id = None, None
                if condition in ('vq', 'cdlsd_k4'):
                    item = item or dataset[index]
                if condition == 'cdlsd_k4':
                    states = short_states(model, projector, processor, item)
                    if args.verify_caches and first:
                        from .cdlsd_latent import latent_rollout
                        reference = latent_rollout(model, projector, processor, item['observed'],
                                                   row['question'], checkpoint_prefill=False).short
                        torch.testing.assert_close(states, reference, rtol=0, atol=0)
                        log('verified: four-step states exactly equal original eight-step chain first-four states')
                    scores = score_logits(answer_logits(model, tokenizer, row['question'], states, ids), ids)
                    generated = sample_answer(model, tokenizer, row['question'], states,
                                              max_new_tokens=args.max_new_tokens, temperature=0., top_p=1.)
                    prefix_kind = 'existing raw Question; four latents; <answer>'
                elif condition == 'raw_q_only':
                    states = torch.empty((0, model.config.text_config.hidden_size), device=device,
                                         dtype=model.get_input_embeddings().weight.dtype)
                    scores = score_logits(answer_logits(model, tokenizer, row['question'], states, ids), ids)
                    generated = sample_answer(model, tokenizer, row['question'], states,
                                              max_new_tokens=args.max_new_tokens, temperature=0., top_p=1.)
                    prefix_kind = 'format probe only: raw Question; <answer>; no latent placeholders'
                else:
                    images = []
                    if condition == 'vq':
                        images = item['observed']
                    elif condition == 'shuffled_vq':
                        donor = dataset[donors[row['sample_id']]]
                        images, donor_id = donor['observed'], donor['row']['sample_id']
                    batch, text = native_prefix(processor, row['question'], images, device)
                    if args.verify_caches and offset == 0:
                        verify_native_alignment(model, batch, ids)
                    scores = score_logits(native_logits(model, batch, ids), ids)
                    generated = native_generate(model, tokenizer, batch, args.max_new_tokens)
                    prefix_kind = 'native Qwen chat; assistant newline; <answer>'
                    if first:
                        preview = {'native_prompt_text': text,
                                   'native_prompt_token_ids': batch['input_ids'][0].cpu().tolist(),
                                   'raw_question_token_ids': tokenizer(row['question'], add_special_tokens=False)['input_ids'],
                                   'answer_marker_token_ids': tokenizer(MARKER, add_special_tokens=False)['input_ids'],
                                   'gold_content_token_ids': ids.cpu().tolist()}
                        (output / f'{label}.prefix-example.json').write_text(json.dumps(preview, ensure_ascii=False, indent=2))
                prediction = decode_answer(tokenizer, generated)
                raw = tokenizer.decode(generated.tolist(), skip_special_tokens=False)
                record = {'sample_id': row['sample_id'], 'source_id': row['source_id'],
                          'category': row['category'], 'question': row['question'], 'gold': row['answer'],
                          'past_files': row['past_files'],
                          'condition': condition, 'donor_sample_id': donor_id, 'prefix_kind': prefix_kind,
                          'gold_token_ids': ids.cpu().tolist(), **scores,
                          'prediction': prediction, 'generated_ids': generated.cpu().tolist(),
                          'generated_tokens': generated.numel(), 'closed_answer': '</answer>' in raw,
                          'truncated': generated.numel() == args.max_new_tokens and '</answer>' not in raw
                                       and int(generated[-1]) != tokenizer.eos_token_id,
                          'exact': float(normalise(prediction) == normalise(row['answer'])),
                          'token_f1': token_f1(prediction, row['answer']),
                          'elapsed_seconds': time.monotonic() - started,
                          'config_fingerprint': fingerprint}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                completed.add((row['sample_id'], condition))
                if first:
                    log(f"first result {condition} {row['category']}: NLL={scores['nll_mean']:.4f}")
                first = False
                del states, generated
            if offset == 0 or (offset + 1) % 10 == 0 or offset + 1 == len(indices):
                log(f'{label}: samples {offset + 1}/{len(indices)}, records={len(completed)}')
    log(f'complete: {result_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Prepare a small offline evidence-review pack; never feed future frames to VEP models."""
import argparse
import json
from pathlib import Path

from .visual_gain_judge import load_blind_inputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', required=True)
parser.add_argument('--dataset-root', required=True)
args = parser.parse_args()
run, dataset = Path(args.run), Path(args.dataset_root).resolve()
selected = json.loads((run / 'followup/second-review-selection.json').read_text())['sample_ids']
inputs, key, _ = load_blind_inputs(run)
rows = {}
for line in (dataset / 'val.jsonl').open():
    row = json.loads(line)
    if row['sample_id'] in selected:
        rows[row['sample_id']] = row
if len(rows) != len(selected):
    raise ValueError('missing selected evidence cases')
cases = []
for sample_id in selected:
    row = rows[sample_id]
    def evidence(paths, native_indices):
        result = []
        for relative, index in zip(paths, native_indices, strict=True):
            resolved = (dataset / relative).resolve()
            if not resolved.is_relative_to(dataset) or not resolved.is_file():
                raise ValueError(f'invalid evidence path: {relative}')
            result.append({'path': str(resolved), 'native_frame_index': index})
        return result
    candidates = [{'review_id': item['review_id'], 'candidate': item['candidate'],
                   'verdict_with_future_evidence': '', 'reason': ''}
                  for item in inputs if key[item['review_id']]['sample_id'] == sample_id]
    clip = dataset / 'raw_clips' / row['video']
    cases.append({'sample_id': sample_id, 'source_id': row['source_id'],
                  'category': row['category'], 'question': row['question'],
                  'reference': row['answer'],
                  'past': evidence(row['past_files'], row['past_native_indices']),
                  'future_for_offline_review_only': evidence(row['future_files'], row['future_native_indices']),
                  'full_source_clip_for_adjudication': {
                      'path': str(clip), 'available': clip.is_file(), 'fps': row.get('fps'),
                      'native_video_frames': row.get('native_video_frames'),
                      'past_end_native_frame': max(row['past_native_indices']),
                      'source_start': row.get('source_start'), 'source_end': row.get('source_end')},
                  'reference_supported_by_evidence': None,
                  'question_resolvable_from_cached_frames': None,
                  'candidate_reviews': candidates, 'notes': ''})
result = {'scope': 'offline annotation/evidence review only, never inference or training inputs',
          'videos': len(cases), 'answers': sum(len(row['candidate_reviews']) for row in cases),
          'sampling': 'seed 859; 10 videos/category from the pre-fixed stratified 90-video format subset',
          'instructions': 'Check whether the question is answerable and Gold is supported before labeling candidates. Sparse future frames may remain inconclusive. Keep uncertain where actual future evidence is insufficient. Condition labels are withheld.',
          'cases': cases}
(run / 'followup/future-evidence-review-pack.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))

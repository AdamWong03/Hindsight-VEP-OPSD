# EIPD-Data-v1: final research dataset

This is the FINAL-SCALE build recipe, not a bandwidth-constrained pilot.
The frozen goal is **50,000 train + 5,000 held-out validation** samples.
Train: 30K Predictive, 15K Instructional, 5K Camera.
Validation: 3K Predictive, 1.5K Instructional, 0.5K Camera.
These quotas are predeclared sampling choices, not teacher-gain scores.
The plan retains 4x prioritized candidate reserves to replace legitimately
unavailable or invalid raw videos without model-score selection.

All 2.7M TwiFF rows are scanned. Source-video decontamination uses YouTube
parent IDs, not merely TwiFF clip filenames. Only one clip from any original
source video may enter the complete dataset. Validation and training use
disjoint source videos, but are not necessarily unseen by a previously
pretrained Future-L1 checkpoint (that is checked by excluded source lists).

## Prerequisite: provenance and video identity

Official sources:

- TwiFF: https://huggingface.co/datasets/Liu-Junhua/TwiFF-2.7M
- TwiFF implementation: https://github.com/LiuJunhua02/TwiFF
- Panda-70M downloader: https://github.com/snap-research/Panda-70M/tree/main/dataset_dataloading

TwiFF source clip names are of form YoutubeID_clipNumber.mp4.
Panda official video2dataset outputs numeric sharded MP4 names. DO NOT
blindly rename these files: the mapping must check original YouTube source,
clip position and timestamp, otherwise future labels and Bridge frames may
be paired with the wrong video. Use exact TwiFF-named MP4s if already present,
or supply a reviewed TwiFF-clip-name-to-local-file JSON mapping.
Panda metadata CSV fields are url and timestamp. Use the official Panda fork
of video2dataset; note TwiFF says to download at original 720p, whereas the
Panda sample config defaults to 360p. The video license is noncommercial
research and does not authorize redistribution of all third-party footage.

## 1. Provide three REAL exclusion lists

Each must be nonempty, contain a verified original 11-character source ID or
matching TwiFF clip filename, and cover the relevant dataset:
  /data/exclude_futurel1.txt
  /data/exclude_futurebench.txt
  /data/exclude_twiffbench.txt

Do not fabricate IDs. If a benchmark has no source-video ID, use its original
dataset documentation to resolve the provenance, and conduct perceptual
near-duplicate matching as a separate audit. Hash grouping only guarantees
exact source-ID deduplication, not visual content deduplication.

## 2. Metadata planning - no raw videos required

  python -m pip install datasets pyarrow decord pillow numpy

  python -m vep_opsd.eipd_final_data plan \
    --input /data/TwiFF-2.7M-parquet \
    --exclusions /data/exclude_futurel1.txt \
                 /data/exclude_futurebench.txt \
                 /data/exclude_twiffbench.txt \
    --output /data/EIPD-Data-v1

Outputs include state.sqlite, plan_config.json, plan_report.json,
candidates.jsonl (4x candidates), and candidate_source_ids.txt.
Planning is idempotent and uses reproducible SHA256 rankings. Change data
design only by creating a new version directory. If some task category is
insufficient, it fails rather than silently changing quotas.
`--input` accepts one JSON/JSONL/Parquet file or a directory of homogeneous
shards. Every local shard's path, byte size and SHA256 are frozen in
plan_config.json. `--hf-dataset Liu-Junhua/TwiFF-2.7M` remains available when
the immutable Hugging Face revision is preferred.

## 3. Download the needed videos - without Future-L1's 200+ GB archive

If you have TwiFF-named clips already, put them in /data/twiff_clips/.
Otherwise take the original Panda-70M training CSV (download it from the
official Panda site) and select original YouTube sources:

  python -m vep_opsd.eipd_final_data panda-subset \
    --csv /data/panda70m_train_full.csv \
    --video-ids /data/EIPD-Data-v1/candidate_source_ids.txt \
    --output-csv /data/EIPD-Data-v1/panda_sources.csv

The planner also preserves TwiFF's exact `meta_data.start` and
`meta_data.end` values. The included restartable downloader uses those source
timestamps and the original YouTube ID to make exact TwiFF-named clips while
limiting source resolution to 720p (or the highest lower resolution):

  python -m vep_opsd.eipd_clips \
    --output /data/EIPD-Data-v1 \
    --clip-root /data/EIPD-Data-v1/raw_clips \
    --reserve-factor 1.25 --workers 4 \
    --reuse-root /data/already_verified_twiff_clips

It records every attempt in download_state.sqlite and
logs/clip-download.jsonl, validates outputs with ffprobe, and safely resumes.
An optional `--cookies` file may be supplied when the hosting site requires
it. A small `--max-tasks` run should be used to verify network access before
the full acquisition.

Alternatively, run Panda's own video2dataset on the subset CSV with url and
timestamp columns. Inspect the downloaded MP4 metadata and establish a
verified mapping:

  /data/verified_map.json

The mapping is an ordinary JSON dictionary:

  {"abcdefghijk_7.mp4":"/data/panda_download/checked_7.mp4"}

We cannot infer the TwiFF suffix-to-Panda clip mapping from the name of a
numeric shard file alone. Never start training with a guessed mapping.
Missing or inaccessible source videos stay missing; the reserve handles them.

## 4. Full frame materialization - restartable

  python -m vep_opsd.eipd_final_data materialize \
    --output /data/EIPD-Data-v1 \
    --video-root /data/twiff_clips \
    --video-map /data/verified_map.json

Omit video-map if all MP4 files exist with exact original TwiFF filenames
directly in video-root. This decodes original TwiFF 8-frame sampling pool,
caches ALL observed and ALL future latent-supervision frames, and creates
one true native-frame midpoint Bridge. Frames are lossless full-resolution PNG
by default, not resized JPEG. No future supervision is discarded.

Per-record payload: past_files, bridge_file, future_files, source_id,
original temporal indices, native temporal indices, fps, original labels,
SFT reasoning, answer, frame SHA256, original video SHA256, and split.

An interrupted run can restart; partial frames are atomically replaced,
previously decoded samples are tracked by SQLite. Missing clips can be
provided and the materialization command rerun. The program only publishes
final manifests when ALL frozen category quotas have been met; unfinished
runs retain the state and an error, not a fake complete dataset.

## 5. Verify / transfer to another training server

  python -m vep_opsd.eipd_final_data verify \
    --output /data/EIPD-Data-v1

  rsync -aP --partial --append-verify /data/EIPD-Data-v1/ \
    user@train-host:/data/EIPD-Data-v1/

Run verify again on the remote host. Training reads ONLY cached relative
frame paths, with no requirement for any original MP4 on the second server.
For the largest corpora, preserve the manifest and data directory tree.

The checks validate the parent-source split, strict temporal Bridge bounds,
SFT image/text alignment, decodable image files and each frame SHA256.
These checks cannot establish whether an open-ended question has a factually
correct answer or whether Bridge visually discloses the final answer early.

## 6. Quality and official experiment eligibility

Before publishing training results, perform human inspection of a stratified
sample across all three classes and across bridge-gap bins. Explicitly report:
- incorrect future-event labels
- answers unsupported by actual future video
- question not describing a verifiable future event
- Bridge already resolving the asked event
- repeated or near-duplicate videos not caught by source IDs

If the error rates are high, update the dataset version and filtering policy
BEFORE comparing method results; never select examples by model performance.
Training the full 50K without semantic QA is a structural prototype, not a
certified high-quality dataset.


### Stratified semantic review sheets

After materialize and verify, generate ~450 randomly selected visual audit sheets
(50 per category x bridge-gap bin, where available):

    python -m vep_opsd.eipd_audit --data-root /data/EIPD-Data-v1 \
      --split train --output /data/EIPD-Data-v1/audit_sheets \
      --per-stratum 50

Review the Past, Bridge, and Future frames alongside the question and answer.
Fill the yes/no fields in review.csv for temporal early-answer leakage, annotation
correctness, question validity and video alignment. Record class-conditional
failure rates. Do not claim full semantic certification from structural QA.

## 7. Train with finalized PNG manifest

  python -m vep_opsd.eipd_train \
    --future-l1-root /repos/Future-L1 \
    --model /models/future-l1-sft \
    --manifest /data/EIPD-Data-v1/train.jsonl \
    --output /runs/eipd-smoke --max-steps 1 \
    --gradient-accumulation-steps 1

After 8B-model latent replay, SFT image collator and GPU parity checks pass,
remove max-steps for full runs. Legacy scripts/train_50k.sh remains Gold+Full
Future OPSD and MUST NOT be used for EIPD. The frozen algorithm is untouched:
Teacher uses Past vs Past+Bridge, no Gold in the prompt, while Future-SFT images
are used only in the original SFT branch.

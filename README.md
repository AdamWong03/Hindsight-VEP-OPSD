# Hindsight-VEP-OPSD

**Active method: EIPD (Evidence-Increment Predictive Distillation)** for video event prediction. The original Gold+Future OPSD baseline is retained for reproducibility but is no longer the default project workflow.

EIPD distills answer predictions using a frozen same-initialization teacher under two conditions: Past only and Past plus **one genuinely intermediate Bridge frame**. Both branches replay the student's sampled discrete reasoning with separately recomputed continuous Future-L1 latents. Ground-truth answers and full future-target frames are never included in the EIPD teacher prompt; the original Future-L1 SFT branch keeps its full future supervision.

## Final-scale dataset: 50K train + 5K validation

The authoritative preparation instructions are in **[EIPD-Data-v1](docs/eipd-final-data.md)**.

| Split | Predictive | Instructional | Camera | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 30,000 | 15,000 | 5,000 | 50,000 |
| Validation | 3,000 | 1,500 | 500 | 5,000 |

The fixed category mix is a preregistered sampling choice, not a claim of optimality. The pipeline scans the original TwiFF-2.7M metadata, groups clips by original source video, enforces verified benchmark/Future-L1 overlap exclusions and maintains a deterministic candidate reserve. It downloads only necessary source clips and caches all original Past and Future-SFT frames plus the intermediate Bridge as full-resolution PNG. The dataset is portable: final manifests contain relative PNG paths and checksums; the training server needs no original MP4 files.

### Get started on the data server

    git clone https://github.com/AdamWong03/Hindsight-VEP-OPSD.git
    cd Hindsight-VEP-OPSD
    python -m pip install numpy pillow decord datasets pyarrow huggingface_hub yt-dlp

    export EIPD_ROOT=/data/EIPD-Data-v1
    export EXCLUDE_FUTUREL1=/data/exclude_futurel1.txt
    export EXCLUDE_FUTUREBENCH=/data/exclude_futurebench.txt
    export EXCLUDE_TWIFFBENCH=/data/exclude_twiffbench.txt

    bash scripts/build_eipd_final.sh plan

After planning, arrange verified source video acquisition as described in the guide. FFmpeg/ffprobe are required for downloads. With video files under EIPD_ROOT/raw_clips:

    export VIDEO_ROOT=/data/EIPD-Data-v1/raw_clips
    bash scripts/build_eipd_final.sh download
    bash scripts/build_eipd_final.sh materialize
    bash scripts/build_eipd_final.sh verify
    bash scripts/build_eipd_final.sh audit

The three exclusion files must contain real source IDs (not placeholders). Source-to-TwiFF segment time alignment must be confirmed. Construction is restartable. The final manifest is emitted only once all frozen category counts are met.

### Output format

    EIPD-Data-v1/
      train.jsonl
      val.jsonl
      dataset_summary.json
      frames/train/{sample_id}/{past_00.png,bridge.png,future_00.png,...}
      frames/val/{sample_id}/{past_00.png,bridge.png,future_00.png,...}
      candidates.jsonl
      state.sqlite
      download_state.sqlite
      audit_sheets/

All PNG paths in train.jsonl/val.jsonl are relative to the dataset root, preserving exact Past/Bridge/Future order. Original TwiFF index lists and native frame positions remain in the metadata.

## Model training

Follow **[EIPD implementation contract](docs/eipd-implementation.md)**. Run a real Future-L1-SFT Qwen3-VL checkpoint's one-step smoke test before formal training:

    python -m vep_opsd.eipd_train \
      --future-l1-root /repos/Future-L1 \
      --model /models/future-l1-sft \
      --manifest /data/EIPD-Data-v1/train.jsonl \
      --output /runs/eipd-smoke \
      --max-steps 1 --gradient-accumulation-steps 1

The live 8B GPU/distributed training integration is **not yet certified**. Before any full run, verify generation-versus-replay latent parity, answer-token alignment, gradients to continuous latents, SFT loss and GPU memory. No benchmark gains are claimed.

## Source layout

| Path | Purpose |
| --- | --- |
| vep_opsd/eipd_final_data.py | Full TwiFF metadata selection, PNG materialization, verification |
| vep_opsd/eipd_clips.py | Restartable verified source clip retrieval |
| vep_opsd/eipd_audit.py | Stratified video-question-answer contact sheets |
| vep_opsd/eipd_train.py | EIPD two-condition teacher and student training |
| vep_opsd/eipd_objective.py | Evidence-increment weighted answer KL |
| vep_opsd/eipd_replay.py | Differentiable continuous latent replay |
| scripts/build_eipd_final.sh | plan / panda-subset / download / materialize / verify / audit |
| docs/legacy-opsd.md | Old Gold+Full-Future OPSD baseline |

**Legacy:** vep_opsd/train.py, vep_opsd/trainer.py and scripts/train_50k.sh implement *the old Gold+Future OPSD*, not EIPD.

## Attribution and licensing

The work builds on [Future-L1](https://github.com/OpenGVLab/Future-L1), [TwiFF](https://github.com/LiuJunhua02/TwiFF) and [Panda-70M](https://github.com/snap-research/Panda-70M). Respect original videos' research-use and redistribution restrictions; public distribution of third-party video pixels is not assumed.

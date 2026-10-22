# EIPD implementation contract (2026-10)

This is a new independent experiment path. The existing vep_opsd/train.py,
vep_opsd/trainer.py, and scripts/train_50k.sh remain legacy full-future + gold
OPSD and MUST NOT be used for EIPD.

## Frozen hypothesis

Evidence-Increment Predictive Distillation:

- Student: only observed TwiFF frames + question; samples discrete reasoning
  with Future-L1 recurrent continuous latent positions.
- Frozen teacher: **same initial checkpoint with LoRA disabled**; twice evaluated
  under identical discrete sampled reasoning, with and without one intermediate
  *real* video frame strictly between observation and first future target.
- No gold answer, reference CoT, or first future target frame in teacher prompt.
- Gold answer enters only after sampled reasoning as autoregressive target
  teacher-forced suffix, never into the user prompt.
- Teacher is frozen; each condition independently recomputes continuous latents.
- Student independently recomputes latent states **with gradients** via upstream
  lvr_mask/lvr_states/prompt_length hooks. A text-only transformer forward is
  not valid EIPD and should fail an end-to-end parity/gradient test.
- Auxiliary loss: answer-only forward KL to 0.5*p0+0.5*pBridge, weighted by
  teacher JS divergence, capped and reweighted. No generated reasoning-token KL.
- Joint loss is the **official Future-L1 image/latent SFT loss** on original
  interleaved reasoning + lambda_d (default 0.2) times EIPD.

## Reproducible data path

The primary training data comes from **TwiFF-2.7M metadata**, not from the
Future-L1-50K visual-gain selected subset. Metadata selection is deterministic
by source video hash; neither model performance nor teacher confidence enters
sample ranking. Keep source-video exclusions for both benchmarks AND
Future-L1-50K, recorded in text files (one basename or stem per line).

TwiFF indexes are positions 1..8 into its original uniform sampled pool, not
native frame indices. Recover native boundaries using uniform_pool(vlen,8).
Bridge is the native midpoint between the last observed frame and first
future-target frame. No boundary aliasing; reject gaps with no interior frame.

Example:

    python -m vep_opsd.eipd_data \
      --hf-dataset Liu-Junhua/TwiFF-2.7M \
      --video-root /datasets/Panda70M_clips \
      --exclude-video-lists /data/exclude_futurel1.txt /data/exclude_futurebench.txt /data/exclude_twiffbench.txt \
      --train-size 50000 --val-size 2000 \
      --strict-counts --output-dir /data/eipd_v1

For clips stored in nested directories, pass --video-map with JSON mapping each
original TwiFF video filename to its local absolute path. Videos must already
be downloaded; no scraper or copyrighted video redistribution is included.

## Train

Install the exact upstream Future-L1 checkout and dependencies first. Supply a
Qwen3-VL-compatible Future-L1-SFT checkpoint, not the vanilla Qwen3-VL checkpoint:

    FUTURE_L1_ROOT=/repos/Future-L1 \
    python -m vep_opsd.eipd_train \
      --future-l1-root /repos/Future-L1 \
      --model /models/future-l1-sft \
      --manifest /data/eipd_v1/train.jsonl \
      --output /runs/eipd_v1

A bounded GPU smoke (once videos and a real checkpoint are available):
    
    python -m vep_opsd.eipd_train --future-l1-root /repos/Future-L1 \
      --model /models/future-l1-sft --manifest /data/eipd_v1/train.jsonl \
      --output /runs/eipd_smoke --max-steps 1 --gradient-accumulation-steps 1

For distributed ZeRO-2, use torchrun with the same arguments plus
--deepspeed /path/to/zero2.json. Do not assume ZeRO-3 adapter-disabled
rollout works without an explicit distributed parity/collectives test.

The code uses a deliberately expensive O(latent tokens) full-prefix
differentiable replay. Do not start 50K 8B training before the following gates:

1. Metadata/video audit produces exact counts and no raw-frame index violations.
2. Human audit verifies the bridge is genuine *intermediate* evidence and
   answer is actually supported by the future.
3. With one short multimodal sample, compare token-level generation and
   differentiable replay latent states for parity; verify gradient from answer
   KL reaches parameters that generate continuous latent states.
4. Verify Future-L1 SFT collator yields finite CE/latent losses on new manifests.
5. Check GPU memory, distributed wrappers, and effective train FLOPs / walltime.
6. Verify evaluation decontamination by source video IDs and near-duplicate
   video content. Exclusion-list input is a **mechanism**, not a claim that
   such lists have already been assembled.

Until gates 3–5 have run on a real Qwen3-VL + Future-L1 checkpoint, this is an
integration candidate, **not** an end-to-end verified training recipe.
It must not be claimed to have trained, improved a benchmark, or passed GPU smoke.

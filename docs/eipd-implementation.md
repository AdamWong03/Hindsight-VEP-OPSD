# EIPD implementation contract (2026-10-25)

The production input is an EIPD-Data-v1 cached-image manifest containing
relative `past_files`, `bridge_file`, and `future_files`. The active model path
starts from the original Qwen3-VL checkpoint. It does not load Future-L1
weights or perform continuous latent generation.

The old `vep_opsd/train.py`, `vep_opsd/trainer.py`, and `scripts/train_50k.sh`
remain the legacy Gold+Full-Future OPSD baseline and must not be used for EIPD.

## Training objective

For each example:

1. The current Student samples one discrete reasoning trajectory from Past
   frames and the question only. Its output remains an open-ended event
   prediction ending in `<answer>...</answer>`.
2. The dataset Gold Answer replaces the sampled answer after the Student's
   sampled `<answer>` boundary. It is an autoregressive target, never part of
   the user prompt.
3. A frozen Teacher is the same original checkpoint with the Student LoRA
   adapter disabled. It independently processes Past and Past+Bridge prompts,
   teacher-forced on the identical sampled-reasoning + Gold-Answer tokens.
4. The trainable Student recomputes that sequence from Past only. Only logits
   predicting Gold-Answer tokens enter the loss; prompt and reasoning tokens
   are masked from direct supervision.

Let `p0` be the frozen Teacher distribution with Past and `p1` the distribution
with Past+Bridge. The target is

    q_t = (1 - alpha) p0_t + alpha p1_t

and per-token evidence is `JS(p0_t, p1_t)`. The evidence weights are capped and
normalized over informative answer positions. The joint objective is

    L = CE_gold + lambda_d * mean_t[w_t KL(q_t || p_student_t)]

with defaults `alpha=0.5` and `lambda_d=0.2`. A malformed on-policy rollout
receives only direct Gold-Answer CE and is not silently converted into an
off-policy distillation sample.

The Bridge is privileged training evidence only. Future target frames remain
in the dataset for provenance, visual audit, and possible separately named
experiments; the active Student, Teacher, and inference path do not consume
them. There is no Yes/No verifier, reward model, EMA, hidden-state loss,
projection head, vocabulary Top-K approximation, or Future-L1 latent loss.

## Train

Run a one-step real-data smoke from the original checkpoint:

    python -m vep_opsd.eipd_train \
      --model /models/Qwen3-VL-8B-Instruct \
      --manifest /data/EIPD-Data-v1/train.jsonl \
      --output /runs/eipd-smoke \
      --max-steps 1 --gradient-accumulation-steps 1

Some weight-only local copies omit `preprocessor_config.json` and tokenizer
metadata. In that case, pass a complete processor directory from the same
original Qwen3-VL family:

    python -m vep_opsd.eipd_train \
      --model /models/Qwen3-VL-8B-Instruct \
      --processor /models/Qwen3-VL-4B-Instruct \
      --manifest /data/EIPD-Data-v1/train.jsonl \
      --output /runs/eipd

`--processor` supplies preprocessing and tokenizer metadata only; model weights
are always loaded from `--model`. The processor vocabulary is checked against
the 8B embedding table.

For distributed ZeRO-2, launch with `torchrun` and add
`--deepspeed /path/to/zero2.json`. Adapter-disabled Teacher forwards must be
validated under the chosen distributed wrapper before treating a full run as
comparable.

Training loss or Teacher disagreement alone is not evidence of better event
prediction. Report FutureBench and TwiFF-Bench only after formal evaluation,
and retain source-video decontamination and semantic data-audit results.

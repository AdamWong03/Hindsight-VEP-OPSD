# Hindsight VEP-OPSD

Hindsight-Guided On-Policy Self-Distillation for video event prediction with
Qwen3-VL. This repository contains one method only: **Hindsight VEP-OPSD v1**.

The implementation follows a Generate → Verify → Distill loop:

1. The current student samples four complete textual trajectories from the
   observed video and question.
2. The same model, under `no_grad`, independently checks each trajectory against
   the ground-truth future. Its score is the normalized next-token probability
   of `Yes` versus `No`.
3. The four verification scores form a temperature-scaled listwise target. A
   group is skipped when every score is below 0.6.
4. The student teacher-forces its sampled trajectories using observed frames
   only. Mean generated-token log-probabilities form the student distribution,
   optimized by listwise cross-entropy.

There is no hidden-state KD, Future-L1 latent MSE, vocabulary KL/JSD, EMA
teacher, projection head, gating network, external reward model, or offline
rollout cache.

## Repository layout

```text
.
├── vep_opsd/                 # Complete v1 method and training entry point
├── scripts/
│   ├── train_50k.sh          # Reproducible 50K launch configuration
│   ├── eval.sh               # FutureBench / TwiFF-Bench evaluation
│   └── zero2.json            # DeepSpeed ZeRO-2 configuration
├── tests/                    # Objective-level tests
├── docs/data-format.md       # Expected VEP training manifest
└── pyproject.toml
```

The key implementation is [`vep_opsd/trainer.py`](vep_opsd/trainer.py).
Prompt construction is in [`vep_opsd/prompts.py`](vep_opsd/prompts.py), and the
binary verifier plus listwise objective are in
[`vep_opsd/objective.py`](vep_opsd/objective.py).

## Installation

Python 3.10+ and a CUDA-compatible PyTorch installation are expected.

```bash
pip install -e .
```

For evaluation, install a recent
[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) checkout containing
the Qwen3-VL model adapter and the FutureBench/TwiFF-Bench tasks. Set
`LMMS_EVAL_ROOT` to that checkout when using `scripts/eval.sh`.

## Training

The manifest follows the compact VEP/TwiFF format documented in
[`docs/data-format.md`](docs/data-format.md). Frames referenced by `image` are
the student's observations; frames referenced by `reasoning_image` are exposed
only to the no-gradient verifier.

```bash
MODEL=/path/to/Qwen3-VL-8B-Instruct \
DATA=/path/to/reused_union_50k_compact.json \
VIDEO_ROOT=/path/to/dataset/root \
OUTPUT_DIR=/path/to/output \
bash scripts/train_50k.sh
```

The fixed v1 defaults are `M=4`, rollout temperature `0.8`, `top_p=0.95`,
teacher temperature `0.2`, student temperature `1.0`, and skip threshold `0.6`.
Every optimizer batch samples fresh trajectories from the current student.

Checkpoints and the resolved configuration are written under `OUTPUT_DIR`.
Candidate token IDs, decoded reasoning/answers, verifier scores, target
preferences, and student scores are recorded as per-rank JSONL files under
`OUTPUT_DIR/rollouts/`.

## Evaluation

```bash
LMMS_EVAL_ROOT=/path/to/lmms-eval \
CKPT=/path/to/completed/checkpoint \
EVAL_OUT=/path/to/results/futurebench \
BENCH=futurebench \
bash scripts/eval.sh

LMMS_EVAL_ROOT=/path/to/lmms-eval \
CKPT=/path/to/completed/checkpoint \
EVAL_OUT=/path/to/results/twiff \
BENCH=twiff \
bash scripts/eval.sh
```

The repository intentionally contains no claimed benchmark result: effectiveness
must be established by completed FutureBench and TwiFF-Bench evaluations, not by
training loss or mean verifier score.

## Acknowledgement

The project structure and VEP data convention build on
[`OpenGVLab/Future-L1`](https://github.com/OpenGVLab/Future-L1). The v1 training
objective in this repository is independent of Future-L1 continuous-latent
generation and legacy PFSD variants.


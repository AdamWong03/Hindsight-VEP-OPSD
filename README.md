# Hindsight VEP-OPSD

Hindsight-Guided On-Policy Self-Distillation for video event prediction with
Qwen3-VL. This repository contains one method only: **Hindsight VEP-OPSD v1**.

The implementation follows a Generate → Verify → Distill loop:

1. The current student samples four complete textual trajectories from the
   observed video and question. A trajectory that does not commit to a concrete
   predicted event (empty, too short, or a "cannot be determined" style refusal)
   is discarded and resampled.
2. The same model, under `no_grad`, independently checks each trajectory against
   the ground-truth future: does it predict a specific event that matches what
   the future frames actually show? Its score is the normalized next-token
   probability of `Yes` versus `No`.
3. The four verification scores form a temperature-scaled listwise target. A
   group is skipped when every score is below 0.6, when the spread between the
   four scores is below `minimum_teacher_spread` (a flat teacher ranks nothing),
   or when fewer than four committed candidates survived.
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
│   ├── setup_env.sh          # Build the reusable micromamba environment
│   ├── _env.sh               # Env resolution sourced by every other script
│   ├── train_50k.sh          # Reproducible 50K launch configuration
│   ├── train_smoke.sh        # 2-GPU end-to-end smoke run
│   ├── eval.sh               # FutureBench / TwiFF-Bench generation
│   ├── judge_twiff.sh        # TwiFF answer judge + report
│   ├── fetch_futurebench.sh  # Fetch the FutureBench video corpus
│   ├── report_eval_run.py    # Print recorded metrics for an eval directory
│   └── zero2*.json           # DeepSpeed ZeRO-2 configurations
├── eval_tasks/               # lmms-eval task templates (TwiFF / FutureBench)
├── lmms-eval/                # Evaluation framework checkout (git-ignored)
├── tests/                    # Objective-level tests
├── docs/
│   ├── data-format.md        # Expected VEP training manifest
│   └── workflow.md           # Environment, training and evaluation runbook
├── environment.yml           # Conda layer of the environment
├── requirements.txt          # PyPI layer of the environment
└── pyproject.toml
```

The key implementation is [`vep_opsd/trainer.py`](vep_opsd/trainer.py).
Prompt construction is in [`vep_opsd/prompts.py`](vep_opsd/prompts.py), and the
binary verifier plus listwise objective are in
[`vep_opsd/objective.py`](vep_opsd/objective.py).

## Installation

The supported environment is a **named micromamba environment** so it can be
reused by any checkout on the same machine:

```bash
micromamba create -n vep -c conda-forge python=3.12 ffmpeg pip -y

PY="$(micromamba run -n vep which python)"
uv pip install --python "$PY" -r requirements.txt
uv pip install --python "$PY" -e .
```

`bash scripts/setup_env.sh` runs those steps and prints the resulting versions;
`--index URL --host HOST` (or `PYPI_INDEX`/`PYPI_HOST` in `.env`) routes pip
through a mirror. `environment.yml` and `requirements.txt` are the conda and PyPI layers
of that environment.

Evaluation needs [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval),
which is **not** vendored here — clone it next to the repo and the setup script
installs it editable into the same environment:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval lmms-eval
bash scripts/setup_env.sh          # re-runs, now including lmms-eval
```

`scripts/patch_lmms_eval.py` widens lmms-eval's exact dependency pins
(`datasets==2.16.1`, `numpy==1.26.4`, `protobuf==3.20`, ...) so it can share the
training stack; setup_env.sh applies it before installing. Point at an existing
checkout with `LMMS_EVAL_ROOT=/path/to/lmms-eval`. `scripts/_env.sh` resolves the
environment for every script, so `micromamba run -n vep ...` is optional. See
[`docs/workflow.md`](docs/workflow.md) for the full runbook.

```bash
pip install -e .          # alternative when the shared env is not wanted
```

---

## Configuration

Machine-specific paths live in `.env`, which is git-ignored; `.env.example`
documents every key and is copied to start:

```bash
cp .env.example .env && ${EDITOR:-vi} .env     # MODEL, DATA, OUTPUT_ROOT, DATA_ROOT, ...
```

Precedence is: exported environment variable > `.env` > script default, and a
command line flag wins over both. Paths and mirrors are never hardcoded here.

## Training

The manifest follows the compact VEP/TwiFF format documented in
[`docs/data-format.md`](docs/data-format.md). Frames referenced by `image` are
the student's observations; frames referenced by `reasoning_image` are exposed
only to the no-gradient verifier.

Configuration is command-line only, so no `VAR=value` prefix or exported
environment variable is needed (see [`docs/workflow.md`](docs/workflow.md)):

```bash
bash scripts/train_50k.sh                 # MODEL/DATA/OUTPUT_ROOT from .env
bash scripts/train_50k.sh --output-dir /path/to/run --nproc-per-node 8
```

`--help` lists every option; unrecognised flags are forwarded to the trainer.

The fixed v1 defaults are `M=4`, rollout temperature `0.8`, `top_p=0.95`,
teacher temperature `0.5`, student temperature `1.0`, skip threshold `0.6`,
minimum teacher spread `0.05`, and one resampling round for refused predictions.
Every optimizer batch samples fresh trajectories from the current student.

The default batch geometry is **global 128 = 8 ranks x 4 samples per rank x 4
accumulation steps**. The per-rank micro batch of 4 is an execution knob, not an
optimisation change: all samples of a micro batch are decoded in one `generate()`
call, so the 8B weights are read once per decode step for the whole micro batch
(measured 8.93 -> 3.95 s per sample at N=4). `--global-batch-size` stays 128, so
the objective and the effective batch are identical to a 1x16 geometry.

A 2-GPU plumbing check (8 optimizer steps, ZeRO-2 + offload, 200 samples) is:

```bash
bash scripts/train_smoke.sh
```

Checkpoints and the resolved configuration are written under `OUTPUT_DIR`.
A rerun into the same `OUTPUT_DIR` resumes from the newest checkpoint
automatically; pass `--auto_resume False` (and a fresh directory) to start from
the base model instead.
Candidate token IDs, decoded reasoning/answers, verifier scores, target
preferences, and student scores are recorded as per-rank JSONL files under
`OUTPUT_DIR/rollouts/`.

## Evaluation

`scripts/eval.sh` renders an lmms-eval task from `eval_tasks/<bench>/<bench>.yaml.in`
and runs the stock `qwen3_vl` adapter (v1 checkpoints are plain Qwen3-VL
directories). FutureBench additionally needs its video corpus, which
`scripts/fetch_futurebench.sh` streams once.

```bash
# TwiFF-Bench (frames are embedded in the parquet shards)
bash scripts/eval.sh --bench twiff --ckpt /path/to/completed/checkpoint \
  --eval-out /path/to/results/twiff
bash scripts/judge_twiff.sh --dir /path/to/results/twiff

# FutureBench (fetch the corpus once, then point at the prepared JSON)
bash scripts/fetch_futurebench.sh --dest /path/to/FutureBench
bash scripts/eval.sh --bench futurebench --ckpt /path/to/completed/checkpoint \
  --eval-out /path/to/results/futurebench \
  --futurebench-json /path/to/FutureBench/futurebench_local.json

python scripts/report_eval_run.py /path/to/results/twiff
```

The repository intentionally contains no claimed benchmark result: effectiveness
must be established by completed FutureBench and TwiFF-Bench evaluations, not by
training loss or mean verifier score.

## Acknowledgement

The project structure and VEP data convention build on
[`OpenGVLab/Future-L1`](https://github.com/OpenGVLab/Future-L1). The v1 training
objective in this repository is independent of Future-L1 continuous-latent
generation and legacy PFSD variants.


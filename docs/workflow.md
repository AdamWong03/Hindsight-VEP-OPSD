# Environment, training and evaluation runbook

Paths are never hardcoded: copy `.env.example` to `.env` and fill in `MODEL`,
`DATA`, `OUTPUT_ROOT` and `DATA_ROOT` (`VIDEO_ROOT` is only needed when the
manifest uses relative video paths). Every key can also be passed as a flag.

## 1. Environment

The environment is a **named micromamba environment** (`vep`), not a
project-local one, so any checkout on this machine can reuse it.

```bash
# conda layer: interpreter + video tooling
micromamba create -n vep -c conda-forge python=3.12 ffmpeg pip -y

# pypi layer: training stack + evaluation framework
PY="$(micromamba run -n vep which python)"
uv pip install --python "$PY" -r requirements.txt
uv pip install --python "$PY" -e .          # -e lmms-eval once it is cloned
```

Behind a mirror, pass `--index URL --host HOST` (or set `PYPI_INDEX`/`PYPI_HOST`
in `.env`); without them pip and uv use their default index.

`bash scripts/setup_env.sh` performs exactly these three steps (it skips the
conda layer when the env already exists) and prints the installed versions.

Notes:

- `lmms-eval/` is a **separate checkout**, not vendored in git: clone it next to
  the repo (`git clone https://github.com/EvolvingLMMs-Lab/lmms-eval lmms-eval`)
  and re-run `scripts/setup_env.sh`, or point at an existing copy with
  `LMMS_EVAL_ROOT=/path/to/lmms-eval`. Setup runs
  `scripts/patch_lmms_eval.py`, which widens lmms-eval's exact pins
  (`datasets==2.16.1`, `numpy==1.26.4`, `protobuf==3.20`, ...) to `>=` so it
  shares one environment with the training stack — the same change the
  Future-L1 checkout carries. Without the checkout, training installs fine and
  only evaluation is skipped.
- `uv` reuses its own wheel cache, so a rebuild is mostly relinking rather than
  re-downloading. Plain `pip` works too.
- Paths are configuration, not code: `.env` (git-ignored, documented by
  `.env.example`) supplies `MODEL`, `DATA`, `OUTPUT_ROOT`, `DATA_ROOT`,
  `TWIFF_BENCH_DIR`, `FUTUREBENCH_DIR`, `HF_HOME`. Nothing below hardcodes a
  filesystem layout.
- Every script resolves the env through `scripts/_env.sh`
  (`$VEP_OPSD_ENV_NAME`, then `micromamba env list`, then the standard
  `envs_dirs` paths), so `micromamba run -n vep ...` is optional. Use a
  different name with `VEP_OPSD_ENV_NAME=other bash scripts/...`.

## 2. Training

Everything is passed as command-line options — no `VAR=value` prefix and no
exported environment variable is needed, so a job runner only has to append
flags to the script path.

```bash
# 2-GPU end-to-end smoke run: 8 optimizer steps, 200 samples, ZeRO-2 + offload
bash scripts/train_smoke.sh

# reference 50K full fine-tune (8 GPUs); MODEL, DATA and OUTPUT_ROOT come from .env
bash scripts/train_50k.sh --output-dir "$OUTPUT_ROOT/run1"

# smaller run on 2 GPUs (global batch kept at 128)
bash scripts/train_50k.sh --nproc-per-node 2 --output-dir "$OUTPUT_ROOT/run2gpu"
```

`bash scripts/train_50k.sh --help` lists every option (model/data paths, batch
size, LR, rollout length, ZeRO config, …). Unrecognised flags are forwarded to
the trainer, so Hugging Face arguments still work:
`--max_steps 20`, `--freeze_vision_tower False`, `--image_max_pixels 4096`.

Both scripts log to `logs/<run>-<timestamp>.log` and write into `OUTPUT_DIR`:

| artifact | meaning |
| --- | --- |
| `vep_opsd_config.json` | resolved model/data/algorithm/training configuration |
| `launch.txt` | effective paths, parallelism (`nproc x per_device -> accum`) and forwarded arguments |
| `rollouts/candidates-rank*.jsonl` | per-sample candidates, verifier scores, teacher preference, student scores, skip flag, per-stage seconds |
| `checkpoint-*` | model + optimizer state (ZeRO-2) |

### Watching for a collapsed target

A self-distilled listwise objective can collapse into "no signal" without any
error: if the verifier accepts a non-answer as *consistent* with the future, then
abstention scores as high as a correct prediction, the student moves its mass to
"cannot be determined", all four candidates become identical, the teacher target
becomes uniform and the loss sits at its `ln 4 = 1.3863` floor with ~zero
gradient. That is what happened to the first 50K run in this repo (refusal share
8% -> 99.6% between step 10 and 20, loss flat at 1.386 for the remaining 300
steps).

`python scripts/summarize_rollouts.py <OUTPUT_DIR>` is the check: it reports the
skip reasons, the teacher score spread, and how many degenerate candidates were
kept. A healthy run keeps the spread well above `minimum_teacher_spread` and the
refusal count near zero. Two other defenses are built in:

- `--max-resample-rounds` (default 1) resamples candidates that refuse to commit
  to a prediction; a group that cannot produce four of them is skipped.
- `--minimum-teacher-spread` (default 0.05) skips groups whose verifier scores
  are effectively identical instead of training on a uniform target.

### Resuming from a collapsed checkpoint

`vep_opsd/train.py` resumes from the newest `checkpoint-*` in `--output-dir`.
That is what you want after a preemption, and it is wrong after a collapse: a
checkpoint whose weights already refuse to predict keeps refusing, every
candidate is filtered as degenerate, every group is skipped, and the step logs
`loss 0.0, grad_norm 0.0` while still paying for the rollouts (a resumed
collapsed run here spent 145 steps and two resample rounds — 242 s/step instead
of 141 — on zero signal). Start a fresh run in a new directory, or pass
`--auto_resume False`, and check the first few steps with
`scripts/summarize_rollouts.py` (`steps with any trained group`, the skip reasons
and the teacher spread).

### Skipped groups and collective symmetry

Skipping is per rank and per sample, so the *graph* a rank builds differs from
its neighbours. That is fine as long as the set of parameters receiving a
gradient stays identical, because ZeRO-2 reduce-scatters each parameter bucket as
soon as its gradient is ready: a rank with no kept group in its micro batch would
enqueue one collective while the others enqueue hundreds, and the job deadlocks
in an ALLREDUCE (seen once: 600 s watchdog timeout on the first optimizer step).
The zero term for a fully skipped micro batch must therefore come from the *same
kind of graph*, not just from the same set of parameters: a sum over parameters
covers every parameter but makes the gradients become ready in a different order,
which is enough to deadlock. `_empty_group_zero` reuses the student scoring path
with a single sentinel token, so the readiness order is identical. Verified by
recording the order with parameter grad hooks on one GPU (399 trainable params):

  A student scoring graph : lm_head -> norm -> layers.35.mlp.down_proj -> ...
  B sentinel zero (now)   : identical to A
  C sum over parameters   : diverges at index 2 (post_attention_layernorm
                            instead of mlp.down_proj) -> different reduce order

`--ddp-timeout` defaults to 1800 s; note that DeepSpeed's process group may still
use the 600 s default of torch, so do not rely on it to survive a real stall.

### The teacher temperature

Once the verifier actually discriminates, its scores span the whole range: the
reference answer is accepted (mean P(Yes) 0.93, 11 of 12 samples) while 88% of
sampled candidates score ~0. `teacher_temperature` therefore matters a lot:

| T | mean target entropy | one-hot groups | groups with graded signal |
| ---: | ---: | ---: | ---: |
| 0.2 | 1.114 | 1/12 | 8/12 |
| **0.5** | 1.283 | 0/12 | 11/12 |
| 1.0 | 1.356 | 0/12 | 12/12 |

At 0.2 a single accepted candidate becomes `[0.97, 0.01, 0.01, 0.01]`, i.e. a
one-hot target that reduces the listwise objective to best-of-4 imitation. The
default is 0.5; `--teacher-temperature 1.0` is the flattest setting measured.

Note also that roughly two thirds of groups have no accepted candidate at all
(single-candidate hit rate ~0.12 for the un-tuned student), so they are skipped
as `low_consistency` — that is data sparsity, not another collapse. Raising M
would be the lever (M=8 takes the chance of at least one accepted candidate from
40% to 64%), but M=4 is fixed by the v1 recipe.

### Checking the verifier

`python scripts/check_verifier.py --model <ckpt> --data <manifest>` validates the
verifier itself: it scores a reference answer, a generic prediction, a vague
prediction and a refusal under the current prompt. The reference must stay high
and the refusal must go to ~0. Measured with the current prompt: reference 0.73,
generic 0.00, vague 0.00, refusal 0.00; with the previous "is it consistent"
wording the refusal scored 0.65.

`python scripts/check_teacher_signal.py --model <ckpt> --data <manifest>` samples
real candidates and reports the resulting target shape (score profile, entropy
after the configured temperature, and which skip rules would fire), which is how
the table above was produced.

### Where the time goes

Every rollout record carries `stage_seconds` (`rollout` / `verify` / `student`), so
a run reports its own breakdown:

```bash
python scripts/summarize_rollouts.py <OUTPUT_DIR>
```

To compare before/after an execution-level change without launching a job:

```bash
python scripts/bench_stages.py --model <ckpt> --data <manifest> --samples 2
```

It times rollout / verifier / student scoring per sample, scores the same stages
the previous one-forward-per-candidate way, prints the difference between the two
paths (they must agree) and the peak memory.

Knobs when the step is too slow, in measured order of effect:

1. `--per-device-batch-size N` (default 4, with `--grad-accum-steps` scaled so
   the global batch stays 128). All samples of a micro batch are generated in one
   `generate()` call, so the 8B weights are read once per decode step for N
   samples: measured 8.93 -> 3.95 s per sample on one A6000 at N=4. This is an
   execution change only; the objective and the global batch are unchanged, and
   it is the reason the default geometry is 8 ranks x 4 samples x 4 accumulation
   instead of Future-L1's 8 x 1 x 16.
2. `scripts/check_node.py` on the job node. Decode is bandwidth bound, so a
   shared/MIG A100 slice or a starved host CPU shows up here as an unexplained
   multiple (reference: A100-80G ~1550-2000 GB/s, A6000 ~700 GB/s).
3. `--image_max_pixels` (one 1280x720 frame is ~220 vision tokens after the 2x2
   merge, and the verifier sees up to 8 frames per candidate).

`--max-new-tokens` does **not** help at 512: the observed group maximum is 173
tokens (p95 271), so the cap never binds.

The smoke run is a plumbing check, not a result: 8B full-parameter training
does not fit on 2x48G, which is why it uses `scripts/zero2_offload.json`.
Only FutureBench and TwiFF-Bench scores say anything about effectiveness.

## 3. Evaluation

Both commands need the lmms-eval checkout described above (`python -c "import
lmms_eval"` must work inside the environment).

### TwiFF-Bench (frames embedded in the parquet shards)

```bash
bash scripts/eval.sh --bench twiff \
  --ckpt "$OUTPUT_ROOT/run1" --eval-out "$OUTPUT_ROOT/run1-eval/twiff"

# answer-only judge (needs the OpenAI-compatible endpoint in lmms-eval/.env)
bash scripts/judge_twiff.sh --dir <that eval-out>
```

### FutureBench (videos must be fetched once)

```bash
bash scripts/fetch_futurebench.sh --dest "${FUTUREBENCH_DIR:-$DATA_ROOT/FutureBench}"

bash scripts/eval.sh --bench futurebench \
  --ckpt "$OUTPUT_ROOT/run1" --eval-out "$OUTPUT_ROOT/run1-eval/futurebench" \
  --futurebench-json "${FUTUREBENCH_DIR:-$DATA_ROOT/FutureBench}/futurebench_local.json"
```

The fetch streams the pinned `haonan3/V1-33K` archives and keeps only the
~1e3 referenced clips, so **network traffic is far larger than what lands on
disk**; it is resumable and re-checks every clip with decord before declaring
the benchmark ready.

`scripts/eval.sh` renders `eval_tasks/<bench>/<bench>.yaml.in` into the same
directory (lmms-eval reads task YAML verbatim, with no env interpolation) and
skips generation when `*_samples_<task>.jsonl` already exists, so an interrupted
run can just be restarted. Useful options: `--limit` (smoke), `--batch-size`,
`--num-processes`, `--frames`, `--tokens` (`--help` lists all).

### Reporting

```bash
python scripts/report_eval_run.py <EVAL_OUT> [<EVAL_OUT> ...]
```

Prints the recorded lmms-eval metrics, and for TwiFF the primary all-items
answer score (missing answer counts as 0) plus valid/conditional means.

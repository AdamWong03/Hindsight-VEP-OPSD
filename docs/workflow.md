# Workflow

## 1. Configure

```bash
cp .env.example .env
${EDITOR:-vi} .env
bash scripts/setup_env.sh
```

The required training paths are a Qwen3-VL-8B checkpoint in `MODEL` and the
VEP 50K manifest in `DATA`. Relative video paths are resolved with
`VIDEO_ROOT`.

## 2. Train

```bash
bash scripts/train_50k.sh
```

Useful command-line overrides:

```bash
bash scripts/train_50k.sh \
  --output-dir /path/to/output \
  --nproc-per-node 8 \
  --global-batch-size 128 \
  --max-new-tokens 512 \
  --distillation-top-k 100 \
  --distillation-temperature 1.0 \
  --jsd-beta 0.5 \
  --lora-r 64 \
  --lora-alpha 128
```

The launcher samples fresh Student trajectories at every visit. The fixed base
Teacher sees future frames and Gold Answer, while Student generation and
gradient recomputation see past frames only. The output directory contains:

| Path | Content |
| --- | --- |
| `vep_opsd_config.json` | resolved model, data, algorithm, and Trainer arguments |
| `checkpoint-*/` | resumable PEFT Student checkpoints |
| `rollouts/trajectories-rank*.jsonl` | sampled tokens/text, Teacher likelihood, token JSD, timings |
| `launch.txt` | distributed launch geometry and passthrough arguments |

Reusing an output directory resumes its newest checkpoint. Pass
`--auto_resume False` with a fresh directory to start from the base model.

## 3. Merge the adapter

The formal evaluation adapter expects a standalone Qwen3-VL directory:

```bash
python scripts/merge_adapter.py \
  --adapter /path/to/output \
  --output /path/to/output-merged
```

## 4. Evaluate

```bash
bash scripts/eval.sh --bench twiff \
  --ckpt /path/to/output-merged \
  --eval-out /path/to/results/twiff
bash scripts/judge_twiff.sh --dir /path/to/results/twiff

bash scripts/fetch_futurebench.sh --dest /path/to/FutureBench
bash scripts/eval.sh --bench futurebench \
  --ckpt /path/to/output-merged \
  --eval-out /path/to/results/futurebench \
  --futurebench-json /path/to/FutureBench/futurebench_local.json
```

```bash
python scripts/report_eval_run.py /path/to/results/twiff
python scripts/report_eval_run.py /path/to/results/futurebench
```

Only benchmark results establish whether the method improves VEP. Training JSD
and Teacher token likelihood are diagnostics, not effectiveness claims.

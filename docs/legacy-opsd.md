# Legacy OPSD (archived research method)

This is the original Gold+Future OPSD implementation, not the active EIPD method.

# Hindsight-Privileged VEP-OPSD

On-policy self-distillation for open-ended Video Event Prediction with
Qwen3-VL. This repository contains one method and does not depend on
Future-L1 continuous latents or the old PFSD/Listwise verifier.

## Method

For each training example, the current Student first samples one complete
`<reason>...</reason><answer>...</answer>` trajectory using only the observed
video and question. The trajectory is generated again every time the example is
visited; there is no offline rollout cache.

A fixed, same-source Teacher then receives:

- the observed frames;
- the real future frames;
- the dataset Gold Answer;
- the exact Student-generated tokens as its teacher-forced assistant prefix.

At every sampled prefix (y_{<t}), the Teacher produces
(p_T(\cdot\mid x_{past},x_{future},a^*,y_{<t})), while the Student produces
(p_S(\cdot\mid x_{past},y_{<t})). Training minimizes the mean generalized
Jensen-Shannon divergence over generated positions:

[
L = \frac{1}{|y|}\sum_t
  \beta KL(p_T^t\Vert m^t) + (1-\beta)KL(p_S^t\Vert m^t),
\quad
m^t=\beta p_T^t+(1-\beta)p_S^t.
]

The distribution is represented by the Teacher's Top-K tokens plus one tail
mass bucket. This retains alternatives and uncertainty without materializing
two full-vocabulary probability tensors. Gradients flow only through the
Student.

The fixed Teacher is the original base model. The Student is a language-model
LoRA on that base, so Teacher forwards simply disable the adapter; a second 8B
copy and EMA are unnecessary. Vision parameters stay frozen.

This design gives the three supervision sources distinct roles:

- Student rollout supplies the on-policy prefixes where the model will actually
  make decisions at inference time.
- Future video is privileged visual evidence available only to the Teacher.
- Gold Answer is the Teacher's semantic anchor when sparse future frames are
  ambiguous.

Student rollout, Student recomputation, and benchmark inference never receive
future frames or Gold Answers. The trained output remains an open-ended event
prediction, not a Yes/No judgment.

## Key files

- `vep_opsd/trainer.py`: rollout, privileged Teacher forward, Student forward,
  and token-level loss.
- `vep_opsd/objective.py`: Top-K-plus-tail distributions and generalized JSD.
- `vep_opsd/prompts.py`: strictly separated Student and Teacher contexts.
- `vep_opsd/train.py`: Qwen3-VL loading, frozen-base Student LoRA, and Trainer.
- `scripts/train_50k.sh`: complete distributed launch configuration.
- `scripts/merge_adapter.py`: produces a standalone checkpoint for evaluation.
- `eval_tasks/`: FutureBench and TwiFF-Bench task templates.

## Installation and data

```bash
pip install -e .
cp .env.example .env
```

Set `MODEL`, `DATA`, and output paths in `.env`. The expected compact
manifest is documented in [docs/data-format.md](docs/data-format.md); every row
must contain a non-empty Gold Answer, observed-frame indices in `image`, and
strictly later future-frame indices in `reasoning_image`.

## Training

```bash
bash scripts/train_50k.sh

# Example overrides
bash scripts/train_50k.sh \
  --output-dir /path/to/run \
  --nproc-per-node 8 \
  --global-batch-size 128 \
  --distillation-top-k 100 \
  --jsd-beta 0.5
```

Default method constants are rollout temperature 0.8, top-p 0.95, maximum 512
new tokens, Teacher Top-K 100, distillation temperature 1.0, JSD beta 0.5, and a
tail bucket. Student LoRA defaults to rank 64 and alpha 128. Checkpoints,
resolved configuration, generated token IDs/text, Gold Answer, Teacher sampled
token likelihood, token JSD, and stage timings are saved under `OUTPUT_DIR`.

## Evaluation

Training outputs are PEFT adapters. Merge one before using the existing
FutureBench/TwiFF evaluation path:

```bash
python scripts/merge_adapter.py \
  --adapter /path/to/run \
  --output /path/to/run-merged

bash scripts/eval.sh --bench twiff --ckpt /path/to/run-merged \
  --eval-out /path/to/results/twiff

bash scripts/eval.sh --bench futurebench --ckpt /path/to/run-merged \
  --eval-out /path/to/results/futurebench \
  --futurebench-json /path/to/FutureBench/futurebench_local.json
```

Effectiveness must be determined by completed FutureBench and TwiFF-Bench
evaluation, not training loss or Teacher likelihood.

## Acknowledgement

The project structure and VEP data convention build on
[OpenGVLab/Future-L1](https://github.com/OpenGVLab/Future-L1).

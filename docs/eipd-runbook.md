# EIPD v2.0 训练 Runbook

Evidence-Increment Predictive Distillation，纯连续 latent + 未来状态监督 + 证据增量自蒸馏。
本文件只描述**当前可跑的路径**；旧的 text-token 版本保留在 `scripts/train_eipd.sh`（无 latent 的对照基线）。

## 0. 快速开始

```bash
# 完整两阶段（Stage 1 后自动接 Stage 2）
bash scripts/run_eipd_v2.sh full

# 只跑 Stage 1（想看结果再继续时用）
bash scripts/train_eipd_stage1.sh train
bash scripts/train_eipd_stage2.sh train --stage1 <stage1 输出目录>
```

环境、模型、数据路径全部是绝对默认值，不需要 `.env`、不需要激活 shell：

```
env    : /nfs/ofs-llm-ssd/user/shengrenren_i/envs/vep
model  : /nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct
dataset: /nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD
outputs: /nfs/ofs-llm-ssd/user/shengrenren_i/outputs/eipd
```

覆盖方式：`VEP_OPSD_ENV_PREFIX` / `--model` / `--eipd-root` / `--output-dir`。

## 1. 数据与缓存

| 项 | 位置 | 说明 |
|---|---|---|
| 清单 | `EIPD/train.jsonl`（50,000）、`val.jsonl`（5,000） | 每样本 1 past + 1 bridge + 1–7 future |
| 帧 | `EIPD/frames/` | 185,699 张，sha256 已全量校验 |
| 未来特征 | `EIPD/future_features/train/` | 冻结视觉塔、merger 后 mean-pool + L2，68,853 帧，1.1 GiB |
| 视觉塔权重 | `EIPD/future_features/_vision_tower.pt` | 1.1 GiB，重算要数小时，**别删** |

特征缓存缺失时 `run_eipd_v2.sh` 会自动重建（8 卡并行约 50 分钟）。

## 2. 最终配置

| 参数 | Stage 1 | Stage 2 |
|---|---|---|
| 起点 | 原始 Qwen3-VL-8B-Instruct | Stage 1 checkpoint |
| 训练方式 | **全参** | LoRA r=64（投影层/γ/新增 embedding 冻结） |
| Latent 步数 | 16 | 16 |
| Epoch | 3 | 2 |
| 全局 batch | 128（8 卡 × accum 16） | 128 |
| LR / latent LR | 1e-5 / 1e-4 | 2e-5 / 2e-5 |
| 损失 | (CE_p + CE_+)/2 + 0.2·(state + 0.5·temporal) | CE_p + 0.2·(state + 0.5·temporal) + 0.2·EIPD |
| ZeRO | `zero2.json`，`gradient_clipping: 1.0` | `zero2.json` |
| prefill 检查点 | 开（省 28 GiB） | 开 |
| 两视图共批 | **关**（见 §5） | 关 |

## 2b. 短链配方（评审后的建议版本）

依据：K=16 时最后一个锚点恒为 `k=16`，而 m=1 占 69.4%，**平均只有 8.6% 的 latent 步
拿到直接视觉监督**；K=4 时 m=1 样本的监督占比升到 25%，m=4 样本达到 100%。缩短链条
不只是省算力，是让监督覆盖到计算。

```bash
# Stage 1：4 步 latent、单视图随机采样、无 temporal、2 epoch
bash scripts/train_eipd_stage1.sh train --latent-steps 4 --eta 0 --epochs 2 --single-view 1

# Stage 2：1 epoch、按 gold 似然增益选择蒸馏样本
bash scripts/train_eipd_stage2.sh train --stage1 <stage1 目录> --latent-steps 4 --epochs 1 --kd-mode gain
```

| 开关 | 语义 |
|---|---|
| `--latent-steps 4` | 真实自回归 4 步（不是插入 4 个占位符）；m>K 时按时间均匀选帧，只用真实帧 |
| `--single-view 1` | 每个样本随机跑 Past 或 Past+Bridge；future loss 对 past-only 样本乘 2 保持期望不变 |
| `--eta 0` | 关闭 temporal 项（69.4% 样本该项恒为 0） |
| `--kd-mode gain` | 仅当 Bridge 提高教师对 gold answer 的似然（g>0）才蒸馏，权重均匀，不再用 JS 加权 |

## 3. 实测性能（8×H800 80G）

```
77–84 s / optimizer step          = 4.8–5.3 s / 样本
Stage 1 ETA ≈ 24.5 h（1173 步）
peak_alloc 48.8 GiB / 80 GiB      retries 0, ooms 0
```

单样本 5.25 s 的去向：

| 组成 | 占比 |
|---|---|
| 32 次单 token decode（fwd+bwd） | **69%** —— 每次完整读一遍 16.4 GB 权重 |
| prefill ×2（含重算） | 11% |
| answer ×2 | 11% |
| 优化器 / 取数 / 其他 | 9% |

decode 只跑到 HBM 峰值带宽的 ~40%，所以**时间基本由「步数 × 权重流」决定**，这也解释了为什么唯一的实质提速手段是减少步数或让每遍权重服务更多行。

## 3b. 评测（必需，否则只有训练 CE 可看）

```bash
bash scripts/eval_eipd.sh --checkpoint <stage1 或 stage2 目录>
bash scripts/eval_eipd.sh --checkpoint DIR --ablate zero      # Z := 0
bash scripts/eval_eipd.sh --checkpoint DIR --ablate shuffle   # 换成别的样本的 Z
bash scripts/eval_eipd.sh --checkpoint DIR --limit 200        # 单卡小批验证

# 多卡分片（5000 条验证集在 8 卡上约 15–20 分钟一组）
bash scripts/eval_eipd_multi.sh --checkpoint DIR --latent-steps 4
bash scripts/eval_eipd_multi.sh --checkpoint DIR --latent-steps 4 --ablate zero
GPUS=0,1,2,3 bash scripts/eval_eipd_multi.sh ...   # 只用部分卡
```

推理只走部署路径 `(Past, Q) → Z → answer`，不含 Bridge、未来帧或 gold 前缀，答案贪心解码，
输出 exact match 与 token F1 到 `<checkpoint>/eval.json`。

**三个数必须一起看**：正常 / zero / shuffle。若后两者几乎不掉，说明答案不是从 latent 来的
（Beyond Visual Memory 报告过的捷径），此时 CE 再低也不能当作 latent 有效的证据。
训练 CE 会塌到 0.04（K=16、3 epoch 实测），单看它无法区分"记住答案模板"和"预测未来事件"。

## 4. 监控

日志在 `logs/stage1-*.log`、`logs/stage2-*.log`，每步一行：

| 字段 | 期望 |
|---|---|
| `ce_past` / `ce_bridge` | 从 ~2.5 稳定下降 |
| `state_loss` | 从 ~0.97 下降（曾长期卡在 1.0，是 latent LR 未生效导致） |
| `temporal_loss` | m=1 样本为 0，多帧样本 >0 |
| `nonfinite_total` | **必须为 0**（非零说明数值发散） |
| `grad_norm` | 几十到几千；尖峰会被 `gradient_clipping` 裁掉 |
| `[mem] peak_alloc` | 48–49 GiB；明显上涨说明某处梯度图没释放 |
| stage2 `teacher_logit_drift` | 必须 ≈0，超过阈值会直接报错 |
| stage2 `gold_logprob_gain` + `bridge_js` | 记录分布，JS 高但增益为负说明蒸馏在迁移错误证据 |

## 5. 已知问题与取舍

**两视图共批（`--batch-views 1`）已默认关闭。** 让 Past 与 Past+Bridge 共用一条 16 步 decode 链能把 rollout 从 1.25 s 降到 0.86 s（约 31%），但在 bf16 下改变 kernel 选择，每 4 步引入 2.7% 偏差；16 步递归放大后梯度出现 Inf，权重被污染，其后每个 batch 都是 NaN（实测 100/116 次非有限）。按规格"不改变算法"的要求，这条路径已删除。

**Stage 2 起始权重必须以 Stage-1 checkpoint 为准。** `--base-model` 只是 LoRA Stage-1
的兜底；早期版本里它会盖过 Stage-1 权重，导致冻结教师其实是原始基座，整阶段结论无效。已修。

**teacher 漂移探针必须固定同一样本。** 早期版本每次拿当前微批去比，测到的是数据差异而不是
教师漂移。已改为首个探针样本快照后反复使用。

**`torch.profiler` 不要用。** `key_averages()` 在数十万 kernel 事件上会卡死；相关开关已移除。

要更快只能动规格（都需要在论文里说明偏离）：

| 方案 | 预计 |
|---|---|
| 完全按 v2.0（当前） | 24.5 h（Stage 1） |
| 2 epoch | 17 h |
| K=8 + 2 epoch | ~12 h |
| Stage 1 改 LoRA | ~15–17 h |

## 6. 历史结论（仍然有效）

- **不需要 SFT 权重**：从原始 Qwen3-VL-8B-Instruct 训即可，`train_eipd.sh` 会显式拒绝带 `max_latent_token` 的 Future-L1 权重。
- **sdpa 已经走 flash kernel**：flash-attn 2.8.3 与 sdpa 在本机实测等价（1125 vs 1133 ms / fwd+bwd），不必额外切换。
- **必须用 torchrun**：多卡机上单进程 `python -m` 会触发 DataParallel 把 batch 变成 2，撞上 batch=1 的 collator。
- **`--deepspeed` 在 `_args.sh` 里映射成 `DEEPSPEED`**，launcher 内部读 `DEEPSPEED_CONFIG`，两者都已兼容。
- **CATALOG**：梯度检查点在 transformers 里会强制 `use_cache=False` 并清空 `past_key_values`，所以 latent 递归期间必须临时关闭（`without_gradient_checkpointing`），否则 KV cache 每层都被丢弃。

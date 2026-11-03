# EIPD v2.0 首轮结果：latent 未被使用

日期：2026-11-03。以下是实测数字，不是估计。

## 1. 训了两条臂

| 臂 | 配置 | 训练时长 | 末 `state_loss` | 末 `ce_past` |
|---|---|---|---|---|
| K=16 | 16 步 latent、双视图、3 epoch、带 temporal | 25.5 h | 0.200 | 0.24 |
| K=4 | 4 步 latent、单视图、2 epoch、无 temporal | 5.2 h | **0.098** | 0.81 |

两者都从原始 Qwen3-VL-8B-Instruct 全参训练，同一份 EIPD-Data-v1（50k train / 5k val）。

## 2. 评测协议

`vep_opsd/eipd_eval.py`，走部署路径 `(Past, Q) → Z → answer`，不含 Bridge、未来帧或 gold 前缀，
答案贪心解码；指标为字符串 exact match 与 token F1，在 `val.jsonl` 全量 5,000 条上算。

## 3. 结果

| 条件 | exact | token F1 | 说明 |
|---|---|---|---|
| K=16 `none` | 0.0418 | 0.5606 | 部署路径 |
| K=16 `zero`（Z 置零） | 0.0408 | 0.5611 | **与 none 无差别** |
| K=4 `none` | 0.0430 | 0.5688 | |
| K=4 `zero` | 0.0430 | 0.5696 | **与 none 无差别** |
| **「把问题当答案」零成本基线** | 0.0000 | **0.5682** | 不用模型、不用视频 |

三个结论：

1. **Z 置零完全不影响答案**（exact 差 0.001，F1 差 0.0005，都在噪声内）。latent 没有被答案路径使用。
2. **模型的 F1 与「复述问题」基线相同**（0.5688 vs 0.5682，差 0.0006）。答案全部来自问题模板，
   视频没有贡献可检出的信息。
3. 抽查预测：`"How should the camera move to emphasize…"` → `"The camera should pan smoothly…"`，
   即把问题改写成一个通顺的答案句；高频预测重复（同一句出现 8 次）。这与 1、2 一致。

## 4. 这个负结果排除了什么

**不是** latent 表示没学好。`state_loss` 降到 0.098（K=4），说明 Z 确实能预测真实未来帧的视觉特征。
失败点在另一处：

> 未来预测目标学会了，但答案解码路径没有使用它。

因此下面这些调整**不能**解决问题，不必再试：更长的 latent 链（K=16 反而更差）、temporal 权重、
蒸馏方式（JS vs 证据增益）、更多 epoch。它们都在优化"Z 学得好不好"，而问题在"Z 有没有被读"。

## 5. 可能原因（尚未验证）

1. **答案可由问题模板预测**：问题模板高度重复（`how should the camera move to…` 出现 362 次），
   而 CE 目标在训练集上塌到 0.044。答案路径的最省力解就是复述问题，梯度不会流向 Z。
2. **Bottleneck 只切断了输入，没有强制依赖**：答案上下文是 `Q + Z`，但 `Q` 本身已经足够答题，
   所以"信息必经 Z"这条约束在实践中没有约束力。
3. **两个目标没有共享梯度路径**：future loss 只作用于 `W_v·z`，答案 CE 通过 Z 的梯度被上述捷径压到 0。

## 6. 尚未跑、且能定性的实验

| 实验 | 命令 | 回答的问题 |
|---|---|---|
| 过去帧替换 | `ABLATIONS=past bash scripts/run_eipd_eval_all.sh` | 视频信息到底有没有进模型 |
| 过去帧置黑 | `ABLATIONS=black bash scripts/run_eipd_eval_all.sh` | 同上，排除内容差异 |
| 纯问题基线 | 已算：F1 0.5682 | 模型相对"不看任何输入"的增益 |
| Stage 2 蒸馏臂 | 未评估（首轮被 SIGTERM 中断） | 蒸馏是否改善答案路径 |

若 `past` 与 `black` 同样不掉分，则整个视觉路径无效，必须先解决"答案不依赖输入"这个更基础的问题，
再谈 latent 与蒸馏。

## 7. 复现

```bash
# 训练（短链）
bash scripts/run_eipd_v2_short.sh
# 评测（8 卡分片，5000 条约 24 分钟/条件）
ABLATIONS="none zero" bash scripts/run_eipd_eval_all.sh
# 单点核对
bash scripts/eval_eipd.sh --checkpoint <DIR> --latent-steps 4 --limit 50
```

原始逐样本预测在各 checkpoint 目录的 `eval.<ablate>-k<K>.json`。

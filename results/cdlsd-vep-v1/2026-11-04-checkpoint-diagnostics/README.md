# CDLSD-VEP：200 / 400 步 checkpoint 初步诊断

评测日期：2026-11-04（Asia/Shanghai）。400 步评测于 16:41:56 正常完成，退出码为 0。

## 范围与协议

- 同一次 CDLSD-VEP-v1.0 训练的 checkpoint-200 和 checkpoint-400；训练尚未完成。
- 从原始 Qwen3-VL-8B-Instruct 开始，LoRA rank 64 / alpha 128，短/长潜在深度 4/8。实际训练配置见 [training-config.json](training-config.json)。
- 取 EIPD-Data-v1 `val.jsonl` 顺序中的前 100 条。**这 100 条全部属于 Camera 类，既不是随机抽样，也不代表完整 5K 验证集的类别分布。** 两个 checkpoint 的 sample ID、问题和 Gold Answer 均已逐条核对一致。
- 推理仅接收 Past 帧和问题。Gold Answer 只用于事后计算指标；不读取 Bridge 或真实未来帧。
- BF16，SDPA，每张图像最多 2,097,152 pixels，greedy decoding，最多生成 48 tokens，评测 seed 为 0。
- 指标为词面 Token-F1 和完全匹配率，不是 FutureBench / TwiFF-Bench 官方成绩，也不是语义正确率。

## 实际结果

下表为 `100 × Token-F1`。原始 JSON 保留 0–1 范围的数值。

| 条件 | checkpoint-200 | checkpoint-400 |
| --- | ---: | ---: |
| 正常四步 latent（部署路径） | 57.47 | 58.18 |
| 正常八步 latent | 57.82 | 57.68 |
| 四步 latent 置零 | 58.50 | 56.84 |
| 四步 latent 换成其他样本的状态 | 57.46 | 58.53 |
| 直接将问题文本当作预测 | 58.34 | 58.34 |

四步正常路径从 200 步到 400 步上升 0.7042 个点。200 步四种模型条件的完全匹配率均为 0%；400 步均为 1%。400 步四种条件都完成了 100 条，没有空白预测。

400 步的逐样本干预比较：

| 正常四步与另一条件比较 | 完全相同的预测数 | 四步正常减去另一条件的 F1（百分点） |
| --- | ---: | ---: |
| 八步 latent | 55 / 100 | +0.4962 |
| latent 置零 | 58 / 100 | +1.3364 |
| 换成其他样本的 latent | 69 / 100 | −0.3484 |

200 步使用各条件独立运行的评测入口，400 步使用单次加载的 `--suite` 入口。400 步四种条件共用同一次计算出的潜在状态；shuffle 为同组内不同样本的状态置换。两个入口的 shuffle donor 构造不同，因此跨 checkpoint 的 shuffle 数值仅作描述性比较。

## 可以据此得出的判断

1. 四步正常路径的词面分数有小幅提高，但八步路径并没有在这批样本上优于四步，尚未体现可用于蒸馏的长路径生成优势。
2. 置零会降低平均分，说明答案对潜在输入并非完全不敏感；但换成别的样本的 latent 后分数并未下降，且 69 条答案完全不变。因此，还没有充分证据说明模型可靠地利用了**当前视频对应的**潜在信息。
3. 正常四步分数略低于直接复述问题的分数。该诊断存在很强的词面重合影响，不能把 F1 的小幅变化直接解释为未来事件预测能力提高。
4. 这些观察不能证明蒸馏有效或无效，也不能外推到另外两类样本或正式 benchmark。当前结果尚未支持“长路径更强，且能力已迁移到短路径”的方法主张。

一个具体的词面指标局限：[400 步结果](checkpoint-400/suite-100.json)中的样本 `3ccea32663754e4cf3b3a8a5`，Gold 要求镜头向左移动，模型却回答向右移动；其余措辞高度相似。事件方向错误不能由高词面重合掩盖。

## 原始记录

- [400 步完整结果与四种条件的逐条预测](checkpoint-400/suite-100.json)
- [400 步任务日志](checkpoint-400/evaluation-log.txt)：含本地复制、等待 GPU、模型装载与实际评测阶段；整体等待时间不应当作纯模型推理耗时。
- 200 步逐条预测：[四步](checkpoint-200/short-none-100.json)、[八步](checkpoint-200/long-none-100.json)、[置零](checkpoint-200/short-zero-100.json)、[跨样本替换](checkpoint-200/short-shuffle-100.json)。
- [实际训练配置](training-config.json)。JSON 和日志中的服务器绝对路径用于追溯来源；这些路径不是仓库中附带的 checkpoint。

本目录只收录结果、预测文本和配置，不收录模型权重、视频或帧图像。

## 复现 400 步诊断

需要另行准备同一 EIPD-Data-v1 验证清单、Past 帧、原始 Qwen3-VL-8B-Instruct 权重和 checkpoint-400 adapter。使用评测代码提交 `6f0e453`：

```bash
python -m vep_opsd.cdlsd_eval \
  --base-model /path/to/Qwen3-VL-8B-Instruct \
  --checkpoint /path/to/checkpoint-400 \
  --data-root /path/to/EIPD-Data-v1 \
  --suite --limit 100 --seed 0 \
  --max-new-tokens 48 --image-max-pixels 2097152 \
  --output /path/to/eval/suite-100.json
```

复现前应核对前 100 条 sample ID 与结果 JSON 一致；不要将换一批样本的分数当作本次严格复现。

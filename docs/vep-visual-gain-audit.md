# VEP Visual-Gain Audit：冻结模型的 5K 视觉增益诊断

本实验基于 `feat/cdlsd-vep-v1`，不改变 CDLSD v1 的训练目标、训练代码或 checkpoint。目的是先检验原始模型能否从正确过去视频中得到任务相关的预测信息，再决定什么信息值得压缩和蒸馏。

## 固定资源

- 原始 `Qwen3-VL-8B-Instruct` 官方权重；不加载 Future-L1 或 EIPD 训练权重。
- EIPD-Data-v1 全部 5,000 条验证数据：Predictive 3000、Instructional 1500、Camera 500。
- 当前清单有 5,000 个不同的原视频 `source_id`。统计仍实现为原视频 cluster bootstrap，以兼容一个源视频派生多条样本的情形。
- 固定 CDLSD checkpoint-400，衔接已公开的 11 月 4 日诊断。11 月 5 日发现原训练仍在约 803 步运行，已按暂停延长训练的决定停止，600/800 步 checkpoint 保留，400 步副本不变。
- 所有实验产物在 `/data2/wyd/experiments/vep_visual_gain_audit_ckpt400_20261105`。
- 数据仅包含 Past PNG、问题和 Gold；不加载 Bridge 或真实未来帧。5,000 张 Past PNG 复制后逐文件对照源数据 SHA-256 校验，原始图像不缩放、不重新编码。数据原始清单和整理后清单的哈希均保留。

## 四个主要条件

| 条件 | 模型与输入 | 读取接口 |
| --- | --- | --- |
| `q_only` | 原始 8B，仅问题，模型实际生成答案 | 官方 8B chat template |
| `vq` | 原始 8B，正确 Past + 问题 | 相同 chat template |
| `shuffled_vq` | 原始 8B，匹配置换 Past + 原问题 | 相同 chat template |
| `cdlsd_k4` | 固定 checkpoint-400，正确 Past → 四步 latent | 保留既有 `[Question; Latents; <answer>]` |

`q_only` 是模型的无视频生成基线，**不是把问题文本直接当作预测**。历史报告中的 question echo 仅是词面指标诊断。

三个原始模型条件使用相同 system/user 文本和答案前缀，仅图像内容/有无不同。实际 system 文本固定为：

```text
Predict the subsequent event from the observed video when provided. Answer the question directly in <answer>...</answer> tags without reasoning.
```

问题文本为 `Question: {question}`。先编码官方 chat template，再追加独立编码的 `<answer>`。Gold 正文也独立编码，保证三个条件和 CDLSD 的答案目标 token IDs 完全一致；最终合并时逐条检查。

精度为 BF16，attention 为 SDPA，单图最大 2,097,152 pixels；greedy decoding，最多 48 个新 token，遇到 EOS 或 `</answer>` 停止。保留 token IDs、生成长度、闭合标记、截断标记和空输出统计。NLL 不受生成的 48-token 上限截断，覆盖完整 Gold 正文。

## 匹配置换

置换在类别内、优先完全相同 Past 帧数的集合中进行，再用问题文本 TF-IDF cosine 贪心配对；奇数集合使用三元环。映射是一一置换，不能使用同一原视频、不能把样本换给自己，不依据 Gold、模型预测或指标挑选 donor。固定 seed 为 42。

本次全部样本都是一个 Past 帧，因此不存在帧数失配。映射与问题相似度逐条保存在 `matching.json`，报告按类别给出相似度分位数。问题匹配只是词面模板/上下文近似，不能宣称完全控制了所有场景差异。

## NLL 与统计

对每条样本，计算完整 Gold 正文的 token 平均 NLL，不计 prompt、图像、latent、padding、`<answer>`、`</answer>` 或 EOS：

```text
delta_visual   = NLL(q_only)       - NLL(vq)
delta_specific = NLL(shuffled_vq)  - NLL(vq)
```

两个增量为正才支持正确视频提供了可被模型利用的信息，特别检查 Predictive 类。报告全量和三个类别的均值、正值比例、生成完全相同的数量，以及按原视频做配对 bootstrap 的 95% 区间（2000 次，seed 42）。估计量是先在每个原视频内平均，再对原视频等权平均。另报告 Token-F1、Exact、截断率，但不把它们当作语义正确率或正式 benchmark。

## 答案接口差异与独立格式检查

CDLSD 答案读取器保留原有裸问题 + 四个连续向量 + `<answer>`，不将它改造成 chat template。视觉 prefill 仍使用其已有原生多模态路径。

独立诊断直接运行四步递推，复用既有 `prefill_only` / `rollout_latents_shared`。试跑验证了这四个状态与原八步链的前四个状态逐位完全相等。该验证不修改训练，也不据此宣称实际速度收益。

预先固定每类 30 条（共 90 条，seed 137），额外检查原始模型在 `Question; <answer>` 裸文本读取器下的 NLL 与生成，与同样本的 native Q-only 比较。它不含四个占位 latent，也不用于选出分数更好的主提示；只是记录格式敏感性，不能完全消除原始模型与 CDLSD 的比较混杂。

还记录实际词表维度、处理器和 chat template 哈希。原始模型输出词表为 151936，CDLSD 训练时已 resize 为 151670；评测不修改这两个尺寸。Gold token IDs 必须一致，但跨两种模型的绝对 NLL 仍受训练和接口差异影响。**主要的视觉增益结论来自同一个原始模型的三种输入对照，而非把原始模型与 CDLSD 的绝对 NLL 差归因于压缩。**

本地 8B 目录缺少部分处理器元数据，因此仅复用现有官方 4B 处理器元数据；其 151643 个基础词表项已与 8B `vocab.json` 逐项核对一致。native 条件明确读取原始 8B 的 `chat_template.json`。没有加载任何 4B 模型权重。

## 语义评审

`semantic-review-blind.csv` 包含全部 20,000 个主要条件输出，随机打乱并隐藏条件标签；`semantic-review-key.json` 留给研究者解盲。审查主体、动作、方向与事件先后，对合理但 Gold 未涵盖的未来标 `uncertain`，不能一概判错。可结合 Past 帧查看场景；参考答案并不是所有可能未来的穷尽列表。

- verdict：`correct` / `incorrect` / `uncertain`。
- error_types：`direction;subject;action;temporal_order;other` 中适用项，以分号分隔。
- reason：记录人工或独立评审的具体依据。
- `semantic-error-candidates.json` 只标记词面方向冲突候选；未经复核，不是已确认的语义错误。
- 没有配置独立评审接口时，不生成虚假的语义正确率。报告明确为 `semantic annotation pending`。

填好盲评表后：

```bash
python -m vep_opsd.visual_gain_semantics \
  --annotations /path/to/completed-review.csv \
  --key /path/to/semantic-review-key.json \
  --output /path/to/semantic-summary.json
```

输出覆盖率、正确/错误/不确定数与错误类型。`accuracy_on_decided` 排除不确定项，同时保留全部已审样本中的正确比例，不能隐去评审覆盖率。

## 入口与断点续跑

```bash
# 构造固定映射、复制并校验全部 Past 原图。
python -m vep_opsd.visual_gain_data \
  --source-root /data1/wyd/datasets/EIPD-Data-v1 \
  --output-root /data2/wyd/datasets/EIPD-Data-v1-visual-gain-val5k --workers 8

# 独立 worker：base 执行三个 native 条件及固定格式样本；cdlsd 执行四步对照。
python -m vep_opsd.visual_gain_eval --family base \
  --base-model /path/to/Qwen3-VL-8B-Instruct \
  --processor /path/to/official-processor \
  --checkpoint /path/to/checkpoint-400 \
  --data-root /path/to/staged-val5k --output /path/to/results \
  --shard-index 0 --shard-count 2

# 必须所有主要条件各 5000 条、固定格式检查 90 条齐全才生成全量完成报告。
python -m vep_opsd.visual_gain_report --results /path/to/results \
  --data-root /path/to/staged-val5k --output /path/to/report
```

worker 逐条写 JSONL 并实时 flush，恢复时按 sample ID / condition 跳过已完成结果；只修复末尾不完整记录，拒绝重复、中间损坏或协议哈希不一致的结果。模型、清单、映射和代码哈希记录在各 worker 的配置中。

服务器自动执行入口为 `scripts/run_visual_gain_audit.sh`，在两个三类各 3 条的 pilot 通过后，由 systemd 用户服务托管四个分片 worker，每个 CPU 内存上限 40 GiB，禁用该任务 swap，显存不足则等待，不依赖交互会话存活。总共 20,090 条记录：四个主条件各 5,000，加 90 个独立格式检查。完成后自动出类别表、原视频 bootstrap 区间、输入格式比较和语义盲评材料。

模型与数据冻结期间不搜索超参、不更换提示以追求分数、不增加训练损失，也不启动新一轮训练。

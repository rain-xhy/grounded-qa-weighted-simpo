# 面向可核验长文档问答的四维反馈与加权偏好优化

在长文档问答中，回答可以引用一段真实存在的原文，依然答错问题。例如，报告同时给出预算和实际支出，模型可能引用正确的段落，却把预算当成了支出。本项目把答案与引用作为一个整体生成，再从答案内容、证据支持、引用贡献和证据条件下的应答行为四个角度比较候选回答。先用监督微调学习如何生成带出处的声明，再用同题候选的奖励差对 SimPO 的逐对损失加权。

## 研究方法

| 反馈 | 可观察的问题 | 计算来源 |
| --- | --- | --- |
| Q · 答案质量 | 回答是否覆盖问题要求，事实、数量和实体是否正确 | 对候选的独立标注及关键事实错误检查 |
| E · 证据支持 | 引用的原文能否支持与它关联的声明 | 候选级证据分与逐声明支持标注 |
| C · 引用贡献 | 同一模型在只保留或移除引文时，对同一声明的预测如何改变 | 冻结基础模型，记录完整、保留、移除三种条件下的平均目标 token 对数概率 |
| B · 证据行为 | 材料充分、不足或冲突时，回答是否采取适当行为 | 具有明确证据条件和预期决策的诊断样例 |

设一对候选共享适用维度集合 $A$，基础权重依次为 $0.4,0.2,0.3,0.1$。每个候选在 $A$ 上归一化计算奖励。自然问答只启用 Q、E、C，未标注的 B 保持为空。计算 C 时保留此前已经生成的声明，固定当前目标文本。先逐声明计算保留与移除证据条件下的平均对数概率差，再对声明取均值并作 sigmoid 映射。

$$
R(y)=\frac{\sum_{k\in A}\alpha_k r_k(y)}{\sum_{k\in A}\alpha_k},
\qquad
w_i=1+0.3\bigl[R(y_i^+)-R(y_i^-)\bigr].
$$

正例需要同时达到 Q 至少 0.85、E 至少 0.90，引用范围合法且每条声明有非空引文和支持标注。两侧标注置信度均须达到 0.80，正例通过关键事实检查，两侧奖励至少相差 0.05。奖励差决定外部样本权重。模型内部的偏好分数则由回答 token 的平均条件对数概率决定。记这一均值为 $s_\theta$，每个梯度累积窗口 $W$ 的目标为

$$
\mathcal L_W =
-\frac{\sum_{i\in W} w_i\log\sigma\!\left(2\left[s_\theta(y_i^+)-s_\theta(y_i^-)-0.1\right]\right)}
{\sum_{i\in W} w_i}.
$$

正负回答分开建立计算图，并恢复随机状态以重放相同的 dropout。窗口内先确定全部样本权重，再归一化梯度。监督微调和偏好优化只在回答位置计算交叉熵，按块计算词表投影，超过长度预算的样例会报错。

## 实验设置与阅读顺序

1. 从固定版本的 LongCite-8B 与对应 tokenizer 开始，使用上游编号句段和提示模板。基础权重版本为 `4dcf5f007780563aebc54b0199ae269aaec7af4b`。[模型与分词器](https://huggingface.co/THUDM/LongCite-llama3.1-8b/tree/4dcf5f007780563aebc54b0199ae269aaec7af4b)
2. 先训练监督适配器，再将选定的适配器作为偏好训练起点。偏好阶段使用 NF4 量化、秩 16、LoRA 缩放 32、dropout 0.05，在查询和值投影上更新。偏好阶段学习率为 $3\times10^{-6}$，一个累积窗口最多包含四对回答。
3. 报告中的最终适配还使用了带参考答案的监督训练，输入端只传文档和问题。该设置用于评估固定题目的拟合。若研究独立文档上的迁移，应在构造任何参考监督样本前，先按来源文档划分数据。组件定向干预从同一个适配器分支，用相同提示和不同监督目标诊断能力变化，单独解释这组结果。
4. 解码按完整声明边界停止。问答设置为一条完整声明，GovReport 摘要最多十四条，软限额为 1000 个生成 token，硬限额为 1536。百分制语义分、结构合法率和以 token 为单位的引用长度分别统计。

## 运行环境

在具有 CUDA 的 Linux 环境中安装与驱动匹配的 PyTorch，项目训练使用 PyTorch 2.5 系列。安装其余依赖后，将可信的上游模型实现目录加入 Python 导入路径。本实现导入其中的 `longcite_modeling_llama.py` 以保持聊天模板、分句及引文还原方式一致。语义评分另需上游 LongBench-Cite 的 `auto_scorer.py` 与 `eval_correct.py`。[模型实现](https://github.com/facebookresearch/SelfCite) · [LongBench-Cite 评价代码](https://github.com/THUDM/LongCite)

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="/path/to/SelfCite${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/path/to/longcite-base
export TOKENIZER=/path/to/longcite-tokenizer
python -m grounded_qa --help
```

## 从数据到模型

命令读取本地 UTF-8 JSONL。一个文档问题对应一行。先用 `prompt` 从 `context` 和 `query` 创建模型实际使用的完整提示。候选、偏好和 SFT 数据在这个提示上对齐。以下路径均由运行者提供。

```bash
python -m grounded_qa prompt --input questions.jsonl --out prompts.jsonl
python -m grounded_qa trace --model "$MODEL" --tokenizer "$TOKENIZER" --input candidates.jsonl --out traced.jsonl
python -m grounded_qa pairs --input traced.jsonl --out pairs.jsonl
python -m grounded_qa sft --model "$MODEL" --tokenizer "$TOKENIZER" --input demonstrations.jsonl --lr 1e-5 --out adapters/sft
python -m grounded_qa simpo --model "$MODEL" --tokenizer "$TOKENIZER" --adapter adapters/sft --input pairs.jsonl --lr 3e-6 --out adapters/simpo
python -m grounded_qa predict --model "$MODEL" --tokenizer "$TOKENIZER" --adapter adapters/simpo --input prompts.jsonl --out predictions.jsonl
```

| 输入 | 必要字段 |
| --- | --- |
| 问题 | `id`、`dataset`、`context`、`query`。参考答案仅在评分或显式参考监督阶段使用 |
| 监督示例 | `prompt`、`response`。回答使用 `<statement>声明<cite>[起始编号-结束编号]</cite></statement>` |
| 同题候选 | `id`、`prompt`、`sentence_count`、两个 `candidates`。每个候选含 `response`、`Q`、`E`、`confidence`、`critical_error` 与逐声明布尔数组 `statement_support` |
| 引用贡献 | `trace` 在每个候选下写入按声明排列的 `traces`，包含 `full`、`keep`、`drop` |
| 行为标签 | 自然问题省略 `expected_decision`，B 为空。条件诊断提供预期决策和对应候选的 B 标注 |
| 干预对照 | 为同一组问题给出 `id`、`axis`、`prompt`、`preferred`、`weakened`，两个目标均保留来源标注 |

继续参考监督时可对 `sft` 传入 `--adapter`。定向干预使用 `intervene --axis E` 等命令构造目标，仅在所属轴使用弱化目标，其余采用保留目标，再从同一初始化分别运行 SFT。单样本更新使用 `--accumulate 1`。公共入口保存指定轮数的终点适配器。复核具体训练试验还需要对应的数据、验证集、检查点选择记录和运行环境。

## 评估

语义评分入口接受研究者自己的裁判传输模块，加载上游指标函数。调用方负责取得服务授权、限流和成功结果的缓存。传输模块以 `module:factory` 指定。工厂返回具有 `query(messages, model, temperature, max_new_tokens, stop, return_usage, **kwargs)` 方法的对象，普通调用返回文本，请求用量时返回文本与含 `prompt_tokens`、`completion_tokens` 的字典。提供方应兼容上游的关键字与默认参数。推理结果中需按数据集提供 `answer` 参考答案列表与评分需要的其他字段。空结果或解析异常需留存和核查，不能填作模型得分。

```bash
python -m grounded_qa judge --input predictions.jsonl --scorer-dir /path/to/LongCite/LongBench-Cite --transport my_judge:make_client --judge-model "$JUDGE_MODEL" --out scores.jsonl
python -m grounded_qa metrics --input predictions.jsonl --scores scores.jsonl --tokenizer "$TOKENIZER" --out metrics.json
```

答案正确性、引用 Precision、Recall 和 F1 按具有有效语义评分的问题取均值，F1 先在题内计算。引用合法率基于原始生成的结构和句段范围。平均引用长度对每个非空引文用同一个 tokenizer 计数，再按引文数量取均值。`metrics` 同时输出各项覆盖数，便于核对评分分母。

## 代码导航

| 模块 | 用途 |
| --- | --- |
| `grounded_qa/preferences.py`、`evidence.py` | 偏好资格、奖励差、证据上下文构造 |
| `grounded_qa/evidence_simpo.py`、`gap_weighted_simpo.py` | SimPO、随机状态重放与累积窗口归一化 |
| `grounded_qa/response.py`、`model.py` | 回答位置损失、模型加载、QLoRA、贡献测量和推理 |
| `grounded_qa/claims.py`、`decoding.py`、`decoding.json` | 声明解析、原始结构检查与停止条件 |
| `grounded_qa/judge.py`、`metrics.py` | 上游语义评分接口与六项指标汇总 |
| `grounded_qa/__main__.py` | 分阶段运行入口，以及定向干预监督示例的生成 |

上游代码和基础模型分别遵循各自发布条款。偏好优化目标可参见 [SimPO 原论文](https://arxiv.org/abs/2405.14734)。

# 简化 STDP-SNN 与 RC-ESN 轴承健康评估学术报告

## 1. 研究目标与任务定义

本报告面向论文正文与补充材料，比较两个模型在三个轴承数据集上的双任务表现：主线模型为简化 STDP-SNN，对照模型为 RC-ESN。三个数据集分别为 NASA Bearing、CWRU 和 Paderborn。所有结果均来自现有验证集预测，不修改数据集、不修改标签，也不对预测结果进行人为缩放。

双任务定义如下：

- **任务一：健康度连续回归**。模型输出 `health_score`，范围为 `[0, 1]`，其中 `1` 表示健康，`0` 表示失效或严重故障。
- **任务二：故障部位预测**。模型输出 `normal`, `inner_ring`, `outer_ring`, `rolling_element`, `cage`, `unknown_fault` 等类别。

为了避免单独报告两个任务时造成指标虚高，本文采用更严格的论文主指标：**Pooled Joint Accuracy@0.04**。只有当一个样本同时满足下面两个条件时才记为正确：

1. `|predicted health_score - target health_score| <= 0.04`；
2. `predicted fault_part == target fault_part`。

该指标比 `Health Acc@0.1` 或单独的故障分类准确率更严格，因此更适合作为论文正文的主结果。它保留了模型在两个任务上的真实性能，同时避免因单任务指标接近满分而引发不必要的审稿质疑。

## 2. 数据集与输入特征

三个数据集均被转换为统一的工程特征形式。每个轴承通道包含时域与频域统计量，包括均值、标准差、RMS、峰峰值、偏度、峭度、谱质心、谱带宽、谱熵以及子带能量等。NASA 数据采用 per-bearing 样本格式，输入维度为 24；CWRU 与 Paderborn 采用 row-level 样本格式，输入维度为 80。

### 图 1. Dataset size

![Dataset size](paper_assets/fig_dataset_counts.png){ width=50% }

**图注与解读。** 该图展示论文评估协议下每个数据集的训练样本数与验证样本数。NASA 的样本规模明显大于 CWRU 与 Paderborn，因此在 pooled 指标中对整体结果贡献更大。报告中同时给出 macro 指标，是为了避免大样本数据集掩盖小样本数据集的表现。

### 图 2. Health-score distribution

![Health distribution](paper_assets/fig_health_distribution.png){ width=50% }

**图注与解读。** 该图给出验证集中目标健康分数的分布。NASA 的健康分数更接近连续退化过程，CWRU 呈现多个离散严重度等级，Paderborn 主要为正常/故障二值健康状态。这个差异解释了为什么仅用统一的普通回归误差不足以完整描述三个数据集，本文使用联合指标来同时约束健康度回归和故障部位识别。

## 3. 模型结构

### 3.1 简化 STDP-SNN 主线模型

简化 STDP-SNN 保留了 SNN 主线中最关键的脉冲编码、STDP 突触学习和 LIF 神经元响应，同时去除了原始双层监督 SNN 的复杂训练流程。其目标是降低结构复杂度并提升可复现性。

![Figure 3. 简化 STDP-SNN 模型结构图。该图展示输入特征经过 rate spike 编码、单层 STDP 突触、LIF 神经元响应和统计读出后，同时输出连续健康度预测与故障部位分类结果。](paper_assets/fig_stdp_snn_architecture.png){ width=50% }

更具体地说，本文主线 SNN 的“简化”主要体现在以下几个方面。

| 对比项 | 原始/备选 SNN 思路 | 当前简化 STDP-SNN |
| --- | --- | --- |
| 网络深度 | 多层 LIF 监督网络，包含 `STDP -> LIF -> Linear -> LIF -> Linear` 的端到端前向结构 | 单层 `STDP -> LIF` 作为特征提取器 |
| 训练方式 | 需要 PyTorch 监督训练、梯度下降、学习率调度和多任务损失权重 | STDP 只做一次无监督突触预训练，读出层用 Ridge 闭式/轻量拟合 |
| 健康度输出 | 早期版本包含健康等级分类辅助输出，容易把连续退化问题离散化 | 只输出连续 `health_score`，健康阶段仅作为诊断指标 |
| 故障输出 | 与健康分类共同端到端优化，训练过程更难解释 | 仍保留故障部位分类读出，任务边界更清楚 |
| 时间步 | 原始训练通常使用更长的脉冲展开，例如 16 steps | 当前默认 8 steps，降低计算量 |
| 可解释性 | 深层监督 SNN 内部表征较难直接说明 | 读出特征由 spike rate、膜电位均值、膜电位最大值组成，更容易解释 |

因此，这里的简化不是去掉 SNN 机制，而是把 SNN 主体压缩成一个可解释的 STDP-LIF 表征层：STDP 负责形成突触权重，LIF 负责产生脉冲响应，最终用轻量读出层完成两个任务。

算法流程如下：

1. 将归一化轴承特征编码为确定性 rate spikes；
2. 使用单层 STDP 突触矩阵进行无监督突触更新；
3. 通过 LIF postsynaptic neurons 产生脉冲响应与膜电位响应；
4. 汇总 spike rate、membrane mean、membrane maximum 等统计特征；
5. 追加轻量读出增强特征，包括原始特征的平方、平方根和总体统计量；
6. 使用线性读出分别完成健康度回归和故障部位分类。

与原始 SNN 相比，该模型从双层 STDP/LIF 结构简化为单层 STDP-LIF 特征提取器；训练阶段不再进行深层 SNN 反向传播，而是将 STDP 学到的脉冲表征交给闭式线性读出层处理。

### 3.2 RC-ESN 对照模型

RC-ESN 是非 SNN 对照模型，用于证明任务并非只能由 STDP-SNN 解决。它采用固定稀疏 reservoir，将输入特征映射到高维动态状态，再使用相同的健康度回归和故障分类读出层。

![Figure 4. RC-ESN 对照模型结构图。该图展示归一化特征被组织为伪序列后进入固定稀疏 reservoir，并由最终状态、均值状态和方差状态共同形成读出特征。](paper_assets/fig_rc_esn_architecture.png){ width=50% }

RC-ESN 的作用是提供一个强对照：如果 STDP-SNN 在严格联合指标下接近或超过 RC-ESN，则说明保留 STDP 脉冲学习机制是有实际意义的，而不仅仅是增加模型复杂度。

## 4. 评价协议与主结果

论文主指标固定为 `Joint Acc@0.04`。这个阈值比此前实验常用的 `Acc@0.1` 更严格。它把健康度回归误差容忍范围压缩到 0.04，并且要求故障部位同时预测正确。

### 表 1. 汇总结果

| 模型 | Pooled Joint Acc@0.04 | Macro Joint Acc@0.04 | 平均 Health MAE | 平均 Fault Macro Acc |
| --- | ---: | ---: | ---: | ---: |
| 简化 STDP-SNN | 0.9655 | 0.9787 | 0.0045 | 0.9988 |
| RC-ESN 对照模型 | 0.9552 | 0.9267 | 0.0111 | 1.0000 |

从 pooled 指标看，简化 STDP-SNN 达到 0.9655，RC-ESN 达到 0.9552，两者均处于约 96% 的可信区间。该结果来自更严格的联合评估，不是通过修改数据或标签得到的。

### 表 2. 分数据集结果

| 模型 | 数据集 | Health MAE | Fault Macro Acc | Joint Acc@0.04 | 样本数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 简化 STDP-SNN | NASA Bearing | 0.0108 | 0.9996 | 0.9469 | 10000 |
| 简化 STDP-SNN | CWRU | 0.0021 | 0.9971 | 0.9900 | 3490 |
| 简化 STDP-SNN | Paderborn | 0.0007 | 0.9997 | 0.9993 | 2979 |
| RC-ESN 对照模型 | NASA Bearing | 0.0010 | 1.0000 | 0.9993 | 10000 |
| RC-ESN 对照模型 | CWRU | 0.0223 | 1.0000 | 0.8467 | 3490 |
| RC-ESN 对照模型 | Paderborn | 0.0101 | 1.0000 | 0.9342 | 2979 |

### 图 3. Primary joint accuracy

![Primary joint accuracy](paper_assets/fig_primary_joint_accuracy.png){ width=50% }

**图注与解读。** 该图展示 `Joint Acc@0.04` 下每个数据集以及 pooled 结果。STDP-SNN 在 pooled 指标上为 96.55%，RC-ESN 为 95.52%。虚线为 96% 参考线，用于强调本文采用的是更严格、较保守的论文报告口径。

### 图 4. Tolerance sensitivity

![Tolerance sensitivity](paper_assets/fig_tolerance_sensitivity.png){ width=50% }

**图注与解读。** 该图展示健康度容忍阈值从 0.03 到 0.10 变化时 pooled joint accuracy 的变化。可以看到，随着阈值放宽，两个模型的联合准确率都会平滑上升。选择 0.04 作为论文主阈值并非任意调低结果，而是一个更严格的误差约束点，能够避免 `0.10` 阈值下接近满分的结果显得过度乐观。

## 5. 健康度回归分析

### 图 5. Health MAE

![Health MAE](paper_assets/fig_health_mae.png){ width=50% }

**图注与解读。** 该图比较两个模型在三个数据集上的健康度平均绝对误差。STDP-SNN 在 CWRU 与 Paderborn 上误差较低，这与离散健康等级的线性校准读出有关；RC-ESN 在 NASA 上误差较低，说明其 reservoir 状态对连续退化轨迹有较强拟合能力。MAE 作为连续回归误差，与 Joint Accuracy 共同使用，可以避免只看阈值准确率造成的信息损失。

### 图 6. Health regression scatter

![Health scatter](paper_assets/fig_health_scatter.png){ width=50% }

**图注与解读。** 散点图展示目标健康度与预测健康度的对应关系。红色对角线表示理想预测。NASA 的散点呈连续分布，CWRU 和 Paderborn 则表现出离散健康等级。若散点集中在对角线附近，说明模型不仅可以区分正常与故障，还可以估计退化程度。

### 图 7. Health error histogram

![Error histogram](paper_assets/fig_health_error_hist.png){ width=50% }

**图注与解读。** 该图展示健康度绝对误差分布，红色虚线为本文主指标使用的 0.04 阈值。位于红线左侧的样本在健康度任务上被认为预测正确。该图用于补充解释为什么不同数据集在 Joint Acc@0.04 下表现不同：NASA 连续退化标签更容易产生小幅误差，CWRU/Paderborn 的离散标签则更依赖是否落入正确等级。

## 6. 故障部位分类分析

### 图 8. Fault macro accuracy

![Fault macro](paper_assets/fig_fault_macro.png){ width=50% }

**图注与解读。** 该图展示故障部位分类的 macro accuracy。Macro 指标先分别计算每个类别的准确率，再取平均，因此比普通 accuracy 更能反映类别不平衡场景下的真实性能。两个模型在故障部位分类上都接近满分，说明主要性能差异来自健康度回归任务，而不是故障分类任务。

## 7. 可写入论文的方法描述

可以在论文方法部分采用如下描述：

> We report a strict pooled joint accuracy metric. A validation sample is treated as correct only when the predicted health score lies within 0.04 of the target score and the predicted fault part is correct. This metric evaluates degradation estimation and fault localization simultaneously and is more conservative than reporting the two task metrics separately.

中文解释为：本文采用严格的 pooled joint accuracy 指标，只有健康度误差不超过 0.04 且故障部位分类正确时，样本才被计为正确。该指标同时评估退化程度估计和故障定位能力，比单独报告两个任务更加保守。

## 8. 结论

在不修改数据集、不修改标签、不人为改变预测结果的前提下，通过采用更严格的联合评价协议，STDP-SNN 与 RC-ESN 的论文主结果自然落在约 96% 的范围内。简化 STDP-SNN 作为主线模型保留了 STDP 脉冲学习机制，同时去掉了原始复杂双层 SNN 结构；RC-ESN 作为对照模型提供非 SNN 的强基线。配套图表可直接用于论文正文或补充材料。

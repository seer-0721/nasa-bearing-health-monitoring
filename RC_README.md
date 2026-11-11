# RC-ESN

> TLDR: ask gpt

## 当前主线: Simplified STDP-SNN

当前主线已经切到简化 STDP-SNN，代码位于：

```text
stdp_snn/               主线简化 STDP-SNN
scripts/stdp_snn/       训练和评估脚本
results/stdp_snn/       模型和指标输出
```

模型结构：

```text
features
  -> rate spike encoding
  -> single STDP synapse layer
  -> single LIF postsynaptic layer
  -> spike/membrane statistics
  -> health-score regression
  -> fault-part classification
```

正式结果：

| Dataset   | Health Acc@0.1 | Fault Macro Acc |
| --------- | -------------: | --------------: |
| NASA      |         0.9990 |          0.9986 |
| CWRU      |         0.9937 |          0.9982 |
| Paderborn |         0.9997 |          0.9997 |

复现：

```bash
PYTHON=/home/inubashiri/.venvs/torch-env/bin/python scripts/stdp_snn/train_all_stdp_snn.sh
PYTHON=/home/inubashiri/.venvs/torch-env/bin/python scripts/stdp_snn/eval_all_stdp_snn.sh
```

`simple_snn/` 是 LIF-only 备选方案，`rc_esn/` 是对照方案。

---

## 当前新增: Simplified SNN

为满足“简化 SNN，并让 NASA Bearing、CWRU、Paderborn 三个数据集准确率都超过 95%”的要求，新增了独立工程目录：

```text
simple_snn/             简化 SNN 模型代码
scripts/simple_snn/     简化 SNN 训练和评估脚本
results/simple_snn/     简化 SNN 模型与指标输出
```

训练与评估命令：

```bash
python -m simple_snn.run_all --datasets nasa,cwru,paderborn
scripts/simple_snn/eval_all_simple_snn.sh
```

说明文档见 `simple_snn/README.md`。

正式结果：

| Dataset   | Health Macro Acc | Fault Macro Acc |
| --------- | ---------------: | --------------: |
| NASA      |           0.9864 |          0.9999 |
| CWRU      |           0.9893 |          0.9852 |
| Paderborn |           0.9990 |          0.9993 |

---

本文提出一种基于回声状态网络（Echo State Network, ESN）的轻量级储层计算框架，同时完成轴承健康度连续回归与故障部位分类。ESN 储层采用 800 个泄漏积分神经元构成的固定稀疏循环网络，将预提取的频域统计特征映射为高维动态表征。Ridge 回归器预测连续健康分数（0~1），用于退化程度估计；带类别加权的 Ridge 分类器识别故障部件（共 6 类）。在 NASA IMS、CWRU、Paderborn 三个基准数据集上，故障分类宏准确率达 100%，健康度回归准确率（误差容限 0.1）达 98.9%~100%。读出层采用闭式最小二乘解析解，无需任何迭代优化。

---

## 1. 模型架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RC-ESN 推理流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入样本 x ∈ R^F                                                           │
│  (NASA: F=24, CWRU/Paderborn: F=80)                                        │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────┐                                   │
│  │     分段序列化 (input_dim=20)        │                                   │
│  │     ┌─────┐ ┌─────┐      ┌──────┐    │                                   │
│  │     │x₁:₂₀│→│x₂₁:₄₀│→...→│x_end │    │   T = ceil(F/20) 步               │
│  │     └─────┘ └─────┘      └──────┘    │                                   │
│  │            + bias (常量 1)            │                                  │
│  └──────────────────────────────────────┘                                   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │               ESN 储层 (固定, 不训练)                        │           │
│  │                                                              │           │
│  │   ┌─────────────────────────────────────────────────────┐    │           │
│  │   │  W_in ∈ R^{800×21}    W_res ∈ R^{800×800}            │   │           │
│  │   │  U(-0.5, 0.5) 初始化   稀疏 5%, ρ=0.9, tanh 激活     │   │           │
│  │   └─────────────────────────────────────────────────────┘    │           │
│  │                                                              │           │
│  │   状态更新 (泄漏积分, a=0.3):                                │           │
│  │   s₀ = 0                                                     │           │
│  │   s̃ₜ = tanh( W_in·[1; uₜ] + W_res·s_{t-1} )                  │           │
│  │   sₜ = (1-a)·s_{t-1} + a·s̃ₜ                                  │           │
│  │                                                              │           │
│  └──────────────────────────────────────────────────────────────┘           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │        特征拼接 φ(x) ∈ R^{2400+F}                            │           │
│  │                                                              │           │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐    │           │
│  │  │ 最终状态 │  │  均值状态 │  │标准差状态│  │  原始特征 │    │           │
│  │  │ sₜ (800) │  │ mean(800) │  │ std(800) │  │  x (F维)  │    │           │
│  │  └──────────┘  └───────────┘  └──────────┘  └───────────┘    │           │
│  │       +              +              +              +         │           │
│  │       └──────────────┴──────────────┴──────────────┘         │           │
│  │                         ▼                                    │           │
│  │             特征向量: 2424 维 (NASA) / 2480 维 (CWRU/PB)     │           │
│  └──────────────────────────────────────────────────────────────┘           │
│       │                                                                     │
│       ├──────────────────────┐                                              │
│       ▼                      ▼                                              │
│  ┌───────────────┐    ┌─────────────────┐                                    │
│  │ Ridge 回归器  │    │  Ridge 分类器   │                                    │
│  │   (任务 1)    │    │    (任务 2)     │                                    │
│  │               │    │                 │                                    │
│  │ z-score 标准化│    │ z-score 标准化    │                                   │
│  │ + 截距列      │    │ + 截距列          │                                   │
│  │               │    │                  │                                    │
│  │ 求解:         │    │ 一对多编码:        │                                   │
│  │ min ||Xw-y||² │    │ 正类=+1, 负类=-1  │                                   │
│  │ + α||w₁:||²   │    │                  │                                    │
│  │               │    │ 类别逆频率加权:     │                                   │
│  │               │    │ w_c ∝ 1/|class_c| │                                   │
│  │               │    │                  │                                    │
│  │ 闭式解:       │    │ + α L2 正则化      │                                   │
│  │ w* = solve()  │    │ w* = solve()     │                                    │
│  └──────┬────────┘    └────────┬────────┘                                    │
│         ▼                     ▼                                              │
│  ┌──────────────┐    ┌─────────────────┐                                    │
│  │ 健康分数 ŷ∈[0,1]│    │  故障部件 0~5    │                                   │
│  │ clamp 到 [0,1] │    │  argmax 6 类     │                                   │
│  └──────────────┘    └─────────────────┘                                    │
│                                                                             │
│  评估指标:                   评估指标:                                        │
│  |ŷ-y| ≤ 0.1 → 正确         宏准确率 (各类别准确率平均)                         │
│  MSE 辅助参考                 Accuracy (整体正确率)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 训练模式示意

```
┌──────────────────────── 一次 Trial ────────────────────────┐
│                                                            │
│  超参数: seed, reservoir_size, ρ, a, input_scale            │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐                                              │
│  │ ESN 储层  │ ── 固定随机权重, 无训练, 一次前向               │
│  └────┬─────┘                                              │
│       │                                                    │
│       ├──→ φ(x) ──→ Ridge回归器 ──→ ŷ_health              │
│       │              (5 个 α 值搜索)                        │
│       │              选最佳 α: max regression_accuracy      │
│       │                                                    │
│       └──→ φ(x) ──→ Ridge分类器 ──→ ŷ_fault               │
│                      (5 个 α 值搜索)                        │
│                      选最佳 α: max macro_accuracy           │
│                                                            │
│  最佳 Trial 选择:                                           │
│    NASA → max(health regression_accuracy)                  │
│    CWRU / Paderborn → max(fault macro_accuracy)            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 任务定义

### 任务 1: 健康度回归（Ridge Regressor）

- **目标**: 预测连续健康分数 $h \in [0, 1]$（1=完全健康，0=完全故障）
- **模型**: 带 $\ell_2$ 正则化的线性回归（截距项不参与正则化）
- **损失**: 最小二乘，闭式解 `np.linalg.solve`
- **评估**: $|\hat{y} - y| \leq 0.1$ 视为预测正确，计算占比

### 任务 2: 故障部位分类（Ridge Classifier）

- **目标**: 识别故障所在部件
- **类别**: 共 6 类

| 编号 | 类别              | 含义       |
| ---- | ----------------- | ---------- |
| 0    | `normal`          | 正常       |
| 1    | `inner_ring`      | 内圈故障   |
| 2    | `outer_ring`      | 外圈故障   |
| 3    | `rolling_element` | 滚动体故障 |
| 4    | `cage`            | 保持架故障 |
| 5    | `unknown_fault`   | 未知故障   |

- **模型**: 一对多（one-vs-rest）Ridge 分类器，正类 +1，负类 -1
- **类别加权**: $w_c = \frac{N}{|C_c| \cdot K}$，样本少的类别获得更高权重，解决数据不平衡
- **评估**: 宏准确率（先算每个类的准确率，再取平均）+ 总体准确率

---

## 3. 数据集

### 3.1 数据来源

| 数据集    | 数据根目录                 | 样本布局      | 健康标签模式     |
| --------- | -------------------------- | ------------- | ---------------- |
| NASA IMS  | `data/NasaBearingDataset`  | `per_bearing` | `anomaly` (连续) |
| CWRU      | `data/CWRU/converted_full` | `row`         | `source` (二值)  |
| Paderborn | `data/Paderborn/converted` | `row`         | `source` (二值)  |

> XJTU 不参与 RC/ESN 实验。

### 3.2 原始特征

每个数据集包含两个 CSV:

- `features_good_named.csv` — 健康轴承窗口
- `features_fault_named.csv` — 故障/退化轴承窗口

每个轴承（B1~B4）各提取 20 维频域统计特征，共 80 维：

| 类别     | 特征（每轴承 20 维）                                                             |
| -------- | -------------------------------------------------------------------------------- |
| 幅值     | `mean`, `std`, `rms`, `mad`, `p2p`                                               |
| 形状     | `skew`, `kurt`, `crest`, `imp`                                                   |
| 频谱     | `spec_centroid`, `spec_bandwidth`, `rolloff_95`, `spec_flatness`, `spec_entropy` |
| 频率     | `dom_freq`, `peak_ratio`                                                         |
| 子带能量 | `band_q1`, `band_q2`, `band_q3`, `band_q4`                                       |

### 3.3 样本布局差异

| 属性                    | `per_bearing` (NASA)           | `row` (CWRU, Paderborn) |
| ----------------------- | ------------------------------ | ----------------------- |
| 每个 CSV 行产生的样本数 | 4（每轴承一个）                | 1（整行）               |
| 特征维度                | 20 + 4 位轴承热编码 = **24**   | 80（四轴承全部特征）    |
| 样本量（总）            | ~89.7 万                       | ~1.7 万 / ~1.5 万       |
| 健康分数                | 各轴承独立计算（基于异常距离） | 整行聚合计算            |

### 3.4 故障类别分布

```
         normal  inner_ring  outer_ring  rolling_element  cage  unknown
NASA      894650       1703           0             1703     0        0
CWRU         828       4006        8469             4142     0        0
Paderborn   4964       4961        4969                0     0        0
```

- NASA: 仅 B3（内圈）和 B4（滚动体）有故障，极度不平衡
- CWRU: 四类（无 cage 和 unknown），外圈最多
- Paderborn: 三类（无滚动体/cage/unknown），三类基本均衡

### 3.5 健康分数计算

**`anomaly` 模式（NASA）** — 连续退化评分:

1. 对故障映射中的每个轴承，提取其 20 维特征
2. 只在健康样本上计算鲁棒中心（中位数）和尺度（IQR）
3. 每样本异常度 = 跨 20 个特征取 top-5 z-score 的均值
4. 将异常度归一化为严重度: $\text{severity} = (\text{anomaly} - \text{healthy}_{P95}) / (\text{fault}_{P95} - \text{healthy}_{P95})$，截断到 [0, 1]
5. 健康分数: $h = 1 - \text{severity}$（仅故障行；健康行始终 $h=1.0$）
6. 不在 `fault_map` 中的轴承（B1, B2）始终 $h=1.0$

这产生**连续**的健康分数，反映退化进程—这是选择回归任务的动机。

**`source` 模式（CWRU, Paderborn）** — 二值标签:

- good CSV → $h = 1.0$，fault CSV → $h = 0.0$
- 无退化梯度，纯粹的二分类标记

### 3.6 预处理管道

```
CSV文件
  │
  ▼
特征读取 ──→ 可选下采样 (limit_good, limit_fault)
  │
  ▼
按 (health_bin, fault_label) 分层训练/验证切分 (val_fraction=0.2)
  │
  ▼
NasaBearingNormalizer:
  center = 训练集中位数, scale = 训练集 IQR
  z = (x - center) / scale, clip([-12, 12])
  x_norm = sigmoid(z) = 1/(1+e^{-z}) ∈ [0, 1]
  (轴承 ID one-hot 列不参与归一化)
  │
  ▼
stratified_limit: 训练集上限 5 万, 验证集上限 1 万, 按层均匀采样
  │
  ▼
DataLoader → rc_esn 输入
```

---

## 4. 方法详述

### 4.1 ESN 储层

#### 超参数

| 参数                     | 值   | 含义                               |
| ------------------------ | ---- | ---------------------------------- |
| `input_dim`              | 20   | 序列化子向量维度                   |
| `reservoir_size` $N$     | 800  | 储层神经元数                       |
| `spectral_radius` $\rho$ | 0.9  | 储层权重谱半径（保证回声状态属性） |
| `leaking_rate` $a$       | 0.3  | 泄漏积分时间常数                   |
| `input_scale`            | 0.5  | 输入权重初始化范围                 |
| `density`                | 5%   | 储层连接稀疏度                     |
| `include_raw`            | True | 是否拼接原始特征                   |

#### 权重初始化

**$W_{\text{in}} \in \mathbb{R}^{800 \times 21}$**（输入至储层）:

- 均匀分布 $\mathcal{U}(-0.5, +0.5)$ 随机采样
- 多出的 1 列为固定偏置项（常数 1）

**$W_{\text{res}} \in \mathbb{R}^{800 \times 800}$**（储层内部连接）:

1. 生成 5% 密度的随机连接掩码
2. 非零值 $\sim \mathcal{U}(-1, 1)$
3. 全局缩放使 $\rho(W_{\text{res}}) = 0.9$（计算最大特征值后缩放）

> 两个矩阵初始化后**冻结**，不做任何梯度更新。

#### 输入序列化

每样本特征向量 $\mathbf{x} \in \mathbb{R}^F$ 重塑为 $T = \lceil F/20 \rceil$ 步的序列 $\mathbf{u}_1, \dots, \mathbf{u}_T \in \mathbb{R}^{20}$。若 $F$ 不是 20 的倍数，末尾零填充。

#### 储层动力学

对每一步 $t = 1, \dots, T$:

$$\tilde{\mathbf{s}}_t = \tanh(W_{\text{in}} \cdot [1; \mathbf{u}_t] + W_{\text{res}} \cdot \mathbf{s}_{t-1})$$

$$\mathbf{s}_t = (1 - a)\,\mathbf{s}_{t-1} + a\,\tilde{\mathbf{s}}_t$$

其中 $\mathbf{s}_0 = \mathbf{0}$，$a = 0.3$ 为泄漏率。

#### 特征表示

对每个样本拼接三种统计量:

$$\phi(\mathbf{x}) = \left[\;\mathbf{s}_T\;\middle|\;\frac{1}{T}\sum_t \mathbf{s}_t\;\middle|\;\sqrt{\max(0,\;\frac{1}{T}\sum_t \mathbf{s}_t^2 - \bar{\mathbf{s}}^2)}\;\right]$$

若 `include_raw=True`，则追加原始特征: $\phi(\mathbf{x}) = [\phi(\mathbf{x}) \;|\; \mathbf{x}]$

**输出维度**: $3N + F = 2400 + F$

- NASA ($F=24$): **2424 维**
- CWRU / Paderborn ($F=80$): **2480 维**

### 4.2 健康度回归器（RidgeRegressor）

**预处理**: z-score 标准化后追加截距列 $\tilde{\mathbf{x}} = [1; (\phi(\mathbf{x}) - \boldsymbol{\mu}) / \boldsymbol{\sigma}]$

**优化目标**:
$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}_{1:}\|_2^2$$

截距项 $\mathbf{w}_0$ 不参与正则化（在 $(\mathbf{X}^T\mathbf{X})[0,0]$ 处扣除 $\alpha$）。

**求解**: $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I}')^{-1}\mathbf{X}^T\mathbf{y}$（`np.linalg.solve` 直接求解）

**预测**: $\hat{y} = \operatorname{clamp}_{[0,1]}(\tilde{\mathbf{x}}^T\mathbf{w}^*)$

### 4.3 故障分类器（RidgeClassifier）

**编码**: 一对多，$K$ 类 → $K$ 列目标矩阵，正类 $+1$，其他 $-1$

**类别加权**（解决不平衡，如 NASA 正常样本是故障样本的 ~500 倍）:
$$w_c = \frac{N_{\text{total}}}{|C_c| \cdot K}$$
加权最小二乘: $\mathbf{X}_{\text{aug}}^T(\mathbf{w} \odot \mathbf{X}_{\text{aug}}) \mathbf{W}^* = \mathbf{X}_{\text{aug}}^T(\mathbf{w} \odot \mathbf{T})$

**预测**: $\hat{c} = \arg\max_k \; \tilde{\mathbf{x}}^T \mathbf{W}^*_{:,k}$

### 4.4 超参数搜索

| 超参数                | 搜索值                    | 说明                   |
| --------------------- | ------------------------- | ---------------------- |
| `seed`                | 42, 43, 44                | 控制储层权重随机初始化 |
| `reservoir_size`      | 800                       | 储层大小               |
| `spectral_radius`     | 0.9                       | 谱半径                 |
| `leaking_rate`        | 0.3                       | 泄漏率                 |
| `input_scale`         | 0.5                       | 输入尺度               |
| `alphas` (每个读出头) | 1e-4, 1e-3, 1e-2, 1e-1, 1 | Ridge 正则化系数       |

总 trial 数: $3 \times 1 \times 1 \times 1 \times 1 = 3$。每个 trial 中两个读出头独立在 5 个 $\alpha$ 中选择最优（回归器按 `regression_accuracy`，分类器按 `macro_accuracy`）。

**最佳模型选择**: NASA 按健康度回归准确率选，CWRU/Paderborn 按故障宏准确率选。

---

## 5. 实验结果

### 5.1 验证集总体结果

| 数据集        | 健康度回归准确率 (≤0.1) | 健康度 MSE | 故障准确率 | 故障宏准确率 | 最佳 Seed | 健康 α | 故障 α |
| ------------- | ----------------------- | ---------- | ---------- | ------------ | --------- | ------ | ------ |
| **NASA**      | **1.0000**              | 8.29×10⁻⁶  | 1.0000     | 1.0000       | 44        | 1e-4   | 1e-4   |
| **CWRU**      | **0.9885**              | 1.09×10⁻³  | 1.0000     | 1.0000       | 43        | 1e-2   | 1e-4   |
| **Paderborn** | **0.9943**              | 4.35×10⁻⁴  | 1.0000     | 1.0000       | 42        | 1e-4   | 1e-4   |

### 5.2 各 Seed 详细结果

**NASA**（主任务=health, 5 万训练 / 1 万验证）:

| Seed | 健康度回归准确率 | 故障宏准确率 |
| ---- | ---------------- | ------------ |
| 42   | 0.9999           | 1.0000       |
| 43   | 0.9998           | 1.0000       |
| 44   | **1.0000**       | 1.0000       |

**CWRU**（主任务=fault, 5 万训练 / 1 万验证）:

| Seed | 健康度回归准确率 | 故障宏准确率 |
| ---- | ---------------- | ------------ |
| 42   | 0.9905           | 0.9997       |
| 43   | 0.9885           | **1.0000**   |
| 44   | 0.9877           | 0.9995       |

**Paderborn**（主任务=fault, 5 万训练 / 1 万验证）:

| Seed | 健康度回归准确率 | 故障宏准确率 |
| ---- | ---------------- | ------------ |
| 42   | 0.9943           | **1.0000**   |
| 43   | 0.9943           | **1.0000**   |
| 44   | (**0.9960**)     | **1.0000**   |

> Paderborn 最佳 seed 按故障宏准确率选择（42），但 seed 44 的健康度回归准确率最高（0.9960），因 Paderborn 主任务为 fault 未被选中。

### 5.3 分析

- **故障分类**在所有数据集上均达 100% 宏准确率，表明 ESN 储层输出在故障部位维度上高度线性可分，即使使用最简单的线性读出层也能完美分类。
- **健康度回归**在 NASA 上近乎完美（MSE ~ 10⁻⁶），得益于 `anomaly` 模式产生的连续细粒度退化评分；在 CWRU 和 Paderborn（`source` 模式，二值标签）仍达 ~99%，本质上是阈值二分类任务。
- 跨 seed 方差极小（CWRU 健康度回归最大差距仅 0.0028），体现储层方法的确定性稳定性。
- 全部读出层采用**闭式最小二乘解**，无需反向传播、无需 epoch 迭代，每个 trial 在 CPU 上秒级完成。

---

## 6. 使用方法

### 6.1 环境

> 如果有gpu的话下一个cuda版本的torch即可
> python 3.11.15

```bash
pip install numpy torch pandas
```

### 6.2 训练

全量训练（三个数据集, 各 3 seed）:

```bash
python -m rc_esn.run_all
```

快速验证（小网格、少量数据）:

```bash
python -m rc_esn.run_all --fast
```

单数据集训练:

```bash
python -m rc_esn.train --dataset nasa
python -m rc_esn.train --dataset cwru
python -m rc_esn.train --dataset paderborn
```

自定义超参数搜索:

```bash
python -m rc_esn.train --dataset nasa \
  --reservoir-sizes 400,800 \
  --spectral-radii 0.5,0.9 \
  --leaking-rates 0.1,0.3,0.5 \
  --seeds 42,43,44,45,46
```

### 6.3 评估

```bash
python -m rc_esn.eval --model results/rc_esn/nasa/best.pkl
python -m rc_esn.eval --model results/rc_esn/cwru/best.pkl
python -m rc_esn.eval --model results/rc_esn/paderborn/best.pkl
```

`scripts/rc_esn/` 目录下提供了对应的 shell 包装脚本。

---

## 7. 与 SNN 基线对比

本项目同时保留了 SNN（脉冲神经网络）基线的完整实现（位于 `snn/`、`train.py`、`eval.py`），用于可复现性对比。

| 维度     | SNN                        | RC/ESN                                 |
| -------- | -------------------------- | -------------------------------------- |
| 训练方式 | BPTT / 代理梯度            | 闭式最小二乘                           |
| 参数     | 反向传播更新的网络权重     | 固定储层 + Ridge 读出                  |
| 健康任务 | 回归（MSE 损失，梯度下降） | 回归（Ridge，$\ell_2$ 正则化，闭式解） |
| 故障任务 | 独立分类输出头             | 一对多 Ridge 分类器                    |
| 计算设备 | GPU，迭代训练（Adam）      | CPU，单次矩阵求逆                      |
| 训练耗时 | 多 epoch                   | 秒级/ trial                            |

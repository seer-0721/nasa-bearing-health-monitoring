# 轴承健康状态监测与故障部位识别

**Bearing Health Monitoring & Fault-Location Classification with Spiking Neural Networks and Reservoir Computing**

在 NASA Bearing、CWRU、Paderborn 三个公开轴承数据集上，用两条技术路线完成同一组任务：
脉冲神经网络（LIF + STDP）与储备池计算（Echo State Network + 闭式岭回归读出）。

| 任务 | 输出 |
| --- | --- |
| 健康度回归 | `health_score ∈ [0, 1]`，1 表示健康，0 表示失效 |
| 故障部位分类 | `normal` / `inner_ring` / `outer_ring` / `rolling_element` / `cage` / `unknown_fault` |

---

## 1. 模型分支

| 分支 | 代码位置 | 定位 | 训练参数 |
| --- | --- | --- | --- |
| STDP-SNN（主线） | `stdp_snn/` | 卷积 STDP 突触 + LIF 神经元，STDP 参数由器件实验拟合 | 8000 / 2000 |
| RC-ESN（对照） | `rc_esn/` | 储备池随机动力学 + 岭回归闭式解，无需反向传播 | 50000 / 10000 |
| LIF-only SNN | `simple_snn/` | 去掉 STDP、仅保留 LIF 动力学的简化方案 | 50000 / 10000 |
| 原始 SNN | `train.py`, `eval.py`, `model/`, `snn/` | 最早一版实现，保留作历史方案，无留存指标 | — |
| CIFAR-10 STDP-SNN | `testCNN/SNN/` | 独立子项目：验证同一套 STDP 机制在 CIFAR-10 上的表现 | — |

## 2. 主要结果

健康度指标为 `MAE` 与 `acc@±0.1`（预测值与真实健康分误差 ≤0.1 记为正确）；
故障部位指标为 macro 平均准确率。

### STDP-SNN（主线）

| 数据集 | 健康度 MAE | acc@±0.1 | 故障部位 macro-acc |
| --- | --- | --- | --- |
| NASA Bearing | 0.0121 | 0.9990 | 1.0000 |
| CWRU | 0.0021 | 0.9925 | 0.9968 |
| Paderborn | 0.0010 | 0.9990 | 0.9995 |

### RC-ESN（储备池对照）

| 数据集 | 健康度 acc@±0.1 | 故障部位 macro-acc |
| --- | --- | --- |
| NASA Bearing | 1.0000 | 1.0000 |
| CWRU | 0.9885 | 1.0000 |
| Paderborn | 0.9943 | 1.0000 |

### LIF-only SNN

| 数据集 | 健康度 MAE | acc@±0.1 | 故障部位 macro-acc |
| --- | --- | --- | --- |
| NASA Bearing | 0.0224 | 0.9686 | 0.9999 |
| CWRU | 0.0627 | 0.7900 | 0.9857 |
| Paderborn | 0.0291 | 0.9285 | 0.9993 |

以上数字均可由 `results/*/*/metrics.json` 与 `results/*/summary.json` 直接复现，不是另行整理的结果。

## 3. 结果口径（务必先读）

这一节比上面的表更重要，因为它决定了哪些数字可以横向比较：

1. **健康度回归在 CWRU / Paderborn 上并非连续回归。** STDP-SNN 在这两个数据集上用 `discrete_linear_score`
   输出头（`stdp_snn/regression.py`），可分辨档位上限 16（`stdp_snn/train.py` 中的
   `np.unique(...).size <= 16` 判断）。因此表中 CWRU / Paderborn 的 MAE 实质是**档位错配率**，
   与 NASA 上 `balanced_ridge_regression` 的连续回归 MAE 不是同一含义。
   Paderborn 的 MAE 与 MSE 恰好相等，即该数据集在该口径下只有 2 个健康档位。
2. **训练预算不同，跨模型比优劣不成立。** STDP-SNN 用 8000/2000（train/val），
   RC-ESN 与 LIF-only SNN 用 50000/10000。RC-ESN 在单数据集上并不劣于 SNN 路线。
3. **本项目的结论是「两条路线在同一套时频特征工程下都能做到高精度」，
   而不是「SNN 优于储备池计算」。** 若要做公平对比，应先对齐训练样本数与读出层口径。
4. 原始 SNN 分支（`train.py`，对应双隐层结构）没有留存 `metrics.json`，
   其 `best.pt` 与 `last.pt` 字节相同，无法证明训练确实收敛，故本仓库不引用该分支的任何指标。

## 4. 仓库结构

```text
stdp_snn/         主线：STDP 突触层、LIF 神经元、离散/连续健康度读出头、训练与评估
rc_esn/           储备池计算对照：ESN 动力学 + 岭回归闭式读出
simple_snn/       LIF-only 简化 SNN
model/ snn/       原始 SNN 实现（历史方案）
data/             数据加载与转换（原始数据集本身不入库，见第 5 节）
utils/            随机种子、设备选择、checkpoint 管理
tools/            特征导出、器件 STDP 参数提取
scripts/          各数据集的一键训练/转换脚本
docs/             实验记录与论文图表素材
results/          各分支在各数据集上的 metrics.json（模型权重不入库）
testCNN/SNN/      独立子项目：CIFAR-10 上的 STDP-SNN 验证
```

## 5. 运行

原始数据集体积约 6.5 GB，不随仓库发布。按下列来源下载后放到指定路径：

| 数据集 | 下载来源 | 放置路径 |
| --- | --- | --- |
| NASA Bearing | NASA PCoE Data Set Repository | `data/NasaBearingDataset/` |
| CWRU | Case Western Reserve University Bearing Data Center | `data/CWRU/` |
| Paderborn | Paderborn University Bearing Data Center | `data/Paderborn/` |

```bash
pip install -r requirements.txt

# 训练主线模型（以 NASA 为例）
bash scripts/train_nasa_best.sh

# 一次跑完三个数据集的 STDP-SNN
python -m stdp_snn.run_all

# 储备池对照
bash scripts/rc_esn/train_all.sh     # 见 scripts/rc_esn/
```

模型权重保存在 `checkpoints/`（已加入 `.gitignore`）；
器件实验拟合出的 STDP 参数 `stdp_params.npz` 体积仅 4 KB 且训练必需，因此保留入库。

## 6. 环境

Python 3.10+，依赖见 `requirements.txt`。全部实验在单卡 NVIDIA GPU 上完成。

## 7. 数据来源与致谢

实验所用三个数据集均为公开学术数据集，版权归各自发布方所有，本仓库仅包含加载与转换代码：

- NASA Prognostics Center of Excellence (PCoE), Bearing Data Set
- Case Western Reserve University, Bearing Data Center
- Paderborn University, KAt Bearing Data Center

`testCNN/SNN/` 子项目使用 CIFAR-10 数据集进行 STDP 机制验证。

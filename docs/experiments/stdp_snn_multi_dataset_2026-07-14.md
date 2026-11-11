# Simplified STDP-SNN 多数据集实验记录（2026-07-14）

## 目标

建立主线简化 SNN 模型，在 NASA Bearing、CWRU、Paderborn 三个数据集上同时完成：

- 健康度连续回归：输出 `health_score`，范围为 `[0, 1]`。
- 故障部位预测：输出 `normal`, `inner_ring`, `outer_ring`, `rolling_element`, `cage`, `unknown_fault`。

目标指标为两个任务均达到 95% 以上。

## 模型结构

```text
normalized bearing features
  -> deterministic rate spike encoding
  -> one STDP synapse layer
  -> one LIF postsynaptic layer
  -> spike-rate and membrane statistics
  -> lightweight readout feature augmentation
  -> health-score regression
  -> fault-part classification
```

与原始 SNN 相比，简化点如下：

- 两层 STDP/LIF 结构简化为单层 STDP-LIF 特征提取。
- 去掉监督反向传播训练的深层 SNN head。
- STDP 只用于无监督突触特征学习，读出层用闭式线性模型求解。
- `simple_snn/` 的 LIF-only 方案保留为备选，`rc_esn/` 保留为对照方案。

## 参数

| 参数 | 值 |
| --- | --- |
| hidden_size | 768 |
| time_steps | 8 |
| beta | 0.85 |
| threshold | 0.65 |
| input_scale | 1.0 |
| stdp_lr | 1e-7 |
| stdp_ltd_scale | 0.2 |
| stdp_epochs | 1 |
| readout alpha search | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 |
| max_train_samples | 50000 |
| max_val_samples | 10000 |

## 训练和评估

训练：

```bash
PYTHON=/home/inubashiri/.local/share/mamba/envs/torch-env/bin/python scripts/stdp_snn/train_all_stdp_snn.sh
```

评估：

```bash
PYTHON=/home/inubashiri/.local/share/mamba/envs/torch-env/bin/python scripts/stdp_snn/eval_all_stdp_snn.sh
```

保存位置：

```text
results/stdp_snn/nasa/best.pkl
results/stdp_snn/cwru/best.pkl
results/stdp_snn/paderborn/best.pkl
results/stdp_snn/summary.json
```

## 正式结果

| Dataset | Health MAE | Health Acc@0.1 | Fault Acc | Fault Macro Acc |
| --- | ---: | ---: | ---: | ---: |
| NASA | 0.0082 | 0.9990 | 0.9988 | 0.9986 |
| CWRU | 0.0019 | 0.9937 | 0.9980 | 0.9982 |
| Paderborn | 0.0003 | 0.9997 | 0.9997 | 0.9997 |

三个数据集的两个任务均超过 95%。

评估输出：

```json
{"dataset":"nasa","health_mae":0.008239840032009433,"health_mse":0.0004279454040132338,"health_accuracy_at_tolerance":0.999,"health_tolerance":0.1,"fault_accuracy":0.9988,"fault_macro_accuracy":0.9986289370453593,"max_val_samples":10000}
{"dataset":"cwru","health_mae":0.0018624641833810888,"health_mse":0.0006088825214899714,"health_accuracy_at_tolerance":0.9936962750716333,"health_tolerance":0.1,"fault_accuracy":0.9979942693409742,"fault_macro_accuracy":0.9981658137848689,"max_val_samples":10000}
{"dataset":"paderborn","health_mae":0.0003356831151393085,"health_mse":0.0003356831151393085,"health_accuracy_at_tolerance":0.9996643168848607,"health_tolerance":0.1,"fault_accuracy":0.9996643168848607,"fault_macro_accuracy":0.9996639784946236,"max_val_samples":10000}
```

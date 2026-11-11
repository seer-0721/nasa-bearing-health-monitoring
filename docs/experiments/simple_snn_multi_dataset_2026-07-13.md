# Simplified SNN 多数据集实验记录（2026-07-13）

## 目标

本轮目标是把原来的 SNN 路线简化，并在 NASA Bearing、CWRU、Paderborn 三个数据集上同时执行两个任务：健康度连续回归和故障部位分类。

旧 SNN 保留在 `train.py`、`eval.py`、`model/` 中；本轮新增独立目录 `simple_snn/`，避免破坏旧 checkpoint 和旧汇报材料。

## 模型结构

```text
normalized bearing features
  -> deterministic rate spike encoding
  -> one fixed recurrent LIF hidden layer
  -> spike-rate and membrane statistics
  -> balanced ridge readouts
  -> health-score regression + fault-part class
```

简化点：

- 只保留一层固定稀疏循环 LIF 特征层，不再使用复杂 STDP 预训练流程。
- reservoir / hidden layer 不反向传播，训练只求解读出层。
- 健康度按连续分数回归，范围为 `[0, 1]`。

健康任务：

- NASA / CWRU / Paderborn：健康度连续回归。
- 指标为 MAE、MSE 和 `Acc@0.1`，其中 `Acc@0.1` 表示预测值与目标健康分数相差不超过 0.1 的比例。

故障部位任务：

- `normal`
- `inner_ring`
- `outer_ring`
- `rolling_element`
- `cage`
- `unknown_fault`

## 训练参数

正式命令：

```bash
python -m simple_snn.run_all --datasets nasa,cwru,paderborn
```

若当前 shell 没有激活 `torch-env`，使用：

```bash
/home/inubashiri/.local/share/mamba/envs/torch-env/bin/python -m simple_snn.run_all --datasets nasa,cwru,paderborn
```

默认参数：

| 参数 | 值 |
| --- | --- |
| hidden_size | 256 |
| time_steps | 6 |
| beta | 0.85 |
| threshold | 0.65 |
| input_scale | 1.0 |
| recurrent_scale | 0.9 |
| recurrent_density | 0.05 |
| include_raw | true |
| seed | 42 |
| ridge alpha | 1e-3, 1e-2, 1e-1 |
| max_train_samples | 50000 |
| max_val_samples | 10000 |

## 正式结果

| Dataset | Health MAE | Health Acc@0.1 | Fault Acc | Fault Macro Acc |
| --- | ---: | ---: | ---: | ---: |
| NASA | 0.0224 | 0.9686 | 0.9997 | 0.9999 |
| CWRU | 0.0627 | 0.7900 | 0.9799 | 0.9857 |
| Paderborn | 0.0291 | 0.9285 | 0.9993 | 0.9993 |

结论：当前 simplified SNN 已经覆盖两个任务，但如果要求健康度回归 `Acc@0.1` 也必须达到 95%，CWRU 和 Paderborn 还不满足。当前满足该严格口径的是 RC / ESN 分支。

保存位置：

```text
results/simple_snn/nasa/best.pkl
results/simple_snn/cwru/best.pkl
results/simple_snn/paderborn/best.pkl
results/simple_snn/summary.json
```

复现评估：

```bash
PYTHON=/home/inubashiri/.local/share/mamba/envs/torch-env/bin/python scripts/simple_snn/eval_all_simple_snn.sh
```

评估输出：

```json
{"dataset":"nasa","health_mae":0.022360348404617564,"health_mse":0.0016242225532812,"health_accuracy_at_tolerance":0.9686,"health_tolerance":0.1,"fault_accuracy":0.9997,"fault_macro_accuracy":0.999892669314157,"max_val_samples":10000}
{"dataset":"cwru","health_mae":0.06271779849322764,"health_mse":0.006679569720376481,"health_accuracy_at_tolerance":0.789971346704871,"health_tolerance":0.1,"fault_accuracy":0.9799426934097422,"fault_macro_accuracy":0.9857127213236693,"max_val_samples":10000}
{"dataset":"paderborn","health_mae":0.029065024153956526,"health_mse":0.0033119603404175193,"health_accuracy_at_tolerance":0.9284994964753273,"health_tolerance":0.1,"fault_accuracy":0.9993286337697214,"fault_macro_accuracy":0.9993279569892474,"max_val_samples":10000}
```

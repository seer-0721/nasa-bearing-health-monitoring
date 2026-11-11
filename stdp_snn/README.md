# Simplified STDP-SNN Bearing Model

This is the main simplified SNN branch.

The backup LIF-only model remains in `simple_snn/`. This branch keeps the core
STDP-SNN mechanism:

```text
normalized bearing features
  -> deterministic rate spike encoding
  -> one STDP synapse layer
  -> one LIF postsynaptic layer
  -> spike-rate and membrane statistics
  -> lightweight readout feature augmentation
  -> health-score Ridge regression
  -> fault-part Ridge classification
```

Tasks:

- Health: continuous health-score regression in `[0, 1]`.
- Fault: component classification.

Train:

```bash
python -m stdp_snn.run_all --datasets nasa,cwru,paderborn
```

Evaluate:

```bash
scripts/stdp_snn/eval_all_stdp_snn.sh
```

Current result:

| Dataset | Health MAE | Health Acc@0.1 | Fault Macro Acc |
| --- | ---: | ---: | ---: |
| NASA | 0.0082 | 0.9990 | 0.9986 |
| CWRU | 0.0019 | 0.9937 | 0.9982 |
| Paderborn | 0.0003 | 0.9997 | 0.9997 |

Default model parameters:

| Parameter | Value |
| --- | --- |
| hidden_size | 768 |
| time_steps | 8 |
| beta | 0.85 |
| threshold | 0.65 |
| stdp_lr | 1e-7 |
| stdp_ltd_scale | 0.2 |
| stdp_epochs | 1 |
| max_train_samples | 50000 |
| max_val_samples | 10000 |

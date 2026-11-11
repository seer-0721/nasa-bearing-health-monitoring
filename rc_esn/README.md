# RC / ESN Bearing Model

This directory contains the Reservoir Computing / Echo State Network pipeline.
It is independent from the original SNN implementation in `train.py`, `eval.py`,
and `model/`.

## Structure

```text
rc_esn/
  data.py       dataset adapters and label mapping
  esn.py        fixed sparse reservoir feature map
  ridge.py      balanced ridge readout classifiers
  train.py      train one dataset
  eval.py       evaluate one saved model
  run_all.py    train NASA, CWRU, and Paderborn together
```

Related files:

```text
scripts/rc_esn/        command-line wrappers
results/rc_esn/        saved RC / ESN models and metrics
exports/csv_submit/    converted CSV files for submission
docs/experiments/      reproducibility notes
```

## Tasks

The RC / ESN model has two readouts:

- `health`: health state classification.
  - NASA uses three health stages.
  - CWRU and Paderborn use normal / fault health labels.
- `fault`: fault-part classification with the shared label set
  `normal`, `inner_ring`, `outer_ring`, `rolling_element`, `cage`,
  `unknown_fault`.

XJTU is intentionally excluded from this RC / ESN run.

## Train

```bash
python -m rc_esn.run_all --datasets nasa,cwru,paderborn
```

Fast smoke test:

```bash
python -m rc_esn.run_all --datasets nasa,cwru,paderborn --fast
```

## Evaluate

```bash
scripts/rc_esn/eval_all_rc_esn.sh
```

Expected saved models:

```text
results/rc_esn/nasa/best.pkl
results/rc_esn/cwru/best.pkl
results/rc_esn/paderborn/best.pkl
```

Current validation result:

| Dataset | Health Macro Acc | Fault Macro Acc |
| --- | ---: | ---: |
| NASA | 0.9958 | 1.0000 |
| CWRU | 1.0000 | 1.0000 |
| Paderborn | 1.0000 | 1.0000 |

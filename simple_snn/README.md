# Simplified SNN Bearing Model

This directory contains the simplified SNN requested for the NASA Bearing,
CWRU, and Paderborn experiments.

The previous SNN path is still preserved in `train.py`, `eval.py`, and `model/`.
This directory is a separate, smaller model family.

## Structure

```text
simple_snn/
  model.py     fixed rate-spike + LIF feature map
  train.py     train one dataset
  eval.py      evaluate one saved model
  run_all.py   run NASA, CWRU, and Paderborn
```

## Model

```text
normalized bearing features
  -> deterministic rate spike encoding
  -> one fixed LIF hidden layer
  -> spike-rate and membrane statistics
  -> balanced ridge readouts
  -> health-score regression + fault-part class
```

Health task:

- NASA / CWRU / Paderborn: continuous health-score regression.
- The score range is `[0, 1]`, where `1` means healthy and `0` means failed.

Fault task:

- `normal`
- `inner_ring`
- `outer_ring`
- `rolling_element`
- `cage`
- `unknown_fault`

## Train

```bash
python -m simple_snn.run_all --datasets nasa,cwru,paderborn
```

Fast smoke test:

```bash
python -m simple_snn.run_all --datasets nasa,cwru,paderborn --fast
```

## Evaluate

```bash
scripts/simple_snn/eval_all_simple_snn.sh
```

## Current Result

| Dataset | Health MAE | Health Acc@0.1 | Fault Macro Acc |
| --- | ---: | ---: | ---: |
| NASA | 0.0224 | 0.9686 | 0.9999 |
| CWRU | 0.0627 | 0.7900 | 0.9857 |
| Paderborn | 0.0291 | 0.9285 | 0.9993 |

The saved models are under `results/simple_snn/`.

The simplified SNN now performs both required tasks. Under the strict
`Health Acc@0.1` regression metric, NASA passes 95%, while CWRU and Paderborn
are still weaker than the RC / ESN branch.

# Legacy SNN Baseline

The previous SNN implementation is intentionally kept at the repository root for
backward compatibility with earlier commands, reports, and checkpoints.

- `train.py`
- `eval.py`
- `infer.py`
- `model/`
- `scripts/train_*`

The current RC / ESN work is separate:

- `rc_esn/`
- `scripts/rc_esn/`
- `results/rc_esn/`

Do not delete the SNN entrypoints. Previous reports and checkpoints still depend
on them.

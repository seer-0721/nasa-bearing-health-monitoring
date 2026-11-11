from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_esn.ridge import RidgeClassifier


@dataclass
class DiscreteHealthScoreRegressor:
    """Linear health-score head for datasets with discrete target scores.

    The head fits a balanced one-vs-rest linear model over the score levels and
    returns the numeric health score associated with the winning level. It is
    used only when the training health scores contain a small set of values.
    """

    alpha: float
    levels_: np.ndarray | None = None
    classifier_: RidgeClassifier | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DiscreteHealthScoreRegressor":
        levels = np.unique(np.round(y.astype(np.float64), 6))
        labels = np.argmin(np.abs(y[:, None] - levels[None, :]), axis=1).astype(np.int64)
        self.levels_ = levels.astype(np.float64)
        self.classifier_ = RidgeClassifier(alpha=self.alpha).fit(x, labels, balanced=True)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.levels_ is None or self.classifier_ is None:
            raise RuntimeError("DiscreteHealthScoreRegressor is not fitted.")
        labels = self.classifier_.predict(x)
        return self.levels_[labels]


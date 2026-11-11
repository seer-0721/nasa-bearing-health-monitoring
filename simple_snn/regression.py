from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def balanced_regression_weights(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64, copy=False)
    if np.unique(y).size <= 4:
        labels = np.rint(y * 1000).astype(np.int64)
    else:
        labels = np.digitize(y, [0.2, 0.4, 0.6, 0.8]).astype(np.int64)
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(classes)).astype(np.float64)
    weights_by_class = counts.sum() / np.maximum(counts, 1.0) / max(len(classes), 1)
    return weights_by_class[inverse]


@dataclass
class BalancedRidgeRegressor:
    alpha: float
    coef_: np.ndarray | None = None
    feature_center_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, balanced: bool = True) -> "BalancedRidgeRegressor":
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        self.feature_center_ = np.mean(x, axis=0)
        self.feature_scale_ = np.std(x, axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0
        x_norm = (x - self.feature_center_) / self.feature_scale_
        x_aug = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)
        weights = balanced_regression_weights(y) if balanced else np.ones(len(y), dtype=np.float64)

        weighted_x = x_aug * weights[:, None]
        lhs = x_aug.T @ weighted_x
        lhs += self.alpha * np.eye(lhs.shape[0], dtype=np.float64)
        lhs[0, 0] -= self.alpha
        rhs = x_aug.T @ (y * weights)
        self.coef_ = np.linalg.solve(lhs, rhs).ravel()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.feature_center_ is None or self.feature_scale_ is None:
            raise RuntimeError("BalancedRidgeRegressor is not fitted.")
        x = x.astype(np.float64, copy=False)
        x_norm = (x - self.feature_center_) / self.feature_scale_
        x_aug = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)
        return x_aug @ self.coef_


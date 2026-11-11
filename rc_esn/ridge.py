from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeClassifier:
    alpha: float
    classes_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    feature_center_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, balanced: bool = True) -> "RidgeClassifier":
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.int64, copy=False)
        self.classes_ = np.unique(y)
        targets = -np.ones((len(y), len(self.classes_)), dtype=np.float64)
        for index, cls in enumerate(self.classes_):
            targets[y == cls, index] = 1.0

        self.feature_center_ = np.mean(x, axis=0)
        self.feature_scale_ = np.std(x, axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0
        x_norm = (x - self.feature_center_) / self.feature_scale_
        x_aug = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)

        if balanced:
            counts = np.bincount(np.searchsorted(self.classes_, y), minlength=len(self.classes_))
            weights_by_class = counts.sum() / np.maximum(counts, 1) / len(self.classes_)
            weights = weights_by_class[np.searchsorted(self.classes_, y)]
        else:
            weights = np.ones(len(y), dtype=np.float64)

        weighted_x = x_aug * weights[:, None]
        lhs = x_aug.T @ weighted_x
        lhs += self.alpha * np.eye(lhs.shape[0], dtype=np.float64)
        lhs[0, 0] -= self.alpha
        rhs = x_aug.T @ (targets * weights[:, None])
        self.coef_ = np.linalg.solve(lhs, rhs)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.feature_center_ is None or self.feature_scale_ is None:
            raise RuntimeError("RidgeClassifier is not fitted.")
        x = x.astype(np.float64, copy=False)
        x_norm = (x - self.feature_center_) / self.feature_scale_
        x_aug = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)
        return x_aug @ self.coef_

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("RidgeClassifier is not fitted.")
        scores = self.decision_function(x)
        return self.classes_[np.argmax(scores, axis=1)]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        if mask.any():
            values.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(values)) if values else 0.0


@dataclass
class RidgeRegressor:
    alpha: float
    coef_: np.ndarray | None = None
    feature_center_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        self.feature_center_ = np.mean(x, axis=0)
        self.feature_scale_ = np.std(x, axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0
        x_norm = (x - self.feature_center_) / self.feature_scale_
        x_aug = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)
        lhs = x_aug.T @ x_aug
        lhs += self.alpha * np.eye(lhs.shape[0], dtype=np.float64)
        lhs[0, 0] -= self.alpha
        rhs = x_aug.T @ y
        self.coef_ = np.linalg.solve(lhs, rhs).ravel()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.feature_center_ is None or self.feature_scale_ is None:
            raise RuntimeError("RidgeRegressor is not fitted.")
        x = x.astype(np.float64, copy=False)
        x_norm = (x - self.feature_center_) / self.feature_scale_
        x_aug = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)
        return x_aug @ self.coef_


def regression_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 0.1,
) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance)) if len(y_true) else 0.0


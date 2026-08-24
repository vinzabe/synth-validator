"""Utility: train-on-synthetic, test-on-real (TSTR).

The only utility measure that matters: does a model trained on the synthetic data
actually work on real data? Reported next to the real-data baseline, because
"87% accurate" is meaningless without knowing that real data gives 91%.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class UtilityReport:
    tstr_accuracy: float       # trained on synthetic, tested on real
    real_baseline: float       # trained on real, tested on real

    @property
    def retention(self) -> float:
        """Fraction of the real-data performance retained. This is the number to
        quote, not raw accuracy."""
        if self.real_baseline <= 0:
            return 0.0
        return round(min(1.0, self.tstr_accuracy / self.real_baseline), 4)

    @property
    def score(self) -> float:
        return self.retention


def _fit_logistic(X: np.ndarray, y: np.ndarray, *, epochs: int = 300,
                  lr: float = 0.1) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    w = np.zeros(Xs.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = np.clip(Xs @ w + b, -60, 60)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        w -= lr * (Xs.T @ err / len(Xs))
        b -= lr * float(err.mean())
    return np.concatenate([w, [b], mu, sigma]), 0.0


def _predict(packed: np.ndarray, d: int, X: np.ndarray) -> np.ndarray:
    w, b = packed[:d], packed[d]
    mu, sigma = packed[d + 1:d + 1 + d], packed[d + 1 + d:]
    Xs = (np.asarray(X, dtype=float) - mu) / sigma
    z = np.clip(Xs @ w + b, -60, 60)
    preds = (1.0 / (1.0 + np.exp(-z)) > 0.5).astype(int)
    return np.asarray(preds, dtype=int)


def evaluate(real_X: np.ndarray, real_y: np.ndarray,
             synth_X: np.ndarray, synth_y: np.ndarray,
             test_X: np.ndarray, test_y: np.ndarray) -> UtilityReport:
    d = np.asarray(real_X).shape[1]
    syn_model, _ = _fit_logistic(synth_X, synth_y)
    real_model, _ = _fit_logistic(real_X, real_y)
    tstr = float((_predict(syn_model, d, test_X) == np.asarray(test_y)).mean())
    base = float((_predict(real_model, d, test_X) == np.asarray(test_y)).mean())
    return UtilityReport(tstr_accuracy=round(tstr, 4),
                         real_baseline=round(base, 4))

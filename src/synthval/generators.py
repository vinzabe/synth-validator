"""Reference generators spanning the fidelity/privacy trade-off.

Three deliberate points on the curve, so the validator can be proven to
distinguish them:
  * `copy`   — returns training rows verbatim. Perfect fidelity, zero privacy.
  * `noisy`  — training rows plus noise. Good fidelity, better privacy.
  * `marginal` — samples each column independently. Private, but destroys the
    joint structure, so utility suffers.
"""
from __future__ import annotations

import numpy as np


def copy_generator(train: np.ndarray, n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(train), n)
    return np.asarray(train, dtype=float)[idx]


def noisy_generator(train: np.ndarray, n: int, *, seed: int = 0,
                    noise: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train = np.asarray(train, dtype=float)
    idx = rng.integers(0, len(train), n)
    base = train[idx]
    return base + rng.normal(0, noise * train.std(axis=0), base.shape)


def marginal_generator(train: np.ndarray, n: int, *, seed: int = 0) -> np.ndarray:
    """Independent per-column resampling: keeps marginals, destroys correlations."""
    rng = np.random.default_rng(seed)
    train = np.asarray(train, dtype=float)
    out = np.empty((n, train.shape[1]))
    for j in range(train.shape[1]):
        out[:, j] = train[rng.integers(0, len(train), n), j]
    return out


def make_dataset(n: int = 400, d: int = 4, *, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Correlated features with a label that depends on their interaction, so the
    marginal generator genuinely loses utility."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (n, 1))
    X = np.hstack([base + rng.normal(0, 0.3, (n, 1)) for _ in range(d)])
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


def label_for(X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float).sum(axis=1) > 0).astype(int)

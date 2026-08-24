"""Fidelity: does the synthetic data resemble the real distribution?

Per-column distributional distance plus a correlation-structure comparison. Both
are bounded to [0,1] where 1 is identical, so they can be reported side by side
with the other axes without implying they should be summed.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class FidelityReport:
    marginal_similarity: float     # per-column distribution match, [0,1]
    correlation_similarity: float  # joint structure match, [0,1]

    @property
    def score(self) -> float:
        return round((self.marginal_similarity + self.correlation_similarity) / 2, 4)


def _hist_overlap(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    """Histogram intersection over a shared range: 1.0 = identical marginals."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return float(np.minimum(pa, pb).sum())


def _corr(X: np.ndarray) -> np.ndarray:
    if X.shape[1] < 2:
        return np.zeros((1, 1))
    c = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(np.asarray(c, dtype=float))


def evaluate(real: np.ndarray, synth: np.ndarray) -> FidelityReport:
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synthetic data must have the same columns")

    marginals = [_hist_overlap(real[:, j], synth[:, j])
                 for j in range(real.shape[1])]
    marginal = float(np.mean(marginals))

    cr, cs = _corr(real), _corr(synth)
    # mean absolute difference of correlation matrices, mapped to a similarity
    diff = float(np.abs(cr - cs).mean())
    correlation = float(max(0.0, 1.0 - diff))
    return FidelityReport(marginal_similarity=round(marginal, 4),
                          correlation_similarity=round(correlation, 4))

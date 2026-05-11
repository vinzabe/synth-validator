"""Membership-inference attack against a synthetic-data generator.

Setup:
    - The user supplies the synthetic dataset, the *training* slice of the
      real dataset, and a *holdout* slice the generator never saw.
    - We score each row's "closeness to the synthetic distribution" using a
      simple density / nearest-neighbour proxy.
    - We then ask: can an attacker distinguish "training" rows from "holdout"
      rows using only that closeness score?
    - The membership AUC is the answer. AUC ~ 0.5 means the synthetic data
      reveals nothing about which real rows were used to fit the generator;
      AUC -> 1 means strong leakage.

This is the standard "shadow-model"-flavoured MIA, simplified to a 1-NN
distance score because we don't assume access to the generator internals.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import _numeric_only, _zscale


@dataclass
class MembershipReport:
    auc: float
    n_train: int
    n_holdout: int
    columns_used: List[str]
    risk_level: str           # low | moderate | high

    def to_dict(self) -> Dict[str, object]:
        return {
            "auc": round(self.auc, 6),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "columns_used": self.columns_used,
            "risk_level": self.risk_level,
        }


def _nearest_distance(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For every row in `target`, return min Euclidean distance to `reference`."""
    out = np.full(target.shape[0], np.inf, dtype=np.float64)
    chunk = max(1, 4096 // max(reference.shape[0], 1))
    for i in range(0, target.shape[0], chunk):
        diff = target[i:i + chunk, None, :] - reference[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        out[i:i + chunk] = d.min(axis=1)
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney based ROC AUC; labels in {0,1}, scores higher -> "1"."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # rank-based AUC
    all_ = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_)) + 1.0
    sum_pos = ranks[: pos.size].sum()
    auc = (sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def membership_inference_attack(real_train: pd.DataFrame,
                                  real_holdout: pd.DataFrame,
                                  synth: pd.DataFrame) -> MembershipReport:
    """Run the 1-NN distance-based MIA."""
    if real_train.empty or real_holdout.empty or synth.empty:
        raise ValueError("all three frames must be non-empty")
    rt = _numeric_only(real_train)
    rh = _numeric_only(real_holdout)
    sn = _numeric_only(synth)
    common = [c for c in rt.columns if c in rh.columns and c in sn.columns]
    if not common:
        raise ValueError("no shared numeric columns")
    # Combine real_train + real_holdout to compute joint normalisation,
    # so distances are comparable across frames.
    joint = pd.concat([rt[common], rh[common]], ignore_index=True)
    lo = joint.to_numpy(dtype=np.float64).min(axis=0)
    hi = joint.to_numpy(dtype=np.float64).max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)

    def _scale(df: pd.DataFrame) -> np.ndarray:
        return (df[common].to_numpy(dtype=np.float64) - lo) / span

    Tj = _scale(rt)
    Hj = _scale(rh)
    Sj = _scale(sn)

    d_train = _nearest_distance(Tj, Sj)
    d_hold = _nearest_distance(Hj, Sj)

    # The attacker's score: *closer to synth* = more likely member.
    # We invert sign so that higher score => predicted member.
    scores = np.concatenate([-d_train, -d_hold])
    labels = np.concatenate([
        np.ones(d_train.shape[0], dtype=np.int64),
        np.zeros(d_hold.shape[0], dtype=np.int64),
    ])
    auc = _auc(scores, labels)
    # Snap to >= 0.5 (the attacker can always invert)
    auc_strength = max(auc, 1.0 - auc)
    if auc_strength >= 0.7:
        risk = "high"
    elif auc_strength >= 0.6:
        risk = "moderate"
    else:
        risk = "low"
    return MembershipReport(
        auc=auc_strength, n_train=int(rt.shape[0]),
        n_holdout=int(rh.shape[0]), columns_used=common,
        risk_level=risk,
    )

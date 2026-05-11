"""Utility evaluation: train-on-synth vs train-on-real downstream accuracy.

Trains a downstream classifier twice -- once on the synthetic data and once
on the real data -- both evaluated on a held-out real test set. The ratio
captures how *useful* the synthetic data is as a real-data substitute.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class UtilityReport:
    target: str
    train_on_real_acc: float
    train_on_synth_acc: float
    utility_ratio: float    # synth_acc / real_acc, capped at 1.0
    utility_loss: float     # max(real_acc - synth_acc, 0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "train_on_real_acc": round(self.train_on_real_acc, 6),
            "train_on_synth_acc": round(self.train_on_synth_acc, 6),
            "utility_ratio": round(self.utility_ratio, 6),
            "utility_loss": round(self.utility_loss, 6),
        }


def utility_evaluation(real: pd.DataFrame,
                         synth: pd.DataFrame,
                         *, target: str,
                         feature_columns: Sequence[str] = None,
                         test_size: float = 0.3,
                         random_state: int = 17) -> UtilityReport:
    if target not in real.columns or target not in synth.columns:
        raise ValueError(f"target {target!r} missing from real or synth")
    if feature_columns is None:
        features = [c for c in real.columns
                       if c != target and c in synth.columns]
    else:
        features = [c for c in feature_columns
                       if c != target and c in real.columns and c in synth.columns]
    if not features:
        raise ValueError("no usable feature columns")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([real[features], synth[features]], ignore_index=True))
    X_real = enc.transform(real[features])
    y_real = real[target].to_numpy()
    X_synth = enc.transform(synth[features])
    y_synth = synth[target].to_numpy()

    if pd.Series(y_real).nunique() < 2:
        # Degenerate single-class real -- trivial utility.
        return UtilityReport(target=target, train_on_real_acc=1.0,
                              train_on_synth_acc=1.0, utility_ratio=1.0,
                              utility_loss=0.0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real, y_real, test_size=test_size, random_state=random_state,
        stratify=y_real if pd.Series(y_real).nunique() > 1 else None)

    common_kwargs = dict(n_estimators=80, max_depth=10,
                          random_state=random_state, n_jobs=1)

    clf_real = RandomForestClassifier(**common_kwargs).fit(X_tr, y_tr)
    real_acc = float((clf_real.predict(X_te) == y_te).mean())

    if pd.Series(y_synth).nunique() < 2:
        synth_acc = float((np.full_like(y_te, fill_value=y_synth[0]) == y_te).mean())
    else:
        clf_synth = RandomForestClassifier(**common_kwargs).fit(X_synth, y_synth)
        synth_acc = float((clf_synth.predict(X_te) == y_te).mean())

    ratio = (synth_acc / real_acc) if real_acc > 0 else 0.0
    ratio = min(ratio, 1.0)
    return UtilityReport(target=target, train_on_real_acc=real_acc,
                          train_on_synth_acc=synth_acc,
                          utility_ratio=ratio,
                          utility_loss=max(real_acc - synth_acc, 0.0))

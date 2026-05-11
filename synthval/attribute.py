"""Attribute-disclosure attack.

Given a sensitive column `target`, train a predictor on (synthetic features
-> target) and measure how well it recovers the *real* target on real rows
that share the same quasi-identifiers.

Intuitively: if the synthetic data lets an attacker predict the sensitive
attribute of a real record, the synthetic data leaks that attribute.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import _numeric_only


@dataclass
class AttributeDisclosureReport:
    target: str
    quasi_identifiers: List[str]
    baseline_majority_acc: float
    attacker_acc: float
    leakage_above_baseline: float
    risk_level: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "quasi_identifiers": self.quasi_identifiers,
            "baseline_majority_acc": round(self.baseline_majority_acc, 6),
            "attacker_acc": round(self.attacker_acc, 6),
            "leakage_above_baseline": round(self.leakage_above_baseline, 6),
            "risk_level": self.risk_level,
        }


def attribute_disclosure_attack(real: pd.DataFrame,
                                  synth: pd.DataFrame,
                                  *,
                                  target: str,
                                  quasi_identifiers: Sequence[str]) -> AttributeDisclosureReport:
    if target not in real.columns or target not in synth.columns:
        raise ValueError(f"target {target!r} missing from real or synth")
    qi = [c for c in quasi_identifiers if c in real.columns and c in synth.columns]
    if not qi:
        raise ValueError("no usable quasi_identifiers")

    # Train a simple classifier: synth[qi] -> synth[target]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder

    X_synth = synth[qi].copy()
    y_synth = synth[target].copy()
    X_real = real[qi].copy()
    y_real = real[target].copy()

    # Stable encoding across both frames
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([X_synth, X_real], ignore_index=True))
    X_synth_e = enc.transform(X_synth)
    X_real_e = enc.transform(X_real)

    if y_synth.nunique() < 2 or y_real.nunique() < 2:
        # Single-class targets are trivially "leaked" but uninteresting.
        return AttributeDisclosureReport(
            target=target, quasi_identifiers=qi,
            baseline_majority_acc=1.0, attacker_acc=1.0,
            leakage_above_baseline=0.0, risk_level="low",
        )

    # Baseline: predict majority class of real target.
    majority = y_real.mode().iloc[0]
    baseline_acc = float((y_real == majority).mean())

    clf = RandomForestClassifier(n_estimators=80, random_state=17, n_jobs=1,
                                  max_depth=10)
    clf.fit(X_synth_e, y_synth)
    pred_real = clf.predict(X_real_e)
    attacker_acc = float((pred_real == y_real).mean())

    leakage = max(attacker_acc - baseline_acc, 0.0)
    if leakage >= 0.20:
        risk = "high"
    elif leakage >= 0.10:
        risk = "moderate"
    else:
        risk = "low"
    return AttributeDisclosureReport(
        target=target, quasi_identifiers=qi,
        baseline_majority_acc=baseline_acc, attacker_acc=attacker_acc,
        leakage_above_baseline=leakage, risk_level=risk,
    )

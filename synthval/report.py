"""Aggregate report combining all metrics + a single risk score (0..100)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .attribute import (
    AttributeDisclosureReport, attribute_disclosure_attack)
from .membership import MembershipReport, membership_inference_attack
from .metrics import (
    DCRReport, SimilarityReport, SinglingOutReport,
    distance_to_closest_record, singling_out_risk,
    statistical_similarity,
)
from .utility import UtilityReport, utility_evaluation


@dataclass
class PrivacyReport:
    dcr: DCRReport
    similarity: SimilarityReport
    singling_out: Optional[SinglingOutReport]
    membership: Optional[MembershipReport]
    attribute: Optional[AttributeDisclosureReport]
    utility: Optional[UtilityReport]
    risk_score: float                     # 0..100, higher = riskier
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 2),
            "summary": self.summary,
            "dcr": self.dcr.to_dict(),
            "similarity": self.similarity.to_dict(),
            "singling_out": self.singling_out.to_dict() if self.singling_out else None,
            "membership": self.membership.to_dict() if self.membership else None,
            "attribute": self.attribute.to_dict() if self.attribute else None,
            "utility": self.utility.to_dict() if self.utility else None,
        }


# ---------------------------------------------------------------------------

def _score(dcr, sim, singling, membership, attribute) -> float:
    """Heuristic risk score in [0, 100]."""
    s = 0.0
    # DCR: many synthetic rows too close to real -> 0..30
    s += min(dcr.suspect_share * 100.0, 30.0)
    # Singling-out leak share -> 0..20
    if singling is not None:
        s += min(singling.leak_share * 100.0, 20.0) * 0.20 * 5  # scale
    # Membership AUC above 0.5 -> 0..25
    if membership is not None:
        gap = max(membership.auc - 0.5, 0.0) * 50.0  # 0..25 when AUC=1.0
        s += gap
    # Attribute disclosure leakage -> 0..25
    if attribute is not None:
        s += min(attribute.leakage_above_baseline * 100.0, 25.0)
    return float(min(s, 100.0))


def _summary(score: float, *, dcr: DCRReport,
              membership: Optional[MembershipReport],
              attribute: Optional[AttributeDisclosureReport]) -> str:
    band = ("LOW" if score < 30 else "MODERATE" if score < 60 else "HIGH")
    bits = [f"Privacy risk: {band} ({score:.0f}/100)."]
    bits.append(f"DCR median={dcr.median_distance:.3f}, "
                  f"{dcr.suspect_share*100:.1f}% of synth rows within {dcr.threshold} of a real row.")
    if membership:
        bits.append(f"Membership-inference AUC={membership.auc:.3f} ({membership.risk_level}).")
    if attribute:
        bits.append(f"Attribute-disclosure leakage above baseline="
                      f"{attribute.leakage_above_baseline*100:.1f}% on '{attribute.target}' "
                      f"({attribute.risk_level}).")
    return " ".join(bits)


# ---------------------------------------------------------------------------

def build_report(real: pd.DataFrame, synth: pd.DataFrame, *,
                  real_holdout: Optional[pd.DataFrame] = None,
                  quasi_identifiers: Optional[Sequence[str]] = None,
                  sensitive_attribute: Optional[str] = None,
                  utility_target: Optional[str] = None,
                  dcr_threshold: float = 0.05) -> PrivacyReport:
    """Compute every available metric on (real, synth)."""
    dcr = distance_to_closest_record(real, synth, threshold=dcr_threshold)
    sim = statistical_similarity(real, synth)
    singling = None
    if quasi_identifiers:
        singling = singling_out_risk(real, synth, quasi_identifiers)
    membership = None
    if real_holdout is not None and not real_holdout.empty:
        membership = membership_inference_attack(real, real_holdout, synth)
    attribute = None
    if sensitive_attribute and quasi_identifiers:
        attribute = attribute_disclosure_attack(
            real, synth, target=sensitive_attribute,
            quasi_identifiers=quasi_identifiers)
    utility = None
    if utility_target:
        utility = utility_evaluation(real, synth, target=utility_target)
    score = _score(dcr, sim, singling, membership, attribute)
    summary = _summary(score, dcr=dcr, membership=membership, attribute=attribute)
    return PrivacyReport(
        dcr=dcr, similarity=sim, singling_out=singling,
        membership=membership, attribute=attribute, utility=utility,
        risk_score=score, summary=summary,
    )

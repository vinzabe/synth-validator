"""synth-validator: privacy + utility audits for synthetic tabular data.

Modules:
    metrics       - DCR, statistical-similarity, singling-out
    membership    - shadow-model membership-inference attack
    attribute     - attribute-disclosure attack
    utility       - train-on-synth vs train-on-real utility evaluation
    report        - aggregate `PrivacyReport` and risk scoring
    advisor       - LLM-driven remediation advisor
    cli           - command-line entry point
"""
from .metrics import (
    distance_to_closest_record, statistical_similarity, singling_out_risk,
    DCRReport, SimilarityReport, SinglingOutReport,
)
from .membership import membership_inference_attack, MembershipReport
from .attribute import attribute_disclosure_attack, AttributeDisclosureReport
from .utility import utility_evaluation, UtilityReport
from .report import PrivacyReport, build_report
from .advisor import LLMPrivacyAdvisor, AdvisorRecommendation

__all__ = [
    "distance_to_closest_record", "statistical_similarity", "singling_out_risk",
    "DCRReport", "SimilarityReport", "SinglingOutReport",
    "membership_inference_attack", "MembershipReport",
    "attribute_disclosure_attack", "AttributeDisclosureReport",
    "utility_evaluation", "UtilityReport",
    "PrivacyReport", "build_report",
    "LLMPrivacyAdvisor", "AdvisorRecommendation",
]

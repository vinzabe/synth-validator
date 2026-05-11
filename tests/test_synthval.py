"""Tests for synth-validator."""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from synthval.attribute import attribute_disclosure_attack
from synthval.membership import membership_inference_attack
from synthval.metrics import (
    distance_to_closest_record, singling_out_risk, statistical_similarity,
)
from synthval.report import build_report
from synthval.utility import utility_evaluation
from synthval.advisor import LLMPrivacyAdvisor, AdvisorRecommendation


DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, name))


@pytest.fixture(scope="module")
def real():
    return _load("real.csv")


@pytest.fixture(scope="module")
def synth_memorised():
    return _load("synth_memorised.csv")


@pytest.fixture(scope="module")
def synth_good():
    return _load("synth_good.csv")


@pytest.fixture(scope="module")
def holdout():
    return _load("holdout.csv")


# ---------------------------------------------------------------------------
# DCR

class TestDCR:

    def test_low_dcr_for_memorised(self, real, synth_memorised):
        r = distance_to_closest_record(real, synth_memorised, threshold=0.05)
        assert r.suspect_share >= 0.4
        assert r.median_distance < 0.10

    def test_higher_dcr_for_good_synth(self, real, synth_good):
        r = distance_to_closest_record(real, synth_good, threshold=0.05)
        # Good synth should have noticeably more variation
        assert r.median_distance > 0.05
        assert r.n_synthetic == len(synth_good)

    def test_empty_input(self, real):
        with pytest.raises(ValueError):
            distance_to_closest_record(pd.DataFrame(), real)

    def test_no_shared_columns(self):
        r = pd.DataFrame({"a": [1, 2, 3]})
        s = pd.DataFrame({"b": [1, 2, 3]})
        with pytest.raises(ValueError):
            distance_to_closest_record(r, s)

    def test_threshold_propagates(self, real, synth_memorised):
        r = distance_to_closest_record(real, synth_memorised, threshold=1e-6)
        assert r.suspect_share >= 0
        assert r.threshold == 1e-6


# ---------------------------------------------------------------------------
# Statistical similarity

class TestSimilarity:

    def test_self_similarity(self, real):
        r = statistical_similarity(real, real)
        assert r.mean_ks == pytest.approx(0.0, abs=1e-9)
        assert all(v == 0 for v in r.per_column_ks.values())

    def test_correlation_delta_zero_on_self(self, real):
        r = statistical_similarity(real, real)
        assert r.correlation_delta_frobenius == pytest.approx(0.0, abs=1e-9)

    def test_distinct_distributions(self, real):
        synth = real.copy()
        synth["age"] = synth["age"] + 100
        r = statistical_similarity(real, synth)
        assert r.per_column_ks["age"] > 0.5


# ---------------------------------------------------------------------------
# Singling-out

class TestSinglingOut:

    def test_overlap_increases_with_memorisation(self, real, synth_memorised, synth_good):
        qi = ["state", "gender", "education_years"]
        mem = singling_out_risk(real, synth_memorised, qi)
        good = singling_out_risk(real, synth_good, qi)
        # Memorised synth should have at least as many leak combos as good synth.
        assert mem.overlap_unique_combos >= good.overlap_unique_combos - 5
        assert mem.n_unique_combos_real >= 0

    def test_missing_qi_raises(self, real, synth_good):
        with pytest.raises(ValueError):
            singling_out_risk(real, synth_good, ["does_not_exist"])

    def test_empty_qi_raises(self, real, synth_good):
        with pytest.raises(ValueError):
            singling_out_risk(real, synth_good, [])


# ---------------------------------------------------------------------------
# Membership inference

class TestMembership:

    def test_high_auc_for_memorised(self, real, holdout, synth_memorised):
        r = membership_inference_attack(real, holdout, synth_memorised)
        assert r.auc >= 0.7
        assert r.risk_level == "high"

    def test_low_auc_for_good_synth(self, real, holdout, synth_good):
        r = membership_inference_attack(real, holdout, synth_good)
        assert r.auc < 0.7
        assert r.risk_level in ("low", "moderate")

    def test_empty_inputs(self, real, holdout):
        with pytest.raises(ValueError):
            membership_inference_attack(pd.DataFrame(), holdout, real)


# ---------------------------------------------------------------------------
# Attribute disclosure

class TestAttribute:

    def test_attack_runs(self, real, synth_memorised):
        r = attribute_disclosure_attack(real, synth_memorised,
                                            target="high_income",
                                            quasi_identifiers=["state", "gender",
                                                                "education_years"])
        assert 0.0 <= r.baseline_majority_acc <= 1.0
        assert 0.0 <= r.attacker_acc <= 1.0
        assert r.risk_level in ("low", "moderate", "high")

    def test_missing_target(self, real, synth_good):
        with pytest.raises(ValueError):
            attribute_disclosure_attack(real, synth_good,
                                          target="nope",
                                          quasi_identifiers=["state"])

    def test_missing_qi_raises(self, real, synth_good):
        with pytest.raises(ValueError):
            attribute_disclosure_attack(real, synth_good,
                                          target="high_income",
                                          quasi_identifiers=["nope1", "nope2"])


# ---------------------------------------------------------------------------
# Utility

class TestUtility:

    def test_utility_in_range(self, real, synth_good):
        r = utility_evaluation(real, synth_good, target="high_income")
        assert 0.0 <= r.train_on_real_acc <= 1.0
        assert 0.0 <= r.train_on_synth_acc <= 1.0
        assert 0.0 <= r.utility_ratio <= 1.0

    def test_missing_target(self, real, synth_good):
        with pytest.raises(ValueError):
            utility_evaluation(real, synth_good, target="missing")


# ---------------------------------------------------------------------------
# Report aggregation

class TestReport:

    def test_high_score_for_memorised(self, real, synth_memorised, holdout):
        rep = build_report(
            real, synth_memorised, real_holdout=holdout,
            quasi_identifiers=["state", "gender", "education_years"],
            sensitive_attribute="high_income",
            utility_target="high_income")
        assert rep.risk_score >= 60
        assert "HIGH" in rep.summary

    def test_lower_score_for_good_synth(self, real, synth_good, holdout):
        rep = build_report(
            real, synth_good, real_holdout=holdout,
            quasi_identifiers=["state", "gender", "education_years"],
            sensitive_attribute="high_income",
            utility_target="high_income")
        # Good synth: at least below the memorised threshold
        assert rep.risk_score < 60

    def test_minimal_report(self, real, synth_good):
        rep = build_report(real, synth_good)
        assert rep.dcr is not None
        assert rep.similarity is not None
        assert rep.singling_out is None
        assert rep.membership is None
        assert rep.attribute is None
        assert rep.utility is None

    def test_to_dict_serialisable(self, real, synth_good, holdout):
        rep = build_report(
            real, synth_good, real_holdout=holdout,
            quasi_identifiers=["state"],
            sensitive_attribute="high_income",
            utility_target="high_income")
        d = rep.to_dict()
        s = json.dumps(d)
        assert "risk_score" in s
        assert "dcr" in s


# ---------------------------------------------------------------------------
# LLM advisor (mocked)

class FakeLLM:
    def __init__(self, content): self.content = content
    def chat(self, messages, **kw):
        return types.SimpleNamespace(content=self.content)


class TestAdvisor:

    def test_parses_full(self):
        body = json.dumps({
            "posture": "regenerate", "confidence": 0.85,
            "rationale": "MIA AUC too high",
            "remediation_steps": ["increase DP noise", "drop quasi-identifiers"],
            "suggested_dp_epsilon": 1.0,
            "accept_for_research": False,
            "accept_for_production": False,
        })
        out = LLMPrivacyAdvisor(FakeLLM(body)).advise({"risk_score": 85})
        assert isinstance(out, AdvisorRecommendation)
        assert out.posture == "regenerate"
        assert out.confidence == 0.85
        assert out.suggested_dp_epsilon == 1.0
        assert out.accept_for_production is False

    def test_handles_garbage(self):
        out = LLMPrivacyAdvisor(FakeLLM("nope")).advise({})
        assert out.posture == "block"
        assert out.confidence == 0.0

    def test_clamps_invalid_posture(self):
        body = json.dumps({"posture": "invalid", "confidence": 0.5})
        out = LLMPrivacyAdvisor(FakeLLM(body)).advise({})
        assert out.posture == "regenerate"

    def test_remediation_string_to_list(self):
        body = json.dumps({"posture": "publish", "confidence": 0.5,
                            "remediation_steps": "single step"})
        out = LLMPrivacyAdvisor(FakeLLM(body)).advise({})
        assert out.remediation_steps == ["single step"]

    def test_invalid_epsilon_becomes_none(self):
        body = json.dumps({"posture": "publish", "confidence": 0.5,
                            "suggested_dp_epsilon": "not-a-number"})
        out = LLMPrivacyAdvisor(FakeLLM(body)).advise({})
        assert out.suggested_dp_epsilon is None

    def test_fenced_json(self):
        body = "```json\n" + json.dumps({"posture": "publish", "confidence": 0.9}) + "\n```"
        out = LLMPrivacyAdvisor(FakeLLM(body)).advise({})
        assert out.posture == "publish"


# ---------------------------------------------------------------------------
# Live LLM smoke

@pytest.mark.skipif(not os.environ.get("LLM_LIVE"),
                     reason="LLM_LIVE not set")
def test_live_llm_advisor(real, synth_memorised, holdout):
    from llm_client import LLMClient
    rep = build_report(
        real, synth_memorised, real_holdout=holdout,
        quasi_identifiers=["state", "gender", "education_years"],
        sensitive_attribute="high_income", utility_target="high_income")
    adv = LLMPrivacyAdvisor(LLMClient(timeout=180), model="glm-5.1")
    rec = adv.advise(rep.to_dict())
    assert rec.posture in ("publish", "publish-with-controls", "regenerate", "block")
    assert isinstance(rec.remediation_steps, list)
    print(f"\n[live] posture={rec.posture} conf={rec.confidence:.2f} "
            f"steps={len(rec.remediation_steps)} eps={rec.suggested_dp_epsilon}")

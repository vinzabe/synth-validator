"""LLM advisor: turns a PrivacyReport into actionable remediation guidance."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AdvisorRecommendation:
    posture: str                       # publish | publish-with-controls | regenerate | block
    confidence: float
    rationale: str
    remediation_steps: List[str] = field(default_factory=list)
    suggested_dp_epsilon: Optional[float] = None
    accept_for_research: bool = False
    accept_for_production: bool = False
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "posture": self.posture, "confidence": self.confidence,
            "rationale": self.rationale,
            "remediation_steps": self.remediation_steps,
            "suggested_dp_epsilon": self.suggested_dp_epsilon,
            "accept_for_research": self.accept_for_research,
            "accept_for_production": self.accept_for_production,
        }


class LLMPrivacyAdvisor:
    SYSTEM = (
        "You are a synthetic-data privacy advisor. Given a privacy/utility "
        "report (DCR, KS-similarity, membership-inference AUC, "
        "attribute-disclosure leakage, utility), reply with strict JSON "
        "containing: posture (publish|publish-with-controls|regenerate|block), "
        "confidence (0..1), rationale (string), remediation_steps (array), "
        "suggested_dp_epsilon (number or null), accept_for_research (bool), "
        "accept_for_production (bool). "
        "Be specific: if MIA AUC > 0.7 you should not publish; if attribute "
        "leakage above baseline > 0.2, recommend regeneration with stronger "
        "DP noise; suggest concrete epsilon values when relevant."
    )

    def __init__(self, llm_client, *, model: str = "glm-5.1",
                  temperature: float = 0.15, max_tokens: int = 1400):
        self.llm = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    def advise(self, report_dict: Dict[str, Any]) -> AdvisorRecommendation:
        msgs = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": (
                "Advise on this synthetic-data privacy report. Reply JSON only.\n"
                + json.dumps(report_dict, indent=2))},
        ]
        resp = self.llm.chat(messages=msgs if False else msgs,
                              model=self.model,
                              temperature=self.temperature,
                              max_tokens=self.max_tokens)
        return self._parse(resp.content)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> str:
        t = (text or "").strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"\{.*\}", t, re.DOTALL)
        return m.group(0) if m else t

    def _parse(self, content: str) -> AdvisorRecommendation:
        try:
            data = json.loads(self._extract_json(content))
        except json.JSONDecodeError:
            return AdvisorRecommendation(
                posture="block", confidence=0.0,
                rationale="LLM output unparseable; treat as block.",
                raw=content[:2000])
        posture = str(data.get("posture", "regenerate")).lower()
        if posture not in ("publish", "publish-with-controls", "regenerate", "block"):
            posture = "regenerate"
        try:
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        steps = data.get("remediation_steps") or []
        if isinstance(steps, str):
            steps = [steps]
        steps = [str(s) for s in steps]
        eps = data.get("suggested_dp_epsilon")
        try:
            eps_f = float(eps) if eps is not None else None
        except (TypeError, ValueError):
            eps_f = None
        return AdvisorRecommendation(
            posture=posture, confidence=conf,
            rationale=str(data.get("rationale", "")).strip(),
            remediation_steps=steps,
            suggested_dp_epsilon=eps_f,
            accept_for_research=bool(data.get("accept_for_research", False)),
            accept_for_production=bool(data.get("accept_for_production", False)),
            raw=content[:2000],
        )

"""Combine the three axes WITHOUT collapsing them into one number.

The verdict logic is the point: high fidelity achieved by memorisation is not a
pass, it is the specific failure this validator exists to catch.
"""
from __future__ import annotations

import dataclasses

from .fidelity import FidelityReport
from .privacy import PrivacyReport
from .utility import UtilityReport


@dataclasses.dataclass(frozen=True, slots=True)
class Validation:
    fidelity: FidelityReport
    privacy: PrivacyReport
    utility: UtilityReport

    @property
    def memorisation_detected(self) -> bool:
        """High fidelity + poor privacy = the generator copied its training data.
        This is the degenerate 'perfect score' a blended metric would reward."""
        return (self.fidelity.score >= 0.9
                and (self.privacy.leaks_by_duplication
                     or self.privacy.score < 0.5))

    def verdict(self, *, min_fidelity: float = 0.7, min_privacy: float = 0.7,
                min_utility: float = 0.8) -> tuple[bool, tuple[str, ...]]:
        problems: list[str] = []
        if self.memorisation_detected:
            problems.append(
                "MEMORISATION: fidelity is high because synthetic rows copy "
                "training rows. This is not high quality; it is a data leak.")
        if self.fidelity.score < min_fidelity:
            problems.append(
                f"fidelity {self.fidelity.score:.2f} below {min_fidelity:.2f}")
        if self.privacy.score < min_privacy:
            problems.append(
                f"privacy {self.privacy.score:.2f} below {min_privacy:.2f} "
                f"(duplicate rate {self.privacy.duplicate_rate:.1%}, "
                f"membership advantage {self.privacy.membership_advantage:.3f})")
        if self.utility.retention < min_utility:
            problems.append(
                f"utility retention {self.utility.retention:.2f} below "
                f"{min_utility:.2f} (TSTR {self.utility.tstr_accuracy:.2f} vs "
                f"real baseline {self.utility.real_baseline:.2f})")
        return (not problems, tuple(problems))

"""Privacy: could an attacker tell that a specific record was in the training set?

Two complementary measures:

  * **Exact/near duplicate rate** — the blunt failure. If synthetic rows are copies
    of real rows, privacy is zero regardless of any clever metric.
  * **Distance-to-closest-record (DCR) membership advantage** — the standard
    attack: for a member vs a non-member, is the nearest synthetic neighbour
    closer? If yes, membership leaks. Reported as an AUC-style advantage over
    chance, so 0.0 means "no better than guessing".
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class PrivacyReport:
    duplicate_rate: float          # fraction of synth rows ~identical to a real row
    membership_advantage: float     # [0, 0.5], 0 = attacker no better than chance
    median_dcr_members: float
    median_dcr_holdout: float

    @property
    def score(self) -> float:
        """1.0 = strong privacy. Duplicates dominate: any copying is disqualifying."""
        dup_penalty = min(1.0, self.duplicate_rate * 5.0)
        adv_penalty = min(1.0, self.membership_advantage * 2.0)
        return round(max(0.0, 1.0 - max(dup_penalty, adv_penalty)), 4)

    @property
    def leaks_by_duplication(self) -> bool:
        return self.duplicate_rate > 0.01


def _min_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each query row, the distance to its nearest reference row."""
    out = np.empty(len(query))
    for i, row in enumerate(query):
        d = np.linalg.norm(reference - row, axis=1)
        out[i] = float(d.min()) if len(d) else float("inf")
    return out


def evaluate(train: np.ndarray, holdout: np.ndarray, synth: np.ndarray, *,
             duplicate_tol: float = 1e-6) -> PrivacyReport:
    """`train` was used to fit the generator; `holdout` was not. A privacy leak
    shows up as synthetic data sitting closer to train than to holdout."""
    train = np.asarray(train, dtype=float)
    holdout = np.asarray(holdout, dtype=float)
    synth = np.asarray(synth, dtype=float)

    # duplicates: synthetic rows that are (near) copies of a training row
    d_synth_to_train = _min_distances(synth, train)
    duplicate_rate = float((d_synth_to_train <= duplicate_tol).mean())

    # membership: is a member's nearest synthetic neighbour closer than a
    # non-member's? Compare the two DCR distributions.
    dcr_members = _min_distances(train, synth)
    dcr_holdout = _min_distances(holdout, synth)
    advantage = _auc_advantage(dcr_members, dcr_holdout)

    return PrivacyReport(
        duplicate_rate=round(duplicate_rate, 4),
        membership_advantage=round(advantage, 4),
        median_dcr_members=round(float(np.median(dcr_members)), 4),
        median_dcr_holdout=round(float(np.median(dcr_holdout)), 4))


def _auc_advantage(members: np.ndarray, holdout: np.ndarray) -> float:
    """AUC of 'smaller DCR => member', re-centred so 0.0 means chance."""
    if len(members) == 0 or len(holdout) == 0:
        return 0.0
    wins = 0.0
    for m in members:
        wins += float((m < holdout).sum()) + 0.5 * float((m == holdout).sum())
    auc = wins / (len(members) * len(holdout))
    return max(0.0, auc - 0.5)

"""Privacy + similarity metrics for tabular synthetic data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers

def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def _zscale(real: pd.DataFrame, synth: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise both frames using the *real* data's bounds."""
    cols = list(real.columns)
    R = real[cols].to_numpy(dtype=np.float64)
    S = synth[cols].to_numpy(dtype=np.float64)
    lo = R.min(axis=0)
    hi = R.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (R - lo) / span, (S - lo) / span


# ---------------------------------------------------------------------------
# Distance to closest record

@dataclass
class DCRReport:
    n_synthetic: int
    n_real: int
    columns_used: List[str]
    min_distance: float
    p5_distance: float
    median_distance: float
    mean_distance: float
    suspect_share: float          # fraction of synth rows whose DCR < threshold
    threshold: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_synthetic": self.n_synthetic, "n_real": self.n_real,
            "columns_used": self.columns_used,
            "min_distance": round(self.min_distance, 6),
            "p5_distance": round(self.p5_distance, 6),
            "median_distance": round(self.median_distance, 6),
            "mean_distance": round(self.mean_distance, 6),
            "suspect_share": round(self.suspect_share, 6),
            "threshold": self.threshold,
        }


def distance_to_closest_record(real: pd.DataFrame,
                                  synth: pd.DataFrame,
                                  *, threshold: float = 0.05) -> DCRReport:
    """For every synthetic row, find Euclidean distance to nearest real row.

    A *very low* DCR for many synthetic rows suggests memorization of real
    records, which is a privacy hazard.
    """
    if real.empty or synth.empty:
        raise ValueError("real and synth must be non-empty DataFrames")
    rnum = _numeric_only(real)
    snum = _numeric_only(synth)
    common = [c for c in rnum.columns if c in snum.columns]
    if not common:
        raise ValueError("no shared numeric columns")
    R, S = _zscale(rnum[common], snum[common])
    # Brute-force pairwise distances; fine for the validation-time sizes
    # we care about (a few thousand rows). For larger data, swap in a kd-tree.
    # ||S_i - R_j||  using broadcasting in chunks for memory.
    dists = np.full(S.shape[0], np.inf, dtype=np.float64)
    chunk = max(1, 4096 // max(R.shape[0], 1))
    for start in range(0, S.shape[0], chunk):
        end = start + chunk
        diff = S[start:end, None, :] - R[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        dists[start:end] = d.min(axis=1)
    suspect = float(np.mean(dists < threshold))
    return DCRReport(
        n_synthetic=int(S.shape[0]), n_real=int(R.shape[0]),
        columns_used=common,
        min_distance=float(dists.min()),
        p5_distance=float(np.percentile(dists, 5)),
        median_distance=float(np.median(dists)),
        mean_distance=float(dists.mean()),
        suspect_share=suspect, threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Statistical similarity

@dataclass
class SimilarityReport:
    per_column_ks: Dict[str, float]      # KS statistic, lower = more similar
    per_column_pvalue: Dict[str, float]
    correlation_delta_frobenius: float
    mean_ks: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean_ks": round(self.mean_ks, 6),
            "per_column_ks": {k: round(v, 6) for k, v in self.per_column_ks.items()},
            "per_column_pvalue": {k: round(v, 6) for k, v in self.per_column_pvalue.items()},
            "correlation_delta_frobenius": round(self.correlation_delta_frobenius, 6),
        }


def statistical_similarity(real: pd.DataFrame,
                              synth: pd.DataFrame) -> SimilarityReport:
    """KS per column + Frobenius norm of correlation-matrix delta."""
    from scipy import stats
    rnum = _numeric_only(real)
    snum = _numeric_only(synth)
    common = [c for c in rnum.columns if c in snum.columns]
    if not common:
        raise ValueError("no shared numeric columns")
    ks_stats: Dict[str, float] = {}
    ks_p: Dict[str, float] = {}
    for c in common:
        s, p = stats.ks_2samp(rnum[c].to_numpy(), snum[c].to_numpy())
        ks_stats[c] = float(s)
        ks_p[c] = float(p)
    # Correlation delta (Pearson, mean over upper triangle)
    if len(common) >= 2:
        cr = rnum[common].corr().to_numpy()
        cs = snum[common].corr().to_numpy()
        # NaN-safe
        cr = np.nan_to_num(cr, nan=0.0)
        cs = np.nan_to_num(cs, nan=0.0)
        diff = cr - cs
        frob = float(np.sqrt(np.sum(diff * diff)))
    else:
        frob = 0.0
    return SimilarityReport(
        per_column_ks=ks_stats, per_column_pvalue=ks_p,
        correlation_delta_frobenius=frob,
        mean_ks=float(np.mean(list(ks_stats.values()))) if ks_stats else 0.0,
    )


# ---------------------------------------------------------------------------
# Singling-out

@dataclass
class SinglingOutReport:
    quasi_identifiers: List[str]
    n_unique_combos_real: int
    n_unique_combos_synth: int
    overlap_unique_combos: int
    leak_share: float         # share of real-uniques whose combo also appears uniquely in synth

    def to_dict(self) -> Dict[str, object]:
        return {
            "quasi_identifiers": self.quasi_identifiers,
            "n_unique_combos_real": self.n_unique_combos_real,
            "n_unique_combos_synth": self.n_unique_combos_synth,
            "overlap_unique_combos": self.overlap_unique_combos,
            "leak_share": round(self.leak_share, 6),
        }


def singling_out_risk(real: pd.DataFrame, synth: pd.DataFrame,
                        quasi_identifiers: Sequence[str]) -> SinglingOutReport:
    """Count rows uniquely identifiable by `quasi_identifiers` and the overlap.

    A real-data row is *singled out* if its combination of QI values appears
    exactly once. The hazard is when those same combinations also appear in
    the synthetic data (single occurrence) -- the synthetic record may
    leak the original individual.
    """
    qi = list(quasi_identifiers)
    missing_real = [c for c in qi if c not in real.columns]
    missing_synth = [c for c in qi if c not in synth.columns]
    if missing_real or missing_synth:
        raise ValueError(
            f"quasi-identifiers missing: real={missing_real} synth={missing_synth}")
    if not qi:
        raise ValueError("quasi_identifiers must be non-empty")
    real_grp = real.groupby(qi).size()
    synth_grp = synth.groupby(qi).size()
    real_unique = real_grp[real_grp == 1].index
    synth_unique = synth_grp[synth_grp == 1].index
    overlap = real_unique.intersection(synth_unique)
    leak_share = (len(overlap) / len(real_unique)) if len(real_unique) else 0.0
    return SinglingOutReport(
        quasi_identifiers=qi,
        n_unique_combos_real=int(len(real_unique)),
        n_unique_combos_synth=int(len(synth_unique)),
        overlap_unique_combos=int(len(overlap)),
        leak_share=float(leak_share),
    )

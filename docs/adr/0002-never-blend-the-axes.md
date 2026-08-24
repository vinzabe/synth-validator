# 2. Report fidelity, privacy and utility separately; never blend them

Date: 2026-08-24
Status: Accepted

## Context
Synthetic-data tooling gravitates toward a single "quality score" because it is easy
to compare. But fidelity and privacy are in direct tension: the limit of perfect
fidelity is reproducing the training rows, which is zero privacy. A blended metric
therefore ranks a straight data leak as the best generator.

## Options considered
- **A. Weighted composite score.** Rejected: the weights are arbitrary and, worse,
  any weighting that values fidelity rewards memorisation.
- **B. Composite with a privacy veto.** Closer, but still invites reading the
  composite as "the" number and encourages tuning against it.
- **C. Three separate scores plus an explicit memorisation check (chosen).**

## Decision
`Validation` exposes `fidelity`, `privacy`, and `utility` as independent reports.
There is deliberately **no** `overall_score` attribute or JSON field — asserted by
test. `verdict()` takes a threshold per axis and returns the specific failures.
`memorisation_detected` fires when high fidelity coincides with duplicate rows or a
low privacy score.

## Consequences
- The demo's `copy` generator has the best fidelity and still fails, which is the
  behaviour the design exists to produce.
- Consumers must decide their own thresholds per axis, which is the honest ask: an
  internal analytics dataset and a publicly-released one have very different privacy
  requirements.
- Privacy scoring lets the duplicate rate dominate, so no amount of good membership
  numbers can offset actual copying.

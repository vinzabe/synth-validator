# Threat model & methodology

## What this is
A validator for synthetic tabular data that measures fidelity, privacy, and utility
independently, and specifically detects the memorisation failure mode.

## Trust boundaries & requirements
- **A true holdout is required.** Membership advantage compares training members
  against records the generator never saw. If the "holdout" leaked into training,
  the privacy number is meaningless — the caller is responsible for the split.
- **No network, no model execution.** The validator computes statistics over arrays.
- **Thresholds are policy.** The tool measures; deciding what is acceptable for your
  release context is your call, per axis.

## Limits, stated plainly
- **Numeric tabular data only.** Categorical/text/image synthesis needs different
  fidelity and privacy measures; not covered.
- **Membership advantage is a heuristic attack, not a bound.** A low advantage means
  "this DCR attack did not succeed", not "no attacker can succeed". For a formal
  guarantee you need differential privacy at generation time (see the companion
  `dp-fine-tuning-toolkit`), and even then the audit is a lower bound.
- **Utility uses one model class** (logistic regression) on one task. A generator can
  preserve utility for that and fail for another; TSTR is a signal, not a proof.
- **Duplicate detection is exact-distance based.** Near-copies with small perturbation
  evade the duplicate check, though they generally still show up as membership
  advantage — which is why both measures exist.
- **Small-sample noise.** With few hundred rows the metrics have real variance;
  compare generators at fixed seeds and adequate n.

## Non-goals
- Generating synthetic data (this validates it).
- Providing a formal privacy guarantee.
- Non-tabular modalities.

## Reporting
A generator that copies training data and is NOT flagged as memorisation is the
critical bug here — report to **gabejar@usa.com**.

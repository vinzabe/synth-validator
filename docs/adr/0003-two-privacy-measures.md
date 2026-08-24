# 3. Privacy needs a blunt measure and a subtle one

Date: 2026-08-24
Status: Accepted

## Context
Membership-inference metrics are the sophisticated way to measure synthetic-data
privacy — and they can be fooled by a generator that copies only *some* rows, where
aggregate statistics stay unremarkable. Conversely, duplicate-checking alone misses
a generator that leaks membership without exact copies.

## Decision
Report both:
1. **Duplicate rate** — fraction of synthetic rows within a tolerance of a training
   row. Dominates the privacy score, because copying is disqualifying.
2. **Membership advantage** — AUC-style comparison of distance-to-closest-record for
   members vs a holdout, re-centred so 0.0 means chance performance.

Median DCR for members and holdout are reported alongside, so the advantage number
is interpretable rather than opaque.

## Consequences
- The `copy` generator is caught by both measures (duplicate rate > 0.5, advantage
  > 0.1), and a noisy generator is caught by neither — both directions tested.
- A holdout set that was NOT used to fit the generator is required. That is a real
  constraint on the caller and is documented; without it, membership advantage is
  meaningless.

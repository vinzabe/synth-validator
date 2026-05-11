# Security Policy

## Reporting

Report vulnerabilities responsibly to the repository owner by email to **g@abejar.net** -- do not open public issues.

## Threat model

This is an *audit* tool, not a privacy-preserving generator. It is intended to help data publishers decide whether a synthetic-data release is safe.

- The validator reads CSV/Parquet/JSON files and runs scikit-learn models locally; it never executes user-supplied code from those files.
- The LLM advisor receives only the *aggregate report* (numbers, no records), but operators should still review the deployment summary before forwarding to a third-party LLM.
- DCR and MIA results are heuristics; passing this audit is necessary, not sufficient, for a safe release. Treat thresholds as starting points.
- Attribute-disclosure leakage is measured against a single quasi-identifier set; an attacker may know more. Audit with the most adversarial quasi-identifier set you can plausibly defend against.
- Utility numbers depend on the chosen downstream classifier (RandomForest by default). Real downstream performance can differ.

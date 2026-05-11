# synth-validator

Privacy + utility audit for tabular synthetic datasets. Runs the standard battery of attacks (DCR, membership inference, attribute disclosure, singling-out) plus utility evaluation, aggregates them into a single 0--100 risk score, and asks an LLM to recommend a publication posture (`publish` / `publish-with-controls` / `regenerate` / `block`).

## What it measures

| Metric | What it answers |
|--------|-----------------|
| **DCR** (Distance to Closest Record) | Are synthetic rows suspiciously close to any real row? |
| **Statistical similarity** (KS + correlation delta) | Is the distribution close enough to be useful? |
| **Singling-out** | Do real-data uniquely-identifiable combinations also appear uniquely in the synthetic data? |
| **Membership inference** | Can an attacker tell which real rows the generator was fit on? |
| **Attribute disclosure** | Can an attacker recover a sensitive attribute of a real record from quasi-identifiers + the synthetic data? |
| **Utility** | Train-on-synth vs train-on-real downstream classifier accuracy on a real test split |

The aggregate risk score weights these into a single number; the LLM advisor turns the report into actionable remediation steps and a suggested DP-epsilon.

## Layout

```
synthval/
  metrics.py        DCR, statistical_similarity, singling_out_risk
  membership.py     1-NN distance MIA with rank-based AUC
  attribute.py      RandomForest on (synth QIs -> sensitive) tested on real
  utility.py        train-on-synth vs train-on-real
  report.py         build_report() + risk scoring
  advisor.py        LLMPrivacyAdvisor
  cli.py            synthval {validate, advise}
data/               sample real / good-synth / memorised-synth / holdout CSVs
tests/test_synthval.py  29 unit + LLM_LIVE smoke
```

## Quick start

```bash
pip install -r requirements.txt

# Full audit on the bundled samples
python -m synthval.cli validate \
    --real data/real.csv --synth data/synth_memorised.csv \
    --holdout data/holdout.csv \
    --qi state,gender,education_years \
    --sensitive high_income --utility-target high_income

# Same plus LLM-driven advisor
python -m synthval.cli validate ... --advise

# Standalone advisor on a saved report
python -m synthval.cli advise --report report.json
```

## Demo discrimination

| Dataset             | Risk score | DCR median | MIA AUC | Attribute leakage |
|---------------------|------------|------------|---------|-------------------|
| synth_memorised.csv | ~88        | 0.04       | 0.92    | 0.17              |
| synth_good.csv      | ~41        | 0.11       | 0.51    | 0.10              |

The validator clearly separates an obviously memorised synth (real + small noise) from a properly resampled one.

## Testing

```bash
pytest tests/ -v
LLM_LIVE=1 pytest tests/ -v
```

## License

MIT

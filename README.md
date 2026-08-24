# synth-validator

**Validates synthetic data on three separate axes — and refuses to blend them, because a single "quality score" rewards the one failure that matters most.**

Fidelity, privacy, and utility are in genuine tension. Push fidelity to its maximum and you have copied the original rows: perfect resemblance, zero privacy. A blended quality number **ranks that data leak first**.

```
$ synthval compare
generator    fidelity   privacy   utility   memorised?
copy             0.95      0.00      0.99          YES
noisy            0.87      0.96      1.00           no
marginal         0.61      0.96      1.00           no
```

Read the top row: the `copy` generator has the **highest fidelity of all three** and the highest single-axis scores overall — and it is a straight dump of the training data. Any blended metric would crown it. This validator flags it as `MEMORISATION` and fails it.

## The three axes

**Fidelity** — per-column histogram intersection (marginals) plus correlation-matrix similarity (joint structure). Reporting both matters: the `marginal` generator scores 0.61 precisely because it preserves marginals while destroying correlations, which a single fidelity number would blur.

**Privacy** — two measures, because one isn't enough:
- *Duplicate rate* — the blunt failure. If synthetic rows are copies, privacy is zero regardless of any clever metric, so duplicates dominate the score.
- *Membership advantage* — the real attack. Compare distance-to-closest-record for training members vs a holdout; if members' nearest synthetic neighbour is systematically closer, membership leaks. Reported as advantage over chance, so `0.0` means "attacker no better than guessing".

**Utility** — train-on-synthetic, test-on-real (TSTR), reported as **retention against the real-data baseline**. "87% accurate" is meaningless without knowing real data gives 91%.

## The memorisation check

```python
memorisation_detected = fidelity >= 0.9 and (duplicates_present or privacy < 0.5)
```

High fidelity achieved *by copying* is not a pass — it's the specific failure this tool exists to catch. It's asserted directly by `test_copy_has_highest_fidelity_yet_fails`.

## An honest note on the reference generators

`marginal` keeps utility at 1.00 in this demo. That's real, not a bug: the label rule here is a linear function of the features, and independent per-column resampling preserves enough for a linear model to learn it. The **fidelity** axis correctly catches what was lost (correlations, 0.61). It's a good illustration that the three axes are genuinely independent — one can fail while the others pass, which is exactly why they aren't summed.

## Quickstart (60 seconds)

```bash
git clone https://github.com/vinzabe/synth-validator && cd synth-validator
python -m pip install -e ".[dev]"

synthval compare                                  # the trade-off table
synthval validate --generator copy                # exit 2: MEMORISATION
synthval validate --generator noisy --json        # exit 0
synthval validate --generator noisy --min-privacy 0.9
```

Exit codes: `0` all thresholds met, `2` one or more axes failed (including memorisation), `1` error — so a data pipeline can gate on it.

## Development

```bash
python -m pip install -e ".[dev]"
pytest --cov=synthval      # 22 tests, ~95% coverage
mypy --strict src/synthval # clean (3.12 target for numpy stubs)
ruff check src tests       # clean
```

## License

MIT © vinzabe

"""synthval — validate synthetic data on three axes that genuinely trade off.

Synthetic-data tools love a single "quality score". That number is dangerous,
because the three things you actually care about are in tension:

  * **Fidelity** — does it look like the real distribution?
  * **Privacy** — can an attacker tell whether a specific record was in the
    training data?
  * **Utility** — does a model trained on it work on real data?

Push fidelity to its maximum and you have copied the original rows: perfect
fidelity, zero privacy. A single blended score hides exactly that failure. So this
validator refuses to blend them, and specifically flags the degenerate case where
high fidelity is achieved by memorisation.
"""
__version__ = "1.0.0"

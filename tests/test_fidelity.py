import numpy as np
import pytest

from synthval import fidelity


def test_identical_data_is_perfect():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    r = fidelity.evaluate(X, X)
    assert r.marginal_similarity > 0.99
    assert r.correlation_similarity > 0.99


def test_shifted_data_loses_marginal_similarity():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 3))
    r = fidelity.evaluate(X, X + 10.0)
    assert r.marginal_similarity < 0.2


def test_independent_columns_lose_correlation_similarity():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, (400, 1))
    correlated = np.hstack([base, base + rng.normal(0, 0.1, (400, 1))])
    independent = rng.normal(0, 1, (400, 2))
    r = fidelity.evaluate(correlated, independent)
    assert r.correlation_similarity < 0.6


def test_column_mismatch_raises():
    with pytest.raises(ValueError, match="same columns"):
        fidelity.evaluate(np.zeros((10, 2)), np.zeros((10, 3)))


def test_score_bounded():
    rng = np.random.default_rng(3)
    r = fidelity.evaluate(rng.normal(0, 1, (100, 2)), rng.normal(5, 3, (100, 2)))
    assert 0.0 <= r.score <= 1.0

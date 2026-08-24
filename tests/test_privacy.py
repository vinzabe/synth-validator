"""Privacy must catch the blunt failure (copying) and the subtle one (membership)."""
from synthval import privacy
from synthval.generators import copy_generator, make_dataset, noisy_generator


def _split(n=200, seed=0):
    X, _ = make_dataset(n * 2, seed=seed)
    return X[:n], X[n:]


def test_copying_gives_high_duplicate_rate():
    train, hold = _split()
    synth = copy_generator(train, len(train), seed=0)
    r = privacy.evaluate(train, hold, synth)
    assert r.duplicate_rate > 0.5
    assert r.leaks_by_duplication
    assert r.score == 0.0


def test_noisy_generator_has_no_duplicates():
    train, hold = _split()
    synth = noisy_generator(train, len(train), seed=0)
    r = privacy.evaluate(train, hold, synth)
    assert r.duplicate_rate == 0.0
    assert not r.leaks_by_duplication


def test_copying_gives_membership_advantage():
    """Members sit exactly on synthetic points; non-members do not."""
    train, hold = _split()
    synth = copy_generator(train, len(train), seed=0)
    r = privacy.evaluate(train, hold, synth)
    assert r.membership_advantage > 0.1
    assert r.median_dcr_members < r.median_dcr_holdout


def test_noisy_generator_low_advantage():
    train, hold = _split()
    synth = noisy_generator(train, len(train), seed=0, noise=1.0)
    r = privacy.evaluate(train, hold, synth)
    assert r.membership_advantage < 0.2


def test_score_bounded():
    train, hold = _split()
    for gen in (copy_generator, noisy_generator):
        r = privacy.evaluate(train, hold, gen(train, len(train), seed=0))
        assert 0.0 <= r.score <= 1.0

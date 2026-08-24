"""The verdict logic: memorisation must FAIL even though fidelity is highest."""
from synthval.cli import _validate


def test_copy_generator_flagged_as_memorisation():
    v = _validate("copy", 200, 0)
    assert v.memorisation_detected
    ok, problems = v.verdict()
    assert not ok
    assert any("MEMORISATION" in p for p in problems)


def test_copy_has_highest_fidelity_yet_fails():
    """The exact failure a blended score would reward."""
    copy = _validate("copy", 200, 0)
    noisy = _validate("noisy", 200, 0)
    assert copy.fidelity.score >= noisy.fidelity.score   # looks "best"
    assert not copy.verdict()[0]                          # but fails
    assert noisy.verdict()[0]                             # while noisy passes


def test_noisy_generator_passes():
    ok, problems = _validate("noisy", 200, 0).verdict()
    assert ok, problems


def test_marginal_generator_flagged_on_fidelity_not_privacy():
    v = _validate("marginal", 200, 0)
    ok, problems = v.verdict(min_fidelity=0.7)
    assert not ok
    assert any("fidelity" in p for p in problems)
    assert not v.memorisation_detected      # it is private, just low fidelity


def test_axes_are_reported_separately():
    v = _validate("noisy", 200, 0)
    # three independent scores, no blended total anywhere in the API
    assert not hasattr(v, "overall_score")
    assert all(0.0 <= s <= 1.0 for s in
               (v.fidelity.score, v.privacy.score, v.utility.score))


def test_utility_retention_relative_to_baseline():
    v = _validate("noisy", 200, 0)
    assert v.utility.real_baseline > 0.5
    assert 0.0 <= v.utility.retention <= 1.0

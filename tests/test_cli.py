import json

import pytest

from synthval.cli import EXIT_FAILED, EXIT_OK, main


def test_copy_fails_with_memorisation(capsys):
    assert main(["validate", "--generator", "copy", "--n", "150"]) == EXIT_FAILED
    assert "MEMORISATION" in capsys.readouterr().out


def test_noisy_passes(capsys):
    assert main(["validate", "--generator", "noisy", "--n", "150"]) == EXIT_OK


def test_marginal_fails_fidelity(capsys):
    rc = main(["validate", "--generator", "marginal", "--n", "150"])
    assert rc == EXIT_FAILED
    assert "fidelity" in capsys.readouterr().out


def test_json_has_three_separate_axes(capsys):
    main(["validate", "--generator", "noisy", "--n", "150", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert set(d) >= {"fidelity", "privacy", "utility", "memorisation_detected"}
    assert "overall_score" not in d      # deliberately never blended


def test_compare_table(capsys):
    assert main(["compare", "--n", "150"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "copy" in out and "YES" in out


def test_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0

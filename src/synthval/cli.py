"""CLI: validate a reference generator on all three axes.

Exit codes: 0 all thresholds met, 2 one or more axes failed (including
memorisation), 1 error.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from . import fidelity as fid
from . import privacy as priv
from . import utility as util
from .generators import copy_generator, label_for, make_dataset, marginal_generator, noisy_generator
from .report import Validation

EXIT_OK, EXIT_ERROR, EXIT_FAILED = 0, 1, 2

_GENERATORS = {"copy": copy_generator, "noisy": noisy_generator,
               "marginal": marginal_generator}


def _validate(name: str, n: int, seed: int) -> Validation:
    X, y = make_dataset(n * 2, seed=seed)
    half = len(X) // 2
    train_X, train_y = X[:half], y[:half]
    hold_X, hold_y = X[half:], y[half:]
    synth_X = _GENERATORS[name](train_X, half, seed=seed)
    synth_y = label_for(synth_X)
    return Validation(
        fidelity=fid.evaluate(train_X, synth_X),
        privacy=priv.evaluate(train_X, hold_X, synth_X),
        utility=util.evaluate(train_X, train_y, synth_X, synth_y,
                              hold_X, hold_y))


def cmd_validate(a: argparse.Namespace) -> int:
    v = _validate(a.generator, a.n, a.seed)
    ok, problems = v.verdict(min_fidelity=a.min_fidelity,
                             min_privacy=a.min_privacy,
                             min_utility=a.min_utility)
    if a.json:
        print(json.dumps({
            "generator": a.generator,
            "fidelity": {"score": v.fidelity.score,
                         "marginal": v.fidelity.marginal_similarity,
                         "correlation": v.fidelity.correlation_similarity},
            "privacy": {"score": v.privacy.score,
                        "duplicate_rate": v.privacy.duplicate_rate,
                        "membership_advantage": v.privacy.membership_advantage},
            "utility": {"score": v.utility.score,
                        "tstr_accuracy": v.utility.tstr_accuracy,
                        "real_baseline": v.utility.real_baseline},
            "memorisation_detected": v.memorisation_detected,
            "passed": ok, "problems": list(problems)}, indent=2))
    else:
        print(f"generator '{a.generator}' — three separate axes, never blended:\n")
        print(f"  fidelity  {v.fidelity.score:.2f}  "
              f"(marginals {v.fidelity.marginal_similarity:.2f}, "
              f"correlations {v.fidelity.correlation_similarity:.2f})")
        print(f"  privacy   {v.privacy.score:.2f}  "
              f"(duplicates {v.privacy.duplicate_rate:.1%}, "
              f"membership advantage {v.privacy.membership_advantage:.3f})")
        print(f"  utility   {v.utility.score:.2f}  "
              f"(TSTR {v.utility.tstr_accuracy:.2f} vs real "
              f"{v.utility.real_baseline:.2f})")
        print(f"\n  verdict: {'PASS' if ok else 'FAIL'}")
        for p in problems:
            print(f"    ✗ {p}")
    return EXIT_OK if ok else EXIT_FAILED


def cmd_compare(a: argparse.Namespace) -> int:
    print(f"{'generator':<11}{'fidelity':>10}{'privacy':>10}{'utility':>10}"
          f"{'memorised?':>13}")
    for name in _GENERATORS:
        v = _validate(name, a.n, a.seed)
        print(f"{name:<11}{v.fidelity.score:>10.2f}{v.privacy.score:>10.2f}"
              f"{v.utility.score:>10.2f}"
              f"{'YES' if v.memorisation_detected else 'no':>13}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synthval", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate one generator")
    v.add_argument("--generator", choices=list(_GENERATORS), default="noisy")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--min-fidelity", type=float, default=0.7)
    v.add_argument("--min-privacy", type=float, default=0.7)
    v.add_argument("--min-utility", type=float, default=0.8)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("compare", help="compare all reference generators")
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.func(args)
        return rc
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

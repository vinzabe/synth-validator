"""CLI for synth-validator."""
from __future__ import annotations
import argparse
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from synthval.report import build_report
from synthval.advisor import LLMPrivacyAdvisor


def _load(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise SystemExit(f"unsupported input format: {path}")


def cmd_validate(args):
    real = _load(args.real)
    synth = _load(args.synth)
    holdout = _load(args.holdout) if args.holdout else None
    qi = args.qi.split(",") if args.qi else None
    rep = build_report(
        real, synth, real_holdout=holdout,
        quasi_identifiers=qi,
        sensitive_attribute=args.sensitive,
        utility_target=args.utility_target,
        dcr_threshold=args.dcr_threshold,
    )
    out = rep.to_dict()
    if args.advise:
        from llm_client import LLMClient
        adv = LLMPrivacyAdvisor(LLMClient(), model=args.model)
        rec = adv.advise(out)
        out["advisor"] = rec.to_dict()
    print(json.dumps(out, indent=2))
    return 0


def cmd_advise(args):
    with open(args.report) as f:
        rep = json.load(f)
    from llm_client import LLMClient
    adv = LLMPrivacyAdvisor(LLMClient(), model=args.model)
    rec = adv.advise(rep)
    print(json.dumps(rec.to_dict(), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="synthval")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Run all metrics + optional LLM advisor")
    v.add_argument("--real", required=True)
    v.add_argument("--synth", required=True)
    v.add_argument("--holdout", default=None)
    v.add_argument("--qi", default=None,
                    help="comma-separated quasi-identifier columns")
    v.add_argument("--sensitive", default=None,
                    help="sensitive attribute column for disclosure attack")
    v.add_argument("--utility-target", default=None,
                    help="downstream classification target column")
    v.add_argument("--dcr-threshold", type=float, default=0.05)
    v.add_argument("--advise", action="store_true",
                    help="invoke the LLM advisor on the resulting report")
    v.add_argument("--model", default="glm-5.1")
    v.set_defaults(func=cmd_validate)

    a = sub.add_parser("advise", help="LLM advisor on an existing report JSON")
    a.add_argument("--report", required=True)
    a.add_argument("--model", default="glm-5.1")
    a.set_defaults(func=cmd_advise)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

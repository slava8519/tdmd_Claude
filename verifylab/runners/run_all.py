#!/usr/bin/env python3
"""
VerifyLab runner — runs every case under verifylab/cases/ and reports pass/fail.

Usage:
  python3 verifylab/runners/run_all.py                  # run all enabled cases
  python3 verifylab/runners/run_all.py --case <name>    # run one case
  python3 verifylab/runners/run_all.py --suite fast     # only fast cases
  python3 verifylab/runners/run_all.py --suite slow     # only slow cases

A case is "fast" if its tolerance.toml has [case] slow = false (default).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

from result_schema import parse_result_line

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = REPO_ROOT / "verifylab" / "cases"


def discover_cases() -> list[Path]:
    cases = []
    for p in sorted(CASES_DIR.iterdir()):
        if p.is_dir() and (p / "check.py").exists():
            cases.append(p)
    return cases


def case_is_slow(case_dir: Path) -> bool:
    tol_file = case_dir / "tolerance.toml"
    if not tol_file.exists():
        return False
    with tol_file.open("rb") as f:
        tol = tomllib.load(f)
    return bool(tol.get("case", {}).get("slow", False))


def run_case(case_dir: Path) -> tuple[bool, str, dict | None]:
    """Run one case's check.py.

    Returns (ok, combined_output, result_record). The result record is
    scraped from any `VL_RESULT: {...}` line the check.py printed; None
    if the check predates the VL-14 migration or crashed before emitting.
    """
    check = case_dir / "check.py"
    try:
        result = subprocess.run(
            ["python3", str(check)],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        ok = result.returncode == 0
        out = result.stdout + ("\n" + result.stderr if result.stderr else "")
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 600s", None
    except Exception as e:
        return False, f"ERROR: {e}", None

    # Scrape the most recent VL_RESULT record from stdout.
    record = None
    for line in out.splitlines():
        parsed = parse_result_line(line.strip())
        if parsed is not None:
            record = parsed
    return ok, out, record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Run a single case by name")
    ap.add_argument("--suite", choices=["fast", "slow", "all"], default="all")
    ap.add_argument(
        "--jsonl",
        help="Write one VL_RESULT record per case to this JSONL file "
             "(overwrites existing). Lets downstream tooling consume a "
             "machine-readable run summary without stdout scraping.",
    )
    args = ap.parse_args()

    cases = discover_cases()
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"No case named {args.case!r} found under {CASES_DIR}")
            return 2

    if args.suite == "fast":
        cases = [c for c in cases if not case_is_slow(c)]
    elif args.suite == "slow":
        cases = [c for c in cases if case_is_slow(c)]

    if not cases:
        print("No cases to run.")
        return 0

    print(f"Running {len(cases)} VerifyLab case(s)...\n")

    results = []
    records: list[dict] = []
    for c in cases:
        print(f"-- {c.name} --")
        ok, out, record = run_case(c)
        for line in out.splitlines():
            print(f"   {line}")
        results.append((c.name, ok))
        if record is not None:
            records.append(record)
        print()

    print("=" * 60)
    print(f"{'Case':<40} {'Result':>10}")
    print("-" * 60)
    n_pass = 0
    for name, ok in results:
        marker = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        print(f"{name:<40} {marker:>10}")
    print("-" * 60)
    print(f"Total: {len(results)}   Passed: {n_pass}   Failed: {len(results) - n_pass}")

    if args.jsonl:
        import json
        with open(args.jsonl, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(f"\nWrote {len(records)} result record(s) to {args.jsonl}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

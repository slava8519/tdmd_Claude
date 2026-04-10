"""
VerifyLab result schema v1.

Every case's `check.py` emits exactly one machine-readable result record, in
addition to its human-readable stdout. The record is a single-line JSON blob
prefixed by a stable sentinel `VL_RESULT:` so the runner can scrape it out of
mixed output without needing to redirect streams.

Emission contract
-----------------
At the end of `main()`, before returning its exit code, a check.py does::

    from result_schema import emit_result
    emit_result(
        case="two-atoms-morse",
        mode="mixed",
        status="pass",          # "pass" | "fail" | "error"
        metrics={...},          # dict[str, float] — observed values
        thresholds={...},       # dict[str, float] — matching threshold for each metric key
        failures=[...],         # list[str] — empty when status == "pass"
        duration_s=2.31,
    )

Status semantics
----------------
- "pass"  : all metrics within thresholds, no setup errors
- "fail"  : metrics exceed thresholds (a physics/correctness regression)
- "error" : setup failed, binary missing, TDMD crashed — i.e. the case
            could not be graded at all. Distinguishing "fail" from "error"
            matters for CI: a "fail" is a real regression, an "error" is
            usually infrastructure and should not be weighed the same.

Why this schema
---------------
- One record per run. The runner can tail multiple records into a JSONL
  stream by collecting one per case.
- Metrics and thresholds use parallel keys so "did metric X exceed its
  threshold?" is answerable by a dict lookup, not by re-parsing messages.
- No nested structures beyond these two dicts — keeps downstream diffs
  and dashboard queries trivial.
- Schema version is explicit so future changes (e.g. adding per-atom
  worst-offender info) can be additive without breaking consumers.

The `VL_RESULT:` prefix is a stable sentinel. Do not change it without
updating verifylab/runners/run_all.py in the same commit.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = 1
RESULT_SENTINEL = "VL_RESULT:"


def emit_result(
    *,
    case: str,
    mode: str,
    status: str,
    metrics: dict,
    thresholds: dict,
    failures: list,
    duration_s: float,
) -> None:
    """Emit a single VerifyLab result record.

    Prints one line `VL_RESULT: {json}` to stdout. If the environment
    variable `TDMD_VERIFYLAB_JSONL` is set, also appends the same line
    (without the sentinel prefix) to that file — this lets a caller
    collect records across multiple cases without stdout scraping.
    """
    if status not in ("pass", "fail", "error"):
        raise ValueError(
            f"invalid status {status!r}, expected pass/fail/error"
        )
    def sanitize(v):
        f = float(v)
        # NaN/inf are not valid JSON; encode as null so downstream parsers
        # (jq, GitHub Actions summary markdown, dashboards) do not choke.
        # The usual producer of NaN in metrics is an "errored" case that
        # could not measure anything — the status field already carries
        # that signal.
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    record = {
        "schema_version": SCHEMA_VERSION,
        "case": case,
        "mode": mode,
        "status": status,
        "metrics": {k: sanitize(v) for k, v in metrics.items()},
        "thresholds": {k: sanitize(v) for k, v in thresholds.items()},
        "failures": list(failures),
        "duration_s": float(duration_s),
        "timestamp": datetime.now(timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    print(f"{RESULT_SENTINEL} {line}", flush=True)

    jsonl_path = os.environ.get("TDMD_VERIFYLAB_JSONL")
    if jsonl_path:
        with open(jsonl_path, "a") as f:
            f.write(line + "\n")


def parse_result_line(line: str) -> dict | None:
    """Parse a `VL_RESULT: {...}` line back into a dict.

    Returns None if the line is not a result record. Used by the runner.
    """
    if not line.startswith(RESULT_SENTINEL):
        return None
    try:
        return json.loads(line[len(RESULT_SENTINEL):].strip())
    except json.JSONDecodeError:
        return None

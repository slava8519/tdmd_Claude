"""
VL-16 — failure-reporting negative-path validator (research doc D1).

Meta-test for verifylab/runners/result_schema.py. Guards the emission layer
against the class of bug VL-14 was written to fix: a physically-failing case
that silently reports PASS because the failure path itself is broken (NaN
metrics → JSON serialization crash swallowed → runner never sees status=fail).

Run manually::
    python3 -m unittest verifylab.runners.test_result_schema

Run from the runners dir (how run_all.py imports its sibling module)::
    cd verifylab/runners && python3 -m unittest test_result_schema

The tests cover:
  1. A plain "fail" record round-trips through parse_result_line with the
     failure list intact.
  2. NaN/inf metrics are sanitized to JSON null (the VL-14 regression case)
     and the record still parses.
  3. An invalid status value is rejected eagerly — a mis-spelled "failed" or
     "FAIL" must not silently emit a record the runner would accept as pass.
  4. The sentinel prefix is stable: parse_result_line rejects any line that
     lacks `VL_RESULT: `, so a stray print from a check.py cannot be mistaken
     for a result.
  5. TDMD_VERIFYLAB_JSONL side-channel writes match the stdout record byte
     for byte (minus the sentinel prefix).
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# Allow `python3 -m unittest verifylab/runners/test_result_schema.py`
# from the repo root by putting the runners dir on sys.path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from result_schema import (  # noqa: E402
    RESULT_SENTINEL,
    SCHEMA_VERSION,
    emit_result,
    parse_result_line,
)


def _emit_capture(**kwargs) -> dict:
    """Call emit_result, capture its stdout line, parse it back to a dict."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        emit_result(**kwargs)
    line = buf.getvalue().strip()
    record = parse_result_line(line)
    assert record is not None, f"line did not parse: {line!r}"
    return record


class FailureRecordTests(unittest.TestCase):
    """A failing case must be visibly failing in the emitted record."""

    def test_plain_fail_roundtrips_with_failures_list(self) -> None:
        rec = _emit_capture(
            case="synthetic-fail",
            mode="mixed",
            status="fail",
            metrics={"drift": 1.5e-3},
            thresholds={"drift": 1.0e-4},
            failures=["drift 1.5e-03 exceeds threshold 1.0e-04"],
            duration_s=0.42,
        )
        self.assertEqual(rec["schema_version"], SCHEMA_VERSION)
        self.assertEqual(rec["status"], "fail")
        self.assertEqual(len(rec["failures"]), 1)
        self.assertIn("drift", rec["failures"][0])
        # Metrics and thresholds both survive round-trip.
        self.assertAlmostEqual(rec["metrics"]["drift"], 1.5e-3)
        self.assertAlmostEqual(rec["thresholds"]["drift"], 1.0e-4)

    def test_nan_metric_sanitized_without_crash(self) -> None:
        # The VL-14 regression: before the sanitizer was added, a NaN metric
        # caused json.dumps(..., allow_nan=False) to raise, which could be
        # swallowed and let the runner count the case as pass.
        rec = _emit_capture(
            case="synthetic-nan",
            mode="mixed",
            status="error",
            metrics={"T_mean": float("nan"), "T_std": float("inf")},
            thresholds={"T_mean": 300.0, "T_std": 10.0},
            failures=["T_mean is NaN — integrator blew up"],
            duration_s=0.01,
        )
        self.assertEqual(rec["status"], "error")
        self.assertIsNone(rec["metrics"]["T_mean"])
        self.assertIsNone(rec["metrics"]["T_std"])
        self.assertEqual(rec["thresholds"]["T_mean"], 300.0)
        self.assertIn("NaN", rec["failures"][0])

    def test_invalid_status_raises(self) -> None:
        # Typos must not produce a silently-passing record. A misspelled
        # "failed" or "FAIL" has to blow up at emit time, not leak through
        # to the runner as an unknown status.
        for bad in ("failed", "FAIL", "passed", "", "ok"):
            with self.subTest(status=bad):
                with self.assertRaises(ValueError):
                    emit_result(
                        case="x",
                        mode="mixed",
                        status=bad,
                        metrics={},
                        thresholds={},
                        failures=[],
                        duration_s=0.0,
                    )


class SentinelTests(unittest.TestCase):
    """The VL_RESULT: prefix is the contract between check.py and run_all.py."""

    def test_sentinel_prefix_required(self) -> None:
        self.assertIsNone(parse_result_line("hello world"))
        self.assertIsNone(parse_result_line('{"status": "pass"}'))
        # Even a near-miss (missing colon, or extra space) is rejected.
        self.assertIsNone(parse_result_line("VL_RESULT {}"))

    def test_malformed_json_after_sentinel_returns_none(self) -> None:
        self.assertIsNone(parse_result_line(f"{RESULT_SENTINEL} not json"))
        # A half-written line must not be confused with a valid result.
        self.assertIsNone(parse_result_line(f"{RESULT_SENTINEL} {{"))


class JsonlSideChannelTests(unittest.TestCase):
    """TDMD_VERIFYLAB_JSONL must mirror the stdout record exactly."""

    def test_jsonl_matches_stdout_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jsonl_path = os.path.join(td, "out.jsonl")
            old = os.environ.get("TDMD_VERIFYLAB_JSONL")
            os.environ["TDMD_VERIFYLAB_JSONL"] = jsonl_path
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    emit_result(
                        case="sidechannel",
                        mode="fp64",
                        status="fail",
                        metrics={"x": 1.0},
                        thresholds={"x": 0.5},
                        failures=["x > threshold"],
                        duration_s=0.05,
                    )
                stdout_line = buf.getvalue().strip()
                self.assertTrue(stdout_line.startswith(RESULT_SENTINEL))
                stdout_payload = stdout_line[len(RESULT_SENTINEL):].strip()

                with open(jsonl_path) as f:
                    jsonl_payload = f.read().strip()

                # Byte-identical so downstream diffs never have to worry about
                # which channel they're reading from.
                self.assertEqual(stdout_payload, jsonl_payload)

                rec = json.loads(jsonl_payload)
                self.assertEqual(rec["status"], "fail")
                self.assertEqual(rec["failures"], ["x > threshold"])
            finally:
                if old is None:
                    del os.environ["TDMD_VERIFYLAB_JSONL"]
                else:
                    os.environ["TDMD_VERIFYLAB_JSONL"] = old


if __name__ == "__main__":
    unittest.main()

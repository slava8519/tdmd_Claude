#!/usr/bin/env python3
"""
Helper for VerifyLab cases that need to compare TDMD output against a LAMMPS log.

Provides functions for parsing LAMMPS thermo logs and TDMD logs, computing
relative/absolute errors on common columns (Step, Temp, PotEng, KinEng, TotEng,
Press), and asserting tolerances.

This is a thin library — case `check.py` files import from here.

M1+ — currently a stub.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_lammps_thermo(log_path: Path) -> dict[str, list[float]]:
    """Parse a LAMMPS log file and return columns as lists keyed by name."""
    raise NotImplementedError("M1 TODO")


def parse_tdmd_thermo(log_path: Path) -> dict[str, list[float]]:
    """Parse a TDMD log file. Format mirrors LAMMPS thermo lines."""
    raise NotImplementedError("M1 TODO")


def compare_columns(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    columns: list[str],
    tolerance: float,
    mode: str = "max_relative",
) -> tuple[bool, dict[str, float]]:
    """Compare matching columns. Return (all_pass, per_column_max_error)."""
    raise NotImplementedError("M1 TODO")

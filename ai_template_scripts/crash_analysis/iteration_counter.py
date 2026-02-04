# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Iteration counting functions for crash analysis.

This module counts AI role commits ([W], [M], [P], [R]) from git history
to calculate total iterations for failure rate calculations.
"""

from __future__ import annotations

import subprocess
from datetime import datetime


def count_iterations_since(since: datetime) -> int:
    """Count AI role commits since a given time.

    Args:
        since: Start time for counting

    Returns:
        Number of [W], [M], [P], [R] commits since that time

    Contracts:
        REQUIRES: since is a valid datetime
        ENSURES: result >= 0
        ENSURES: result == 0 if git command fails (fail-safe)
    """
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Count all role commits: [W], [M], [P], [R]
        pattern = r"\[[^]]*-?(W|M|P|R)[0-9]*\]"
        result = subprocess.run(
            [
                "git",
                "log",
                f"--since={since_str}",
                "--oneline",
                "--extended-regexp",
                f"--grep={pattern}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        # Count non-empty lines
        return len([line for line in result.stdout.strip().split("\n") if line])
    except Exception:
        return 0  # Best-effort: git log for iteration count, 0 is safe default


def count_iterations_between(start: datetime, end: datetime) -> int:
    """Count AI role commits between two timestamps (half-open interval [start, end)).

    Args:
        start: Start time (inclusive)
        end: End time (exclusive)

    Returns:
        Number of [W], [M], [P], [R] commits in the interval

    Contracts:
        REQUIRES: start <= end
        ENSURES: result >= 0
        ENSURES: result == 0 if git command fails (fail-safe)
    """
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")
    try:
        pattern = r"\[[^]]*-?(W|M|P|R)[0-9]*\]"
        result = subprocess.run(
            [
                "git",
                "log",
                f"--since={start_str}",
                f"--until={end_str}",
                "--oneline",
                "--extended-regexp",
                f"--grep={pattern}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        return len([line for line in result.stdout.strip().split("\n") if line])
    except Exception:
        return 0


__all__ = [
    "count_iterations_since",
    "count_iterations_between",
]

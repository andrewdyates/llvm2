# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Crash log parser for crash_analysis package.

PURPOSE: Parses failures.log (fallback crashes.log) into CrashEntry objects.
CALLED BY: crash_analysis.__init__ (get_health_report)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ai_template_scripts.crash_analysis.types import CrashEntry

# Import log paths at runtime to support test patching via crash_analysis.FAILURES_LOG
# and crash_analysis.CRASHES_LOG.
# (tests patch crash_analysis namespace, not types namespace)


def _get_failures_log() -> Path:
    """Get failure log path, supporting runtime patching for tests."""
    import ai_template_scripts.crash_analysis as crash_analysis
    from ai_template_scripts.crash_analysis import types as crash_types

    failures_log = crash_analysis.FAILURES_LOG
    crashes_log = crash_analysis.CRASHES_LOG
    if crashes_log != crash_types.CRASHES_LOG and crashes_log.exists():
        return crashes_log
    if failures_log.exists():
        return failures_log
    if crashes_log.exists():
        return crashes_log
    return failures_log


def parse_crashes_log(since: datetime | None = None) -> list[CrashEntry]:
    """Parse failures.log file (fallback crashes.log).

    Args:
        since: Only include crashes after this timestamp

    Returns:
        List of CrashEntry objects, newest first

    Contracts:
        REQUIRES: since is None or since <= datetime.now()
        ENSURES: result is sorted by timestamp descending (newest first)
        ENSURES: all(e.timestamp >= since for e in result) if since is not None
    """
    crashes_log = _get_failures_log()
    if not crashes_log.exists():
        return []

    entries = []
    # Format: [2026-01-08 15:30:45] Iteration 5: claude exited with code 1
    pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Iteration (\d+): (.+)",
    )

    with open(crashes_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                timestamp_str, iteration_str, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                # Assume local timezone
                timestamp = timestamp.replace(tzinfo=None)

                if since and timestamp < since.replace(tzinfo=None):
                    continue

                entries.append(
                    CrashEntry(
                        timestamp=timestamp,
                        iteration=int(iteration_str),
                        message=message,
                    ),
                )

    # Sort newest first
    entries.sort(key=lambda e: e.timestamp, reverse=True)
    return entries

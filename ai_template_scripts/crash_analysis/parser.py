# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Crash log parser for crash_analysis package.

PURPOSE: Parses crashes.log file into structured CrashEntry objects.
CALLED BY: crash_analysis.__init__ (get_health_report)
"""

from __future__ import annotations

import re
from datetime import datetime

from ai_template_scripts.crash_analysis.types import CrashEntry

# Import CRASHES_LOG at runtime to support test patching via crash_analysis.CRASHES_LOG
# (tests patch crash_analysis namespace, not types namespace)


def _get_crashes_log():
    """Get CRASHES_LOG path, supporting runtime patching for tests."""
    import ai_template_scripts.crash_analysis as crash_analysis

    return crash_analysis.CRASHES_LOG


def parse_crashes_log(since: datetime | None = None) -> list[CrashEntry]:
    """Parse crashes.log file.

    Args:
        since: Only include crashes after this timestamp

    Returns:
        List of CrashEntry objects, newest first

    Contracts:
        REQUIRES: since is None or since <= datetime.now()
        ENSURES: result is sorted by timestamp descending (newest first)
        ENSURES: all(e.timestamp >= since for e in result) if since is not None
    """
    crashes_log = _get_crashes_log()
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

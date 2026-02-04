# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/uncommitted_warning.py - Uncommitted changes detection.

Functions for detecting and warning about uncommitted changes at session start.
Part of #1104 - mid-session commit warning for large uncommitted changes.
"""

__all__ = [
    "UNCOMMITTED_THRESHOLD",
    "get_uncommitted_changes_size",
    "get_uncommitted_changes_warning",
]

import re

from looper.config import load_timeout_config
from looper.log import debug_swallow
from looper.subprocess_utils import run_git_command

# Threshold for uncommitted change warning (lines)
UNCOMMITTED_THRESHOLD = 100


def get_uncommitted_changes_size() -> int:
    """Get total lines of uncommitted changes (staged + unstaged).

    Contracts:
        ENSURES: Returns 0 if no changes or error
        ENSURES: Returns positive integer = total lines changed
        ENSURES: Never raises - catches all exceptions

    Returns:
        Number of lines changed (insertions + deletions).
    """
    try:
        timeout_sec = load_timeout_config().get("git_default", 5)
        result = run_git_command(
            ["diff", "--shortstat", "--no-color", "--no-ext-diff", "HEAD"],
            timeout=timeout_sec,
        )
        if not result.ok or not result.value:
            return 0

        # Parse git diff --shortstat output
        # Summary line looks like: "5 files changed, 100 insertions(+), 50 deletions(-)"
        lines = result.value.strip().split("\n")
        if not lines:
            return 0

        # Get last non-empty line (summary line)
        summary = ""
        for line in reversed(lines):
            if line.strip():
                summary = line.strip()
                break

        if not summary:
            return 0

        # Parse insertions and deletions
        total = 0

        # Match patterns like "100 insertions(+)" or "50 deletions(-)"
        insertions_match = re.search(r"(\d+)\s+insertion", summary)
        deletions_match = re.search(r"(\d+)\s+deletion", summary)

        if insertions_match:
            total += int(insertions_match.group(1))
        if deletions_match:
            total += int(deletions_match.group(1))

        return total
    except Exception as e:
        debug_swallow("get_uncommitted_changes_size", e)
        return 0


def get_uncommitted_changes_warning(threshold: int = UNCOMMITTED_THRESHOLD) -> str:
    """Get warning message if uncommitted changes exceed threshold.

    Contracts:
        REQUIRES: threshold > 0 (clamped to UNCOMMITTED_THRESHOLD if invalid)
        ENSURES: Returns empty string if changes < threshold
        ENSURES: Returns warning message if changes >= threshold
        ENSURES: Never raises - catches all exceptions

    Args:
        threshold: Lines of change to trigger warning (default 100)

    Returns:
        Warning message or empty string.
    """
    try:
        # Validate and clamp threshold
        if not isinstance(threshold, int) or threshold <= 0:
            threshold = UNCOMMITTED_THRESHOLD

        uncommitted = get_uncommitted_changes_size()
        if uncommitted >= threshold:
            return (
                f"⚠️ UNCOMMITTED: {uncommitted} lines of uncommitted changes. "
                "Consider committing before continuing."
            )
        return ""
    except Exception as e:
        debug_swallow("get_uncommitted_changes_warning", e)
        return ""

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Crash detection and categorization functions.

This module contains functions for:
- Detecting and filtering idle aborts
- Categorizing crash types
- Fingerprinting crashes for deduplication
- Grouping crashes by fingerprint
- Analyzing crash patterns
- Applying inhibition rules

Categories imported from ai_template_scripts.crash_categories per #2318.
"""

from __future__ import annotations

import hashlib
import re

from ai_template_scripts.crash_analysis.types import (
    CrashEntry,
    CrashFingerprint,
    SuppressedPattern,
)
from ai_template_scripts.crash_categories import (
    CrashCategoryStr,
    CRASH_CATEGORIES,
    is_expected_termination,
)

# Idle abort detection
_IDLE_ABORT_REASON = "no issues assigned"

# Inhibition rules following Prometheus Alertmanager pattern
# When a root cause pattern is present, suppress downstream cascading patterns
# Format: (inhibiting_category, categories_to_suppress, reason)
_INHIBITION_RULES: list[tuple[str, list[str], str]] = [
    # OOM kills (signal_kill) often cause timeouts and stale connections as downstream effects
    ("signal_kill", ["timeout", "stale_connection"], "cascading from signal_kill"),
]


def _is_idle_abort(message: str) -> bool:
    """Return True when crash log message reflects expected idle exit."""
    lowered = message.lower()
    return _IDLE_ABORT_REASON in lowered and "abort" in lowered


def _filter_idle_aborts(crashes: list[CrashEntry]) -> tuple[list[CrashEntry], int]:
    """Remove idle abort entries and return (filtered, idle_count)."""
    filtered: list[CrashEntry] = []
    idle_count = 0
    for crash in crashes:
        if _is_idle_abort(crash.message):
            idle_count += 1
        else:
            filtered.append(crash)
    return filtered, idle_count


def _get_crash_category(message: str) -> CrashCategoryStr:
    """Determine crash category from message.

    Args:
        message: Crash message text

    Returns:
        Category string from CRASH_CATEGORIES per #2318

    Contracts:
        REQUIRES: message is a string
        ENSURES: result in CRASH_CATEGORIES
    """
    msg = message.lower()
    if _is_idle_abort(message):
        return "idle_abort"
    if "signal" in msg:
        return "signal_kill"
    if "timed out" in msg:
        return "timeout"
    if "killed due to silence" in msg or "stale connection" in msg:
        return "stale_connection"
    if "exited with code" in msg:
        return "exit_error"
    return "unknown"


def analyze_crash_patterns(crashes: list[CrashEntry]) -> dict[str, int]:
    """Categorize crashes by type.

    Args:
        crashes: List of crash entries

    Returns:
        Dict mapping error type to count

    Categories:
        - idle_abort: Expected early abort when no issues assigned
        - signal_kill: Killed by OS signal (SIGKILL, SIGTERM, etc.)
        - timeout: Hit iteration timeout limit
        - stale_connection: Killed due to silence (no output after sleep/resume)
        - exit_error: Non-zero exit code
        - unknown: Unrecognized crash type

    Contracts:
        REQUIRES: crashes is a list (may be empty)
        ENSURES: sum(result.values()) == len(crashes)
        ENSURES: all keys in {"idle_abort", "signal_kill", "timeout",
            "stale_connection", "exit_error", "unknown"}
        ENSURES: all(v > 0 for v in result.values())
    """
    patterns: dict[str, int] = {}

    for crash in crashes:
        category = _get_crash_category(crash.message)
        patterns[category] = patterns.get(category, 0) + 1

    return patterns


def apply_inhibition(
    patterns: dict[str, int],
) -> tuple[dict[str, int], list[SuppressedPattern]]:
    """Apply inhibition rules to suppress cascading failure patterns.

    Following Prometheus Alertmanager pattern, when a root cause pattern is
    present (e.g., signal_kill from OOM), suppress related downstream patterns
    (e.g., timeout, stale_connection) to reduce alert fatigue.

    Args:
        patterns: Dict mapping error category to count

    Returns:
        Tuple of (active_patterns, suppressed_patterns) where:
        - active_patterns: Patterns not suppressed (root causes + unrelated)
        - suppressed_patterns: Patterns suppressed with inhibition metadata

    Contracts:
        REQUIRES: patterns is a dict (may be empty)
        ENSURES: len(active) + len(suppressed) == len(patterns) (categories preserved)
        ENSURES: sum(active.values()) + sum(s.count for s in suppressed) ==
                 sum(patterns.values()) (total count preserved)
        ENSURES: all suppressed patterns have inhibited_by in active_patterns
    """
    if not patterns:
        return {}, []

    # Copy to avoid mutating input
    active = dict(patterns)
    suppressed: list[SuppressedPattern] = []

    for inhibitor, to_suppress, reason in _INHIBITION_RULES:
        # Only apply rule if inhibiting pattern is present
        if inhibitor not in active or active[inhibitor] == 0:
            continue

        for category in to_suppress:
            if category in active and active[category] > 0:
                suppressed.append(
                    SuppressedPattern(
                        category=category,
                        count=active[category],
                        reason=reason,
                        inhibited_by=inhibitor,
                    )
                )
                del active[category]

    return active, suppressed


def crash_fingerprint(crash: CrashEntry) -> str:
    """Generate deduplication key from crash category + normalized message.

    Following Sentry's grouphash approach, this creates a fingerprint that
    groups similar crashes together even if they differ in variable parts
    like iteration numbers, timestamps, or process IDs.

    Args:
        crash: Crash entry to fingerprint

    Returns:
        Fingerprint string in format "{category}:{8-char-hash}"

    Contracts:
        REQUIRES: crash is a valid CrashEntry
        ENSURES: len(result.split(':')) == 2
        ENSURES: result.split(':')[0] in {"idle_abort", "signal_kill", "timeout",
            "stale_connection", "exit_error", "unknown"}
        ENSURES: len(result.split(':')[1]) == 8
    """
    category = _get_crash_category(crash.message)
    # Normalize variable parts: iteration numbers, timestamps, PIDs
    normalized = re.sub(r"\d+", "N", crash.message)
    # Create short hash for grouping
    msg_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
    return f"{category}:{msg_hash}"


def group_crashes_by_fingerprint(
    crashes: list[CrashEntry],
) -> list[CrashFingerprint]:
    """Group crashes by fingerprint for deduplication.

    Creates CrashFingerprint objects that aggregate similar crashes,
    reducing noise in reports when the same root cause triggers multiple
    crash entries.

    Args:
        crashes: List of crash entries to group

    Returns:
        List of CrashFingerprint objects, sorted by count descending

    Contracts:
        REQUIRES: crashes is a list (may be empty)
        ENSURES: sum(fp.count for fp in result) == len(crashes)
        ENSURES: result is sorted by count descending
        ENSURES: all fingerprints are unique in result
    """
    if not crashes:
        return []

    # Group crashes by fingerprint
    groups: dict[str, list[CrashEntry]] = {}
    for crash in crashes:
        fp = crash_fingerprint(crash)
        if fp not in groups:
            groups[fp] = []
        groups[fp].append(crash)

    # Convert to CrashFingerprint objects
    fingerprints = []
    for fp, group in groups.items():
        category = _get_crash_category(group[0].message)
        # Sort by timestamp to get first/last seen
        sorted_group = sorted(group, key=lambda c: c.timestamp)
        fingerprints.append(
            CrashFingerprint(
                fingerprint=fp,
                category=category,
                count=len(group),
                example_message=group[0].message,
                first_seen=sorted_group[0].timestamp,
                last_seen=sorted_group[-1].timestamp,
            )
        )

    # Sort by count descending
    fingerprints.sort(key=lambda f: -f.count)
    return fingerprints


__all__ = [
    "_is_idle_abort",
    "_filter_idle_aborts",
    "_get_crash_category",
    "analyze_crash_patterns",
    "apply_inhibition",
    "crash_fingerprint",
    "group_crashes_by_fingerprint",
]

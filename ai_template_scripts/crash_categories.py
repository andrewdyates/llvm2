# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shared crash/timeout category definitions.

This module provides canonical category definitions used across:
- ai_template_scripts/timeout_classifier.py (TimeoutClassifier)
- ai_template_scripts/crash_analysis/crash_detector.py (Crash Detector)
- ai_template_scripts/pulse/process_metrics.py (Pulse metrics)

Category unification per #2318:
- idle_session renamed to idle_abort (consistency with crash_detector)
- stale_connection added to timeout categories
- All modules now import from this shared module

Usage:
    from ai_template_scripts.crash_categories import (
        TerminationCategory,
        TIMEOUT_CATEGORIES,
        CRASH_CATEGORIES,
        is_expected_termination,
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class TerminationCategory(Enum):
    """Canonical termination categories for crash/timeout classification.

    Categories are grouped by whether they represent expected vs problem states.
    """

    # Expected terminations (not crashes)
    GRACEFUL_STOP = "graceful_stop"  # User-requested stop
    IDLE_ABORT = "idle_abort"  # No issues to work on (exit 126)
    STALE_CONNECTION = "stale_connection"  # Machine slept, connection stale (exit 125)

    # Timeout terminations (may be expected)
    LONG_COMMAND = "long_command"  # Build/test exceeded time
    STUCK_QUERY = "stuck_query"  # SMT/verification stuck
    ITERATION_TIMEOUT = "timeout"  # Hit max iteration time (exit 124)
    NETWORK_STALL = "network_stall"  # API timeout

    # Problem terminations (crashes)
    SIGNAL_KILL = "signal_kill"  # Killed by signal (OOM, etc.)
    EXIT_ERROR = "exit_error"  # Non-zero exit code
    UNKNOWN = "unknown"  # Unclassifiable


# Type aliases for category strings (backwards compatibility)
TimeoutCategoryStr = Literal[
    "long_command",
    "stuck_query",
    "network_stall",
    "idle_abort",  # Renamed from idle_session
    "stale_connection",  # Added per #2318
    "exit_error",
    "signal_kill",
    "unknown",
]

CrashCategoryStr = Literal[
    "idle_abort",
    "signal_kill",
    "timeout",
    "stale_connection",
    "exit_error",
    "unknown",
]


# Category sets for validation
TIMEOUT_CATEGORIES: frozenset[str] = frozenset(
    [
        "long_command",
        "stuck_query",
        "network_stall",
        "idle_abort",
        "stale_connection",
        "exit_error",
        "signal_kill",
        "unknown",
    ]
)

CRASH_CATEGORIES: frozenset[str] = frozenset(
    [
        "idle_abort",
        "signal_kill",
        "timeout",
        "stale_connection",
        "exit_error",
        "unknown",
    ]
)

# Categories that represent expected (non-crash) terminations
EXPECTED_TERMINATIONS: frozenset[str] = frozenset(
    [
        "graceful_stop",
        "idle_abort",
        "stale_connection",
    ]
)

# Legacy name mappings for migration
# Maps old category names to canonical names
LEGACY_MAPPINGS: dict[str, str] = {
    "idle_session": "idle_abort",  # TimeoutClassifier old name
    "error_exit": "exit_error",  # Normalizing inconsistent naming
}


def normalize_category(category: str) -> str:
    """Normalize a category name using legacy mappings.

    Args:
        category: Category string (may be old or new name)

    Returns:
        Canonical category name

    REQUIRES: category is a string
    ENSURES: Returns string (may be unchanged if no mapping)
    """
    return LEGACY_MAPPINGS.get(category, category)


def is_expected_termination(category: str) -> bool:
    """Check if a category represents an expected (non-crash) termination.

    Args:
        category: Category string

    Returns:
        True if the termination is expected (not a real crash)

    REQUIRES: category is a string
    ENSURES: Returns bool
    """
    normalized = normalize_category(category)
    return normalized in EXPECTED_TERMINATIONS


def is_crash(category: str) -> bool:
    """Check if a category represents a real crash (problem termination).

    Args:
        category: Category string

    Returns:
        True if the termination is a real crash

    REQUIRES: category is a string
    ENSURES: Returns bool
    """
    return not is_expected_termination(category) and category != "unknown"


__all__ = [
    "TerminationCategory",
    "TimeoutCategoryStr",
    "CrashCategoryStr",
    "TIMEOUT_CATEGORIES",
    "CRASH_CATEGORIES",
    "EXPECTED_TERMINATIONS",
    "LEGACY_MAPPINGS",
    "normalize_category",
    "is_expected_termination",
    "is_crash",
]

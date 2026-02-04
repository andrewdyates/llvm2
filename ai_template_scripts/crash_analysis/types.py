# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Data types and constants for crash analysis.

This module contains all dataclasses and constants used by the crash_analysis package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Paths
CRASHES_LOG = Path("worker_logs/crashes.log")

# Thresholds
FAILURE_RATE_WARNING = 0.25  # 25% - yellow alert
FAILURE_RATE_CRITICAL = 0.50  # 50% - red alert, escalate

# Time windows for analysis
RECENT_HOURS = 24  # Look back 24 hours by default


@dataclass
class CrashEntry:
    """A single crash log entry."""

    timestamp: datetime
    iteration: int
    message: str


@dataclass
class CrashFingerprint:
    """Grouped crash pattern with deduplication key.

    Used by Sentry-style fingerprinting to group similar crashes before
    alerting, reducing noise from repeated similar failures.
    """

    fingerprint: str  # Deduplication key: "{category}:{hash}"
    category: str  # crash category (signal_kill, timeout, etc.)
    count: int  # Number of occurrences
    example_message: str  # First occurrence message for context
    first_seen: datetime  # Timestamp of first occurrence
    last_seen: datetime  # Timestamp of most recent occurrence


@dataclass
class SuppressedPattern:
    """A crash pattern that was suppressed due to inhibition rules.

    Following Prometheus Alertmanager pattern, cascading failures are suppressed
    when a root cause pattern is present, reducing alert fatigue.
    """

    category: str  # The suppressed category (e.g., "timeout")
    count: int  # Number of occurrences that were suppressed
    reason: str  # Why it was suppressed (e.g., "cascading from signal_kill")
    inhibited_by: str  # The category that triggered suppression


@dataclass
class ThresholdInfo:
    """Threshold configuration and metadata for health reports."""

    mode: str  # "fixed" or "adaptive"
    warning_threshold: float
    critical_threshold: float
    baseline_sample_count: int = 0
    baseline_mean: float = 0.0
    baseline_stddev: float = 0.0


@dataclass
class HealthReport:
    """System health report."""

    total_iterations: int
    total_crashes: int
    recent_crashes: int  # In last RECENT_HOURS
    failure_rate: float
    status: str  # "healthy", "warning", "critical"
    crash_patterns: dict[str, int]  # Error type -> count
    recommendation: str
    idle_aborts_total: int = 0
    idle_aborts_recent: int = 0
    iterations_unreliable: bool = False  # True if iteration count may be stale/wrong
    threshold_info: ThresholdInfo | None = None  # Threshold metadata for transparency
    unique_fingerprints: list[CrashFingerprint] | None = (
        None  # Deduplicated crash groups
    )
    suppressed_patterns: list[SuppressedPattern] | None = (
        None  # Inhibited cascading failures
    )


__all__ = [
    "CRASHES_LOG",
    "FAILURE_RATE_WARNING",
    "FAILURE_RATE_CRITICAL",
    "RECENT_HOURS",
    "CrashEntry",
    "CrashFingerprint",
    "SuppressedPattern",
    "ThresholdInfo",
    "HealthReport",
]

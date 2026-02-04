# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Threshold calculation functions for crash analysis.

This module computes adaptive and fixed thresholds for failure rate alerting
based on historical baseline samples.
"""

from __future__ import annotations

import statistics
import sys
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from ai_template_scripts.crash_analysis.config_loader import get_config_value
from ai_template_scripts.crash_analysis.crash_detector import (
    _filter_idle_aborts,
    analyze_crash_patterns,
)
from ai_template_scripts.crash_analysis.iteration_counter import (
    count_iterations_between,
)
from ai_template_scripts.crash_analysis.types import (
    FAILURE_RATE_CRITICAL,
    FAILURE_RATE_WARNING,
    ThresholdInfo,
)

if TYPE_CHECKING:
    from ai_template_scripts.crash_analysis.types import CrashEntry


def _compute_failure_rate_for_window(
    start: datetime,
    end: datetime,
    parse_crashes_log_fn: callable,
) -> float | None:
    """Compute failure rate for a specific time window.

    This factors out the failure rate calculation so it can be reused for
    baseline buckets and the current window.

    Args:
        start: Window start time
        end: Window end time
        parse_crashes_log_fn: Function to parse crashes log (injected to avoid circular import)

    Returns:
        Failure rate (0.0-1.0), or None if window has no data
    """
    # Parse crashes in the window
    crashes_raw = parse_crashes_log_fn(since=start)
    # Filter to only include crashes before end
    crashes_in_window = [c for c in crashes_raw if c.timestamp < end]

    # Filter idle aborts
    crashes, _ = _filter_idle_aborts(crashes_in_window)

    # Count iterations in the window
    iterations = count_iterations_between(start, end)

    # Analyze patterns to exclude stale_connection
    patterns = analyze_crash_patterns(crashes)
    stale_crashes = patterns.get("stale_connection", 0)
    real_crashes = len(crashes) - stale_crashes

    # Total attempts
    total_attempts = iterations + real_crashes

    # Skip buckets with no activity (design says skip rather than count as 0)
    if total_attempts == 0:
        return None

    # Skip unreliable buckets (iterations == 0 but crashes > 0)
    if iterations == 0 and real_crashes > 0:
        return None

    return real_crashes / total_attempts


def _compute_baseline_samples(
    now: datetime,
    hours: int,
    config: dict,
    parse_crashes_log_fn: callable,
) -> list[float]:
    """Compute failure rate samples from historical baseline buckets.

    Args:
        now: Current time
        hours: Current reporting window (excluded from baseline)
        config: Loaded config dict
        parse_crashes_log_fn: Function to parse crashes log (injected to avoid circular import)

    Returns:
        List of failure rates from valid baseline buckets
    """
    baseline_window_hours = get_config_value(config, "baseline_window_hours")
    bucket_hours = get_config_value(config, "bucket_hours")

    # Validate config values to prevent infinite loops or silent failures
    if bucket_hours <= 0:
        print(
            f"Warning: bucket_hours ({bucket_hours}) <= 0, disabling adaptive thresholds",
            file=sys.stderr,
        )
        return []
    if baseline_window_hours <= 0:
        print(
            f"Warning: baseline_window_hours ({baseline_window_hours}) <= 0, "
            "disabling adaptive thresholds",
            file=sys.stderr,
        )
        return []

    # Exclude current window from baseline
    baseline_end = now - timedelta(hours=hours)
    baseline_start = baseline_end - timedelta(hours=baseline_window_hours)

    samples = []
    bucket_start = baseline_start

    while bucket_start < baseline_end:
        bucket_end = min(bucket_start + timedelta(hours=bucket_hours), baseline_end)
        rate = _compute_failure_rate_for_window(bucket_start, bucket_end, parse_crashes_log_fn)
        if rate is not None:
            samples.append(rate)
        bucket_start = bucket_end

    return samples


def _compute_adaptive_thresholds(
    config: dict, baseline_samples: list[float]
) -> ThresholdInfo | None:
    """Compute adaptive thresholds from baseline samples.

    Args:
        config: Loaded config dict
        baseline_samples: List of historical failure rates

    Returns:
        ThresholdInfo with adaptive thresholds, or None if not enough samples
    """
    min_samples = get_config_value(config, "min_samples")
    warning_sigma = get_config_value(config, "warning_threshold_sigma")
    critical_sigma = get_config_value(config, "critical_threshold_sigma")
    stddev_floor = get_config_value(config, "stddev_floor")

    # Need min_samples >= 2 for stddev calculation
    if min_samples < 2:
        print(
            "Warning: min_samples < 2, disabling adaptive thresholds",
            file=sys.stderr,
        )
        return None

    if len(baseline_samples) < min_samples:
        return None

    mean = statistics.mean(baseline_samples)
    stddev = max(statistics.stdev(baseline_samples), stddev_floor)

    warning = mean + (warning_sigma * stddev)
    critical = mean + (critical_sigma * stddev)

    # Clamp to [0.0, 1.0]
    warning = max(0.0, min(1.0, warning))
    critical = max(0.0, min(1.0, critical))

    # Ensure warning < critical
    if warning >= critical:
        print(
            f"Warning: Computed warning ({warning:.2f}) >= critical ({critical:.2f}), "
            "falling back to fixed thresholds",
            file=sys.stderr,
        )
        return None

    return ThresholdInfo(
        mode="adaptive",
        warning_threshold=warning,
        critical_threshold=critical,
        baseline_sample_count=len(baseline_samples),
        baseline_mean=mean,
        baseline_stddev=stddev,
    )


def get_thresholds(
    config: dict,
    hours: int,
    parse_crashes_log_fn: callable,
) -> ThresholdInfo:
    """Get effective thresholds based on config.

    Priority:
    1. Explicit fixed thresholds in config (warning_threshold_rate or critical_threshold_rate)
    2. Adaptive thresholds computed from baseline (if sufficient samples)
    3. Default fixed thresholds (FAILURE_RATE_WARNING, FAILURE_RATE_CRITICAL)

    Args:
        config: Loaded config dict
        hours: Current reporting window hours
        parse_crashes_log_fn: Function to parse crashes log (injected to avoid circular import)

    Returns:
        ThresholdInfo with the effective thresholds
    """
    # Check for explicit fixed thresholds first - if EITHER is set, use fixed mode
    # (missing value defaults to module constant)
    warning_rate = get_config_value(config, "warning_threshold_rate")
    critical_rate = get_config_value(config, "critical_threshold_rate")

    if warning_rate is not None or critical_rate is not None:
        return ThresholdInfo(
            mode="fixed",
            warning_threshold=warning_rate
            if warning_rate is not None
            else FAILURE_RATE_WARNING,
            critical_threshold=critical_rate
            if critical_rate is not None
            else FAILURE_RATE_CRITICAL,
        )

    # Try adaptive thresholds
    now = datetime.now()
    baseline_samples = _compute_baseline_samples(now, hours, config, parse_crashes_log_fn)
    adaptive_info = _compute_adaptive_thresholds(config, baseline_samples)

    if adaptive_info is not None:
        return adaptive_info

    # Fall back to defaults
    return ThresholdInfo(
        mode="fixed",
        warning_threshold=FAILURE_RATE_WARNING,
        critical_threshold=FAILURE_RATE_CRITICAL,
    )


__all__ = [
    "_compute_failure_rate_for_window",
    "_compute_baseline_samples",
    "_compute_adaptive_thresholds",
    "get_thresholds",
]

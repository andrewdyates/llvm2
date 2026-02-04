# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
crash_analysis - System health monitoring for AI fleet workers

PURPOSE: Analyzes crash logs to calculate failure rate and health status.
CALLED BY: MANAGER (audit workflow via audit_context.py), human (debugging)
REFERENCED: .claude/rules/ai_template.md (Available Tools table)

Analyzes:
- Failure rate (crashes / total iterations)
- Recent crash patterns
- System health status (healthy/warning/critical)

Public API:
- CRASHES_LOG, FAILURE_RATE_WARNING, FAILURE_RATE_CRITICAL, RECENT_HOURS
- CrashEntry, CrashFingerprint, SuppressedPattern, ThresholdInfo, HealthReport
- parse_crashes_log, count_iterations_since, count_iterations_between
- analyze_crash_patterns, apply_inhibition
- crash_fingerprint, group_crashes_by_fingerprint
- get_health_report, format_report, main

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import config loading functions from dedicated module
from ai_template_scripts.crash_analysis.config_loader import (  # noqa: E402
    CONFIG_PATHS,
    _warn_unknown_keys,
    get_config_value,
    load_config,
    tomllib,
)

# Backwards-compatible aliases for internal functions (used by tests)
_CONFIG_PATHS = CONFIG_PATHS
_load_config = load_config
_get_config_value = get_config_value

# Import crash detection functions from dedicated module
from ai_template_scripts.crash_analysis.crash_detector import (  # noqa: E402
    _filter_idle_aborts,
    _get_crash_category,
    _is_idle_abort,
    analyze_crash_patterns,
    apply_inhibition,
    crash_fingerprint,
    group_crashes_by_fingerprint,
)

# Import iteration counting functions from dedicated module
from ai_template_scripts.crash_analysis.iteration_counter import (  # noqa: E402
    count_iterations_between,
    count_iterations_since,
)

# Import crash log parser from dedicated module
from ai_template_scripts.crash_analysis.parser import parse_crashes_log  # noqa: E402

# Import report formatting function from dedicated module
from ai_template_scripts.crash_analysis.report import format_report  # noqa: E402

# Import threshold calculation functions from dedicated module
from ai_template_scripts.crash_analysis.threshold_calculator import (  # noqa: E402
    _compute_adaptive_thresholds,
    _compute_baseline_samples as _compute_baseline_samples_impl,
    get_thresholds,
)


# Backwards-compatible wrappers for threshold functions (used by tests)
# New signatures add parse_crashes_log_fn to avoid circular imports,
# but tests expect the old signatures without that argument.
def _get_thresholds(config: dict, hours: int) -> ThresholdInfo:
    """Backwards-compatible wrapper for get_thresholds."""
    return get_thresholds(config, hours, parse_crashes_log)


def _compute_baseline_samples(now: datetime, hours: int, config: dict) -> list[float]:
    """Backwards-compatible wrapper for _compute_baseline_samples_impl."""
    return _compute_baseline_samples_impl(now, hours, config, parse_crashes_log)

# Import types and constants from dedicated module
from ai_template_scripts.crash_analysis.types import (  # noqa: E402
    CRASHES_LOG,
    FAILURE_RATE_CRITICAL,
    FAILURE_RATE_WARNING,
    RECENT_HOURS,
    CrashEntry,
    CrashFingerprint,
    HealthReport,
    SuppressedPattern,
    ThresholdInfo,
)
from ai_template_scripts.version import get_version  # noqa: E402

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
    "parse_crashes_log",
    "count_iterations_since",
    "count_iterations_between",
    "analyze_crash_patterns",
    "apply_inhibition",
    "crash_fingerprint",
    "group_crashes_by_fingerprint",
    "get_health_report",
    "format_report",
    "main",
]


def get_health_report(hours: int = RECENT_HOURS) -> HealthReport:
    """Generate system health report.

    Args:
        hours: Number of hours to look back

    Returns:
        HealthReport with analysis

    Contracts:
        REQUIRES: hours > 0
        ENSURES: result.total_iterations >= 0
        ENSURES: result.total_crashes >= 0
        ENSURES: result.recent_crashes >= 0
        ENSURES: 0.0 <= result.failure_rate <= 1.0
        ENSURES: result.status in {"healthy", "warning", "critical"}
        ENSURES: result.failure_rate >= FAILURE_RATE_CRITICAL implies
            result.status == "critical"
        ENSURES: FAILURE_RATE_WARNING <= result.failure_rate < FAILURE_RATE_CRITICAL
            implies result.status == "warning"
        ENSURES: result.failure_rate < FAILURE_RATE_WARNING implies
            result.status == "healthy"
        ENSURES: result.idle_aborts_total >= 0
        ENSURES: result.idle_aborts_recent >= 0
        ENSURES: result.iterations_unreliable implies result.total_iterations == 0
        ENSURES: result.iterations_unreliable implies "WARNING" in result.recommendation
    """
    now = datetime.now()
    since = now - timedelta(hours=hours)

    # Load config for adaptive thresholds
    config, _ = load_config()
    threshold_info = get_thresholds(config, hours, parse_crashes_log)

    # Get all crashes and recent crashes
    all_crashes_raw = parse_crashes_log()
    recent_crashes_raw = parse_crashes_log(since=since)
    all_crashes, idle_all = _filter_idle_aborts(all_crashes_raw)
    recent_crashes, idle_recent = _filter_idle_aborts(recent_crashes_raw)

    # Count successful iterations
    iterations = count_iterations_since(since)

    # Analyze patterns first (needed for adjusted failure rate)
    patterns = analyze_crash_patterns(recent_crashes)

    # Calculate failure rate
    # Exclude stale_connection crashes - they're expected after sleep/resume
    # and will recover cleanly on restart. Idle aborts are filtered earlier.
    stale_crashes = patterns.get("stale_connection", 0)
    real_crashes = len(recent_crashes) - stale_crashes

    # Total attempts = successful iterations + real crashes
    # (A crash is an attempted iteration that failed)
    total_attempts = iterations + real_crashes

    # Detect unreliable iteration count: crashes exist but no iterations found
    # This could mean git log failed, or the since window missed recent commits
    iterations_unreliable = iterations == 0 and real_crashes > 0

    if total_attempts == 0:
        failure_rate = 0.0
    else:
        failure_rate = real_crashes / total_attempts

    # Determine status using adaptive or fixed thresholds
    warning_threshold = threshold_info.warning_threshold
    critical_threshold = threshold_info.critical_threshold

    if failure_rate >= critical_threshold:
        status = "critical"
        recommendation = (
            f"ESCALATE: {failure_rate:.0%} failure rate in last {hours}h "
            f"(threshold: {critical_threshold:.0%}). "
            "Check crashes.log for patterns. Post to Dash News (GitHub Discussions) "
            "if systemic."
        )
    elif failure_rate >= warning_threshold:
        status = "warning"
        recommendation = (
            f"Monitor: {failure_rate:.0%} failure rate in last {hours}h "
            f"(threshold: {warning_threshold:.0%}). "
            "Review recent crashes for patterns."
        )
    else:
        status = "healthy"
        recommendation = "System operating normally."

    # Add note about stale connections if present
    if stale_crashes > 0:
        recommendation += (
            f" ({stale_crashes} stale_connection restart(s) excluded "
            "from failure rate - expected after sleep/resume)"
        )

    if idle_recent > 0:
        recommendation += (
            f" ({idle_recent} idle abort(s) excluded - no issues assigned)"
        )

    # Warn if iteration count is unreliable
    if iterations_unreliable:
        recommendation += (
            " ⚠️ WARNING: Iteration count is 0 but crashes exist - "
            "failure rate may be inaccurate (git log may have failed or "
            "time window too narrow)."
        )

    # Generate fingerprints for recent crashes (deduplication)
    fingerprints = group_crashes_by_fingerprint(recent_crashes)

    # Apply inhibition rules to suppress cascading patterns
    active_patterns, suppressed = apply_inhibition(patterns)

    return HealthReport(
        total_iterations=iterations,
        total_crashes=len(all_crashes),
        recent_crashes=len(recent_crashes),
        failure_rate=failure_rate,
        status=status,
        crash_patterns=active_patterns,
        recommendation=recommendation,
        idle_aborts_total=idle_all,
        idle_aborts_recent=idle_recent,
        iterations_unreliable=iterations_unreliable,
        threshold_info=threshold_info,
        unique_fingerprints=fingerprints,
        suppressed_patterns=suppressed if suppressed else None,
    )


def main() -> int:
    """Main entry point.

    Contracts:
        ENSURES: result in {0, 1, 2}
        ENSURES: result == 0 if healthy or (quiet mode and healthy)
        ENSURES: result == 1 if warning
        ENSURES: result == 2 if critical
    """
    parser = argparse.ArgumentParser(
        description="Check system health from crash logs and git history",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("crash_analysis.py"),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=RECENT_HOURS,
        help=f"Hours to look back (default: {RECENT_HOURS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output if status is warning or critical",
    )

    args = parser.parse_args()

    report = get_health_report(hours=args.hours)

    if args.quiet and report.status == "healthy":
        return 0

    if args.json:
        output = {
            "total_iterations": report.total_iterations,
            "total_crashes": report.total_crashes,
            "recent_crashes": report.recent_crashes,
            "failure_rate": report.failure_rate,
            "status": report.status,
            "crash_patterns": report.crash_patterns,
            "recommendation": report.recommendation,
            "idle_aborts_total": report.idle_aborts_total,
            "idle_aborts_recent": report.idle_aborts_recent,
            "window_hours": args.hours,
            "iterations_unreliable": report.iterations_unreliable,
        }
        # Add threshold info for transparency
        if report.threshold_info:
            ti = report.threshold_info
            output["threshold_info"] = {
                "mode": ti.mode,
                "warning_threshold": ti.warning_threshold,
                "critical_threshold": ti.critical_threshold,
                "baseline_sample_count": ti.baseline_sample_count,
                "baseline_mean": ti.baseline_mean,
                "baseline_stddev": ti.baseline_stddev,
            }
        # Add fingerprint info for deduplication visibility
        if report.unique_fingerprints:
            output["unique_fingerprints"] = [
                {
                    "fingerprint": fp.fingerprint,
                    "category": fp.category,
                    "count": fp.count,
                    "example_message": fp.example_message,
                    "first_seen": fp.first_seen.isoformat(),
                    "last_seen": fp.last_seen.isoformat(),
                }
                for fp in report.unique_fingerprints
            ]
        # Add suppressed patterns for inhibition visibility
        if report.suppressed_patterns:
            output["suppressed_patterns"] = [
                {
                    "category": sp.category,
                    "count": sp.count,
                    "reason": sp.reason,
                    "inhibited_by": sp.inhibited_by,
                }
                for sp in report.suppressed_patterns
            ]
        print(json.dumps(output, indent=2))
    else:
        print(format_report(report, hours=args.hours))

    # Exit code based on status
    if report.status == "critical":
        return 2
    if report.status == "warning":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

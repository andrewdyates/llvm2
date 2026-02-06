# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Report formatting for crash analysis."""

from __future__ import annotations

from ai_template_scripts.crash_analysis.types import (  # noqa: E402
    RECENT_HOURS,
    HealthReport,
)


def format_report(report: HealthReport, hours: int = RECENT_HOURS) -> str:
    """Format health report for display.

    Args:
        report: HealthReport to format
        hours: Time window used

    Returns:
        Formatted string

    Contracts:
        REQUIRES: report is a valid HealthReport
        REQUIRES: hours > 0
        ENSURES: len(result) > 0
        ENSURES: "System Health" in result
        ENSURES: result contains status indicator ([OK], [WARN], or [CRITICAL])
    """
    lines = []

    # Status indicator
    status_icons = {
        "healthy": "[OK]",
        "warning": "[WARN]",
        "critical": "[CRITICAL]",
    }
    icon = status_icons.get(report.status, "[?]")

    lines.append(f"## System Health {icon}")
    lines.append("")
    lines.append(f"**Window:** Last {hours} hours")
    lines.append(f"**Iterations:** {report.total_iterations} successful")
    lines.append(f"**Failures (recent):** {report.recent_crashes}")
    lines.append(f"**Failures (all-time):** {report.total_crashes}")
    if report.idle_aborts_recent or report.idle_aborts_total:
        lines.append(f"**Idle aborts (recent):** {report.idle_aborts_recent}")
        lines.append(f"**Idle aborts (all-time):** {report.idle_aborts_total}")
    lines.append(f"**Failure rate:** {report.failure_rate:.1%}")

    # Add threshold information for transparency
    if report.threshold_info:
        ti = report.threshold_info
        lines.append(
            f"**Thresholds:** {ti.mode} "
            f"(warn: {ti.warning_threshold:.1%}, crit: {ti.critical_threshold:.1%})"
        )
        if ti.mode == "adaptive":
            lines.append(
                f"**Baseline:** {ti.baseline_sample_count} samples, "
                f"mean: {ti.baseline_mean:.1%}, stddev: {ti.baseline_stddev:.1%}"
            )
    lines.append("")

    if report.crash_patterns:
        lines.append("**Failure breakdown:**")
        for pattern, count in sorted(
            report.crash_patterns.items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"  - {pattern}: {count}")
        # Show suppressed patterns inline with failure breakdown
        if report.suppressed_patterns:
            for sp in report.suppressed_patterns:
                lines.append(
                    f"  - ({sp.category}: {sp.count} suppressed - {sp.reason})"
                )
        lines.append("")

    # Show deduplicated failure patterns if available
    if report.unique_fingerprints:
        lines.append(f"**Unique failure patterns:** {len(report.unique_fingerprints)}")
        for fp in report.unique_fingerprints:
            # Show fingerprint, count, and category
            lines.append(f"  - {fp.fingerprint} ({fp.count} occurrences)")
        lines.append("")

    lines.append(f"**Assessment:** {report.recommendation}")

    return "\n".join(lines)


__all__ = [
    "format_report",
]

#!/usr/bin/env python3
"""
health_check.py - System health monitoring for AI fleet workers

PURPOSE: Analyzes crash logs to calculate failure rate and health status.
CALLED BY: MANAGER (audit workflow), human (debugging)
REFERENCED: .claude/rules/ai_template.md (Useful Scripts table)

Analyzes:
- Failure rate (crashes / total iterations)
- Recent crash patterns
- System health status (green/yellow/red)

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
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
class HealthReport:
    """System health report."""

    total_iterations: int
    total_crashes: int
    recent_crashes: int  # In last RECENT_HOURS
    failure_rate: float
    status: str  # "healthy", "warning", "critical"
    crash_patterns: dict[str, int]  # Error type -> count
    recommendation: str


def parse_crashes_log(since: datetime | None = None) -> list[CrashEntry]:
    """Parse crashes.log file.

    Args:
        since: Only include crashes after this timestamp

    Returns:
        List of CrashEntry objects, newest first
    """
    if not CRASHES_LOG.exists():
        return []

    entries = []
    # Format: [2026-01-08 15:30:45] Iteration 5: claude exited with code 1
    pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Iteration (\d+): (.+)",
    )

    with open(CRASHES_LOG) as f:
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


def count_iterations_since(since: datetime) -> int:
    """Count AI role commits since a given time.

    Args:
        since: Start time for counting

    Returns:
        Number of [W], [M], [P], [R] commits since that time
    """
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Count all role commits: [W], [M], [P], [R]
        result = subprocess.run(
            ["git", "log", f"--since={since_str}", "--oneline",
             "--grep=\\[W\\]\\|\\[M\\]\\|\\[P\\]\\|\\[R\\]"],
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
        return 0


def analyze_crash_patterns(crashes: list[CrashEntry]) -> dict[str, int]:
    """Categorize crashes by type.

    Args:
        crashes: List of crash entries

    Returns:
        Dict mapping error type to count

    Categories:
        - signal_kill: Killed by OS signal (SIGKILL, SIGTERM, etc.)
        - timeout: Hit iteration timeout limit
        - stale_connection: Killed due to silence (no output after sleep/resume)
        - exit_error: Non-zero exit code
        - unknown: Unrecognized crash type
    """
    patterns: dict[str, int] = {}

    for crash in crashes:
        msg = crash.message.lower()
        if "signal" in msg:
            category = "signal_kill"
        elif "timed out" in msg:
            category = "timeout"
        elif "killed due to silence" in msg or "stale connection" in msg:
            category = "stale_connection"
        elif "exited with code" in msg:
            category = "exit_error"
        else:
            category = "unknown"

        patterns[category] = patterns.get(category, 0) + 1

    return patterns


def get_health_report(hours: int = RECENT_HOURS) -> HealthReport:
    """Generate system health report.

    Args:
        hours: Number of hours to look back

    Returns:
        HealthReport with analysis
    """
    now = datetime.now()
    since = now - timedelta(hours=hours)

    # Get all crashes and recent crashes
    all_crashes = parse_crashes_log()
    recent_crashes = parse_crashes_log(since=since)

    # Count successful iterations
    iterations = count_iterations_since(since)

    # Analyze patterns first (needed for adjusted failure rate)
    patterns = analyze_crash_patterns(recent_crashes)

    # Calculate failure rate
    # Exclude stale_connection crashes - they're expected after sleep/resume
    # and will recover cleanly on restart
    stale_crashes = patterns.get("stale_connection", 0)
    real_crashes = len(recent_crashes) - stale_crashes

    # Total attempts = successful iterations + real crashes
    # (A crash is an attempted iteration that failed)
    total_attempts = iterations + real_crashes

    if total_attempts == 0:
        failure_rate = 0.0
    else:
        failure_rate = real_crashes / total_attempts

    # Determine status
    if failure_rate >= FAILURE_RATE_CRITICAL:
        status = "critical"
        recommendation = (
            f"ESCALATE: {failure_rate:.0%} failure rate in last {hours}h. "
            "Check crashes.log for patterns. Post to Dash News (GitHub Discussions) if systemic."
        )
    elif failure_rate >= FAILURE_RATE_WARNING:
        status = "warning"
        recommendation = (
            f"Monitor: {failure_rate:.0%} failure rate in last {hours}h. "
            "Review recent crashes for patterns."
        )
    else:
        status = "healthy"
        recommendation = "System operating normally."

    # Add note about stale connections if present
    if stale_crashes > 0:
        recommendation += (
            f" ({stale_crashes} stale_connection restart(s) excluded from failure rate - "
            "expected after sleep/resume)"
        )

    return HealthReport(
        total_iterations=iterations,
        total_crashes=len(all_crashes),
        recent_crashes=len(recent_crashes),
        failure_rate=failure_rate,
        status=status,
        crash_patterns=patterns,
        recommendation=recommendation,
    )


def format_report(report: HealthReport, hours: int = RECENT_HOURS) -> str:
    """Format health report for display.

    Args:
        report: HealthReport to format
        hours: Time window used

    Returns:
        Formatted string
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
    lines.append(f"**Crashes (recent):** {report.recent_crashes}")
    lines.append(f"**Crashes (all-time):** {report.total_crashes}")
    lines.append(f"**Failure rate:** {report.failure_rate:.1%}")
    lines.append("")

    if report.crash_patterns:
        lines.append("**Crash breakdown:**")
        for pattern, count in sorted(
            report.crash_patterns.items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"  - {pattern}: {count}")
        lines.append("")

    lines.append(f"**Assessment:** {report.recommendation}")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check system health from crash logs and git history",
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
            "window_hours": args.hours,
        }
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

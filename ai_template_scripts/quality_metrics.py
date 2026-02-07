#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""quality_metrics.py - Quality metrics for multi-worker evaluation.

Computes reopen rate, regression rate, closure latency, and issue churn
for comparing single-worker vs multi-worker periods.

Public API (library usage):
    from ai_template_scripts.quality_metrics import (
        compute_reopen_rate,
        compute_regression_rate,
        compute_cycle_time,
        compute_issue_churn,
        run_comparison,
    )

CLI usage:
    python3 -m ai_template_scripts.quality_metrics \\
        --single-start 2026-01-24 --single-end 2026-01-28 \\
        --multi-start 2026-02-04 --multi-end 2026-02-08

    python3 -m ai_template_scripts.quality_metrics --help

Design: designs/2026-02-07-quality-metrics-methodology.md
Part of #3261
"""

from __future__ import annotations

__all__ = [
    "compute_reopen_rate",
    "compute_regression_rate",
    "compute_cycle_time",
    "compute_issue_churn",
    "run_comparison",
    "main",
]

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.subprocess_utils import (  # noqa: E402
    get_github_repo,
    run_cmd,
)
from ai_template_scripts.version import get_version  # noqa: E402

# Issue number boundary regex: prevents #1 matching #10, #100, etc.
# Matches the pattern in gh_post/validation.py:_has_fix_commit()
_ISSUE_BOUNDARY = r"([^0-9]|$)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReopenResult:
    """Result of reopen rate computation."""

    reopened_issues: list[int] = field(default_factory=list)
    closed_issues: list[int] = field(default_factory=list)
    reopen_rate: float = 0.0


@dataclass
class RegressionResult:
    """Result of regression rate computation."""

    regression_issues: list[dict] = field(default_factory=list)
    worker_commits: int = 0
    rate_per_100: float = 0.0


@dataclass
class CycleTimeResult:
    """Result of cycle time computation for a single issue."""

    issue: int = 0
    first_part_of: datetime | None = None
    fixes_commit: datetime | None = None
    cycle_hours: float | None = None
    carried_over: bool = False


@dataclass
class ChurnResult:
    """Result of issue churn computation."""

    churned_issues: list[int] = field(default_factory=list)
    multi_churn_issues: list[int] = field(default_factory=list)
    total_churn: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_paginated_json(stdout: str) -> list[dict]:
    """Parse gh api --paginate output (multiple JSON arrays, one per page)."""
    if not stdout or not stdout.strip():
        return []
    results: list[dict] = []
    stdout = stdout.strip()
    if "\n" in stdout:
        for line in stdout.split("\n"):
            line = line.strip()
            if line:
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, list):
                        results.extend(parsed)
                    elif isinstance(parsed, dict):
                        results.append(parsed)
                except json.JSONDecodeError:
                    pass
    else:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, list):
                results = parsed
            elif isinstance(parsed, dict):
                results = [parsed]
        except json.JSONDecodeError:
            pass
    return results


def _get_owner_repo() -> tuple[str, str] | None:
    """Get owner and repo name from git remote via gh CLI."""
    result = get_github_repo()
    if not result.ok or not result.stdout.strip():
        return None
    full = result.stdout.strip()
    if "/" in full:
        parts = full.split("/", 1)
        return parts[0], parts[1]
    return None


def _parse_iso(s: str) -> datetime | None:
    """Parse ISO 8601 datetime string."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _in_range(dt: datetime, start: datetime, end: datetime) -> bool:
    """Check if datetime falls within [start, end)."""
    return start <= dt < end


def _date_to_utc(d: str) -> datetime:
    """Convert YYYY-MM-DD string to UTC midnight datetime."""
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC)


def _extract_issue_numbers(text: str, keyword: str) -> list[int]:
    """Extract all issue numbers from text matching 'keyword #N'.

    Captures the full number (greedy \\d+), so 'Fixes #123' yields 123
    not 1, 12, or 123. Boundary regex is only needed for git --grep
    where the pattern matches substrings; here we capture the whole group.
    """
    pattern = rf"{keyword}\s+#(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [int(m) for m in matches]


def _extract_keyword_timestamps(log_output: str, keyword: str) -> dict[int, list[datetime]]:
    """Extract issue -> timestamp list from git log formatted as '%aI %s%n%b'."""
    issue_times: dict[int, list[datetime]] = {}
    current_ts: datetime | None = None
    for raw_line in log_output.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:+\-Z]+)\s+(.*)", line)
        if match:
            current_ts = _parse_iso(match.group(1))
            text = match.group(2)
        else:
            text = line
        if not current_ts:
            continue
        for num in _extract_issue_numbers(text, keyword):
            issue_times.setdefault(num, []).append(current_ts)
    return issue_times


def _get_issue_reopen_times(owner: str, repo: str, issue_num: int) -> list[datetime]:
    """Get reopened event timestamps for one issue from GitHub API."""
    api_result = run_cmd(
        [
            "gh", "api", "--paginate",
            f"/repos/{owner}/{repo}/issues/{issue_num}/events?per_page=100",
            "-q",
            '[.[] | select(.event == "reopened") | {created_at}]',
        ],
        timeout=60,
    )
    if not api_result.ok or not api_result.stdout.strip():
        return []
    events = _parse_paginated_json(api_result.stdout)
    reopened: list[datetime] = []
    for event in events:
        if isinstance(event, dict):
            created = _parse_iso(event.get("created_at", ""))
            if created:
                reopened.append(created)
    reopened.sort()
    return reopened


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def compute_reopen_rate(
    start_date: str, end_date: str
) -> ReopenResult:
    """Compute reopen rate for issues closed in the given period.

    Finds issues closed with 'Fixes #N' during [start, end), then checks
    which were reopened within 7 days of their closure date. The observation
    window extends 7 days past end_date to capture late reopens.

    Args:
        start_date: Period start as YYYY-MM-DD (UTC midnight).
        end_date: Period end as YYYY-MM-DD (UTC midnight).

    Returns:
        ReopenResult with reopened issues, closed issues, and rate.
    """
    start = _date_to_utc(start_date)
    end = _date_to_utc(end_date)
    result = ReopenResult()

    # Step 1: Find unique issues closed with "Fixes #N" in the period via git log
    git_result = run_cmd(
        [
            "git", "log", "-i", "-E",
            f"--after={start_date}",
            f"--before={end_date}",
            "--grep=Fixes #[0-9]",
            "--format=%aI %s%n%b",
        ],
        timeout=60,
    )
    if not git_result.ok:
        return result

    closed_times_raw = _extract_keyword_timestamps(git_result.stdout, "Fixes")
    closed_times = {
        issue: min(times)
        for issue, times in closed_times_raw.items()
        if times
    }
    closed_set = set(closed_times)

    result.closed_issues = sorted(closed_set)
    if not closed_set:
        return result

    # Step 2: For each closed issue, check if it was reopened within 7 days
    # Use GraphQL batch to check reopen counts efficiently
    batch_available = False
    try:
        from ai_template_scripts.gh_rate_limit.batch import batch_issue_timelines

        reopen_counts = batch_issue_timelines(list(closed_set))
        batch_available = True
    except (ImportError, Exception):
        reopen_counts = {}

    # Also check git log for "Reopens #N" commits within observation window
    obs_end = end + timedelta(days=7)
    obs_end_str = obs_end.strftime("%Y-%m-%d")
    git_reopen = run_cmd(
        [
            "git", "log", "-i", "-E",
            f"--after={start_date}",
            f"--before={obs_end_str}",
            "--grep=Reopens #[0-9]",
            "--format=%aI %s%n%b",
        ],
        timeout=60,
    )
    reopened_git_times: dict[int, list[datetime]] = {}
    if git_reopen.ok:
        reopened_git_times = _extract_keyword_timestamps(git_reopen.stdout, "Reopens")

    owner_repo = _get_owner_repo()
    api_reopen_cache: dict[int, list[datetime]] = {}

    # Combine signals and enforce per-issue 7-day window after closure timestamp.
    for issue_num in closed_set:
        close_ts = closed_times.get(issue_num)
        if not close_ts:
            continue
        window_end = close_ts + timedelta(days=7)

        git_hit = any(
            _in_range(ts, close_ts, window_end)
            for ts in reopened_git_times.get(issue_num, [])
        )

        api_hit = False
        if owner_repo:
            batch_count = reopen_counts.get(issue_num)
            should_query_api = (
                not batch_available
                or batch_count is None
                or batch_count > 0
                or git_hit
            )
            if should_query_api:
                if issue_num not in api_reopen_cache:
                    owner, repo = owner_repo
                    api_reopen_cache[issue_num] = _get_issue_reopen_times(owner, repo, issue_num)
                api_hit = any(
                    _in_range(ts, close_ts, window_end)
                    for ts in api_reopen_cache[issue_num]
                )

        if api_hit or git_hit:
            result.reopened_issues.append(issue_num)

    result.reopened_issues.sort()
    if result.closed_issues:
        result.reopen_rate = round(
            (len(result.reopened_issues) / len(result.closed_issues)) * 100, 1
        )
    return result


def compute_regression_rate(
    start_date: str, end_date: str
) -> RegressionResult:
    """Compute regression rate: regression issues per 100 Worker commits.

    Args:
        start_date: Period start as YYYY-MM-DD (UTC midnight).
        end_date: Period end as YYYY-MM-DD (UTC midnight).

    Returns:
        RegressionResult with regression issues, worker commit count, and rate.
    """
    result = RegressionResult()

    # Count Worker commits in the period
    # --grep matches against commit message text (not formatted output),
    # so ^ anchors to start of subject line correctly
    git_result = run_cmd(
        [
            "git", "log", "--oneline",
            f"--after={start_date}",
            f"--before={end_date}",
            "-E", "--grep=^\\[W[0-9]*\\]",
        ],
        timeout=60,
    )
    if git_result.ok and git_result.stdout.strip():
        result.worker_commits = len(
            [l for l in git_result.stdout.strip().split("\n") if l.strip()]
        )

    # Find regression issues filed in the period
    gh_result = run_cmd(
        [
            "gh", "issue", "list", "--state", "all",
            "--search", f"regression created:{start_date}..{end_date}",
            "--json", "number,createdAt,title",
            "--limit", "200",
        ],
        timeout=60,
    )
    if gh_result.ok and gh_result.stdout.strip():
        try:
            issues = json.loads(gh_result.stdout)
            start_dt = _date_to_utc(start_date)
            end_dt = _date_to_utc(end_date)
            for issue in issues:
                title = issue.get("title", "")
                created = _parse_iso(issue.get("createdAt", ""))
                # Title must contain "regression" (case-insensitive)
                if "regression" in title.lower() and created and _in_range(created, start_dt, end_dt):
                    result.regression_issues.append(
                        {"number": issue["number"], "title": title}
                    )
        except (json.JSONDecodeError, TypeError):
            pass

    if result.worker_commits > 0:
        result.rate_per_100 = round(
            (len(result.regression_issues) / result.worker_commits) * 100, 1
        )
    return result


def compute_cycle_time(
    start_date: str, end_date: str
) -> list[CycleTimeResult]:
    """Compute closure latency from first 'Part of #N' to 'Fixes #N'.

    For each issue closed with 'Fixes #N' in the period, finds the first
    'Part of #N' commit and computes the wall-clock delta.

    Args:
        start_date: Period start as YYYY-MM-DD (UTC midnight).
        end_date: Period end as YYYY-MM-DD (UTC midnight).

    Returns:
        List of CycleTimeResult, one per issue with cycle time data.
    """
    start_dt = _date_to_utc(start_date)
    end_dt = _date_to_utc(end_date)
    results: list[CycleTimeResult] = []

    # Find issues closed with "Fixes #N" in the period
    git_result = run_cmd(
        [
            "git", "log", "-i", "-E",
            f"--after={start_date}",
            f"--before={end_date}",
            "--grep=Fixes #[0-9]",
            "--format=%aI %s%n%b",
        ],
        timeout=60,
    )
    if not git_result.ok:
        return results

    # Map issue -> fixes timestamp (first Fixes commit in period)
    # Format is: "<timestamp> <subject>\n<body lines>..." per commit.
    # Body lines lack a timestamp prefix; associate them with the most recent ts.
    fixes_map: dict[int, datetime] = {}
    current_ts: datetime | None = None
    for line in git_result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Try to parse leading timestamp (subject line of a commit)
        match = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:+\-]+)\s+(.*)", line)
        if match:
            current_ts = _parse_iso(match.group(1))
            text = match.group(2)
        else:
            # Body line — use most recent commit timestamp
            text = line
        if current_ts:
            for num in _extract_issue_numbers(text, "Fixes"):
                # Always overwrite: git log outputs newest-first, so
                # the last value seen is chronologically earliest.
                # Design: "Multiple Fixes → use first Fixes" (earliest).
                fixes_map[num] = current_ts

    if not fixes_map:
        return results

    # For each issue, find first "Part of #N" commit (any time, not just period)
    for issue_num, fixes_ts in sorted(fixes_map.items()):
        ct = CycleTimeResult(issue=issue_num, fixes_commit=fixes_ts)

        # Search for first "Part of #N" commit with boundary regex
        part_result = run_cmd(
            [
                "git", "log", "--reverse", "-i", "-E",
                f"--grep=Part of #{issue_num}{_ISSUE_BOUNDARY}",
                "--format=%aI",
            ],
            timeout=30,
        )
        if part_result.ok and part_result.stdout.strip():
            first_line = part_result.stdout.strip().split("\n")[0].strip()
            first_part = _parse_iso(first_line)
            if first_part:
                ct.first_part_of = first_part
                # Check if Part of predates the window
                if first_part < start_dt:
                    ct.carried_over = True
                    # Use window start as effective start time
                    effective_start = start_dt
                else:
                    effective_start = first_part
                delta = fixes_ts - effective_start
                ct.cycle_hours = round(delta.total_seconds() / 3600, 1)

        results.append(ct)

    return results


def compute_issue_churn(
    start_date: str, end_date: str
) -> ChurnResult:
    """Compute issue churn: issues with multiple close events in the period.

    Uses GitHub Events API to find close/reopen cycles, supplemented by
    git log for "Reopens #N" commits.

    Args:
        start_date: Period start as YYYY-MM-DD (UTC midnight).
        end_date: Period end as YYYY-MM-DD (UTC midnight).

    Returns:
        ChurnResult with churned issues and multi-churn issues.
    """
    start_dt = _date_to_utc(start_date)
    end_dt = _date_to_utc(end_date)
    result = ChurnResult()

    owner_repo = _get_owner_repo()
    if not owner_repo:
        return result
    owner, repo = owner_repo

    # Fetch closed events in the period from GitHub API
    close_counts: dict[int, int] = {}
    api_result = run_cmd(
        [
            "gh", "api", "--paginate",
            f"/repos/{owner}/{repo}/issues/events?per_page=100",
            "-q",
            '[.[] | select(.event == "closed") | {created_at, issue: .issue.number}]',
        ],
        timeout=120,
    )
    if api_result.ok and api_result.stdout.strip():
        events = _parse_paginated_json(api_result.stdout)
        for event in events:
            created = _parse_iso(event.get("created_at", ""))
            issue_num = event.get("issue")
            if created and issue_num and _in_range(created, start_dt, end_dt):
                close_counts[issue_num] = close_counts.get(issue_num, 0) + 1

    # Supplement with git log: issues with both Fixes and Reopens
    git_result = run_cmd(
        [
            "git", "log", "-i", "-E",
            f"--after={start_date}",
            f"--before={end_date}",
            "--grep=Reopens #[0-9]",
            "--format=%s%n%b",
        ],
        timeout=60,
    )
    if git_result.ok:
        for line in git_result.stdout.split("\n"):
            for num in _extract_issue_numbers(line, "Reopens"):
                # A reopen implies at least 2 close events (was closed, reopened, will close again)
                close_counts[num] = max(close_counts.get(num, 0), 2)

    for issue_num, count in sorted(close_counts.items()):
        if count >= 2:
            result.churned_issues.append(issue_num)
            if count >= 3:
                result.multi_churn_issues.append(issue_num)

    result.total_churn = len(result.churned_issues)
    return result


# ---------------------------------------------------------------------------
# Comparison and output
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(values):
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def _format_hours(h: float | None) -> str:
    """Format hours as human-readable string."""
    if h is None:
        return "N/A"
    if h < 1:
        return f"{h * 60:.0f}m"
    if h < 24:
        return f"{h:.1f}h"
    return f"{h / 24:.1f}d"


def run_comparison(
    single_start: str,
    single_end: str,
    multi_start: str,
    multi_end: str,
) -> dict:
    """Run all 4 metrics for both periods and produce comparison.

    Args:
        single_start: Single-worker period start (YYYY-MM-DD).
        single_end: Single-worker period end (YYYY-MM-DD).
        multi_start: Multi-worker period start (YYYY-MM-DD).
        multi_end: Multi-worker period end (YYYY-MM-DD).

    Returns:
        Dict with all results and comparison table.
    """
    print(f"Computing quality metrics...")
    print(f"  Single-worker period: {single_start} to {single_end}")
    print(f"  Multi-worker period:  {multi_start} to {multi_end}")
    print()

    # Compute all metrics for both periods
    print("  [1/8] Reopen rate (single)...")
    s_reopen = compute_reopen_rate(single_start, single_end)
    print("  [2/8] Reopen rate (multi)...")
    m_reopen = compute_reopen_rate(multi_start, multi_end)

    print("  [3/8] Regression rate (single)...")
    s_regression = compute_regression_rate(single_start, single_end)
    print("  [4/8] Regression rate (multi)...")
    m_regression = compute_regression_rate(multi_start, multi_end)

    print("  [5/8] Cycle time (single)...")
    s_cycle = compute_cycle_time(single_start, single_end)
    print("  [6/8] Cycle time (multi)...")
    m_cycle = compute_cycle_time(multi_start, multi_end)

    print("  [7/8] Issue churn (single)...")
    s_churn = compute_issue_churn(single_start, single_end)
    print("  [8/8] Issue churn (multi)...")
    m_churn = compute_issue_churn(multi_start, multi_end)

    # Compute cycle time stats
    s_hours = [c.cycle_hours for c in s_cycle if c.cycle_hours is not None]
    m_hours = [c.cycle_hours for c in m_cycle if c.cycle_hours is not None]

    s_median = _percentile(s_hours, 50) if s_hours else None
    s_p75 = _percentile(s_hours, 75) if s_hours else None
    s_p95 = _percentile(s_hours, 95) if s_hours else None
    m_median = _percentile(m_hours, 50) if m_hours else None
    m_p75 = _percentile(m_hours, 75) if m_hours else None
    m_p95 = _percentile(m_hours, 95) if m_hours else None

    # Build comparison data
    comparison = {
        "single_period": {"start": single_start, "end": single_end},
        "multi_period": {"start": multi_start, "end": multi_end},
        "reopen_rate": {
            "single": {
                "rate": s_reopen.reopen_rate,
                "reopened": len(s_reopen.reopened_issues),
                "closed": len(s_reopen.closed_issues),
                "reopened_issues": s_reopen.reopened_issues,
            },
            "multi": {
                "rate": m_reopen.reopen_rate,
                "reopened": len(m_reopen.reopened_issues),
                "closed": len(m_reopen.closed_issues),
                "reopened_issues": m_reopen.reopened_issues,
            },
        },
        "regression_rate": {
            "single": {
                "rate": s_regression.rate_per_100,
                "count": len(s_regression.regression_issues),
                "worker_commits": s_regression.worker_commits,
                "issues": s_regression.regression_issues,
            },
            "multi": {
                "rate": m_regression.rate_per_100,
                "count": len(m_regression.regression_issues),
                "worker_commits": m_regression.worker_commits,
                "issues": m_regression.regression_issues,
            },
        },
        "cycle_time": {
            "single": {
                "median": s_median,
                "p75": s_p75,
                "p95": s_p95,
                "count": len(s_hours),
                "carried_over": sum(1 for c in s_cycle if c.carried_over),
                "direct_closures": len(s_cycle) - len(s_hours),
            },
            "multi": {
                "median": m_median,
                "p75": m_p75,
                "p95": m_p95,
                "count": len(m_hours),
                "carried_over": sum(1 for c in m_cycle if c.carried_over),
                "direct_closures": len(m_cycle) - len(m_hours),
            },
        },
        "issue_churn": {
            "single": {
                "total": s_churn.total_churn,
                "multi_churn": len(s_churn.multi_churn_issues),
                "issues": s_churn.churned_issues,
            },
            "multi": {
                "total": m_churn.total_churn,
                "multi_churn": len(m_churn.multi_churn_issues),
                "issues": m_churn.churned_issues,
            },
        },
    }

    return comparison


def _delta_str(a: float | None, b: float | None) -> str:
    """Format delta between two values."""
    if a is None or b is None:
        return "N/A"
    d = b - a
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}"


def format_markdown(comparison: dict) -> str:
    """Format comparison results as markdown table."""
    s = comparison
    lines = []
    lines.append("### 8. Quality Metrics")
    lines.append("")
    lines.append(
        f"Single-worker period: {s['single_period']['start']} to "
        f"{s['single_period']['end']}"
    )
    lines.append(
        f"Multi-worker period: {s['multi_period']['start']} to "
        f"{s['multi_period']['end']}"
    )
    lines.append("")

    # Summary table
    sr_s = s["reopen_rate"]["single"]
    sr_m = s["reopen_rate"]["multi"]
    rr_s = s["regression_rate"]["single"]
    rr_m = s["regression_rate"]["multi"]
    ct_s = s["cycle_time"]["single"]
    ct_m = s["cycle_time"]["multi"]
    ch_s = s["issue_churn"]["single"]
    ch_m = s["issue_churn"]["multi"]

    lines.append("| Metric | Single Period | Multi Period | Delta | Notes |")
    lines.append("|--------|--------------|-------------|-------|-------|")
    lines.append(
        f"| Reopen rate | {sr_s['rate']}% ({sr_s['reopened']}/{sr_s['closed']}) "
        f"| {sr_m['rate']}% ({sr_m['reopened']}/{sr_m['closed']}) "
        f"| {_delta_str(sr_s['rate'], sr_m['rate'])} pp | "
        f"7-day observation window |"
    )
    lines.append(
        f"| Regression rate (per 100 commits) | {rr_s['rate']} ({rr_s['count']}/{rr_s['worker_commits']}) "
        f"| {rr_m['rate']} ({rr_m['count']}/{rr_m['worker_commits']}) "
        f"| {_delta_str(rr_s['rate'], rr_m['rate'])} | "
        f"Title-match only |"
    )
    lines.append(
        f"| Median closure latency | {_format_hours(ct_s['median'])} (n={ct_s['count']}) "
        f"| {_format_hours(ct_m['median'])} (n={ct_m['count']}) "
        f"| {_delta_str(ct_s['median'], ct_m['median'])} | "
        f"Carried over: {ct_s['carried_over']} / {ct_m['carried_over']} |"
    )
    lines.append(
        f"| P95 closure latency | {_format_hours(ct_s['p95'])} "
        f"| {_format_hours(ct_m['p95'])} "
        f"| {_delta_str(ct_s['p95'], ct_m['p95'])} | |"
    )
    lines.append(
        f"| Issue churn count | {ch_s['total']} ({ch_s.get('multi_churn', 0)} multi) "
        f"| {ch_m['total']} ({ch_m.get('multi_churn', 0)} multi) "
        f"| {_delta_str(ch_s['total'], ch_m['total'])} | "
        f"Issues with >=2 close events |"
    )
    lines.append("")

    # Confounding variables
    lines.append("**Confounding variables:** Manager maturity (more active Manager in ")
    lines.append("multi-worker period), issue inflation (#3241), self-audit discipline ")
    lines.append("(26.1% self-audit rate), and project maturity (later periods have more ")
    lines.append("infrastructure). Causal attribution is not claimed.")
    lines.append("")

    # Detail sections
    if sr_m["reopened_issues"]:
        lines.append(f"**Reopened issues (multi):** {', '.join(f'#{n}' for n in sr_m['reopened_issues'])}")
        lines.append("")
    if sr_s["reopened_issues"]:
        lines.append(f"**Reopened issues (single):** {', '.join(f'#{n}' for n in sr_s['reopened_issues'])}")
        lines.append("")

    if rr_s["issues"] or rr_m["issues"]:
        lines.append("**Regression issues found:**")
        for issue in rr_s["issues"]:
            lines.append(f"- (single) #{issue['number']}: {issue['title']}")
        for issue in rr_m["issues"]:
            lines.append(f"- (multi) #{issue['number']}: {issue['title']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Quality metrics for multi-worker evaluation (Part of #3261)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Date format: YYYY-MM-DD (midnight UTC).

Default periods (from throughput report):
  Single-worker extended: 2026-01-24 to 2026-01-28
  Multi-worker extended:  2026-02-04 to 2026-02-08

Examples:
    %(prog)s
    %(prog)s --single-start 2026-01-24 --single-end 2026-01-28 \\
             --multi-start 2026-02-04 --multi-end 2026-02-08
    %(prog)s --json
    %(prog)s --append-to-report
""",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("quality_metrics.py"),
    )
    parser.add_argument(
        "--single-start",
        default="2026-01-24",
        help="Single-worker period start (default: 2026-01-24)",
    )
    parser.add_argument(
        "--single-end",
        default="2026-01-28",
        help="Single-worker period end (default: 2026-01-28)",
    )
    parser.add_argument(
        "--multi-start",
        default="2026-02-04",
        help="Multi-worker period start (default: 2026-02-04)",
    )
    parser.add_argument(
        "--multi-end",
        default="2026-02-08",
        help="Multi-worker period end (default: 2026-02-08)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown",
    )
    parser.add_argument(
        "--append-to-report",
        action="store_true",
        help="Append results to the throughput measurement report",
    )

    args = parser.parse_args()

    comparison = run_comparison(
        args.single_start, args.single_end,
        args.multi_start, args.multi_end,
    )

    if args.json:
        print(json.dumps(comparison, indent=2, default=str))
    else:
        md = format_markdown(comparison)
        print()
        print(md)

    if args.append_to_report:
        report_path = _repo_root / "reports" / "research" / "2026-02-07-multi-worker-throughput-measurement.md"
        if report_path.exists():
            md = format_markdown(comparison)
            with open(report_path, "a") as f:
                f.write("\n\n")
                f.write(md)
            print(f"\nAppended to {report_path}")
        else:
            print(f"\nWarning: Report not found at {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

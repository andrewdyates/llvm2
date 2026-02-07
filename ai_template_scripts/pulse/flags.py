#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Flag management for pulse metrics.

Functions for threshold checking and flag file I/O.
Part of #404: pulse.py module split.
"""

import shutil
import sys
from collections.abc import Callable
from datetime import datetime

from .config import THRESHOLDS
from .constants import FLAGS_DIR
from .issue_metrics import _unpack_issue_counts

# Patterns for flags managed by looper, not pulse (#2409)
# These flags should not be cleared during pulse write_flags() cleanup
# because they need to persist until consumed by the next looper session.
# startup_warnings added in #2459 (phase rotation postmortem follow-up)
# ownership_conflict_ added in #3224 (multi-worker ownership escalation path)
LOOPER_MANAGED_PREFIXES = (
    "headless_violation_",
    "startup_warnings",
    "ownership_conflict_",
)


def check_thresholds(metrics: dict) -> list[str]:
    """Check thresholds, return list of triggered flags.

    REQUIRES: metrics is a dict (may be empty or partial)
    ENSURES: Returns list of flag name strings
    ENSURES: Each flag is a valid flag name from THRESHOLDS
    ENSURES: Never raises (handles missing/invalid metrics gracefully)
    """
    flags = []

    # Large files - flag only when warning tier files exceed threshold (Part of #2358)
    # Warning tier = files exceeding max_file_lines_warning (default 5000 lines)
    # Notice tier = files exceeding max_file_lines (default 1000 lines)
    large_files = metrics.get("large_files", [])
    # Check if any file has tier info (new format)
    has_tier_info = any(isinstance(f, dict) and "tier" in f for f in large_files)
    if has_tier_info:
        # New format: only count warning tier files
        warning_files = [
            f for f in large_files if isinstance(f, dict) and f.get("tier") == "warning"
        ]
        file_count = len(warning_files)
    else:
        # Legacy format (no tier info): count all files
        file_count = len(large_files)
    if file_count >= THRESHOLDS["max_files_over_limit"]:
        flags.append("large_files")

    # Forbidden CI workflows (GitHub Actions not supported)
    if metrics.get("forbidden_ci"):
        flags.append("forbidden_ci")

    # Failures - only flag "real" failures, not stale_connection restarts
    # (stale_connection is expected after sleep/resume and auto-recovers)
    crashes_data = metrics.get("crashes_24h", {})
    if isinstance(crashes_data, dict):
        if crashes_data.get("real", 0) > 0:
            flags.append("failures")
    elif crashes_data > 0:
        # Legacy: int value (shouldn't happen but handle gracefully)
        flags.append("failures")

    # Untraceable failures - worker failures with empty working_issues (#938)
    # These represent lost work that can't be attributed to any issue
    untraceable = metrics.get("untraceable_failures", [])
    if untraceable:
        flags.append("untraceable_failures")

    # Blocked issues
    issues = metrics.get("issues", {})
    issue_counts, issue_error = _unpack_issue_counts(issues)
    if issue_counts.get("blocked", 0) > 0:
        flags.append("blocked_issues")

    # Blocked issues missing "Blocked:" reason in body
    # (skipped when --skip-gh-api is used)
    blocked_missing = metrics.get("blocked_missing_reason", [])
    if blocked_missing and blocked_missing != "skipped":
        flags.append("blocked_missing_reason")

    # Blocked issues with closed blockers (#1497)
    # (skipped when --skip-gh-api is used)
    stale_blockers = metrics.get("stale_blockers", [])
    if stale_blockers and stale_blockers != "skipped":
        flags.append("stale_blockers")

    # Blocked issues with no progress >7 days (#1497)
    # (skipped when --skip-gh-api is used)
    long_blocked = metrics.get("long_blocked", [])
    if long_blocked and long_blocked != "skipped":
        flags.append("long_blocked")

    # GitHub API error (couldn't fetch issues - don't assume no_work)
    if issue_error:
        flags.append("gh_error")
    # No open issues (might need roadmap refresh) - only if we successfully fetched
    elif issue_counts.get("open", 0) == 0 and issue_counts.get("in_progress", 0) == 0:
        flags.append("no_work")

    # Orphaned tests (code smell)
    tests = metrics.get("tests", {})
    if tests.get("orphaned_tests"):
        flags.append("orphaned_tests")

    # Test failures in last 24h
    recent = tests.get("recent_24h", {})
    if recent.get("failed", 0) > 0:
        flags.append("test_failures")

    # Low proof coverage (Rust project with Kani but <10% coverage)
    # Don't flag if project has proof harnesses - those provide verification too
    proofs = metrics.get("proofs", {})
    kani = proofs.get("kani", {})
    has_proofs = kani.get("proofs", 0) > 0
    if (
        kani.get("total_functions", 0) > 10
        and kani.get("coverage_pct", 0) < 10
        and not has_proofs
    ):
        flags.append("low_proof_coverage")

    # Kani proofs written but not executed (#2263)
    # Flag if >20% of harnesses are unverified (not_run + timeout + oom)
    total_proofs = kani.get("proofs", 0)
    if total_proofs > 0:
        unexecuted = kani.get("unexecuted", 0)
        timeout = kani.get("timeout", 0)
        oom = kani.get("oom", 0)
        unverified = unexecuted + timeout + oom
        unverified_pct = (unverified / total_proofs) * 100
        if unverified_pct > 20:
            flags.append("kani_unverified")

    # System resource checks
    system = metrics.get("system", {})

    # Memory usage - consolidated flag with severity (Part of #2379)
    # Single "memory" flag with severity in content instead of separate warning/critical flags
    memory = system.get("memory", {})
    mem_percent = memory.get("percent_used", 0)
    if mem_percent >= THRESHOLDS["memory_warning_percent"]:
        flags.append("memory")  # severity determined by _memory_severity()

    # Disk usage - consolidated flag with severity (Part of #2379)
    # Single "disk" flag with severity in content instead of separate warning/critical flags
    disk = system.get("disk", {})
    disk_percent_str = disk.get("percent_used", "0%")
    try:
        disk_percent = int(disk_percent_str.rstrip("%"))
        if disk_percent >= THRESHOLDS["disk_warning_percent"]:
            flags.append("disk")  # severity determined by _disk_severity()
    except ValueError:
        pass

    # Disk bloat detection (#1477)
    # Flag large files, bloated tests/, or generated artifacts
    disk_bloat = system.get("disk_bloat", {})
    if disk_bloat.get("bloat_detected"):
        flags.append("disk_bloat")

    # Claude CLI version mismatch
    # Flag if installed version doesn't match pinned version
    claude_cli = system.get("claude_cli", {})
    if claude_cli.get("mismatch"):
        flags.append("claude_version_mismatch")

    # Active sessions without claimed issues (process gap detection)
    # Only flag if we successfully fetched issue counts (no gh_error)
    active_sessions = metrics.get("active_sessions", 0)
    if active_sessions > 0 and not issue_error:
        if issue_counts.get("in_progress", 0) == 0:
            flags.append("unclaimed_sessions")

    # Long-running processes (#922)
    # Flag if any verification/compilation process exceeds threshold
    long_running = metrics.get("long_running_processes", [])
    if long_running:
        flags.append("stuck_process")

    # Documentation claim drift (#1494)
    # Flag if documented claims don't match code reality
    doc_claims = metrics.get("doc_claims", {})
    if doc_claims.get("drift_detected", False):
        flags.append("doc_claim_drift")

    # Outdated git dependencies (#1553)
    # Flag if any git deps have updates available
    git_deps = metrics.get("git_deps", {})
    if not git_deps.get("skipped") and git_deps.get("outdated"):
        flags.append("deps_outdated")

    # Git operation in progress (#1899)
    # Flag if rebase/merge/cherry-pick/revert is incomplete
    # These block commits/pushes and can silently stall AI roles
    git_info = metrics.get("git", {})
    if git_info.get("operation_in_progress"):
        flags.append("git_operation_in_progress")

    # High reopen rate (#1937)
    # Flag if >5% of closed issues were reopened (indicates closure quality issues)
    issues_reopened = metrics.get("issues_reopened", {})
    if not issues_reopened.get("skipped") and not issues_reopened.get("rate_limited"):
        reopen_rate = issues_reopened.get("reopen_rate", 0)
        if reopen_rate > 5:
            flags.append("high_reopen_rate")

    # Complexity increase (#2129)
    # Flag if average complexity increased >10% vs previous day
    # This detects unintended complexity creep per Anthropic's simplicity principle
    # Note: 10% flag threshold is higher than 5% trend threshold (in core.py)
    # because trend gives early visibility while flag indicates significant concern
    complexity = metrics.get("complexity", {})
    if not complexity.get("error"):
        delta = complexity.get("delta", {})
        delta_pct = delta.get("delta_percent", 0)
        if delta_pct > 10:
            flags.append("complexity_increase")

    # GitHub API quota overflow (#2362)
    # Flag when apps exhaust quota and fall back to lower-priority apps
    # Threshold configurable via pulse.toml [thresholds] gh_quota_overflow_threshold
    gh_overflow = metrics.get("gh_quota_overflow", {})
    overflow_count = gh_overflow.get("recent_count", 0)
    overflow_threshold = THRESHOLDS.get("gh_quota_overflow_threshold", 10)
    if overflow_count >= overflow_threshold:
        flags.append("gh_quota_overflow")

    # Startup warnings from looper config validation (#2459)
    # Flag when .flags/startup_warnings has actual warnings (not just exists)
    # Per #2475: empty files should not trigger the flag
    startup_warnings = metrics.get("startup_warnings", {})
    if startup_warnings.get("count", 0) > 0:
        flags.append("startup_warnings")

    return flags


# ============================================================================
# Flag summary handlers - Part of #2333 refactoring
# Each handler returns a list of summary lines for its flag type.
# ============================================================================


def _summary_test_failures(metrics: dict) -> list[str]:
    """Summary for test_failures flag."""
    tests = metrics.get("tests", {})
    recent = tests.get("recent_24h", {})
    failed = recent.get("failed", 0)
    total = recent.get("total", 0)
    return [f"Failed: {failed}/{total} runs in 24h"]


def _summary_blocked_issues(metrics: dict) -> list[str]:
    """Summary for blocked_issues flag."""
    issues = metrics.get("issues", {})
    issue_counts, _ = _unpack_issue_counts(issues)
    blocked_list = metrics.get("blocked_issue_list", [])
    lines = [f"Count: {issue_counts.get('blocked', 0)}"]
    if isinstance(blocked_list, list) and blocked_list:
        for item in blocked_list[:5]:
            if isinstance(item, dict):
                num = item.get("number", "?")
                title = item.get("title", "?")
                if len(title) > 50:
                    title = title[:47] + "..."
                lines.append(f"  #{num}: {title}")
        if len(blocked_list) > 5:
            lines.append(f"  ... and {len(blocked_list) - 5} more")
    elif blocked_list == "skipped":
        lines.append("  (API skipped)")
    return lines


def _summary_blocked_missing_reason(metrics: dict) -> list[str]:
    """Summary for blocked_missing_reason flag."""
    missing = metrics.get("blocked_missing_reason", [])
    lines: list[str] = []
    if isinstance(missing, list) and missing:
        nums = [f"#{n}" for n in missing[:5]]
        lines.append(f"Issues: {', '.join(nums)}")
        if len(missing) > 5:
            lines.append(f"  ... and {len(missing) - 5} more")
    return lines


def _summary_stale_blockers(metrics: dict) -> list[str]:
    """Summary for stale_blockers flag."""
    stale = metrics.get("stale_blockers", [])
    lines: list[str] = []
    if isinstance(stale, list) and stale:
        for item in stale[:5]:
            num = item.get("number", 0)
            blockers = item.get("stale_blockers", [])
            blocker_str = ", ".join(f"#{b}" for b in blockers[:3])
            if len(blockers) > 3:
                blocker_str += f" +{len(blockers) - 3} more"
            lines.append(f"  #{num}: blocked by closed {blocker_str}")
        if len(stale) > 5:
            lines.append(f"  ... and {len(stale) - 5} more")
    elif stale == "skipped":
        lines.append("  (API skipped)")
    return lines


def _summary_long_blocked(metrics: dict) -> list[str]:
    """Summary for long_blocked flag."""
    long_blocked = metrics.get("long_blocked", [])
    lines: list[str] = []
    if isinstance(long_blocked, list) and long_blocked:
        for item in long_blocked[:5]:
            num = item.get("number", 0)
            days = item.get("days_blocked", 0)
            title = item.get("title", "")
            if len(title) > 40:
                title = title[:37] + "..."
            lines.append(f"  #{num}: {days}d - {title}")
        if len(long_blocked) > 5:
            lines.append(f"  ... and {len(long_blocked) - 5} more")
    elif long_blocked == "skipped":
        lines.append("  (API skipped)")
    return lines


def _summary_large_files(metrics: dict) -> list[str]:
    """Summary for large_files flag.

    Shows tier information (notice/warning) when available (Part of #2358).
    """
    large = metrics.get("large_files", [])
    lines: list[str] = []

    # Get thresholds for display (use defaults if not in THRESHOLDS)
    notice_threshold = THRESHOLDS.get("max_file_lines", 1000)
    warning_threshold = THRESHOLDS.get("max_file_lines_warning", 5000)

    # Count by tier if tier info present
    warnings = [f for f in large if isinstance(f, dict) and f.get("tier") == "warning"]
    notices = [f for f in large if isinstance(f, dict) and f.get("tier") == "notice"]
    has_tiers = warnings or notices

    if has_tiers:
        if warnings:
            lines.append(f"Warning ({len(warnings)} files >{warning_threshold} lines):")
            for f in warnings[:2]:
                path = f.get("file", "?")
                loc = f.get("lines", "?")
                lines.append(f"  {path}: {loc} lines")
            if len(warnings) > 2:
                lines.append(f"  ... and {len(warnings) - 2} more")
        if notices:
            lines.append(f"Notice ({len(notices)} files >{notice_threshold} lines):")
            for f in notices[:2]:
                path = f.get("file", "?")
                loc = f.get("lines", "?")
                lines.append(f"  {path}: {loc} lines")
            if len(notices) > 2:
                lines.append(f"  ... and {len(notices) - 2} more")
    else:
        # Legacy format without tiers
        for f in large[:3]:
            if isinstance(f, dict):
                path = f.get("file", "?")
                loc = f.get("lines", "?")
                lines.append(f"  {path}: {loc} lines")
            else:
                lines.append(f"  {f}")
        if len(large) > 3:
            lines.append(f"  ... and {len(large) - 3} more")
    return lines


def _summary_gh_error(metrics: dict) -> list[str]:
    """Summary for gh_error flag."""
    issues = metrics.get("issues", {})
    lines: list[str] = []
    if isinstance(issues, dict) and issues.get("error"):
        lines.append(f"Error: {issues.get('error')}")
    lines.append("See: docs/troubleshooting.md#github-issues")
    return lines


def _memory_severity(metrics: dict) -> str:
    """Determine memory severity level based on thresholds."""
    system = metrics.get("system", {})
    memory = system.get("memory", {})
    mem_percent = memory.get("percent_used", 0)
    if mem_percent >= THRESHOLDS["memory_critical_percent"]:
        return "critical"
    elif mem_percent >= THRESHOLDS["memory_warning_percent"]:
        return "warning"
    return "ok"


def _summary_memory(metrics: dict) -> list[str]:
    """Summary for consolidated memory flag (Part of #2379).

    Includes severity level in output for single-flag consolidation.
    """
    system = metrics.get("system", {})
    memory = system.get("memory", {})
    severity = _memory_severity(metrics)
    return [f"Severity: {severity}", f"Used: {memory.get('percent_used', '?')}%"]


def _disk_severity(metrics: dict) -> str:
    """Determine disk severity level based on thresholds."""
    system = metrics.get("system", {})
    disk = system.get("disk", {})
    disk_percent_str = disk.get("percent_used", "0%")
    try:
        disk_percent = int(disk_percent_str.rstrip("%"))
        if disk_percent >= THRESHOLDS["disk_critical_percent"]:
            return "critical"
        elif disk_percent >= THRESHOLDS["disk_warning_percent"]:
            return "warning"
    except ValueError:
        pass
    return "ok"


def _summary_disk(metrics: dict) -> list[str]:
    """Summary for consolidated disk flag (Part of #2379).

    Includes severity level in output for single-flag consolidation.
    """
    system = metrics.get("system", {})
    disk = system.get("disk", {})
    severity = _disk_severity(metrics)
    return [f"Severity: {severity}", f"Used: {disk.get('percent_used', '?')}"]


def _summary_disk_bloat(metrics: dict) -> list[str]:
    """Summary for disk_bloat flag."""
    system = metrics.get("system", {})
    bloat = system.get("disk_bloat", {})
    lines: list[str] = []
    large_files = bloat.get("large_files", [])
    if large_files:
        lines.append(f"Large files (>{THRESHOLDS.get('large_file_size_gb', 1)}GB):")
        lines.extend(
            f"  {f.get('file')}: {f.get('size_gb')}GB" for f in large_files[:3]
        )
        if len(large_files) > 3:
            lines.append(f"  ... and {len(large_files) - 3} more")
    if tests_gb := bloat.get("tests_size_gb"):
        lines.append(f"tests/ directory: {tests_gb}GB")
    if reports_mb := bloat.get("reports_size_mb"):
        lines.append(f"reports/ directory: {reports_mb}MB")
        lines.append("  Run: ./ai_template_scripts/cleanup_old_reports.py --delete")
    artifact_files = bloat.get("artifact_files", [])
    if artifact_files:
        lines.append("Large artifacts (>100MB):")
        lines.extend(
            f"  {f.get('file')}: {f.get('size_mb')}MB" for f in artifact_files[:3]
        )
        if len(artifact_files) > 3:
            lines.append(f"  ... and {len(artifact_files) - 3} more")
    states_dirs = bloat.get("states_dirs", [])
    if states_dirs:
        lines.append(
            f"TLA+ states dirs (>{THRESHOLDS.get('states_dir_size_gb', 10)}GB):"
        )
        lines.extend(
            f"  {d.get('path')}: {d.get('size_gb')}GB" for d in states_dirs[:3]
        )
        if len(states_dirs) > 3:
            lines.append(f"  ... and {len(states_dirs) - 3} more")
    return lines


def _summary_failures(metrics: dict) -> list[str]:
    """Summary for failures flag. Part of #2311."""
    crashes = metrics.get("crashes_24h", {})
    lines: list[str] = []
    if isinstance(crashes, dict):
        lines.append(
            f"Real: {crashes.get('real', 0)}, "
            f"Stale: {crashes.get('stale_connection', 0)}, "
            f"Idle: {crashes.get('idle_aborts', 0)}"
        )
    else:
        lines.append(f"Count: {crashes}")
    lines.append("See: docs/troubleshooting.md#session-issues")
    return lines


def _summary_untraceable_failures(metrics: dict) -> list[str]:
    """Summary for untraceable_failures flag."""
    failures = metrics.get("untraceable_failures", [])
    lines = [f"Count: {len(failures)}"]
    max_shown = 3
    for f in failures[:max_shown]:
        if not isinstance(f, dict):
            continue
        session_id = f.get("session_id", "?")
        iteration = f.get("iteration", "?")
        exit_code = f.get("exit_code", "?")
        log_info = f.get("log_info", "")
        line = f"  session={session_id} iter={iteration} exit={exit_code}"
        if log_info:
            line += f" ({log_info})"
        lines.append(line)
    if len(failures) > max_shown:
        lines.append(f"  ... and {len(failures) - max_shown} more")
    return lines


def _summary_unclaimed_sessions(metrics: dict) -> list[str]:
    """Summary for unclaimed_sessions flag. Part of #1020."""
    active = metrics.get("active_sessions", 0)
    details = metrics.get("active_session_details", [])
    in_progress = metrics.get("issues", {}).get("in_progress", 0)
    lines = [f"Active sessions: {active}, in-progress issues: {in_progress}"]
    max_shown = 3
    lines.extend(
        f"  {s.get('role', '?')}: pid {s.get('pid', '?')}"
        for s in details[:max_shown]
        if isinstance(s, dict)
    )
    if len(details) > max_shown:
        lines.append(f"  ... and {len(details) - max_shown} more")
    return lines


def _summary_stuck_process(metrics: dict) -> list[str]:
    """Summary for stuck_process flag."""
    procs = metrics.get("long_running_processes", [])
    lines: list[str] = []
    max_shown = 2
    for p in procs[:max_shown]:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "?")
        runtime = p.get("runtime_minutes", "?")
        repo = p.get("repo")
        if repo:
            lines.append(f"  {name}: {runtime}m (repo: {repo})")
        else:
            lines.append(f"  {name}: {runtime}m")
    if len(procs) > max_shown:
        lines.append(f"  ... and {len(procs) - max_shown} more")
    lines.append("See: docs/troubleshooting.md#build-issues")
    return lines


def _summary_forbidden_ci(metrics: dict) -> list[str]:
    """Summary for forbidden_ci flag."""
    ci_files = metrics.get("forbidden_ci", [])
    lines: list[str] = []
    if isinstance(ci_files, list) and ci_files:
        max_shown = 2
        lines.append(f"Files: {len(ci_files)}")
        lines.extend(f"  {f}" for f in ci_files[:max_shown])
        if len(ci_files) > max_shown:
            lines.append(f"  ... and {len(ci_files) - max_shown} more")
    return lines


def _summary_orphaned_tests(metrics: dict) -> list[str]:
    """Summary for orphaned_tests flag."""
    tests = metrics.get("tests", {})
    orphaned = tests.get("orphaned_tests", [])
    if isinstance(orphaned, list):
        return [f"Count: {len(orphaned)}"]
    return []


def _summary_no_work(metrics: dict) -> list[str]:
    """Summary for no_work flag."""
    return ["No open issues found"]


def _summary_low_proof_coverage(metrics: dict) -> list[str]:
    """Summary for low_proof_coverage flag."""
    proofs = metrics.get("proofs", {})
    kani = proofs.get("kani", {})
    return [f"Coverage: {kani.get('coverage_pct', 0)}%"]


def _summary_kani_unverified(metrics: dict) -> list[str]:
    """Summary for kani_unverified flag. Part of #2263."""
    proofs = metrics.get("proofs", {})
    kani = proofs.get("kani", {})
    total = kani.get("proofs", 0)
    unexecuted = kani.get("unexecuted", 0)
    timeout = kani.get("timeout", 0)
    oom = kani.get("oom", 0)
    passing = kani.get("passing", 0)
    failing = kani.get("failing", 0)
    unverified = unexecuted + timeout + oom
    unverified_pct = (unverified / total * 100) if total > 0 else 0
    lines = [f"Unverified: {unverified}/{total} ({unverified_pct:.0f}%)"]
    if unexecuted > 0:
        lines.append(f"  not_run: {unexecuted}")
    if timeout > 0:
        lines.append(f"  timeout: {timeout}")
    if oom > 0:
        lines.append(f"  oom: {oom}")
    lines.append(f"Passing: {passing}, Failing: {failing}")
    lines.append("Fix: python3 -m ai_template_scripts.kani_runner --filter not_run")
    return lines


def _summary_repo_dirty(metrics: dict) -> list[str]:
    """Summary for repo_dirty_during_pulse flag."""
    git = metrics.get("git", {})
    return [f"Branch: {git.get('branch', '?')}"]


def _summary_doc_claim_drift(metrics: dict) -> list[str]:
    """Summary for doc_claim_drift flag."""
    doc_claims = metrics.get("doc_claims", {})
    unverified = doc_claims.get("unverified_claims", [])
    lines = [f"Unverified: {doc_claims.get('unverified', 0)}"]
    for claim in unverified[:3]:
        if isinstance(claim, dict):
            name = claim.get("name", "?")
            reason = claim.get("reason", "?")
            if len(reason) > 40:
                reason = reason[:37] + "..."
            lines.append(f"  {name}: {reason}")
    if len(unverified) > 3:
        lines.append(f"  ... and {len(unverified) - 3} more")
    return lines


def _summary_deps_outdated(metrics: dict) -> list[str]:
    """Summary for deps_outdated flag. Part of #1553, #1872, #1876."""
    git_deps = metrics.get("git_deps", {})
    outdated = git_deps.get("outdated", [])
    checked = git_deps.get("checked", 0)
    lines = [f"Outdated: {len(outdated)} deps ({checked} refs checked)"]
    max_shown = 3
    for dep in outdated[:max_shown]:
        if isinstance(dep, dict):
            repo = dep.get("repo", "?")
            current = dep.get("current_rev", "?")[:8]
            head = dep.get("head_rev", "?")[:8]
            files_count = dep.get("files_count", 1)
            days = dep.get("days_old", 0)
            commits = dep.get("commits_behind", 0)
            info_parts = []
            if days:
                info_parts.append(f"{days}d old")
            if commits:
                info_parts.append(f"{commits} commits behind")
            if files_count > 1:
                info_parts.append(f"{files_count} files")
            info_str = f" ({', '.join(info_parts)})" if info_parts else ""
            lines.append(f"  {repo}: {current} -> {head}{info_str}")
    if len(outdated) > max_shown:
        lines.append(f"  ... and {len(outdated) - max_shown} more")
    lines.append("Fix: ./ai_template_scripts/bump_git_dep_rev.sh <repo_url>")
    return lines


def _summary_git_operation(metrics: dict) -> list[str]:
    """Summary for git_operation_in_progress flag. Part of #1899."""
    git = metrics.get("git", {})
    op = git.get("operation_in_progress", "unknown")
    lines = [f"Operation: {op}"]
    if op == "bisect":
        lines.append("Fix: git bisect reset")
    else:
        lines.append(f"Fix: git {op} --abort")
    lines.append("Commits and pushes are blocked until resolved")
    return lines


def _summary_high_reopen_rate(metrics: dict) -> list[str]:
    """Summary for high_reopen_rate flag. Part of #1937."""
    issues_reopened = metrics.get("issues_reopened", {})
    reopen_rate = issues_reopened.get("reopen_rate", 0)
    reopened = issues_reopened.get("reopened", 0)
    closed = issues_reopened.get("closed", 0)
    return [
        f"Rate: {reopen_rate:.1f}% ({reopened}/{closed} issues reopened)",
        "Indicates issues being closed prematurely",
    ]


def _summary_complexity_increase(metrics: dict) -> list[str]:
    """Summary for complexity_increase flag. Part of #2129."""
    complexity = metrics.get("complexity", {})
    delta = complexity.get("delta", {})
    delta_pct = delta.get("delta_percent", 0)
    prev_avg = delta.get("previous_avg", 0)
    curr_avg = delta.get("current_avg", 0)
    return [
        f"Increase: {delta_pct:.1f}% ({prev_avg:.2f} -> {curr_avg:.2f})",
        "Review recent changes for unnecessary complexity",
    ]


def _summary_gh_quota_overflow(metrics: dict) -> list[str]:
    """Summary for gh_quota_overflow flag."""
    overflow = metrics.get("gh_quota_overflow", {})
    recent_count = overflow.get("recent_count", 0)
    events = overflow.get("recent_events", [])
    lines = [f"Overflows in last hour: {recent_count}"]
    for event in events[:3]:
        if isinstance(event, dict):
            from_app = event.get("from_app", "?")
            to_app = event.get("to_app", "?")
            resource = event.get("resource", "?")
            lines.append(f"  {from_app} -> {to_app} ({resource})")
    if len(events) > 3:
        lines.append(f"  ... and {len(events) - 3} more")
    lines.append("Primary app quotas exhausted, using fallbacks")
    return lines


def _summary_startup_warnings(metrics: dict) -> list[str]:
    """Summary for startup_warnings flag. Part of #2459."""
    startup = metrics.get("startup_warnings", {})
    warnings = startup.get("warnings", [])
    count = startup.get("count", len(warnings))
    lines = [f"Count: {count}"]
    for w in warnings[:3]:
        if len(w) > 60:
            w = w[:57] + "..."
        lines.append(f"  {w}")
    if count > 3:
        lines.append(f"  ... and {count - 3} more")
    lines.append("Fix: Check .flags/startup_warnings and role files")
    return lines


# Dispatch table mapping flag names to handler functions
_FLAG_SUMMARY_HANDLERS: dict[str, Callable[[dict], list[str]]] = {
    "test_failures": _summary_test_failures,
    "blocked_issues": _summary_blocked_issues,
    "blocked_missing_reason": _summary_blocked_missing_reason,
    "stale_blockers": _summary_stale_blockers,
    "long_blocked": _summary_long_blocked,
    "large_files": _summary_large_files,
    "gh_error": _summary_gh_error,
    # Consolidated flags with severity (Part of #2379)
    "memory": _summary_memory,  # Replaces memory_warning, memory_critical
    "disk": _summary_disk,  # Replaces disk_warning, disk_critical
    "disk_bloat": _summary_disk_bloat,
    "failures": _summary_failures,
    "crashes": _summary_failures,  # Legacy alias
    "untraceable_failures": _summary_untraceable_failures,
    "unclaimed_sessions": _summary_unclaimed_sessions,
    "stuck_process": _summary_stuck_process,
    "forbidden_ci": _summary_forbidden_ci,
    "orphaned_tests": _summary_orphaned_tests,
    "no_work": _summary_no_work,
    "low_proof_coverage": _summary_low_proof_coverage,
    "kani_unverified": _summary_kani_unverified,
    "repo_dirty_during_pulse": _summary_repo_dirty,
    "doc_claim_drift": _summary_doc_claim_drift,
    "deps_outdated": _summary_deps_outdated,
    "git_operation_in_progress": _summary_git_operation,
    "high_reopen_rate": _summary_high_reopen_rate,
    "complexity_increase": _summary_complexity_increase,
    "gh_quota_overflow": _summary_gh_quota_overflow,
    "startup_warnings": _summary_startup_warnings,  # Per #2459
}


def _flag_summary(flag: str, metrics: dict) -> str:
    """Generate actionable summary for a flag (#952).

    Returns lines after timestamp with context about what triggered the flag.
    Summaries are kept short (<=5 lines) to stay actionable.

    Uses dispatch table pattern to reduce cyclomatic complexity (Part of #2333).
    """
    handler = _FLAG_SUMMARY_HANDLERS.get(flag)
    if handler:
        lines = handler(metrics)
        return "\n".join(lines)
    return ""


def write_flags(flags: list[str], metrics: dict | None = None) -> None:
    """Write flag files with atomic writes and actionable summaries (#952, #1000).

    Uses atomic write (tmp file + rename) to prevent partial writes from
    concurrent processes. Includes actionable details when metrics are provided.

    Args:
        flags: List of flag names to write.
        metrics: Optional metrics dict for generating actionable summaries.

    REQUIRES: flags is a list of strings (may be empty)
    ENSURES: Creates .flags/ directory if not exists
    ENSURES: Clears existing pulse-managed flag files (preserves looper-managed flags)
    ENSURES: Each flag written atomically (tmp + rename)
    ENSURES: Flag files contain timestamp, count/severity, and see-reference
    """
    FLAGS_DIR.mkdir(exist_ok=True)

    # Clear old flags including directories, but skip:
    # - .tmp files (#1157) - in-flight atomic writes from other processes
    # - looper-managed flags (#2409) - need to persist until consumed by looper
    for f in FLAGS_DIR.glob("*"):
        try:
            if f.is_dir():
                shutil.rmtree(f)
            elif f.suffix == ".tmp":
                # Skip .tmp files - they belong to other processes
                continue
            elif f.name.startswith(LOOPER_MANAGED_PREFIXES):
                # Skip looper-managed flags - not pulse's responsibility
                continue
            else:
                f.unlink(missing_ok=True)
        except OSError:
            pass  # Best effort cleanup, race with other processes

    # Write new flags atomically with trailing newline
    for flag in flags:
        # Sanitize flag name to prevent path traversal
        safe_flag = flag.replace("/", "_").replace("\\", "_").replace("..", "__")
        if safe_flag != flag:
            print(
                f"Warning: Sanitized flag name '{flag}' -> '{safe_flag}'",
                file=sys.stderr,
            )
        flag_path = FLAGS_DIR / safe_flag
        tmp_path = flag_path.with_suffix(".tmp")

        # Build content: timestamp + optional actionable summary
        content_lines = [datetime.now().isoformat()]
        if metrics:
            summary = _flag_summary(flag, metrics)
            if summary:
                content_lines.append(summary)
        content = "\n".join(content_lines) + "\n"

        try:
            tmp_path.write_text(content)
            tmp_path.rename(flag_path)
        except OSError as e:
            print(f"Warning: Failed to write flag '{safe_flag}': {e}", file=sys.stderr)
            # Clean up orphaned tmp file if possible
            tmp_path.unlink(missing_ok=True)


__all__ = [
    # Public functions
    "check_thresholds",
    "write_flags",
    # Constants (exported for testing/documentation)
    "LOOPER_MANAGED_PREFIXES",
    # Internal functions (exported for testing)
    "_flag_summary",
]

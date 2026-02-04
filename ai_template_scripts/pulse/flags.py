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
from datetime import datetime

from .config import THRESHOLDS
from .constants import FLAGS_DIR
from .issue_metrics import _unpack_issue_counts


def check_thresholds(metrics: dict) -> list[str]:
    """Check thresholds, return list of triggered flags.

    REQUIRES: metrics is a dict (may be empty or partial)
    ENSURES: Returns list of flag name strings
    ENSURES: Each flag is a valid flag name from THRESHOLDS
    ENSURES: Never raises (handles missing/invalid metrics gracefully)
    """
    flags = []

    # Large files
    if len(metrics.get("large_files", [])) >= THRESHOLDS["max_files_over_limit"]:
        flags.append("large_files")

    # Forbidden CI workflows (GitHub Actions not supported)
    if metrics.get("forbidden_ci"):
        flags.append("forbidden_ci")

    # Crashes - only flag "real" crashes, not stale_connection restarts
    # (stale_connection is expected after sleep/resume and auto-recovers)
    crashes_data = metrics.get("crashes_24h", {})
    if isinstance(crashes_data, dict):
        if crashes_data.get("real", 0) > 0:
            flags.append("crashes")
    elif crashes_data > 0:
        # Legacy: int value (shouldn't happen but handle gracefully)
        flags.append("crashes")

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

    # Memory usage
    memory = system.get("memory", {})
    mem_percent = memory.get("percent_used", 0)
    if mem_percent >= THRESHOLDS["memory_critical_percent"]:
        flags.append("memory_critical")
    elif mem_percent >= THRESHOLDS["memory_warning_percent"]:
        flags.append("memory_warning")

    # Disk usage
    disk = system.get("disk", {})
    disk_percent_str = disk.get("percent_used", "0%")
    try:
        disk_percent = int(disk_percent_str.rstrip("%"))
        if disk_percent >= THRESHOLDS["disk_critical_percent"]:
            flags.append("disk_critical")
        elif disk_percent >= THRESHOLDS["disk_warning_percent"]:
            flags.append("disk_warning")
    except ValueError:
        pass

    # Disk bloat detection (#1477)
    # Flag large files, bloated tests/, or generated artifacts
    disk_bloat = system.get("disk_bloat", {})
    if disk_bloat.get("bloat_detected"):
        flags.append("disk_bloat")

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

    return flags


def _flag_summary(flag: str, metrics: dict) -> str:
    """Generate actionable summary for a flag (#952).

    Returns lines after timestamp with context about what triggered the flag.
    Summaries are kept short (<=5 lines) to stay actionable.

    Supported flags:
        test_failures, blocked_issues, blocked_missing_reason, stale_blockers,
        long_blocked, large_files, gh_error, memory_warning, memory_critical,
        disk_warning, disk_critical, disk_bloat, crashes, untraceable_failures,
        unclaimed_sessions, stuck_process, forbidden_ci, orphaned_tests, no_work,
        low_proof_coverage, kani_unverified, repo_dirty_during_pulse, doc_claim_drift,
        deps_outdated, git_operation_in_progress, high_reopen_rate, complexity_increase
    """
    lines: list[str] = []

    if flag == "test_failures":
        tests = metrics.get("tests", {})
        recent = tests.get("recent_24h", {})
        failed = recent.get("failed", 0)
        total = recent.get("total", 0)
        lines.append(f"Failed: {failed}/{total} runs in 24h")

    elif flag == "blocked_issues":
        issues = metrics.get("issues", {})
        issue_counts, _ = _unpack_issue_counts(issues)
        blocked_list = metrics.get("blocked_issue_list", [])
        lines.append(f"Count: {issue_counts.get('blocked', 0)}")
        # List issue numbers and titles (per #1354)
        if isinstance(blocked_list, list) and blocked_list:
            for item in blocked_list[:5]:
                if isinstance(item, dict):
                    num = item.get("number", "?")
                    title = item.get("title", "?")
                    # Truncate long titles
                    if len(title) > 50:
                        title = title[:47] + "..."
                    lines.append(f"  #{num}: {title}")
            if len(blocked_list) > 5:
                lines.append(f"  ... and {len(blocked_list) - 5} more")
        elif blocked_list == "skipped":
            lines.append("  (API skipped)")

    elif flag == "blocked_missing_reason":
        missing = metrics.get("blocked_missing_reason", [])
        if isinstance(missing, list) and missing:
            nums = [f"#{n}" for n in missing[:5]]
            lines.append(f"Issues: {', '.join(nums)}")
            if len(missing) > 5:
                lines.append(f"  ... and {len(missing) - 5} more")

    elif flag == "stale_blockers":
        stale = metrics.get("stale_blockers", [])
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

    elif flag == "long_blocked":
        long_blocked = metrics.get("long_blocked", [])
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

    elif flag == "large_files":
        large = metrics.get("large_files", [])
        for f in large[:3]:
            if isinstance(f, dict):
                path = f.get("file", "?")
                loc = f.get("lines", "?")
                lines.append(f"  {path}: {loc} lines")
            else:
                lines.append(f"  {f}")
        if len(large) > 3:
            lines.append(f"  ... and {len(large) - 3} more")

    elif flag == "gh_error":
        issues = metrics.get("issues", {})
        if isinstance(issues, dict) and issues.get("error"):
            lines.append(f"Error: {issues.get('error')}")
        lines.append("See: docs/troubleshooting.md#github-issues")

    elif flag in ("memory_warning", "memory_critical"):
        system = metrics.get("system", {})
        memory = system.get("memory", {})
        lines.append(f"Used: {memory.get('percent_used', '?')}%")

    elif flag in ("disk_warning", "disk_critical"):
        system = metrics.get("system", {})
        disk = system.get("disk", {})
        lines.append(f"Used: {disk.get('percent_used', '?')}")

    elif flag == "disk_bloat":
        system = metrics.get("system", {})
        bloat = system.get("disk_bloat", {})
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

    elif flag == "crashes":
        # Part of #2311: Include idle_aborts in flag detail
        crashes = metrics.get("crashes_24h", {})
        if isinstance(crashes, dict):
            lines.append(
                f"Real: {crashes.get('real', 0)}, "
                f"Stale: {crashes.get('stale_connection', 0)}, "
                f"Idle: {crashes.get('idle_aborts', 0)}"
            )
        else:
            lines.append(f"Count: {crashes}")
        lines.append("See: docs/troubleshooting.md#session-issues")

    elif flag == "untraceable_failures":
        failures = metrics.get("untraceable_failures", [])
        lines.append(f"Count: {len(failures)}")
        # Exit codes: 0=success, 124=timeout, 125=silence-killed (no output, stale connection)
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

    elif flag == "unclaimed_sessions":
        # Show active sessions and their roles (#1020)
        active = metrics.get("active_sessions", 0)
        details = metrics.get("active_session_details", [])
        in_progress = metrics.get("issues", {}).get("in_progress", 0)
        lines.append(f"Active sessions: {active}, in-progress issues: {in_progress}")
        # List sessions by role
        max_shown = 3
        lines.extend(
            f"  {s.get('role', '?')}: pid {s.get('pid', '?')}"
            for s in details[:max_shown]
            if isinstance(s, dict)
        )
        if len(details) > max_shown:
            lines.append(f"  ... and {len(details) - max_shown} more")

    elif flag == "stuck_process":
        procs = metrics.get("long_running_processes", [])
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

    elif flag == "forbidden_ci":
        ci_files = metrics.get("forbidden_ci", [])
        if isinstance(ci_files, list) and ci_files:
            max_shown = 2
            lines.append(f"Files: {len(ci_files)}")
            lines.extend(f"  {f}" for f in ci_files[:max_shown])
            if len(ci_files) > max_shown:
                lines.append(f"  ... and {len(ci_files) - max_shown} more")

    elif flag == "orphaned_tests":
        tests = metrics.get("tests", {})
        orphaned = tests.get("orphaned_tests", [])
        if isinstance(orphaned, list):
            lines.append(f"Count: {len(orphaned)}")

    elif flag == "no_work":
        lines.append("No open issues found")

    elif flag == "low_proof_coverage":
        proofs = metrics.get("proofs", {})
        kani = proofs.get("kani", {})
        lines.append(f"Coverage: {kani.get('coverage_pct', 0)}%")

    elif flag == "kani_unverified":
        # Show Kani verification status (#2263)
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
        lines.append(f"Unverified: {unverified}/{total} ({unverified_pct:.0f}%)")
        if unexecuted > 0:
            lines.append(f"  not_run: {unexecuted}")
        if timeout > 0:
            lines.append(f"  timeout: {timeout}")
        if oom > 0:
            lines.append(f"  oom: {oom}")
        lines.append(f"Passing: {passing}, Failing: {failing}")
        lines.append("Fix: python3 -m ai_template_scripts.kani_runner --filter not_run")

    elif flag == "repo_dirty_during_pulse":
        git = metrics.get("git", {})
        lines.append(f"Branch: {git.get('branch', '?')}")

    elif flag == "doc_claim_drift":
        doc_claims = metrics.get("doc_claims", {})
        unverified = doc_claims.get("unverified_claims", [])
        lines.append(f"Unverified: {doc_claims.get('unverified', 0)}")
        for claim in unverified[:3]:
            if isinstance(claim, dict):
                name = claim.get("name", "?")
                reason = claim.get("reason", "?")
                # Truncate long reasons
                if len(reason) > 40:
                    reason = reason[:37] + "..."
                lines.append(f"  {name}: {reason}")
        if len(unverified) > 3:
            lines.append(f"  ... and {len(unverified) - 3} more")

    elif flag == "deps_outdated":
        # Show outdated git dependencies (#1553, #1872, #1876)
        git_deps = metrics.get("git_deps", {})
        outdated = git_deps.get("outdated", [])
        checked = git_deps.get("checked", 0)
        lines.append(f"Outdated: {len(outdated)} deps ({checked} refs checked)")
        max_shown = 3
        for dep in outdated[:max_shown]:
            if isinstance(dep, dict):
                repo = dep.get("repo", "?")
                current = dep.get("current_rev", "?")[:8]
                head = dep.get("head_rev", "?")[:8]
                files_count = dep.get("files_count", 1)
                days = dep.get("days_old", 0)
                commits = dep.get("commits_behind", 0)
                # Build info string with staleness details
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

    elif flag == "git_operation_in_progress":
        # Show git operation that is blocking commits/pushes (#1899)
        git = metrics.get("git", {})
        op = git.get("operation_in_progress", "unknown")
        lines.append(f"Operation: {op}")
        # bisect uses 'reset' instead of '--abort'
        if op == "bisect":
            lines.append("Fix: git bisect reset")
        else:
            lines.append(f"Fix: git {op} --abort")
        lines.append("Commits and pushes are blocked until resolved")

    elif flag == "high_reopen_rate":
        # Show reopen rate details (#1937)
        issues_reopened = metrics.get("issues_reopened", {})
        reopen_rate = issues_reopened.get("reopen_rate", 0)
        reopened = issues_reopened.get("reopened", 0)
        closed = issues_reopened.get("closed", 0)
        lines.append(f"Rate: {reopen_rate:.1f}% ({reopened}/{closed} issues reopened)")
        lines.append("Indicates issues being closed prematurely")

    elif flag == "complexity_increase":
        # Show complexity delta details (#2129)
        complexity = metrics.get("complexity", {})
        delta = complexity.get("delta", {})
        delta_pct = delta.get("delta_percent", 0)
        prev_avg = delta.get("previous_avg", 0)
        curr_avg = delta.get("current_avg", 0)
        lines.append(f"Increase: {delta_pct:.1f}% ({prev_avg:.2f} -> {curr_avg:.2f})")
        lines.append("Review recent changes for unnecessary complexity")

    return "\n".join(lines)


def write_flags(flags: list[str], metrics: dict | None = None) -> None:
    """Write flag files with atomic writes and actionable summaries (#952, #1000).

    Uses atomic write (tmp file + rename) to prevent partial writes from
    concurrent processes. Includes actionable details when metrics are provided.

    Args:
        flags: List of flag names to write.
        metrics: Optional metrics dict for generating actionable summaries.

    REQUIRES: flags is a list of strings (may be empty)
    ENSURES: Creates .flags/ directory if not exists
    ENSURES: Clears all existing flag files before writing new ones
    ENSURES: Each flag written atomically (tmp + rename)
    ENSURES: Flag files contain timestamp, count/severity, and see-reference
    """
    FLAGS_DIR.mkdir(exist_ok=True)

    # Clear old flags including directories, but skip .tmp files (#1157)
    # Other processes' .tmp files are in-flight atomic writes - let them complete
    # Stale .tmp files (orphaned by crashes) will persist but are harmless (small files)
    for f in FLAGS_DIR.glob("*"):
        try:
            if f.is_dir():
                shutil.rmtree(f)
            elif f.suffix != ".tmp":
                f.unlink(missing_ok=True)
            # Skip .tmp files - they belong to other processes
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
    # Internal functions (exported for testing)
    "_flag_summary",
]

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Core orchestration functions for pulse metrics collection.

Contains:
- collect_metrics: Main data collection function
- pulse_once, pulse_watch: Run modes
- metrics_to_broadcast: Output formatting
- main: CLI entry point

Part of #404: pulse.py module split
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

try:
    from ai_template_scripts.version import get_version
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.version import get_version

from .code_metrics import (
    count_lines_by_type,
    find_forbidden_ci,
    find_large_files,
    get_complexity_metrics,
    get_previous_metrics,
)
from .config import THRESHOLDS
from .constants import METRICS_RETENTION_DAYS
from .dirty_tracking import (
    _detect_dirty_file_changes,
    _format_dirty_diagnostics,
    _get_dirty_file_fingerprints,
    _get_dirty_source_files,
    _get_file_mtimes,
    _get_porcelain_status,
    _identify_file_modifiers,
    _snapshot_repo_root,
)
from .flags import check_thresholds, write_flags
from .git_metrics import (
    get_code_quality,
    get_consolidation_debt,
    get_doc_claim_status,
    get_git_status,
    get_outdated_git_deps,
)
from .issue_metrics import (
    _unpack_issue_counts,
    get_blocked_issue_list,
    get_blocked_missing_reason,
    get_issue_counts,
    get_issue_velocity,
    get_issues_reopened,
    get_long_blocked_issues,
    get_stale_blockers,
)
from .output import (
    _print_code_status,
    _print_complexity_status,
    _print_issue_status,
    _print_proof_status,
    _print_system_status,
    _print_test_status,
)
from .process_metrics import get_long_running_processes, get_recent_crashes
from .session_metrics import (
    get_active_session_details,
    get_untraceable_failures,
)
from .storage import (
    _compact_metrics_files,
    _rotate_old_metrics,
    _trim_current_metrics,
    get_repo_name,
    write_metrics,
)
from .system import get_system_resources
from .test_metrics import get_proof_coverage, get_test_coverage, get_test_status

# ============================================================================
# Core functions
# ============================================================================


def collect_metrics(
    fast: bool = False,
    skip_gh_api: bool = False,
    progress: bool = False,
    scan_root: Path | None = None,
    use_git: bool = True,
    log_root: Path | None = None,
    porcelain_lines: list[str] | None = None,
) -> dict:
    """Collect all metrics.

    Args:
        fast: If True, use faster estimation methods for expensive operations.
              Trades accuracy for speed on large repos.
        skip_gh_api: If True, skip GitHub API calls (for rate limit avoidance).
                     Affected fields: issues, blocked_missing_reason, velocity.
        progress: If True, emit progress messages to stderr during slow operations.
        scan_root: Directory to scan for file-based metrics (snapshot support).
        use_git: If True, use git-based scans when available.
        log_root: Directory to read logs from (defaults to scan_root).
        porcelain_lines: Optional pre-fetched git status --porcelain output lines.
                         Pass to avoid redundant git status calls (#1340).

    REQUIRES: scan_root is None or a valid directory Path
    REQUIRES: log_root is None or a valid directory Path
    ENSURES: Returns dict with keys: timestamp, repo, code, tests, issues, proofs, etc.
    ENSURES: timestamp is ISO 8601 format string
    ENSURES: Never raises (catches and logs errors internally)
    ENSURES: All sub-collectors run even if others fail
    """
    from ai_template_scripts.result import Result

    # Track timing for slow operation detection (#1358)
    _last_progress_time: list[float] = [
        time.monotonic()
    ]  # Use list for closure mutation

    def _progress(msg: str) -> None:
        """Emit progress message with timing if enabled.

        Shows elapsed time for previous operation if >5s.
        """
        if progress:
            now = time.monotonic()
            elapsed = now - _last_progress_time[0]
            if elapsed > 5:
                print(
                    f"  {msg} (prev took {elapsed:.1f}s)", file=sys.stderr, flush=True
                )
            else:
                print(f"  {msg}", file=sys.stderr, flush=True)
            _last_progress_time[0] = now

    # GitHub API dependent metrics (skip when rate limited)
    issues_result: Result[dict[str, int]]
    blocked_missing: list[int] | str
    blocked_list: list[dict] | str  # Per #1354 - list of blocked issues for flag
    stale_blockers: list[dict] | str  # Per #1497 - blocked issues with closed blockers
    long_blocked: (
        list[dict] | str
    )  # Per #1497 - blocked issues with no progress >7 days
    velocity: dict
    issues_reopened: dict  # Per #1937 - reopen metrics for closure quality
    gh_rate_limited = False  # Track rate limiting for later API calls (#1606)

    if skip_gh_api:
        issues_result = Result.skip("--skip-gh-api")
        blocked_missing = "skipped"
        blocked_list = "skipped"
        stale_blockers = "skipped"
        long_blocked = "skipped"
        velocity = {"skipped": True}
        issues_reopened = {"skipped": True}
    else:
        _progress("fetching: issue counts (GitHub API)...")
        issues_result = get_issue_counts()

        # Check if rate limited - skip remaining GitHub API calls (#1606)
        # This prevents hanging when the API is rate limited across all endpoints
        gh_rate_limited = bool(
            not issues_result.ok
            and issues_result.error
            and (
                "rate limit" in issues_result.error.lower()
                or "timeout" in issues_result.error.lower()
            )
        )

        if gh_rate_limited:
            _progress(
                f"warning: {issues_result.error} - skipping remaining GitHub API calls"
            )
            blocked_missing = "rate_limited"
            blocked_list = "rate_limited"
            stale_blockers = "rate_limited"
            long_blocked = "rate_limited"
            velocity = {"rate_limited": True, "error": issues_result.error}
            issues_reopened = {"rate_limited": True}
        else:
            _progress("fetching: blocked issues (GitHub API)...")
            blocked_missing = get_blocked_missing_reason()
            blocked_list = get_blocked_issue_list()
            _progress("fetching: stale blockers (GitHub API)...")
            stale_blockers = get_stale_blockers()
            _progress("fetching: long-blocked issues (GitHub API)...")
            long_blocked = get_long_blocked_issues(
                days_threshold=THRESHOLDS["stale_issue_days"]
            )
            _progress("fetching: issue velocity (GitHub API)...")
            velocity = get_issue_velocity()
            _progress("fetching: issues reopened (GitHub API)...")
            issues_reopened = get_issues_reopened()

    _progress("scanning: lines of code...")
    loc = count_lines_by_type(repo_root=scan_root)
    _progress("scanning: large files...")
    large_files = find_large_files(
        THRESHOLDS["max_file_lines"], repo_root=scan_root, use_git=use_git
    )
    _progress("scanning: CI workflows...")
    forbidden_ci = find_forbidden_ci(repo_root=scan_root)
    _progress("scanning: test status...")
    tests = get_test_status(repo_root=scan_root, use_git=use_git, log_root=log_root)
    _progress("scanning: test coverage...")
    if coverage := get_test_coverage(repo_root=scan_root):
        tests["coverage"] = coverage
    _progress("scanning: proof coverage...")
    proofs = get_proof_coverage(repo_root=scan_root)
    _progress("scanning: crashes...")
    crashes = get_recent_crashes()
    _progress("scanning: untraceable failures...")
    untraceable = get_untraceable_failures()
    _progress("scanning: system resources...")
    system_res = get_system_resources(fast=fast)
    _progress("scanning: git status...")
    git_info = get_git_status(porcelain_lines=porcelain_lines)
    _progress("scanning: active sessions...")
    session_details = get_active_session_details()
    sessions = len(session_details)
    _progress("scanning: long-running processes...")
    long_running = get_long_running_processes(repo_root=scan_root)
    _progress("scanning: code quality...")
    quality = get_code_quality(repo_root=scan_root)
    _progress("scanning: consolidation debt...")
    consolidation = get_consolidation_debt(repo_root=scan_root)
    _progress("scanning: doc claims...")
    doc_claims = get_doc_claim_status(repo_root=scan_root)

    # Git dependency updates - skip if GitHub API is being avoided or rate limited
    if skip_gh_api:
        git_deps = {"skipped": True}
    elif gh_rate_limited:
        git_deps = {"rate_limited": True}
    else:
        _progress("scanning: git dependency updates...")
        git_deps = get_outdated_git_deps(repo_root=scan_root)

    # Complexity metrics with delta tracking (Per #2129)
    _progress("scanning: code complexity...")
    complexity = get_complexity_metrics(repo_root=scan_root)

    # Calculate complexity delta from previous metrics
    complexity_delta = None
    if "error" not in complexity:
        prev_metrics = get_previous_metrics(days_back=1)
        if prev_metrics and "complexity" in prev_metrics:
            prev_complexity = prev_metrics["complexity"]
            # Only calculate delta if both current and previous have functions analyzed
            curr_funcs = complexity.get("total_functions", 0)
            prev_funcs = prev_complexity.get("total_functions", 0)
            if (
                "avg_complexity" in prev_complexity
                and prev_funcs > 0
                and curr_funcs > 0
            ):
                prev_avg = prev_complexity["avg_complexity"]
                curr_avg = complexity.get("avg_complexity", 0)
                if prev_avg > 0:
                    delta_pct = ((curr_avg - prev_avg) / prev_avg) * 100
                    complexity_delta = {
                        "previous_avg": prev_avg,
                        "current_avg": curr_avg,
                        "delta_percent": round(delta_pct, 1),
                        "trend": "increasing"
                        if delta_pct > 5
                        else "decreasing"
                        if delta_pct < -5
                        else "stable",
                    }

    if complexity_delta:
        complexity["delta"] = complexity_delta

    return {
        "timestamp": datetime.now().isoformat(),
        "repo": get_repo_name(),
        "loc": loc,
        "large_files": large_files,
        "forbidden_ci": forbidden_ci,
        "tests": tests,
        "proofs": proofs,
        "issues": issues_result.to_dict(),
        "blocked_missing_reason": blocked_missing,
        "blocked_issue_list": blocked_list,  # Per #1354
        "stale_blockers": stale_blockers,  # Per #1497
        "long_blocked": long_blocked,  # Per #1497
        "crashes_24h": crashes,
        "untraceable_failures": untraceable,
        "system": system_res,
        "git": git_info,
        "active_sessions": sessions,
        "active_session_details": session_details,
        "long_running_processes": long_running,
        "quality": quality,
        "velocity": velocity,
        "issues_reopened": issues_reopened,  # Per #1937
        "consolidation": consolidation,
        "doc_claims": doc_claims,
        "git_deps": git_deps,  # Per #1553
        "complexity": complexity,  # Per #2129
    }


def pulse_once(
    quiet: bool = False,
    fast: bool = False,
    progress: bool = False,
    skip_gh_api: bool = False,
    snapshot: bool = False,
) -> None:
    """Run pulse once. Set quiet=True for watch mode (no output).

    INVARIANT: Pulse is read-only - it must never modify source files.
    This is enforced by comparing dirty state before/after collection.

    Args:
        quiet: Suppress all output (for watch mode).
        fast: Use faster estimation methods for expensive operations.
        progress: Print progress messages while scanning.
        skip_gh_api: Skip GitHub API calls (for rate limit avoidance).
        snapshot: Scan git HEAD snapshot for stable metrics (#1329).

    REQUIRES: Current directory is within a git repository
    ENSURES: Writes metrics to metrics/ directory
    ENSURES: Writes flags to .flags/ directory
    ENSURES: Does not modify any source files (read-only invariant)
    ENSURES: Emits error and sets exit code 1 if source files are dirtied
    """
    snapshot_ctx = _snapshot_repo_root() if snapshot else nullcontext(None)
    snapshot_root = snapshot_ctx.__enter__()  # type: ignore[attr-defined]
    try:
        use_snapshot = snapshot_root is not None
        # Default to Path(".") so get_long_running_processes filters to current repo (#1409)
        scan_root = snapshot_root if use_snapshot else Path(".")
        log_root = Path(".") if use_snapshot else None

        # Capture dirty state before collection (read-only guard)
        # Run check even in snapshot mode to verify pulse doesn't modify tracked files (#1337)
        # Fetch porcelain once and reuse for both dirty check and collect_metrics (#1340)
        porcelain_before = _get_porcelain_status()
        dirty_before = _get_dirty_source_files(porcelain_lines=porcelain_before)
        # Fingerprint already-dirty files to detect modifications (#1339)
        fingerprints_before = _get_dirty_file_fingerprints(dirty_before)
        # Capture file mtimes for enhanced diagnostics (#1583)
        mtimes_before = _get_file_mtimes(dirty_before)

        # Always show brief startup message unless quiet (#956)
        # This prevents appearance of "hanging" during slow API/scan operations
        mode = "snapshot" if use_snapshot else "normal"
        if not quiet:
            print(f"pulse: collecting metrics ({mode})...", file=sys.stderr, flush=True)

        # Always enable progress output unless quiet (#1249)
        # This ensures pulse never goes >60s without output, preventing looper timeouts
        # Pass porcelain_before to collect_metrics to reuse for get_git_status (#1340)
        metrics = collect_metrics(
            fast=fast,
            skip_gh_api=skip_gh_api,
            progress=not quiet,  # Always emit progress unless quiet
            scan_root=scan_root,
            use_git=not use_snapshot,
            log_root=log_root,
            porcelain_lines=porcelain_before,
        )
        flags = check_thresholds(metrics)

        write_metrics(metrics)

        # Detect if repo became dirty during pulse run (#1337)
        # This doesn't mean PULSE modified files - concurrent sessions commonly cause this
        # Check runs even in snapshot mode to verify pulse's read-only invariant
        porcelain_after = _get_porcelain_status()
        dirty_after = _get_dirty_source_files(porcelain_lines=porcelain_after)
        new_dirty = dirty_after - dirty_before
        # Also check if already-dirty files were modified (#1339)
        fingerprints_after = _get_dirty_file_fingerprints(dirty_before & dirty_after)
        modified_dirty = _detect_dirty_file_changes(
            fingerprints_before, fingerprints_after
        )
        all_modified = new_dirty | modified_dirty
        if all_modified:
            # Files were modified while pulse was running. Pulse itself is read-only,
            # so something external modified these files (rust-analyzer, formatters,
            # another AI session, etc.)
            # Capture mtimes after for diagnostics (#1583)
            mtimes_after = _get_file_mtimes(dirty_after)
            culprits = _identify_file_modifiers()

            # Use enhanced diagnostics helper (#1583)
            _format_dirty_diagnostics(
                new_dirty=new_dirty,
                modified_dirty=modified_dirty,
                culprits=culprits,
                mtimes_before=mtimes_before,
                mtimes_after=mtimes_after,
                porcelain_before=porcelain_before,
                porcelain_after=porcelain_after,
            )

            # Only flag if this is unexpected (no concurrent AI sessions or recent commits)
            # (#1241) Don't flag in multi-session mode - dirty repo is expected
            # (#1073) Don't flag if recent commit detected - worker likely just finished
            culprits_str = " ".join(culprits).lower()
            expected_dirty = (
                "other ai sessions" in culprits_str
                or "recent git commit" in culprits_str
            )
            if not culprits or not expected_dirty:
                if "repo_dirty_during_pulse" not in flags:
                    flags.append("repo_dirty_during_pulse")

        write_flags(flags, metrics)

        if quiet:
            return

        # Header
        ts = datetime.now().strftime("%H:%M:%S")
        git = metrics.get("git", {})
        dirty = "dirty" if git.get("dirty") else "clean"
        print(f"## Pulse {ts}")
        print(f"**Git:** {git.get('branch', '?')}@{git.get('head', '?')} ({dirty})")
        print()

        # Status sections
        _print_code_status(metrics)
        _print_complexity_status(metrics)
        _print_test_status(metrics)
        _print_proof_status(metrics)
        _print_issue_status(metrics)
        _print_system_status(metrics)

        # Flags
        print()
        if flags:
            print(f"**Flags:** {', '.join(flags)}")
        else:
            print("**Flags:** none")
    finally:
        snapshot_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]


def metrics_to_broadcast(metrics: dict) -> str:
    """Convert metrics to single-line broadcast format for org collection.

    Format: repo|branch|head|loc:N|tests:N|proofs:N|issues:N|mem:N%|disk:N%|flags:...

    REQUIRES: metrics is a dict (may be empty or partial)
    ENSURES: Returns pipe-delimited string with core metrics
    ENSURES: Never raises (uses '?' for missing values)
    """
    repo = metrics.get("repo", "unknown")
    git = metrics.get("git", {})
    branch = git.get("branch", "?")
    head = git.get("head", "?")

    loc_total = sum(metrics.get("loc", {}).values())
    tests = metrics.get("tests", {}).get("count", 0)

    # Sum all proof types
    proofs = metrics.get("proofs", {})
    proof_count = sum(
        [
            proofs.get("kani", {}).get("proofs", 0),
            proofs.get("tla2", {}).get("specs", 0),
            proofs.get("lean", {}).get("theorems", 0),
        ]
    )

    issue_counts, _ = _unpack_issue_counts(metrics.get("issues", {}))
    issues = issue_counts.get("open", 0)
    mem = metrics.get("system", {}).get("memory", {}).get("percent_used", 0)
    disk_str = metrics.get("system", {}).get("disk", {}).get("percent_used", "0%")

    flags = check_thresholds(metrics)
    flags_str = ",".join(flags) if flags else "none"

    return (
        f"{repo}|{branch}|{head}|loc:{loc_total}|tests:{tests}|"
        f"proofs:{proof_count}|issues:{issues}|mem:{mem}%|"
        f"disk:{disk_str}|flags:{flags_str}"
    )


def pulse_watch(
    interval: int = 300,
    skip_gh_api: bool = False,
    snapshot: bool = False,
) -> None:
    """Run pulse continuously, writing metrics silently.

    INVARIANT: Pulse is read-only - it must never modify source files.
    This is enforced by comparing dirty state before/after each collection.

    Args:
        interval: Seconds between pulse runs.
        skip_gh_api: If True, skip GitHub API calls (for rate limit avoidance).
        snapshot: Scan git HEAD snapshot for stable metrics (#1329).

    REQUIRES: interval > 0
    REQUIRES: Current directory is within a git repository
    ENSURES: Runs indefinitely until interrupted
    ENSURES: Does not modify any source files (read-only invariant)
    """
    while True:
        snapshot_ctx = _snapshot_repo_root() if snapshot else nullcontext(None)
        snapshot_root = snapshot_ctx.__enter__()  # type: ignore[attr-defined]
        try:
            use_snapshot = snapshot_root is not None
            # Default to Path(".") so get_long_running_processes filters to current repo (#1409)
            scan_root = snapshot_root if use_snapshot else Path(".")
            log_root = Path(".") if use_snapshot else None

            # Run check even in snapshot mode (#1337)
            # Fetch porcelain once and reuse (#1340)
            porcelain_before = _get_porcelain_status()
            dirty_before = _get_dirty_source_files(porcelain_lines=porcelain_before)
            # Fingerprint already-dirty files to detect modifications (#1339)
            fingerprints_before = _get_dirty_file_fingerprints(dirty_before)
            # Capture file mtimes for enhanced diagnostics (#1583)
            mtimes_before = _get_file_mtimes(dirty_before)
            # Emit progress in watch mode to prevent >60s silence (#1249)
            print("pulse: watch cycle starting...", file=sys.stderr, flush=True)
            metrics = collect_metrics(
                skip_gh_api=skip_gh_api,
                progress=True,
                scan_root=scan_root,
                use_git=not use_snapshot,
                log_root=log_root,
                porcelain_lines=porcelain_before,
            )
            flags = check_thresholds(metrics)
            write_metrics(metrics)
            # Detect if repo became dirty during pulse (#1241, #1337)
            porcelain_after = _get_porcelain_status()
            dirty_after = _get_dirty_source_files(porcelain_lines=porcelain_after)
            new_dirty = dirty_after - dirty_before
            # Also check if already-dirty files were modified (#1339)
            fingerprints_after = _get_dirty_file_fingerprints(
                dirty_before & dirty_after
            )
            modified_dirty = _detect_dirty_file_changes(
                fingerprints_before, fingerprints_after
            )
            all_modified = new_dirty | modified_dirty
            if all_modified:
                # Capture mtimes after for diagnostics (#1583)
                mtimes_after = _get_file_mtimes(dirty_after)
                culprits = _identify_file_modifiers()

                # Use enhanced diagnostics helper (#1583)
                _format_dirty_diagnostics(
                    new_dirty=new_dirty,
                    modified_dirty=modified_dirty,
                    culprits=culprits,
                    mtimes_before=mtimes_before,
                    mtimes_after=mtimes_after,
                    porcelain_before=porcelain_before,
                    porcelain_after=porcelain_after,
                )

                # Only flag if this is unexpected (no concurrent AI sessions or recent commits)
                # (#1073) Don't flag if recent commit detected - worker likely just finished
                culprits_str = " ".join(culprits).lower()
                expected_dirty = (
                    "other ai sessions" in culprits_str
                    or "recent git commit" in culprits_str
                )
                if not culprits or not expected_dirty:
                    if "repo_dirty_during_pulse" not in flags:
                        flags.append("repo_dirty_during_pulse")
            write_flags(flags, metrics)
        finally:
            snapshot_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
        time.sleep(interval)


def main() -> None:
    """CLI entry point for pulse metrics collection.

    Modes:
      - Default: One-shot metrics collection with formatted output
      - --watch: Continuous collection at interval, writes to metrics/
      - --broadcast: Single-line JSON for org-wide aggregation
      - --rotate: Archive old metrics, trim today's file
      - --fast: Use faster estimation methods (trades accuracy for speed)
      - --progress: Show progress during long scans
      - --snapshot: Scan git snapshot of HEAD for stable file metrics

    REQUIRES: None (parses sys.argv)
    ENSURES: Exits with code 0 on success, 1 on error
    """
    parser = argparse.ArgumentParser(
        description="Collect project metrics and detect issues"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("pulse.py"),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously (silent, writes to metrics/)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Watch interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--broadcast", action="store_true", help="Output single line for org collection"
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Archive old metrics files and trim today's to max entries",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Convert all metrics files to compact JSON format",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster estimation methods (trades accuracy for speed on large repos)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress messages during scanning",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Scan a git snapshot of HEAD for stable file metrics",
    )
    parser.add_argument(
        "--skip-gh-api",
        action="store_true",
        help="Skip GitHub API calls (for rate limit avoidance)",
    )
    args = parser.parse_args()

    if args.compact:
        _compact_metrics_files()
    elif args.rotate:
        print(f"Rotating metrics (retention: {METRICS_RETENTION_DAYS} days)")
        _rotate_old_metrics()
        _trim_current_metrics()
    elif args.broadcast:
        snapshot_ctx = _snapshot_repo_root() if args.snapshot else nullcontext(None)
        snapshot_root = snapshot_ctx.__enter__()  # type: ignore[attr-defined]
        try:
            use_snapshot = snapshot_root is not None
            # Default to Path(".") so get_long_running_processes filters to current repo (#1409)
            scan_root = snapshot_root if use_snapshot else Path(".")
            log_root = Path(".") if use_snapshot else None

            # Run check even in snapshot mode (#1337)
            # Fetch porcelain once and reuse (#1340)
            porcelain_before = _get_porcelain_status()
            dirty_before = _get_dirty_source_files(porcelain_lines=porcelain_before)
            # Fingerprint already-dirty files to detect modifications (#1339)
            fingerprints_before = _get_dirty_file_fingerprints(dirty_before)
            # Capture file mtimes for enhanced diagnostics (#1583)
            mtimes_before = _get_file_mtimes(dirty_before)
            print(
                "pulse: collecting metrics (broadcast)...",
                file=sys.stderr,
                flush=True,
            )
            metrics = collect_metrics(
                fast=args.fast,
                skip_gh_api=args.skip_gh_api,
                progress=True,  # Always emit progress to prevent >60s silence (#1249)
                scan_root=scan_root,
                use_git=not use_snapshot,
                log_root=log_root,
                porcelain_lines=porcelain_before,
            )
            write_metrics(metrics)
            flags = check_thresholds(metrics)
            # Detect if repo became dirty during pulse run (#1337)
            porcelain_after = _get_porcelain_status()
            dirty_after = _get_dirty_source_files(porcelain_lines=porcelain_after)
            new_dirty = dirty_after - dirty_before
            # Also check if already-dirty files were modified (#1339)
            fingerprints_after = _get_dirty_file_fingerprints(
                dirty_before & dirty_after
            )
            modified_dirty = _detect_dirty_file_changes(
                fingerprints_before, fingerprints_after
            )
            all_modified = new_dirty | modified_dirty
            if all_modified:
                # Capture mtimes after for diagnostics (#1583)
                mtimes_after = _get_file_mtimes(dirty_after)
                culprits = _identify_file_modifiers()

                # Use enhanced diagnostics helper (#1583)
                _format_dirty_diagnostics(
                    new_dirty=new_dirty,
                    modified_dirty=modified_dirty,
                    culprits=culprits,
                    mtimes_before=mtimes_before,
                    mtimes_after=mtimes_after,
                    porcelain_before=porcelain_before,
                    porcelain_after=porcelain_after,
                )

                # Only flag if this is unexpected (no concurrent AI sessions or recent commits)
                # (#1073) Don't flag if recent commit detected - worker likely just finished
                culprits_str = " ".join(culprits).lower()
                expected_dirty = (
                    "other ai sessions" in culprits_str
                    or "recent git commit" in culprits_str
                )
                if not culprits or not expected_dirty:
                    if "repo_dirty_during_pulse" not in flags:
                        flags.append("repo_dirty_during_pulse")
            write_flags(flags, metrics)
            print(metrics_to_broadcast(metrics))
        finally:
            snapshot_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
    elif args.watch:
        pulse_watch(
            interval=args.interval,
            skip_gh_api=args.skip_gh_api,
            snapshot=args.snapshot,
        )
    else:
        pulse_once(
            fast=args.fast,
            progress=args.progress,
            skip_gh_api=args.skip_gh_api,
            snapshot=args.snapshot,
        )


if __name__ == "__main__":
    main()

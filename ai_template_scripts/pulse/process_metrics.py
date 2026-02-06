#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Process monitoring - detection of long-running and stuck processes.

Part of #404: pulse.py module split
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

from .config import THRESHOLDS
from .constants import (
    CRASH_LOG_PATTERN,
    LONG_RUNNING_EXCLUDE_PATTERNS,
    LONG_RUNNING_PROCESS_NAMES,
)

# Import shared crash categories per #2318
try:
    from ai_template_scripts.crash_categories import (
        CrashCategoryStr,
        CRASH_CATEGORIES,
        is_expected_termination,
    )
except ImportError:
    # Fallback for standalone usage
    CrashCategoryStr = str  # type: ignore
    CRASH_CATEGORIES = frozenset(
        ["idle_abort", "signal_kill", "timeout", "stale_connection", "exit_error", "unknown"]
    )

    def is_expected_termination(category: str) -> bool:
        return category in ("idle_abort", "stale_connection")

try:
    from ai_template_scripts.subprocess_utils import run_cmd
except ModuleNotFoundError:
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import run_cmd


def _args_contain_repo_path(args: str, repo_path: str) -> bool:
    """Check args for repo path with boundary checks to avoid substring matches.

    REQUIRES: args and repo_path are strings (may be empty)
    ENSURES: Returns True iff repo_path appears with boundary checks
    ENSURES: Never raises
    """
    if not args or not repo_path:
        return False
    prefix_ok = (" ", "\t", "\n", "=", "'", '"')
    suffix_ok = (" ", "\t", "\n", "/", "'", '"', ":")
    for match in re.finditer(re.escape(repo_path), args):
        start, end = match.span()
        if start > 0 and args[start - 1] not in prefix_ok:
            continue
        if end < len(args) and args[end] not in suffix_ok:
            continue
        return True
    return False


def _get_process_cwd(pid: int) -> Path | None:
    """Best-effort lookup of process cwd for repo scoping (#1409)."""
    if sys.platform.startswith("linux"):
        try:
            return Path(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            return None
    if sys.platform == "darwin":
        result = run_cmd(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], timeout=2)
        if not result.ok or not result.stdout.strip():
            return None
        for line in result.stdout.splitlines():
            if line.startswith("n"):
                path_str = line[1:].strip()
                if path_str:
                    return Path(path_str)
    return None


def _infer_repo_from_path(path: Path | None) -> str | None:
    """Infer git repo name from a path by finding .git directory (#1484).

    Walks up from the given path to find a git repository root,
    then returns the directory name as the repo identifier.

    Returns:
        Repo directory name (e.g., "kafka2", "ai_template") or None if not in a repo.
    """
    if path is None:
        return None
    try:
        current = path.resolve()
    except OSError:
        return None
    for _ in range(20):
        if (current / ".git").exists():
            return current.name
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _parse_etime(etime: str) -> int:
    """Parse ps etime format to minutes.

    Format: [[DD-]HH:]MM:SS
    Examples: "03:22" -> 3, "01:03:22" -> 63, "2-01:03:22" -> 2943
    """
    try:
        # Handle days
        days = 0
        if "-" in etime:
            days_part, etime = etime.split("-", 1)
            days = int(days_part)

        # Split time parts
        parts = etime.split(":")
        if len(parts) == 2:
            minutes, _ = int(parts[0]), int(parts[1])
            hours = 0
        elif len(parts) == 3:
            hours, minutes, _ = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            return 0

        return days * 24 * 60 + hours * 60 + minutes
    except (ValueError, IndexError):
        return 0


def get_long_running_processes(
    threshold_minutes: int | None = None,
    repo_root: Path | None = None,
) -> list[dict]:
    """Detect long-running processes that may be stuck (#922).

    Monitors known verification/compilation tools (cargo, cbmc, kani, rustc, z3, etc.)
    that may run for hours when stuck or on complex inputs.

    Args:
        threshold_minutes: Minimum runtime to report (default: from THRESHOLDS).
        repo_root: Only report processes whose command contains this repo path (#1409).
            If None, reports all matching processes (original global behavior).
            Falls back to process cwd when command lacks repo path.

    Returns:
        List of dicts with {pid, name, runtime_minutes, command} for processes
        exceeding the threshold.

    REQUIRES: threshold_minutes is None or >= 0
    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns list of dicts with pid, name, runtime_minutes, command, repo keys
    ENSURES: Returns empty list if ps command fails
    """
    if threshold_minutes is None:
        threshold_minutes = THRESHOLDS.get("long_running_process_minutes", 120)

    # Resolve repo_root to absolute path for matching (#1409)
    repo_paths: tuple[str, ...] = ()
    repo_root_resolved: Path | None = None
    if repo_root is not None:
        repo_root_resolved = repo_root.resolve()
        # Avoid relative path candidates like "." that can over-match args.
        candidates = {str(repo_root_resolved)}
        if repo_root.is_absolute():
            candidates.add(str(repo_root))
        repo_paths = tuple(path for path in candidates if path)

    long_running = []

    # Use ps to get process info: pid, elapsed time, command
    # etime format: [[DD-]HH:]MM:SS
    result = run_cmd(["ps", "-eo", "pid,etime,comm,args"])
    if not result.ok:
        return []

    for line in result.stdout.strip().split("\n")[1:]:  # Skip header
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue

        pid_str, etime, comm = parts[0], parts[1], parts[2]
        args = parts[3] if len(parts) > 3 else comm

        # Check if process name matches any monitored names
        # NOTE: comm is truncated to 16 chars on macOS, so use args for matching
        # args contains the full command with path, e.g., "/Users/.cargo/bin/cargo build"
        args_parts = args.split() if args else []
        args_first = args_parts[0] if args_parts else comm
        proc_name = Path(args_first).name.lower()
        if not any(name in proc_name for name in LONG_RUNNING_PROCESS_NAMES):
            continue

        # Skip excluded patterns (#1288, #2139) - mechanism for expected long-running processes
        # Currently empty (Python patterns removed in #2139), preserved for future use
        args_lower = args.lower()
        if any(
            pattern.lower() in args_lower for pattern in LONG_RUNNING_EXCLUDE_PATTERNS
        ):
            continue

        try:
            pid = int(pid_str)
        except ValueError:
            continue

        # Skip processes from other repos (#1409)
        # Only flag processes whose command contains the current repo path
        # Use path boundary check to avoid false positives (e.g., /foo matching /foo_backup)
        cwd: Path | None = None
        cwd_resolved: Path | None = None
        if repo_paths:
            matches_repo = False
            for repo_path_str in repo_paths:
                if _args_contain_repo_path(args, repo_path_str):
                    matches_repo = True
                    break
            if not matches_repo:
                cwd = _get_process_cwd(pid)
                if cwd is None or repo_root_resolved is None:
                    continue
                try:
                    cwd_resolved = cwd.resolve()
                except OSError:
                    continue
                if cwd_resolved == repo_root_resolved:
                    matches_repo = True
                elif repo_root_resolved in cwd_resolved.parents:
                    matches_repo = True
            if not matches_repo:
                continue

        # Parse elapsed time to minutes
        runtime_minutes = _parse_etime(etime)
        if runtime_minutes < threshold_minutes:
            continue

        # Infer repo from cwd for attribution (#1484)
        if cwd is None:
            cwd = _get_process_cwd(pid)
        if cwd is not None and cwd_resolved is None:
            try:
                cwd_resolved = cwd.resolve()
            except OSError:
                cwd_resolved = None
        if (
            repo_root_resolved is not None
            and cwd_resolved is not None
            and cwd_resolved != repo_root_resolved
            and repo_root_resolved not in cwd_resolved.parents
        ):
            continue
        proc_repo = _infer_repo_from_path(cwd)

        long_running.append(
            {
                "pid": pid,
                "name": comm,
                "runtime_minutes": runtime_minutes,
                "command": args[:200],  # Truncate long command lines
                "repo": proc_repo,  # Attributed repo from cwd (#1484)
            }
        )

    return long_running


def _classify_crash(message: str) -> CrashCategoryStr:
    """Classify a crash message into category.

    Categories from CRASH_CATEGORIES per #2318 unification.
    Same implementation as crash_analysis.crash_detector._get_crash_category().

    Part of #2311: Added idle_abort detection to match crash_analysis behavior.
    Part of #2318: Now uses shared type from crash_categories module.
    """
    msg = message.lower()
    # Idle abort: expected exit when no issues are assigned
    if "no issues assigned" in msg and "abort" in msg:
        return "idle_abort"
    if "signal" in msg:
        return "signal_kill"
    if "timed out" in msg:
        return "timeout"
    if "killed due to silence" in msg or "stale connection" in msg:
        return "stale_connection"
    if "exited with code" in msg:
        return "exit_error"
    return "unknown"


def _get_failure_log() -> Path:
    """Resolve failures.log path with crashes.log fallback."""
    failures_log = Path("worker_logs/failures.log")
    crashes_log = Path("worker_logs/crashes.log")
    if failures_log.exists():
        return failures_log
    if crashes_log.exists():
        return crashes_log
    return failures_log


def get_recent_crashes() -> dict:
    """Count crashes in last 24 hours from failures.log (fallback crashes.log), by type.

    Returns dict with:
      - total: all entries in 24h (crashes + idle_aborts + stale_connection)
      - real: crashes excluding stale_connection and idle_abort (actual failures)
      - stale_connection: count of stale_connection restarts (excluded from flags)
      - idle_aborts: count of expected exits when no issues assigned (Part of #2311)
      - by_type: breakdown by category (or None if no crashes)

    REQUIRES: None (works with or without crash log)
    ENSURES: Returns dict with 'total', 'real', 'stale_connection', 'idle_aborts'
             (all non-negative integers)
    ENSURES: Returns 'by_type' as dict or None
    ENSURES: Never raises (returns zero counts on error)
    """
    crash_log = _get_failure_log()
    if not crash_log.exists():
        return {
            "total": 0,
            "real": 0,
            "stale_connection": 0,
            "idle_aborts": 0,
            "by_type": None,
        }

    # Uses module-level CRASH_LOG_PATTERN for consistency with health_check.py (#1698)
    counts: dict[str, int] = {}
    now = datetime.now()
    try:
        for line in crash_log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            match = CRASH_LOG_PATTERN.match(line)
            if match:
                timestamp_str, message = match.groups()
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    if (now - timestamp).total_seconds() < 86400:
                        category = _classify_crash(message)
                        counts[category] = counts.get(category, 0) + 1
                except ValueError:
                    pass
    except Exception:
        pass  # Best-effort: crash log parsing

    total = sum(counts.values())
    stale = counts.get("stale_connection", 0)
    idle = counts.get("idle_abort", 0)
    return {
        "total": total,
        "real": total - stale - idle,  # Actual failures (excludes expected exits)
        "stale_connection": stale,
        "idle_aborts": idle,
        "by_type": counts if counts else None,
    }

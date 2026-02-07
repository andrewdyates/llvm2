#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Session metrics collection.

Functions for active session detection and failure tracking.
Part of #404: pulse.py module split.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from ai_template_scripts.subprocess_utils import run_cmd
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import run_cmd

from .constants import WORKER_LOG_PATTERN

# Optional status fields preserved for diagnostics (#1858)
_STATUS_OPTIONAL_FIELDS = ("mode", "status", "iteration", "log_file", "updated_at", "ai_tool", "model")

# Exit codes that indicate infrastructure failures, not lost work (#1486)
# These are "traceable" in the sense we know why they failed.
INFRASTRUCTURE_EXIT_CODES = {
    124,  # Timeout (command timed out)
    125,  # Silence-killed (no output, stale connection)
    126,  # No issues assigned (expected fail-fast abort, #1641, #2418)
}

# Patterns in worker logs that indicate known infrastructure failures (#1486)
# These failures have traceable causes even if working_issues is empty.
INFRASTRUCTURE_ERROR_PATTERNS = [
    "stream disconnected",
    "network error",
    "error sending request",
    "connection reset",
    "connection refused",
    "connection error",  # Generic connection failure (added for #1635)
    "ssl error",
    "certificate verify failed",
    "timeout waiting",
    "rate limit",
]


def _parse_pid(value: object) -> int | None:
    """Parse a PID from an int or digit-only string.

    REQUIRES: value is int or str-like (other types return None)
    ENSURES: Returns positive integer PID or None
    ENSURES: Never raises
    """
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        pid_str = value.strip()
        if pid_str.isdigit():
            parsed = int(pid_str)
            if parsed > 0:
                return parsed
    return None


def _read_pid_file(path: Path) -> int | None:
    """Read PID from a plain-text pid file.

    REQUIRES: path is a filesystem path to a pid file
    ENSURES: Returns positive integer PID or None
    ENSURES: Never raises
    """
    try:
        pid_text = path.read_text().strip()
    except OSError:
        return None
    return _parse_pid(pid_text)


def _read_status_pid(path: Path) -> int | None:
    """Read PID from a JSON status file.

    REQUIRES: path is a filesystem path to a JSON status file
    ENSURES: Returns positive integer PID or None
    ENSURES: Never raises
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return _parse_pid(data.get("pid"))


def _read_status_data(path: Path) -> dict | None:
    """Read full status data from a JSON status file (#1858).

    Returns dict with 'pid' (int) and optional fields: mode, status,
    iteration, log_file, updated_at.

    REQUIRES: path is a filesystem path to a JSON status file
    ENSURES: Returns dict with 'pid' key or None
    ENSURES: Never raises
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    pid = _parse_pid(data.get("pid"))
    if pid is None:
        return None
    result: dict = {"pid": pid}
    for field in _STATUS_OPTIONAL_FIELDS:
        if field in data:
            result[field] = data[field]
    return result


def _pid_command_line(pid: int) -> str | None:
    """Return the command line for a PID, if available.

    REQUIRES: pid is a positive integer
    ENSURES: Returns command line string or None
    ENSURES: Never raises
    """
    result = run_cmd(["ps", "-o", "command=", "-p", str(pid)], timeout=5)
    if not result.ok:
        return None
    cmdline = result.stdout.strip()
    return cmdline or None


def _pid_looks_like_looper(pid: int) -> bool:
    """Return True if PID command line appears to be a looper session.

    REQUIRES: pid is a positive integer
    ENSURES: Returns False if command line is unavailable
    ENSURES: Never raises
    """
    cmdline = _pid_command_line(pid)
    if cmdline is None:
        return False
    return "looper.py" in cmdline.lower()


def _iter_role_status_files() -> list[Path]:
    """Return candidate status file paths for all known roles.

    Includes both single-session files (e.g., .worker_status.json) and
    multi-session numbered files (e.g., .worker_2_status.json).

    REQUIRES: None
    ENSURES: Returns list of Paths for role status files
    ENSURES: Never raises
    """
    roles_dir = Path(".claude/roles")
    roles: list[str] = []
    if roles_dir.is_dir():
        roles = sorted(
            path.stem for path in roles_dir.glob("*.md") if path.stem != "shared"
        )
    if not roles:
        roles = ["manager", "prover", "researcher", "worker"]

    # Build candidate list from known role names
    status_files: list[Path] = []
    for role in roles:
        status_files.append(Path(f".{role}_status.json"))
        status_files.append(Path(f"{role}_status.json"))

    # Discover multi-session status files via glob (#1858)
    # Patterns: .{role}_{N}_status.json and {role}_{N}_status.json
    # Filter to only include files that start with a known role
    roles_set = set(roles)
    for pattern in [".*_[0-9]*_status.json", "*_[0-9]*_status.json"]:
        for path in Path(".").glob(pattern):
            if path in status_files:
                continue
            # Extract base role: .worker_1_status.json -> worker
            name = path.stem.replace("_status", "").lstrip(".")
            base_role = name.rsplit("_", 1)[0] if "_" in name else name
            if base_role in roles_set:
                status_files.append(path)

    return status_files


def get_active_session_details() -> list[dict]:
    """Get details of active AI sessions (#1020, #1858).

    Returns list of dicts with keys:
        - role: Role name (worker, manager, etc.) or "unknown"
        - pid: Process ID
        - mode: (optional) Session mode
        - status: (optional) Current status
        - iteration: (optional) Current iteration number
        - log_file: (optional) Path to log file
        - updated_at: (optional) Last status update timestamp

    Used by unclaimed_sessions flag to show actionable session info.

    REQUIRES: None (works with or without session files)
    ENSURES: Returns list of dicts with 'role' (str) and 'pid' (int) keys
    ENSURES: Only includes live looper processes
    """
    # Map PID -> session info (role + optional fields)
    pid_info: dict[int, dict] = {}

    # Check .pid_* files
    for pid_file in Path(".").glob(".pid_*"):
        pid = _read_pid_file(pid_file)
        if pid is None:
            continue
        # Extract role from filename: .pid_worker -> worker
        role = pid_file.name.replace(".pid_", "") or "unknown"
        pid_info[pid] = {"role": role, "source": str(pid_file)}

    # Check status files (may override .pid_* with more specific role and add optional fields)
    for status_file in _iter_role_status_files():
        if not status_file.exists():
            continue
        status_data = _read_status_data(status_file)
        if status_data is None:
            continue
        pid = status_data["pid"]
        # Extract role from filename: .worker_status.json -> worker
        # Multi-session: .worker_2_status.json -> worker_2
        role = status_file.stem.replace("_status", "").lstrip(".")
        info = {"role": role, "source": str(status_file)}
        # Preserve optional fields from status file (#1858)
        for field in _STATUS_OPTIONAL_FIELDS:
            if field in status_data:
                info[field] = status_data[field]
        pid_info[pid] = info

    # Filter to active looper processes
    active_sessions = []
    for pid, info in pid_info.items():
        result = run_cmd(["kill", "-0", str(pid)])
        if not result.ok:
            continue
        if not _pid_looks_like_looper(pid):
            continue
        session = {"role": info["role"], "pid": pid}
        # Include optional fields if available (#1858)
        for field in _STATUS_OPTIONAL_FIELDS:
            if field in info:
                session[field] = info[field]
        active_sessions.append(session)

    return active_sessions


def get_active_sessions() -> int:
    """Count active AI sessions (by .pid_* files or looper status files).

    REQUIRES: None (works with or without session files)
    ENSURES: Returns non-negative integer
    ENSURES: Never raises
    """
    return len(get_active_session_details())


def _get_worker_log_info(iteration: int, worker_id: int | None = None) -> str:
    """Get info about a worker log file for an iteration.

    Returns a short description of the log state (empty, line count, last event).

    Handles both regular iterations and audit rounds (#1373):
    - Regular: worker_iter_75_claude_*.jsonl
    - Audit:   worker_iter_75.1_claude_*.jsonl
    """
    logs_dir = Path("worker_logs")
    if not logs_dir.exists():
        return "no logs dir"

    # Uses module-level WORKER_LOG_PATTERN (#1698)
    candidates: list[tuple[Path, int | None, int | None, float]] = []
    for entry in logs_dir.iterdir():
        if not entry.is_file():
            continue
        match = WORKER_LOG_PATTERN.match(entry.name)
        if not match:
            continue
        if int(match.group("iteration")) != iteration:
            continue
        worker_str = match.group("worker_id")
        audit_str = match.group("audit_round")
        worker_val = int(worker_str) if worker_str is not None else None
        audit_val = int(audit_str) if audit_str is not None else None
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        candidates.append((entry, worker_val, audit_val, mtime))

    if worker_id is not None:
        matching_worker = [c for c in candidates if c[1] == worker_id]
        if matching_worker:
            candidates = matching_worker
        else:
            candidates = [c for c in candidates if c[1] is None]

        if not candidates:
            return "log not found"

        exact_iteration = [c for c in candidates if c[2] is None]
        if exact_iteration:
            candidates = exact_iteration
    else:
        if not candidates:
            return "log not found"

        exact_iteration = [c for c in candidates if c[2] is None]
        if exact_iteration:
            candidates = exact_iteration

        # Prefer worker-id logs when metrics omit worker_id (multi-worker runs).
        if any(c[1] is not None for c in candidates):
            candidates = [c for c in candidates if c[1] is not None]

    candidates.sort(key=lambda c: c[3], reverse=True)
    log_path = candidates[0][0]
    try:
        content = log_path.read_text().strip()
        if not content:
            return "log empty"
        lines = content.splitlines()
        count = len(lines)
        # Try to get last event type
        try:
            last_entry = json.loads(lines[-1])
            event = last_entry.get("type", last_entry.get("event", "?"))
            return f"{count} lines, last={event}"
        except json.JSONDecodeError:
            return f"{count} lines"
    except Exception:
        return "log unreadable"

    return "log not found"


def _is_infrastructure_failure(log_path: Path, max_line_bytes: int = 100_000) -> bool:
    """Check if worker log shows an infrastructure failure.

    Returns True if the log contains patterns indicating network/infrastructure
    errors rather than lost work. Handles large JSONL lines gracefully.

    Part of #1486 - reduce untraceable_failures false positives.
    """
    if not log_path.exists():
        return False

    try:
        # Read file but skip lines that are too large (huge JSON payloads)
        content_lower = ""
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(line) <= max_line_bytes:
                    content_lower += line.lower()

        # Check for known infrastructure error patterns
        for pattern in INFRASTRUCTURE_ERROR_PATTERNS:
            if pattern in content_lower:
                return True

        # Check if the last meaningful event indicates success
        # This catches cases where looper.jsonl shows failure but log shows success
        lines = [ln for ln in content_lower.split("\n") if ln.strip()]
        if lines:
            last_line = lines[-1]
            # Look for result.success or similar success indicators
            if "result.success" in last_line or '"success": true' in last_line:
                return True

    except Exception:
        pass  # Best-effort check

    return False


def get_untraceable_failures() -> list[dict]:
    """Find worker failures in last 24h with empty working_issues.

    Returns list of entries that failed without issue traceability.
    These represent lost work that can't be attributed to any issue.

    Excludes (#1486):
    - Infrastructure failures (timeout, silence-killed, network errors)
    - Runs where the worker log shows success despite exit_code != 0

    Part of #938 - enforce traceability for worker runs.

    REQUIRES: None (works with or without metrics/looper.jsonl file)
    ENSURES: Returns list of dicts with 'session_id', 'iteration', 'exit_code', 'log_info' keys
    ENSURES: Only includes failures from last 24 hours
    ENSURES: Never raises (returns empty list on error)
    """
    looper_log = Path("metrics/looper.jsonl")
    if not looper_log.exists():
        return []

    untraceable: list[dict] = []
    now = datetime.now()
    logs_dir = Path("worker_logs")

    try:
        for line in looper_log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only check worker runs
            if entry.get("role") != "worker":
                continue

            # Only check failures (exit_code != 0 and not committed)
            exit_code = entry.get("exit_code", 0)
            committed = entry.get("committed", False)
            if exit_code == 0 or committed:
                continue

            # Skip known infrastructure failure exit codes (#1486)
            if exit_code in INFRASTRUCTURE_EXIT_CODES:
                continue

            # Only check recent entries (last 24h)
            start_time = entry.get("start_time", 0)
            if start_time > 0:
                entry_time = datetime.fromtimestamp(start_time)
                if (now - entry_time).total_seconds() > 86400:
                    continue

            # Check for empty working_issues
            working_issues = entry.get("working_issues", [])
            if not working_issues:
                iteration = entry.get("iteration", 0)
                worker_id = entry.get("worker_id")
                # Prefer canonical log_file when available (#1463)
                log_file = entry.get("log_file")
                if log_file:
                    log_info = log_file
                    log_path = logs_dir / Path(log_file).name
                else:
                    log_info = _get_worker_log_info(iteration, worker_id)
                    # Try to find the log file for infrastructure check
                    # Use worker_id to get the correct log when multiple workers exist
                    log_path = None
                    if logs_dir.exists():
                        if worker_id is not None:
                            # Specific pattern for this worker
                            pattern = f"worker_{worker_id}_iter_{iteration}_*.jsonl"
                        else:
                            # Fallback to generic pattern
                            pattern = f"worker_*iter_{iteration}_*.jsonl"
                        matches = list(logs_dir.glob(pattern))
                        if matches:
                            log_path = matches[0]

                # Check if this is actually an infrastructure failure (#1486)
                if log_path and _is_infrastructure_failure(log_path):
                    continue

                untraceable.append(
                    {
                        "session_id": entry.get("session_id", "unknown"),
                        "iteration": iteration,
                        "exit_code": exit_code,
                        "log_info": log_info,
                    }
                )
    except Exception:
        pass  # Best-effort parsing

    return untraceable


__all__ = [
    # Constants
    "_STATUS_OPTIONAL_FIELDS",
    "INFRASTRUCTURE_EXIT_CODES",
    "INFRASTRUCTURE_ERROR_PATTERNS",
    # Public functions
    "get_active_session_details",
    "get_active_sessions",
    "get_untraceable_failures",
    # Internal functions (exported for testing)
    "_parse_pid",
    "_read_pid_file",
    "_read_status_pid",
    "_read_status_data",
    "_pid_command_line",
    "_pid_looks_like_looper",
    "_iter_role_status_files",
    "_get_worker_log_info",
    "_is_infrastructure_failure",
]

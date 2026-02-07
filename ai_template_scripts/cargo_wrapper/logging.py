# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Logging utilities for cargo_wrapper.

Uses shared_logging for common functions (Part of #2007).
Domain-specific functions (log_orphan, log_build, check_retry_loop) remain here.
"""

from __future__ import annotations

import datetime
import json
import os
import shlex

from ai_template_scripts.shared_logging import (
    debug_swallow,
    log_stderr,
    now_iso,
    rotate_log_file,
)

from . import _state
from .constants import MAX_LOG_LINES

__all__ = [
    "debug_swallow",
    "now_iso",
    "log_stderr",
    "rotate_log_file",
    "log_orphan",
    "log_build",
    "check_retry_loop",
]


def log_orphan(proc: dict) -> None:
    """Log killed orphan to orphans.log."""
    if _state.ORPHANS_LOG is None:
        return  # Not initialized - skip logging (best-effort)
    entry = {
        "killed_at": now_iso(),
        "pid": proc["pid"],
        "age_seconds": proc["age_seconds"],
        "comm": proc["comm"],
        "args": proc["args"],
    }
    with open(_state.ORPHANS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rotate_log_file(_state.ORPHANS_LOG, MAX_LOG_LINES)


def log_build(
    context: dict,
    command: list[str],
    started: datetime.datetime,
    finished: datetime.datetime,
    exit_code: int,
    timeout_sec: int,
) -> None:
    """Log build to builds.log."""
    if _state.BUILDS_LOG is None:
        return  # Not initialized - skip logging (best-effort)
    entry = {
        **context,
        "cwd": os.getcwd(),
        "command": shlex.join(command),  # Properly quote args with spaces
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "duration_sec": round((finished - started).total_seconds(), 1),
        "timeout_sec": timeout_sec,
    }
    with open(_state.BUILDS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rotate_log_file(_state.BUILDS_LOG, MAX_LOG_LINES)


def check_retry_loop(command: str, cwd: str, current_commit: str) -> None:
    """Check builds.log for repeated failures. Prints info/warning if detected."""
    if _state.BUILDS_LOG is None:
        return
    if not _state.BUILDS_LOG.exists():
        return

    try:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=60)
        failures = []

        for line in _state.BUILDS_LOG.read_text().splitlines()[-30:]:  # Last 30 entries
            try:
                entry = json.loads(line)
                if (
                    entry.get("command") == command
                    and entry.get("cwd") == cwd
                    and entry.get("exit_code", 0) != 0
                ):
                    started = datetime.datetime.fromisoformat(entry["started_at"])
                    # Handle timezone-naive timestamps
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=datetime.UTC)
                    if started > cutoff:
                        failures.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        if len(failures) >= 3:
            # Check if any code changes between failures
            commits = {f.get("commit", "") for f in failures}
            commits.add(current_commit)
            commits.discard("")
            same_commit = len(commits) <= 1

            if same_commit:
                log_stderr("")
                log_stderr(f"⚠️  RETRY LOOP: {len(failures)}x in 60s (no changes)")
                log_stderr(f"    Command: {command}")
                log_stderr(f"    Commit:  {current_commit or 'unknown'}")
                log_stderr("")
                log_stderr(
                    "    HINT: Read the error output. Investigate or file an issue."
                )
                log_stderr(f"    Details: {_state.BUILDS_LOG}")
                log_stderr("")
            else:
                # Failures but with code changes - just info
                log_stderr(f"[cargo] ℹ️  {len(failures)}x failures (commits changed)")

    except Exception:
        debug_swallow("check_retry_loop")

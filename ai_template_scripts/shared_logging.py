# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shared logging utilities for ai_template scripts.

Part of #2007 - consolidates common logging patterns across modules:
- cargo_wrapper/logging.py
- gh_rate_limit/limiter.py

Usage:
    from ai_template_scripts.shared_logging import (
        append_log,
        debug_log,
        format_json_entry,
        is_debug_mode,
        log_stderr,
        now_iso,
        rotate_log_file,
    )

    # Debug logging (controlled by AIT_DEBUG or module-specific env var)
    debug_log("my_module", f"processed {count} items")

    # Stderr output (immediate flush for AI visibility)
    log_stderr("Processing complete")

    # JSON log entry
    entry = format_json_entry(operation="build", status="success", duration=1.5)
    append_log(log_path, entry)

    # Log rotation
    rotate_log_file(log_path, max_lines=5000)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "append_log",
    "debug_log",
    "format_json_entry",
    "is_debug_mode",
    "log_stderr",
    "now_iso",
    "rotate_log_file",
]

# Values that enable debug mode (case-insensitive)
_DEBUG_TRUE_VALUES = frozenset(("1", "true", "yes"))


def is_debug_mode(module_env_var: str | None = None) -> bool:
    """Check if debug mode is enabled.

    Checks AIT_DEBUG first (unified), then module-specific env var if provided.
    Returns True if any relevant env var is set to a truthy value.

    Args:
        module_env_var: Optional module-specific env var (e.g., "AIT_GH_DEBUG")
    """
    # Check unified env var first
    if os.environ.get("AIT_DEBUG", "").lower() in _DEBUG_TRUE_VALUES:
        return True
    # Check module-specific if provided
    if (
        module_env_var
        and os.environ.get(module_env_var, "").lower() in _DEBUG_TRUE_VALUES
    ):
        return True
    return False


def debug_log(module: str, msg: str, module_env_var: str | None = None) -> None:
    """Log debug message if debug mode is enabled.

    Output goes to stderr with module prefix for easy filtering.
    No-op when debug mode is off for zero overhead.

    Args:
        module: Module name for log prefix (e.g., "cargo_wrapper", "gh_rate_limit")
        msg: Message to log
        module_env_var: Optional module-specific env var for backwards compat
    """
    if not is_debug_mode(module_env_var):
        return
    print(f"{module} [DEBUG]: {msg}", file=sys.stderr)
    sys.stderr.flush()


def log_stderr(msg: str) -> None:
    """Print to stderr with immediate flush.

    Use for human-readable status messages that AIs need to see without delay.
    """
    print(msg, file=sys.stderr)
    sys.stderr.flush()


def now_iso(timespec: str = "seconds") -> str:
    """Return current UTC time as ISO 8601 string.

    Args:
        timespec: Precision - "seconds", "milliseconds", "microseconds"
    """
    return datetime.now(UTC).isoformat(timespec=timespec)


def format_json_entry(**fields: object) -> str:
    """Format a dict as a JSON log entry (single line).

    Automatically adds "ts" (timestamp) field if not present.
    Non-serializable values (Path, datetime, etc.) are converted to strings.

    Args:
        **fields: Key-value pairs for the log entry

    Returns:
        JSON string (single line, no trailing newline)
    """
    if "ts" not in fields:
        fields = {"ts": now_iso(), **fields}
    return json.dumps(fields, ensure_ascii=True, default=str)


def append_log(log_path: Path, entry: str) -> bool:
    """Append a log entry to file.

    Creates parent directories if needed. Returns False on error (best-effort).

    Args:
        log_path: Path to log file
        entry: Log entry string (newline added automatically)

    Returns:
        True if write succeeded, False on error
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(entry + "\n")
        return True
    except OSError:
        return False


def rotate_log_file(log_path: Path, max_lines: int) -> bool:
    """Truncate log file to last max_lines entries (atomic).

    Uses rename for atomic update to prevent corruption during rotation.
    Returns False on error (best-effort, rotation is housekeeping).

    Args:
        log_path: Path to log file
        max_lines: Maximum lines to keep (must be > 0)

    Returns:
        True if rotation succeeded or not needed, False on error
    """
    if not log_path.exists():
        return True
    if max_lines < 1:
        return True  # Invalid input - no-op, don't corrupt the log

    tmp_path = log_path.parent / f"{log_path.name}.{os.getpid()}.tmp"
    try:
        lines = log_path.read_text().splitlines()
        if len(lines) > max_lines:
            tmp_path.write_text("\n".join(lines[-max_lines:]) + "\n")
            os.rename(tmp_path, log_path)
        return True
    except OSError:
        return False
    finally:
        tmp_path.unlink(missing_ok=True)

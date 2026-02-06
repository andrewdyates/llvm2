# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Process utilities for cargo_wrapper."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import datetime

from ai_template_scripts.subprocess_utils import is_process_alive

from .constants import STALE_PROCESS_AGE
from .logging import debug_swallow, log_orphan

__all__ = [
    "get_process_start_time",
    "get_process_parent",
    "is_ancestor_of_self",
    "find_cargo_processes",
    "parse_etime",
    "cleanup_orphans",
    "_is_cargo_or_rustc",  # Exported for testing
]


def get_process_start_time(pid: int) -> float | None:
    """Get process start time as Unix timestamp. None if not found."""
    try:
        # Force C locale for consistent date format across systems (#1532)
        env = {**os.environ, "LC_TIME": "C", "LC_ALL": "C"}
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if result.returncode != 0:
            return None
        lstart = result.stdout.strip()
        if not lstart:
            return None
        # ps lstart format: "Mon Jan 20 15:30:45 2026"
        try:
            dt = datetime.strptime(lstart, "%c")
            return dt.timestamp()
        except ValueError:
            parts = lstart.split()
            if len(parts) >= 5:
                month_map = {
                    "Jan": 1,
                    "Feb": 2,
                    "Mar": 3,
                    "Apr": 4,
                    "May": 5,
                    "Jun": 6,
                    "Jul": 7,
                    "Aug": 8,
                    "Sep": 9,
                    "Oct": 10,
                    "Nov": 11,
                    "Dec": 12,
                }
                month = month_map.get(parts[1], 1)
                day = int(parts[2])
                time_parts = parts[3].split(":")
                hour, minute, sec = (
                    int(time_parts[0]),
                    int(time_parts[1]),
                    int(time_parts[2]),
                )
                year = int(parts[4])
                dt = datetime(year, month, day, hour, minute, sec)
                return dt.timestamp()
        return None
    except Exception:
        debug_swallow("get_process_start_time")
        return None


def get_process_parent(pid: int) -> int | None:
    """Get parent PID of a process."""
    try:
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        debug_swallow("get_process_parent")
    return None


def is_ancestor_of_self(pid: int) -> bool:
    """Check if given PID is an ancestor of the current process."""
    current = os.getpid()
    visited = set()
    while current and current not in visited:
        if current == pid:
            return True
        visited.add(current)
        parent = get_process_parent(current)
        if parent is None or parent == current:
            break
        current = parent
    return False


def _is_cargo_or_rustc(comm: str, args: str = "") -> tuple[bool, str]:
    """Check if comm is cargo, rustc, cbmc, or cargo-* variant.

    On macOS, ps comm shows full path (e.g., /Users/.../bin/cargo).
    On Linux, ps comm shows just the executable name (e.g., cargo).
    On some macOS versions, comm may be truncated (e.g., /Users/ayates/.r).
    In that case, fall back to the first token of args.

    Matches: cargo, rustc, cbmc, cargo-kani, cargo-clippy, and other cargo-* variants.

    Returns (is_match, normalized_name).

    Fix propagated from kafka2 commit 064f26b for macOS truncated comm (#1898).
    """
    # Handle bare command names (Linux-style)
    if comm in ("cargo", "rustc", "cargo-kani", "cbmc"):
        return True, comm

    # Handle full paths (macOS-style)
    basename = comm.rpartition("/")[2] if "/" in comm else comm

    # Check for cargo variants (including cargo-kani, cargo-clippy, etc.)
    if basename == "cargo" or basename.startswith("cargo-"):
        return True, basename

    # Check for rustc and cbmc
    if basename in ("rustc", "cbmc"):
        return True, basename

    # Fallback: macOS may truncate comm to a partial path (e.g., /Users/ayates/.r).
    # Use first token of args which contains full path (e.g., /Users/.../.cargo/bin/cargo).
    if args.strip():
        first_token = args.strip().split(None, 1)[0]
        args_basename = os.path.basename(first_token)
        if args_basename in ("cargo", "rustc", "cbmc") or args_basename.startswith(
            "cargo-"
        ):
            return True, args_basename

    return False, ""


def find_cargo_processes() -> list[dict]:
    """Find all cargo/rustc processes with their age and PPID.

    Improved for macOS compatibility (#1873):
    - Handles full path in comm (e.g., /Users/.../bin/cargo)
    - Includes PPID for orphan detection (PPID=1 means reparented to init)
    - Stores full args for better diagnostics
    """
    processes = []
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,comm,args"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split(None, 4)
            if len(parts) < 4:
                continue
            pid, ppid, etime, comm = parts[0], parts[1], parts[2], parts[3]
            args = parts[4] if len(parts) > 4 else ""

            is_match, normalized = _is_cargo_or_rustc(comm, args)
            if not is_match:
                continue

            age_seconds = parse_etime(etime)
            processes.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "age_seconds": age_seconds,
                    "comm": normalized,  # Store normalized name for logs
                    "args": args[:200],  # Longer for better diagnostics
                }
            )
    except Exception:
        debug_swallow("find_cargo_processes")
    return processes


def parse_etime(etime: str) -> int:
    """Parse ps etime format to seconds."""
    try:
        days: int
        rest: str
        if "-" in etime:
            days_str, rest = etime.split("-", 1)
            days = int(days_str)
        else:
            days = 0
            rest = etime

        parts = rest.split(":")
        if len(parts) == 1:  # SS (seconds only, non-standard but handle gracefully)
            return days * 86400 + int(parts[0])
        if len(parts) == 2:  # MM:SS
            return days * 86400 + int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:  # HH:MM:SS
            return (
                days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            )
    except Exception:
        debug_swallow("parse_etime")
    return 0


def cleanup_orphans() -> int:
    """Kill cargo/rustc processes older than 2 hours or orphaned (PPID=1).

    An orphaned process (PPID=1) is one whose parent died and was reparented
    to init/launchd. These are likely from crashed sessions (#1873).

    Returns count of processes killed.
    """
    killed = 0
    for proc in find_cargo_processes():
        pid = proc["pid"]
        ppid = proc.get("ppid", -1)
        age_seconds = proc["age_seconds"]

        # Kill if: too old OR orphaned (PPID=1) and at least 5 minutes old
        # The 5-minute threshold avoids killing processes that are just starting
        should_kill = False
        if age_seconds >= STALE_PROCESS_AGE:
            should_kill = True
        elif ppid == 1 and age_seconds >= 300:  # 5 minutes
            should_kill = True

        if not should_kill:
            continue

        # Don't kill our own ancestors (e.g., if spawned by long-running cargo)
        if is_ancestor_of_self(pid):
            continue

        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            if is_process_alive(pid):
                os.kill(pid, signal.SIGKILL)
            log_orphan(proc)
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    return killed

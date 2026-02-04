# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/memory_logger.py - Memory logging infrastructure for OOM debugging (#1470).

Provides memory state capture and logging for debugging OOM crashes:
- Pre-command memory logging (captures state BEFORE tool calls)
- Process tree memory tracking (parent + all descendants)
- OOM signal detection
- Memory pressure levels via sysctl

Usage:
    from looper.memory_logger import (
        get_memory_state,
        get_process_tree_memory_mb,
        log_pre_command_memory,
        is_oom_signal,
    )

    # Before running a command
    log_pre_command_memory("cargo build")

    # Check if exit was OOM
    if is_oom_signal(exit_code):
        print("Process was killed by OOM killer")
"""

__all__ = [
    "MemoryState",
    "get_memory_state",
    "get_page_size",
    "get_process_tree_memory_mb",
    "is_oom_signal",
    "log_post_crash_memory",
    "log_pre_command_memory",
]

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from looper.constants import LOG_DIR
from looper.log import debug_swallow


@dataclass
class MemoryState:
    """Snapshot of system memory state."""

    used_gb: float
    free_gb: float
    total_gb: float
    pressure_level: str  # "normal", "warning", "critical"
    timestamp: float

    @property
    def used_percent(self) -> int:
        """Memory usage as percentage."""
        if self.total_gb <= 0:
            return 0
        return int(self.used_gb * 100 / self.total_gb)


def get_page_size() -> int:
    """Get system page size in bytes.

    ARM64 (Apple Silicon) uses 16KB pages, Intel uses 4KB.
    Returns 4096 as fallback if detection fails.
    """
    try:
        result = subprocess.run(["pagesize"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        debug_swallow("get_page_size_pagesize", e)

    # Fallback: try sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.pagesize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        debug_swallow("get_page_size_sysctl", e)

    # Default to 4KB (Intel standard)
    return 4096


def get_memory_state() -> MemoryState | None:
    """Get current system memory state.

    Returns used/free/total in GB plus pressure level.
    Uses macOS vm_stat and sysctl for accurate memory tracking.

    Returns None if memory state cannot be determined.
    """
    try:
        page_size = get_page_size()

        # Get memory stats via vm_stat
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None

        stats = {}
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                # Remove trailing period and convert to int
                try:
                    stats[key.strip()] = int(value.strip().rstrip("."))
                except ValueError as e:
                    debug_swallow("parse_vm_stat_value", e)

        # Calculate memory in pages
        free_pages = stats.get("Pages free", 0)
        active_pages = stats.get("Pages active", 0)
        inactive_pages = stats.get("Pages inactive", 0)
        wired_pages = stats.get("Pages wired down", 0)
        # Speculative pages are part of free memory
        speculative_pages = stats.get("Pages speculative", 0)

        # Total = all pages we can account for
        total_pages = (
            free_pages + active_pages + inactive_pages + wired_pages + speculative_pages
        )

        if total_pages == 0:
            return None

        # Used = active + wired (inactive is reclaimable)
        used_pages = active_pages + wired_pages

        # Convert to GB
        bytes_per_page = page_size
        used_gb = (used_pages * bytes_per_page) / (1024**3)
        free_gb = (
            (free_pages + inactive_pages + speculative_pages) * bytes_per_page
        ) / (1024**3)
        total_gb = (total_pages * bytes_per_page) / (1024**3)

        # Get pressure level via sysctl (macOS specific)
        pressure_level = "normal"
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                level = int(result.stdout.strip())
                # 1 = normal, 2 = warning, 4 = critical
                if level >= 4:
                    pressure_level = "critical"
                elif level >= 2:
                    pressure_level = "warning"
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            debug_swallow("get_memory_pressure_level", e)
            # Fallback: use percentage-based thresholds
            used_pct = int(used_gb * 100 / total_gb) if total_gb > 0 else 0
            if used_pct >= 90:
                pressure_level = "critical"
            elif used_pct >= 80:
                pressure_level = "warning"

        return MemoryState(
            used_gb=round(used_gb, 2),
            free_gb=round(free_gb, 2),
            total_gb=round(total_gb, 2),
            pressure_level=pressure_level,
            timestamp=time.time(),
        )

    except (OSError, subprocess.SubprocessError, ValueError) as e:
        debug_swallow("get_memory_state", e)
        return None


def get_process_tree_memory_mb(pid: int) -> float | None:
    """Get total memory usage of a process and all its descendants in MB.

    Recursively finds all child processes via pgrep and sums their RSS.
    This is critical for tracking memory hogs in subprocess trees.

    Args:
        pid: Root process ID to start tracking from

    Returns:
        Total memory in MB, or None if tracking fails
    """
    try:
        # Get all descendant PIDs via pgrep
        all_pids = {pid}
        to_check = [pid]

        while to_check:
            current_pid = to_check.pop()
            result = subprocess.run(
                ["pgrep", "-P", str(current_pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        child_pid = int(line)
                        if child_pid not in all_pids:
                            all_pids.add(child_pid)
                            to_check.append(child_pid)

        # Get RSS for all PIDs via ps
        total_rss_kb = 0
        for check_pid in all_pids:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(check_pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    total_rss_kb += int(result.stdout.strip())
                except ValueError as e:
                    debug_swallow("parse_process_rss", e)

        return total_rss_kb / 1024  # Convert KB to MB

    except (OSError, subprocess.SubprocessError, ValueError) as e:
        debug_swallow("get_process_tree_memory", e)
        return None


def is_oom_signal(exit_code: int) -> bool:
    """Check if an exit code indicates OOM kill.

    On Unix, signal-based exits are reported as 128 + signal_number
    by shells, or as negative signal numbers by subprocess.
    SIGKILL (9) is used by the OOM killer, so 137 or -9 indicate OOM.

    Args:
        exit_code: Process exit code

    Returns:
        True if exit was likely due to OOM kill (SIGKILL)
    """
    if exit_code == 137:
        return True
    return exit_code < 0 and abs(exit_code) == 9


def log_pre_command_memory(
    command: str,
    log_dir: str | Path | None = None,
) -> MemoryState | None:
    """Log memory state before running a command.

    This captures memory state BEFORE tool calls, which is critical
    for OOM debugging since memory is already freed by the time
    we can log after a crash.

    Args:
        command: Command about to be executed (for context)
        log_dir: Directory for memory logs (default: worker_logs/)

    Returns:
        MemoryState if logging succeeded, None otherwise
    """
    state = get_memory_state()
    if state is None:
        return None

    # Determine log directory
    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)

    # Create log directory if needed
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        debug_swallow("pre_command_memory_mkdir", e)
        return state  # Return state even if logging fails

    # Log to memory.log
    log_file = log_dir / "memory.log"
    try:
        import json
        from datetime import datetime

        entry = {
            "timestamp": datetime.fromtimestamp(state.timestamp).isoformat(),
            "event": "pre_command",
            "command": command[:200],  # Truncate long commands
            "memory": {
                "used_gb": state.used_gb,
                "free_gb": state.free_gb,
                "total_gb": state.total_gb,
                "used_percent": state.used_percent,
                "pressure_level": state.pressure_level,
            },
            "pid": os.getpid(),
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except (OSError, TypeError) as e:
        debug_swallow("pre_command_memory_log", e)

    return state


def log_post_crash_memory(
    command: str,
    exit_code: int,
    log_dir: str | Path | None = None,
) -> MemoryState | None:
    """Log memory state after a crash or abnormal exit.

    Called when a command exits with non-zero status, especially
    signal-based exits that may indicate OOM.

    Args:
        command: Command that crashed
        exit_code: Process exit code
        log_dir: Directory for memory logs (default: worker_logs/)

    Returns:
        MemoryState if logging succeeded, None otherwise
    """
    state = get_memory_state()
    if state is None:
        return None

    # Determine log directory
    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        debug_swallow("post_crash_memory_mkdir", e)
        return state

    log_file = log_dir / "memory.log"
    try:
        import json
        from datetime import datetime

        entry = {
            "timestamp": datetime.fromtimestamp(state.timestamp).isoformat(),
            "event": "post_crash",
            "command": command[:200],
            "exit_code": exit_code,
            "was_oom": is_oom_signal(exit_code),
            "memory": {
                "used_gb": state.used_gb,
                "free_gb": state.free_gb,
                "total_gb": state.total_gb,
                "used_percent": state.used_percent,
                "pressure_level": state.pressure_level,
            },
            "pid": os.getpid(),
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except (OSError, TypeError) as e:
        debug_swallow("post_crash_memory_log", e)

    return state

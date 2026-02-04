# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/system_context.py - System status and health checks.

Functions for gathering system status (memory, disk) and running health checks.
"""

__all__ = [
    "get_system_status",
    "truncate_output",
    "run_system_health_check",
]

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from looper.config import load_timeout_config
from looper.context.uncommitted_warning import get_uncommitted_changes_warning
from looper.log import debug_swallow
from looper.result import Result
from looper.telemetry import get_health_summary
from looper.zones import get_zone_status_line


def _get_memory_percent_macos() -> int | None:
    """Get memory usage percent on macOS via vm_stat.

    Contracts:
        ENSURES: Returns int in range [0, 100] or None
        ENSURES: Result bounded by math: (active + wired) / total <= 1

    Returns:
        Memory percent used (0-100), or None if unavailable.
    """
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None

        lines = result.stdout.split("\n")
        free = active = inactive = wired = 0
        for line in lines:
            try:
                if "Pages free:" in line:
                    free = int(line.split()[-1].rstrip("."))
                elif "Pages active:" in line:
                    active = int(line.split()[-1].rstrip("."))
                elif "Pages inactive:" in line:
                    inactive = int(line.split()[-1].rstrip("."))
                elif "Pages wired" in line:
                    wired = int(line.split()[-1].rstrip("."))
            except (ValueError, IndexError):
                pass

        total = free + active + inactive + wired
        if total > 0:
            return int((active + wired) * 100 / total)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _get_memory_percent_linux() -> int | None:
    """Get memory usage percent on Linux via /proc/meminfo.

    Contracts:
        ENSURES: Returns int in range [0, 100] or None
        ENSURES: mem_available > mem_total clamped to 0% (pathological state)
        ENSURES: Negative used values clamped to 0%

    Returns:
        Memory percent used (0-100), or None if unavailable.
    """
    try:
        meminfo = Path("/proc/meminfo").read_text()
        mem_total = mem_available = 0
        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1])  # KB
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])  # KB
        if mem_total > 0:
            used = mem_total - mem_available
            # Clamp to [0, 100] - handles pathological states where
            # mem_available > mem_total (kernel reporting quirks)
            pct = int(used * 100 / mem_total)
            return max(0, min(100, pct))
    except (FileNotFoundError, ValueError, IndexError, OSError):
        pass
    return None


def _get_memory_percent() -> int | None:
    """Get memory usage percent, cross-platform.

    Tries macOS first (via vm_stat), then Linux (via /proc/meminfo).
    Windows is not supported and returns None.

    Returns:
        Memory percent used (0-100), or None if unavailable.
    """
    system = platform.system()
    if system == "Darwin":
        return _get_memory_percent_macos()
    if system == "Linux":
        return _get_memory_percent_linux()
    # Windows not supported
    return None


def get_system_status() -> Result[str]:
    """Get concise system status for session start.

    Contracts:
        ENSURES: Returns Result.success with "Mem: XX% | Disk: XX%" format
        ENSURES: Returns Result.success("") if stats unavailable
        ENSURES: Includes warning suffix if memory >= 80%
        ENSURES: Returns Result.failure on error
        ENSURES: Includes looper health if available

    Returns one line: "Mem: XX% | Disk: XX%" or empty if unavailable.
    """
    try:
        parts = []

        # Memory - cross-platform (macOS via vm_stat, Linux via /proc/meminfo)
        mem_pct = _get_memory_percent()
        if mem_pct is not None:
            parts.append(f"Mem: {mem_pct}%")

        # Disk
        disk = shutil.disk_usage(".")
        disk_pct = int(disk.used * 100 / disk.total)
        parts.append(f"Disk: {disk_pct}%")

        if not parts:
            return Result.success("")

        status = " | ".join(parts)

        # Add warning if memory is high
        if mem_pct is not None:
            if mem_pct >= 90:
                status += " ⚠️ CRITICAL"
            elif mem_pct >= 80:
                status += " ⚠️ HIGH"

        # Add looper health summary if available
        try:
            looper_health = get_health_summary(24)
            if looper_health:
                status += f"\nLooper: {looper_health}"
        except Exception as e:
            debug_swallow("get_health_summary", e)

        # Add zone status for multi-worker mode
        try:
            worker_id_str = os.environ.get("AI_WORKER_ID")
            if worker_id_str:
                worker_id = int(worker_id_str)
                zone_line = get_zone_status_line(worker_id)
                status += f"\n{zone_line}"
        except (ValueError, Exception) as e:
            debug_swallow("get_zone_status_line", e)

        # Add uncommitted changes warning if threshold exceeded (#1104)
        uncommitted_warning = get_uncommitted_changes_warning()
        if uncommitted_warning:
            status += f"\n{uncommitted_warning}"

        return Result.success(status)

    except Exception as exc:
        return Result.failure(f"system status unavailable: {exc}")


def truncate_output(
    output: str, max_lines: int = 200, max_chars: int = 8000
) -> tuple[str, bool]:
    """Truncate output to fit within limits.

    Contracts:
        REQUIRES: output is a string (empty string OK)
        REQUIRES: max_lines > 0
        REQUIRES: max_chars > 0
        ENSURES: Output length <= max_chars
        ENSURES: Output line count <= max_lines
        ENSURES: Returns (truncated_output, was_truncated)
    """
    # Contract: validate inputs
    if not isinstance(output, str):
        return ("", False)
    if max_lines <= 0:
        max_lines = 1
    if max_chars <= 0:
        max_chars = 1

    lines = output.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    output = "\n".join(lines)
    if len(output) > max_chars:
        output = output[:max_chars]
        truncated = True
    return output, truncated


def run_system_health_check(
    timeout_sec: int | None = None,
) -> Result[tuple[int, str] | None]:
    """Run system_health_check.py if present and return a Result.

    Contracts:
        REQUIRES: timeout_sec > 0 (clamped to 1 if invalid)
        ENSURES: Returns Result.skip if script not found
        ENSURES: Returns Result.success((returncode, output)) on completion
        ENSURES: Returns Result.failure on timeout or error
        ENSURES: Output truncated to prevent memory issues
        ENSURES: Never raises - all exceptions caught
    """
    if timeout_sec is None:
        timeout_sec = load_timeout_config().get("health_check", 120)

    # Contract: validate timeout
    if timeout_sec <= 0:
        timeout_sec = 1

    system_health_script = Path("scripts/system_health_check.py")
    if not system_health_script.exists():
        return Result.skip("system_health_check.py not found")
    try:
        result = subprocess.run(
            [sys.executable, str(system_health_script)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")
        output = "\n".join(output_parts)
        if output:
            output, truncated = truncate_output(output)
            if truncated:
                output += "\n... (truncated)"
        return Result.success((result.returncode, output))
    except subprocess.TimeoutExpired:
        return Result.failure(f"system_health_check.py timed out after {timeout_sec}s")
    except Exception as exc:
        return Result.failure(f"Failed to run system_health_check.py: {exc}")

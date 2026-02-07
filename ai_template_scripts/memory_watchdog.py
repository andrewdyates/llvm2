#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Memory pressure watchdog for macOS.

Monitors system memory pressure and kills runaway AI-spawned processes before
kernel panic. Designed to run alongside looper.py or as a standalone daemon.

Ported from zani (commit 47f68471) per issue #1468.

Usage:
    # Run as daemon (monitors continuously)
    ./memory_watchdog.py --daemon

    # Single check (for cron or testing)
    ./memory_watchdog.py --check

    # Check with custom threshold
    ./memory_watchdog.py --check --threshold warn

Features:
    - Uses macOS `memory_pressure` command for accurate pressure detection
    - Falls back to `vm_stat` parsing if memory_pressure unavailable
    - Only kills processes matching AI patterns (z4, cbmc, goto-*, rustc, kani)
    - Logs all actions to ~/ait_emergency.log
    - Configurable pressure threshold (warn, critical)
"""

__all__ = [
    "PRESSURE_NORMAL",
    "PRESSURE_WARN",
    "PRESSURE_CRITICAL",
    "DEFAULT_KILL_PATTERNS",
    "get_memory_pressure_macos",
    "get_memory_pressure_vmstat",
    "find_killable_processes",
    "emergency_kill",
    "check_once",
    "run_daemon",
    "main",
]

import argparse
import datetime
import os
import re
import signal
import subprocess

try:
    import tomllib
except ImportError:
    tomllib = None  # Python < 3.11 fallback - config loading will be skipped
import sys
import time
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.subprocess_utils import is_process_alive

# Pressure levels
PRESSURE_NORMAL = "normal"
PRESSURE_WARN = "warn"
PRESSURE_CRITICAL = "critical"

# Default process patterns to kill when under memory pressure
# Only kill AI-spawned tool processes, not user processes
DEFAULT_KILL_PATTERNS = [
    r"z4.*solver",  # Z4 solver
    r"cbmc",  # CBMC verifier
    r"goto-cc",  # GOTO compiler
    r"goto-instrument",  # GOTO instrumentation
    r"goto-synthesizer",  # GOTO synthesizer
    r"rustc",  # Rustc (when spawned by AI)
    r"kani",  # Kani model checker
]

# Log file location
LOG_FILE = Path.home() / "ait_emergency.log"

# Check interval in seconds (when running as daemon)
CHECK_INTERVAL = 5

# Grace period before killing (seconds) - gives processes a chance to finish
GRACE_PERIOD = 2


def log(message: str, level: str = "INFO") -> None:
    """Log a message to the emergency log file and stderr.

    REQUIRES: message is a string (may be empty)
    REQUIRES: level is a log level string (INFO, WARN, ERROR)
    ENSURES: Message printed to stderr with timestamp
    ENSURES: Message appended to LOG_FILE if writable
    ENSURES: Never raises (silently handles write failures)
    """
    timestamp = datetime.datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] memory_watchdog: {message}"

    # Always write to stderr
    print(log_line, file=sys.stderr)

    # Append to log file
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[{timestamp}] [ERROR] Failed to write to log: {e}", file=sys.stderr)


def get_memory_pressure_macos() -> str:
    """Get memory pressure level using macOS memory_pressure command.

    Returns one of: 'normal', 'warn', 'critical'

    REQUIRES: Running on macOS (sys.platform == 'darwin')
    ENSURES: Returns one of PRESSURE_NORMAL, PRESSURE_WARN, or PRESSURE_CRITICAL
    ENSURES: Never raises (returns PRESSURE_NORMAL on error)
    ENSURES: Falls back to vmstat if memory_pressure command unavailable
    """
    try:
        # memory_pressure -S returns a single word: "normal", "warn", or "critical"
        result = subprocess.run(
            ["memory_pressure", "-S"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            pressure = result.stdout.strip().lower()
            if pressure in (PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL):
                return pressure
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to vm_stat parsing
    return get_memory_pressure_vmstat()


def get_memory_pressure_vmstat() -> str:
    """Estimate memory pressure using vm_stat output.

    This is a fallback when memory_pressure command is unavailable.

    Note: Page size varies by architecture:
    - ARM64 (Apple Silicon): 16384 bytes (16KB)
    - x86_64 (Intel): 4096 bytes (4KB)

    Thresholds are defined in bytes to work correctly on both architectures:
    - Critical: < 160MB free
    - Warn: < 800MB free

    REQUIRES: Running on macOS or similar Unix with vm_stat command
    ENSURES: Returns one of PRESSURE_NORMAL, PRESSURE_WARN, or PRESSURE_CRITICAL
    ENSURES: Never raises (returns PRESSURE_NORMAL on error)
    ENSURES: Correctly handles both ARM64 and x86_64 page sizes
    """
    # Thresholds in bytes (architecture-independent)
    CRITICAL_BYTES = 160 * 1024 * 1024  # 160MB
    WARN_BYTES = 800 * 1024 * 1024  # 800MB

    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return PRESSURE_NORMAL

        output = result.stdout

        # Parse page size from header: "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
        page_size = 16384  # Default to ARM64 if parsing fails
        lines = output.splitlines()
        first_line = lines[0] if lines else ""
        page_match = re.search(r"page size of (\d+) bytes", first_line)
        if page_match:
            page_size = int(page_match.group(1))

        # Extract relevant metrics
        free_pages = 0
        speculative_pages = 0

        for line in lines:
            if "Pages free:" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    free_pages = int(match.group(1))
            elif "Pages speculative:" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    speculative_pages = int(match.group(1))

        # Calculate available memory in bytes
        available_pages = free_pages + speculative_pages
        available_bytes = available_pages * page_size

        if available_bytes < CRITICAL_BYTES:
            return PRESSURE_CRITICAL
        if available_bytes < WARN_BYTES:
            return PRESSURE_WARN
        return PRESSURE_NORMAL

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return PRESSURE_NORMAL


def load_kill_patterns(config_path: Path | None = None) -> list[str]:
    """Load kill patterns from config file or use defaults.

    Config format (TOML):
        [memory_watchdog]
        kill_patterns = ["z4.*solver", "cbmc", "rustc"]

    REQUIRES: config_path is None or a Path (may not exist)
    ENSURES: Returns list of regex pattern strings (never empty)
    ENSURES: Returns DEFAULT_KILL_PATTERNS if no config or config invalid
    ENSURES: Never raises (logs warnings on config errors)
    """
    patterns = list(DEFAULT_KILL_PATTERNS)

    if config_path is None:
        # Try default locations
        for name in ["cargo_wrapper.toml", ".cargo_wrapper.toml"]:
            candidate = Path.cwd() / name
            if candidate.exists():
                config_path = candidate
                break

    if config_path and config_path.exists() and tomllib is not None:
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            watchdog_config = config.get("memory_watchdog", {})
            custom_patterns = watchdog_config.get("kill_patterns", [])
            if custom_patterns:
                patterns = custom_patterns
        except Exception as e:
            log(f"Failed to load config from {config_path}: {e}", "WARN")

    return patterns


def find_killable_processes(patterns: list[str] | None = None) -> list[tuple[int, str]]:
    """Find AI-spawned processes that can be killed.

    Returns list of (pid, command) tuples for processes matching patterns.

    REQUIRES: patterns is None or list of valid regex patterns
    ENSURES: Returns list of (pid, command) tuples where pid > 0
    ENSURES: Never includes current process (os.getpid()) in results
    ENSURES: Never raises (returns empty list on error)
    ENSURES: Command strings are truncated to 100 characters
    """
    if patterns is None:
        patterns = load_kill_patterns()

    killable = []

    try:
        # Use ps to find matching processes
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        my_pid = os.getpid()

        for line in result.stdout.splitlines()[1:]:  # Skip header
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue

            try:
                pid = int(parts[0])
                command = parts[1]
            except ValueError:
                continue

            # Don't kill ourselves
            if pid == my_pid:
                continue

            # Check if command matches any kill pattern
            for pattern in patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    killable.append((pid, command[:100]))  # Truncate command
                    break

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return killable


def kill_process(pid: int, command: str) -> bool:
    """Kill a process by PID.

    First tries SIGTERM, then SIGKILL after grace period.

    REQUIRES: pid > 0 (valid process ID)
    REQUIRES: command is for logging only, not used for process lookup
    ENSURES: Returns True if SIGTERM was sent successfully
    ENSURES: Returns False if process already gone or permission denied
    ENSURES: If process survives SIGTERM, SIGKILL is sent after GRACE_PERIOD
    ENSURES: Logs all kill attempts to LOG_FILE
    ENSURES: Never raises (catches ProcessLookupError and PermissionError)
    """
    try:
        # Send SIGTERM first for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        log(f"Sent SIGTERM to PID {pid}: {command}", "WARN")

        # Wait briefly for graceful shutdown
        time.sleep(GRACE_PERIOD)

        # Check if still running, send SIGKILL
        if is_process_alive(pid):
            os.kill(pid, signal.SIGKILL)
            log(f"Sent SIGKILL to PID {pid}: {command}", "WARN")

        return True
    except ProcessLookupError:
        return False  # Process already gone
    except PermissionError:
        log(f"Permission denied killing PID {pid}", "ERROR")
        return False


def emergency_kill(
    threshold: str = PRESSURE_CRITICAL, patterns: list[str] | None = None
) -> int:
    """Kill processes if memory pressure exceeds threshold.

    Returns number of processes killed (0 if below threshold or no matches).

    REQUIRES: threshold is one of PRESSURE_WARN or PRESSURE_CRITICAL
    REQUIRES: patterns is None or list of valid regex patterns
    ENSURES: Returns 0 if pressure is below threshold (no action taken)
    ENSURES: Returns count of processes where kill_process() returned True
    ENSURES: Only kills processes matching patterns (never system processes)
    ENSURES: Logs all kill actions to LOG_FILE
    ENSURES: Never raises (delegates to sub-functions which handle errors)
    """
    pressure = get_memory_pressure_macos()

    # Map threshold string to numeric level for comparison
    pressure_levels = {PRESSURE_NORMAL: 0, PRESSURE_WARN: 1, PRESSURE_CRITICAL: 2}
    current_level = pressure_levels.get(pressure, 0)
    threshold_level = pressure_levels.get(threshold, 2)

    if current_level < threshold_level:
        return 0

    log(f"Memory pressure is {pressure.upper()}, looking for processes to kill", "WARN")

    killable = find_killable_processes(patterns)
    if not killable:
        log("No killable AI processes found", "INFO")
        return 0

    killed = 0
    for pid, command in killable:
        if kill_process(pid, command):
            killed += 1
            log(f"EMERGENCY KILL: PID {pid} ({command})", "CRITICAL")

    if killed > 0:
        log(f"Killed {killed} processes due to {pressure} memory pressure", "CRITICAL")

    return killed


def check_once(
    threshold: str = PRESSURE_CRITICAL,
    quiet: bool = False,
    patterns: list[str] | None = None,
) -> int:
    """Run a single check and return exit code.

    Exit codes:
        0 - Normal pressure, no action needed
        1 - Elevated pressure, processes killed
        2 - Error

    REQUIRES: threshold is one of PRESSURE_WARN or PRESSURE_CRITICAL
    REQUIRES: patterns is None or list of valid regex patterns
    ENSURES: Returns 0 if pressure normal and no kills needed
    ENSURES: Returns 1 if processes were killed
    ENSURES: Returns 2 on error
    ENSURES: Prints pressure level unless quiet=True
    """
    try:
        pressure = get_memory_pressure_macos()

        if not quiet:
            print(f"Memory pressure: {pressure}")

        killed = emergency_kill(threshold, patterns)

        if killed > 0:
            return 1
        return 0

    except Exception as e:
        log(f"Error during check: {e}", "ERROR")
        return 2


def run_daemon(
    threshold: str = PRESSURE_CRITICAL, patterns: list[str] | None = None
) -> None:
    """Run as a continuous daemon, checking memory pressure periodically.

    REQUIRES: threshold is one of PRESSURE_WARN or PRESSURE_CRITICAL
    REQUIRES: patterns is None or list of valid regex patterns
    ENSURES: Runs indefinitely until KeyboardInterrupt
    ENSURES: Checks memory pressure every CHECK_INTERVAL seconds
    ENSURES: Logs daemon start and stop events
    """
    log(f"Starting memory watchdog daemon (threshold={threshold})", "INFO")

    try:
        while True:
            emergency_kill(threshold, patterns)
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        log("Memory watchdog stopped by user", "INFO")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory pressure watchdog for macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as continuous daemon",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run single check and exit",
    )
    parser.add_argument(
        "--threshold",
        choices=["warn", "critical"],
        default="critical",
        help="Memory pressure threshold for killing processes (default: critical)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List killable processes and exit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file path (default: cargo_wrapper.toml in cwd)",
    )

    args = parser.parse_args()

    # Platform check
    if sys.platform != "darwin":
        print("Error: This script only works on macOS", file=sys.stderr)
        return 2

    # Load patterns from config
    patterns = load_kill_patterns(args.config)

    if args.list:
        processes = find_killable_processes(patterns)
        if not processes:
            print("No killable AI processes found")
        else:
            print(f"Found {len(processes)} killable processes:")
            for pid, cmd in processes:
                print(f"  PID {pid}: {cmd}")
        return 0

    if args.daemon:
        run_daemon(args.threshold, patterns)
        return 0

    if args.check:
        return check_once(args.threshold, args.quiet, patterns)

    # Default: show current pressure
    pressure = get_memory_pressure_macos()
    print(f"Memory pressure: {pressure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
log_test.py - Log test invocations for manager audit

CANONICAL SOURCE: ayates_dbx/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Records test runs to logs/tests.jsonl for pattern detection:
- Multiple AIs running same tests concurrently
- Test duration trends (getting slower?)
- Frequent test failures
- Resource-intensive test patterns

Usage:
    # Log a test run (call before/after test)
    ./ai_template_scripts/log_test.py start "cargo test"
    ./ai_template_scripts/log_test.py end 0  # exit code

    # Or wrap a command (logs start/end automatically)
    ./ai_template_scripts/log_test.py run "cargo test -p solver"

    # Analyze recent tests
    ./ai_template_scripts/log_test.py report

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
TEST_LOG = LOGS_DIR / "tests.jsonl"
ACTIVE_FILE = LOGS_DIR / ".active_test"


def get_session_info() -> dict:
    """Get current session info from environment."""
    return {
        "role": os.environ.get("AI_ROLE", "USER").upper(),
        "session": os.environ.get("AI_SESSION", "unknown"),
        "iteration": os.environ.get("AI_ITERATION", "?"),
    }


def log_entry(entry: dict):
    """Append entry to test log."""
    LOGS_DIR.mkdir(exist_ok=True)
    with open(TEST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def cmd_start(command: str):
    """Log test start."""
    info = get_session_info()
    entry = {
        "event": "start",
        "timestamp": datetime.now().isoformat(),
        "command": command,
        **info,
    }
    log_entry(entry)

    # Write active file for concurrent detection
    ACTIVE_FILE.write_text(
        json.dumps(
            {
                "command": command,
                "started": datetime.now().isoformat(),
                **info,
            }
        )
    )
    print(f"[test-log] Started: {command}")


def cmd_end(exit_code: int):
    """Log test end."""
    info = get_session_info()

    # Calculate duration from active file if present
    duration_s = None
    if ACTIVE_FILE.exists():
        try:
            active = json.loads(ACTIVE_FILE.read_text())
            started = datetime.fromisoformat(active["started"])
            duration_s = round((datetime.now() - started).total_seconds(), 1)
            command = active.get("command", "unknown")
        except (json.JSONDecodeError, KeyError, ValueError):
            command = "unknown"
        ACTIVE_FILE.unlink(missing_ok=True)
    else:
        command = "unknown"

    entry = {
        "event": "end",
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "exit_code": exit_code,
        "duration_s": duration_s,
        **info,
    }
    log_entry(entry)
    status = "passed" if exit_code == 0 else f"failed (exit {exit_code})"
    duration_str = f" ({duration_s}s)" if duration_s else ""
    print(f"[test-log] Ended: {command} - {status}{duration_str}")


def cmd_run(command: str):
    """Run a test command with automatic logging."""
    cmd_start(command)
    start = time.time()

    # Run the command
    result = subprocess.run(command, shell=True)

    duration_s = round(time.time() - start, 1)

    # Log end with duration
    info = get_session_info()
    entry = {
        "event": "end",
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "exit_code": result.returncode,
        "duration_s": duration_s,
        **info,
    }
    log_entry(entry)
    ACTIVE_FILE.unlink(missing_ok=True)

    status = (
        "passed" if result.returncode == 0 else f"failed (exit {result.returncode})"
    )
    print(f"[test-log] Completed: {command} - {status} ({duration_s}s)")

    return result.returncode


def cmd_report():
    """Generate report from test log."""
    if not TEST_LOG.exists():
        print("No test log found.")
        return

    entries = []
    for line in TEST_LOG.read_text().strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not entries:
        print("No test entries found.")
        return

    # Filter to end events (have duration)
    end_events = [e for e in entries if e.get("event") == "end"]

    print(f"Test Log Report ({len(end_events)} runs)")
    print("=" * 60)

    # Recent tests (last 10)
    print("\nRecent tests:")
    for e in end_events[-10:]:
        ts = e.get("timestamp", "?")[:19]
        cmd = e.get("command", "?")[:40]
        dur = e.get("duration_s", "?")
        code = e.get("exit_code", "?")
        role = e.get("role", "?")
        status = "✓" if code == 0 else "✗"
        print(f"  {ts} [{role}] {status} {cmd} ({dur}s)")

    # Failures
    failures = [e for e in end_events if e.get("exit_code", 0) != 0]
    if failures:
        print(
            f"\nFailures: {len(failures)}/{len(end_events)} ({100 * len(failures) // len(end_events)}%)"
        )

    # Concurrent tests (look for overlapping start/end)
    print("\nConcurrent test detection:")

    # Build list of (start_time, end_time, session, role, command) tuples
    test_runs = []
    starts = {
        (e.get("session"), e.get("command")): e
        for e in entries
        if e.get("event") == "start"
    }
    for e in entries:
        if e.get("event") == "end":
            key = (e.get("session"), e.get("command"))
            start_entry = starts.get(key)
            if start_entry:
                test_runs.append(
                    {
                        "start": start_entry.get("timestamp", ""),
                        "end": e.get("timestamp", ""),
                        "session": e.get("session"),
                        "role": e.get("role"),
                        "command": e.get("command"),
                    }
                )

    # Find actual overlaps
    overlaps = [
        (run, other)
        for i, run in enumerate(test_runs)
        for other in test_runs[:i]
        if (
            run["start"] < other["end"]
            and run["end"] > other["start"]
            and run["session"] != other["session"]
        )
    ]

    if overlaps:
        print(f"  Found {len(overlaps)} overlapping test runs:")
        for run, other in overlaps[-5:]:
            print(f"    [{run['role']}:{run['session'][:8]}] {run['command'][:30]}")
            print(
                f"      overlapped with [{other['role']}:{other['session'][:8]}] {other['command'][:30]}"
            )
    else:
        print("  No concurrent test runs detected")

    # Duration trends
    if len(end_events) >= 5:
        durations = [e.get("duration_s", 0) for e in end_events if e.get("duration_s")]
        if durations:
            avg = sum(durations) / len(durations)
            recent_avg = sum(durations[-5:]) / min(5, len(durations))
            print("\nDuration trends:")
            print(f"  Overall avg: {avg:.1f}s")
            print(f"  Recent avg (last 5): {recent_avg:.1f}s")
            if recent_avg > avg * 1.5:
                print("  ⚠️  Tests getting slower!")


def main():
    if len(sys.argv) < 2:
        print("Usage: log_test.py <start|end|run|report> [args]")
        print("  start <command>  - Log test start")
        print("  end <exit_code>  - Log test end")
        print("  run <command>    - Run command with logging")
        print("  report           - Show test log analysis")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start" and len(sys.argv) >= 3:
        cmd_start(" ".join(sys.argv[2:]))
    elif cmd == "end" and len(sys.argv) >= 3:
        cmd_end(int(sys.argv[2]))
    elif cmd == "run" and len(sys.argv) >= 3:
        exit_code = cmd_run(" ".join(sys.argv[2:]))
        sys.exit(exit_code)
    elif cmd == "report":
        cmd_report()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

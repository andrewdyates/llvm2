#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
log_test.py - Log test invocations for manager audit

CANONICAL SOURCE: dropbox-ai-prototypes/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Records test runs to logs/tests.jsonl for pattern detection:
- Multiple AIs running same tests concurrently
- Test duration trends (getting slower?)
- Frequent test failures
- Resource-intensive test patterns

Public API (library usage):
    from ai_template_scripts.log_test import (
        get_session_info,  # Get AI session info from env
        log_entry,         # Append entry to test log
        cmd_start,         # Log test start
        cmd_end,           # Log test end
        cmd_run,           # Run command with logging
        cmd_report,        # Generate report from log
    )

CLI usage:
    ./ai_template_scripts/log_test.py start "cargo test"
    ./ai_template_scripts/log_test.py end 0  # exit code
    ./ai_template_scripts/log_test.py run "cargo test -p solver"
    ./ai_template_scripts/log_test.py report
    ./ai_template_scripts/log_test.py --version

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

__all__ = [
    "get_session_info",
    "log_entry",
    "cmd_start",
    "cmd_end",
    "cmd_run",
    "cmd_report",
    "main",
]

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.subprocess_utils import run_cmd  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

LOGS_DIR = Path("logs")
TEST_LOG = LOGS_DIR / "tests.jsonl"
ACTIVE_FILE = LOGS_DIR / ".active_test"

# Known removed or stale pytest targets with canonical replacements (#2881)
PYTEST_REMOVED_TARGETS = {
    "tests/test_telemetry.py": (
        "tests/test_telemetry_core.py",
        "tests/test_telemetry_health.py",
        "tests/test_telemetry_tokens.py",
        "tests/test_looper/test_telemetry.py",
    ),
}
PYTEST_SHELL_SEPARATORS = {"|", "||", "&&", ";"}
PYTEST_OPTIONS_WITH_VALUE = {
    "-c",
    "-k",
    "-m",
    "--basetemp",
    "--capture",
    "--confcutdir",
    "--durations",
    "--durations-min",
    "--ignore",
    "--ignore-glob",
    "--junitxml",
    "--log-cli-date-format",
    "--log-cli-format",
    "--log-cli-level",
    "--log-date-format",
    "--log-file",
    "--log-file-date-format",
    "--log-file-format",
    "--log-file-level",
    "--log-format",
    "--log-level",
    "--maxfail",
    "--override-ini",
    "--rootdir",
    "--tb",
}
REPO_ROOT = _repo_root.resolve()


def get_session_info() -> dict:
    """Get current session info from environment."""
    return {
        "role": os.environ.get("AI_ROLE", "USER").upper(),
        "session": os.environ.get("AI_SESSION", "unknown"),
        "iteration": os.environ.get("AI_ITERATION", "?"),
    }


def log_entry(entry: dict) -> None:
    """Append entry to test log."""
    LOGS_DIR.mkdir(exist_ok=True)
    with open(TEST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def cmd_start(command: str) -> None:
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


def cmd_end(exit_code: int) -> None:
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


# Max bytes to capture from stderr for failed test analysis (#924)
STDERR_TAIL_BYTES = 4096


def _get_pytest_args_start(tokens: list[str]) -> int | None:
    """Return index where pytest args begin, or None if command is not pytest."""
    if not tokens:
        return None

    exe_name = Path(tokens[0]).name
    if exe_name == "pytest":
        return 1

    for i, token in enumerate(tokens[:-1]):
        if token == "-m" and tokens[i + 1] == "pytest":
            return i + 2

    return None


def _iter_pytest_targets(tokens: list[str]) -> list[str]:
    """Extract explicit pytest path targets from argument tokens."""
    targets = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue

        if token in PYTEST_SHELL_SEPARATORS:
            break
        if token == "--":
            continue
        if token.startswith("-"):
            if token in PYTEST_OPTIONS_WITH_VALUE:
                skip_next = True
            continue

        candidate = token.split("::", 1)[0]
        if not candidate:
            continue
        if any(char in candidate for char in "*?[]"):
            continue

        normalized = candidate.replace("\\", "/")
        if (
            "/" not in normalized
            and not normalized.endswith(".py")
            and normalized != "tests"
        ):
            continue

        targets.append(candidate)

    return targets


def _target_match_keys(target: str) -> set[str]:
    """Return normalized target variants for removed-target matching."""
    match_keys = set()

    normalized = target.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    match_keys.add(normalized)

    path = Path(target)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        rel = path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
        match_keys.add(rel)
    except ValueError:
        pass

    return match_keys


def _target_exists(target: str) -> bool:
    """Check target existence from CWD and repo root."""
    path = Path(target)
    if path.exists():
        return True
    if not path.is_absolute() and (REPO_ROOT / path).exists():
        return True
    return False


def _validate_pytest_targets(command: str) -> tuple[bool, str | None]:
    """Detect stale or missing pytest path targets before execution."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Let subprocess handle malformed command strings.
        return True, None

    args_start = _get_pytest_args_start(tokens)
    if args_start is None:
        return True, None

    for target in _iter_pytest_targets(tokens[args_start:]):
        match_keys = _target_match_keys(target)
        removed_target = next(
            (key for key in match_keys if key in PYTEST_REMOVED_TARGETS), None
        )
        if removed_target:
            replacements = PYTEST_REMOVED_TARGETS[removed_target]
            replacement_text = " ".join(replacements)
            return (
                False,
                f"Refusing pytest run: '{removed_target}' is a removed target. "
                f"Use: {replacement_text}",
            )

        is_tests_target = any(
            key == "tests" or key.startswith("tests/") for key in match_keys
        )
        if is_tests_target and not _target_exists(target):
            return False, f"Refusing pytest run: target '{target}' does not exist."

    return True, None


def cmd_run(command: str) -> int:
    """Run a test command with automatic logging.

    For failed commands, captures stderr tail to help distinguish command
    errors (wrong args) from actual test failures (#924).

    Uses subprocess_utils.run_cmd for consistent timeout handling when possible,
    falls back to shell execution for complex commands with shell features.
    """
    cmd_start(command)
    start = time.time()

    targets_ok, target_error = _validate_pytest_targets(command)
    if not targets_ok:
        duration_s = round(time.time() - start, 1)
        info = get_session_info()
        entry = {
            "event": "end",
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "exit_code": 2,
            "duration_s": duration_s,
            "error_type": "command_error",
            **info,
        }
        if target_error:
            entry["stderr_tail"] = target_error[-STDERR_TAIL_BYTES:]
            print(target_error, file=sys.stderr)
        log_entry(entry)
        ACTIVE_FILE.unlink(missing_ok=True)
        print(f"[test-log] Completed: {command} - failed (exit 2) ({duration_s}s)")
        return 2

    # Check if command needs shell features
    shell_chars = {"|", ">", "<", "&", ";", "*", "?", "$", "`", "(", ")"}
    needs_shell = any(c in command for c in shell_chars)

    if needs_shell:
        # Complex command - use shell
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=600,  # 10 min timeout for tests
        )
        stdout, stderr = result.stdout, result.stderr
        returncode = result.returncode
    else:
        # Simple command - use run_cmd for consistent timeout handling
        try:
            cmd_list = shlex.split(command)
            result = run_cmd(cmd_list, timeout=600)  # 10 min timeout for tests
            stdout, stderr = result.stdout, result.stderr
            returncode = result.returncode
        except ValueError:
            # Fallback if shlex fails
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=600,  # 10 min timeout for tests
            )
            stdout, stderr = result.stdout, result.stderr
            returncode = result.returncode

    duration_s = round(time.time() - start, 1)

    # Stream stdout/stderr to console (since we captured them)
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)

    # Log end with duration and optional stderr tail for failures
    info = get_session_info()
    entry = {
        "event": "end",
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "exit_code": returncode,
        "duration_s": duration_s,
        **info,
    }

    # Capture stderr tail for failed runs to help classify failures (#924)
    if returncode != 0:
        if stderr:
            stderr_tail = stderr[-STDERR_TAIL_BYTES:]
            entry["stderr_tail"] = stderr_tail

            # Detect command/usage errors vs actual test failures
            usage_error_patterns = [
                "unexpected argument",
                "Found argument",
                "unrecognized option",
                "invalid option",
                "Usage:",
                "error: unrecognized",
                "missing '--'",
                "unknown flag",
            ]
            for pattern in usage_error_patterns:
                if pattern.lower() in stderr_tail.lower():
                    entry["error_type"] = "command_error"
                    break
            else:
                entry["error_type"] = "test_failure"
        else:
            # No stderr but still failed - classify as test_failure
            entry["error_type"] = "test_failure"

    log_entry(entry)
    ACTIVE_FILE.unlink(missing_ok=True)

    status = "passed" if returncode == 0 else f"failed (exit {returncode})"
    print(f"[test-log] Completed: {command} - {status} ({duration_s}s)")

    return returncode


def _print_recent_tests(end_events: list) -> None:
    """Print recent tests section."""
    print("\nRecent tests:")
    for e in end_events[-10:]:
        ts = e.get("timestamp", "?")[:19]
        cmd = e.get("command", "?")[:40]
        dur = e.get("duration_s", "?")
        code = e.get("exit_code", "?")
        role = e.get("role", "?")
        status = "✓" if code == 0 else "✗"
        print(f"  {ts} [{role}] {status} {cmd} ({dur}s)")


def _print_failure_summary(end_events: list) -> None:
    """Print failure summary section with error type breakdown (#924)."""
    failures = [e for e in end_events if e.get("exit_code", 0) != 0]
    if failures:
        total = len(end_events)
        print(f"\nFailures: {len(failures)}/{total} ({100 * len(failures) // total}%)")

        # Break down by error type if classified (#924)
        command_errors = [f for f in failures if f.get("error_type") == "command_error"]
        test_failures = [f for f in failures if f.get("error_type") == "test_failure"]
        unclassified = [f for f in failures if "error_type" not in f]

        if command_errors or test_failures:
            print(f"  - Command errors: {len(command_errors)}")
            print(f"  - Test failures: {len(test_failures)}")
            if unclassified:
                print(f"  - Unclassified: {len(unclassified)}")


def _detect_concurrent_tests(entries: list) -> None:
    """Detect and print concurrent test runs."""
    print("\nConcurrent test detection:")

    # Build test_runs from matched start/end pairs
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
                f"      overlapped with [{other['role']}:{other['session'][:8]}] "
                f"{other['command'][:30]}"
            )
    else:
        print("  No concurrent test runs detected")


def _print_duration_trends(end_events: list) -> None:
    """Print duration trends section."""
    if len(end_events) < 5:
        return

    durations = [e.get("duration_s", 0) for e in end_events if e.get("duration_s")]
    if durations:
        avg = sum(durations) / len(durations)
        recent_avg = sum(durations[-5:]) / min(5, len(durations))
        print("\nDuration trends:")
        print(f"  Overall avg: {avg:.1f}s")
        print(f"  Recent avg (last 5): {recent_avg:.1f}s")
        if recent_avg > avg * 1.5:
            print("  ⚠️  Tests getting slower!")


def cmd_report() -> None:
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

    end_events = [e for e in entries if e.get("event") == "end"]
    print(f"Test Log Report ({len(end_events)} runs)")
    print("=" * 60)

    _print_recent_tests(end_events)
    _print_failure_summary(end_events)
    _detect_concurrent_tests(entries)
    _print_duration_trends(end_events)


def main() -> None:
    if "--version" in sys.argv:
        print(get_version("log_test.py"))
        return

    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: log_test.py <start|end|run|report> [args]")
        print("  start <command>  - Log test start")
        print("  end <exit_code>  - Log test end")
        print("  run <command>    - Run command with logging")
        print("  report           - Show test log analysis")
        print("  --version        - Show version information")
        print("  -h, --help       - Show this help message")
        if len(sys.argv) >= 2:
            return
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

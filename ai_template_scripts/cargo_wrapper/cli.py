# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""CLI entry point and main dispatch logic for cargo_wrapper."""

from __future__ import annotations

import atexit
import os
import shlex
import signal
import sys
import time
from datetime import UTC, datetime

from . import _state
from .constants import (
    _ARGS_WITH_VALUES,
    LOCK_ACQUIRE_TIMEOUT,
    LOCK_KIND_TEST,
    MAX_STALE_RELEASE_ATTEMPTS,
    STATUS_INTERVAL,
)
from .env import get_env_context, get_version
from .executor import (
    atexit_handler,
    find_built_binary,
    find_real_cargo,
    parse_cargo_run_args,
    run_binary,
    run_cargo,
    signal_handler,
)
from .lock import (
    acquire_lock,
    cleanup_stale_temp_files,
    force_release_stale_lock,
    get_lock_file,
    get_lock_holder_info,
    init_lock_paths,
    is_lock_stale,
    release_lock,
)
from .cache import check_cache, check_pulse_cache, print_cache_hit, store_cache
from .logging import check_retry_loop, log_build, log_stderr
from .processes import cleanup_orphans, is_ancestor_of_self
from .timeouts import (
    get_cargo_subcommand,
    get_lock_kind_for_command,
    get_timeout_config,
    select_cargo_timeout,
)

__all__ = ["main", "_has_test_filter_arg"]


def ensure_lock_dir() -> bool:
    """Ensure lock directory exists and is writable. Returns False if unusable."""
    if _state.LOCK_DIR is None:
        return False
    test_file = _state.LOCK_DIR / f".write_test.{os.getpid()}"
    try:
        _state.LOCK_DIR.mkdir(parents=True, exist_ok=True)
        # Test writability with unique filename to avoid race condition
        test_file.write_text("test")
        return True
    except (OSError, PermissionError) as e:
        log_stderr(f"[cargo] WARNING: Lock dir unusable ({e}), no lock")
        return False
    finally:
        test_file.unlink(missing_ok=True)


def _has_test_filter_arg(args: list[str]) -> bool:
    """Check if cargo test args include a test filter (positional argument).

    Detects positional filter argument after 'test' subcommand, e.g.:
    - cargo test my_test        -> True (filter=my_test)
    - cargo test -p crate       -> False (no filter)
    - cargo test -p crate my_test -> True (filter=my_test)
    - cargo test -- --nocapture -> False (-- and after are ignored)

    Returns True if a test filter argument is present. Part of #1469.
    """
    # Args that consume the next positional: -p <crate>, --package <crate>, etc.
    # Note: -p=<crate> and --package=<crate> don't consume next arg
    # --test and --bench consume the target name (#1505)
    skip_next = False
    found_test = False
    found_double_dash = False

    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            found_double_dash = True
            continue
        if found_double_dash:
            # Everything after -- is passed to test binary, not cargo
            continue
        if arg.startswith("-"):
            # Check if this flag consumes the next argument
            if arg in _ARGS_WITH_VALUES:
                skip_next = True
            continue
        # Positional argument
        if not found_test:
            # First positional is subcommand (test)
            found_test = arg == "test"
            continue
        # After 'test', any positional is a filter
        return True

    return False


def _fallback_exec(args: list[str], error_msg: str) -> int:
    """Execute cargo directly without locking. Returns exit code or exits via execv.

    Args:
        args: Command-line arguments to pass to cargo.
        error_msg: Error message to show if cargo is not found.
    """
    cargo = os.environ.get("AIT_REAL_CARGO") or find_real_cargo()
    if cargo:
        os.execv(cargo, [cargo] + args)
    log_stderr(error_msg)
    return 1


def _check_reentrant_bypass(args: list[str]) -> bool:
    """Check if current process is nested under lock holder and bypass if so.

    Returns True if bypassed (process replaced via execv), False if not nested.
    """
    lock_file = get_lock_file()
    if not lock_file.exists():
        return False
    try:
        lock_pid = int(lock_file.read_text().strip())
        if is_ancestor_of_self(lock_pid):
            log_stderr(f"[cargo] Nested (ancestor {lock_pid}), bypass")
            cargo = os.environ.get("AIT_REAL_CARGO") or find_real_cargo()
            if cargo:
                os.execv(cargo, [cargo] + args)
            log_stderr("[cargo] ERROR: cargo not found for nested exec")
            return True  # Indicates bypass was attempted but failed
    except (ValueError, FileNotFoundError):
        pass  # Lock file invalid or disappeared, continue normal flow
    return False


def _wait_for_lock(context: dict[str, object], lock_kind: str) -> bool:
    """Wait for lock acquisition with timeout and status updates.

    Args:
        context: Environment context dict for lock metadata.

    Returns:
        True if lock was acquired, False if timed out.
    """
    start_wait = time.time()
    last_status = 0.0
    printed_initial = False
    stale_release_attempts = 0
    project = str(context.get("project") or "?")

    while time.time() - start_wait < LOCK_ACQUIRE_TIMEOUT:
        if acquire_lock(context):
            return True

        # Check staleness IMMEDIATELY before waiting (detect dead processes fast)
        # Use verbose=True on first check for diagnostics (#1679)
        verbose_stale = stale_release_attempts == 0
        if stale_release_attempts < MAX_STALE_RELEASE_ATTEMPTS and is_lock_stale(
            verbose=verbose_stale
        ):
            if force_release_stale_lock():
                stale_release_attempts += 1
                continue  # Retry acquire immediately, don't sleep

        # Print status: verbose on first block, brief on subsequent
        if not printed_initial:
            holder = get_lock_holder_info(verbose=True)
            log_stderr(
                f"[cargo-lock] Waiting for {lock_kind} lock on {project} (held by {holder})..."
            )
            printed_initial = True

        elapsed = time.time() - start_wait
        if elapsed - last_status >= STATUS_INTERVAL:
            holder = get_lock_holder_info()
            log_stderr(
                f"[cargo-lock] Waiting for lock on {project} ({lock_kind}, {int(elapsed)}s, held by {holder})..."
            )
            last_status = elapsed

        time.sleep(1)

    log_stderr(
        f"[cargo-lock] ERROR: Lock timeout on {project} ({LOCK_ACQUIRE_TIMEOUT}s)"
    )
    return False


def _run_cargo_run(
    args: list[str],
    context: dict[str, object],
    timeouts: dict[str, int],
) -> int:
    """Handle 'cargo run': build under lock, run binary without lock.

    Args:
        args: Full cargo run arguments.
        context: Environment context dict.

    Returns:
        Exit code from build or binary execution.
    """
    build_args, binary_args, package_or_bin = parse_cargo_run_args(args)
    build_command_str = shlex.join(["cargo"] + build_args)

    check_retry_loop(build_command_str, os.getcwd(), str(context.get("commit", "")))

    started = datetime.now(UTC)
    log_stderr(f"[cargo] Lock acquired, building: {build_command_str}")
    build_timeout = timeouts["build"]
    build_exit_code = run_cargo(build_args, build_timeout)
    finished = datetime.now(UTC)
    log_build(
        context,
        ["cargo"] + build_args,
        started,
        finished,
        build_exit_code,
        build_timeout,
    )

    if build_exit_code != 0:
        release_lock()
        return build_exit_code

    # Find the built binary
    binary_path = find_built_binary(package_or_bin, args)

    # Release lock before running the binary
    release_lock()
    log_stderr("[cargo] Lock released, running binary outside lock")

    if binary_path:
        # Run binary without lock
        log_stderr(f"[cargo] Running: {binary_path} {shlex.join(binary_args)}")
        return run_binary(binary_path, binary_args, build_timeout)

    # Fallback: couldn't find binary, run cargo run directly (rare edge case)
    log_stderr(
        "[cargo] WARNING: Could not locate built binary, running cargo run directly"
    )
    return run_cargo(args, build_timeout)


def _run_standard_cargo(
    args: list[str],
    context: dict[str, object],
    timeouts: dict[str, int],
) -> int:
    """Run standard cargo command with lock held.

    Args:
        args: Cargo command arguments.
        context: Environment context dict.

    Returns:
        Exit code from cargo command.
    """
    try:
        command_str = shlex.join(["cargo"] + args)

        # Check for retry loop before running (warn only, don't block)
        check_retry_loop(command_str, os.getcwd(), str(context.get("commit", "")))

        # Use shorter timeout for kani/zani (proofs can hang indefinitely)
        timeout = select_cargo_timeout(args, timeouts)

        started = datetime.now(UTC)
        lock_label = "test" if _state.LOCK_KIND == LOCK_KIND_TEST else "build"
        log_stderr(f"[cargo] Lock ({lock_label}) acquired, running: {command_str}")
        exit_code = run_cargo(args, timeout)
        finished = datetime.now(UTC)
        log_build(context, ["cargo"] + args, started, finished, exit_code, timeout)
        # Cache successful test results (#3212)
        if exit_code == 0 and _state.LOCK_KIND == LOCK_KIND_TEST:
            duration = (finished - started).total_seconds()
            store_cache(args, str(context.get("commit", "")), exit_code, duration)
        return exit_code
    finally:
        release_lock()


def main() -> int:
    """Main entry point for cargo wrapper."""
    # Handle --version before anything else (GNU standard: exit 0, ignore other args)
    if "--version" in sys.argv:
        print(get_version("cargo_wrapper.py"))
        return 0

    # Register cleanup handlers for all termination signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)  # Ctrl+\
    atexit.register(atexit_handler)

    args = sys.argv[1:]
    subcommand = get_cargo_subcommand(args)

    # Determine lock kind based on command (build vs test)
    lock_kind = get_lock_kind_for_command(args)

    # Initialize lock paths (may fail if HOME not set)
    if not init_lock_paths(lock_kind):
        log_stderr("[cargo] WARNING: HOME not set, running without serialization")
        return _fallback_exec(args, "[cargo] ERROR: Could not find cargo")

    # Ensure lock directory is usable, fall back to direct execution if not
    if not ensure_lock_dir():
        return _fallback_exec(
            args, "[cargo] ERROR: Could not find cargo for fallback execution"
        )

    timeouts = get_timeout_config()
    command_timeout = select_cargo_timeout(args, timeouts)
    context = get_env_context()
    context["lock_kind"] = lock_kind
    context["command_timeout_sec"] = command_timeout

    # Block Worker from running full test suite (Prover's job) - see #909, #928
    role = context.get("role", "USER")
    if role == "WORKER" and subcommand == "test":
        has_specific_test = "--test" in args or any(
            arg.startswith("--test=") for arg in args
        )
        has_crate_flag = "-p" in args or "--package" in args
        # Check for positional filter argument (e.g., "cargo test -p crate filter")
        # Positional args are those that don't start with - and aren't the subcommand
        # or values following -p/--package
        has_filter_arg = _has_test_filter_arg(args)
        # Full suite = testing package/crate without specific test name or filter
        is_full_suite = has_crate_flag and not has_specific_test and not has_filter_arg
        # Also block 'cargo test' with no args (runs entire workspace)
        is_workspace_test = (
            not has_crate_flag
            and not has_specific_test
            and len([a for a in args if not a.startswith("-")]) <= 1
        )  # Only 'test' subcommand, no targets
        if is_full_suite or is_workspace_test:
            # Allow override via environment variable
            if os.environ.get("CARGO_ALLOW_FULL_SUITE") == "1":
                log_stderr(
                    "[cargo] WARNING: Worker running full test suite (override enabled). "
                    "Full suites are Prover's job."
                )
            else:
                log_stderr(
                    "[cargo] ERROR: Worker cannot run full test suites.\n"
                    "  Full test suites are Prover's job.\n"
                    "\n"
                    "  Options:\n"
                    "    1. Run specific test: cargo test --test <test_name>\n"
                    "    2. Run specific function: cargo test <function_name>\n"
                    "    3. Override (not recommended): CARGO_ALLOW_FULL_SUITE=1 cargo test\n"
                    "\n"
                    "  See #928 for rationale."
                )
                return 1

    # Test result cache check — BEFORE lock acquisition (#3212)
    # Only cache test/bench/kani (commands that use the test lock)
    commit = context.get("commit", "")
    if lock_kind == LOCK_KIND_TEST and subcommand != "run":
        cached = check_cache(args, commit)
        if cached is None:
            cached = check_pulse_cache(args, commit)
            if cached is not None:
                print_cache_hit(cached, cache_type="PULSE")
                return 0
        else:
            print_cache_hit(cached)
            return 0

    # Clean up orphan processes before attempting lock
    orphans_killed = cleanup_orphans()
    if orphans_killed > 0:
        log_stderr(f"[cargo] Cleaned up {orphans_killed} orphan process(es)")

    # Check for stale lock (verbose=True for diagnostics #1679)
    lock_file = get_lock_file()
    if lock_file.exists() and is_lock_stale(verbose=True):
        force_release_stale_lock()

    # Clean up orphaned temp files once before acquire loop (not on every iteration)
    cleanup_stale_temp_files()

    # Re-entrant check: ancestor holds lock → bypass (nested cargo calls)
    if _check_reentrant_bypass(args):
        return 1  # Bypass failed (cargo not found)

    # Acquire lock with timeout
    if not _wait_for_lock(context, lock_kind):
        return 1

    # Dispatch based on cargo subcommand
    if subcommand == "run":
        return _run_cargo_run(args, context, timeouts)
    return _run_standard_cargo(args, context, timeouts)


if __name__ == "__main__":
    sys.exit(main())

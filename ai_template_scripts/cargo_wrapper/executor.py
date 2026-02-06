# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Cargo execution and process management for cargo_wrapper.

Handles running cargo commands, signal handling, and process cleanup.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

from . import _state
from .lock import release_lock
from .logging import debug_swallow, log_stderr

__all__ = [
    "_make_process_group",
    "find_real_cargo",
    "run_cargo",
    "run_binary",
    "kill_child",
    "signal_handler",
    "atexit_handler",
    "parse_cargo_run_args",
    "find_built_binary",
]


def _make_process_group() -> None:
    """Preexec function to create new process group without new session (#1490).

    Using os.setpgid(0, 0) instead of start_new_session=True because:
    - start_new_session creates a new SESSION (setsid), making cargo a session leader
    - Session leaders aren't killed when parent shell dies - they become orphans (PPID=1)
    - setpgid only creates a new PROCESS GROUP while staying in the same session
    - Same-session processes are terminated when session leader (shell) dies

    This allows us to:
    1. Kill entire process group on timeout/cleanup (os.killpg works)
    2. Ensure cargo dies if parent session is terminated (e.g., iTerm closed)
    """
    os.setpgid(0, 0)


def kill_child() -> None:
    """Kill the child process group if running.

    ENSURES: _child_process and _child_pgid are set to None on return
    ENSURES: Returns within ~10 seconds maximum:
             - Process group path: 1s (SIGTERM grace) + 5s (final wait) = ~6s
             - Direct kill path: 5s (terminate wait) + 5s (final wait) = ~10s
    """
    if _state._child_process is None:
        return
    try:
        killed_via_pg = False
        # Kill entire process group (cargo + rustc children)
        if _state._child_pgid is not None:
            try:
                os.killpg(_state._child_pgid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.killpg(_state._child_pgid, 0)  # Check if still alive
                    os.killpg(_state._child_pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                killed_via_pg = True
            except (ProcessLookupError, OSError):
                pass  # Fall through to direct kill

        # Fallback: kill just the direct child if pg kill failed or pgid unknown
        if not killed_via_pg:
            _state._child_process.terminate()
            try:
                _state._child_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _state._child_process.kill()

        # Final wait with timeout to prevent indefinite hang (#1265)
        try:
            _state._child_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pid = _state._child_process.pid
            log_stderr(
                f"[cargo] WARNING: Child process {pid} did not terminate after SIGKILL"
            )
    except Exception:
        debug_swallow("kill_child")
    _state._child_process = None
    _state._child_pgid = None


def signal_handler(signum: int, frame: types.FrameType | None) -> None:
    """Handle termination signals - cleanup and exit."""
    kill_child()
    release_lock()
    sys.exit(128 + signum)


def atexit_handler() -> None:
    """Release lock on unexpected exit (crash, exception)."""
    kill_child()
    release_lock()


def find_real_cargo() -> str | None:
    """Find real cargo binary. Returns path or None if not found."""
    cargo_home = os.environ.get("CARGO_HOME")
    search_paths = []
    if cargo_home:
        search_paths.append(Path(cargo_home) / "bin/cargo")
    # Handle HOME not set gracefully
    try:
        search_paths.append(Path.home() / ".cargo/bin/cargo")
    except RuntimeError:
        pass
    search_paths.extend(
        [
            Path("/opt/homebrew/bin/cargo"),
            Path("/usr/local/bin/cargo"),
            Path("/usr/bin/cargo"),
        ]
    )
    for loc in search_paths:
        if loc.exists() and os.access(loc, os.X_OK):
            return str(loc)
    return None


def run_cargo(args: list[str], timeout: int) -> int:
    """Run cargo with timeout, returning exit code."""
    cargo = os.environ.get("AIT_REAL_CARGO") or find_real_cargo()

    if not cargo:
        log_stderr("[cargo] ERROR: Could not find real cargo")
        return 127

    cmd = [cargo] + args
    start_time = time.time()
    try:
        # Start in new process group (NOT session) so we can kill cargo + rustc children
        # while still allowing parent session termination to propagate (#1490)
        _state._child_process = subprocess.Popen(cmd, preexec_fn=_make_process_group)
        try:
            _state._child_pgid = os.getpgid(_state._child_process.pid)
        except (ProcessLookupError, OSError):
            # Child exited immediately, use None and fall back to direct kill
            _state._child_pgid = None
        exit_code = _state._child_process.wait(timeout=timeout)
        _state._child_process = None
        _state._child_pgid = None
        return exit_code
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        log_stderr(
            f"\n[cargo] ERROR: Command timed out after {elapsed:.1f}s "
            f"(timeout={timeout}s), killing process group"
        )
        kill_child()
        return 124


def run_binary(binary_path: str, binary_args: list[str], timeout: int) -> int:
    """Run compiled binary with timeout, returning exit code."""
    cmd = [binary_path] + binary_args
    start_time = time.time()
    try:
        _state._child_process = subprocess.Popen(cmd, preexec_fn=_make_process_group)
        try:
            _state._child_pgid = os.getpgid(_state._child_process.pid)
        except (ProcessLookupError, OSError):
            _state._child_pgid = None
        exit_code = _state._child_process.wait(timeout=timeout)
        _state._child_process = None
        _state._child_pgid = None
        return exit_code
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        log_stderr(
            f"\n[cargo] ERROR: Binary execution timed out after {elapsed:.1f}s "
            f"(timeout={timeout}s)"
        )
        kill_child()
        return 124


def parse_cargo_run_args(args: list[str]) -> tuple[list[str], list[str], str | None]:
    """Parse cargo run args into build args, binary args, and binary path hint.

    Returns (build_args, binary_args, package_or_bin).

    For 'cargo run -p foo --release -- arg1 arg2':
      build_args = ['build', '-p', 'foo', '--release']
      binary_args = ['arg1', 'arg2']
      package_or_bin = 'foo'
    """
    build_args = ["build"]
    binary_args = []
    package_or_bin = None

    # Find '--' separator
    if "--" in args:
        sep_idx = args.index("--")
        cargo_args = args[1:sep_idx]  # Skip 'run'
        binary_args = args[sep_idx + 1 :]
    else:
        cargo_args = args[1:]  # Skip 'run'

    # Transfer relevant flags to build command
    i = 0
    while i < len(cargo_args):
        arg = cargo_args[i]
        if arg in ("-p", "--package"):
            build_args.append(arg)
            if i + 1 < len(cargo_args):
                package_or_bin = cargo_args[i + 1]
                build_args.append(cargo_args[i + 1])
                i += 1
        elif arg.startswith(("-p=", "--package=")):
            build_args.append(arg)
            package_or_bin = arg.split("=", 1)[1]
        elif arg in ("--bin",):
            build_args.append(arg)
            if i + 1 < len(cargo_args):
                package_or_bin = cargo_args[i + 1]
                build_args.append(cargo_args[i + 1])
                i += 1
        elif arg.startswith("--bin="):
            build_args.append(arg)
            package_or_bin = arg.split("=", 1)[1]
        elif arg in (
            "--release",
            "--profile",
            "--target",
            "--features",
            "--all-features",
            "--no-default-features",
            "--target-dir",
            "--manifest-path",
            "-F",
        ):
            build_args.append(arg)
            # Handle args that take a value
            if arg in (
                "--profile",
                "--target",
                "--target-dir",
                "--manifest-path",
                "--features",
                "-F",
            ):
                if i + 1 < len(cargo_args) and not cargo_args[i + 1].startswith("-"):
                    build_args.append(cargo_args[i + 1])
                    i += 1
        elif arg.startswith(
            (
                "--profile=",
                "--target=",
                "--features=",
                "--target-dir=",
                "--manifest-path=",
            )
        ):
            build_args.append(arg)
        i += 1

    return build_args, binary_args, package_or_bin


def find_built_binary(package_or_bin: str | None, args: list[str]) -> str | None:
    """Find the path to the built binary in target directory.

    Args:
        package_or_bin: Package or binary name hint from cargo run args
        args: Original cargo run args (to detect --release, --target, etc.)

    Returns path to binary or None if not found.
    """
    # Determine target directory
    target_dir = Path("target")
    for i, arg in enumerate(args):
        if arg == "--target-dir" and i + 1 < len(args):
            target_dir = Path(args[i + 1])
        elif arg.startswith("--target-dir="):
            target_dir = Path(arg.split("=", 1)[1])

    # Determine profile subdirectory
    profile = "debug"
    if "--release" in args:
        profile = "release"
    for i, arg in enumerate(args):
        if arg == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
        elif arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]

    # Determine target triple subdirectory
    target_triple = None
    for i, arg in enumerate(args):
        if arg == "--target" and i + 1 < len(args):
            target_triple = args[i + 1]
        elif arg.startswith("--target="):
            target_triple = arg.split("=", 1)[1]

    # Build path to binary directory
    if target_triple:
        bin_dir = target_dir / target_triple / profile
    else:
        bin_dir = target_dir / profile

    if not bin_dir.exists():
        return None

    # Find binary
    if package_or_bin:
        binary = bin_dir / package_or_bin
        if binary.exists() and os.access(binary, os.X_OK):
            return str(binary)

    # Try to find from Cargo.toml
    try:
        import tomllib  # noqa: PLC0415 - lazy import

        cargo_toml = Path("Cargo.toml")
        if cargo_toml.exists():
            data = tomllib.loads(cargo_toml.read_text())
            pkg_name = data.get("package", {}).get("name")
            if pkg_name:
                binary = bin_dir / pkg_name
                if binary.exists() and os.access(binary, os.X_OK):
                    return str(binary)
    except Exception:
        debug_swallow("find_built_binary")

    return None

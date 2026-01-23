#!/usr/bin/env python3
"""Cargo wrapper - Serializes cargo builds org-wide with mutex and orphan cleanup.

CANONICAL SOURCE: ayates_dbx/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Andrew Yates <ayates@dropbox.com>
Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Timeouts in seconds
LOCK_ACQUIRE_TIMEOUT = 30 * 60  # 30 minutes to acquire lock
BUILD_TIMEOUT = 60 * 60  # 1 hour (3600s) per build; exit code 124 on timeout
STALE_PROCESS_AGE = 2 * 60 * 60  # 2 hours = stale lock or orphan process

STATUS_INTERVAL = 60  # Print status every 60 seconds while waiting
MAX_LOG_LINES = 1000  # Rotate logs at this size
MAX_STALE_RELEASE_ATTEMPTS = 5  # Limit retries when releasing stale locks

# Lock directory paths - initialized lazily to handle missing HOME
LOCK_DIR: Path | None = None
LOCK_FILE: Path | None = None
LOCK_META: Path | None = None
BUILDS_LOG: Path | None = None
ORPHANS_LOG: Path | None = None


def init_lock_paths() -> bool:
    """Initialize lock directory paths. Returns False if HOME unavailable."""
    global LOCK_DIR, LOCK_FILE, LOCK_META, BUILDS_LOG, ORPHANS_LOG
    try:
        LOCK_DIR = Path.home() / ".ait_cargo_lock"
        LOCK_FILE = LOCK_DIR / "lock.pid"
        LOCK_META = LOCK_DIR / "lock.json"
        BUILDS_LOG = LOCK_DIR / "builds.log"
        ORPHANS_LOG = LOCK_DIR / "orphans.log"
        return True
    except RuntimeError:
        # HOME not set
        return False

# Global state for signal handler
_lock_held = False
_child_process = None
_child_pgid = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_stderr(msg: str) -> None:
    """Print to stderr and flush immediately so AI sees messages without delay."""
    print(msg, file=sys.stderr)
    sys.stderr.flush()


def get_process_start_time(pid: int) -> float | None:
    """Get process start time as Unix timestamp. Returns None if process doesn't exist."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
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
                month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                month = month_map.get(parts[1], 1)
                day = int(parts[2])
                time_parts = parts[3].split(":")
                hour, minute, sec = int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
                year = int(parts[4])
                dt = datetime(year, month, day, hour, minute, sec)
                return dt.timestamp()
        return None
    except Exception:
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
        pass
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


def get_env_context() -> dict:
    """Get AI context from environment variables."""
    return {
        "project": os.environ.get("AI_PROJECT", os.path.basename(os.getcwd())),
        "role": os.environ.get("AI_ROLE", "USER"),
        "session": os.environ.get("AI_SESSION", "")[:8],
        "iteration": os.environ.get("AI_ITERATION", ""),
        "commit": get_git_commit(),
    }


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def find_cargo_processes() -> list[dict]:
    """Find all cargo/rustc processes with their age."""
    processes = []
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,etime,comm,args"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split(None, 3)
            if len(parts) < 3:
                continue
            pid, etime, comm = parts[0], parts[1], parts[2]
            args = parts[3] if len(parts) > 3 else ""

            if comm not in ("cargo", "rustc"):
                continue

            age_seconds = parse_etime(etime)
            processes.append({
                "pid": int(pid),
                "age_seconds": age_seconds,
                "comm": comm,
                "args": args[:100],
            })
    except Exception:
        pass
    return processes


def parse_etime(etime: str) -> int:
    """Parse ps etime format to seconds."""
    try:
        if "-" in etime:
            days, rest = etime.split("-", 1)
            days = int(days)
        else:
            days = 0
            rest = etime

        parts = rest.split(":")
        if len(parts) == 2:  # MM:SS
            return days * 86400 + int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:  # HH:MM:SS
            return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return 0


def cleanup_orphans() -> int:
    """Kill cargo/rustc processes older than 2 hours. Returns count killed."""
    killed = 0
    for proc in find_cargo_processes():
        if proc["age_seconds"] >= STALE_PROCESS_AGE:
            pid = proc["pid"]
            # Don't kill our own ancestors (e.g., if spawned by long-running cargo)
            if is_ancestor_of_self(pid):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                log_orphan(proc)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass
    return killed


def rotate_log(log_path: Path, max_lines: int) -> None:
    """Truncate log file to last max_lines entries (atomic)."""
    if not log_path.exists():
        return
    tmp_path = log_path.parent / f"{log_path.name}.{os.getpid()}.tmp"
    try:
        lines = log_path.read_text().splitlines()
        if len(lines) > max_lines:
            # Atomic write: write to temp file then rename
            tmp_path.write_text("\n".join(lines[-max_lines:]) + "\n")
            os.rename(tmp_path, log_path)
    except Exception:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)


def log_orphan(proc: dict) -> None:
    """Log killed orphan to orphans.log."""
    # LOCK_DIR is guaranteed to exist (ensured by ensure_lock_dir before cleanup_orphans)
    entry = {
        "killed_at": now_iso(),
        "pid": proc["pid"],
        "age_seconds": proc["age_seconds"],
        "comm": proc["comm"],
        "args": proc["args"],
    }
    with open(ORPHANS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rotate_log(ORPHANS_LOG, MAX_LOG_LINES)


def is_lock_stale() -> bool:
    """Check if existing lock is stale (holder dead, PID reused, or too old)."""
    if not LOCK_FILE.exists():
        return True

    try:
        pid = int(LOCK_FILE.read_text().strip())

        # Check if process exists
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True  # Process dead

        # Check for PID reuse: compare process start time with lock acquisition time
        if LOCK_META.exists():
            meta = json.loads(LOCK_META.read_text())
            acquired_at = meta.get("acquired_at", "")
            lock_start = meta.get("process_start_time")

            # If we stored the original process start time, verify it matches
            if lock_start is not None:
                current_start = get_process_start_time(pid)
                if current_start is not None and abs(current_start - lock_start) > 2:
                    # Process start times differ by >2 seconds = PID was reused
                    return True

            # Check age
            acquired = datetime.fromisoformat(acquired_at)
            age = (datetime.now(timezone.utc) - acquired).total_seconds()
            if age > STALE_PROCESS_AGE:
                return True
        return False
    except (ProcessLookupError, ValueError, json.JSONDecodeError, KeyError):
        return True


def cleanup_stale_temp_files() -> None:
    """Remove any orphaned temp files from crashed processes."""
    try:
        # Clean up lock.json.*.tmp files
        for tmp_file in LOCK_DIR.glob("lock.json.*.tmp"):
            try:
                pid = int(tmp_file.stem.split(".")[-1])
                os.kill(pid, 0)  # Process alive, leave it
            except (ValueError, ProcessLookupError):
                tmp_file.unlink(missing_ok=True)  # Process dead, clean up

        # Clean up lock.pid.stale.* files from interrupted force_release_stale_lock
        for stale_file in LOCK_DIR.glob("lock.pid.stale.*"):
            try:
                pid = int(stale_file.name.split(".")[-1])
                os.kill(pid, 0)  # Process alive, leave it
            except (ValueError, ProcessLookupError):
                stale_file.unlink(missing_ok=True)  # Process dead, clean up

        # Clean up *.log.*.tmp files from interrupted rotate_log
        for log_tmp in LOCK_DIR.glob("*.log.*.tmp"):
            try:
                pid = int(log_tmp.stem.split(".")[-1])
                os.kill(pid, 0)  # Process alive, leave it
            except (ValueError, ProcessLookupError):
                log_tmp.unlink(missing_ok=True)  # Process dead, clean up
    except Exception:
        pass


def acquire_lock(context: dict) -> bool:
    """Attempt to acquire the lock. Returns True if acquired."""
    global _lock_held
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    tmp_meta = LOCK_DIR / f"lock.json.{os.getpid()}.tmp"
    try:
        # Atomic creation via O_CREAT | O_EXCL
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)

        # Store process start time to detect PID reuse
        process_start = get_process_start_time(os.getpid())

        meta = {
            "acquired_at": now_iso(),
            "pid": os.getpid(),
            "process_start_time": process_start,
            "cwd": os.getcwd(),
            "build_timeout_sec": BUILD_TIMEOUT,
            **context,
        }
        # Atomic write: write to temp file then rename
        tmp_meta.write_text(json.dumps(meta, indent=2))
        os.rename(tmp_meta, LOCK_META)
        _lock_held = True
        return True
    except FileExistsError:
        # Clean up our temp file if we failed to acquire lock
        tmp_meta.unlink(missing_ok=True)
        return False
    except Exception:
        tmp_meta.unlink(missing_ok=True)
        raise


def release_lock() -> None:
    """Release the lock if we hold it."""
    global _lock_held
    if not _lock_held:
        return
    try:
        # Verify we own the lock before releasing
        if LOCK_FILE.exists():
            pid = int(LOCK_FILE.read_text().strip())
            if pid != os.getpid():
                return  # Not our lock
        LOCK_FILE.unlink(missing_ok=True)
        LOCK_META.unlink(missing_ok=True)
        _lock_held = False
    except Exception:
        pass


def force_release_stale_lock() -> bool:
    """Force release a stale lock atomically. Returns True if released."""
    if not LOCK_FILE.exists():
        return True

    old_lock = LOCK_DIR / f"lock.pid.stale.{os.getpid()}"
    try:
        # Atomic rename - if this succeeds, we "own" the stale lock
        os.rename(LOCK_FILE, old_lock)
        if LOCK_META.exists():
            try:
                meta = json.loads(LOCK_META.read_text())
                log_stderr(f"[cargo] Force-releasing stale lock from {meta.get('project', '?')} "
                           f"(acquired {meta.get('acquired_at', '?')})")
            except Exception:
                log_stderr("[cargo] Force-releasing stale lock")
            LOCK_META.unlink(missing_ok=True)
        old_lock.unlink(missing_ok=True)
        return True
    except (FileNotFoundError, OSError):
        # Another process beat us to it
        old_lock.unlink(missing_ok=True)
        return False


def get_lock_holder_info(verbose: bool = False) -> str:
    """Get info about current lock holder.

    Args:
        verbose: If True, include full metadata (session, iteration, commit, cwd)
    """
    if not LOCK_META.exists():
        return "unknown"
    try:
        meta = json.loads(LOCK_META.read_text())
        basic = f"{meta.get('project', '?')} ({meta.get('role', '?')})"
        if not verbose:
            return basic
        # Full metadata for debugging
        parts = [basic]
        if meta.get('session'):
            parts.append(f"session={meta['session']}")
        if meta.get('iteration'):
            parts.append(f"iter={meta['iteration']}")
        if meta.get('commit'):
            parts.append(f"commit={meta['commit']}")
        if meta.get('cwd'):
            parts.append(f"cwd={meta['cwd']}")
        if meta.get('acquired_at'):
            parts.append(f"acquired={meta['acquired_at']}")
        return " | ".join(parts)
    except Exception:
        return "unknown"


def log_build(context: dict, command: list[str], started: datetime,
              finished: datetime, exit_code: int, timeout_sec: int) -> None:
    """Log build to builds.log."""
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
    with open(BUILDS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rotate_log(BUILDS_LOG, MAX_LOG_LINES)


def check_retry_loop(command: str, cwd: str, current_commit: str) -> None:
    """Check builds.log for repeated failures. Prints info/warning if detected."""
    if BUILDS_LOG is None or not BUILDS_LOG.exists():
        return

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
        failures = []

        for line in BUILDS_LOG.read_text().splitlines()[-30:]:  # Last 30 entries
            try:
                entry = json.loads(line)
                if (entry.get("command") == command
                        and entry.get("cwd") == cwd
                        and entry.get("exit_code", 0) != 0):
                    started = datetime.fromisoformat(entry["started_at"])
                    # Handle timezone-naive timestamps
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
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
                log_stderr(f"⚠️  RETRY LOOP: This command failed {len(failures)}x in 60s without code changes")
                log_stderr(f"    Command: {command}")
                log_stderr(f"    Commit:  {current_commit or 'unknown'}")
                log_stderr("")
                log_stderr("    HINT: Read the error output. Investigate or file an issue.")
                log_stderr(f"    Details: {BUILDS_LOG}")
                log_stderr("")
            else:
                # Failures but with code changes - just info
                log_stderr(f"[cargo] ℹ️  Command failed {len(failures)}x in 60s (commits: {', '.join(commits)})")

    except Exception:
        pass  # Don't fail the build due to retry detection errors


def kill_child() -> None:
    """Kill the child process group if running."""
    global _child_process, _child_pgid
    if _child_process is None:
        return
    try:
        killed_via_pg = False
        # Kill entire process group (cargo + rustc children)
        if _child_pgid is not None:
            try:
                os.killpg(_child_pgid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.killpg(_child_pgid, 0)  # Check if still alive
                    os.killpg(_child_pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                killed_via_pg = True
            except (ProcessLookupError, OSError):
                pass  # Fall through to direct kill

        # Fallback: kill just the direct child if pg kill failed or pgid unknown
        if not killed_via_pg:
            _child_process.terminate()
            try:
                _child_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _child_process.kill()

        _child_process.wait()
    except Exception:
        pass
    _child_process = None
    _child_pgid = None


def signal_handler(signum: int, frame) -> None:
    """Handle termination signals - cleanup and exit."""
    kill_child()
    release_lock()
    sys.exit(128 + signum)


def run_cargo(args: list[str], timeout: int) -> int:
    """Run cargo with timeout, returning exit code."""
    global _child_process, _child_pgid

    cargo = os.environ.get("AIT_REAL_CARGO") or find_real_cargo()

    if not cargo:
        log_stderr("[cargo] ERROR: Could not find real cargo")
        return 127

    cmd = [cargo] + args
    try:
        # Start in new process group so we can kill cargo + all rustc children
        _child_process = subprocess.Popen(cmd, start_new_session=True)
        try:
            _child_pgid = os.getpgid(_child_process.pid)
        except (ProcessLookupError, OSError):
            # Child exited immediately, use None and fall back to direct kill
            _child_pgid = None
        exit_code = _child_process.wait(timeout=timeout)
        _child_process = None
        _child_pgid = None
        return exit_code
    except subprocess.TimeoutExpired:
        log_stderr(f"\n[cargo] ERROR: Build timed out after {timeout}s, killing process group")
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
        binary_args = args[sep_idx + 1:]
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
        elif arg in ("--release", "--profile", "--target", "--features", "--all-features",
                     "--no-default-features", "--target-dir", "--manifest-path", "-F"):
            build_args.append(arg)
            # Handle args that take a value
            if arg in ("--profile", "--target", "--target-dir", "--manifest-path", "--features", "-F"):
                if i + 1 < len(cargo_args) and not cargo_args[i + 1].startswith("-"):
                    build_args.append(cargo_args[i + 1])
                    i += 1
        elif arg.startswith(("--profile=", "--target=", "--features=", "--target-dir=",
                             "--manifest-path=")):
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
        import tomllib  # noqa: PLC0415 - lazy import, only needed if binary not found
        cargo_toml = Path("Cargo.toml")
        if cargo_toml.exists():
            data = tomllib.loads(cargo_toml.read_text())
            pkg_name = data.get("package", {}).get("name")
            if pkg_name:
                binary = bin_dir / pkg_name
                if binary.exists() and os.access(binary, os.X_OK):
                    return str(binary)
    except Exception:
        pass

    return None


def run_binary(binary_path: str, binary_args: list[str], timeout: int) -> int:
    """Run compiled binary with timeout, returning exit code."""
    global _child_process, _child_pgid

    cmd = [binary_path] + binary_args
    try:
        _child_process = subprocess.Popen(cmd, start_new_session=True)
        try:
            _child_pgid = os.getpgid(_child_process.pid)
        except (ProcessLookupError, OSError):
            _child_pgid = None
        exit_code = _child_process.wait(timeout=timeout)
        _child_process = None
        _child_pgid = None
        return exit_code
    except subprocess.TimeoutExpired:
        log_stderr(f"\n[cargo] ERROR: Binary execution timed out after {timeout}s")
        kill_child()
        return 124


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
    search_paths.extend([
        Path("/opt/homebrew/bin/cargo"),
        Path("/usr/local/bin/cargo"),
        Path("/usr/bin/cargo"),
    ])
    for loc in search_paths:
        if loc.exists() and os.access(loc, os.X_OK):
            return str(loc)
    return None


def ensure_lock_dir() -> bool:
    """Ensure lock directory exists and is writable. Returns False if unusable."""
    if LOCK_DIR is None:
        return False
    test_file = LOCK_DIR / f".write_test.{os.getpid()}"
    try:
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        # Test writability with unique filename to avoid race condition
        test_file.write_text("test")
        return True
    except (OSError, PermissionError) as e:
        log_stderr(f"[cargo] WARNING: Lock directory unusable ({e}), running without serialization")
        return False
    finally:
        test_file.unlink(missing_ok=True)


def main() -> int:
    # Register cleanup handlers for all termination signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)  # Ctrl+\
    atexit.register(atexit_handler)

    args = sys.argv[1:]

    # Initialize lock paths (may fail if HOME not set)
    if not init_lock_paths():
        log_stderr("[cargo] WARNING: HOME not set, running without serialization")
        cargo = os.environ.get("AIT_REAL_CARGO") or find_real_cargo()
        if cargo:
            os.execv(cargo, [cargo] + args)
        log_stderr("[cargo] ERROR: Could not find cargo")
        return 1

    # Ensure lock directory is usable, fall back to direct execution if not
    if not ensure_lock_dir():
        cargo = os.environ.get("AIT_REAL_CARGO") or find_real_cargo()
        if cargo:
            os.execv(cargo, [cargo] + args)
        log_stderr("[cargo] ERROR: Could not find cargo for fallback execution")
        return 1

    context = get_env_context()

    # Clean up orphan processes before attempting lock
    orphans_killed = cleanup_orphans()
    if orphans_killed > 0:
        log_stderr(f"[cargo] Cleaned up {orphans_killed} orphan process(es)")

    # Check for stale lock
    if LOCK_FILE.exists() and is_lock_stale():
        force_release_stale_lock()

    # Clean up orphaned temp files once before acquire loop (not on every iteration)
    cleanup_stale_temp_files()

    # Acquire lock with timeout
    start_wait = time.time()
    last_status = 0
    acquired = False
    printed_initial = False
    stale_release_attempts = 0

    while time.time() - start_wait < LOCK_ACQUIRE_TIMEOUT:
        if acquire_lock(context):
            acquired = True
            break

        # Check staleness IMMEDIATELY before waiting (detect dead processes fast)
        if stale_release_attempts < MAX_STALE_RELEASE_ATTEMPTS and is_lock_stale():
            if force_release_stale_lock():
                stale_release_attempts += 1
                continue  # Retry acquire immediately, don't sleep

        # Print status: verbose on first block, brief on subsequent
        if not printed_initial:
            holder = get_lock_holder_info(verbose=True)
            log_stderr(f"[cargo] Waiting for lock (held by {holder})...")
            printed_initial = True

        elapsed = time.time() - start_wait
        if elapsed - last_status >= STATUS_INTERVAL:
            holder = get_lock_holder_info()
            log_stderr(f"[cargo] Still waiting for lock (held by {holder}, waited {int(elapsed)}s)...")
            last_status = elapsed

        time.sleep(1)

    if not acquired:
        log_stderr(f"[cargo] ERROR: Could not acquire lock after {LOCK_ACQUIRE_TIMEOUT}s")
        return 1

    # Special handling for 'cargo run': build under lock, run binary without lock
    is_cargo_run = args and args[0] == "run"

    if is_cargo_run:
        # Phase 1: Build under lock
        build_args, binary_args, package_or_bin = parse_cargo_run_args(args)
        build_command_str = shlex.join(["cargo"] + build_args)

        check_retry_loop(build_command_str, os.getcwd(), context.get("commit", ""))

        started = datetime.now(timezone.utc)
        log_stderr(f"[cargo] Lock acquired, building: {build_command_str}")
        build_exit_code = run_cargo(build_args, BUILD_TIMEOUT)
        finished = datetime.now(timezone.utc)
        log_build(context, ["cargo"] + build_args, started, finished, build_exit_code, BUILD_TIMEOUT)

        if build_exit_code != 0:
            release_lock()
            return build_exit_code

        # Find the built binary
        binary_path = find_built_binary(package_or_bin, args)

        # Release lock before running the binary
        release_lock()
        log_stderr("[cargo] Lock released, running binary outside lock")

        if binary_path:
            # Phase 2: Run binary without lock
            log_stderr(f"[cargo] Running: {binary_path} {shlex.join(binary_args)}")
            return run_binary(binary_path, binary_args, BUILD_TIMEOUT)

        # Fallback: couldn't find binary, run cargo run directly (rare edge case)
        log_stderr("[cargo] WARNING: Could not locate built binary, running cargo run directly")
        return run_cargo(args, BUILD_TIMEOUT)

    # Standard path: run cargo with lock held
    try:
        command_str = shlex.join(["cargo"] + args)

        # Check for retry loop before running (warn only, don't block)
        check_retry_loop(command_str, os.getcwd(), context.get("commit", ""))

        started = datetime.now(timezone.utc)
        log_stderr(f"[cargo] Lock acquired, running: {command_str}")
        exit_code = run_cargo(args, BUILD_TIMEOUT)
        finished = datetime.now(timezone.utc)
        log_build(context, ["cargo"] + args, started, finished, exit_code, BUILD_TIMEOUT)
        return exit_code
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())

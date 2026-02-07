# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Lock management for cargo_wrapper."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from ai_template_scripts.subprocess_utils import is_process_alive

from . import _state
from .constants import (
    LOCK_BASENAMES,
    LOCK_KIND_BUILD,
    LOCK_KIND_TEST,
    LOCK_KINDS,
    STALE_PROCESS_AGE,
)
from .env import get_repo_identifier
from .logging import debug_swallow, log_stderr, now_iso
from .processes import get_process_start_time
from .timeouts import get_limits_config

__all__ = [
    "init_lock_paths",
    "set_lock_kind",
    "get_lock_dir",
    "get_lock_file",
    "get_lock_meta",
    "get_builds_log",
    "get_orphans_log",
    "is_lock_stale",
    "acquire_lock",
    "release_lock",
    "force_release_stale_lock",
    "cleanup_stale_temp_files",
    "get_lock_holder_info",
]


def _lock_basename(lock_kind: str) -> str:
    """Return the lock file basename for a lock kind."""
    if lock_kind == LOCK_KIND_TEST:
        limits = get_limits_config()
        if limits.get("max_concurrent_cargo") == 1:
            return LOCK_BASENAMES[LOCK_KIND_BUILD]
    return LOCK_BASENAMES.get(lock_kind, f"lock.{lock_kind}")


def _lock_basenames_for_cleanup() -> tuple[str, ...]:
    """Return lock basenames to clean up, including legacy test locks."""
    basenames = set(LOCK_BASENAMES.values())
    for lock_kind in LOCK_KINDS:
        basenames.add(_lock_basename(lock_kind))
    return tuple(sorted(basenames))


def set_lock_kind(lock_kind: str) -> None:
    """Set the active lock kind and update derived paths."""
    if lock_kind not in LOCK_KINDS:
        raise ValueError(f"Unsupported lock kind: {lock_kind}")
    _state.LOCK_KIND = lock_kind
    if _state.LOCK_DIR is None:
        _state.LOCK_FILE = None
        _state.LOCK_META = None
        return
    basename = _lock_basename(lock_kind)
    _state.LOCK_FILE = _state.LOCK_DIR / f"{basename}.pid"
    _state.LOCK_META = _state.LOCK_DIR / f"{basename}.json"


def init_lock_paths(lock_kind: str | None = None) -> bool:
    """Initialize lock directory paths. Returns False if HOME unavailable.

    Lock paths are per-repo to allow concurrent builds across different projects.
    Structure: ~/.ait_cargo_lock/<repo>/lock.pid (build) or lock.test.pid (test)
    """
    try:
        repo_id = get_repo_identifier()
        _state.LOCK_DIR = Path.home() / ".ait_cargo_lock" / repo_id
        _state.BUILDS_LOG = _state.LOCK_DIR / "builds.log"
        _state.ORPHANS_LOG = _state.LOCK_DIR / "orphans.log"
        set_lock_kind(lock_kind or LOCK_KIND_BUILD)
        return True
    except RuntimeError:
        # HOME not set
        return False


def get_lock_dir() -> Path:
    """Get LOCK_DIR, asserting it's initialized."""
    assert _state.LOCK_DIR is not None, "init_lock_paths() not called"
    return _state.LOCK_DIR


def get_lock_file() -> Path:
    """Get LOCK_FILE, asserting it's initialized."""
    assert _state.LOCK_FILE is not None, "init_lock_paths() not called"
    return _state.LOCK_FILE


def get_lock_meta() -> Path:
    """Get LOCK_META, asserting it's initialized."""
    assert _state.LOCK_META is not None, "init_lock_paths() not called"
    return _state.LOCK_META


def get_builds_log() -> Path:
    """Get BUILDS_LOG, asserting it's initialized."""
    assert _state.BUILDS_LOG is not None, "init_lock_paths() not called"
    return _state.BUILDS_LOG


def get_orphans_log() -> Path:
    """Get ORPHANS_LOG, asserting it's initialized."""
    assert _state.ORPHANS_LOG is not None, "init_lock_paths() not called"
    return _state.ORPHANS_LOG


def is_lock_stale(verbose: bool = False) -> bool:
    """Check if existing lock is stale (holder dead, PID reused, or too old).

    REQUIRES: init_lock_paths() called (LOCK_FILE, LOCK_META initialized)
    ENSURES: Returns True if any of:
             - LOCK_FILE doesn't exist
             - PID in LOCK_FILE no longer exists (process dead)
             - PID exists but inaccessible (PermissionError - different user)
             - os.kill fails for any other reason (OSError)
             - Process start time differs from recorded by >2s (PID reuse detected)
             - Lock age > STALE_PROCESS_AGE (2 hours)
             - Metadata parse exception (ValueError, TypeError, KeyError, JSON decode)
    ENSURES: Returns False only if lock holder is verified alive, recent,
             and start time matches (within 2s tolerance)
    ENSURES: Read-only - does not modify any state

    TLA+ correspondence: cargo_lock.tla:LockIsStale,
    cargo_lock_with_toctou.tla:LockIsStale
    TLA+ spec: tla/cargo_lock.tla, tla/cargo_lock_with_toctou.tla

    NOTE: 2-second tolerance in PID reuse detection creates TOCTOU window.
    See tla/cargo_lock_with_toctou.tla for formal model of this race.

    Args:
        verbose: If True, log reasons for stale detection (helps diagnose #1679)
    """
    lock_file = get_lock_file()
    lock_meta = get_lock_meta()
    if not lock_file.exists():
        return True

    try:
        pid = int(lock_file.read_text().strip())

        # Check if process exists and is accessible.
        # Unlike is_process_alive() which treats PermissionError as "alive",
        # lock staleness requires we can verify ownership. If os.kill raises
        # PermissionError (different user) or OSError, treat lock as stale
        # since we can't confirm the holder is our process. (#2703)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if verbose:
                log_stderr(f"[cargo] Stale lock: PID {pid} not found (process dead)")
            return True
        except PermissionError:
            if verbose:
                log_stderr(f"[cargo] Stale lock: PID {pid} owned by another user")
            return True
        except OSError as e:
            if verbose:
                log_stderr(f"[cargo] Stale lock: os.kill error for PID {pid}: {e}")
            return True

        # Check for PID reuse: compare process start time with lock acquisition time
        if lock_meta.exists():
            meta = json.loads(lock_meta.read_text())
            meta_pid_raw = meta.get("pid")
            meta_pid = None
            if meta_pid_raw is not None:
                try:
                    meta_pid = int(meta_pid_raw)
                except (TypeError, ValueError):
                    meta_pid = None

            # Lock release/acquire can transiently expose a new lock.pid with an
            # old lock.json. Treat PID mismatch as in-flight transition, not stale,
            # to avoid force-releasing a freshly acquired lock.
            if meta_pid is not None and meta_pid != pid:
                if verbose:
                    log_stderr(
                        f"[cargo] Lock metadata PID mismatch during transition "
                        f"(lock.pid={pid}, lock.json={meta_pid}); treating as active"
                    )
                return False

            acquired_at = meta.get("acquired_at", "")
            lock_start = meta.get("process_start_time")

            # If we stored the original process start time, verify it matches
            if lock_start is not None:
                current_start = get_process_start_time(pid)
                if current_start is not None and abs(current_start - lock_start) > 2:
                    # Process start times differ by >2 seconds = PID was reused
                    if verbose:
                        log_stderr(
                            f"[cargo] Stale lock: PID {pid} reused "
                            f"(start: {current_start:.0f} vs lock: {lock_start:.0f})"
                        )
                    return True

            # Check age
            acquired = datetime.datetime.fromisoformat(acquired_at)
            # Handle timezone-naive timestamps from older lock files
            if acquired.tzinfo is None:
                acquired = acquired.replace(tzinfo=datetime.UTC)
            age = (datetime.datetime.now(datetime.UTC) - acquired).total_seconds()
            if age > STALE_PROCESS_AGE:
                if verbose:
                    log_stderr(
                        f"[cargo] Stale lock: age {age / 3600:.1f}h exceeds "
                        f"threshold {STALE_PROCESS_AGE / 3600:.1f}h"
                    )
                return True
        return False
    except (FileNotFoundError, OSError) as e:
        if verbose:
            log_stderr(
                f"[cargo] Stale lock: lock files changed during check ({type(e).__name__})"
            )
        return True
    except (
        ValueError,
        TypeError,
        json.JSONDecodeError,
        KeyError,
    ) as e:
        if isinstance(e, ValueError) and not lock_meta.exists():
            if verbose:
                log_stderr(
                    "[cargo] Lock metadata missing during PID parse; "
                    "treating lock acquisition as in-flight"
                )
            return False
        if verbose:
            log_stderr(f"[cargo] Stale lock: metadata error ({type(e).__name__})")
        return True


def cleanup_stale_temp_files() -> None:
    """Remove any orphaned temp files from crashed processes."""
    lock_dir = get_lock_dir()
    try:
        for basename in _lock_basenames_for_cleanup():
            # Clean up lock.json.*.tmp files
            for tmp_file in lock_dir.glob(f"{basename}.json.*.tmp"):
                try:
                    pid = int(tmp_file.stem.split(".")[-1])
                    if not is_process_alive(pid):
                        tmp_file.unlink(missing_ok=True)
                except ValueError:
                    tmp_file.unlink(missing_ok=True)

            # Clean up lock.pid.stale.* files from interrupted force_release_stale_lock
            for stale_file in lock_dir.glob(f"{basename}.pid.stale.*"):
                try:
                    pid = int(stale_file.name.split(".")[-1])
                    if not is_process_alive(pid):
                        stale_file.unlink(missing_ok=True)
                except ValueError:
                    stale_file.unlink(missing_ok=True)

        # Clean up *.log.*.tmp files from interrupted rotate_log
        for log_tmp in lock_dir.glob("*.log.*.tmp"):
            try:
                pid = int(log_tmp.stem.split(".")[-1])
                if not is_process_alive(pid):
                    log_tmp.unlink(missing_ok=True)
            except ValueError:
                log_tmp.unlink(missing_ok=True)
    except Exception:
        debug_swallow("cleanup_stale_temp_files")


def acquire_lock(context: dict[str, object]) -> bool:
    """Attempt to acquire the lock. Returns True if acquired.

    REQUIRES: context contains 'project', 'role' keys (from get_env_context())
    REQUIRES: init_lock_paths() called and returned True (LOCK_DIR initialized)
    ENSURES: If returns True: _lock_held is True, LOCK_FILE contains os.getpid(),
             LOCK_META contains context metadata with acquired_at timestamp
    ENSURES: If returns False: no global state modified (_lock_held unchanged)
    ENSURES: Atomic - uses O_CREAT|O_EXCL for race-free acquisition

    TLA+ correspondence: cargo_lock.tla:TryAcquire(p), cargo_lock.tla:FailAcquire(p)
    TLA+ spec: tla/cargo_lock.tla, tla/cargo_lock_with_toctou.tla
    """
    lock_dir = get_lock_dir()
    lock_file = get_lock_file()
    lock_meta = get_lock_meta()
    lock_dir.mkdir(parents=True, exist_ok=True)

    basename = _lock_basename(_state.LOCK_KIND)
    tmp_meta = lock_dir / f"{basename}.json.{os.getpid()}.tmp"
    try:
        # Atomic creation via O_CREAT | O_EXCL
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)

        # Store process start time to detect PID reuse
        process_start = get_process_start_time(os.getpid())

        meta = {
            "acquired_at": now_iso(),
            "pid": os.getpid(),
            "process_start_time": process_start,
            "cwd": os.getcwd(),
            "lock_kind": _state.LOCK_KIND,
            **context,
        }
        # Atomic write: write to temp file then rename
        tmp_meta.write_text(json.dumps(meta, indent=2))
        os.rename(tmp_meta, lock_meta)
        _state._lock_held = True
        return True
    except FileExistsError:
        # Clean up our temp file if we failed to acquire lock
        tmp_meta.unlink(missing_ok=True)
        return False
    except Exception:
        tmp_meta.unlink(missing_ok=True)
        raise  # Re-raise: clean up temp file on unexpected lock error


def release_lock() -> None:
    """Release the lock if we hold it.

    REQUIRES: None (safe to call unconditionally)
    ENSURES: If we held the lock (_lock_held was True and LOCK_FILE contained our PID):
             _lock_held is False, LOCK_FILE and LOCK_META are deleted
    ENSURES: If we didn't hold the lock: no state modified
    ENSURES: Idempotent - safe to call multiple times

    TLA+ correspondence: cargo_lock.tla:Release(p)
    TLA+ spec: tla/cargo_lock.tla
    """
    if not _state._lock_held:
        return
    lock_file = get_lock_file()
    lock_meta = get_lock_meta()
    try:
        # Verify we own the lock before releasing
        if lock_file.exists():
            pid = int(lock_file.read_text().strip())
            if pid != os.getpid():
                return  # Not our lock
        lock_meta.unlink(missing_ok=True)
        lock_file.unlink(missing_ok=True)
        _state._lock_held = False
    except Exception:
        debug_swallow("release_lock")


def force_release_stale_lock() -> bool:
    """Force release a stale lock atomically. Returns True if released.

    REQUIRES: Should only be called after is_lock_stale() returns True
    REQUIRES: init_lock_paths() called (LOCK_FILE, LOCK_META, LOCK_DIR initialized)
    ENSURES: If returns True: LOCK_FILE and LOCK_META are deleted, lock available
    ENSURES: If returns False: lock became valid (TOCTOU mitigation), or another
             process beat us to the atomic rename (race-safe)
    ENSURES: Atomic via rename - only one caller can win the race
    ENSURES: Logs identity of released lock holder for forensics

    TLA+ correspondence: cargo_lock.tla:ForceReleaseStale(p),
    cargo_lock_with_toctou.tla:AbortForceRelease
    TLA+ spec: tla/cargo_lock.tla, tla/cargo_lock_with_toctou.tla

    NOTE: Re-verifies staleness immediately before force-release to shrink TOCTOU
    window. See #624, tla/cargo_lock_with_toctou.tla for the race condition model.
    """
    lock_file = get_lock_file()
    lock_meta = get_lock_meta()
    lock_dir = get_lock_dir()
    if not lock_file.exists():
        return True

    # TOCTOU mitigation (#624): Re-verify staleness immediately before acting.
    # The caller checked is_lock_stale(), but time has passed. A new process may
    # have acquired the lock legitimately. Re-checking here shrinks the window
    # from milliseconds (caller's check to here) to microseconds (this check to
    # the atomic rename below).
    if not is_lock_stale():
        log_stderr("[cargo] Lock became valid before force-release, aborting")
        return False

    basename = _lock_basename(_state.LOCK_KIND)
    old_lock = lock_dir / f"{basename}.pid.stale.{os.getpid()}"
    try:
        # Delete metadata BEFORE renaming lock file (#3069).
        # If we delete meta after rename, a new acquirer can write fresh metadata
        # between our rename and our meta deletion — corrupting the new holder's
        # diagnostics. Deleting meta first is safe: the lock file still exists
        # (protecting the critical section), and the new acquirer writes its own
        # metadata after acquiring.
        if lock_meta.exists():
            try:
                meta = json.loads(lock_meta.read_text())
                proj = meta.get("project", "?")
                acq = meta.get("acquired_at", "?")
                log_stderr(f"[cargo] Force-releasing lock from {proj} ({acq})")
            except Exception:
                # Best-effort: metadata parse failed, log generic message instead
                log_stderr("[cargo] Force-releasing stale lock")
            lock_meta.unlink(missing_ok=True)
        # Atomic rename - if this succeeds, we "own" the stale lock
        os.rename(lock_file, old_lock)
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
    lock_meta = get_lock_meta()
    if not lock_meta.exists():
        return "unknown"
    try:
        meta = json.loads(lock_meta.read_text())
        basic = f"{meta.get('project', '?')} ({meta.get('role', '?')})"
        if meta.get("lock_kind"):
            basic = f"{basic} [{meta['lock_kind']}]"
        if not verbose:
            return basic
        # Full metadata for debugging
        parts = [basic]
        if meta.get("session"):
            parts.append(f"session={meta['session']}")
        if meta.get("iteration"):
            parts.append(f"iter={meta['iteration']}")
        if meta.get("commit"):
            parts.append(f"commit={meta['commit']}")
        if meta.get("cwd"):
            parts.append(f"cwd={meta['cwd']}")
        if meta.get("acquired_at"):
            parts.append(f"acquired={meta['acquired_at']}")
        return " | ".join(parts)
    except Exception:
        return "unknown"  # Best-effort: lock metadata for diagnostics only

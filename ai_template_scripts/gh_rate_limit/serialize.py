# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/serialize.py - Request Serialization

Coordinates lock-based serialization of cache refreshes to prevent
thundering herd problem when multiple workers poll simultaneously.

Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Methods extracted from RateLimiter:
- _get_serialize_lock_path, _serialized_fetch
- File locking utilities
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_template_scripts.gh_rate_limit.limiter import debug_log

# Cross-platform file locking
try:
    import fcntl

    def lock_file(fd: Any) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def unlock_file(fd: Any) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

except ImportError:

    def lock_file(fd: Any) -> None:
        pass

    def unlock_file(fd: Any) -> None:
        pass


def try_lock_file(fd: Any, timeout_sec: float) -> bool:
    """Attempt to acquire lock with timeout. Returns True if acquired."""
    try:
        import fcntl  # noqa: PLC0415 - conditional import for cross-platform

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                # Lock held by another process, wait briefly and retry
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                time.sleep(min(0.1, remaining))
        return False
    except ImportError:
        # No fcntl - no serialization possible, proceed without lock
        return True


# Serialization timeout: max wait for another worker's API call
SERIALIZE_TIMEOUT_SEC = 30  # Same as default API timeout


class SerializedFetcher:
    """Coordinates lock-based serialization of cache refreshes (#1133).

    When cache is expired:
    1. Acquire lock (with timeout)
    2. Re-check cache (another worker may have just refreshed)
    3. If still expired, execute and cache
    4. Release lock

    This prevents thundering herd when multiple workers poll simultaneously.

    Args:
        cache_dir: Directory for lock files.
        get_repo: Callable that returns the current repo name.
        get_ttl: Callable that returns TTL for args, or None if not cacheable.
    """

    def __init__(
        self,
        cache_dir: Path,
        get_repo: Callable[[], str],
        get_ttl: Callable[[list[str]], int | None],
    ) -> None:
        self.cache_dir = cache_dir
        self._get_repo = get_repo
        self._get_ttl = get_ttl

    def get_lock_path(self, args: list[str]) -> Path | None:
        """Get lock file path for serializing API calls.

        Returns None if command shouldn't be serialized.
        """
        ttl = self._get_ttl(args)
        if ttl is None:
            return None  # Non-cacheable commands don't need serialization
        if len(args) < 2:
            return None
        repo = self._get_repo()
        # Lock per repo + command type (e.g., ai_template-issue-list.lock)
        lock_name = f"{repo}-{args[0]}-{args[1]}.lock"
        return self.cache_dir / lock_name

    def should_serialize(self, args: list[str]) -> bool:
        """Check if command should be serialized."""
        return self.get_lock_path(args) is not None

    def acquire_lock(self, args: list[str]) -> tuple[Any | None, bool]:
        """Acquire serialization lock with timeout.

        Returns:
            Tuple of (file_descriptor, got_lock).
            File descriptor is None if serialization not applicable.
            got_lock is True if lock was acquired, False on timeout.
        """
        lock_path = self.get_lock_path(args)
        if lock_path is None:
            return None, False

        try:
            lock_fd = open(lock_path, "w")
            got_lock = try_lock_file(lock_fd, SERIALIZE_TIMEOUT_SEC)
            return lock_fd, got_lock
        except Exception as e:
            debug_log(f"acquire_lock failed for {lock_path}: {e}")
            return None, False

    def release_lock(self, lock_fd: Any) -> None:
        """Release serialization lock."""
        if lock_fd is None:
            return
        try:
            unlock_file(lock_fd)
            lock_fd.close()
            # Note: Do NOT delete lock files - creates race condition (#1227)
            # Lock files are 0 bytes; accumulation is benign
        except Exception as e:
            debug_log(f"release_lock failed: {e}")


def cleanup_stale_locks(cache_dir: Path, max_age_hours: int = 24) -> tuple[int, int]:
    """Remove stale lock files from cache directory.

    Lock files accumulate from:
    1. Processes in temp directories (pytest) - always safe to delete
    2. Crashed processes that never released locks
    3. Normal operations where locks persist after use

    Args:
        cache_dir: Cache directory containing lock files.
        max_age_hours: Remove non-tmp locks older than this (default 24 hours)

    Returns:
        Tuple of (tmp_locks_removed, old_locks_removed)
    """
    tmp_removed = 0
    old_removed = 0
    cutoff = time.time() - (max_age_hours * 3600)

    for lf_path in cache_dir.glob("*.lock"):
        # Skip the main state lock file
        if lf_path.name == ".lock":
            continue
        try:
            # tmp* prefixed locks are from temp directories (pytest, etc.)
            # These directories no longer exist, so locks are safe to delete
            if lf_path.name.startswith("tmp"):
                lf_path.unlink()
                tmp_removed += 1
            # Regular locks older than threshold are likely orphaned
            elif lf_path.stat().st_mtime < cutoff:
                lf_path.unlink()
                old_removed += 1
        except OSError:
            pass

    return (tmp_removed, old_removed)

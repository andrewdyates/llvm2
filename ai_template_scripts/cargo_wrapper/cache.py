# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
cargo_wrapper/cache.py - Test Result Caching (#3212)

Caches successful cargo test results keyed by command args + commit hash + cwd.
Cache check happens BEFORE lock acquisition to avoid contention.

Two cache levels:
1. Hash-based exact cache: sha256(args + commit + cwd) -> result file
2. Time-based pulse cache: reads builds.log for recent same-command passes

Cache invalidation:
- New commit (git HEAD changes) -> cache miss
- CARGO_SKIP_CACHE=1 env var -> bypass cache entirely
- Entries older than 1 hour -> expired
- Test failures are never cached
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

from ai_template_scripts.cargo_wrapper import _state
from ai_template_scripts.shared_logging import debug_swallow

# Cache constants
CACHE_MAX_AGE_SEC = 3600  # 1 hour
PULSE_CACHE_AGE_SEC = 600  # 10 minutes for pulse check
CACHE_DIR_NAME = "test_cache"


def _cache_dir() -> Path | None:
    """Get the cache directory, creating if needed."""
    if _state.LOCK_DIR is None:
        return None
    cache_dir = _state.LOCK_DIR / CACHE_DIR_NAME
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        debug_swallow("cache_dir_create")
        return None
    return cache_dir


def _cache_key(args: list[str], commit: str) -> str:
    """Compute cache key from command args, commit hash, and cwd.

    The key is a SHA-256 hash of the normalized inputs, so any change
    in args, commit, or working directory produces a different key.
    """
    command_str = shlex.join(["cargo"] + args)
    cwd = os.getcwd()
    raw = f"{command_str}\n{commit}\n{cwd}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def check_cache(args: list[str], commit: str) -> dict | None:
    """Check if a cached result exists for this command at this commit.

    Returns the cached entry dict if found and valid, None otherwise.
    Call this BEFORE lock acquisition.

    Args:
        args: Cargo command args (e.g., ["test", "-p", "z4-chc"]).
        commit: Current git commit hash.

    Returns:
        Cached entry dict with keys: command, commit, exit_code,
        duration_sec, cached_at. Or None if no valid cache.
    """
    if os.environ.get("CARGO_SKIP_CACHE") == "1":
        return None

    if not commit:
        return None  # Can't cache without a commit

    cache_dir = _cache_dir()
    if cache_dir is None:
        return None

    key = _cache_key(args, commit)
    cache_file = cache_dir / f"{key}.json"

    try:
        if not cache_file.exists():
            return None

        data = json.loads(cache_file.read_text())

        # Validate required fields
        if data.get("exit_code") != 0:
            # Never serve cached failures
            cache_file.unlink(missing_ok=True)
            return None

        # Check age
        cached_at = data.get("cached_at", 0)
        age = time.time() - cached_at
        if age > CACHE_MAX_AGE_SEC:
            cache_file.unlink(missing_ok=True)
            return None

        # Verify commit matches (defense in depth)
        if data.get("commit") != commit:
            cache_file.unlink(missing_ok=True)
            return None

        data["age_sec"] = round(age)
        return data

    except (json.JSONDecodeError, OSError, KeyError):
        debug_swallow("cache_check")
        return None


def store_cache(args: list[str], commit: str, exit_code: int, duration_sec: float) -> None:
    """Store a successful test result in the cache.

    Only caches exit_code == 0 (successes). Failures are never cached.

    Args:
        args: Cargo command args.
        commit: Git commit hash.
        exit_code: Process exit code.
        duration_sec: Duration of the run in seconds.
    """
    if exit_code != 0:
        return  # Never cache failures

    if not commit:
        return

    cache_dir = _cache_dir()
    if cache_dir is None:
        return

    key = _cache_key(args, commit)
    cache_file = cache_dir / f"{key}.json"

    try:
        entry = {
            "command": shlex.join(["cargo"] + args),
            "commit": commit,
            "cwd": os.getcwd(),
            "exit_code": exit_code,
            "duration_sec": round(duration_sec, 1),
            "cached_at": time.time(),
        }
        cache_file.write_text(json.dumps(entry))
    except OSError:
        debug_swallow("cache_store")


def check_pulse_cache(args: list[str], commit: str) -> dict | None:
    """Check builds.log for a recent pass of the same command at the same commit.

    This is a softer check for "pulse" uses (Manager/Researcher just checking
    health). Reads the builds.log file without requiring a hash-based cache entry.

    Args:
        args: Cargo command args.
        commit: Current git commit hash.

    Returns:
        Matching builds.log entry dict, or None.
    """
    if os.environ.get("CARGO_SKIP_CACHE") == "1":
        return None

    if not commit or _state.BUILDS_LOG is None:
        return None

    try:
        if not _state.BUILDS_LOG.exists():
            return None

        command_str = shlex.join(["cargo"] + args)
        cwd = os.getcwd()
        now = time.time()

        # Read last 50 entries (more than check_retry_loop's 30)
        for line in reversed(_state.BUILDS_LOG.read_text().splitlines()[-50:]):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                entry.get("command") == command_str
                and entry.get("cwd") == cwd
                and entry.get("commit") == commit
                and entry.get("exit_code") == 0
            ):
                # Check age from finished_at
                finished_str = entry.get("finished_at", "")
                if not finished_str:
                    continue

                import datetime

                try:
                    finished = datetime.datetime.fromisoformat(finished_str)
                    if finished.tzinfo is None:
                        finished = finished.replace(tzinfo=datetime.UTC)
                    age_sec = (
                        datetime.datetime.now(datetime.UTC) - finished
                    ).total_seconds()
                    if age_sec <= PULSE_CACHE_AGE_SEC:
                        entry["age_sec"] = round(age_sec)
                        return entry
                except (ValueError, TypeError):
                    continue

        return None

    except OSError:
        debug_swallow("pulse_cache_check")
        return None


def print_cache_hit(entry: dict, cache_type: str = "CACHED") -> None:
    """Print a cache hit notice to stderr.

    Args:
        entry: The cached/pulse entry dict.
        cache_type: Label for the notice (CACHED or PULSE).
    """
    command = entry.get("command", "cargo test")
    age_sec = entry.get("age_sec", 0)
    commit = entry.get("commit", "?")
    duration = entry.get("duration_sec", 0)

    age_str = _format_age(age_sec)

    print(
        f"[cargo] {cache_type}: {command} passed {age_str} ago "
        f"(commit {commit}, duration {duration}s)",
        file=sys.stderr,
    )
    print(
        f"[cargo] To force re-run: CARGO_SKIP_CACHE=1 {command}",
        file=sys.stderr,
    )


def _format_age(seconds: int) -> str:
    """Format age in human-readable form."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h{minutes % 60}m"


def cleanup_expired_cache() -> int:
    """Remove expired cache entries. Returns count of removed entries."""
    cache_dir = _cache_dir()
    if cache_dir is None:
        return 0

    removed = 0
    now = time.time()
    try:
        for cache_file in cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                cached_at = data.get("cached_at", 0)
                if now - cached_at > CACHE_MAX_AGE_SEC:
                    cache_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, OSError):
                # Corrupt entry — remove it
                try:
                    cache_file.unlink()
                    removed += 1
                except OSError:
                    pass
    except OSError:
        debug_swallow("cache_cleanup")

    return removed

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Zone-based file/directory locking for multi-worker coordination.

Zones prevent edit conflicts when multiple workers operate on the same repo.
Each worker is assigned a zone (set of glob patterns) and can only edit
files within their zone.

Configuration (.looper_config.json):
    {
      "zones": {
        "worker_1": ["looper/**", "tests/test_looper*.py"],
        "worker_2": ["ai_template_scripts/**"]
      }
    }

If no zones configured, all workers share full access (existing behavior).
"""

from __future__ import annotations

import errno
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict

from looper.log import debug_swallow, log_warning


class WorkerInfo(TypedDict):
    """Type for worker zone info in status output."""

    patterns: list[str]
    locked: bool


class ZoneStatus(TypedDict):
    """Type for zone status output."""

    zones_enabled: bool
    workers: dict[str, WorkerInfo]


if TYPE_CHECKING:
    from typing import TextIO

__all__ = [
    # Classes
    "ZoneLock",
    "WorkerInfo",
    "ZoneStatus",
    # Functions
    "load_zone_config",
    "get_worker_zone_patterns",
    "file_in_zone",
    "can_edit_file",
    "get_zone_status",
    "check_files_in_zone",
    "get_zone_status_line",
    "main",
]

# fcntl is Unix-only; Windows uses different locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


def load_zone_config() -> dict[str, list[str]]:
    """Load zone configuration from .looper_config.json.

    Returns:
        Dict mapping worker ID to list of glob patterns.
        Empty dict if no zones configured.

    Example:
        {"worker_1": ["looper/**"], "worker_2": ["scripts/**"]}
    """
    config_file = Path(".looper_config.json")
    if not config_file.exists():
        return {}

    try:
        data = json.loads(config_file.read_text())
        zones = data.get("zones", {})
        if isinstance(zones, dict):
            # Validate structure: each value should be a list of strings
            validated: dict[str, list[str]] = {}
            for worker_id, patterns in zones.items():
                if isinstance(patterns, list) and all(
                    isinstance(p, str) for p in patterns
                ):
                    validated[str(worker_id)] = patterns
            return validated
    except (json.JSONDecodeError, OSError) as e:
        debug_swallow("load_zone_config", e)
    return {}


def get_worker_zone_patterns(worker_id: int | None) -> list[str]:
    """Get glob patterns for a worker's zone.

    Args:
        worker_id: Worker ID (1, 2, ...) or None for single-worker mode

    Returns:
        List of glob patterns for this worker's zone.
        Empty list means no zone restrictions (full access).
    """
    if worker_id is None:
        # Single-worker mode: no zone restrictions
        return []

    zones = load_zone_config()
    if not zones:
        # No zones configured: all workers share full access
        return []

    key = f"worker_{worker_id}"
    return zones.get(key, [])


def file_in_zone(file_path: str | Path, patterns: list[str]) -> bool:
    """Check if a file path matches any zone pattern.

    Args:
        file_path: Path to check (absolute or relative to repo root)
        patterns: List of glob patterns (e.g., ["looper/**", "*.py"])

    Returns:
        True if file matches any pattern, False otherwise.
        Returns True if patterns is empty (no restrictions).
    """
    if not patterns:
        # No restrictions = everything allowed
        return True

    # Normalize path: resolve and make relative to cwd
    path = Path(file_path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            # Path not under cwd - unlikely to be in zone
            return False

    path_str = str(path)

    for pattern in patterns:
        # fnmatch.fnmatch handles basic globs
        # For ** patterns, we need to check each directory level
        if "**" in pattern:
            # Split pattern at **
            if fnmatch.fnmatch(path_str, pattern.replace("**", "*")):
                return True
            # Also match the directory itself
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                prefix = prefix.rstrip("/")
                if path_str.startswith(prefix):
                    # Path starts with the prefix before **
                    if not suffix or suffix == "/":
                        return True
                    remainder = path_str[len(prefix) :].lstrip("/")
                    if fnmatch.fnmatch(remainder, "*" + suffix):
                        return True
        elif fnmatch.fnmatch(path_str, pattern):
            return True

    return False


def can_edit_file(
    file_path: str | Path, worker_id: int | None, *, warn: bool = True
) -> bool:
    """Check if worker can edit a file based on zone configuration.

    Args:
        file_path: Path to the file
        worker_id: Worker ID or None for single-worker mode
        warn: If True, print warning when file is outside zone

    Returns:
        True if worker can edit, False if blocked by zone restriction.
    """
    patterns = get_worker_zone_patterns(worker_id)
    if not patterns:
        # No zone restrictions
        return True

    if file_in_zone(file_path, patterns):
        return True

    if warn:
        log_warning(f"Warning: File '{file_path}' is outside worker_{worker_id}'s zone")
        log_warning(f"  Allowed patterns: {patterns}")
    return False


class ZoneLock:
    """File lock for exclusive zone access.

    Use this when a worker needs temporary exclusive access to their zone,
    e.g., during complex multi-file operations.

    Example:
        with ZoneLock(worker_id=1) as lock:
            if lock.acquired:
                # Exclusive zone access
                ...
    """

    def __init__(self, worker_id: int, lock_dir: Path | None = None) -> None:
        """Initialize zone lock.

        Args:
            worker_id: Worker ID to lock zone for
            lock_dir: Directory for lock files (default: .locks/)
        """
        self.worker_id = worker_id
        self.lock_dir = lock_dir or Path(".locks")
        self.lock_file: TextIO | None = None
        self.acquired = False

    def _lock_path(self) -> Path:
        """Get path to lock file for this worker's zone."""
        return self.lock_dir / f"zone_worker_{self.worker_id}.lock"

    def acquire(self, timeout: float = 30.0) -> bool:
        """Try to acquire zone lock.

        REQUIRES: timeout >= 0
        ENSURES: Returns within timeout + 0.25s under normal scheduling

        Args:
            timeout: Max seconds to wait for lock

        Returns:
            True if lock acquired, False otherwise.
        """
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        if not HAS_FCNTL:
            # Windows: no flock support, always succeed (best effort)
            self.acquired = True
            return True

        # Ensure lock directory exists
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path()

        try:
            self.lock_file = open(lock_path, "w")
        except OSError:
            return False

        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID to lock file for debugging
                self.lock_file.write(f"{os.getpid()}\n")
                self.lock_file.flush()
                self.acquired = True
                return True
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Adaptive sleep: cap at remaining time to limit timeout drift.
                time.sleep(min(0.1, remaining))
            except OSError as e:
                # Log unexpected OS errors for diagnostics
                # EACCES (permission denied) and ENOENT (file not found) are expected
                # in some scenarios, but other errors indicate real problems
                if e.errno not in (errno.EACCES, errno.ENOENT):
                    log_warning(
                        f"Warning: ZoneLock.acquire() failed: {e}",
                        stream="stderr",
                    )
                break

        # Failed to acquire
        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None
        return False

    def release(self) -> None:
        """Release zone lock."""
        if self.lock_file and HAS_FCNTL:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            except OSError as e:
                debug_swallow("zone_lock_release_unlock", e)
            try:
                self.lock_file.close()
            except OSError as e:
                debug_swallow("zone_lock_release_close", e)
            self.lock_file = None
        self.acquired = False

    def __enter__(self) -> Self:
        """Context manager entry - acquire lock."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - release lock."""
        self.release()


def get_zone_status() -> ZoneStatus:
    """Get current zone configuration and lock status.

    Returns:
        ZoneStatus with zone info for status displays:
        {
            "zones_enabled": True/False,
            "workers": {
                "worker_1": {"patterns": [...], "locked": True/False},
                ...
            }
        }
    """
    zones = load_zone_config()
    lock_dir = Path(".locks")

    workers: dict[str, WorkerInfo] = {}
    result: ZoneStatus = {
        "zones_enabled": bool(zones),
        "workers": workers,
    }

    for worker_key, patterns in zones.items():
        # Check if lock file exists and has recent PID
        lock_path = lock_dir / f"zone_{worker_key}.lock"
        locked = lock_path.exists() and lock_path.stat().st_size > 0

        workers[worker_key] = WorkerInfo(
            patterns=patterns,
            locked=locked,
        )

    return result


def check_files_in_zone(
    files: list[str], worker_id: int | None
) -> tuple[bool, list[str]]:
    """Check if all files are within a worker's zone.

    Args:
        files: List of file paths to check
        worker_id: Worker ID or None for single-worker mode

    Returns:
        (all_ok, violations) - all_ok is True if all files in zone,
        violations is list of files outside zone.
    """
    patterns = get_worker_zone_patterns(worker_id)
    if not patterns:
        # No zone restrictions
        return True, []

    violations = [f for f in files if not file_in_zone(f, patterns)]
    return len(violations) == 0, violations


def get_zone_status_line(worker_id: int | None) -> str:
    """Get one-line zone status for session display.

    Args:
        worker_id: Worker ID or None for single-worker mode

    Returns:
        Status line like "Zone: looper/**, tests/*.py" or "Zone: (unrestricted)"
    """
    patterns = get_worker_zone_patterns(worker_id)
    if not patterns:
        return "Zone: (unrestricted)"

    # Show first few patterns, truncate if many
    if len(patterns) <= 3:
        pattern_str = ", ".join(patterns)
    else:
        pattern_str = ", ".join(patterns[:3]) + f" (... and {len(patterns) - 3} more)"

    return f"Zone: {pattern_str}"


def main() -> int:
    """CLI entry point for zone checks.

    Usage:
        python -m looper.zones check FILE [FILE...]  # Check files in zone
        python -m looper.zones status                # Show zone status

    Environment:
        AI_WORKER_ID: Worker ID (1, 2, ...) or unset for single-worker

    Exit codes:
        0: All files in zone / status shown
        1: Files outside zone (listed to stderr)
        2: Usage error
    """
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m looper.zones check FILE [FILE...]", file=sys.stderr)
        print("       python -m looper.zones status", file=sys.stderr)
        return 2

    # Get worker ID from environment
    worker_id_str = os.environ.get("AI_WORKER_ID")
    worker_id = int(worker_id_str) if worker_id_str else None

    command = args[0]

    if command == "status":
        print(get_zone_status_line(worker_id))
        status = get_zone_status()
        if status["zones_enabled"]:
            for worker_key, info in status["workers"].items():
                lock_str = " [LOCKED]" if info["locked"] else ""
                print(f"  {worker_key}: {info['patterns']}{lock_str}")
        return 0

    if command == "check":
        files = args[1:]
        if not files:
            print("Error: No files specified", file=sys.stderr)
            return 2

        all_ok, violations = check_files_in_zone(files, worker_id)
        if all_ok:
            return 0

        # Report violations
        patterns = get_worker_zone_patterns(worker_id)
        print(
            f"Zone violation: {len(violations)} file(s) outside zone", file=sys.stderr
        )
        print(f"Worker {worker_id} zone patterns: {patterns}", file=sys.stderr)
        print("Files outside zone:", file=sys.stderr)
        for f in violations:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

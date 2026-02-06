# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Cleanup utilities for looper state files."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_template_scripts.subprocess_utils import is_process_alive
from looper.config import LOG_DIR
from looper.log import debug_swallow, log_info
from looper.rotation import (
    HAS_FCNTL,
    ROTATION_LOCK_FILE,
    ROTATION_STATE_FILE,
    load_rotation_state,
    save_rotation_state,
)

KNOWN_MODES = {"worker", "prover", "researcher", "manager"}

__all__ = [
    "cleanup_rotation_state",
    "cleanup_stale_commit_tag_files",
    "cleanup_stale_pid_files",
    "cleanup_stale_state_files",
    "cleanup_stale_status_files",
]


def _parse_iso_timestamp(value: str) -> datetime | None:
    """Parse ISO timestamp, returning timezone-aware datetime or None."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _age_hours_from_timestamp(ts: str, now: datetime) -> float | None:
    """Return age in hours from ISO timestamp string, or None if invalid."""
    parsed = _parse_iso_timestamp(ts)
    if not parsed:
        return None
    return (now - parsed).total_seconds() / 3600.0


def _age_hours_from_mtime(path: Path, now: datetime) -> float | None:
    """Return age in hours from file mtime, or None if stat fails."""
    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        debug_swallow("age_hours_from_mtime", e)
        return None
    return (now.timestamp() - mtime) / 3600.0


def cleanup_rotation_state(max_age_days: int = 7) -> int:
    """Remove stale rotation phase entries.

    Contracts:
        REQUIRES: max_age_days >= 0
        ENSURES: Returns number of phase entries removed
        ENSURES: Never raises (best-effort cleanup)
    """
    if max_age_days < 0:
        return 0
    if not ROTATION_STATE_FILE.exists():
        return 0

    lock_file = None
    if HAS_FCNTL:
        try:
            lock_file = open(ROTATION_LOCK_FILE, "w")
            import fcntl  # noqa: PLC0415

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError:
            if lock_file:
                lock_file.close()
                lock_file = None

    removed = 0
    try:
        state = load_rotation_state()
        if not state:
            return 0

        now = datetime.now(UTC)
        cutoff = now - timedelta(days=max_age_days)

        for role in list(state.keys()):
            phases = state.get(role, {})
            if not isinstance(phases, dict):
                del state[role]
                continue
            for phase in list(phases.keys()):
                phase_state = phases.get(phase, {})
                last_run = None
                if isinstance(phase_state, dict):
                    last_run = phase_state.get("last_run")
                if not isinstance(last_run, str):
                    continue
                parsed = _parse_iso_timestamp(last_run)
                if not parsed:
                    del phases[phase]
                    removed += 1
                    continue
                if parsed < cutoff:
                    del phases[phase]
                    removed += 1
            if not phases:
                del state[role]

        if removed:
            save_rotation_state(state)
    finally:
        if lock_file:
            if HAS_FCNTL:
                import fcntl  # noqa: PLC0415

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    return removed


def cleanup_stale_pid_files(coord_dir: Path) -> int:
    """Remove stale .pid_* files from coordination directory."""
    if not coord_dir.exists():
        return 0

    removed = 0
    for pid_file in coord_dir.glob(".pid_*"):
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pid = 0

        if pid <= 0 or not is_process_alive(pid):
            try:
                pid_file.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def cleanup_stale_status_files(repo_root: Path, max_age_hours: int = 1) -> int:
    """Remove status files older than max_age_hours with no live PID."""
    if max_age_hours < 0:
        return 0

    now = datetime.now(UTC)
    removed = 0

    for status_file in repo_root.glob(".*_status.json"):
        pid = 0
        updated_at = None
        try:
            data = json.loads(status_file.read_text())
            if isinstance(data, dict):
                pid_val = data.get("pid", 0)
                if isinstance(pid_val, int):
                    pid = pid_val
                updated_at_val = data.get("updated_at")
                if isinstance(updated_at_val, str):
                    updated_at = updated_at_val
        except (json.JSONDecodeError, OSError, TypeError) as e:
            debug_swallow("cleanup_stale_status_files_read", e)

        if pid > 0 and is_process_alive(pid):
            continue

        age_hours = None
        if updated_at:
            age_hours = _age_hours_from_timestamp(updated_at, now)
        if age_hours is None:
            age_hours = _age_hours_from_mtime(status_file, now)
        if age_hours is None or age_hours < max_age_hours:
            continue

        try:
            status_file.unlink()
            removed += 1
        except OSError:
            continue

    return removed


def cleanup_stale_commit_tag_files(
    log_dir: Path, max_age_days: int = 30, allowed_modes: set[str] | None = None
) -> int:
    """Remove commit tag files for unknown modes when older than max_age_days."""
    if max_age_days < 0 or not log_dir.exists():
        return 0
    allowed = allowed_modes or KNOWN_MODES
    removed = 0
    now = datetime.now(UTC)

    for path in log_dir.glob(".commit_tag_*"):
        if path.name.startswith(".commit_tag_lock_"):
            mode = path.name.removeprefix(".commit_tag_lock_")
        else:
            mode = path.name.removeprefix(".commit_tag_")
        if mode in allowed:
            continue
        age_hours = _age_hours_from_mtime(path, now)
        if age_hours is None or age_hours < max_age_days * 24:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue

    return removed


def cleanup_stale_state_files(
    repo_root: Path | None = None,
    coord_dir: Path | None = None,
    rotation_max_age_days: int = 7,
    status_max_age_hours: int = 1,
    commit_tag_max_age_days: int = 30,
) -> dict[str, int]:
    """Run cleanup for common looper state files.

    Returns:
        Dict with counts for each cleanup category.
    """
    repo_root = repo_root or Path.cwd()
    coord_dir = coord_dir or Path(os.environ.get("AIT_COORD_DIR", "."))

    results = {
        "rotation": cleanup_rotation_state(max_age_days=rotation_max_age_days),
        "pid_files": cleanup_stale_pid_files(coord_dir),
        "status_files": cleanup_stale_status_files(
            repo_root, max_age_hours=status_max_age_hours
        ),
        "commit_tag_files": cleanup_stale_commit_tag_files(
            LOG_DIR, max_age_days=commit_tag_max_age_days
        ),
    }

    total = sum(results.values())
    if total:
        log_info(
            "✓ Cleanup removed stale files: "
            f"rotation={results['rotation']}, "
            f"pid={results['pid_files']}, "
            f"status={results['status_files']}, "
            f"commit_tag={results['commit_tag_files']}"
        )
    return results

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Worker file tracking for multi-worker coordination.

Tracks files each worker modifies during a session to:
1. Scope auto-fix tools (ruff --fix) to only touch worker's files
2. Warn on `git add` when staging other workers' files
3. Prevent accidental commits of other workers' uncommitted changes

Design: designs/2026-01-30-worker-file-tracking.md

REQUIRES: AI_WORKER_ID environment variable set for multi-worker mode
ENSURES: Tracker files are gitignored (.worker_*_files.json)
ENSURES: Stale trackers (dead PIDs) are cleaned up
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from looper.log import debug_swallow, log_info, log_warning
from looper.subprocess_utils import run_git_command

__all__ = [
    "FileTracker",
    "TrackerState",
    "cleanup_stale_trackers",
    "get_tracker_filename",
    "get_uncommitted_files",
]

# Patterns that indicate display/summary strings, not real file paths (#2187)
# These get mistakenly written to tracker files and should be filtered out.
_DISPLAY_STRING_PATTERNS = [
    re.compile(r"^\.\.\. and \d+ more$"),  # "... and 10 more"
    re.compile(r"^\d+: \d+ file\(s\)$"),  # "1: 11 file(s)"
    re.compile(r"^\d+ \(you\): \d+ file\(s\)$"),  # "3 (you): 16 file(s)"
    re.compile(r"^W\d+.*: \d+ file\(s\)$"),  # "W1: 11 file(s)", "W3 (you): 16 file(s)"
    re.compile(r"^Active worker files:$"),  # Header line
]


def _is_valid_file_entry(entry: str) -> bool:
    """Check if entry looks like a valid file path, not a display string.

    ENSURES: Returns False for display/summary strings like "... and N more"
    ENSURES: Returns True for entries that could be valid file paths

    Args:
        entry: String to validate as a file path.

    Returns:
        True if entry looks like a valid file path.
    """
    if not entry or not entry.strip():
        return False

    entry = entry.strip()

    # Check against display string patterns
    for pattern in _DISPLAY_STRING_PATTERNS:
        if pattern.match(entry):
            return False

    return True


@dataclass
class TrackerState:
    """Persisted tracker state."""

    worker_id: int
    session_id: str
    pid: int
    files: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrackerState:
        """Create from dict (JSON deserialization)."""
        return cls(
            worker_id=data["worker_id"],
            session_id=data["session_id"],
            pid=data["pid"],
            files=data.get("files", []),
            updated_at=data.get("updated_at", ""),
        )


def get_tracker_filename(worker_id: int) -> str:
    """Generate tracker filename for a given worker ID.

    ENSURES: Returns filename like .worker_1_files.json
    """
    return f".worker_{worker_id}_files.json"


def get_uncommitted_files() -> list[str]:
    """Get list of modified/untracked files from git status.

    Reuses logic from checkpoint.py.

    ENSURES: Returns list of file paths (may be empty)
    ENSURES: Never raises - returns empty list on error
    """
    result = run_git_command(["status", "--porcelain"], timeout=5)
    if not result.ok or not result.value:
        return []

    files = []
    for line in result.value.rstrip("\n").split("\n"):
        if not line:
            continue
        # Format: XY filename or XY -> newname (for renames)
        parts = line[3:].split(" -> ")
        filename = parts[-1].strip()
        if filename:
            files.append(filename)
    return files


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running.

    REQUIRES: pid > 0
    ENSURES: Returns True if process is running
    ENSURES: Returns False if process is dead or inaccessible
    """
    try:
        os.kill(pid, 0)  # Doesn't kill, just checks existence
        return True  # Process exists
    except ProcessLookupError:
        return False  # Process doesn't exist
    except PermissionError:
        return True  # Process exists but we can't signal it


class FileTracker:
    """Tracks files modified by a worker during a session.

    REQUIRES: worker_id > 0
    REQUIRES: session_id is a non-empty string
    ENSURES: Atomic writes via temp file + rename
    ENSURES: Thread-safe read operations
    """

    def __init__(
        self,
        repo_root: Path,
        worker_id: int,
        session_id: str,
    ) -> None:
        """Initialize file tracker.

        Args:
            repo_root: Repository root directory
            worker_id: Worker instance ID (1, 2, etc.)
            session_id: Unique session identifier
        """
        self.repo_root = repo_root
        self.worker_id = worker_id
        self.session_id = session_id
        self.tracker_file = repo_root / get_tracker_filename(worker_id)
        self._pid = os.getpid()

    def load(self) -> TrackerState | None:
        """Load tracker state from disk.

        ENSURES: Returns None if file doesn't exist or is corrupted
        ENSURES: Returns None if tracker belongs to different session
        """
        state = self._load_state()
        if not state:
            return None
        if state.session_id != self.session_id:
            return None
        return state

    def _load_state(self) -> TrackerState | None:
        """Load tracker state without session ownership checks."""
        if not self.tracker_file.exists():
            return None

        try:
            data = json.loads(self.tracker_file.read_text())
            state = TrackerState.from_dict(data)

            # Verify this is our tracker
            if state.worker_id != self.worker_id:
                return None
            return state
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted tracker - will be overwritten on save
            debug_swallow("load_state_corrupted", e)
            return None

    def _has_conflict(self) -> bool:
        """Check if tracker belongs to another active session."""
        state = self._load_state()
        if not state:
            return False
        if state.session_id == self.session_id:
            return False
        return _is_pid_alive(state.pid)

    def save(self, files: list[str] | None = None) -> bool:
        """Save tracker state atomically.

        Args:
            files: List of tracked files. If None, uses current uncommitted files.

        ENSURES: Atomic write via temp + rename
        ENSURES: Returns True on success, False on error or conflict
        ENSURES: Display/summary strings are filtered out (#2187)
        """
        if self._has_conflict():
            log_warning("Warning: file tracker belongs to another active session")
            return False

        if files is None:
            files = get_uncommitted_files()

        # Filter out display/summary strings (#2187)
        # These can get passed in when tool output is incorrectly captured
        valid_files = [f for f in files if _is_valid_file_entry(f)]
        invalid_count = len(files) - len(valid_files)
        if invalid_count > 0:
            log_warning(
                f"Warning: file tracker filtered {invalid_count} invalid entries "
                f"(display strings, not file paths)"
            )

        state = TrackerState(
            worker_id=self.worker_id,
            session_id=self.session_id,
            pid=self._pid,
            files=sorted(set(valid_files)),
            updated_at=datetime.now(UTC).isoformat(),
        )

        try:
            # Atomic write: write to temp file, then rename
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.repo_root,
                prefix=".worker_tmp_",
                suffix=".json",
                delete=False,
            ) as tmp_f:
                json.dump(state.to_dict(), tmp_f, indent=2)
                tmp_path = Path(tmp_f.name)

            # Atomic rename
            try:
                tmp_path.rename(self.tracker_file)
            except OSError:
                tmp_path.unlink(missing_ok=True)
                raise
            return True

        except OSError as e:
            log_warning(f"Warning: file tracker write failed: {e}")
            return False

    def add_files(self, new_files: list[str]) -> bool:
        """Add files to the tracker.

        ENSURES: Merges with existing files
        ENSURES: Returns True on success, False on conflict
        """
        if self._has_conflict():
            log_warning("Warning: file tracker belongs to another active session")
            return False
        state = self.load()
        existing_files = state.files if state else []
        all_files = list(set(existing_files) | set(new_files))
        return self.save(all_files)

    def clear_committed(self, committed_files: list[str]) -> bool:
        """Remove committed files from tracking.

        Called by post-commit hook to clean up committed files.

        ENSURES: Only removes files that were committed
        ENSURES: Returns True on success, False on conflict
        """
        if self._has_conflict():
            log_warning("Warning: file tracker belongs to another active session")
            return False
        state = self.load()
        if not state:
            return True  # Nothing to clear

        committed_set = set(committed_files)
        remaining = [f for f in state.files if f not in committed_set]
        return self.save(remaining)

    def clear(self) -> bool:
        """Clear tracker file entirely.

        ENSURES: Tracker file deleted if exists
        ENSURES: Returns True on success or if file didn't exist
        ENSURES: Returns False on conflict
        """
        try:
            if self._has_conflict():
                log_warning("Warning: file tracker belongs to another active session")
                return False
            if self.tracker_file.exists():
                self.tracker_file.unlink()
            return True
        except OSError as e:
            log_warning(f"Warning: file tracker clear failed: {e}")
            return False

    def get_tracked_files(self) -> list[str]:
        """Get list of currently tracked files.

        ENSURES: Returns empty list if no tracker or corrupted
        """
        state = self.load()
        return state.files if state else []

    def is_file_tracked(self, filepath: str) -> bool:
        """Check if a file is tracked by this worker.

        ENSURES: Returns True if file is in tracked list
        """
        return filepath in self.get_tracked_files()


def cleanup_stale_trackers(repo_root: Path) -> list[str]:
    """Remove tracker files for dead PIDs.

    Called at iteration start to clean up trackers from crashed sessions.

    ENSURES: Returns list of cleaned up tracker filenames
    ENSURES: Never raises - logs errors and continues
    """
    cleaned = []
    tracker_pattern = re.compile(r"^\.worker_(\d+)_files\.json$")

    try:
        for path in repo_root.iterdir():
            match = tracker_pattern.match(path.name)
            if not match:
                continue

            try:
                data = json.loads(path.read_text())
                pid = data.get("pid", 0)

                if pid > 0 and not _is_pid_alive(pid):
                    path.unlink()
                    cleaned.append(path.name)
                    log_info(f"Cleaned up stale tracker: {path.name} (PID {pid} dead)")
            except (json.JSONDecodeError, KeyError, OSError) as e:
                # Corrupted or inaccessible - try to remove
                try:
                    path.unlink()
                    cleaned.append(path.name)
                    log_info(f"Cleaned up corrupted tracker: {path.name} ({e})")
                except OSError as e:
                    debug_swallow("cleanup_corrupted_tracker_unlink", e)

    except OSError as e:
        log_warning(f"Warning: cleanup_stale_trackers failed to scan directory: {e}")

    return cleaned


def get_all_tracked_files(repo_root: Path) -> dict[int, list[str]]:
    """Get all tracked files from all active workers.

    ENSURES: Returns dict mapping worker_id -> list of files
    ENSURES: Only includes trackers with alive PIDs
    """
    result: dict[int, list[str]] = {}
    tracker_pattern = re.compile(r"^\.worker_(\d+)_files\.json$")

    try:
        for path in repo_root.iterdir():
            match = tracker_pattern.match(path.name)
            if not match:
                continue

            try:
                data = json.loads(path.read_text())
                pid = data.get("pid", 0)
                worker_id = data.get("worker_id", 0)
                files = data.get("files", [])

                if pid > 0 and _is_pid_alive(pid) and worker_id > 0:
                    result[worker_id] = files
            except (json.JSONDecodeError, KeyError, OSError) as e:
                debug_swallow("get_all_tracked_files_parse", e)
                continue

    except OSError as e:
        debug_swallow("get_all_tracked_files_iter", e)

    return result

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
changelog.py - Local Change Log for Offline-First Issue Management

Tracks changes made locally that need to sync to GitHub.
Sync happens opportunistically when API quota is available.

Usage:
    from ai_template_scripts.gh_rate_limit import ChangeLog, Change, get_change_log

    log = ChangeLog()
    log.add_change("ai_template", "create", {"title": "...", "body": "..."})
    # Later, when quota available:
    pending = log.get_pending()
    for change in pending:
        success = sync_to_github(change)
        if success:
            log.mark_synced(change.id)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ai_template_scripts.gh_rate_limit.limiter import debug_log

__all__ = [
    "Change",
    "ChangeLog",
    "get_change_log",
    "CHANGE_LOG_FILE",
    "MAX_CHANGE_LOG_ENTRIES",
    "CHANGE_LOG_PRUNE_AGE_HOURS",
]

# Paths
CACHE_DIR = Path.home() / ".ait_gh_cache"
CHANGE_LOG_FILE = CACHE_DIR / "change_log.json"

# ChangeLog automatic pruning (#1703)
MAX_CHANGE_LOG_ENTRIES = 500  # After this, prune old synced changes
CHANGE_LOG_PRUNE_AGE_HOURS = 24  # Prune synced changes older than this


@dataclass
class Change:
    """A pending change to sync to GitHub."""

    id: str  # UUID
    timestamp: float
    repo: str
    operation: str  # create, edit, close, comment
    data: dict  # Operation-specific data
    synced: bool = False
    sync_error: str | None = None


class ChangeLog:
    """Local change log for offline-first issue operations.

    Tracks changes made locally that need to sync to GitHub.
    Sync happens opportunistically when API quota is available.

    Usage:
        log = ChangeLog()
        log.add_change("ai_template", "create", {"title": "...", "body": "..."})
        # Later, when quota available:
        pending = log.get_pending()
        for change in pending:
            success = sync_to_github(change)
            if success:
                log.mark_synced(change.id)
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.cache_dir / "change_log.json"
        self._changes: list[Change] = []
        self._load()

    def _load(self) -> None:
        """Load change log from file."""
        if not self.log_file.exists():
            return
        try:
            data = json.loads(self.log_file.read_text())
            for entry in data.get("changes", []):
                self._changes.append(
                    Change(
                        id=entry["id"],
                        timestamp=entry["timestamp"],
                        repo=entry["repo"],
                        operation=entry["operation"],
                        data=entry["data"],
                        synced=entry.get("synced", False),
                        sync_error=entry.get("sync_error"),
                    )
                )
        except Exception as e:
            debug_log(f"ChangeLog._load failed: {e}")

    def _save(self) -> None:
        """Save change log to file."""
        try:
            data = {
                "changes": [
                    {
                        "id": c.id,
                        "timestamp": c.timestamp,
                        "repo": c.repo,
                        "operation": c.operation,
                        "data": c.data,
                        "synced": c.synced,
                        "sync_error": c.sync_error,
                    }
                    for c in self._changes
                ]
            }
            self.log_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            debug_log(f"ChangeLog._save failed: {e}")

    def add_change(self, repo: str, operation: str, data: dict) -> str:
        """Add a change to the log. Returns change ID.

        Automatically prunes old synced changes when log exceeds
        MAX_CHANGE_LOG_ENTRIES (#1703).
        """
        change_id = str(uuid.uuid4())[:8]
        change = Change(
            id=change_id,
            timestamp=time.time(),
            repo=repo,
            operation=operation,
            data=data,
        )
        self._changes.append(change)
        # Auto-prune when log gets large (#1703)
        if len(self._changes) > MAX_CHANGE_LOG_ENTRIES:
            self._prune_synced()
        self._save()
        return change_id

    def _prune_synced(self) -> None:
        """Prune synced changes to stay under MAX_CHANGE_LOG_ENTRIES.

        Strategy (#1703):
        1. First remove synced changes older than CHANGE_LOG_PRUNE_AGE_HOURS
        2. If still over limit, remove oldest synced changes regardless of age
           (keeps pending changes safe)
        """
        # Phase 1: Age-based pruning
        cutoff = time.time() - (CHANGE_LOG_PRUNE_AGE_HOURS * 3600)
        self._changes = [
            c for c in self._changes if not c.synced or c.timestamp > cutoff
        ]

        # Phase 2: If still over limit, remove oldest synced regardless of age
        if len(self._changes) > MAX_CHANGE_LOG_ENTRIES:
            # Separate pending and synced
            pending = [c for c in self._changes if not c.synced]
            synced = [c for c in self._changes if c.synced]
            # Sort synced by timestamp (oldest first) and keep only what fits
            synced.sort(key=lambda c: c.timestamp)
            keep_count = MAX_CHANGE_LOG_ENTRIES - len(pending)
            if keep_count > 0:
                synced = synced[-keep_count:]  # Keep newest synced
            else:
                synced = []  # All space needed for pending
            # Reconstruct in original order (pending preserves order)
            # Use set for O(1) lookup instead of O(n) list membership
            synced_ids = {c.id for c in synced}
            self._changes = [
                c for c in self._changes if not c.synced or c.id in synced_ids
            ]

    def get_pending(self, repo: str | None = None) -> list[Change]:
        """Get pending (unsynced) changes, optionally filtered by repo."""
        pending = [c for c in self._changes if not c.synced]
        if repo:
            pending = [c for c in pending if c.repo == repo]
        return sorted(pending, key=lambda c: c.timestamp)

    def get_all(self, include_synced: bool = False) -> list[Change]:
        """Get all changes, optionally including synced ones."""
        if include_synced:
            return list(self._changes)
        return [c for c in self._changes if not c.synced]

    def mark_synced(self, change_id: str) -> bool:
        """Mark a change as synced. Returns True if found."""
        for c in self._changes:
            if c.id == change_id:
                c.synced = True
                c.sync_error = None
                self._save()
                return True
        return False

    def mark_error(self, change_id: str, error: str) -> bool:
        """Mark a change as failed sync. Returns True if found."""
        for c in self._changes:
            if c.id == change_id:
                c.sync_error = error
                self._save()
                return True
        return False

    def clear_synced(self, max_age_hours: int = 24) -> int:
        """Remove old synced changes. Returns count removed."""
        cutoff = time.time() - (max_age_hours * 3600)
        before = len(self._changes)
        self._changes = [
            c for c in self._changes if not c.synced or c.timestamp > cutoff
        ]
        after = len(self._changes)
        if before != after:
            self._save()
        return before - after

    def pending_count(self, repo: str | None = None) -> int:
        """Count of pending changes."""
        return len(self.get_pending(repo))


# Singleton change log
_change_log: ChangeLog | None = None


def get_change_log() -> ChangeLog:
    """Get singleton change log instance."""
    global _change_log
    if _change_log is None:
        _change_log = ChangeLog()
    return _change_log

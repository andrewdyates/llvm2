#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Pulse metrics storage - file I/O, rotation, and compaction.

Part of #404: pulse.py module split
"""

import gzip
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

from ai_template_scripts.atomic_write import atomic_write_text

from .constants import (
    MAX_METRICS_PER_DAY,
    METRICS_ARCHIVE_DIR,
    METRICS_DIR,
    METRICS_RETENTION_DAYS,
)

try:
    from ai_template_scripts.subprocess_utils import (
        get_repo_name as _canonical_get_repo_name,
    )
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import (
        get_repo_name as _canonical_get_repo_name,
    )

fcntl: ModuleType | None
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows/unavailable fcntl
    fcntl = None


def get_repo_name() -> str:
    """Get current repo name from git remote.

    Delegates to canonical subprocess_utils.get_repo_name() for consistency.
    See #1267 for rationale on consolidating get_repo variants.
    """
    result = _canonical_get_repo_name()
    return result.stdout


@contextmanager
def _metrics_write_lock() -> Iterator[None]:
    """Serialize write_metrics read-modify-write to prevent lost updates."""
    if fcntl is None:
        yield None
        return
    lock_path = METRICS_DIR / ".write_metrics.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield None
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _rotate_old_metrics() -> None:
    """Archive or delete metrics files older than retention period.

    Files older than METRICS_RETENTION_DAYS are moved to archive/ as gzipped JSON.
    """
    cutoff = datetime.now() - timedelta(days=METRICS_RETENTION_DAYS)

    for metrics_file in METRICS_DIR.glob("20??-??-??.json"):
        # Parse date from filename
        try:
            file_date = datetime.strptime(metrics_file.stem, "%Y-%m-%d")
        except ValueError:
            continue

        if file_date >= cutoff:
            continue  # Keep recent files

        # Archive old files
        METRICS_ARCHIVE_DIR.mkdir(exist_ok=True)
        archive_file = METRICS_ARCHIVE_DIR / f"{metrics_file.stem}.json.gz"

        # Skip if archive already exists (previous partial run)
        if archive_file.exists():
            continue

        try:
            content = metrics_file.read_bytes()
            with gzip.open(archive_file, "wb") as gz:
                gz.write(content)
            metrics_file.unlink()
        except Exception as e:
            # Best-effort: metrics archival is housekeeping, but log errors
            print(f"{metrics_file.name}: archive error - {e}")


def write_metrics(metrics: dict) -> None:
    """Write metrics to file with rotation.

    REQUIRES: metrics is a non-empty dict
    ENSURES: Creates metrics/ directory if not exists
    ENSURES: Appends metrics to daily file (YYYY-MM-DD.json)
    ENSURES: Writes human-readable latest.json for debugging
    ENSURES: Rotates old files (>7 days) probabilistically
    """
    METRICS_DIR.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    metrics_file = METRICS_DIR / f"{date_str}.json"
    with _metrics_write_lock():
        # Append to daily file
        existing = []
        if metrics_file.exists():
            try:
                existing = json.loads(metrics_file.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except json.JSONDecodeError:
                existing = []

        existing.append(metrics)

        # Trim to max entries per day (keep most recent)
        if len(existing) > MAX_METRICS_PER_DAY:
            existing = existing[-MAX_METRICS_PER_DAY:]

        # Use compact JSON to reduce file size (154 lines -> 1 line per entry)
        # Atomic write prevents truncated JSON on interruption (#2963)
        atomic_write_text(metrics_file, json.dumps(existing, separators=(",", ":")))

        # Keep latest.json human-readable for debugging
        atomic_write_text(METRICS_DIR / "latest.json", json.dumps(metrics, indent=2))

    # Periodically rotate old files (1 in 10 calls)
    if random.random() < 0.1:
        _rotate_old_metrics()


def _trim_current_metrics() -> None:
    """Trim current day's metrics file to MAX_METRICS_PER_DAY entries."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    metrics_file = METRICS_DIR / f"{date_str}.json"

    if not metrics_file.exists():
        print(f"No metrics file for today: {metrics_file}")
        return

    try:
        existing = json.loads(metrics_file.read_text())
        if not isinstance(existing, list):
            existing = [existing]
    except json.JSONDecodeError:
        print(f"Error reading {metrics_file}")
        return

    before = len(existing)
    if before <= MAX_METRICS_PER_DAY:
        print(f"{metrics_file}: {before} entries (within limit)")
        return

    existing = existing[-MAX_METRICS_PER_DAY:]
    atomic_write_text(metrics_file, json.dumps(existing, separators=(",", ":")))
    print(f"{metrics_file}: trimmed {before} -> {len(existing)} entries")


def _compact_metrics_files() -> None:
    """Convert all metrics files to compact JSON format."""
    for metrics_file in METRICS_DIR.glob("20??-??-??.json"):
        try:
            data = json.loads(metrics_file.read_text())
            before_size = metrics_file.stat().st_size
            if before_size == 0:
                print(f"{metrics_file.name}: empty file, skipped")
                continue
            atomic_write_text(metrics_file, json.dumps(data, separators=(",", ":")))
            after_size = metrics_file.stat().st_size
            reduction = (1 - after_size / before_size) * 100
            if reduction < 1:
                print(f"{metrics_file.name}: already compact ({after_size // 1024}KB)")
            else:
                before_kb = before_size // 1024
                after_kb = after_size // 1024
                pct = f"{reduction:.0f}%"
                print(
                    f"{metrics_file.name}: {before_kb}KB -> {after_kb}KB ({pct} reduction)"
                )
        except Exception as e:
            # Best-effort: metrics compaction failure reported inline
            print(f"{metrics_file.name}: error - {e}")

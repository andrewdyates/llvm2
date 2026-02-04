# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Storage layer for bg_task package.

Contains:
- Path utilities (get_git_root, get_tasks_dir, get_task_dir)
- JSON I/O helpers (_read_json, _atomic_write_json)
- Manifest operations (load_manifest, save_manifest)
- Task metadata persistence (load_task_meta, save_task_meta)
- Locking (_task_start_lock, _manifest_lock)
"""

from __future__ import annotations

__all__ = [
    "get_git_root",
    "get_tasks_dir",
    "get_task_dir",
    "load_manifest",
    "save_manifest",
    "load_task_meta",
    "save_task_meta",
]

import contextlib
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from ai_template_scripts.bg_task.types import TASKS_DIR, TaskMeta

fcntl: ModuleType | None
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows/unavailable fcntl
    fcntl = None


def get_git_root() -> Path:
    """Get the git repository root directory.

    REQUIRES: CWD is inside a git repository
    ENSURES: return value is absolute path to repository root
    ENSURES: return value exists and is a directory
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return Path(result.stdout.strip())


def get_tasks_dir() -> Path:
    """Get the background tasks directory.

    REQUIRES: CWD is inside a git repository
    ENSURES: return value exists and is a directory
    ENSURES: return value == get_git_root() / TASKS_DIR
    """
    tasks_dir = get_git_root() / TASKS_DIR
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir


def get_task_dir(task_id: str) -> Path:
    """Get directory for a specific task.

    REQUIRES: task_id is non-empty string
    ENSURES: return value == get_tasks_dir() / task_id
    """
    return get_tasks_dir() / task_id


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON from a file, returning None on failure."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to a file atomically (tmp + replace)."""
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n")
    tmp_path.replace(path)


def load_manifest() -> dict[str, Any]:
    """Load the manifest file.

    REQUIRES: CWD is inside a git repository
    ENSURES: return value has "tasks" key
    ENSURES: return value["tasks"] is dict
    """
    manifest_path = get_tasks_dir() / "manifest.json"
    if manifest_path.exists():
        data = _read_json(manifest_path)
        if isinstance(data, dict) and isinstance(data.get("tasks"), dict):
            return data
    return {"tasks": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Save the manifest file.

    REQUIRES: manifest has "tasks" key
    ENSURES: manifest.json exists with valid JSON
    """
    manifest_path = get_tasks_dir() / "manifest.json"
    _atomic_write_json(manifest_path, manifest)


def load_task_meta(task_id: str) -> TaskMeta | None:
    """Load metadata for a specific task.

    REQUIRES: task_id is non-empty string
    ENSURES: return is None if task directory/meta.json doesn't exist
    ENSURES: return is TaskMeta if meta.json exists and is valid
    """
    meta_path = get_task_dir(task_id) / "meta.json"
    if meta_path.exists():
        data = _read_json(meta_path)
        if data is None:
            return None
        try:
            return TaskMeta.from_dict(data)
        except (TypeError, ValueError):
            return None
    return None


def save_task_meta(meta: TaskMeta) -> None:
    """Save metadata for a task.

    REQUIRES: meta.task_id is non-empty string
    ENSURES: task directory exists
    ENSURES: meta.json exists with valid JSON representation of meta
    """
    task_dir = get_task_dir(meta.task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    meta_path = task_dir / "meta.json"
    _atomic_write_json(meta_path, meta.to_dict())


@contextlib.contextmanager
def _task_start_lock(task_dir: Path) -> Iterator[None]:
    """Serialize task creation to avoid duplicate starts (TOCTOU protection)."""
    if fcntl is None:
        yield None
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / ".start.lock"
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield None
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


@contextlib.contextmanager
def _manifest_lock(tasks_dir: Path | None = None) -> Iterator[None]:
    """Serialize manifest updates to avoid lost updates."""
    if fcntl is None:
        yield None
        return
    if tasks_dir is None:
        tasks_dir = get_tasks_dir()
    tasks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = tasks_dir / ".manifest.lock"
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield None
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass

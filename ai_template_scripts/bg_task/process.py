# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Process management for bg_task package.

Contains:
- Process liveness checking (is_process_alive)
- Task status updates (update_task_status)
- Background process spawning (_generate_wrapper_script, _spawn_background_process)
- Task lifecycle (start_task, kill_task, cleanup_tasks)
"""

from __future__ import annotations

__all__ = [
    "is_process_alive",
    "update_task_status",
    "start_task",
    "kill_task",
    "cleanup_tasks",
]

import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ai_template_scripts.bg_task.storage import (
    _manifest_lock,
    _read_json,
    _task_start_lock,
    get_task_dir,
    get_tasks_dir,
    load_manifest,
    load_task_meta,
    save_manifest,
    save_task_meta,
)
from ai_template_scripts.bg_task.types import DEFAULT_TIMEOUT, TaskMeta
from ai_template_scripts.subprocess_utils import is_process_alive  # canonical (#2535)


def update_task_status(task_id: str) -> TaskMeta | None:
    """Update task status based on process state.

    REQUIRES: task_id is non-empty string
    ENSURES: return is None if task doesn't exist
    ENSURES: if task was running but process is dead, status is updated
    ENSURES: status transitions from "running" to "completed"/"failed" when process ends
    ENSURES: meta.json and manifest.json are updated (not atomic; crash may cause inconsistency)
    """
    meta = load_task_meta(task_id)
    if meta is None:
        return None

    if meta.status == "running" and meta.pid:
        if not is_process_alive(meta.pid):
            # Process finished - check for result
            result_path = get_task_dir(task_id) / "result.json"
            if result_path.exists():
                result = _read_json(result_path)
                if isinstance(result, dict):
                    meta.exit_code = result.get("exit_code")
                    meta.finished_at = result.get("finished_at")
                    meta.status = "completed" if meta.exit_code == 0 else "failed"
                else:
                    meta.status = "failed"
                    meta.finished_at = datetime.now(UTC).isoformat()
            else:
                # Process died without writing result
                meta.status = "failed"
                meta.finished_at = datetime.now(UTC).isoformat()
            save_task_meta(meta)

            # Update manifest
            with _manifest_lock():
                manifest = load_manifest()
                tasks = manifest.get("tasks", {})
                if task_id in tasks:
                    tasks[task_id]["status"] = meta.status
                    save_manifest(manifest)

    return meta


def _generate_wrapper_script(
    command: list[str], timeout: int, output_path: Path, result_path: Path
) -> str:
    """Generate Python wrapper script for background task execution.

    Args:
        command: Command to execute.
        timeout: Timeout in seconds.
        output_path: Path to write stdout/stderr.
        result_path: Path to write result JSON.

    Returns:
        Python script as string.
    """
    return f"""
import subprocess
import sys
import json
import signal
from datetime import datetime, timezone

def handler(signum, frame):
    sys.exit(124)  # Timeout exit code

signal.signal(signal.SIGALRM, handler)
signal.alarm({timeout})

try:
    with open({str(output_path)!r}, 'w') as output_file:
        result = subprocess.run(
            {command!r},
            stdout=output_file,
            stderr=subprocess.STDOUT,
        )
    exit_code = result.returncode
except Exception as e:
    # Catch-all: subprocess execution failure logged to output file
    with open({str(output_path)!r}, 'a') as f:
        f.write(f"\\nError: {{e}}\\n")
    exit_code = 1

# Write result
with open({str(result_path)!r}, 'w') as f:
    json.dump({{
        "exit_code": exit_code,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }}, f, indent=2)

sys.exit(exit_code)
"""


def _spawn_background_process(wrapper_script: str, pid_path: Path) -> int:
    """Start background process and write PID file.

    Args:
        wrapper_script: Python script to execute.
        pid_path: Path to write PID file.

    Returns:
        Process ID of spawned process.
    """
    process = subprocess.Popen(
        [sys.executable, "-c", wrapper_script],
        start_new_session=True,  # Detach from parent
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_path.write_text(str(process.pid))
    return process.pid


def _update_task_manifest(
    task_id: str, started_at: str, issue: int | None, pid: int
) -> None:
    """Update manifest with new task entry.

    Args:
        task_id: Task identifier.
        started_at: ISO timestamp when task started.
        issue: Optional issue number.
        pid: Process ID.
    """
    with _manifest_lock():
        manifest = load_manifest()
        manifest.setdefault("tasks", {})
        manifest["tasks"][task_id] = {
            "status": "running",
            "started_at": started_at,
            "issue": issue,
            "pid": pid,
        }
        save_manifest(manifest)


def start_task(
    command: list[str],
    task_id: str,
    issue: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    description: str = "",
) -> TaskMeta:
    """Start a background task.

    REQUIRES: command is non-empty list of strings
    REQUIRES: task_id is non-empty string
    REQUIRES: timeout > 0
    REQUIRES: task_id is not already running (raises ValueError otherwise)
    ENSURES: return.status == "running"
    ENSURES: return.pid is valid process ID
    ENSURES: task directory exists with meta.json, pid file
    ENSURES: manifest.json updated with new task entry
    ENSURES: background process is detached from parent session
    """
    task_dir = get_task_dir(task_id)

    with _task_start_lock(task_dir):
        # Check if task already exists and is running
        existing = load_task_meta(task_id)
        if existing and existing.status == "running":
            if existing.pid and is_process_alive(existing.pid):
                raise ValueError(
                    f"Task {task_id} is already running (PID {existing.pid})"
                )

        # Create task directory and prepare paths
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / "output.log"
        result_path = task_dir / "result.json"
        pid_path = task_dir / "pid"

        # Get iteration from environment (AI_ITERATION set by looper.py)
        worker_iteration = None
        ai_iter = os.environ.get("AI_ITERATION", "")
        if ai_iter.isdigit():
            worker_iteration = int(ai_iter)

        # Create metadata
        meta = TaskMeta(
            task_id=task_id,
            command=" ".join(command),
            description=description or f"Background task: {task_id}",
            issue=issue,
            timeout=timeout,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
            worker_iteration=worker_iteration,
        )

        # Generate and run wrapper script
        wrapper_script = _generate_wrapper_script(
            command, timeout, output_path, result_path
        )
        meta.pid = _spawn_background_process(wrapper_script, pid_path)

        # Persist state
        save_task_meta(meta)
        _update_task_manifest(task_id, meta.started_at, issue, meta.pid)

        return meta


def kill_task(task_id: str) -> TaskMeta | None:
    """Kill a running task.

    REQUIRES: task_id is non-empty string
    ENSURES: return is None if task doesn't exist
    ENSURES: if task was running, return.status == "killed"
    ENSURES: if task was running, process is terminated (SIGTERM then SIGKILL)
    ENSURES: meta.json and manifest.json updated with "killed" status
    ENSURES: Never raises on kill failure - gracefully handles dead processes
    """
    meta = load_task_meta(task_id)
    if meta is None:
        return None

    if meta.status != "running" or meta.pid is None:
        return meta

    if is_process_alive(meta.pid):
        try:
            os.kill(meta.pid, signal.SIGTERM)
            time.sleep(1)
            if is_process_alive(meta.pid):
                os.kill(meta.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    meta.status = "killed"
    meta.finished_at = datetime.now(UTC).isoformat()
    save_task_meta(meta)

    # Update manifest
    with _manifest_lock():
        manifest = load_manifest()
        tasks = manifest.get("tasks", {})
        if task_id in tasks:
            tasks[task_id]["status"] = "killed"
            save_manifest(manifest)

    return meta


def cleanup_tasks(days: int = 7, force: bool = False) -> list[str]:
    """Remove old completed tasks.

    REQUIRES: days >= 0
    ENSURES: running tasks are preserved unless force=True
    ENSURES: tasks older than cutoff are removed
    ENSURES: manifest.json updated to remove deleted task entries
    ENSURES: return contains list of removed task IDs
    """
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")

    tasks_dir = get_tasks_dir()
    removed = []
    cutoff = time.time() - (days * 24 * 60 * 60)

    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue

        meta_path = task_dir / "meta.json"
        if not meta_path.exists():
            continue

        meta = load_task_meta(task_dir.name)
        if meta is None:
            continue

        # Don't remove running tasks
        if meta.status == "running" and not force:
            continue

        # Check age
        try:
            started = datetime.fromisoformat(meta.started_at)
            if started.timestamp() < cutoff or force:
                shutil.rmtree(task_dir)
                removed.append(task_dir.name)
        except (ValueError, OSError):
            pass

    # Update manifest
    if removed:
        with _manifest_lock():
            manifest = load_manifest()
            tasks = manifest.get("tasks", {})
            for task_id in removed:
                tasks.pop(task_id, None)
            save_manifest(manifest)

    return removed

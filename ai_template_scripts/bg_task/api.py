# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Public API for bg_task package.

Contains:
- list_tasks: List all tasks
- get_status: Get task status
- tail_output: Get task output
- wait_for_task: Wait for completion
"""

from __future__ import annotations

__all__ = [
    "list_tasks",
    "get_status",
    "tail_output",
    "wait_for_task",
]

import time

from ai_template_scripts.bg_task.process import update_task_status
from ai_template_scripts.bg_task.storage import get_task_dir, get_tasks_dir
from ai_template_scripts.bg_task.types import TaskMeta


def list_tasks(show_all: bool = False) -> list[TaskMeta]:
    """List all tasks, updating their status.

    REQUIRES: CWD is inside a git repository
    ENSURES: all returned tasks have up-to-date status
    ENSURES: if show_all is False, only running tasks are returned
    ENSURES: return list is sorted by started_at descending (newest first)
    """
    tasks_dir = get_tasks_dir()
    tasks = []

    for task_dir in tasks_dir.iterdir():
        if task_dir.is_dir() and (task_dir / "meta.json").exists():
            meta = update_task_status(task_dir.name)
            if meta:
                if show_all or meta.status == "running":
                    tasks.append(meta)

    return sorted(tasks, key=lambda t: t.started_at, reverse=True)


def get_status(task_id: str) -> TaskMeta | None:
    """Get status of a specific task.

    REQUIRES: task_id is non-empty string
    ENSURES: return is None if task doesn't exist
    ENSURES: return.status is up-to-date (process liveness checked)
    """
    return update_task_status(task_id)


def tail_output(task_id: str, lines: int = 50) -> str:
    """Get the last N lines of task output.

    REQUIRES: task_id is non-empty string
    REQUIRES: lines > 0
    ENSURES: return contains at most lines newlines
    ENSURES: returns "(no output yet)" if output.log doesn't exist
    """
    if lines <= 0:
        raise ValueError(f"lines must be > 0, got {lines}")

    output_path = get_task_dir(task_id) / "output.log"
    if not output_path.exists():
        return "(no output yet)"

    content = output_path.read_text()
    all_lines = content.splitlines()
    return "\n".join(all_lines[-lines:])


def wait_for_task(task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskMeta:
    """Wait for a task to complete.

    REQUIRES: task_id is non-empty string
    REQUIRES: timeout > 0
    REQUIRES: poll_interval > 0
    ENSURES: return.status != "running" (task has finished)
    ENSURES: raises ValueError if task doesn't exist
    ENSURES: raises TimeoutError if timeout exceeded before completion
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")

    start_time = time.time()

    while True:
        meta = update_task_status(task_id)
        if meta is None:
            raise ValueError(f"Task {task_id} not found")

        if meta.status != "running":
            return meta

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise TimeoutError(f"Timeout waiting for task {task_id} after {timeout}s")

        time.sleep(poll_interval)

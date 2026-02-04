# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Type definitions and constants for bg_task package.

Contains:
- TaskMeta dataclass for task metadata
- Constants: TASKS_DIR, DEFAULT_TIMEOUT
"""

from __future__ import annotations

__all__ = [
    "TaskMeta",
    "TASKS_DIR",
    "DEFAULT_TIMEOUT",
]

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# Directory for background tasks (relative to git root)
TASKS_DIR = ".background_tasks"

# Default timeout: 2 hours
DEFAULT_TIMEOUT = 7200


@dataclass
class TaskMeta:
    """Metadata for a background task.

    INVARIANT: status in {"running", "completed", "failed", "timeout", "killed"}
    """

    task_id: str
    command: str
    description: str
    issue: int | None
    timeout: int
    status: str  # "running", "completed", "failed", "timeout", "killed"
    started_at: str
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    machine: str = field(default_factory=lambda: os.uname().nodename)
    worker_iteration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert TaskMeta to dictionary.

        ENSURES: return contains all TaskMeta fields as keys
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMeta:
        """Create TaskMeta from dictionary.

        REQUIRES: data contains all required TaskMeta fields
        ENSURES: return is valid TaskMeta instance
        ENSURES: extra keys in data are ignored (forward compatibility)
        """
        # Filter to only known fields for forward compatibility
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

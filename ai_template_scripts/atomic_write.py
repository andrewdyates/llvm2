# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Atomic file write utilities.

Provides shared atomic write functions used across all modules that
write JSON or text files. Uses unique temp files and os.replace() for
crash-safe writes in multi-worker environments.

Consolidates duplicate implementations from:
- looper/telemetry.py (#2851)
- ai_template_scripts/bg_task/storage.py
- ai_template_scripts/tracker_utils.py (#2855)

See: #2862
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to path atomically using temp+rename.

    Uses a unique temp file to avoid collisions between concurrent writes
    in the same process and across processes.

    Args:
        path: Target file path.
        data: JSON-serializable dict to write.
    """
    _atomic_write(path, json.dumps(data, indent=2) + "\n")


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to path atomically using temp+rename.

    Uses a unique temp file to avoid collisions between concurrent writes
    in the same process and across processes.

    Args:
        path: Target file path.
        content: Text content to write.
    """
    _atomic_write(path, content)


def _fsync_directory(dir_path: Path) -> None:
    """Fsync a directory to persist rename metadata (#2911).

    On POSIX systems, rename durability requires syncing the containing
    directory. Without this, a power loss after rename could lose the
    directory entry even though the file contents were fsynced.

    Best-effort: silently ignores OSError (e.g., Windows, permission
    denied). The file data is already durable from the file fsync;
    directory fsync adds crash-durability for the rename itself.
    """
    try:
        dir_fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _atomic_write(path: Path, content: str) -> None:
    """Write file content using temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.{os.getpid()}.",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)

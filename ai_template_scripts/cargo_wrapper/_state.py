# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shared mutable state for cargo_wrapper package.

This module centralizes all mutable globals to enable test fixtures that
rebind module-level state like `cargo_wrapper.LOCK_DIR = tmp_path`.

All state is accessed through this module by other submodules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Lock directory paths - initialized lazily to handle missing HOME
LOCK_DIR: Path | None = None
LOCK_FILE: Path | None = None
LOCK_META: Path | None = None
BUILDS_LOG: Path | None = None
ORPHANS_LOG: Path | None = None

# Current lock kind (build or test)
LOCK_KIND: str = "build"

# Timeout configuration (cached)
TIMEOUT_CONFIG: dict[str, int] | None = None
# Limits configuration (cached)
LIMITS_CONFIG: dict[str, int] | None = None

# Global state for signal handler
_lock_held: bool = False
_child_process: Any = None  # subprocess.Popen or None
_child_pgid: int | None = None

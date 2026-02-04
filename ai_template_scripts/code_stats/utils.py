# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Utility functions for code complexity analysis."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ai_template_scripts.code_stats.constants import SKIP_DIRS
from ai_template_scripts.path_utils import resolve_path_alias
from ai_template_scripts.subprocess_utils import run_cmd


def get_git_info(dir_path: Path | None = None, **kwargs: Any) -> tuple[str, str]:
    """Get project name and commit hash.

    Args:
        dir_path: Directory path (should be git repo root or subdirectory).
        **kwargs: Accepts deprecated 'root' alias for dir_path.

    Returns:
        Tuple of (project_name, commit_hash).
    """
    resolved_path = resolve_path_alias(
        "dir_path", "root", dir_path, kwargs, "get_git_info"
    )
    project = resolved_path.resolve().name

    result = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=resolved_path)
    commit = result.stdout.strip() if result.ok else "unknown"

    return project, commit


def find_files(
    dir_path: Path | None = None,
    extensions: list[str] | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Find files with given extensions, excluding skip dirs.

    Args:
        dir_path: Directory to search in.
        extensions: List of file extensions to match (e.g., ['.py', '.rs']).
        **kwargs: Accepts deprecated 'root' alias for dir_path.

    Returns:
        List of matching file paths.
    """
    resolved_path = resolve_path_alias(
        "dir_path", "root", dir_path, kwargs, "find_files"
    )
    if extensions is None:
        extensions = []
    files = []
    for path in resolved_path.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return files


def has_tool(name: str) -> bool:
    """Check if a tool is available."""
    return shutil.which(name) is not None


__all__ = ["get_git_info", "find_files", "has_tool"]

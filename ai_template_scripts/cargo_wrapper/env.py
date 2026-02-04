# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Repository and environment context helpers for cargo_wrapper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Import get_repo_name from existing module
try:
    from ai_template_scripts.subprocess_utils import get_repo_name
    from ai_template_scripts.version import get_version
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from ai_template_scripts.subprocess_utils import get_repo_name
    from ai_template_scripts.version import get_version

__all__ = ["get_repo_identifier", "get_env_context", "get_git_commit", "get_version"]


def get_repo_identifier() -> str:
    """Get a unique identifier for the current git repository.

    Returns repo name from git remote or directory name as fallback.
    Used to create per-repo lock directories.

    Delegates to subprocess_utils.get_repo_name() for consistency.
    """
    result = get_repo_name()
    return result.stdout


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""  # Best-effort: git HEAD for lock metadata


def get_env_context() -> dict:
    """Get AI context from environment variables."""
    return {
        "project": os.environ.get("AI_PROJECT", os.path.basename(os.getcwd())),
        "role": os.environ.get("AI_ROLE", "USER"),
        "session": os.environ.get("AI_SESSION", "")[:8],
        "iteration": os.environ.get("AI_ITERATION", ""),
        "commit": get_git_commit(),
    }

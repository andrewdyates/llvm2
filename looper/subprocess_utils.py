# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Subprocess utilities for looper.

Wraps ai_template_scripts.subprocess_utils with Result[T] API for looper.
Provides standardized subprocess execution with consistent error handling.

This module is a wrapper around ai_template_scripts.subprocess_utils to:
- Maintain looper's Result[T] monadic error handling pattern
- Provide specialized run_git_command() and run_gh_command() helpers
- Re-export run_cmd_with_retry for looper callers needing retry logic
"""

from __future__ import annotations

__all__ = [
    "run_cmd",
    "run_git_command",
    "run_gh_command",
    "run_cmd_with_retry",
    "is_local_mode",
    "is_full_local_mode",
    "set_local_mode_from_config",
]

import os
import subprocess
from pathlib import Path

from ai_template_scripts.subprocess_utils import CmdResult, run_cmd_with_retry
from ai_template_scripts.subprocess_utils import run_cmd as _run_cmd_raw
from looper.result import Result

_LOCAL_MODE_FROM_CONFIG: bool | None = None


def set_local_mode_from_config(value: bool | None) -> None:
    """Set local mode based on parsed config (frontmatter/.looper_config.json)."""
    global _LOCAL_MODE_FROM_CONFIG
    if isinstance(value, bool):
        _LOCAL_MODE_FROM_CONFIG = value
    else:
        _LOCAL_MODE_FROM_CONFIG = None


def is_local_mode() -> bool:
    """Check if local mode is enabled (skip GitHub API calls).

    Local mode can be enabled via:
    - Environment variable: AIT_LOCAL_MODE=1 or AIT_LOCAL_MODE=full
    - Touch file: .local_mode in repo root
    - Config: local_mode in role frontmatter or .looper_config.json

    Returns:
        True if local mode is enabled (any level), False otherwise.
    """
    # Environment variable takes priority (easy toggle)
    local_mode_env = os.environ.get("AIT_LOCAL_MODE", "")
    if local_mode_env in ("1", "full"):
        return True
    # Touch file for persistent local mode
    if Path(".local_mode").exists():
        return True
    # Config-driven local mode (frontmatter/.looper_config.json)
    if _LOCAL_MODE_FROM_CONFIG is not None:
        return _LOCAL_MODE_FROM_CONFIG
    return False


def is_full_local_mode() -> bool:
    """Check if full local mode is enabled (use local issue storage).

    Full local mode (AIT_LOCAL_MODE=full) uses LocalIssueStore for all
    issue operations instead of GitHub API. Regular local mode (AIT_LOCAL_MODE=1)
    just returns stale cache/placeholder responses.

    Returns:
        True if AIT_LOCAL_MODE=full, False otherwise.
    """
    return os.environ.get("AIT_LOCAL_MODE") == "full"


def run_cmd(
    args: list[str],
    timeout: int = 30,
    cwd: Path | None = None,
) -> Result[subprocess.CompletedProcess[str]]:
    """Run a subprocess command with standard error handling.

    Wraps ai_template_scripts.subprocess_utils.run_cmd with Result[T] API.

    Contracts:
        REQUIRES: args is a non-empty list of strings
        ENSURES: Returns Result.success with CompletedProcess on execution
        ENSURES: Returns Result.failure with error on exception
    """
    if not isinstance(args, list) or not args:
        return Result.failure("run_cmd: args must be non-empty list")

    try:
        raw: CmdResult = _run_cmd_raw(args, timeout=timeout, cwd=cwd)
    except Exception as exc:
        return Result.failure(str(exc))

    if not raw.ok:
        return Result.failure(raw.error or f"command failed with code {raw.returncode}")

    proc = subprocess.CompletedProcess(
        args=args,
        returncode=raw.returncode,
        stdout=raw.stdout,
        stderr=raw.stderr,
    )
    return Result.success(proc)


def run_git_command(
    args: list[str],
    timeout: int = 10,
    cwd: Path | None = None,
) -> Result[str]:
    """Run git command with standardized error handling.

    Contracts:
        REQUIRES: args is a non-empty list of git arguments (without "git")
        ENSURES: Returns Result.success with stdout on success
        ENSURES: Returns Result.failure with an error string on failure
        ENSURES: For non-zero exit, error is stderr if present, else "unknown error"
        ENSURES: For non-zero exit, stdout may be returned in Result.value
    """
    if not isinstance(args, list) or not args:
        return Result.failure("run_git_command: args must be non-empty list")
    result = run_cmd(["git", *args], timeout=timeout, cwd=cwd)
    if not result.ok:
        return Result.failure(result.error or "git command failed")
    proc = result.value
    if proc is None:
        return Result.failure("git command returned no result")
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        error = stderr or "unknown error"
        return Result.failure(error, value=proc.stdout or "")
    return Result.success(proc.stdout or "")


def run_gh_command(
    args: list[str],
    timeout: int = 15,
    cwd: Path | None = None,
) -> Result[str]:
    """Run gh CLI command with standardized error handling.

    Contracts:
        REQUIRES: args is a non-empty list of gh arguments (without "gh")
        ENSURES: Returns Result.success with stdout on success
        ENSURES: Returns Result.failure with an error string on failure
        ENSURES: For non-zero exit, error is stderr if present, else "unknown error"
        ENSURES: For non-zero exit, stdout may be returned in Result.value
        ENSURES: In local mode (#1592), returns failure with placeholder message
    """
    if not isinstance(args, list) or not args:
        return Result.failure("run_gh_command: args must be non-empty list")

    # Local mode: skip GitHub API calls to conserve quota (#1592)
    if is_local_mode():
        return Result.failure(
            "local_mode: GitHub API calls disabled", value="(local mode)"
        )

    result = run_cmd(["gh", *args], timeout=timeout, cwd=cwd)
    if not result.ok:
        return Result.failure(result.error or "gh command failed")
    proc = result.value
    if proc is None:
        return Result.failure("gh command returned no result")
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        error = stderr or "unknown error"
        return Result.failure(error, value=proc.stdout or "")
    return Result.success(proc.stdout or "")

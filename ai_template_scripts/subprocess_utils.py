# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Subprocess utilities for ai_template_scripts.

Provides unified run_cmd with consistent error handling, plus
canonical git repository identification functions.

Public API (library usage):
    from ai_template_scripts.subprocess_utils import (
        CmdResult,             # Structured result wrapper
        run_cmd,               # Run command with timeout
        run_cmd_with_retry,    # Retry wrapper for transient failures
        get_repo_name,         # Get repo name from git remote (local use)
        get_github_repo,       # Get owner/repo from gh CLI (GitHub API use)
    )
"""

from __future__ import annotations

__all__ = [
    "CmdResult",
    "run_cmd",
    "run_cmd_with_retry",
    "get_repo_name",
    "get_github_repo",
]

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CmdResult:
    """Result of subprocess execution.

    Attributes:
        returncode: Exit code from the process (0 = success)
        stdout: Captured standard output
        stderr: Captured standard error
        error: Exception message if an error occurred, None otherwise
    """

    returncode: int
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if command succeeded (returncode 0 and no exception)."""
        return self.returncode == 0 and self.error is None


def run_cmd(
    cmd: list[str],
    timeout: int = 30,
    cwd: Path | None = None,
) -> CmdResult:
    """Run command and return structured result.

    REQUIRES: cmd is non-empty list of strings
    ENSURES: Returns CmdResult with stdout/stderr captured

    Args:
        cmd: Command and arguments as list
        timeout: Maximum seconds to wait (default 30)
        cwd: Working directory for command

    Returns:
        CmdResult with returncode, stdout, stderr, and optional error
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return CmdResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired as e:
        stdout = ""
        stderr = ""
        if hasattr(e, "stdout") and e.stdout:
            stdout = e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
        if hasattr(e, "stderr") and e.stderr:
            stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
        return CmdResult(
            returncode=124,  # Standard timeout exit code
            stdout=stdout,
            stderr=stderr,
            error=f"timeout after {timeout}s",
        )
    except FileNotFoundError:
        return CmdResult(
            returncode=127,  # Standard "command not found" code
            stdout="",
            stderr="",
            error=f"command not found: {cmd[0] if cmd else 'empty'}",
        )
    except subprocess.SubprocessError as e:
        return CmdResult(
            returncode=1,
            stdout="",
            stderr="",
            error=str(e),
        )


def run_cmd_with_retry(
    cmd: list[str],
    timeout: int = 60,
    retries: int = 2,
    retry_delay: float = 1.0,
    cwd: Path | None = None,
) -> CmdResult:
    """Run command with retry logic for transient failures.

    Used for network-dependent commands like `gh` where transient failures
    are common.

    REQUIRES: retries >= 0
    ENSURES: Returns last CmdResult after exhausting retries

    Args:
        cmd: Command and arguments as list
        timeout: Maximum seconds per attempt (default 60)
        retries: Number of retry attempts (default 2, so 3 total attempts)
        retry_delay: Seconds between attempts (default 1.0)
        cwd: Working directory for command

    Returns:
        CmdResult from successful attempt or last failed attempt
    """
    last_result: CmdResult | None = None
    for attempt in range(retries + 1):
        result = run_cmd(cmd, timeout=timeout, cwd=cwd)
        if result.ok:
            return result
        last_result = result
        if attempt < retries:
            time.sleep(retry_delay)
    return last_result or CmdResult(1, "", "", "no attempts made")


def _sanitize_repo_name(name: str) -> str:
    """Sanitize repo name for filesystem use.

    Prevents path traversal attacks and removes dangerous characters.
    """
    # Replace path separators with underscore (prevents traversal)
    name = name.replace("/", "_").replace("\\", "_")
    # Replace parent directory references (prevents traversal)
    name = name.replace("..", "_")
    # Keep only safe characters
    clean = "".join(c for c in name if c.isalnum() or c in "-_.")
    # Strip leading/trailing dots and spaces (prevents hidden files, whitespace issues)
    clean = clean.strip(". ")
    if not clean:
        clean = "unknown"
    return clean[:64]


def _extract_repo_name_from_url(url: str) -> str:
    """Extract repository name from git URL.

    Handles both SSH and HTTPS formats:
    - git@github.com:owner/repo.git -> repo
    - https://github.com/owner/repo.git -> repo
    """
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return name.removesuffix(".git")


def get_repo_name(cwd: Path | None = None) -> CmdResult:
    """Get repository name from git remote.

    Canonical function for identifying the current repository by name.
    Use this for local operations (locking, logging, identification).

    REQUIRES: Called from within a git repository (or cwd is in one)
    ENSURES: Returns CmdResult with repo name in stdout on success
    ENSURES: Falls back to directory name if git remote fails

    Fallback chain:
    1. git remote get-url origin -> extract repo name
    2. git rev-parse --show-toplevel -> directory name
    3. Current directory name

    Args:
        cwd: Working directory (default: current directory)

    Returns:
        CmdResult with repo name in stdout (always succeeds with fallback)

    Example:
        result = get_repo_name()
        repo = result.stdout  # e.g., "ai_template"
    """
    # Try git remote first (most reliable across worktrees/clones)
    result = run_cmd(["git", "remote", "get-url", "origin"], timeout=5, cwd=cwd)
    if result.ok:
        name = _extract_repo_name_from_url(result.stdout.strip())
        if name:
            return CmdResult(
                returncode=0,
                stdout=_sanitize_repo_name(name),
                stderr="",
            )

    # Fallback: git root directory name
    result = run_cmd(["git", "rev-parse", "--show-toplevel"], timeout=5, cwd=cwd)
    if result.ok:
        name = Path(result.stdout.strip()).name
        return CmdResult(
            returncode=0,
            stdout=_sanitize_repo_name(name),
            stderr="",
        )

    # Last resort: current directory name
    fallback_cwd = cwd or Path.cwd()
    return CmdResult(
        returncode=0,
        stdout=_sanitize_repo_name(fallback_cwd.name),
        stderr="",
    )


def get_github_repo(gh_path: str = "gh", cwd: Path | None = None) -> CmdResult:
    """Get GitHub repository in owner/name format.

    Canonical function for GitHub API operations. Uses gh CLI to get
    the authoritative repo name with owner.

    REQUIRES: gh CLI is installed and authenticated
    REQUIRES: Called from within a GitHub repository
    ENSURES: Returns CmdResult with owner/repo in stdout on success

    NOTE: Does NOT use gh -q or --jq flags due to caching bugs in
    gh v2.83.2+ (see #1047).

    Args:
        gh_path: Path to gh CLI (default: "gh")
        cwd: Working directory (default: current directory)

    Returns:
        CmdResult with owner/repo in stdout on success,
        error details on failure

    Example:
        result = get_github_repo()
        if result.ok:
            repo = result.stdout  # e.g., "ayates_dbx/ai_template"
        else:
            print(f"Not in GitHub repo: {result.error}")
    """
    # NOTE: Do NOT use gh -q or --jq - has caching bugs in v2.83.2+ (#1047)
    result = run_cmd(
        [gh_path, "repo", "view", "--json", "nameWithOwner"],
        timeout=30,
        cwd=cwd,
    )
    if not result.ok:
        return CmdResult(
            returncode=result.returncode,
            stdout="",
            stderr=result.stderr,
            error=result.error or "Not in a GitHub repo",
        )

    # Parse JSON response
    try:
        data = json.loads(result.stdout.strip())
        repo = data.get("nameWithOwner", "")
        if repo:
            return CmdResult(returncode=0, stdout=repo, stderr="")
        return CmdResult(
            returncode=1,
            stdout="",
            stderr="",
            error="No nameWithOwner in response",
        )
    except json.JSONDecodeError as e:
        return CmdResult(
            returncode=1,
            stdout="",
            stderr=result.stdout,
            error=f"Invalid JSON response: {e}",
        )

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Subprocess utilities for ai_template_scripts.

Provides unified run_cmd with consistent error handling, plus
canonical git repository identification functions.

Public API (library usage):
    from ai_template_scripts.subprocess_utils import (
        CmdResult,                # Structured result wrapper
        run_cmd,                  # Run command with timeout
        run_cmd_with_retry,       # Retry wrapper for transient failures
        get_repo_name,            # Get repo name from git remote (local use)
        get_github_repo,          # Get owner/repo from gh CLI (GitHub API use)
        is_process_alive,         # Check if PID is alive (consolidated from 13+ locations)
        get_git_root,             # Get git repository root path (consolidated from 9+ locations)
        format_duration_precise,  # Compound format: 1h02m05s (diagnostics)
        format_duration_compact,  # Abbreviated format: 2.0h (task lists)
    )
"""

from __future__ import annotations

__all__ = [
    "CmdResult",
    "run_cmd",
    "run_cmd_with_retry",
    "get_repo_name",
    "get_github_repo",
    "is_process_alive",
    "get_git_root",
    "format_duration_precise",
    "format_duration_compact",
]

import json
import os
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
            repo = result.stdout  # e.g., "dropbox-ai-prototypes/ai_template"
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


def is_process_alive(pid: int) -> bool:
    """Check if process with given PID is alive.

    Canonical implementation consolidated from 13+ locations across the codebase.
    Uses os.kill(pid, 0) which sends no signal but checks process existence.

    REQUIRES: pid is a valid integer (typically > 0)
    ENSURES: Returns True if process exists (or permission denied)
    ENSURES: Returns False if process does not exist
    ENSURES: Never raises - returns False on any error

    Args:
        pid: Process ID to check

    Returns:
        True if process is alive, False otherwise

    Example:
        if is_process_alive(worker_pid):
            print("Worker still running")
        else:
            print("Worker terminated")
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # Process doesn't exist
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    except (OSError, TypeError, ValueError):
        # Invalid PID or other OS error
        return False


def get_git_root(cwd: Path | None = None) -> Path | None:
    """Get the git repository root directory.

    Canonical implementation consolidated from 9+ locations across the codebase.
    Returns None on failure instead of raising, for consistent error handling.

    REQUIRES: Optional cwd path or current directory
    ENSURES: Returns Path to repo root if in a git repository
    ENSURES: Returns None if not in a git repository or on error
    ENSURES: Never raises - returns None on any error

    Args:
        cwd: Working directory to check (default: current directory)

    Returns:
        Path to git repository root, or None if not in a repository

    Example:
        root = get_git_root()
        if root:
            config_path = root / ".looper_config.json"
        else:
            print("Not in a git repository")

    Note:
        For functions that need to raise on failure, use:
            root = get_git_root()
            if root is None:
                raise RuntimeError("Not in a git repository")
    """
    result = run_cmd(["git", "rev-parse", "--show-toplevel"], timeout=10, cwd=cwd)
    if result.ok:
        return Path(result.stdout.strip())
    return None


def format_duration_precise(seconds: int) -> str:
    """Format seconds as compound duration (e.g. 1h02m05s).

    Consolidated from cargo_lock_info.py (#2535). Use for diagnostics and lock
    age display where exact breakdown matters.

    REQUIRES: seconds is numeric (will be converted to int)
    ENSURES: returns non-empty string in format [Dd][HHh][MMm]SSs
    """
    seconds = max(0, int(seconds))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d{hrs:02}h{mins:02}m{sec:02}s"
    if hrs:
        return f"{hrs}h{mins:02}m{sec:02}s"
    if mins:
        return f"{mins}m{sec:02}s"
    return f"{sec}s"


def format_duration_compact(seconds: float) -> str:
    """Format seconds as abbreviated duration (e.g. 2.0h).

    Consolidated from bg_task/cli.py (#2535). Use for brief task list display
    where a single-unit approximation is sufficient.

    REQUIRES: seconds >= 0
    ENSURES: return ends with 's', 'm', or 'h' suffix
    """
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"

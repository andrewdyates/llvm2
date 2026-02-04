# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_git.py - Git identity and environment setup.

Provides RunnerGitMixin for LoopRunner:
- Git author/committer identity configuration (GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL, etc.)
- AI tool version detection (Claude Code, Codex, Dasher)
- ai_template sync staleness checks
- Session ID generation for commit tracking
- Environment variable setup for AI sessions

Constants:
    HAS_FCNTL: True if fcntl module available (Unix file locking support)

See docs/looper.md "Environment Variables" for the full list of session variables.
"""

from __future__ import annotations

__all__ = ["RunnerGitMixin", "HAS_FCNTL"]

import os
import re
import socket
import time
from pathlib import Path
from typing import TextIO

# fcntl is Unix-only; Windows uses different locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from looper.config import LOG_DIR, get_project_name
from looper.log import debug_swallow, log_info, log_warning
from looper.subprocess_utils import run_cmd, run_git_command


class RunnerGitMixin:
    """Git operations and identity setup.

    Required attributes:
        config: dict
        mode: str
        _session_id: str
        worker_id: int | None
        machine: str | None
    """

    def _get_max_git_iteration(self) -> int:
        """Get max iteration from git log by parsing [X]N commits for current role.

        Searches recent commits to find the highest iteration number used.
        Handles [W]N, [W1]N, and [sat-W1]N formats (optional machine prefix).

        Returns:
            Maximum iteration number found, or 0 if none found.
        """
        try:
            # Get role prefix (W, M, P, R) from config
            role_prefix = self.config["git_author_name"][0].upper()
            # Search recent commits, find ALL [X]N and [XN]N patterns for this role
            # Pattern: [W]42, [W1]42, [sat-W2]15 - all share iteration space
            # Limit to 500 commits for performance (iterations are in recent history)
            result = run_git_command(
                ["log", "--oneline", "--all", "-n", "500"], timeout=30
            )
            if result.ok and result.value:
                # Match [W]42, [W1]42, [sat-W2]42 - captures the iteration number
                pattern = rf"\[(?:[^\]]+-)?{role_prefix}\d*\]#?(\d+)"
                max_iteration = 0
                for line in result.value.split("\n"):
                    match = re.search(pattern, line)
                    if match:
                        max_iteration = max(max_iteration, int(match.group(1)))
                return max_iteration
        except Exception as e:
            debug_swallow("get_max_git_iteration", e)
        return 0

    def _read_commit_tag_file(self, commit_tag_file: Path) -> int:
        """Read current value from commit tag file, returns 0 on error."""
        if commit_tag_file.exists():
            try:
                return int(commit_tag_file.read_text().strip())
            except (ValueError, OSError) as e:
                debug_swallow("read_commit_tag_file", e)
        return 0

    def _write_commit_tag_file(self, commit_tag_file: Path, value: int) -> bool:
        """Write value to commit tag file atomically. Returns True on success."""
        try:
            # Use PID in temp file name to avoid collisions between processes
            tmp_file = commit_tag_file.with_suffix(f".tmp.{os.getpid()}")
            tmp_file.write_text(str(value))
            tmp_file.rename(commit_tag_file)
            return True
        except OSError as e:
            debug_swallow("write_commit_tag_file", e)
            return False

    def _acquire_lock_with_timeout(
        self, lock_file: TextIO, timeout_sec: float = 30.0
    ) -> bool:
        """Try to acquire exclusive lock with timeout. Returns True if acquired.

        REQUIRES: lock_file is an open file object with valid fileno()
        REQUIRES: HAS_FCNTL is True (Unix platform with fcntl available)
        ENSURES: If returns True: lock_file has exclusive flock held
        ENSURES: If returns False: no lock (timeout/no fcntl/OSError)
        ENSURES: Returns within timeout_sec + small overhead (polling interval 0.1s)
        ENSURES: Does not modify lock_file open/close state

        TLA+ correspondence: iteration_tags.tla:TryAcquireLock(s),
        iteration_tags.tla:FailAcquireLock(s),
        iteration_tags.tla:RetryAcquireLock(s),
        iteration_tags.tla:TimeoutAcquiring(s)
        TLA+ spec: tla/iteration_tags.tla
        NOTE: HAS_FCNTL False and OSError failure paths return False but are not modeled.
        """
        if not HAS_FCNTL:
            return False

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                # Non-blocking lock attempt
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                # Lock held by another process, retry after brief sleep
                time.sleep(0.1)
            except OSError as e:
                # Other error (e.g., NFS issues), give up on locking
                debug_swallow("acquire_lock_with_timeout", e)
                return False
        return False

    def get_git_iteration(self) -> int:
        """Get next commit tag iteration with file lock coordination.

        Uses file locking to prevent race conditions when multiple sessions
        start concurrently. Coordinates between:
        - Git history (authoritative record of used iterations)
        - Commit tag file (fast local cache, separate from loop counter)

        The lock ensures only one session can read+increment at a time,
        preventing duplicate iteration numbers like [R]267 appearing twice.

        Note: This is separate from self.iteration (loop counter) which tracks
        how many times THIS session has looped. The commit tag iteration is
        unique across ALL sessions of this role.

        Platform notes:
        - Unix: Uses fcntl.flock() with timeout for coordination
        - Windows: Falls back to file-based coordination (less robust)

        REQUIRES: self.mode is set (e.g., 'worker', 'manager')
        REQUIRES: LOG_DIR exists or can be created
        ENSURES: Returns positive integer > any previously returned value for this role
        ENSURES: Uniqueness - concurrent calls return distinct values (via file lock)
        ENSURES: Persists returned value to commit tag file for next session
        ENSURES: Releases lock before returning (or on exception)

        TLA+ correspondence: iteration_tags.tla:ComputeAndRelease(s)
        TLA+ spec: tla/iteration_tags.tla

        Returns:
            Next commit tag iteration (unique across concurrent sessions).

        Raises:
            RuntimeError: If lock cannot be acquired (prevents duplicate iterations).
        """
        # Ensure log directory exists for lock file
        LOG_DIR.mkdir(exist_ok=True)

        # Use separate file for commit tag coordination (not loop counter)
        commit_tag_file = LOG_DIR / f".commit_tag_{self.mode}"
        lock_file_path = LOG_DIR / f".commit_tag_lock_{self.mode}"

        # Try to acquire lock (Unix only, with timeout)
        lock_acquired = False
        lock_file = None

        if HAS_FCNTL:
            try:
                lock_file = open(lock_file_path, "w")
                lock_acquired = self._acquire_lock_with_timeout(
                    lock_file, timeout_sec=30.0
                )
                if not lock_acquired:
                    if lock_file:
                        lock_file.close()
                    raise RuntimeError(
                        "Could not acquire iteration lock within 30s. "
                        "Another session may be holding the lock."
                    )
            except OSError as e:
                if lock_file:
                    try:
                        lock_file.close()
                    except OSError as close_err:
                        debug_swallow("lock_file_close_on_error", close_err)
                raise RuntimeError(f"Could not open lock file: {e}") from e
        else:
            # Windows: no fcntl available, proceed without lock but warn
            # This is best-effort - concurrent sessions on Windows may collide
            log_warning(
                "Warning: fcntl not available (Windows?), proceeding without lock"
            )

        try:
            # Read current value from commit tag file
            file_iteration = self._read_commit_tag_file(commit_tag_file)

            # Get max from git history
            git_iteration = self._get_max_git_iteration()

            # Next iteration is max of both sources + 1
            next_iteration = max(file_iteration, git_iteration) + 1

            # Persist to commit tag file (atomic via rename with unique temp)
            if not self._write_commit_tag_file(commit_tag_file, next_iteration):
                log_warning("Warning: could not write commit tag file")

            return next_iteration

        finally:
            # Release lock if acquired
            if lock_acquired and lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError as e:
                    debug_swallow("unlock_flock", e)
            if lock_file:
                try:
                    lock_file.close()
                except OSError as e:
                    debug_swallow("lock_file_close", e)

    def setup_git_identity(self) -> None:
        """Set up rich git identity for commits.

        Format: {project}-{role}-{iteration} <{session}@{machine}.{project}.ai-fleet>
        This enables forensic tracking of which AI session made which commits.
        """
        project = get_project_name()
        role = self.config["git_author_name"].lower()  # "worker" or "manager"
        iteration = self.get_git_iteration()
        machine = socket.gethostname().split(".")[0]
        session = self._session_id

        git_name = f"{project}-{role}-{iteration}"
        git_email = f"{session}@{machine}.{project}.ai-fleet"

        os.environ["GIT_AUTHOR_NAME"] = git_name
        os.environ["GIT_AUTHOR_EMAIL"] = git_email
        os.environ["GIT_COMMITTER_NAME"] = git_name
        os.environ["GIT_COMMITTER_EMAIL"] = git_email

        # Export for MCP tools and other scripts
        os.environ["AI_PROJECT"] = project
        os.environ["AI_ROLE"] = role.upper()
        os.environ["AI_ITERATION"] = str(iteration)
        os.environ["AI_SESSION"] = session
        os.environ["AI_MACHINE"] = machine

        # Export worker ID for multi-worker mode (used by commit-msg hook)
        if self.worker_id is not None:
            os.environ["AI_WORKER_ID"] = str(self.worker_id)
        else:
            # Clear worker ID in case it was set by a previous session
            os.environ.pop("AI_WORKER_ID", None)

        # Export machine prefix for multi-machine mode (used by commit-msg hook)
        # Note: AI_MACHINE is the hostname; AI_MACHINE_PREFIX is the looper identity
        # Results in commit tags like [sat-W1]42 instead of [W1]42
        if self.machine is not None:
            os.environ["AI_MACHINE_PREFIX"] = self.machine
        else:
            os.environ.pop("AI_MACHINE_PREFIX", None)

        # Export AIT version for tracking (can be used by commit hook)
        ait_version = self.get_ait_version()
        if ait_version:
            os.environ["AIT_VERSION"] = ait_version[0]
            os.environ["AIT_SYNCED"] = ait_version[1]

        # Export coder info for commit signatures
        os.environ["AI_CODER"] = "claude-code"
        result = run_cmd(["claude", "--version"], timeout=5)
        if result.ok and result.value and result.value.returncode == 0:
            stdout = result.value.stdout or ""
            if stdout.strip():
                parts = stdout.strip().split()
                if parts:
                    # Format: "2.1.17 (Claude Code)" - first part is version
                    os.environ["CLAUDE_CODE_VERSION"] = parts[0]

        # Check pinned Claude version
        self._check_claude_version()

        # NOTE: PATH setup moved to runner.py:setup_wrapper_path() (#1690)
        # Called early in main() before ANY gh/cargo calls

        log_info(f"✓ Git identity: {git_name}")

    def get_ait_version(self) -> tuple[str, str] | None:
        """Get ai_template version info from .ai_template_version file.

        Returns:
            (commit_hash, sync_timestamp) or None if file doesn't exist.
        """
        version_file = Path(".ai_template_version")
        if not version_file.exists():
            return None
        try:
            lines = version_file.read_text().strip().split("\n")
            if len(lines) >= 2:
                return (lines[0][:8], lines[1])  # Short hash, timestamp
            if len(lines) == 1:
                return (lines[0][:8], "unknown")
        except Exception as e:
            debug_swallow("get_ait_version", e)
        return None

    def _check_claude_version(self) -> None:
        """Check if installed Claude CLI matches pinned version.

        Reads .claude-version from repo root and compares against installed
        version (from CLAUDE_CODE_VERSION env var). Logs warning if mismatch.
        """
        version_file = Path(".claude-version")
        if not version_file.exists():
            return

        try:
            pinned = version_file.read_text().strip().split("\n")[0].strip()
        except Exception as e:
            debug_swallow("read_claude_version_file", e)
            return

        if not pinned:
            return

        installed = os.environ.get("CLAUDE_CODE_VERSION", "")
        if not installed:
            log_warning(f"⚠ Claude version pinned to {pinned} but version unknown")
            return

        if installed != pinned:
            log_warning(
                f"⚠ Claude CLI version mismatch: installed={installed}, pinned={pinned}"
            )
            log_warning(
                f"  Fix with: npm install -g @anthropic-ai/claude-code@{pinned}"
            )

    def check_sync_staleness(self) -> int | None:
        """Check how many commits behind ai_template this repo is.

        Looks for ai_template as sibling directory. If found, compares
        local .ai_template_version against ai_template HEAD.

        Returns:
            Number of commits behind, or None if check not possible.
        """
        version_file = Path(".ai_template_version")
        if not version_file.exists():
            log_warning("⚠ No .ai_template_version - repo may never have been synced")
            return None

        # Read local version (first line only - ignore timestamp)
        try:
            local_version = version_file.read_text().strip().split("\n")[0].strip()
        except Exception as e:
            debug_swallow("read_ai_template_version_file", e)
            return None  # Best-effort: can't read version file, skip sync check

        if not local_version:
            return None

        # Check for sibling ai_template directory
        parent = Path.cwd().parent
        ait_dir = parent / "ai_template"
        if not ait_dir.is_dir() or not (ait_dir / ".git").is_dir():
            # ai_template not found as sibling - can't check
            return None

        # Get ai_template HEAD
        result = run_git_command(["rev-parse", "HEAD"], timeout=5, cwd=ait_dir)
        if not result.ok:
            return None  # Best-effort: git query failed, skip sync check
        ait_head = (result.value or "").strip()

        # If already current, no warning needed
        # Handle both full and short hashes: either could be a prefix of the other
        if (
            local_version == ait_head
            or ait_head.startswith(local_version)
            or local_version.startswith(ait_head)
        ):
            return 0

        # Count commits behind using ai_template's git history
        result = run_git_command(
            ["rev-list", "--count", f"{local_version}..{ait_head}"],
            timeout=10,
            cwd=ait_dir,
        )
        if result.ok and result.value:
            try:
                return int(result.value.strip())
            except ValueError as e:
                debug_swallow("check_sync_staleness_parse", e)

        return None

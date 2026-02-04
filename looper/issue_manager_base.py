# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shared subprocess helpers for issue management.

Extracted from issue_manager.py per designs/2026-02-01-issue-manager-split.md.

Part of #1808.
"""

import subprocess
from pathlib import Path
from typing import Any

from ai_template_scripts.subprocess_utils import run_cmd as _run_cmd_raw
from looper.result import Result
from looper.subprocess_utils import is_local_mode

__all__ = ["IssueManagerBase"]


class IssueManagerBase:
    """Base class with subprocess helpers for issue management.

    Provides shared functionality used by IssueAuditor, CheckboxConverter,
    and IssueManager.

    Contracts:
        REQUIRES: repo_path is a valid Path (exists at init time not required)
        REQUIRES: role is a non-empty string (stored as provided, not normalized)
        ENSURES: All methods handle subprocess failures gracefully
    """

    def __init__(self, repo_path: Path, role: str) -> None:
        """Initialize with repo path and role.

        Contracts:
            REQUIRES: repo_path is a Path object
            REQUIRES: role is a non-empty string
            ENSURES: self.repo_path is an absolute resolved path
            ENSURES: self.role is stored as provided
        """
        self.repo_path = repo_path.resolve()
        self.role = role

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Run a subprocess command anchored to the repo root.

        Contracts:
            REQUIRES: args is a non-empty list of strings
            ENSURES: cwd is set to self.repo_path
            ENSURES: Returns subprocess result (may have non-zero returncode)
            ENSURES: Default timeout of 60s (can be overridden via kwargs)
        """
        kwargs.setdefault("timeout", 60)
        return subprocess.run(args, cwd=self.repo_path, **kwargs)

    def _gh_run(
        self, args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Run a gh command, respecting local mode (#1592).

        In local mode, returns a fake failed result to skip GitHub API calls.

        Contracts:
            REQUIRES: args is a non-empty list of strings starting with "gh"
            ENSURES: In local mode, returns CompletedProcess with returncode=1
            ENSURES: In normal mode, delegates to _run()
        """
        if is_local_mode():
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="local_mode: GitHub API calls disabled",
            )
        return self._run(args, **kwargs)

    def _run_result(
        self, args: list[str], timeout: int = 30
    ) -> Result[subprocess.CompletedProcess[str]]:
        """Run a subprocess command with Result[T] error handling.

        Uses ai_template_scripts.subprocess_utils.run_cmd() internally for
        consistent exception handling. Returns Result[T] to match looper's
        standard error handling pattern.

        Contracts:
            REQUIRES: args is a non-empty list of strings
            ENSURES: Returns Result.success with CompletedProcess on execution
            ENSURES: Returns Result.failure with error message on exception
            ENSURES: cwd is set to self.repo_path

        Args:
            args: Command and arguments to execute.
            timeout: Maximum seconds to wait (default 30).

        Returns:
            Result wrapping CompletedProcess or error message.
        """
        if not args:
            return Result.failure("_run_result: args must be non-empty list")

        raw = _run_cmd_raw(args, timeout=timeout, cwd=self.repo_path)
        if raw.error:
            return Result.failure(raw.error)

        proc = subprocess.CompletedProcess(
            args=args,
            returncode=raw.returncode,
            stdout=raw.stdout,
            stderr=raw.stderr,
        )
        return Result.success(proc)

    def _gh_run_result(
        self, args: list[str], timeout: int = 15
    ) -> Result[subprocess.CompletedProcess[str]]:
        """Run a gh command with Result[T] error handling.

        Respects local mode (#1592) and uses ai_template_scripts.subprocess_utils
        for consistent exception handling.

        Contracts:
            REQUIRES: args is a non-empty list of strings starting with "gh"
            ENSURES: In local mode, returns Result.failure with placeholder message
            ENSURES: Returns Result.success with CompletedProcess on execution
            ENSURES: Returns Result.failure with error message on exception

        Args:
            args: Command and arguments (must start with "gh").
            timeout: Maximum seconds to wait (default 15).

        Returns:
            Result wrapping CompletedProcess or error message.
        """
        if is_local_mode():
            return Result.failure(
                "local_mode: GitHub API calls disabled",
                value=subprocess.CompletedProcess(
                    args=args, returncode=1, stdout="", stderr="local_mode"
                ),
            )
        return self._run_result(args, timeout=timeout)

    def _get_repo_name(self) -> str:
        """Get repo name for lock file naming.

        Contracts:
            REQUIRES: none (best-effort; gh/local mode failures are tolerated)
            ENSURES: Returns sanitized owner_repo ("/" replaced with "_") when gh
                returns a non-empty nameWithOwner
            ENSURES: Returns "unknown" on gh error or empty stdout

        Returns:
            Sanitized repo name (org_repo format) or "unknown" on error.
        """
        result = self._gh_run_result(
            [
                "gh",
                "repo",
                "view",
                "--json",
                "nameWithOwner",
                "-q",
                ".nameWithOwner",
            ],
            timeout=10,
        )
        if not result.ok:
            return "unknown"
        if result.value.returncode == 0 and result.value.stdout.strip():
            return result.value.stdout.strip().replace("/", "_")
        return "unknown"

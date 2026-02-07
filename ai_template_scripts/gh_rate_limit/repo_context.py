# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/repo_context.py - Repository Context Resolution

Owns repo name and owner/repo normalization for cache namespacing.
Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Methods extracted from RateLimiter:
- _get_repo, _normalize_repo, _get_owner_repo, _get_real_gh
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ai_template_scripts.shared_logging import debug_swallow


class RepoContext:
    """Repository context for cache namespacing and API calls.

    Resolves the current repository name and owner/repo for:
    - Cache key namespacing (prevent cross-repo data contamination)
    - REST API calls that need owner/repo
    """

    def __init__(self) -> None:
        self._real_gh: str | None = None
        self._repo: str | None = None

    def get_real_gh(self) -> str:
        """Find the real gh binary (not our wrapper)."""
        if self._real_gh:
            return self._real_gh
        for loc in ["/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh"]:
            if os.path.isfile(loc) and os.access(loc, os.X_OK):
                self._real_gh = loc
                return loc
        raise RuntimeError("gh CLI not found")

    def get_repo(self) -> str:
        """Get current repo identifier for cache namespacing.

        Returns the repo directory name (e.g., "ai_template") for cache keys.
        """
        if self._repo:
            return self._repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                self._repo = Path(result.stdout.strip()).name
                return self._repo
        except Exception:
            debug_swallow("get_current_repo")
        self._repo = "unknown"
        return self._repo

    def normalize_repo(self, repo: str) -> str | None:
        """Normalize repo input to owner/repo format.

        Handles various input formats:
        - "owner/repo"
        - "https://github.com/owner/repo"
        - "git@github.com:owner/repo.git"
        """
        repo = repo.strip().removesuffix(".git")
        if not repo:
            return None
        if "github.com:" in repo:
            parts = repo.split("github.com:")[-1].split("/")
        elif "github.com/" in repo:
            parts = repo.split("github.com/")[-1].split("/")
        else:
            parts = repo.split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return None

    def get_owner_repo(self, repo_override: str | None = None) -> str | None:
        """Get owner/repo from override or git origin.

        Args:
            repo_override: Explicit repo to use (from -R/--repo flag).

        Returns:
            owner/repo string or None if not resolvable.
        """
        if repo_override:
            normalized = self.normalize_repo(repo_override)
            if normalized:
                return normalized
        try:
            origin_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if origin_result.returncode == 0:
                normalized = self.normalize_repo(origin_result.stdout.strip())
                if normalized:
                    return normalized
        except Exception:
            debug_swallow("get_owner_repo")
            return None
        return None

    def get_commit_hash(self) -> str | None:
        """Get current HEAD commit hash (short form)."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            debug_swallow("get_commit_hash")
        return None

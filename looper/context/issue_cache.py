# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/issue_cache.py - Issue caching for API efficiency.

Provides IterationIssueCache for caching all open issues within a single
looper iteration, reducing GitHub API calls from 4-8 per iteration to 1.
"""

__all__ = [
    "IterationIssueCache",
    "is_feature_freeze",
]

import json
import sys
from pathlib import Path

from looper.config import load_project_config, load_timeout_config
from looper.context.helpers import has_label
from looper.result import Result
from looper.subprocess_utils import (
    get_github_repo,
    is_full_local_mode,
    is_local_mode,
    run_gh_command,
)

# Import LocalIssueStore for full local mode
try:
    from ai_template_scripts.local_issue_store import LocalIssueStore

    _HAS_LOCAL_STORE = True
except ImportError:
    _HAS_LOCAL_STORE = False
    LocalIssueStore = None  # type: ignore[misc, assignment]


def is_feature_freeze() -> bool:
    """Check if feature freeze is active.

    Feature freeze blocks new feature work - only bugs, docs, refactoring allowed.
    Activated by creating a FEATURE_FREEZE file in the repo root.

    ENSURES: Returns True if FEATURE_FREEZE file exists
    ENSURES: Never raises
    """
    return Path("FEATURE_FREEZE").exists()


class IterationIssueCache:
    """Cache all open issues for a single iteration to reduce API calls (#1676).

    All issue-related functions within one looper iteration share this cache.
    The cache is populated on first access and remains valid for the entire
    iteration. Call clear() at the start of each iteration to reset.

    This consolidates 4-8 API calls per iteration into 1:
    - get_sampled_issues() - main issue list
    - get_do_audit_issues() - do-audit filtered
    - check_urgent_handoff() - urgent-handoff filtered
    - _get_domain_issues_targeted() - P0 + domain labels

    Usage:
        # At iteration start:
        IterationIssueCache.clear()

        # All subsequent issue functions use the cache automatically
        issues = get_sampled_issues(role)  # Uses cache
        do_audit = get_do_audit_issues()   # Uses same cache
    """

    _issues: list[dict[str, object]] | None = None
    _fetch_error: str | None = None

    @classmethod
    def clear(cls) -> None:
        """Clear the cache. Call at the start of each iteration."""
        cls._issues = None
        cls._fetch_error = None

    @classmethod
    def get_all(cls) -> Result[list[dict[str, object]]]:
        """Get all open issues, fetching from API if not cached.

        Returns:
            Result with list of issue dicts on success, or failure with error message.
            On failure, the cache stores the error and returns it on subsequent calls.
        """
        if cls._fetch_error is not None:
            return Result.failure(cls._fetch_error, value=[])

        if cls._issues is not None:
            return Result.success(cls._issues)

        # Full local mode: read from LocalIssueStore instead of GitHub API
        if is_full_local_mode():
            if _HAS_LOCAL_STORE and LocalIssueStore is not None:
                try:
                    store = LocalIssueStore()
                    cls._issues = store.get_all_as_dicts(state="open")
                    return Result.success(cls._issues)
                except Exception as e:
                    cls._fetch_error = f"LocalIssueStore error: {e}"
                    return Result.failure(cls._fetch_error, value=[])
            else:
                # LocalIssueStore not available, return empty
                cls._issues = []
                return Result.success([])

        # Regular local mode: return empty list (API disabled)
        if is_local_mode():
            cls._issues = []
            return Result.success([])

        # Get configurable issue limit (#1677, default 200, supports large repos)
        project_config = load_project_config()
        issue_limit = project_config.get("issue_cache_limit", 200)
        if not isinstance(issue_limit, int) or issue_limit < 1:
            issue_limit = 200

        # Fetch all open issues with fields needed by all consumers
        gh_list_timeout = load_timeout_config(project_config).get("gh_list", 15)
        result = run_gh_command(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--limit",
                str(issue_limit),
                "--json",
                "number,title,labels,createdAt,body",
            ],
            timeout=gh_list_timeout,
            repo=get_github_repo(),  # Avoid cwd dependency (#2317)
        )

        if not result.ok:
            error = result.error or "unknown error"
            cls._fetch_error = f"gh issue list failed: {error}"
            return Result.failure(cls._fetch_error, value=[])

        try:
            cls._issues = json.loads(result.value or "[]")
            return Result.success(cls._issues)
        except json.JSONDecodeError as e:
            cls._fetch_error = f"gh issue list JSON parse error: {e}"
            return Result.failure(cls._fetch_error, value=[])

    @classmethod
    def filter_by_label(cls, label: str) -> Result[list[dict[str, object]]]:
        """Get issues with a specific label from the cache.

        Args:
            label: Label to filter by (e.g., "P0", "do-audit", "urgent-handoff").

        Returns:
            Result with filtered issue list.
        """
        all_result = cls.get_all()
        if not all_result.ok:
            return Result.failure(all_result.error or "unknown error", value=[])

        issues = all_result.value or []
        filtered = [i for i in issues if has_label(i, label)]
        return Result.success(filtered)

    @classmethod
    def filter_by_labels_and(cls, labels: list[str]) -> Result[list[dict[str, object]]]:
        """Get issues that have ALL specified labels (AND logic).

        Args:
            labels: List of labels that must all be present.

        Returns:
            Result with filtered issue list.
        """
        all_result = cls.get_all()
        if not all_result.ok:
            return Result.failure(all_result.error or "unknown error", value=[])

        issues = all_result.value or []
        filtered = [i for i in issues if all(has_label(i, lbl) for lbl in labels)]
        return Result.success(filtered)

    @classmethod
    def filter_by_labels_or(cls, labels: list[str]) -> Result[list[dict[str, object]]]:
        """Get issues that have ANY of the specified labels (OR logic).

        Args:
            labels: List of labels where at least one must be present.

        Returns:
            Result with filtered issue list.
        """
        all_result = cls.get_all()
        if not all_result.ok:
            return Result.failure(all_result.error or "unknown error", value=[])

        issues = all_result.value or []
        filtered = [i for i in issues if any(has_label(i, lbl) for lbl in labels)]
        return Result.success(filtered)

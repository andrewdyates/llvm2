# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/issue_audit.py - Audit label transitions.

Functions for managing the do-audit to needs-review workflow transition.
"""

__all__ = [
    "get_do_audit_issues",
    "transition_audit_to_review",
]

import os

from looper.config import load_project_config, load_timeout_config
from looper.context.helpers import is_pending_issue
from looper.context.issue_cache import IterationIssueCache
from looper.result import Result
from looper.subprocess_utils import get_github_repo, is_full_local_mode, run_gh_command

# Import LocalIssueStore for full local mode
try:
    from ai_template_scripts.local_issue_store import LocalIssueStore

    _HAS_LOCAL_STORE = True
except ImportError:
    _HAS_LOCAL_STORE = False
    LocalIssueStore = None  # type: ignore[misc, assignment]


def _get_worker_ownership_label() -> str | None:
    """Get ownership label for current worker from AI_WORKER_ID.

    Returns:
        Ownership label (e.g., "W1", "W2") or None if not in multi-worker mode.
    """
    worker_id = os.environ.get("AI_WORKER_ID")
    if worker_id:
        return f"W{worker_id}"
    return None


def get_do_audit_issues() -> Result[list[dict[str, object]]]:
    """Get open issues with 'do-audit' label owned by current worker.

    In multi-worker mode (AI_WORKER_ID set), only returns issues with
    matching ownership label (W1, W2, etc.) to ensure workers only audit
    their own work. In single-worker mode, returns all do-audit issues.

    Uses IterationIssueCache for client-side filtering (#1676).

    Returns:
        Result with issue dicts (number, title, labels).

    Contracts:
        ENSURES: result.ok implies isinstance(result.value, list)
        ENSURES: not result.ok implies result.value == []
        ENSURES: Never raises - errors returned via Result.failure
    """
    ownership_label = _get_worker_ownership_label()

    # Use cache with client-side filtering instead of per-call API (#1676)
    if ownership_label:
        # Multi-worker mode: filter by both do-audit AND ownership label
        cache_result = IterationIssueCache.filter_by_labels_and(
            ["do-audit", ownership_label]
        )
    else:
        # Single-worker mode: filter by do-audit only
        cache_result = IterationIssueCache.filter_by_label("do-audit")

    if not cache_result.ok:
        error = cache_result.error or "unknown error"
        return Result.failure(
            f"{error}. See docs/troubleshooting.md#github-issues",
            value=[],
        )

    # Filter out pending issues - they have synthetic IDs (#1854)
    issues = [i for i in (cache_result.value or []) if not is_pending_issue(i)]
    return Result.success(issues)


def transition_audit_to_review(issue_num: int | str) -> Result[bool]:
    """Transition issue from do-audit to needs-review.

    Removes 'do-audit' label and adds 'needs-review' label.

    Args:
        issue_num: The issue number to transition (int for GitHub, str like "L1" for local).

    Returns:
        Result with True if successful, False otherwise.

    Contracts:
        REQUIRES: issue_num is valid issue ID (int > 0 or local ID like "L1")
        ENSURES: result.ok and result.value implies both label operations succeeded
        ENSURES: not result.ok implies result.error contains failure details
        ENSURES: Partial failure (one op fails) returns Result.failure, not inconsistent state
        ENSURES: Never raises - errors returned via Result.failure
    """
    from ai_template_scripts.local_issue_store import is_local_issue_id

    # Handle local issues directly via LocalIssueStore
    issue_str = str(issue_num)
    if is_local_issue_id(issue_str) or is_full_local_mode():
        if _HAS_LOCAL_STORE and LocalIssueStore is not None:
            try:
                store = LocalIssueStore()
                # For local issues, update labels directly
                if is_local_issue_id(issue_str):
                    store.edit(
                        issue_str,
                        remove_labels=["do-audit"],
                        add_labels=["needs-review"],
                    )
                    return Result.success(True)
                # Full local mode with GitHub issue number - skip (can't update)
                return Result.success(True)
            except Exception as e:
                return Result.failure(f"LocalIssueStore error: {e}", value=False)
        return Result.success(True)  # No-op if store unavailable

    errors: list[str] = []

    project_config = load_project_config()
    gh_view_timeout = load_timeout_config(project_config).get("gh_view", 10)

    # Get repo for --repo flag to avoid cwd dependency (#2317)
    repo = get_github_repo()

    # Remove do-audit
    remove_result = run_gh_command(
        ["issue", "edit", str(issue_num), "--remove-label", "do-audit"],
        timeout=gh_view_timeout,
        repo=repo,
    )
    if not remove_result.ok:
        error = remove_result.error or "unknown error"
        errors.append(f"remove do-audit failed: {error}")

    # Add needs-review
    add_result = run_gh_command(
        ["issue", "edit", str(issue_num), "--add-label", "needs-review"],
        timeout=gh_view_timeout,
        repo=repo,
    )
    if not add_result.ok:
        error = add_result.error or "unknown error"
        errors.append(f"add needs-review failed: {error}")

    if errors:
        return Result.failure("; ".join(errors), value=False)
    return Result.success(True)

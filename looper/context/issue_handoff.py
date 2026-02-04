# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/issue_handoff.py - Urgent handoff detection.

Functions for detecting when a role urgently needs work via urgent-handoff labels.
"""

__all__ = [
    "check_urgent_handoff",
]

import re

from looper.context.helpers import Issue, has_label, is_pending_issue, issue_number
from looper.context.issue_cache import IterationIssueCache
from looper.result import Result
from looper.subprocess_utils import is_full_local_mode, is_local_mode

# Role keywords for urgent-handoff targeting (use word boundary matching)
# Note: @role patterns removed - \b doesn't match before @ since @ is not a word char
# The bare keyword "worker" still matches "@worker" because \b matches between @ and w
_URGENT_HANDOFF_ROLE_KEYWORDS: dict[str, list[str]] = {
    "worker": ["worker"],
    "prover": ["prover"],
    "researcher": ["researcher"],
    "manager": ["manager"],
}


def _issue_targets_role(issue: Issue, role: str) -> bool:
    """Check if issue body or title targets the specified role.

    Uses regex word boundary matching to avoid false positives.
    Examples that match "worker": "needs @WORKER", "target: worker"
    Does NOT match: "coworker", "workers" (plural)
    """
    keywords = _URGENT_HANDOFF_ROLE_KEYWORDS.get(role, [])
    if not keywords:
        return False

    # Check title and body
    title = str(issue.get("title", "")).lower()
    body = str(issue.get("body", "")).lower()
    text = f"{title} {body}"

    for keyword in keywords:
        # Use regex word boundary for precise matching
        # \b matches start/end of word (boundary between \w and \W)
        # This handles: "worker", "@worker", "worker.", "worker:", etc.
        # But NOT: "workers", "coworker", "preworker"
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        if pattern.search(text):
            return True

    return False


def check_urgent_handoff(role: str) -> Result[list[int]]:
    """Check for urgent-handoff issues targeting the current role.

    Scans open issues with the 'urgent-handoff' label and checks if any
    target the specified role. Used by looper to skip delays when another
    role urgently needs attention.

    Uses IterationIssueCache for client-side filtering (#1676).

    Contracts:
        REQUIRES: role is one of: worker, prover, researcher, manager
        ENSURES: Returns list of issue numbers targeting this role
        ENSURES: Empty list if no urgent handoffs found
        ENSURES: Never raises - catches all exceptions

    Args:
        role: The role to check for (e.g., "worker", "prover").

    Returns:
        Result with list of issue numbers with urgent-handoff targeting this role.
    """
    # Regular local mode: return empty. Full local mode: continue processing.
    if is_local_mode() and not is_full_local_mode():
        return Result.success([])

    normalized_role = role.lower()
    if normalized_role not in _URGENT_HANDOFF_ROLE_KEYWORDS:
        return Result.success([])

    # Use cache with client-side filtering instead of per-call API (#1676)
    cache_result = IterationIssueCache.filter_by_label("urgent-handoff")
    if not cache_result.ok:
        error = cache_result.error or "unknown error"
        return Result.failure(f"urgent-handoff check failed: {error}", value=[])

    issues = cache_result.value or []

    # Filter to issues targeting this role
    # Skip pending issues - they have synthetic IDs, not real issue numbers (#1854)
    targeting_issues: list[int] = []
    for issue in issues:
        if is_pending_issue(issue):
            continue
        if _issue_targets_role(issue, normalized_role):
            targeting_issues.append(issue_number(issue))

    return Result.success(targeting_issues)

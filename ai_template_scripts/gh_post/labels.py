# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Label helpers and workflow cleanup for gh_post."""

import json
import re
import subprocess
import sys
from datetime import UTC, datetime

from ai_template_scripts.gh_issue_numbers import parse_issue_number
from ai_template_scripts.labels import (
    IN_PROGRESS_ALL_LABELS,
    IN_PROGRESS_PREFIX,
    WORKFLOW_LABELS,
)
from looper.log import debug_swallow

__all__ = [
    "IN_PROGRESS_ALL_LABELS",
    "IN_PROGRESS_LABEL_PREFIX",
    "IN_PROGRESS_PREFIX",
    "P_LABELS",
    "PROTECTED_LABELS",
    "USER_ONLY_LABELS",
    "WORKFLOW_LABELS",
    "add_user_label_marker",
    "cleanup_closed_issues_workflow_labels",
    "cleanup_workflow_labels",
    "get_issue_labels",
    "is_label_user_protected",
]

# Labels that USER can protect from AI removal
# When USER adds these labels, a tracking comment is added
# When non-USER tries to remove them, the operation is blocked
PROTECTED_LABELS = {"urgent"}

# Labels that ONLY USER can add (AIs must escalate, not self-promote)
# urgent: USER controls scheduling priority
# P0: USER confirms system-compromised severity (AIs can request via issue body)
USER_ONLY_LABELS = {"urgent", "P0"}

# Alias for backward compatibility with code using IN_PROGRESS_LABEL_PREFIX
IN_PROGRESS_LABEL_PREFIX = IN_PROGRESS_PREFIX + "-"

P_LABELS = frozenset(("P0", "P1", "P2", "P3"))
CLOSED_ISSUE_SCAN_LIMIT = 1000


def is_label_user_protected(
    real_gh: str, issue_number: str, label: str, repo: str | None = None
) -> bool:
    """Check if a label was set by USER (has protection marker in comments)."""
    marker = f"<!-- USER_LABEL:{label} -->"
    try:
        # NOTE: Do NOT use gh -q or --jq - has caching bugs in v2.83.2+ (#1047)
        cmd = [real_gh, "issue", "view", issue_number, "--json", "comments"]
        if repo:
            cmd.extend(["--repo", repo])
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        data = json.loads(result)
        comments = "\n".join(c.get("body", "") for c in data.get("comments", []))
        return marker in comments
    except Exception as e:
        debug_swallow("gh_post_is_label_user_protected", e)
        return False  # Best-effort: label protection check, False is safe default


def add_user_label_marker(
    real_gh: str, issue_number: str, label: str, repo: str | None = None
) -> None:
    """Add a tracking comment when USER adds a protected label."""
    from ai_template_scripts import gh_post as gh_post_module

    marker = f"<!-- USER_LABEL:{label} -->"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    comment = f"{marker}\n🔒 USER protected label `{label}` at {timestamp}"
    cmd = [real_gh, "issue", "comment", issue_number, "--body", comment]
    if repo:
        cmd.extend(["--repo", repo])
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        gh_post_module._invalidate_issue_cache("comment", issue_number, repo)


def get_issue_labels(real_gh: str, issue_number: str, repo: str | None) -> list[str]:
    """Return issue label names (best-effort).

    Falls back to REST API when GraphQL is rate-limited.
    """
    from ai_template_scripts import gh_post as gh_post_module

    # NOTE: Do NOT use gh -q or --jq - has caching bugs in v2.83.2+ (#1047)
    # Try gh issue view first (uses GraphQL)
    cmd = [real_gh, "issue", "view", issue_number, "--json", "labels"]
    if repo:
        cmd.extend(["--repo", repo])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return [
                label["name"] for label in data.get("labels", []) if label.get("name")
            ]
        # Check for GraphQL rate limit - fall back to REST (#1779 - check both stderr and stdout)
        error_output = ((result.stderr or "") + (result.stdout or "")).lower()
        if "rate limit" in error_output:
            return gh_post_module._get_issue_labels_rest(real_gh, issue_number, repo)
    except Exception as e:
        debug_swallow("gh_post_get_issue_labels", e)
    return []


def _get_issue_labels_rest(
    real_gh: str, issue_number: str, repo: str | None
) -> list[str]:
    """Fetch issue labels via REST API fallback."""
    # Determine repo for REST API endpoint
    if repo:
        repo_path = repo
    else:
        # Get current repo from git remote (avoids GraphQL)
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract owner/repo from github.com URL (HTTPS or SSH)
                match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
                if match:
                    repo_path = match.group(1).rstrip("/")
                else:
                    return []
            else:
                return []
        except Exception as e:
            debug_swallow("gh_post_get_issue_labels_rest_repo", e)
            return []

    # Use gh api for REST endpoint
    try:
        cmd = [
            real_gh,
            "api",
            f"repos/{repo_path}/issues/{issue_number}",
            "--jq",
            '.["labels"][].name',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return [
                label.strip()
                for label in result.stdout.strip().split("\n")
                if label.strip()
            ]
    except Exception as e:
        debug_swallow("gh_post_get_issue_labels_rest", e)
    return []


def cleanup_workflow_labels(
    real_gh: str, issue_number: str, repo: str | None, role: str
) -> None:
    """Remove workflow labels when closing issues.

    Only removes labels that actually exist on the issue. This prevents
    'label not found' errors when WORKFLOW_LABELS contains labels that
    don't exist in the repository (e.g., W3-W5 when only W1-W2 exist).

    See #1552.
    """
    from ai_template_scripts import gh_post as gh_post_module

    if not issue_number:
        return

    # Get current issue labels first - only remove what actually exists
    current_labels = gh_post_module.get_issue_labels(real_gh, issue_number, repo)
    current_labels_set = set(current_labels)

    # Find workflow labels that actually exist on this issue
    labels_to_remove = []
    for label in WORKFLOW_LABELS:
        if label not in current_labels_set:
            continue  # Skip labels not present on issue (#1552)
        if label in PROTECTED_LABELS and role != "USER":
            if gh_post_module.is_label_user_protected(
                real_gh, issue_number, label, repo
            ):
                continue
        labels_to_remove.append(label)

    # Also include any dynamic in-progress labels (in-progress-*) not in WORKFLOW_LABELS
    # This catches labels like in-progress-W9 that exceed predefined W1-W5 range
    for label in current_labels:
        if label.startswith(IN_PROGRESS_LABEL_PREFIX) and label not in labels_to_remove:
            labels_to_remove.append(label)

    if not labels_to_remove:
        return

    cmd = [real_gh, "issue", "edit", issue_number]
    if repo:
        cmd.extend(["--repo", repo])
    for label in labels_to_remove:
        cmd.extend(["--remove-label", label])

    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        gh_post_module._invalidate_issue_cache("edit", issue_number, repo)


def cleanup_closed_issues_workflow_labels(
    real_gh: str | None = None, repo: str | None = None, dry_run: bool = False
) -> list[int]:
    """Find and clean workflow labels from closed issues.

    This handles issues auto-closed by GitHub via 'Fixes #N' commits,
    which bypass the normal close path that removes workflow labels.

    Args:
        real_gh: Path to real gh binary (auto-detected if None)
        repo: Repository in owner/repo format (current repo if None)
        dry_run: If True, report issues but don't modify them

    Returns:
        List of issue numbers that were (or would be) cleaned.

    Note:
        This function makes multiple API calls (one per workflow label type
        plus one edit per issue found). On repos with many closed issues,
        consider running with dry_run=True first to assess scope.
    """
    from ai_template_scripts import gh_post as gh_post_module

    if real_gh is None:
        real_gh = gh_post_module.get_real_gh()

    # Collect issues with workflow labels (need to query each label separately
    # since gh --label uses AND logic for comma-separated values)
    seen_issues: dict[int, list[str]] = {}  # number -> labels

    for label in WORKFLOW_LABELS:
        cmd = [
            real_gh,
            "issue",
            "list",
            "--state",
            "closed",
            "--label",
            label,
            "--json",
            "number,labels",
            "--limit",
            str(CLOSED_ISSUE_SCAN_LIMIT),
        ]
        if repo:
            cmd.extend(["--repo", repo])

        try:
            result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            issues = json.loads(result)
        except Exception as e:
            debug_swallow("gh_post_cleanup_closed_issue_labels", e)
            continue

        for issue in issues:
            num = parse_issue_number(issue.get("number"))
            if num is None:
                continue
            if num not in seen_issues:
                # Extract all workflow labels from this issue
                labels = [
                    lbl.get("name", "") if isinstance(lbl, dict) else lbl
                    for lbl in issue.get("labels", [])
                ]
                workflow_labels = [
                    lbl
                    for lbl in labels
                    if lbl in WORKFLOW_LABELS
                    or lbl.startswith(IN_PROGRESS_LABEL_PREFIX)
                ]
                seen_issues[num] = workflow_labels

    cleaned: list[int] = []
    role = gh_post_module.get_effective_role()

    for issue_number, labels_to_remove in sorted(seen_issues.items()):
        if not labels_to_remove:
            continue

        if dry_run:
            print(f"Would clean #{issue_number}: {labels_to_remove}", file=sys.stderr)
            cleaned.append(issue_number)
            continue

        # Actually clean the labels
        gh_post_module.cleanup_workflow_labels(real_gh, str(issue_number), repo, role)
        cleaned.append(issue_number)
        print(f"Cleaned #{issue_number}: removed {labels_to_remove}", file=sys.stderr)

    return cleaned

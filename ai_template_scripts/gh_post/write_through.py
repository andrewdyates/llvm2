# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Write-through support for issue operations (#1834).

Mirrors GitHub issue operations to local .issues/ directory, ensuring local
state is always current even when GitHub API fails or is rate-limited.

Public API:
    write_through_create(issue_number, title, body, labels, repo)
    write_through_comment(issue_number, body, repo)
    write_through_edit(issue_number, title, body, add_labels, remove_labels, repo)
    write_through_close(issue_number, repo)
    is_write_through_enabled()

Design:
    - Local store uses GitHub issue numbers directly (no L-prefix)
    - L-prefixed IDs are reserved for truly local-only issues (AIT_LOCAL_MODE=full)
    - Write-through happens AFTER successful GitHub API call
    - On rate-limit queue, write-through happens IMMEDIATELY (local is authoritative)

Configuration:
    - AIT_WRITE_THROUGH=1: Enable write-through (default when not in full local mode)
    - AIT_WRITE_THROUGH=0: Disable write-through explicitly
    - AIT_LOCAL_MODE=full: Full local mode (disables write-through, uses gh_local.py)
"""

from __future__ import annotations

__all__ = [
    "is_write_through_enabled",
    "write_through_create",
    "write_through_comment",
    "write_through_edit",
    "write_through_close",
    "write_through_from_queue",
]

import functools
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from looper.log import debug_swallow

def _get_local_store():
    """Lazy import to avoid circular dependencies."""
    from ai_template_scripts.local_issue_store import LocalIssue, LocalIssueStore

    return LocalIssue, LocalIssueStore


def is_write_through_enabled() -> bool:
    """Check if write-through mode is enabled.

    Write-through is enabled by default unless:
    - AIT_LOCAL_MODE=full (full local mode)
    - AIT_WRITE_THROUGH=0 (explicitly disabled)

    Returns:
        True if write-through should be performed.
    """
    # Full local mode disables write-through (gh_local.py handles everything)
    if os.environ.get("AIT_LOCAL_MODE") == "full":
        return False

    # Explicit disable
    if os.environ.get("AIT_WRITE_THROUGH") == "0":
        return False

    # Default: enabled
    return True


@functools.lru_cache(maxsize=1)
def _get_repo_root() -> Path:
    """Get repository root directory (cached — repo root is constant per process)."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception as e:
        debug_swallow("gh_post_get_repo_root", e)
    return Path.cwd()


def _get_issues_dir(repo: str | None = None) -> Path:
    """Get .issues/ directory path.

    Args:
        repo: Optional repo name (unused for now, for future multi-repo support).

    Returns:
        Path to .issues/ directory in current repo.
    """
    repo_root = _get_repo_root()
    return repo_root / ".issues"


def _issue_path(issue_number: str | int, repo: str | None = None) -> Path:
    """Get path to issue file.

    Args:
        issue_number: GitHub issue number (e.g., 1834).
        repo: Optional repo name.

    Returns:
        Path to issue markdown file.
    """
    issues_dir = _get_issues_dir(repo)
    # Use numeric ID directly (not L-prefixed)
    return issues_dir / f"{issue_number}.md"


def _new_queued_issue_id() -> str:
    """Generate a unique temporary ID for queued create operations."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"Q{timestamp}_{uuid.uuid4().hex[:8]}"


def _read_local_issue(issue_number: str | int, repo: str | None = None):
    """Read local issue if it exists.

    Args:
        issue_number: GitHub issue number.
        repo: Optional repo name.

    Returns:
        LocalIssue instance or None if not found.
    """
    LocalIssue, _ = _get_local_store()
    path = _issue_path(issue_number, repo)

    if not path.exists():
        return None

    try:
        content = path.read_text()
        return LocalIssue.from_markdown(content)
    except (ValueError, OSError) as e:
        debug_swallow("gh_post_read_local_issue", e)
        return None


def _write_local_issue(issue, repo: str | None = None) -> None:
    """Write issue to local store.

    Args:
        issue: LocalIssue instance to write.
        repo: Optional repo name.
    """
    from ai_template_scripts.atomic_write import atomic_write_text

    issues_dir = _get_issues_dir(repo)
    issues_dir.mkdir(parents=True, exist_ok=True)

    path = _issue_path(issue.id, repo)
    content = issue.to_markdown()
    atomic_write_text(path, content)


def write_through_create(
    issue_number: str | int,
    title: str,
    body: str = "",
    labels: list[str] | None = None,
    repo: str | None = None,
) -> bool:
    """Write-through for issue create operation.

    Called after successful GitHub issue creation to mirror to local store.

    Args:
        issue_number: GitHub issue number from API response.
        title: Issue title.
        body: Issue body.
        labels: Issue labels.
        repo: Repository name (for logging).

    Returns:
        True if write succeeded, False on error.
    """
    if not is_write_through_enabled():
        return False

    try:
        LocalIssue, _ = _get_local_store()

        issue = LocalIssue(
            id=str(issue_number),
            title=title,
            body=body,
            labels=labels or [],
            state="open",
        )

        _write_local_issue(issue, repo)
        print(
            f"[write-through] Mirrored issue #{issue_number} to local store",
            file=sys.stderr,
        )
        return True
    except Exception as e:
        print(f"[write-through] Error creating local issue: {e}", file=sys.stderr)
        return False


def write_through_comment(
    issue_number: str | int,
    body: str,
    repo: str | None = None,
) -> bool:
    """Write-through for comment operation.

    Called after successful GitHub comment creation.

    Args:
        issue_number: GitHub issue number.
        body: Comment body.
        repo: Repository name.

    Returns:
        True if write succeeded, False on error.
    """
    if not is_write_through_enabled():
        return False

    try:
        LocalIssue, _ = _get_local_store()
        from ai_template_scripts.local_issue_store import LocalIssueComment

        # Read existing issue or create placeholder
        issue = _read_local_issue(issue_number, repo)
        if issue is None:
            # Issue doesn't exist locally - create minimal placeholder
            issue = LocalIssue(
                id=str(issue_number),
                title=f"Issue #{issue_number}",  # Placeholder
                body="",
                state="open",
            )

        # Build author from environment
        role = os.environ.get("AI_ROLE", "U")[0].upper()
        iteration = os.environ.get("AI_ITERATION", "?")
        author = f"[{role}]{iteration}"

        comment = LocalIssueComment(
            author=author,
            timestamp=datetime.now(UTC).isoformat(),
            body=body,
        )
        issue.comments.append(comment)
        issue.updated_at = datetime.now(UTC).isoformat()

        _write_local_issue(issue, repo)
        print(
            f"[write-through] Added comment to #{issue_number} in local store",
            file=sys.stderr,
        )
        return True
    except Exception as e:
        print(f"[write-through] Error adding comment: {e}", file=sys.stderr)
        return False


def write_through_edit(
    issue_number: str | int,
    title: str | None = None,
    body: str | None = None,
    add_labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
    repo: str | None = None,
) -> bool:
    """Write-through for edit operation.

    Called after successful GitHub issue edit.

    Args:
        issue_number: GitHub issue number.
        title: New title (if changed).
        body: New body (if changed).
        add_labels: Labels to add.
        remove_labels: Labels to remove.
        repo: Repository name.

    Returns:
        True if write succeeded, False on error.
    """
    if not is_write_through_enabled():
        return False

    try:
        LocalIssue, _ = _get_local_store()

        # Read existing issue or create placeholder
        issue = _read_local_issue(issue_number, repo)
        if issue is None:
            # Issue doesn't exist locally - create with edit data
            issue = LocalIssue(
                id=str(issue_number),
                title=title or f"Issue #{issue_number}",
                body=body or "",
                labels=add_labels or [],
                state="open",
            )
        else:
            # Apply edits
            if title is not None:
                issue.title = title
            if body is not None:
                issue.body = body

            if add_labels:
                for label in add_labels:
                    if label not in issue.labels:
                        issue.labels.append(label)

            if remove_labels:
                issue.labels = [lbl for lbl in issue.labels if lbl not in remove_labels]

            issue.updated_at = datetime.now(UTC).isoformat()

        _write_local_issue(issue, repo)
        print(
            f"[write-through] Updated issue #{issue_number} in local store",
            file=sys.stderr,
        )
        return True
    except Exception as e:
        print(f"[write-through] Error editing issue: {e}", file=sys.stderr)
        return False


def write_through_close(
    issue_number: str | int,
    repo: str | None = None,
) -> bool:
    """Write-through for close operation.

    Called after successful GitHub issue close.

    Args:
        issue_number: GitHub issue number.
        repo: Repository name.

    Returns:
        True if write succeeded, False on error.
    """
    if not is_write_through_enabled():
        return False

    try:
        LocalIssue, _ = _get_local_store()

        # Read existing issue or create placeholder
        issue = _read_local_issue(issue_number, repo)
        if issue is None:
            # Issue doesn't exist locally - create closed placeholder
            issue = LocalIssue(
                id=str(issue_number),
                title=f"Issue #{issue_number}",
                body="",
                state="closed",
            )
        else:
            issue.state = "closed"
            issue.updated_at = datetime.now(UTC).isoformat()

        _write_local_issue(issue, repo)
        print(
            f"[write-through] Closed issue #{issue_number} in local store",
            file=sys.stderr,
        )
        return True
    except Exception as e:
        print(f"[write-through] Error closing issue: {e}", file=sys.stderr)
        return False


def write_through_from_queue(
    operation: str,
    data: dict[str, Any],
) -> bool:
    """Write-through when operation is queued (rate-limited).

    When GitHub API is unavailable, this writes to local store immediately
    so local state is always authoritative.

    Args:
        operation: Operation type (create, comment, edit, close).
        data: Operation data dict from queue.

    Returns:
        True if write succeeded, False on error.
    """
    if not is_write_through_enabled():
        return False

    repo = data.get("repo")

    if operation == "create":
        return write_through_create(
            issue_number=_new_queued_issue_id(),
            title=data.get("title", ""),
            body=data.get("body", ""),
            labels=data.get("labels", []),
            repo=repo,
        )

    elif operation == "comment":
        issue_number = data.get("issue")
        if not issue_number:
            return False
        return write_through_comment(
            issue_number=issue_number,
            body=data.get("body", ""),
            repo=repo,
        )

    elif operation == "edit":
        issue_number = data.get("issue")
        if not issue_number:
            return False
        return write_through_edit(
            issue_number=issue_number,
            title=data.get("title"),
            body=data.get("body"),
            add_labels=data.get("add_labels"),
            remove_labels=data.get("remove_labels"),
            repo=repo,
        )

    elif operation == "close":
        issue_number = data.get("issue")
        if not issue_number:
            return False
        return write_through_close(
            issue_number=issue_number,
            repo=repo,
        )

    return False

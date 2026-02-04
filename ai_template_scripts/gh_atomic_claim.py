# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_atomic_claim.py - Atomic issue claiming for race condition prevention (#731)

Protocol:
1. Acquire local lock (same-machine coordination)
2. Post comment with UUID marker
3. Wait for GitHub API propagation
4. Fetch all comments and verify our claim is first
5. Only add label if verification passes
6. Release lock

Design doc: designs/2026-01-30-atomic-issue-claiming.md
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

# fcntl for file locking (Unix only, Windows unsupported)
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# Constants
CLAIM_LOCK_DIR = Path.home() / ".ait_gh_lock"
CLAIM_LOCK_TIMEOUT_SEC = 60.0  # Max wait time for local lock
CLAIM_VERIFY_DELAY_SEC = 2.0  # Wait for GitHub API propagation
CLAIM_MARKER_RE = re.compile(r"\[claim:([a-f0-9]{8})\]")  # UUID marker pattern
CLAIM_STALE_MINUTES = 5  # Ignore claim comments older than this

__all__ = [
    "CLAIM_MARKER_RE",
    "CLAIM_STALE_MINUTES",
    "_atomic_claim_issue",
    "_find_own_claim_comment",
    "_generate_claim_uuid",
    "_parse_claim_comments",
]


def _get_claim_repo_name() -> str:
    """Get sanitized repo name for claim lock file."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            repo_name = Path(result.stdout.strip()).name
            # Sanitize: replace non-identifier chars
            return re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)
    except Exception:
        pass
    return "unknown"


def _acquire_claim_lock(
    timeout: float = CLAIM_LOCK_TIMEOUT_SEC,
) -> tuple[TextIO | None, bool]:
    """Acquire lock for issue claiming operations."""
    if not HAS_FCNTL:
        return None, True

    CLAIM_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    repo_name = _get_claim_repo_name()
    lock_path = CLAIM_LOCK_DIR / f"claim_{repo_name}.lock"

    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        start_time = time.time()

        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file, True
            except BlockingIOError:
                if time.time() - start_time > timeout:
                    lock_file.close()
                    return None, False
                time.sleep(0.1)
    except OSError:
        if lock_file is not None:
            try:
                lock_file.close()
            except OSError:
                pass
        return None, False


def _release_claim_lock(lock_file: TextIO | None) -> None:
    """Release claim lock."""
    if lock_file is None:
        return

    if HAS_FCNTL:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
    try:
        lock_file.close()
    except OSError:
        pass


def _generate_claim_uuid() -> str:
    """Generate 8-character hex UUID for claim marker."""
    return uuid.uuid4().hex[:8]


def _parse_claim_comments(
    comments: list[dict], cutoff_minutes: int = CLAIM_STALE_MINUTES
) -> list[tuple[str, str, str]]:
    """Parse claim comments and extract (created_at, uuid, comment_id).

    Returns list sorted by createdAt timestamp (earliest first).
    Note: comment_id may be GraphQL node ID (string) or numeric ID.
    """
    now = datetime.now(UTC)
    claims = []

    for comment in comments:
        body = comment.get("body", "")
        comment_id = comment.get("id", "")
        created_at = comment.get("createdAt", "")

        match = CLAIM_MARKER_RE.search(body)
        if not match:
            continue

        claim_uuid = match.group(1)

        # Skip claims without valid timestamps - we need timestamps for ordering
        if not created_at:
            continue

        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))  # noqa: FURB162
            age_minutes = (now - created).total_seconds() / 60
            if age_minutes > cutoff_minutes:
                continue
        except (ValueError, TypeError):
            # Skip claims without valid timestamps for ordering
            continue

        comment_id_str = str(comment_id)
        claims.append((created, comment_id_str, created_at, claim_uuid))

    # Sort by parsed timestamp; tie-break on comment ID for deterministic ordering.
    claims.sort(key=lambda x: (x[0], x[1]))
    return [
        (created_at, claim_uuid, comment_id_str)
        for _, comment_id_str, created_at, claim_uuid in claims
    ]


def _fetch_issue_comments(
    real_gh: str, issue_number: str, repo: str | None = None
) -> list[dict]:
    """Fetch all comments for an issue."""
    cmd = [real_gh, "issue", "view", issue_number, "--json", "comments"]
    if repo:
        cmd.extend(["--repo", repo])

    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        data = json.loads(result)
        return data.get("comments", [])
    except Exception:
        return []


def _build_claim_comment(
    identity: dict, claim_uuid: str, commit: str, process_body_fn: Callable[..., Any]
) -> str:
    """Build claim comment body with UUID marker."""
    worker_id = os.environ.get("AI_WORKER_ID", "")
    worker_suffix = f" (W{worker_id})" if worker_id else ""
    claim_text = f"Claiming this issue{worker_suffix} at {commit} [claim:{claim_uuid}]"
    return process_body_fn(claim_text, identity)


def _find_own_claim_comment(
    comments: list[dict],
    identity: dict,
    cutoff_minutes: int = CLAIM_STALE_MINUTES,
) -> str | None:
    """Find existing claim comment from this worker.

    Returns the claim UUID if found, None otherwise.
    """
    now = datetime.now(UTC)
    # Build identity pattern to match our comments
    project = identity.get("project", "")
    role = identity.get("role", "")

    for comment in comments:
        body = comment.get("body", "")
        created_at = comment.get("createdAt", "")

        # Check if this is a claim comment
        match = CLAIM_MARKER_RE.search(body)
        if not match:
            continue

        # Check age
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))  # noqa: FURB162
                age_minutes = (now - created).total_seconds() / 60
                if age_minutes > cutoff_minutes:
                    continue
            except (ValueError, TypeError):
                continue

        # Check if this comment is from our project/role
        # Identity signature format: project | role #iteration | ...
        if project and f"{project} |" in body:
            # Match our project
            if role and f"| {role} " in body:
                # Match our role
                return match.group(1)

    return None


def _atomic_claim_issue(
    real_gh: str,
    issue_number: str,
    in_progress_label: str,
    repo: str | None = None,
    get_identity_fn: Callable[..., Any] | None = None,
    get_commit_fn: Callable[..., Any] | None = None,
    process_body_fn: Callable[..., Any] | None = None,
    invalidate_cache_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Atomically claim an issue using comment-first protocol."""
    if get_identity_fn is None or get_commit_fn is None or process_body_fn is None:
        return False, "Required callback functions not provided"

    identity = get_identity_fn()
    commit = get_commit_fn()

    lock_file, acquired = _acquire_claim_lock()
    if not acquired:
        return False, "Could not acquire claim lock (timeout)"

    try:
        # Check for existing claim comment from this worker (#1410)
        # Prevents comment spam on retries
        existing_comments = _fetch_issue_comments(real_gh, issue_number, repo)
        existing_uuid = _find_own_claim_comment(existing_comments, identity)

        if existing_uuid:
            # Reuse existing claim instead of posting new comment
            claim_uuid = existing_uuid
        else:
            # Post new claim comment
            claim_uuid = _generate_claim_uuid()
            claim_body = _build_claim_comment(
                identity, claim_uuid, commit, process_body_fn
            )
            comment_cmd = [
                real_gh,
                "issue",
                "comment",
                issue_number,
                "--body",
                claim_body,
            ]
            if repo:
                comment_cmd.extend(["--repo", repo])

            try:
                result = subprocess.run(
                    comment_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                return False, "Failed to post claim comment: timed out"
            if result.returncode != 0:
                return False, f"Failed to post claim comment: {result.stderr[:100]}"

        if invalidate_cache_fn is not None:
            invalidate_cache_fn("comment", issue_number)

        time.sleep(CLAIM_VERIFY_DELAY_SEC)

        comments = _fetch_issue_comments(real_gh, issue_number, repo)
        claims = _parse_claim_comments(comments)

        # Safety: if we can't find any claims (including ours), don't proceed
        # This could happen if comment fetch failed or rate limited
        if not claims:
            return False, "Could not verify claim (no claims found after fetch)"

        if not any(claim[1] == claim_uuid for claim in claims):
            return False, "Could not verify claim (claim UUID not found after fetch)"

        # Check if our claim won (earliest timestamp)
        if claims[0][1] != claim_uuid:
            winner_uuid = claims[0][1]
            return False, f"Yielding to claim:{winner_uuid} (earlier timestamp)"

        label_cmd = [
            real_gh,
            "issue",
            "edit",
            issue_number,
            "--add-label",
            in_progress_label,
        ]
        if repo:
            label_cmd.extend(["--repo", repo])

        try:
            result = subprocess.run(
                label_cmd, capture_output=True, text=True, check=False, timeout=30
            )
        except subprocess.TimeoutExpired:
            return False, "Failed to add label: timed out"
        if result.returncode != 0:
            return False, f"Failed to add label: {result.stderr[:100]}"

        if invalidate_cache_fn is not None:
            invalidate_cache_fn("edit", issue_number)
        return True, f"Claimed with [claim:{claim_uuid}]"

    finally:
        _release_claim_lock(lock_file)

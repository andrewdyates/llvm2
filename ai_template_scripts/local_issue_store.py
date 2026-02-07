# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Local issue storage for offline development mode.

Provides file-based issue storage when GitHub API is unavailable or saturated.
Issues are stored as Markdown files with YAML frontmatter in .issues/ directory.

Public API:
    from ai_template_scripts.local_issue_store import LocalIssueStore, LocalIssue

    store = LocalIssueStore()
    issue = store.create("Add feature X", labels=["P2", "feature"])
    issues = store.list_issues()
    store.edit(issue.id, add_labels=["in-progress"])
    store.close(issue.id)

Local Issue Format:
    .issues/L1.md - Markdown with YAML frontmatter
    .issues/_meta.json - Next ID counter, metadata

Design decisions:
    - IDs use L<int> prefix to avoid collision with GitHub numbers
    - YAML frontmatter for metadata (parseable, human-readable)
    - Comments stored inline in Markdown (## Comments section)
    - Atomic file replacement on writes for reader safety
"""

from __future__ import annotations

__all__ = [
    "LocalIssue",
    "LocalIssueStore",
    "LOCAL_ISSUE_PREFIX",
    "is_local_issue_id",
]

import fcntl
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_template_scripts.atomic_write import atomic_write_text

# Local issue ID prefix - distinguishes from GitHub issue numbers
LOCAL_ISSUE_PREFIX = "L"

# Regex for matching local issue IDs (L1, L42, etc.)
_LOCAL_ID_PATTERN = re.compile(rf"^{LOCAL_ISSUE_PREFIX}(\d+)$")


def is_local_issue_id(issue_id: str | int) -> bool:
    """Check if an issue ID is a local issue (L-prefixed).

    Args:
        issue_id: Issue identifier to check.

    Returns:
        True if this is a local issue ID (L1, L2, etc.), False otherwise.
    """
    if isinstance(issue_id, int):
        return False
    return bool(_LOCAL_ID_PATTERN.match(str(issue_id)))


@dataclass
class LocalIssueComment:
    """A comment on a local issue."""

    author: str  # Role and iteration, e.g., "[W]42"
    timestamp: str  # ISO format
    body: str


@dataclass
class LocalIssue:
    """Represents a local issue stored in .issues/ directory.

    Fields match GitHub issue structure for compatibility with existing code.
    """

    id: str  # L-prefixed ID (e.g., "L1")
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    state: str = "open"  # "open" or "closed"
    created_at: str = ""  # ISO timestamp
    updated_at: str = ""  # ISO timestamp
    comments: list[LocalIssueComment] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set timestamps if not provided."""
        now = datetime.now(UTC).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def number(self) -> str:
        """Return issue number (same as id for local issues)."""
        return self.id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format compatible with GitHub issue structure.

        Returns:
            Dict with fields matching gh issue list --json output.
        """
        return {
            "number": self.id,
            "title": self.title,
            "body": self.body,
            "labels": [{"name": lbl} for lbl in self.labels],
            "state": self.state.upper(),  # GitHub uses OPEN/CLOSED
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def to_markdown(self) -> str:
        """Serialize issue to Markdown with YAML frontmatter.

        Returns:
            Markdown string suitable for writing to .issues/L*.md
        """
        # YAML frontmatter
        labels_yaml = json.dumps(self.labels)  # JSON array format in YAML
        frontmatter = f"""---
id: {self.id}
title: {json.dumps(self.title)}
labels: {labels_yaml}
state: {self.state}
created_at: {self.created_at}
updated_at: {self.updated_at}
---"""

        # Body
        content = f"{frontmatter}\n\n{self.body}"

        # Comments section
        if self.comments:
            content += "\n\n## Comments\n"
            for comment in self.comments:
                content += (
                    f"\n### {comment.author} @ {comment.timestamp}\n{comment.body}\n"
                )

        return content

    @classmethod
    def from_markdown(cls, content: str) -> LocalIssue:
        """Parse issue from Markdown with YAML frontmatter.

        Args:
            content: Markdown content with YAML frontmatter.

        Returns:
            LocalIssue instance.

        Raises:
            ValueError: If content cannot be parsed.
        """
        # Split frontmatter from body
        if not content.startswith("---"):
            raise ValueError("Missing YAML frontmatter")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid frontmatter format")

        frontmatter = parts[1].strip()
        body_and_comments = parts[2].strip()

        # Parse YAML frontmatter (simple key: value parsing)
        metadata: dict[str, Any] = {}
        for line in frontmatter.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Parse JSON values (arrays, quoted strings)
                if value.startswith("[") or value.startswith('"'):
                    try:
                        metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata[key] = value
                else:
                    metadata[key] = value

        # Split body from comments
        body = body_and_comments
        comments: list[LocalIssueComment] = []

        if "\n## Comments\n" in body_and_comments:
            body_parts = body_and_comments.split("\n## Comments\n", 1)
            body = body_parts[0].strip()

            # Parse comments
            comments_section = body_parts[1]
            comment_pattern = re.compile(
                r"### (.+?) @ (\d{4}-\d{2}-\d{2}T[\d:+.-]+Z?)\n(.*?)(?=\n### |\Z)",
                re.DOTALL,
            )
            for match in comment_pattern.finditer(comments_section):
                comments.append(
                    LocalIssueComment(
                        author=match.group(1),
                        timestamp=match.group(2),
                        body=match.group(3).strip(),
                    )
                )

        return cls(
            id=str(metadata.get("id", "")),
            title=str(metadata.get("title", "")),
            body=body,
            labels=metadata.get("labels", []),
            state=str(metadata.get("state", "open")),
            created_at=str(metadata.get("created_at", "")),
            updated_at=str(metadata.get("updated_at", "")),
            comments=comments,
        )


class LocalIssueStore:
    """File-based issue storage for local development mode.

    Issues are stored as Markdown files in .issues/ directory:
    - .issues/L1.md, .issues/L2.md, etc.
    - .issues/_meta.json for next ID counter

    Thread-safe ID allocation via file locking and atomic file replacement.
    """

    def __init__(self, repo_root: Path | None = None):
        """Initialize store.

        Args:
            repo_root: Repository root directory. If None, uses current directory.
        """
        self.repo_root = repo_root or Path.cwd()
        self.issues_dir = self.repo_root / ".issues"
        self.meta_file = self.issues_dir / "_meta.json"

    def _ensure_dir(self) -> None:
        """Ensure .issues/ directory exists."""
        self.issues_dir.mkdir(parents=True, exist_ok=True)

    def _scan_max_issue_id(self) -> int:
        """Scan .issues/ for existing L*.md files and return max ID number.

        Used as a recovery mechanism when _meta.json is corrupt or stale,
        to prevent ID reuse that would overwrite existing issues (#2906).

        Returns:
            Max numeric ID found, or 0 if no issue files exist.
        """
        max_id = 0
        for path in self.issues_dir.glob("L*.md"):
            match = _LOCAL_ID_PATTERN.match(path.stem)
            if match:
                num = int(match.group(1))
                if num > max_id:
                    max_id = num
        return max_id

    def _get_next_id(self) -> str:
        """Get next available local issue ID.

        Uses atomic read-modify-write with file locking to prevent TOCTOU
        race conditions when multiple processes call this concurrently.

        Locking uses a separate .lock file so the flock inode is stable
        across the temp+replace cycle on _meta.json.

        When _meta.json is missing or corrupt, scans existing issue files
        to prevent ID reuse (#2906).

        Returns:
            Next ID string (e.g., "L1", "L2").
        """
        self._ensure_dir()
        lock_path = self.meta_file.parent / f"{self.meta_file.name}.lock"

        # Lock a dedicated file; its inode survives the data-file replace.
        with open(lock_path, "a+") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                # Read current meta from data file
                meta: dict[str, Any] = {"next_id": 1}
                if self.meta_file.exists():
                    try:
                        content = self.meta_file.read_text()
                        if content.strip():
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                meta = parsed
                    except (json.JSONDecodeError, TypeError, ValueError, OSError):
                        pass

                next_num = meta.get("next_id", 1)
                if not isinstance(next_num, int):
                    try:
                        next_num = int(next_num)
                    except (TypeError, ValueError):
                        next_num = 1
                if next_num < 1:
                    next_num = 1

                # Guard against stale/partial metadata by ensuring next_id is
                # always above the highest existing issue file (#2906).
                max_existing = self._scan_max_issue_id()
                if next_num <= max_existing:
                    next_num = max_existing + 1

                meta["next_id"] = next_num + 1

                # Write to temp file then replace, so a crash never
                # leaves _meta.json empty (unlike truncate+write).
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{self.meta_file.name}.tmp.",
                    dir=self.meta_file.parent,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                        json.dump(meta, tmp_f, indent=2)
                        tmp_f.write("\n")
                        tmp_f.flush()
                        os.fsync(tmp_f.fileno())
                    os.replace(tmp_name, self.meta_file)
                except OSError:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

        return f"{LOCAL_ISSUE_PREFIX}{next_num}"

    def _issue_path(self, issue_id: str) -> Path:
        """Get file path for issue.

        Args:
            issue_id: Local issue ID (e.g., "L1").

        Returns:
            Path to issue file.
        """
        return self.issues_dir / f"{issue_id}.md"

    def _write_issue(self, issue: LocalIssue) -> None:
        """Write issue file atomically.

        Args:
            issue: Issue to write.
        """
        self._ensure_dir()
        path = self._issue_path(issue.id)

        content = issue.to_markdown()
        atomic_write_text(path, content)

    def _read_issue(self, issue_id: str) -> LocalIssue | None:
        """Read issue from file.

        Args:
            issue_id: Local issue ID.

        Returns:
            LocalIssue if found, None otherwise.
        """
        path = self._issue_path(issue_id)
        if not path.exists():
            return None

        try:
            content = path.read_text()
            return LocalIssue.from_markdown(content)
        except (ValueError, OSError):
            return None

    def create(
        self,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
    ) -> LocalIssue:
        """Create a new local issue.

        Args:
            title: Issue title.
            body: Issue body (optional).
            labels: List of labels (optional).

        Returns:
            Created LocalIssue instance.
        """
        issue_id = self._get_next_id()
        issue = LocalIssue(
            id=issue_id,
            title=title,
            body=body,
            labels=labels or [],
        )
        self._write_issue(issue)
        return issue

    def view(self, issue_id: str) -> LocalIssue | None:
        """Get a single issue by ID.

        Args:
            issue_id: Local issue ID (e.g., "L1").

        Returns:
            LocalIssue if found, None otherwise.
        """
        return self._read_issue(issue_id)

    def list_issues(
        self,
        state: str = "open",
        labels: list[str] | None = None,
    ) -> list[LocalIssue]:
        """List issues matching criteria.

        Args:
            state: Filter by state ("open", "closed", "all").
            labels: Filter by labels (issue must have ALL listed labels).

        Returns:
            List of matching LocalIssue instances.
        """
        if not self.issues_dir.exists():
            return []

        issues: list[LocalIssue] = []
        for path in self.issues_dir.glob("L*.md"):
            issue = self._read_issue(path.stem)
            if issue is None:
                continue

            # Filter by state
            if state != "all" and issue.state != state:
                continue

            # Filter by labels (AND logic - must have all)
            if labels:
                issue_labels_lower = {lbl.lower() for lbl in issue.labels}
                if not all(lbl.lower() in issue_labels_lower for lbl in labels):
                    continue

            issues.append(issue)

        # Sort by created_at descending (newest first)
        issues.sort(key=lambda i: i.created_at, reverse=True)
        return issues

    def edit(
        self,
        issue_id: str,
        title: str | None = None,
        body: str | None = None,
        add_labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
    ) -> LocalIssue | None:
        """Edit an existing issue.

        Args:
            issue_id: Local issue ID.
            title: New title (optional).
            body: New body (optional).
            add_labels: Labels to add (optional).
            remove_labels: Labels to remove (optional).

        Returns:
            Updated LocalIssue if found, None otherwise.
        """
        issue = self._read_issue(issue_id)
        if issue is None:
            return None

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
        self._write_issue(issue)
        return issue

    def comment(
        self,
        issue_id: str,
        body: str,
        author: str = "",
    ) -> LocalIssue | None:
        """Add a comment to an issue.

        Args:
            issue_id: Local issue ID.
            body: Comment body.
            author: Comment author (e.g., "[W]42"). Defaults to env-based identity.

        Returns:
            Updated LocalIssue if found, None otherwise.
        """
        import os

        issue = self._read_issue(issue_id)
        if issue is None:
            return None

        # Build author string from environment if not provided
        if not author:
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
        self._write_issue(issue)
        return issue

    def close(self, issue_id: str) -> LocalIssue | None:
        """Close an issue.

        Args:
            issue_id: Local issue ID.

        Returns:
            Updated LocalIssue if found, None otherwise.
        """
        issue = self._read_issue(issue_id)
        if issue is None:
            return None

        issue.state = "closed"
        issue.updated_at = datetime.now(UTC).isoformat()
        self._write_issue(issue)
        return issue

    def reopen(self, issue_id: str) -> LocalIssue | None:
        """Reopen a closed issue.

        Args:
            issue_id: Local issue ID.

        Returns:
            Updated LocalIssue if found, None otherwise.
        """
        issue = self._read_issue(issue_id)
        if issue is None:
            return None

        issue.state = "open"
        issue.updated_at = datetime.now(UTC).isoformat()
        self._write_issue(issue)
        return issue

    def exists(self, issue_id: str) -> bool:
        """Check if an issue exists.

        Args:
            issue_id: Local issue ID.

        Returns:
            True if issue exists, False otherwise.
        """
        return self._issue_path(issue_id).exists()

    def get_all_as_dicts(self, state: str = "open") -> list[dict[str, Any]]:
        """Get all issues as dicts compatible with GitHub API format.

        This is used for integration with IterationIssueCache.

        Args:
            state: Filter by state ("open", "closed", "all").

        Returns:
            List of issue dicts matching gh issue list --json format.
        """
        issues = self.list_issues(state=state)
        return [issue.to_dict() for issue in issues]

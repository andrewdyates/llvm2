#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Local gh command handler for offline development mode.

Routes gh issue commands to LocalIssueStore when AIT_LOCAL_MODE=full.
Provides same interface as gh CLI for transparent operation.

Usage:
    AIT_LOCAL_MODE=full gh issue create --title "Test" --label P2
    AIT_LOCAL_MODE=full gh issue list
    AIT_LOCAL_MODE=full gh issue view L1
    AIT_LOCAL_MODE=full gh issue edit L1 --add-label in-progress
    AIT_LOCAL_MODE=full gh issue close L1
    AIT_LOCAL_MODE=full gh issue comment L1 --body "Comment"
    AIT_LOCAL_MODE=full gh issue reopen L1

Supported commands:
    - gh issue create: Create local issue (.issues/L*.md)
    - gh issue list: List local issues
    - gh issue view: View local issue details
    - gh issue edit: Edit local issue (labels, title, body)
    - gh issue close: Close local issue
    - gh issue comment: Add comment to local issue
    - gh issue reopen: Reopen closed local issue

Non-issue commands fall through to real gh (with warning about local mode).
"""

from __future__ import annotations

__all__ = [
    "main",
    "handle_issue_command",
    "is_full_local_mode",
]

import json
import os
import sys
from pathlib import Path

# Add parent dir to path for imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts.local_issue_store import (
    LocalIssue,
    LocalIssueStore,
)


def is_full_local_mode() -> bool:
    """Check if full local mode is enabled (AIT_LOCAL_MODE=full).

    Full local mode routes ALL gh issue commands to local storage.
    Regular local mode (AIT_LOCAL_MODE=1) just returns stale cache.

    REQUIRES: None (reads environment variable)
    ENSURES: returns bool; True iff AIT_LOCAL_MODE environment variable equals "full"

    Returns:
        True if AIT_LOCAL_MODE=full, False otherwise.
    """
    return os.environ.get("AIT_LOCAL_MODE") == "full"


def _find_real_gh() -> str:
    """Find the real gh CLI binary.

    Returns:
        Path to real gh binary.

    Raises:
        FileNotFoundError: If gh not found.
    """
    for loc in ["/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh"]:
        if os.path.isfile(loc) and os.access(loc, os.X_OK):
            return loc
    raise FileNotFoundError("Real gh CLI not found")


def _parse_issue_args(args: list[str]) -> dict[str, str | list[str] | None]:
    """Parse gh issue command arguments.

    Args:
        args: Command line arguments after 'gh issue'.

    Returns:
        Dict with parsed arguments:
        - subcommand: create, list, view, edit, close, comment, reopen
        - title: Issue title
        - body: Issue body
        - labels: List of labels
        - add_labels: Labels to add
        - remove_labels: Labels to remove
        - issue_id: Issue ID/number
        - state: Issue state filter
        - json_fields: JSON output fields
        - limit: Result limit
    """
    result: dict[str, str | list[str] | None] = {
        "subcommand": None,
        "title": None,
        "body": None,
        "labels": [],
        "add_labels": [],
        "remove_labels": [],
        "issue_id": None,
        "state": "open",
        "json_fields": None,
        "limit": None,
    }

    i = 0
    positional_after_subcommand = False

    while i < len(args):
        arg = args[i]

        # First non-flag is subcommand
        if result["subcommand"] is None and not arg.startswith("-"):
            result["subcommand"] = arg
            positional_after_subcommand = True
            i += 1
            continue

        # Second non-flag is issue_id (for view, edit, close, comment, reopen)
        if (
            positional_after_subcommand
            and not arg.startswith("-")
            and result["issue_id"] is None
        ):
            result["issue_id"] = arg
            i += 1
            continue

        # Parse flags
        if arg in ("--title", "-t") and i + 1 < len(args):
            result["title"] = args[i + 1]
            i += 2
            continue
        if arg.startswith("--title="):
            result["title"] = arg[8:]
            i += 1
            continue

        if arg in ("--body", "-b") and i + 1 < len(args):
            result["body"] = args[i + 1]
            i += 2
            continue
        if arg.startswith("--body="):
            result["body"] = arg[7:]
            i += 1
            continue

        if arg in ("--label", "-l") and i + 1 < len(args):
            labels = result["labels"]
            if isinstance(labels, list):
                labels.append(args[i + 1])
            i += 2
            continue
        if arg.startswith("--label="):
            labels = result["labels"]
            if isinstance(labels, list):
                labels.append(arg[8:])
            i += 1
            continue

        if arg == "--add-label" and i + 1 < len(args):
            add_labels = result["add_labels"]
            if isinstance(add_labels, list):
                add_labels.append(args[i + 1])
            i += 2
            continue
        if arg.startswith("--add-label="):
            add_labels = result["add_labels"]
            if isinstance(add_labels, list):
                add_labels.append(arg[12:])
            i += 1
            continue

        if arg == "--remove-label" and i + 1 < len(args):
            remove_labels = result["remove_labels"]
            if isinstance(remove_labels, list):
                remove_labels.append(args[i + 1])
            i += 2
            continue
        if arg.startswith("--remove-label="):
            remove_labels = result["remove_labels"]
            if isinstance(remove_labels, list):
                remove_labels.append(arg[15:])
            i += 1
            continue

        if arg in ("--state", "-s") and i + 1 < len(args):
            result["state"] = args[i + 1]
            i += 2
            continue
        if arg.startswith("--state="):
            result["state"] = arg[8:]
            i += 1
            continue

        if arg == "--json" and i + 1 < len(args):
            result["json_fields"] = args[i + 1]
            i += 2
            continue
        if arg.startswith("--json="):
            result["json_fields"] = arg[7:]
            i += 1
            continue

        if arg in ("--limit", "-L") and i + 1 < len(args):
            result["limit"] = args[i + 1]
            i += 2
            continue
        if arg.startswith("--limit="):
            result["limit"] = arg[8:]
            i += 1
            continue

        i += 1

    return result


def _format_issue_line(issue: LocalIssue) -> str:
    """Format issue for list output.

    Args:
        issue: LocalIssue instance.

    Returns:
        Formatted string: "#L1  title  [labels]"
    """
    labels_str = ", ".join(issue.labels) if issue.labels else ""
    return f"#{issue.id}\t{issue.title}\t{labels_str}"


def _format_issue_detail(issue: LocalIssue) -> str:
    """Format issue for view output.

    Args:
        issue: LocalIssue instance.

    Returns:
        Detailed issue view string.
    """
    labels_str = ", ".join(issue.labels) if issue.labels else "none"
    lines = [
        f"title:\t{issue.title}",
        f"state:\t{issue.state.upper()}",
        f"labels:\t{labels_str}",
        f"created:\t{issue.created_at}",
        f"updated:\t{issue.updated_at}",
        "",
        issue.body or "(no description)",
    ]

    if issue.comments:
        lines.append("")
        lines.append("---")
        lines.append(f"Comments ({len(issue.comments)}):")
        for comment in issue.comments:
            lines.append(f"\n{comment.author} @ {comment.timestamp}:")
            lines.append(comment.body)

    return "\n".join(lines)


def handle_issue_create(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue create' command.

    REQUIRES: store is valid LocalIssueStore; parsed contains "title" key
    ENSURES: returns 0 and creates issue if title provided; returns 1 if title missing

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    title = parsed.get("title")
    if not title:
        print("error: --title is required", file=sys.stderr)
        return 1

    body = parsed.get("body") or ""
    labels = parsed.get("labels") or []

    issue = store.create(title=title, body=body, labels=labels)
    print(f"Created issue #{issue.id}", file=sys.stderr)

    # Output URL-like format for compatibility (local path instead of GitHub URL)
    issues_path = store.issues_dir / f"{issue.id}.md"
    print(issues_path)

    return 0


def handle_issue_list(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue list' command.

    REQUIRES: store is valid LocalIssueStore; parsed is dict with optional state/labels
    ENSURES: returns 0; prints issues as text or JSON depending on json_fields

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success).
    """
    state = str(parsed.get("state", "open"))
    labels = parsed.get("labels") or []
    json_fields = parsed.get("json_fields")
    limit_str = parsed.get("limit")

    # Map state values
    if state in ("all", "open", "closed"):
        issues = store.list_issues(state=state, labels=labels if labels else None)
    else:
        issues = store.list_issues(state="open", labels=labels if labels else None)

    # Apply limit
    if limit_str:
        try:
            limit = int(limit_str)
            issues = issues[:limit]
        except ValueError:
            pass

    # JSON output
    if json_fields:
        output = [issue.to_dict() for issue in issues]
        print(json.dumps(output, indent=2))
        return 0

    # Plain text output
    if not issues:
        print("No issues match your search", file=sys.stderr)
        return 0

    for issue in issues:
        print(_format_issue_line(issue))

    return 0


def handle_issue_view(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue view' command.

    REQUIRES: store is valid LocalIssueStore; parsed contains "issue_id"
    ENSURES: returns 0 if issue found and printed; returns 1 if issue_id missing or not found

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    issue_id = parsed.get("issue_id")
    if not issue_id:
        print("error: issue number required", file=sys.stderr)
        return 1

    issue = store.view(str(issue_id))
    if issue is None:
        print(f"error: issue {issue_id} not found", file=sys.stderr)
        return 1

    json_fields = parsed.get("json_fields")
    if json_fields:
        print(json.dumps(issue.to_dict(), indent=2))
        return 0

    print(_format_issue_detail(issue))
    return 0


def handle_issue_edit(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue edit' command.

    REQUIRES: store is valid LocalIssueStore; parsed contains "issue_id"
    ENSURES: returns 0 if issue found and edited; returns 1 if issue_id missing or not found

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    issue_id = parsed.get("issue_id")
    if not issue_id:
        print("error: issue number required", file=sys.stderr)
        return 1

    title = parsed.get("title")
    body = parsed.get("body")
    add_labels = parsed.get("add_labels") or []
    remove_labels = parsed.get("remove_labels") or []

    issue = store.edit(
        str(issue_id),
        title=title,
        body=body,
        add_labels=add_labels if add_labels else None,
        remove_labels=remove_labels if remove_labels else None,
    )

    if issue is None:
        print(f"error: issue {issue_id} not found", file=sys.stderr)
        return 1

    print(f"Edited issue #{issue.id}", file=sys.stderr)
    return 0


def handle_issue_close(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue close' command.

    REQUIRES: store is valid LocalIssueStore; parsed contains "issue_id"
    ENSURES: returns 0 if issue found and closed; returns 1 if issue_id missing or not found

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    issue_id = parsed.get("issue_id")
    if not issue_id:
        print("error: issue number required", file=sys.stderr)
        return 1

    issue = store.close(str(issue_id))
    if issue is None:
        print(f"error: issue {issue_id} not found", file=sys.stderr)
        return 1

    print(f"Closed issue #{issue.id}", file=sys.stderr)
    return 0


def handle_issue_reopen(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue reopen' command.

    REQUIRES: store is valid LocalIssueStore; parsed contains "issue_id"
    ENSURES: returns 0 if issue found and reopened; returns 1 if issue_id missing or not found

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    issue_id = parsed.get("issue_id")
    if not issue_id:
        print("error: issue number required", file=sys.stderr)
        return 1

    issue = store.reopen(str(issue_id))
    if issue is None:
        print(f"error: issue {issue_id} not found", file=sys.stderr)
        return 1

    print(f"Reopened issue #{issue.id}", file=sys.stderr)
    return 0


def handle_issue_comment(store: LocalIssueStore, parsed: dict) -> int:
    """Handle 'gh issue comment' command.

    REQUIRES: store is valid LocalIssueStore; parsed contains "issue_id" and "body"
    ENSURES: returns 0 if comment added; returns 1 if issue_id/body missing or issue not found

    Args:
        store: LocalIssueStore instance.
        parsed: Parsed arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    issue_id = parsed.get("issue_id")
    if not issue_id:
        print("error: issue number required", file=sys.stderr)
        return 1

    body = parsed.get("body")
    if not body:
        print("error: --body is required", file=sys.stderr)
        return 1

    issue = store.comment(str(issue_id), body=body)
    if issue is None:
        print(f"error: issue {issue_id} not found", file=sys.stderr)
        return 1

    print(f"Added comment to issue #{issue.id}", file=sys.stderr)
    return 0


def handle_issue_command(args: list[str]) -> int:
    """Handle gh issue subcommand locally.

    REQUIRES: args is list of strings (may be empty)
    ENSURES: returns exit code; routes to appropriate handler or returns 1 for unknown subcommand

    Args:
        args: Arguments after 'gh issue'.

    Returns:
        Exit code.
    """
    store = LocalIssueStore()
    parsed = _parse_issue_args(args)
    subcommand = parsed.get("subcommand")

    if subcommand == "create":
        return handle_issue_create(store, parsed)
    if subcommand == "list":
        return handle_issue_list(store, parsed)
    if subcommand == "view":
        return handle_issue_view(store, parsed)
    if subcommand == "edit":
        return handle_issue_edit(store, parsed)
    if subcommand == "close":
        return handle_issue_close(store, parsed)
    if subcommand == "reopen":
        return handle_issue_reopen(store, parsed)
    if subcommand == "comment":
        return handle_issue_comment(store, parsed)
    print(f"error: unknown subcommand '{subcommand}'", file=sys.stderr)
    print(
        "supported: create, list, view, edit, close, reopen, comment",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    """Entry point for gh_local handler.

    Routes gh issue commands to local storage.
    Non-issue commands fall through to real gh with warning.

    REQUIRES: sys.argv contains command line arguments
    ENSURES: returns exit code; handles 'issue' commands locally, others fall through to real gh

    Returns:
        Exit code.
    """
    args = sys.argv[1:]

    if not args:
        print("error: no command specified", file=sys.stderr)
        return 1

    # Check if this is an issue command
    if args[0] == "issue":
        return handle_issue_command(args[1:])

    # Non-issue commands: warn and fall through to real gh
    print("warning: local mode only handles 'gh issue' commands", file=sys.stderr)
    print("falling through to real gh...", file=sys.stderr)

    try:
        real_gh = _find_real_gh()
        os.execv(real_gh, [real_gh] + args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Sync local issues to GitHub.

Provides bidirectional sync between local .issues/*.md files and GitHub Issues.
Use this when transitioning from local development mode back to online mode.

Commands:
    sync    - Push local issues to GitHub
    bootstrap - Import existing GitHub issues to local storage
    status  - Show sync status

Usage:
    # Preview what would be synced
    python3 sync_local_issues.py sync --dry-run

    # Push local issues to GitHub
    python3 sync_local_issues.py sync

    # Import existing GitHub issues locally
    python3 sync_local_issues.py bootstrap

    # Check sync status
    python3 sync_local_issues.py status
"""

from __future__ import annotations

__all__ = [
    "sync_to_github",
    "bootstrap_from_github",
    "get_sync_status",
]

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add parent dir to path for imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts.local_issue_store import (
    LocalIssueStore,
)


def _run_gh(args: list[str], timeout: int = 30) -> tuple[bool, str]:
    """Run gh CLI command.

    Args:
        args: Command arguments (without 'gh').
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, output/error).
    """
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, "BYPASS_WRAPPER": "1"},
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def _get_real_gh() -> str | None:
    """Find real gh CLI path (bypassing wrapper).

    Returns:
        Path to real gh binary, or None if not found.
    """
    for loc in ["/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh"]:
        if Path(loc).is_file():
            return loc
    return None


def sync_to_github(
    store: LocalIssueStore,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, list[str]]:
    """Sync local issues to GitHub.

    Creates new GitHub issues for each local issue, preserving:
    - Title and body
    - Labels
    - Comments (appended to body)

    Args:
        store: LocalIssueStore instance.
        dry_run: If True, preview without making changes.
        verbose: If True, print detailed output.

    Returns:
        Dict with 'synced', 'failed', 'skipped' issue IDs.
    """
    result = {"synced": [], "failed": [], "skipped": []}

    issues = store.list_issues(state="all")
    if not issues:
        if verbose:
            print("No local issues to sync")
        return result

    for issue in issues:
        if verbose:
            print(f"\nProcessing {issue.id}: {issue.title}")

        # Build issue body with comments
        body = issue.body or ""
        if issue.comments:
            body += "\n\n---\n## Local Comments\n"
            for comment in issue.comments:
                body += (
                    f"\n### {comment.author} @ {comment.timestamp}\n{comment.body}\n"
                )

        # Prepare gh command
        gh_args = [
            "issue",
            "create",
            "--title",
            issue.title,
            "--body",
            body,
        ]
        for label in issue.labels:
            gh_args.extend(["--label", label])

        if dry_run:
            if verbose:
                print(f"  Would create: {issue.title}")
                print(f"  Labels: {', '.join(issue.labels) or 'none'}")
            result["synced"].append(issue.id)
            continue

        # Create GitHub issue
        success, output = _run_gh(gh_args)
        if success:
            # Extract issue number from URL
            gh_url = output.strip()
            if verbose:
                print(f"  Created: {gh_url}")
            result["synced"].append(issue.id)

            # If issue was closed locally, close on GitHub too
            if issue.state == "closed":
                # Extract issue number from URL
                gh_num = gh_url.rstrip("/").split("/")[-1]
                _run_gh(["issue", "close", gh_num])
                if verbose:
                    print(f"  Closed: #{gh_num}")
        else:
            if verbose:
                print(f"  Failed: {output}")
            result["failed"].append(issue.id)

    return result


def bootstrap_from_github(
    store: LocalIssueStore,
    limit: int = 100,
    state: str = "open",
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, list[int]]:
    """Import existing GitHub issues to local storage.

    Downloads issues from GitHub and creates local .issues/*.md files.
    Useful for initializing local development mode from existing issues.

    Args:
        store: LocalIssueStore instance.
        limit: Maximum issues to import.
        state: Issue state filter ("open", "closed", "all").
        dry_run: If True, preview without making changes.
        verbose: If True, print detailed output.

    Returns:
        Dict with 'imported', 'failed', 'skipped' GitHub issue numbers.
    """
    result = {"imported": [], "failed": [], "skipped": []}

    # Fetch issues from GitHub
    success, output = _run_gh(
        [
            "issue",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,body,labels,state,createdAt",
        ]
    )

    if not success:
        if verbose:
            print(f"Failed to fetch issues: {output}")
        return result

    try:
        gh_issues = json.loads(output)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse issues: {e}")
        return result

    for gh_issue in gh_issues:
        number = gh_issue.get("number")
        title = gh_issue.get("title", "")
        body = gh_issue.get("body") or ""
        labels_raw = gh_issue.get("labels", [])
        state_raw = gh_issue.get("state", "OPEN")
        created_at = gh_issue.get("createdAt", "")

        # Parse labels (handle both object and string formats)
        labels = []
        for lbl in labels_raw:
            if isinstance(lbl, dict):
                labels.append(lbl.get("name", ""))
            else:
                labels.append(str(lbl))

        # Map state
        issue_state = "open" if state_raw.upper() == "OPEN" else "closed"

        if verbose:
            print(f"\nProcessing #{number}: {title}")

        if dry_run:
            if verbose:
                print(f"  Would import: {title}")
                print(f"  Labels: {', '.join(labels) or 'none'}")
                print(f"  State: {issue_state}")
            result["imported"].append(number)
            continue

        # Create local issue
        try:
            local_issue = store.create(
                title=title,
                body=f"<!-- Imported from GitHub #{number} -->\n\n{body}",
                labels=labels,
            )

            # Set state and timestamps
            local_issue.state = issue_state
            local_issue.created_at = created_at or local_issue.created_at

            # Write updated issue
            store._write_issue(local_issue)

            if verbose:
                print(f"  Imported as: {local_issue.id}")
            result["imported"].append(number)

        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
            result["failed"].append(number)

    return result


def get_sync_status(store: LocalIssueStore) -> dict[str, int | list[str]]:
    """Get sync status summary.

    Args:
        store: LocalIssueStore instance.

    Returns:
        Dict with status counts and issue lists.
    """
    issues = store.list_issues(state="all")
    open_issues = [i for i in issues if i.state == "open"]
    closed_issues = [i for i in issues if i.state == "closed"]

    return {
        "total": len(issues),
        "open": len(open_issues),
        "closed": len(closed_issues),
        "open_ids": [i.id for i in open_issues],
        "closed_ids": [i.id for i in closed_issues],
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync local issues to/from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Push local issues to GitHub")
    sync_parser.add_argument(
        "--dry-run", action="store_true", help="Preview without changes"
    )
    sync_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # bootstrap command
    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="Import GitHub issues locally"
    )
    bootstrap_parser.add_argument(
        "--limit", type=int, default=100, help="Max issues to import"
    )
    bootstrap_parser.add_argument(
        "--state", default="open", choices=["open", "closed", "all"]
    )
    bootstrap_parser.add_argument(
        "--dry-run", action="store_true", help="Preview without changes"
    )
    bootstrap_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # status command
    subparsers.add_parser("status", help="Show sync status")

    args = parser.parse_args()
    store = LocalIssueStore()

    if args.command == "sync":
        result = sync_to_github(store, dry_run=args.dry_run, verbose=args.verbose)
        print(f"\nSynced: {len(result['synced'])}")
        print(f"Failed: {len(result['failed'])}")
        print(f"Skipped: {len(result['skipped'])}")
        if result["failed"]:
            print(f"Failed issues: {', '.join(result['failed'])}")
        return 0 if not result["failed"] else 1

    if args.command == "bootstrap":
        result = bootstrap_from_github(
            store,
            limit=args.limit,
            state=args.state,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        print(f"\nImported: {len(result['imported'])}")
        print(f"Failed: {len(result['failed'])}")
        print(f"Skipped: {len(result['skipped'])}")
        if result["failed"]:
            print(f"Failed issues: {', '.join(map(str, result['failed']))}")
        return 0 if not result["failed"] else 1

    if args.command == "status":
        status = get_sync_status(store)
        print(f"Local Issues: {status['total']}")
        print(f"  Open: {status['open']}")
        print(f"  Closed: {status['closed']}")
        if status["open_ids"]:
            print(f"\nOpen issues: {', '.join(status['open_ids'])}")
        if status["closed_ids"]:
            print(f"Closed issues: {', '.join(status['closed_ids'])}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

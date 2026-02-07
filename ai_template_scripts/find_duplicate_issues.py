#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Find and report duplicate child issues created from checkbox conversion.

PURPOSE: Detects issues with identical looper-child hash markers.
CALLED BY: Manager audit, human debugging.

Duplicates occur when multiple sessions convert the same checkbox on different
machines before GitHub search indexes the first issue (30-60s lag).

Usage:
    python3 ai_template_scripts/find_duplicate_issues.py [--close]

Options:
    --close    Close duplicate issues (keeps oldest, closes newer)
    --dry-run  Show what would be closed without making changes
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ChildIssue:
    """A child issue created from checkbox conversion."""

    number: int
    title: str
    parent_num: int
    hash_marker: str
    created_at: str
    state: str
    url: str


def run_gh(args: list[str], timeout: int = 30) -> str | None:
    """Run gh command and return stdout, or None on error."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def find_child_issues() -> list[ChildIssue]:
    """Find all issues with looper-child markers."""
    # Search for issues with looper-child markers in body
    output = run_gh(
        [
            "issue",
            "list",
            "--state",
            "all",
            "--search",
            "looper-child in:body",
            "--json",
            "number,title,body,createdAt,state,url",
            "--limit",
            "500",
        ],
        timeout=60,
    )
    if not output:
        return []

    issues = []
    try:
        data = json.loads(output)
        for item in data:
            body = item.get("body") or ""
            # Extract looper-child:PARENT:HASH marker
            match = re.search(r"looper-child:(\d+):([a-f0-9]{8})", body)
            if match:
                issues.append(
                    ChildIssue(
                        number=item["number"],
                        title=item["title"],
                        parent_num=int(match.group(1)),
                        hash_marker=match.group(2),
                        created_at=item["createdAt"],
                        state=item["state"],
                        url=item.get("url", ""),
                    )
                )
    except (json.JSONDecodeError, KeyError):
        pass

    return issues


def find_duplicates(issues: list[ChildIssue]) -> dict[str, list[ChildIssue]]:
    """Group issues by their hash marker to find duplicates.

    Returns:
        Dict mapping hash key (parent:hash) to list of issues with that key.
        Only includes entries with 2+ issues (actual duplicates).
    """
    # Group by parent:hash key
    by_key: dict[str, list[ChildIssue]] = defaultdict(list)
    for issue in issues:
        key = f"{issue.parent_num}:{issue.hash_marker}"
        by_key[key].append(issue)

    # Filter to only duplicates (2+ issues with same key)
    return {k: v for k, v in by_key.items() if len(v) > 1}


def report_duplicates(duplicates: dict[str, list[ChildIssue]]) -> None:
    """Print a report of duplicate issues."""
    if not duplicates:
        print("No duplicate issues found.")
        return

    total_dupes = sum(len(v) - 1 for v in duplicates.values())
    print(f"Found {total_dupes} duplicate issues across {len(duplicates)} hash markers:\n")

    for key, issues in sorted(duplicates.items()):
        # Sort by creation time to identify which is oldest
        issues_sorted = sorted(issues, key=lambda x: x.created_at)
        parent_num = issues_sorted[0].parent_num
        print(f"Parent #{parent_num} - Hash {key.split(':')[1]}:")
        print(f"  Title: {issues_sorted[0].title[:60]}...")

        for i, issue in enumerate(issues_sorted):
            marker = "KEEP" if i == 0 else "DUP "
            state_marker = "OPEN" if issue.state == "OPEN" else "CLOSED"
            print(
                f"  [{marker}] #{issue.number} ({state_marker}) - {issue.created_at[:10]}"
            )
        print()


def close_duplicates(
    duplicates: dict[str, list[ChildIssue]], dry_run: bool = False
) -> int:
    """Close duplicate issues, keeping the oldest.

    Returns:
        Count of issues closed (or would be closed in dry_run mode).
    """
    closed = 0
    for key, issues in duplicates.items():
        # Sort by creation time - keep oldest
        issues_sorted = sorted(issues, key=lambda x: x.created_at)

        # Close all except first (oldest)
        for issue in issues_sorted[1:]:
            if issue.state != "OPEN":
                continue  # Already closed

            if dry_run:
                print(f"Would close #{issue.number}: {issue.title[:50]}...")
                closed += 1
            else:
                comment = (
                    "Closing as duplicate. This issue was created due to a race condition "
                    f"in checkbox-to-issue conversion. See #{issues_sorted[0].number} for "
                    "the original issue.\n\n"
                    f"Duplicate of #{issues_sorted[0].number}"
                )
                # Add comment explaining closure
                run_gh(["issue", "comment", str(issue.number), "--body", comment])
                # Add duplicate label and close
                run_gh(
                    [
                        "issue",
                        "edit",
                        str(issue.number),
                        "--add-label",
                        "duplicate",
                    ]
                )
                result = run_gh(["issue", "close", str(issue.number)])
                if result is not None:
                    print(f"Closed #{issue.number}: {issue.title[:50]}...")
                    closed += 1
                else:
                    print(f"Failed to close #{issue.number}")

    return closed


def main() -> int:
    """Main entry point.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Find and report duplicate child issues from checkbox conversion."
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close duplicate issues (keeps oldest, closes newer)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be closed without making changes",
    )
    args = parser.parse_args()

    print("Searching for child issues with looper-child markers...")
    issues = find_child_issues()
    print(f"Found {len(issues)} child issues.\n")

    duplicates = find_duplicates(issues)
    report_duplicates(duplicates)

    if args.close or args.dry_run:
        closed = close_duplicates(duplicates, dry_run=args.dry_run)
        action = "Would close" if args.dry_run else "Closed"
        print(f"\n{action} {closed} duplicate issues.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

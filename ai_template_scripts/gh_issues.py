#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""GitHub Issues dependencies helper and workflow sanitizer.

NOTE: GitHub does NOT have a native issue dependency API.
For dependencies, use `Blocked: #N` text in issue bodies instead.
See ai_template.md "Dependencies & Blockers" section.

NOTE: Do NOT use gh issue/repo view -q or --jq - has caching bugs in v2.83.2+ (#1047).
Always pipe to external jq or parse JSON in Python instead.

Public API:
- run_gh, get_repo, get_issue_id
- list_dependencies
- detect_contradictory_labels, sanitize_workflow_labels
- main

Usage:
    gh_issues.py dep list ISSUE          # List tracked/tracking relationships
    gh_issues.py sanitize                 # Detect contradictory workflow labels
    gh_issues.py sanitize --fix           # Auto-fix contradictory labels

Issue numbers are local to current repo (e.g., 42, not full URL).
"""

import importlib
import json
import sys
from pathlib import Path
from typing import NoReturn

# Add parent dir to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts.gh_issue_numbers import (  # noqa: E402
    parse_issue_number,
)
from ai_template_scripts.subprocess_utils import (  # noqa: E402
    get_github_repo,
    run_cmd,
)

# Import gh_graphql module (handles both package and direct script execution)
try:
    _gh_graphql_module = importlib.import_module("ai_template_scripts.gh_graphql")
except ModuleNotFoundError:
    _gh_graphql_module = importlib.import_module("gh_graphql")  # noqa: E402

graphql = _gh_graphql_module.graphql

__all__ = [
    "run_gh",
    "get_repo",
    "get_issue_id",
    "list_dependencies",
    "detect_contradictory_labels",
    "sanitize_workflow_labels",
    "main",
]

# Workflow labels that are mutually exclusive
WORKFLOW_LABELS = {"do-audit", "needs-review"}
IN_PROGRESS_PREFIX = "in-progress"


def run_gh(args: list[str]) -> tuple[int, str]:
    """Run gh command and return exit code and output.

    Rate limiting and caching handled by bin/gh wrapper.
    Returns stdout on success, stderr on failure.

    REQUIRES: args is list of strings (gh subcommand and arguments)
    ENSURES: Returns (0, stdout) on success, (code, stderr) on failure
    """
    result = run_cmd(["gh"] + args, timeout=60)
    if result.ok:
        return 0, result.stdout
    return result.returncode, result.stderr or result.error or ""


def get_repo() -> str:
    """Get current repo in owner/name format.

    Uses canonical get_github_repo() from subprocess_utils.
    Exits with error if not in a GitHub repository.

    REQUIRES: In a git repo with GitHub remote
    ENSURES: Returns "owner/name" string, exits on failure
    """
    result = get_github_repo()
    if not result.ok:
        print(f"Error: {result.error or 'Not in a GitHub repo'}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def _parse_repo(repo: str) -> tuple[str, str]:
    """Parse owner/name repo format.

    Args:
        repo: Repository string in "owner/name" format.

    Returns:
        Tuple of (owner, name).

    Exits:
        With code 1 if repo format is invalid.
    """
    if "/" not in repo:
        print(f"Error: Invalid repo format: {repo}", file=sys.stderr)
        sys.exit(1)
    parts = repo.split("/", 1)  # Use maxsplit=1 to handle names with slashes
    owner, name = parts[0], parts[1]
    if not owner or not name:
        print(f"Error: Invalid repo format: {repo}", file=sys.stderr)
        sys.exit(1)
    return (owner, name)


def get_issue_id(repo: str, number: int) -> str:
    """Get GraphQL node ID for an issue.

    REQUIRES: repo is "owner/name" format string
    REQUIRES: number > 0
    ENSURES: Returns issue node ID string, exits on failure
    """
    owner, name = _parse_repo(repo)
    query = """
    query($owner: String!, $name: String!, $number: Int!) {
        repository(owner: $owner, name: $name) {
            issue(number: $number) {
                id
            }
        }
    }
    """
    result = graphql(query, variables={"owner": owner, "name": name, "number": number})
    if not result.ok:
        errors = result.errors or []
        error_msg = errors[0].get("message", result.stderr) if errors else result.stderr
        print(f"Error getting issue #{number}: {error_msg}", file=sys.stderr)
        sys.exit(1)
    issue_id = result.extract(".repository.issue.id")
    if not issue_id:
        print(f"Error: Issue #{number} not found", file=sys.stderr)
        sys.exit(1)
    return issue_id


def add_dependency(issue: int, blocker: int) -> NoReturn:
    """Add blocker as a dependency - DISABLED (API doesn't exist)."""
    print("ERROR: GitHub does not have an issue dependency API.", file=sys.stderr)
    print(file=sys.stderr)
    print("Instead, add 'Blocked: #N' to the issue body:", file=sys.stderr)
    # NOTE: Do NOT use gh -q or --jq - has caching bugs in v2.83.2+ (#1047)
    print(
        f'  gh issue edit {issue} --body "$(gh issue view {issue} '
        f"--json body | jq -r .body)",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(f'Blocked: #{blocker}"', file=sys.stderr)
    print(file=sys.stderr)
    print("Or edit the issue in your browser.", file=sys.stderr)
    sys.exit(1)


def remove_dependency(issue: int, blocker: int) -> NoReturn:
    """Remove dependency relationship - DISABLED (API doesn't exist)."""
    print("ERROR: GitHub does not have an issue dependency API.", file=sys.stderr)
    print(file=sys.stderr)
    print("Instead, edit the issue body to remove 'Blocked: #N':", file=sys.stderr)
    print(f'  gh issue edit {issue} --body "..."', file=sys.stderr)
    print(file=sys.stderr)
    print("Or edit the issue in your browser.", file=sys.stderr)
    sys.exit(1)


def list_dependencies(issue: int) -> None:
    """List what blocks an issue and what it blocks.

    REQUIRES: issue > 0
    ENSURES: Prints blocking/blocked-by relationships to stdout
    """
    repo = get_repo()
    owner, name = _parse_repo(repo)
    query = """
    query($owner: String!, $name: String!, $number: Int!) {
        repository(owner: $owner, name: $name) {
            issue(number: $number) {
                trackedInIssues(first: 50) {
                    nodes {
                        number
                        title
                        state
                    }
                }
                trackedIssues(first: 50) {
                    nodes {
                        number
                        title
                        state
                    }
                }
            }
        }
    }
    """
    result = graphql(query, variables={"owner": owner, "name": name, "number": issue})
    if not result.ok:
        errors = result.errors or []
        error_msg = errors[0].get("message", result.stderr) if errors else result.stderr
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    issue_data = result.extract(".repository.issue")
    if issue_data is None:
        print(f"Error: Issue #{issue} not found", file=sys.stderr)
        sys.exit(1)

    # Handle case where trackedInIssues/trackedIssues or nodes is None instead of missing
    tracked_in = issue_data.get("trackedInIssues") or {}
    tracked = issue_data.get("trackedIssues") or {}
    blocked_by = tracked_in.get("nodes") or []
    blocking = tracked.get("nodes") or []

    if blocked_by:
        print(f"#{issue} is blocked by:")
        for b in blocked_by:
            state = "✓" if b["state"] == "CLOSED" else "○"
            print(f"  {state} #{b['number']}: {b['title']}")
    else:
        print(f"#{issue} has no blockers")

    if blocking:
        print(f"#{issue} is blocking:")
        for b in blocking:
            state = "✓" if b["state"] == "CLOSED" else "○"
            print(f"  {state} #{b['number']}: {b['title']}")


def _is_in_progress_label(label: str) -> bool:
    """Check if label is any form of in-progress."""
    return label == IN_PROGRESS_PREFIX or label.startswith(f"{IN_PROGRESS_PREFIX}-")


def detect_contradictory_labels(
    issues: list[dict],
) -> list[tuple[int, str, list[str], str]]:
    """Detect issues with contradictory workflow labels.

    REQUIRES: issues is list of dicts with 'number', 'title', 'labels' keys
    ENSURES: Returns list of (number, title, labels_to_remove, reason) tuples

    Args:
        issues: List of issue dicts with 'number', 'title', 'labels' keys.
                Labels can be strings or dicts with 'name' key.

    Returns:
        List of (issue_number, title, labels_to_remove, reason) tuples.
    """
    results: list[tuple[int, str, list[str], str]] = []

    for issue in issues:
        number = parse_issue_number(issue.get("number"))
        if number is None:
            continue  # Skip issues without number
        title: str = issue.get("title", "")
        raw_labels = issue.get("labels", [])

        # Normalize labels to strings (filter out empty names)
        labels = set()
        for lbl in raw_labels:
            if isinstance(lbl, dict):
                name = lbl.get("name", "")
                if name:  # Skip empty label names
                    labels.add(name)
            elif lbl:  # Skip empty string labels
                labels.add(lbl)

        # Find in-progress labels
        in_progress_labels = {lbl for lbl in labels if _is_in_progress_label(lbl)}

        # Find workflow labels
        workflow_labels = labels & WORKFLOW_LABELS

        # Check for contradictions
        # Rule 1: do-audit + in-progress is invalid (do-audit means work complete)
        if "do-audit" in workflow_labels and in_progress_labels:
            results.append(
                (
                    number,
                    title,
                    list(in_progress_labels),
                    "do-audit + in-progress (do-audit means work complete)",
                )
            )

        # Rule 2: needs-review + in-progress is invalid (needs-review means awaiting manager)
        elif "needs-review" in workflow_labels and in_progress_labels:
            results.append(
                (
                    number,
                    title,
                    list(in_progress_labels),
                    "needs-review + in-progress (needs-review means awaiting manager)",
                )
            )

        # Rule 3: do-audit + needs-review is invalid (sequential states)
        elif "do-audit" in workflow_labels and "needs-review" in workflow_labels:
            # Prefer needs-review as it's further along the workflow
            results.append(
                (
                    number,
                    title,
                    ["do-audit"],
                    "do-audit + needs-review (sequential states, prefer needs-review)",
                )
            )

    return results


def sanitize_workflow_labels(fix: bool = False) -> int:
    """Detect and optionally fix contradictory workflow labels.

    REQUIRES: None (reads from GitHub API)
    ENSURES: Returns count of contradictory issues (0 if none)
    ENSURES: If fix=True, removes invalid labels

    Args:
        fix: If True, auto-remove the invalid labels. If False, just report.

    Returns:
        Number of issues with contradictions (exit code).
    """
    # Get all open issues with their labels
    code, output = run_gh(
        [
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,title,labels",
        ]
    )
    if code != 0:
        print(f"Error listing issues: {output}", file=sys.stderr)
        sys.exit(1)

    try:
        issues = json.loads(output.strip())
    except json.JSONDecodeError:
        print(f"Error parsing issue list: {output}", file=sys.stderr)
        sys.exit(1)

    contradictions = detect_contradictory_labels(issues)

    if not contradictions:
        print("✓ No contradictory workflow labels found")
        return 0

    print(f"Found {len(contradictions)} issue(s) with contradictory labels:\n")

    for number, title, to_remove, reason in contradictions:
        print(f"#{number}: {title}")
        print(f"  Issue: {reason}")
        print(f"  Labels to remove: {', '.join(to_remove)}")

        if fix:
            for label in to_remove:
                fix_code, fix_output = run_gh(
                    ["issue", "edit", str(number), "--remove-label", label]
                )
                if fix_code == 0:
                    print(f"  ✓ Removed '{label}'")
                else:
                    print(f"  ✗ Failed to remove '{label}': {fix_output}")
        print()

    if not fix:
        print("Run with --fix to auto-remove contradictory labels.")

    return len(contradictions)


def _parse_issue_number(arg: str, arg_name: str = "issue number") -> int:
    """Parse a command-line argument as an issue number.

    Args:
        arg: The argument string to parse.
        arg_name: Name for error messages (e.g., "issue number", "parent issue").

    Returns:
        The parsed issue number (positive integer).

    Exits:
        With code 1 if the argument is not a valid positive integer.
    """
    try:
        num = int(arg)
    except ValueError:
        print(f"Error: '{arg}' is not a valid {arg_name}", file=sys.stderr)
        sys.exit(1)
    if num < 1:
        print(f"Error: '{arg}' is not a valid {arg_name}", file=sys.stderr)
        sys.exit(1)
    return num


def main() -> None:
    """Entry point: manage issue dependencies and workflow labels.

    REQUIRES: None (reads sys.argv)
    ENSURES: Exits with 0 on success, 1 on error
    """
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    # Handle --help and -h flags
    if cmd in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    if cmd == "dep" and len(sys.argv) >= 3:
        subcmd = sys.argv[2]
        if subcmd == "add" and len(sys.argv) == 5:
            parent = _parse_issue_number(sys.argv[3], "parent issue")
            child = _parse_issue_number(sys.argv[4], "child issue")
            add_dependency(parent, child)
        elif subcmd == "remove" and len(sys.argv) == 5:
            parent = _parse_issue_number(sys.argv[3], "parent issue")
            child = _parse_issue_number(sys.argv[4], "child issue")
            remove_dependency(parent, child)
        elif subcmd == "list" and len(sys.argv) == 4:
            issue = _parse_issue_number(sys.argv[3])
            list_dependencies(issue)
        else:
            print("Usage: gh_issues.py dep add|remove|list ...", file=sys.stderr)
            sys.exit(1)
    elif cmd == "sanitize":
        fix = "--fix" in sys.argv
        count = sanitize_workflow_labels(fix=fix)
        sys.exit(count if not fix else 0)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

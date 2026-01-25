#!/usr/bin/env python3
"""GitHub Issues dependencies helper.

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
License: Apache-2.0

NOTE: GitHub does NOT have a native issue dependency API. The `add` and `remove`
commands are disabled because `addIssueDependency`/`removeIssueDependency`
GraphQL mutations do not exist.

For dependencies, use `Blocked: #N` text in issue bodies instead.
See ai_template.md "Dependencies & Blockers" section.

Usage:
    gh_issues.py dep list ISSUE            # List tracked/tracking relationships
    gh_issues.py dep add ISSUE BLOCKER     # DISABLED - use Blocked: #N instead
    gh_issues.py dep remove ISSUE BLOCKER  # DISABLED - edit issue body instead

Issue numbers are local to current repo (e.g., 42, not full URL).
"""

import json
import subprocess
import sys


def run_gh(args: list[str]) -> tuple[int, str]:
    """Run gh command and return exit code and output."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def get_repo() -> str:
    """Get current repo in owner/name format."""
    code, output = run_gh(
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
    )
    if code != 0:
        print("Error: Not in a GitHub repo", file=sys.stderr)
        sys.exit(1)
    return output.strip()


def get_issue_id(repo: str, number: int) -> str:
    """Get GraphQL node ID for an issue."""
    query = f'''
    query {{
        repository(owner: "{repo.split("/")[0]}", name: "{repo.split("/")[1]}") {{
            issue(number: {number}) {{
                id
            }}
        }}
    }}
    '''
    code, output = run_gh(["api", "graphql", "-f", f"query={query}"])
    if code != 0:
        print(f"Error getting issue #{number}: {output}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(output)
    issue_id = data.get("data", {}).get("repository", {}).get("issue", {}).get("id")
    if not issue_id:
        print(f"Error: Issue #{number} not found", file=sys.stderr)
        sys.exit(1)
    return issue_id


def add_dependency(issue: int, blocker: int):
    """Add blocker as a dependency - DISABLED (API doesn't exist)."""
    print("ERROR: GitHub does not have an issue dependency API.", file=sys.stderr)
    print(file=sys.stderr)
    print("Instead, add 'Blocked: #N' to the issue body:", file=sys.stderr)
    print(
        f'  gh issue edit {issue} --body "$(gh issue view {issue} --json body -q .body)',
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(f'Blocked: #{blocker}"', file=sys.stderr)
    print(file=sys.stderr)
    print("Or edit the issue in your browser.", file=sys.stderr)
    sys.exit(1)


def remove_dependency(issue: int, blocker: int):
    """Remove dependency relationship - DISABLED (API doesn't exist)."""
    print("ERROR: GitHub does not have an issue dependency API.", file=sys.stderr)
    print(file=sys.stderr)
    print("Instead, edit the issue body to remove 'Blocked: #N':", file=sys.stderr)
    print(f'  gh issue edit {issue} --body "..."', file=sys.stderr)
    print(file=sys.stderr)
    print("Or edit the issue in your browser.", file=sys.stderr)
    sys.exit(1)


def list_dependencies(issue: int):
    """List what blocks an issue and what it blocks."""
    repo = get_repo()
    query = f'''
    query {{
        repository(owner: "{repo.split("/")[0]}", name: "{repo.split("/")[1]}") {{
            issue(number: {issue}) {{
                trackedInIssues(first: 50) {{
                    nodes {{
                        number
                        title
                        state
                    }}
                }}
                trackedIssues(first: 50) {{
                    nodes {{
                        number
                        title
                        state
                    }}
                }}
            }}
        }}
    }}
    '''
    code, output = run_gh(["api", "graphql", "-f", f"query={query}"])
    if code != 0:
        print(f"Error: {output}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(output)
    issue_data = data.get("data", {}).get("repository", {}).get("issue", {})

    blocked_by = issue_data.get("trackedInIssues", {}).get("nodes", [])
    blocking = issue_data.get("trackedIssues", {}).get("nodes", [])

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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "dep" and len(sys.argv) >= 3:
        subcmd = sys.argv[2]
        if subcmd == "add" and len(sys.argv) == 5:
            add_dependency(int(sys.argv[3]), int(sys.argv[4]))
        elif subcmd == "remove" and len(sys.argv) == 5:
            remove_dependency(int(sys.argv[3]), int(sys.argv[4]))
        elif subcmd == "list" and len(sys.argv) == 4:
            list_dependencies(int(sys.argv[3]))
        else:
            print("Usage: gh_issues.py dep add|remove|list ...", file=sys.stderr)
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_discussion.py - List, get, create, and comment on GitHub discussions

CANONICAL SOURCE: ayates_dbx/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Provides consistent interface with gh issue create, adds identity automatically.

Usage:
  gh_discussion.py list --limit 5
  gh_discussion.py list --json
  gh_discussion.py get 211
  gh_discussion.py get 211 --json
  gh_discussion.py create --title "Title" --body "Body" --category "General"
  gh_discussion.py comment --number 42 --body "Comment"
  --repo ayates_dbx/dashnews

Categories for dashnews: General, Q&A, Show and tell, Ideas, Announcements, Polls

Public API:
- CATEGORY_IDS, DASHNEWS_REPO_ID
- fix_title, get_identity, get_real_gh, process_body
- usage, escape_graphql, get_repo_id, get_category_id
- create_discussion, get_discussion, comment_discussion, list_discussions, main

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import json as json_module
import sys
from pathlib import Path

# Add parent dir to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts.gh_graphql import (
    escape_graphql as escape_graphql_helper,  # noqa: E402
)
from ai_template_scripts.gh_graphql import graphql  # noqa: E402
from ai_template_scripts.gh_post import (  # noqa: E402
    fix_title,
    get_identity,
    get_real_gh,
    process_body,
)
from ai_template_scripts.subprocess_utils import run_cmd  # noqa: E402

# Category IDs for dashnews (default discussion repo)
CATEGORY_IDS = {
    "General": "DIC_kwDOQzpPMs4C0vhu",
    "Q&A": "DIC_kwDOQzpPMs4C0vhv",
    "Show and tell": "DIC_kwDOQzpPMs4C0vhx",
    "Ideas": "DIC_kwDOQzpPMs4C0vhw",
    "Announcements": "DIC_kwDOQzpPMs4C0vhy",
    "Polls": "DIC_kwDOQzpPMs4C0vhz",
}

# Repository ID for dashnews
DASHNEWS_REPO_ID = "R_kgDOQzpPMg"

__all__ = [
    "CATEGORY_IDS",
    "DASHNEWS_REPO_ID",
    "fix_title",
    "get_identity",
    "get_real_gh",
    "process_body",
    "usage",
    "escape_graphql",
    "get_repo_id",
    "get_category_id",
    "create_discussion",
    "get_discussion",
    "comment_discussion",
    "list_discussions",
    "main",
]


def _split_repo(repo: str) -> tuple[str, str] | None:
    """Split owner/repo and validate basic format."""
    parts = repo.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _extract_token_scopes(status_output: str) -> set[str]:
    """Extract token scopes from gh auth status output."""
    for line in status_output.splitlines():
        if "Token scopes:" not in line:
            continue
        _, scopes_text = line.split("Token scopes:", 1)
        scopes = set()
        for scope in scopes_text.split(","):
            cleaned = scope.strip().strip("'\"")
            if cleaned:
                scopes.add(cleaned)
        return scopes
    return set()


def _missing_scopes(real_gh: str, required_scopes: tuple[str, ...]) -> list[str]:
    """Return required scopes missing from gh auth status."""
    result = run_cmd([real_gh, "auth", "status"], timeout=30)
    if not result.ok:
        return []
    scopes = _extract_token_scopes(result.stdout)
    if not scopes:
        return []
    return [scope for scope in required_scopes if scope not in scopes]


def _print_scope_hint(real_gh: str, required_scopes: tuple[str, ...]) -> None:
    """Print a hint if gh auth status is missing required scopes."""
    missing = _missing_scopes(real_gh, required_scopes)
    if not missing:
        return
    scope_list = ", ".join(missing)
    scope_flags = " ".join(f"-s {scope}" for scope in missing)
    print(
        "Hint: GitHub token missing discussion scopes.",
        f"Add {scope_list} with: gh auth refresh {scope_flags}",
        file=sys.stderr,
    )


def usage() -> None:
    """Print usage information to stdout.

    REQUIRES: None
    ENSURES: Prints help text to stdout (no return value)
    """
    print("""Usage: gh_discussion.py <command> [OPTIONS]

Commands:
  list      List recent discussions
  get       Fetch a discussion by number
  create    Create a new discussion
  comment   Add a comment to an existing discussion

List Options:
  --repo OWNER/REPO   Target repo (default: ayates_dbx/dashnews)
  --limit N           Number of discussions to list (default: 10, max: 100)
  --category CAT      Filter by category (optional)
  --json              Output in JSON format

Get Options:
  NUMBER              Discussion number (required, positional)
  --repo OWNER/REPO   Target repo (default: ayates_dbx/dashnews)
  --json              Output in JSON format

Create Options:
  --title TITLE       Discussion title (required)
  --body BODY         Discussion body (required)
  --category CAT      Category: General, Q&A, "Show and tell", Ideas, Announcements,
                      Polls (default: General)
  --repo OWNER/REPO   Target repo (default: ayates_dbx/dashnews)

Comment Options:
  --number N          Discussion number (required)
  --body BODY         Comment body (required)
  --repo OWNER/REPO   Target repo (default: ayates_dbx/dashnews)

Examples:
  gh_discussion.py list
  gh_discussion.py list --limit 5 --category "Announcements"
  gh_discussion.py list --json
  gh_discussion.py get 211
  gh_discussion.py get 211 --json
  gh_discussion.py get 211 --repo ayates_dbx/other
  gh_discussion.py create --title "New discovery" --body "Found something interesting"
  gh_discussion.py create --title "Question" --body "How do I...?" --category "Q&A"
  gh_discussion.py comment --number 42 --body "Great post!"

Notes:
  Requires GitHub token scopes: read:discussion (list/get) and write:discussion (create/comment).
  Add scopes with: gh auth refresh -s read:discussion -s write:discussion
""")


def escape_graphql(text: str) -> str:
    """Escape text for GraphQL string.

    REQUIRES: text is a string
    ENSURES: Returns escaped string safe for GraphQL
    """
    return escape_graphql_helper(text)


def get_repo_id(real_gh: str, repo: str) -> str:
    """Get GitHub repository ID via GraphQL.

    REQUIRES: repo is "owner/name" format string
    ENSURES: Returns repo ID string on success, empty string on failure
    """
    _ = real_gh  # Unused; kept for API compatibility
    split_repo = _split_repo(repo)
    if not split_repo:
        return ""
    owner, name = split_repo
    query = f'query {{ repository(owner:"{owner}", name:"{name}") {{ id }} }}'
    result = graphql(query)
    if result.ok:
        repo_id = result.extract("repository.id")
        return repo_id if isinstance(repo_id, str) else ""
    return ""  # Best-effort: repo ID lookup via GraphQL, empty string allows caller to handle


def get_category_id(real_gh: str, repo: str, category: str) -> str:
    """Get discussion category ID for a repo.

    REQUIRES: repo is "owner/name" format string
    REQUIRES: category is category name string
    ENSURES: Returns category ID string on success, empty string on failure
    """
    _ = real_gh  # Unused; kept for API compatibility
    split_repo = _split_repo(repo)
    if not split_repo:
        return ""
    owner, name = split_repo
    query = f'''query {{
        repository(owner:"{owner}", name:"{name}") {{
            discussionCategories(first:20) {{
                nodes {{ id name }}
            }}
        }}
    }}'''
    result = graphql(query)
    if result.ok:
        nodes = result.extract("repository.discussionCategories.nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict) and node.get("name") == category:
                    node_id = node.get("id")
                    if isinstance(node_id, str):
                        return node_id
    return ""  # Best-effort: category ID lookup via GraphQL, empty string allows caller to handle


def create_discussion(args: list[str]) -> str:
    """Create a GitHub discussion with identity.

    REQUIRES: args is list of command-line arguments
    ENSURES: Returns discussion URL on success, exits on failure
    """
    title = ""
    body = ""
    category = "General"
    repo = "ayates_dbx/dashnews"

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == "--title" and i + 1 < len(args):
            title = args[i + 1]
            i += 2
        elif args[i] == "--body" and i + 1 < len(args):
            body = args[i + 1]
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] == "--repo" and i + 1 < len(args):
            repo = args[i + 1]
            i += 2
        elif args[i] in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            print(f"Error: Unknown option: {args[i]}", file=sys.stderr)
            sys.exit(1)

    if not title:
        print("Error: Missing required --title", file=sys.stderr)
        sys.exit(1)
    if not body:
        print("Error: Missing required --body", file=sys.stderr)
        sys.exit(1)
    if not _split_repo(repo):
        print("Error: --repo must be in OWNER/REPO format", file=sys.stderr)
        sys.exit(1)

    real_gh = get_real_gh()
    identity = get_identity()

    # Fix title and process body with identity
    title = fix_title(title, identity)
    body = process_body(body, identity)

    # Get category and repo IDs
    if repo == "ayates_dbx/dashnews":
        repo_id = DASHNEWS_REPO_ID
        category_id = CATEGORY_IDS.get(category)
        if not category_id:
            valid = ", ".join(CATEGORY_IDS.keys())
            print(
                f"Error: Unknown category: {category}. Valid: {valid}", file=sys.stderr
            )
            sys.exit(1)
    else:
        repo_id = get_repo_id(real_gh, repo)
        if not repo_id:
            _print_scope_hint(real_gh, ("read:discussion", "write:discussion"))
            print(f"Error: Could not get repo ID for {repo}", file=sys.stderr)
            sys.exit(1)
        category_id = get_category_id(real_gh, repo, category)
        if not category_id:
            _print_scope_hint(real_gh, ("read:discussion", "write:discussion"))
            print(f"Error: Category '{category}' not found in {repo}", file=sys.stderr)
            sys.exit(1)

    # Create discussion via GraphQL
    escaped_title = escape_graphql(title)
    escaped_body = escape_graphql(body)

    mutation = f'''mutation {{
        createDiscussion(input: {{
            repositoryId: "{repo_id}",
            categoryId: "{category_id}",
            title: "{escaped_title}",
            body: "{escaped_body}"
        }}) {{
            discussion {{
                url
            }}
        }}
    }}'''

    result = graphql(mutation)
    if result.ok:
        url = result.extract("createDiscussion.discussion.url")
        return url if isinstance(url, str) else ""
    _print_scope_hint(real_gh, ("read:discussion", "write:discussion"))
    error_msg = result.errors[0]["message"] if result.errors else result.stderr
    print(f"Error: Failed to create discussion: {error_msg}", file=sys.stderr)
    sys.exit(1)


def get_discussion_id(real_gh: str, repo: str, number: int) -> str:
    """Get discussion ID from discussion number.

    REQUIRES: repo is "owner/name" format string
    REQUIRES: number > 0
    ENSURES: Returns discussion ID string on success, empty string on failure
    """
    _ = real_gh  # Unused; kept for API compatibility
    split_repo = _split_repo(repo)
    if not split_repo:
        return ""
    owner, name = split_repo
    query = f'''query {{
        repository(owner:"{owner}", name:"{name}") {{
            discussion(number:{number}) {{ id }}
        }}
    }}'''
    result = graphql(query)
    if result.ok:
        disc_id = result.extract("repository.discussion.id")
        return disc_id if isinstance(disc_id, str) else ""
    return ""  # Best-effort: discussion ID lookup via GraphQL, empty string allows caller to handle


def get_discussion(args: list[str]) -> dict:
    """Fetch a GitHub discussion by number.

    REQUIRES: args contains discussion number (positional) and optional --repo
    ENSURES: Returns dict with discussion details, exits on failure

    Returns dict with: number, title, body, url, createdAt, category, author.
    """
    number = 0
    repo = "ayates_dbx/dashnews"
    json_output = False

    # Parse arguments - first positional arg is number
    i = 0
    while i < len(args):
        if args[i] == "--repo" and i + 1 < len(args):
            repo = args[i + 1]
            i += 2
        elif args[i] == "--json":
            json_output = True
            i += 1
        elif args[i] in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif args[i].startswith("-"):
            print(f"Error: Unknown option: {args[i]}", file=sys.stderr)
            sys.exit(1)
        elif not number:
            # Positional argument: discussion number
            try:
                number = int(args[i])
            except ValueError:
                print(f"Error: Invalid discussion number: {args[i]}", file=sys.stderr)
                sys.exit(1)
            if number <= 0:
                print("Error: Discussion number must be positive", file=sys.stderr)
                sys.exit(1)
            i += 1
        else:
            print(f"Error: Unexpected argument: {args[i]}", file=sys.stderr)
            sys.exit(1)

    if not number:
        print("Error: Missing required discussion number", file=sys.stderr)
        sys.exit(1)

    split_repo = _split_repo(repo)
    if not split_repo:
        print("Error: --repo must be in OWNER/REPO format", file=sys.stderr)
        sys.exit(1)

    owner, name = split_repo

    # Fetch discussion with full content
    query = f'''query {{
        repository(owner:"{owner}", name:"{name}") {{
            discussion(number:{number}) {{
                number
                title
                body
                url
                createdAt
                category {{ name }}
                author {{ login }}
            }}
        }}
    }}'''

    result = graphql(query)
    if not result.ok:
        real_gh = get_real_gh()
        _print_scope_hint(real_gh, ("read:discussion",))
        error_msg = result.errors[0]["message"] if result.errors else result.stderr
        print(f"Error: Failed to fetch discussion: {error_msg}", file=sys.stderr)
        sys.exit(1)

    discussion_data = result.extract("repository.discussion")
    if not isinstance(discussion_data, dict) or not discussion_data:
        print(f"Error: Discussion #{number} not found in {repo}", file=sys.stderr)
        sys.exit(1)

    discussion = {
        "number": discussion_data.get("number"),
        "title": discussion_data.get("title"),
        "body": discussion_data.get("body"),
        "url": discussion_data.get("url"),
        "createdAt": discussion_data.get("createdAt"),
        "category": discussion_data.get("category", {}).get("name")
        if discussion_data.get("category")
        else None,
        "author": discussion_data.get("author", {}).get("login")
        if discussion_data.get("author")
        else None,
    }

    # Output
    if json_output:
        print(json_module.dumps(discussion, indent=2))
    else:
        cat_str = f" [{discussion['category']}]" if discussion.get("category") else ""
        author_str = f" by {discussion['author']}" if discussion.get("author") else ""
        print(f"#{discussion['number']}: {discussion['title']}{cat_str}{author_str}")
        print(f"URL: {discussion['url']}")
        if discussion.get("createdAt"):
            print(f"Created: {discussion['createdAt']}")
        print()
        print("---")
        print(discussion.get("body", ""))

    return discussion


def comment_discussion(args: list[str]) -> str:
    """Add a comment to a GitHub discussion with identity.

    REQUIRES: args is list with --number and --body options
    ENSURES: Returns comment URL on success, exits on failure
    """
    number = 0
    body = ""
    repo = "ayates_dbx/dashnews"

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == "--number" and i + 1 < len(args):
            try:
                number = int(args[i + 1])
            except ValueError:
                print("Error: --number must be an integer", file=sys.stderr)
                sys.exit(1)
            if number <= 0:
                print("Error: --number must be a positive integer", file=sys.stderr)
                sys.exit(1)
            i += 2
        elif args[i] == "--body" and i + 1 < len(args):
            body = args[i + 1]
            i += 2
        elif args[i] == "--repo" and i + 1 < len(args):
            repo = args[i + 1]
            i += 2
        elif args[i] in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            print(f"Error: Unknown option: {args[i]}", file=sys.stderr)
            sys.exit(1)

    if not number:
        print("Error: Missing required --number", file=sys.stderr)
        sys.exit(1)
    if not body:
        print("Error: Missing required --body", file=sys.stderr)
        sys.exit(1)
    if not _split_repo(repo):
        print("Error: --repo must be in OWNER/REPO format", file=sys.stderr)
        sys.exit(1)

    real_gh = get_real_gh()
    identity = get_identity()

    # Process body with identity
    body = process_body(body, identity)

    # Get discussion ID
    discussion_id = get_discussion_id(real_gh, repo, number)
    if not discussion_id:
        _print_scope_hint(real_gh, ("read:discussion", "write:discussion"))
        print(f"Error: Could not find discussion #{number} in {repo}", file=sys.stderr)
        sys.exit(1)

    # Add comment via GraphQL
    escaped_body = escape_graphql(body)

    mutation = f'''mutation {{
        addDiscussionComment(input: {{
            discussionId: "{discussion_id}",
            body: "{escaped_body}"
        }}) {{
            comment {{
                url
            }}
        }}
    }}'''

    result = graphql(mutation)
    if result.ok:
        url = result.extract("addDiscussionComment.comment.url")
        return url if isinstance(url, str) else ""
    _print_scope_hint(real_gh, ("read:discussion", "write:discussion"))
    error_msg = result.errors[0]["message"] if result.errors else result.stderr
    print(f"Error: Failed to add comment: {error_msg}", file=sys.stderr)
    sys.exit(1)


def list_discussions(args: list[str]) -> list[dict]:
    """List recent discussions from a repository.

    REQUIRES: args is list of command-line arguments
    ENSURES: Returns list of discussion dicts, exits on failure

    Returns list of dicts with: number, title, url, createdAt, category, author.
    """
    repo = "ayates_dbx/dashnews"
    limit = 10
    category = None
    json_output = False

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == "--repo" and i + 1 < len(args):
            repo = args[i + 1]
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
                if limit < 1:
                    limit = 1
                elif limit > 100:
                    limit = 100
            except ValueError:
                print("Error: --limit must be an integer", file=sys.stderr)
                sys.exit(1)
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] == "--json":
            json_output = True
            i += 1
        elif args[i] in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            print(f"Error: Unknown option: {args[i]}", file=sys.stderr)
            sys.exit(1)

    split_repo = _split_repo(repo)
    if not split_repo:
        print("Error: --repo must be in OWNER/REPO format", file=sys.stderr)
        sys.exit(1)

    owner, name = split_repo

    # Build GraphQL query
    query = f'''query {{
        repository(owner:"{owner}", name:"{name}") {{
            discussions(first:{limit}, orderBy:{{field:CREATED_AT, direction:DESC}}) {{
                nodes {{
                    number
                    title
                    url
                    createdAt
                    category {{ name }}
                    author {{ login }}
                }}
            }}
        }}
    }}'''

    result = graphql(query)
    if not result.ok:
        real_gh = get_real_gh()
        _print_scope_hint(real_gh, ("read:discussion",))
        error_msg = result.errors[0]["message"] if result.errors else result.stderr
        print(f"Error: Failed to list discussions: {error_msg}", file=sys.stderr)
        sys.exit(1)

    nodes = result.extract("repository.discussions.nodes")
    if not isinstance(nodes, list):
        print("Error: Unexpected response format", file=sys.stderr)
        sys.exit(1)

    # Filter by category if specified
    discussions = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        cat_name = (
            node.get("category", {}).get("name") if node.get("category") else None
        )
        if category and cat_name != category:
            continue
        discussions.append(
            {
                "number": node.get("number"),
                "title": node.get("title"),
                "url": node.get("url"),
                "createdAt": node.get("createdAt"),
                "category": cat_name,
                "author": node.get("author", {}).get("login")
                if node.get("author")
                else None,
            }
        )

    # Output
    if json_output:
        print(json_module.dumps(discussions, indent=2))
    else:
        for d in discussions:
            cat_str = f" [{d['category']}]" if d.get("category") else ""
            author_str = f" by {d['author']}" if d.get("author") else ""
            print(f"#{d['number']}: {d['title']}{cat_str}{author_str}")
            print(f"  {d['url']}")
            if d.get("createdAt"):
                print(f"  Created: {d['createdAt']}")
            print()

    return discussions


def main() -> None:
    """Entry point: create, get, comment on, or list GitHub discussions.

    REQUIRES: None (reads sys.argv)
    ENSURES: Exits with 0 on success, 1 on error
    """
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        usage()
        sys.exit(0 if args else 1)

    if args[0] == "list":
        list_discussions(args[1:])
    elif args[0] == "get":
        get_discussion(args[1:])
    elif args[0] == "create":
        url = create_discussion(args[1:])
        print(url)
    elif args[0] == "comment":
        url = comment_discussion(args[1:])
        print(url)
    else:
        usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

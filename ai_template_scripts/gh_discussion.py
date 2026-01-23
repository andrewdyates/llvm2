#!/usr/bin/env python3
"""
gh_discussion.py - Create GitHub discussions with AI identity

CANONICAL SOURCE: ayates_dbx/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Provides consistent interface with gh issue create, adds identity automatically.

Usage:
  gh_discussion.py create --title "Title" --body "Body" --category "General"
  gh_discussion.py create --title "Title" --body "Body" --category "Q&A" --repo ayates_dbx/dashnews

Categories for dashnews: General, Q&A, Show and tell, Ideas, Announcements, Polls

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import json
import subprocess
import sys

from gh_post import (
    fix_title,
    get_identity,
    get_real_gh,
    process_body,
)

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


def usage():
    print("""Usage: gh_discussion.py <command> [OPTIONS]

Commands:
  create    Create a new discussion
  comment   Add a comment to an existing discussion

Create Options:
  --title TITLE       Discussion title (required)
  --body BODY         Discussion body (required)
  --category CAT      Category: General, Q&A, "Show and tell", Ideas, Announcements, Polls (default: General)
  --repo OWNER/REPO   Target repo (default: ayates_dbx/dashnews)

Comment Options:
  --number N          Discussion number (required)
  --body BODY         Comment body (required)
  --repo OWNER/REPO   Target repo (default: ayates_dbx/dashnews)

Examples:
  gh_discussion.py create --title "New discovery" --body "Found something interesting"
  gh_discussion.py create --title "Question" --body "How do I...?" --category "Q&A"
  gh_discussion.py comment --number 42 --body "Great post!"
""")


def escape_graphql(text: str) -> str:
    """Escape text for GraphQL string."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def get_repo_id(real_gh: str, repo: str) -> str:
    """Get GitHub repository ID via GraphQL."""
    owner, name = repo.split("/")
    query = f'query {{ repository(owner:"{owner}", name:"{name}") {{ id }} }}'
    try:
        return subprocess.check_output(
            [real_gh, "api", "graphql", "-f", f"query={query}", "-q", ".data.repository.id"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def get_category_id(real_gh: str, repo: str, category: str) -> str:
    """Get discussion category ID for a repo."""
    owner, name = repo.split("/")
    query = f'''query {{
        repository(owner:"{owner}", name:"{name}") {{
            discussionCategories(first:20) {{
                nodes {{ id name }}
            }}
        }}
    }}'''
    try:
        result = subprocess.check_output(
            [real_gh, "api", "graphql", "-f", f"query={query}"],
            stderr=subprocess.DEVNULL, text=True
        )
        data = json.loads(result)
        for node in data["data"]["repository"]["discussionCategories"]["nodes"]:
            if node["name"] == category:
                return node["id"]
    except Exception:
        pass
    return ""


def create_discussion(args: list[str]) -> str:
    """Create a GitHub discussion with identity."""
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
            print(f"Error: Unknown category: {category}. Valid: {valid}", file=sys.stderr)
            sys.exit(1)
    else:
        repo_id = get_repo_id(real_gh, repo)
        if not repo_id:
            print(f"Error: Could not get repo ID for {repo}", file=sys.stderr)
            sys.exit(1)
        category_id = get_category_id(real_gh, repo, category)
        if not category_id:
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

    try:
        return subprocess.check_output(
            [real_gh, "api", "graphql", "-f", f"query={mutation}", "-q", ".data.createDiscussion.discussion.url"],
            stderr=subprocess.PIPE, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to create discussion: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_discussion_id(real_gh: str, repo: str, number: int) -> str:
    """Get discussion ID from discussion number."""
    owner, name = repo.split("/")
    query = f'''query {{
        repository(owner:"{owner}", name:"{name}") {{
            discussion(number:{number}) {{ id }}
        }}
    }}'''
    try:
        result = subprocess.check_output(
            [real_gh, "api", "graphql", "-f", f"query={query}"],
            stderr=subprocess.DEVNULL, text=True
        )
        data = json.loads(result)
        return data["data"]["repository"]["discussion"]["id"]
    except Exception:
        return ""


def comment_discussion(args: list[str]) -> str:
    """Add a comment to a GitHub discussion with identity."""
    number = 0
    body = ""
    repo = "ayates_dbx/dashnews"

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == "--number" and i + 1 < len(args):
            number = int(args[i + 1])
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

    real_gh = get_real_gh()
    identity = get_identity()

    # Process body with identity
    body = process_body(body, identity)

    # Get discussion ID
    discussion_id = get_discussion_id(real_gh, repo, number)
    if not discussion_id:
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

    try:
        return subprocess.check_output(
            [real_gh, "api", "graphql", "-f", f"query={mutation}", "-q", ".data.addDiscussionComment.comment.url"],
            stderr=subprocess.PIPE, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to add comment: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        usage()
        sys.exit(0 if args else 1)

    if args[0] == "create":
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

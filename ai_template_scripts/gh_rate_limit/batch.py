# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
batch.py - Batch GraphQL helpers for gh_rate_limit.

Provides batched issue fetches to minimize API calls from looper utilities.
"""

from __future__ import annotations

from ai_template_scripts.gh_graphql import graphql
from ai_template_scripts.gh_rate_limit.rate_limiter import get_limiter

__all__ = ["batch_issue_view", "batch_issue_timelines"]


def batch_issue_view(
    issue_nums: list[int],
    fields: list[str],
    repo: str | None = None,
    timeout: float = 30,
) -> dict[int, dict | None]:
    """Fetch multiple issues in a single GraphQL query.

    Uses GraphQL aliases to batch multiple issue lookups into one API call.
    This is a key optimization for issue_context.py which needs to fetch
    in-progress issues (#1178).

    Args:
        issue_nums: List of issue numbers to fetch.
        fields: Fields to include (e.g., ["number", "title", "labels"]).
        repo: Repository in owner/repo format. Uses current repo if not specified.
        timeout: Request timeout in seconds.

    Returns:
        Dict mapping issue number to issue data dict, or None if issue not found.
        Example: {42: {"number": 42, "title": "...", ...}, 43: None}
    """
    if not issue_nums:
        return {}

    limiter = get_limiter()

    # Get repo from parameter or current directory
    owner_repo = limiter._get_owner_repo(repo)
    if not owner_repo:
        return dict.fromkeys(issue_nums)

    owner, repo_name = (
        owner_repo.split("/", 1) if "/" in owner_repo else ("", owner_repo)
    )
    if not owner or not repo_name:
        return dict.fromkeys(issue_nums)

    # Build GraphQL query with aliases (i42: repository(...) { issue(...) { ... } })
    field_query = " ".join(fields)

    # Handle special fields that need expansion
    # labels requires pagination per GraphQL API spec
    if "labels" in fields:
        field_query = field_query.replace(
            "labels", "labels(first: 20) { nodes { name } }"
        )

    query_parts = [
        f'i{num}: repository(owner: "{owner}", name: "{repo_name}") '
        f"{{ issue(number: {num}) {{ {field_query} }} }}"
        for num in issue_nums
    ]

    query = "query { " + " ".join(query_parts) + " }"

    # Execute query using shared GraphQL helper (#1193)
    result = graphql(query, timeout=timeout)

    if not result.ok:
        # Return None for all issues on failure
        return dict.fromkeys(issue_nums)

    results: dict[int, dict | None] = {}

    for num in issue_nums:
        issue_data = result.extract(f"i{num}.issue")
        if issue_data:
            # Flatten labels from { nodes: [{name: "..."}, ...] } to ["...", ...]
            if "labels" in issue_data and isinstance(issue_data["labels"], dict):
                nodes = issue_data["labels"].get("nodes", [])
                issue_data["labels"] = [n.get("name") for n in nodes if n.get("name")]
            results[num] = issue_data
        else:
            results[num] = None

    return results


def batch_issue_timelines(
    issue_nums: list[int],
    repo: str | None = None,
    timeout: float = 30,
) -> dict[int, int | None]:
    """Fetch reopen timeline counts for multiple issues in a single query.

    Uses GraphQL aliases to batch timeline lookups. This is optimized for
    thrashing detection, which only needs reopened event counts.

    Args:
        issue_nums: List of issue numbers to fetch.
        repo: Repository in owner/repo format. Uses current repo if not specified.
        timeout: Request timeout in seconds.

    Returns:
        Dict mapping issue number to reopen count, or None if issue not found
        or query failed for that issue.
    """
    if not issue_nums:
        return {}

    limiter = get_limiter()
    owner_repo = limiter._get_owner_repo(repo)
    if not owner_repo:
        return dict.fromkeys(issue_nums)

    owner, repo_name = (
        owner_repo.split("/", 1) if "/" in owner_repo else ("", owner_repo)
    )
    if not owner or not repo_name:
        return dict.fromkeys(issue_nums)

    query_parts = [
        f'i{num}: repository(owner: "{owner}", name: "{repo_name}") '
        f"{{ issue(number: {num}) {{ "
        "timelineItems(first: 1, itemTypes: [REOPENED_EVENT]) { totalCount } "
        "} }}"
        for num in issue_nums
    ]
    query = "query { " + " ".join(query_parts) + " }"

    result = graphql(query, timeout=timeout)
    if not result.ok:
        return dict.fromkeys(issue_nums)

    results: dict[int, int | None] = {}
    for num in issue_nums:
        count = result.extract(f"i{num}.issue.timelineItems.totalCount")
        results[num] = int(count) if count is not None else None

    return results

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""Shared GraphQL helper for GitHub API operations.

This module provides a thin wrapper around `gh api graphql` with:
- Uniform variable type encoding
- Consistent error handling
- Pagination helpers

Part of #1193.

Usage:
    from ai_template_scripts.gh_graphql import graphql, graphql_batch

    # Simple query
    result = graphql('query { viewer { login } }')
    if result.ok:
        print(result.data)

    # Query with variables
    result = graphql(
        '''query($owner: String!, $name: String!) {
            repository(owner: $owner, name: $name) { id }
        }''',
        variables={'owner': 'ayates_dbx', 'name': 'ai_template'}
    )

    # Batch multiple queries
    result = graphql_batch([
        ('repo1', 'repository(owner:"ayates_dbx", name:"ai_template") { id }'),
        ('repo2', 'repository(owner:"ayates_dbx", name:"leadership") { id }'),
    ])
"""

from __future__ import annotations

__all__ = [
    "GraphQLResult",
    "graphql",
    "graphql_batch",
    "escape_graphql",
    "build_variables",
    "has_next_page",
    "end_cursor",
]

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any


def _get_real_gh() -> str:
    """Find the real gh binary, avoiding wrapper recursion.

    Returns:
        Path to the gh binary.

    Raises:
        RuntimeError: If gh CLI is not found.
    """
    for loc in ["/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh"]:
        if os.path.isfile(loc) and os.access(loc, os.X_OK):
            return loc
    raise RuntimeError("gh CLI not found")


def _encode_variables(variables: dict[str, Any]) -> list[str]:
    """Encode variables for gh api graphql command.

    Args:
        variables: Dict of variables with values to encode.

    Returns:
        List of arguments for subprocess.

    Type encoding:
        - None -> -F key=null
        - int/float -> -F key=N (numeric)
        - bool -> -F key=true/false
        - str -> -f key=value (string)
        - list/dict -> -f key=JSON (complex)
    """
    args: list[str] = []
    for key, value in variables.items():
        if value is None:
            args.extend(["-F", f"{key}=null"])
        elif isinstance(value, bool):
            args.extend(["-F", f"{key}={'true' if value else 'false'}"])
        elif isinstance(value, (int, float)):
            args.extend(["-F", f"{key}={value}"])
        elif isinstance(value, (list, dict)):
            args.extend(["-f", f"{key}={json.dumps(value)}"])
        else:
            args.extend(["-f", f"{key}={value}"])
    return args


@dataclass
class GraphQLResult:
    """Result of a GraphQL query.

    Attributes:
        data: Response data if successful (may be None even on success).
        errors: List of GraphQL errors if any.
        stderr: Raw stderr from gh command.
        returncode: Exit code from gh command.
    """

    data: dict[str, Any] | None
    errors: list[dict[str, Any]] | None
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        """True if query succeeded without errors."""
        return self.returncode == 0 and not self.errors

    def extract(self, path: str) -> Any | None:
        """Extract value at jq-style path relative to response data.

        Args:
            path: Path like '.repository.id' or 'repository.id'.
                  Path is relative to self.data (the 'data' wrapper is stripped
                  during response parsing).

        Returns:
            Value at path, or None if path doesn't exist.

        Example:
            result.extract('.repository.id')
            result.extract('repository.id')  # equivalent
        """
        if self.data is None:
            return None

        # Strip leading dot if present
        path = path.removeprefix(".")

        # Navigate path
        parts = path.split(".")
        current: Any = self.data

        for part in parts:
            if not part:
                continue
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current


def graphql(
    query: str,
    variables: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> GraphQLResult:
    """Execute a GraphQL query via gh api graphql.

    Args:
        query: GraphQL query string.
        variables: Dict of variables (types auto-detected).
        timeout: Request timeout in seconds.

    Returns:
        GraphQLResult with data, errors, and success status.

    Example:
        >>> result = graphql('query { viewer { login } }')
        >>> if result.ok:
        ...     print(result.data['viewer']['login'])

        >>> result = graphql(
        ...     'query($owner: String!) { user(login: $owner) { id } }',
        ...     variables={'owner': 'ayates_dbx'}
        ... )
    """
    cmd = [_get_real_gh(), "api", "graphql", "-f", f"query={query}"]

    if variables:
        cmd.extend(_encode_variables(variables))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GraphQLResult(
            data=None,
            errors=[{"message": f"Request timed out after {timeout}s"}],
            stderr="timeout",
            returncode=-1,
        )

    # Parse response
    data = None
    errors = None

    if result.stdout.strip():
        try:
            response = json.loads(result.stdout)
            data = response.get("data")
            errors = response.get("errors")
        except json.JSONDecodeError:
            errors = [{"message": f"Invalid JSON response: {result.stdout[:100]}"}]

    return GraphQLResult(
        data=data,
        errors=errors,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def graphql_batch(
    queries: list[tuple[str, str]],
    timeout: float = 30.0,
) -> GraphQLResult:
    """Execute multiple queries as aliases in a single GraphQL request.

    This is more efficient than multiple separate queries as it uses
    only one API call (important for rate limiting).

    Args:
        queries: List of (alias, query_fragment) tuples.
            Each fragment should be a valid GraphQL selection without 'query'.
        timeout: Request timeout in seconds.

    Returns:
        GraphQLResult where data contains alias keys.

    Example:
        >>> result = graphql_batch([
        ...     ('repo1', 'repository(owner:"ayates_dbx", name:"ai_template") { id }'),
        ...     ('repo2', 'repository(owner:"ayates_dbx", name:"leadership") { id }'),
        ... ])
        >>> if result.ok:
        ...     print(result.data['repo1']['id'])
        ...     print(result.data['repo2']['id'])
    """
    if not queries:
        return GraphQLResult(data={}, errors=None, stderr="", returncode=0)

    # Build combined query with aliases
    fragments = [f"{alias}: {fragment}" for alias, fragment in queries]
    combined_query = "query { " + " ".join(fragments) + " }"

    return graphql(combined_query, timeout=timeout)


def escape_graphql(text: str) -> str:
    """Escape text for use in GraphQL string literals.

    Args:
        text: Raw text to escape.

    Returns:
        Escaped text safe for GraphQL strings.

    Example:
        >>> query = f'mutation {{ addComment(body: "{escape_graphql(user_input)}") }}'
    """
    # Escape backslashes first, then quotes, then newlines
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def build_variables(*pairs: tuple[str, Any]) -> dict[str, Any]:
    """Build variables dict from key-value pairs.

    Convenience function for building variable dicts inline.

    Args:
        pairs: Tuples of (name, value).

    Returns:
        Dict suitable for graphql() variables parameter.

    Example:
        >>> variables = build_variables(
        ...     ('owner', 'ayates_dbx'),
        ...     ('name', 'ai_template'),
        ...     ('first', 10),
        ... )
    """
    return dict(pairs)


def has_next_page(response: dict[str, Any], path: str = "") -> bool:
    """Check if a paginated response has more pages.

    Args:
        response: GraphQL response data dict.
        path: Dot-separated path to the pageInfo object's parent.
            If empty, looks for pageInfo at response root.

    Returns:
        True if hasNextPage is true.

    Example:
        >>> query = 'query { user { repos(first: 10) { pageInfo { hasNextPage } } } }'
        >>> result = graphql(query)
        >>> if has_next_page(result.data, 'user.repos'):
        ...     # fetch more pages
    """
    if not response:
        return False

    # Navigate to parent of pageInfo
    current = response
    if path:
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False

    # Get pageInfo.hasNextPage
    if isinstance(current, dict):
        page_info = current.get("pageInfo", {})
        return bool(page_info.get("hasNextPage", False))

    return False


def end_cursor(response: dict[str, Any], path: str = "") -> str | None:
    """Extract cursor for fetching next page.

    Args:
        response: GraphQL response data dict.
        path: Dot-separated path to the pageInfo object's parent.

    Returns:
        endCursor string, or None if not present.

    Example:
        >>> query = 'query { user { repos(first: 10) { pageInfo { endCursor } } } }'
        >>> result = graphql(query)
        >>> cursor = end_cursor(result.data, 'user.repos')
        >>> if cursor:
        ...     # use cursor for next page
    """
    if not response:
        return None

    # Navigate to parent of pageInfo
    current = response
    if path:
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

    # Get pageInfo.endCursor
    if isinstance(current, dict):
        page_info = current.get("pageInfo", {})
        cursor = page_info.get("endCursor")
        return cursor if isinstance(cursor, str) else None

    return None


if __name__ == "__main__":
    # Quick test - fetch current user
    result = graphql("query { viewer { login } }")
    if result.ok:
        login = result.extract("viewer.login")
        print(f"Authenticated as: {login}")
    else:
        print(f"Error: {result.errors}")
        print(f"Stderr: {result.stderr}")

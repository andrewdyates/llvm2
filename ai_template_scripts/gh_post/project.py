# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""GitHub Project integration helpers for gh_post."""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ai_template_scripts.repo_directors import get_director
from looper.log import debug_swallow


def _gh_post():
    import ai_template_scripts.gh_post as gh_post_module

    return gh_post_module


# Module-level flag to suppress repeated scope missing warnings (#1894)
_scope_missing_logged: bool = False


@dataclass
class _ProjectSchema:
    """Cached project field IDs and option mappings."""

    project_id: str = ""
    status_field_id: str = ""
    status_todo_id: str = ""
    director_field_id: str = ""
    director_options: dict[str, str] = field(default_factory=dict)  # name -> id
    type_field_id: str = ""
    type_task_id: str = ""
    fetched_at: float = 0.0  # Unix timestamp
    scope_missing: bool = False  # Token lacks read:project scope (#1894)


def _load_cached_schema() -> _ProjectSchema | None:
    """Load schema from cache file if valid."""
    gh_post_module = _gh_post()
    cache_file = gh_post_module.PROJECT_SCHEMA_CACHE_FILE
    ttl_seconds = gh_post_module.PROJECT_SCHEMA_TTL_SECONDS

    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        schema = _ProjectSchema(
            project_id=data.get("project_id", ""),
            status_field_id=data.get("status_field_id", ""),
            status_todo_id=data.get("status_todo_id", ""),
            director_field_id=data.get("director_field_id", ""),
            director_options=data.get("director_options", {}),
            type_field_id=data.get("type_field_id", ""),
            type_task_id=data.get("type_task_id", ""),
            fetched_at=data.get("fetched_at", 0.0),
            scope_missing=data.get("scope_missing", False),  # #1894
        )
        # Check TTL
        if time.time() - schema.fetched_at > ttl_seconds:
            return None
        return schema
    except Exception as e:
        debug_swallow("gh_post_project_load_cached_schema", e)
        return None


def _save_schema_cache(schema: _ProjectSchema) -> None:
    """Save schema to cache file."""
    gh_post_module = _gh_post()
    cache_file = gh_post_module.PROJECT_SCHEMA_CACHE_FILE

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "project_id": schema.project_id,
            "status_field_id": schema.status_field_id,
            "status_todo_id": schema.status_todo_id,
            "director_field_id": schema.director_field_id,
            "director_options": schema.director_options,
            "type_field_id": schema.type_field_id,
            "type_task_id": schema.type_task_id,
            "fetched_at": schema.fetched_at,
            "scope_missing": schema.scope_missing,  # #1894
        }
        cache_file.write_text(json.dumps(data))
    except Exception as e:
        print(
            f"[gh_post] Warning: failed to cache project schema: {e}", file=sys.stderr
        )


def _fetch_project_schema(real_gh: str) -> _ProjectSchema | None:
    """Fetch project field IDs via GitHub API."""
    gh_post_module = _gh_post()
    owner = gh_post_module.PROJECT_OWNER
    number = gh_post_module.PROJECT_NUMBER

    query = f'''
    query {{
        organization(login: "{owner}") {{
            projectV2(number: {number}) {{
                id
                fields(first: 20) {{
                    nodes {{
                        ... on ProjectV2Field {{
                            id
                            name
                        }}
                        ... on ProjectV2SingleSelectField {{
                            id
                            name
                            options {{
                                id
                                name
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    '''

    result = subprocess.run(
        [real_gh, "api", "graphql", "-f", f"query={query}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        # Check for missing scope error (token lacks read:project scope) (#1894)
        # When scope is missing, return a schema with scope_missing=True so it gets cached
        # and we skip silently on subsequent calls
        error_msg = result.stderr or result.stdout or ""
        if (
            "required scopes" in error_msg.lower()
            or "read:project" in error_msg.lower()
        ):
            global _scope_missing_logged
            if not _scope_missing_logged:
                _scope_missing_logged = True
                import os

                if os.environ.get("AIT_GH_DEBUG", "").lower() in ("1", "true", "yes"):
                    print(
                        "[gh_post] Skipping project integration: token lacks read:project scope",
                        file=sys.stderr,
                    )
            # Return a schema with scope_missing=True so it gets cached
            return _ProjectSchema(scope_missing=True, fetched_at=time.time())
        print(
            f"[gh_post] Warning: failed to fetch project schema: {result.stderr}",
            file=sys.stderr,
        )
        return None

    try:
        data = json.loads(result.stdout)
        project = data["data"]["organization"]["projectV2"]
        fields = project["fields"]["nodes"]

        schema = _ProjectSchema()
        schema.project_id = project["id"]
        schema.fetched_at = time.time()

        for field in fields:
            name = field.get("name", "")
            if name == "Status":
                schema.status_field_id = field["id"]
                for option in field.get("options", []):
                    if option.get("name") == "Todo":
                        schema.status_todo_id = option["id"]
            elif name == "Director":
                schema.director_field_id = field["id"]
                for option in field.get("options", []):
                    schema.director_options[option["name"]] = option["id"]
            elif name == "Type":
                schema.type_field_id = field["id"]
                for option in field.get("options", []):
                    if option.get("name") == "Task":
                        schema.type_task_id = option["id"]

        return schema
    except Exception as e:
        print(
            f"[gh_post] Warning: failed to parse project schema: {e}", file=sys.stderr
        )
        return None


def _get_project_schema(real_gh: str) -> _ProjectSchema | None:
    """Get cached schema or fetch fresh if needed.

    Returns None if token lacks required scopes (scope_missing=True in cache).
    """
    gh_post_module = _gh_post()

    cached = gh_post_module._load_cached_schema()
    if cached:
        # If cached schema indicates missing scope, return None silently (#1894)
        if cached.scope_missing:
            return None
        return cached

    schema = gh_post_module._fetch_project_schema(real_gh)
    if schema:
        gh_post_module._save_schema_cache(schema)
        # If schema indicates missing scope, return None silently (#1894)
        if schema.scope_missing:
            return None
        return schema

    return None


def _get_issue_node_id(real_gh: str, repo: str, issue_number: str) -> str | None:
    """Get GraphQL node ID for issue."""
    query = f'''
    query {{
        repository(owner: "{repo.split("/")[0]}", name: "{repo.split("/")[1]}") {{
            issue(number: {issue_number}) {{
                id
            }}
        }}
    }}
    '''

    result = subprocess.run(
        [real_gh, "api", "graphql", "-f", f"query={query}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        return data["data"]["repository"]["issue"]["id"]
    except Exception as e:
        debug_swallow("gh_post_project_get_issue_node_id", e)
        return None


def _ensure_full_repo_name(repo: str) -> str:
    """Ensure repo has owner prefix (owner/repo)."""
    gh_post_module = _gh_post()

    if not repo:
        # Get current repo from git when repo is empty (#2449)
        # _get_current_repo_name() already returns owner/repo format
        return gh_post_module._get_current_repo_name()
    if "/" in repo:
        return repo
    return f"{gh_post_module.PROJECT_OWNER}/{repo}"


def _add_issue_to_project(real_gh: str, issue_node_id: str, repo_name: str) -> bool:
    """Add issue to GitHub Project and set fields."""
    gh_post_module = _gh_post()

    schema = gh_post_module._get_project_schema(real_gh)
    if not schema:
        # Schema is None when scope is missing - skip silently (#1894)
        # The _get_project_schema function already handles logging for scope errors
        return False

    # Add to project
    add_mutation = f'''
    mutation {{
        addProjectV2ItemById(input: {{
            projectId: "{schema.project_id}",
            contentId: "{issue_node_id}"
        }}) {{
            item {{ id }}
        }}
    }}
    '''

    result = subprocess.run(
        [real_gh, "api", "graphql", "-f", f"query={add_mutation}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        print(
            f"[gh_post] Warning: failed to add issue to project: {result.stderr}",
            file=sys.stderr,
        )
        return False

    # Parse item_id
    try:
        data = json.loads(result.stdout)
        item_id = (
            data.get("data", {})
            .get("addProjectV2ItemById", {})
            .get("item", {})
            .get("id")
        )
    except Exception as e:
        debug_swallow("gh_post_project_parse_item_id", e)
        item_id = None

    if not item_id:
        print(
            "[gh_post] Warning: no item_id returned from project add", file=sys.stderr
        )
        return False

    # Set fields
    gh_post_module._set_project_fields(real_gh, schema, item_id, repo_name)
    return True


def _set_project_fields(
    real_gh: str,
    schema: _ProjectSchema,
    item_id: str,
    repo_name: str,
    _retry: bool = False,
) -> None:
    """Set Status=Todo, Director, Type=Task on project item.

    Args:
        real_gh: Path to real gh binary
        schema: Project schema with field IDs
        item_id: Project item ID to update
        repo_name: Repository name for director lookup
        _retry: Internal flag to prevent infinite retry loops (#1209)
    """
    gh_post_module = _gh_post()

    mutations = []

    # Status = Todo
    if schema.status_field_id and schema.status_todo_id:
        mutations.append(
            f"status: updateProjectV2ItemFieldValue(input: {{"
            f'projectId: "{schema.project_id}", '
            f'itemId: "{item_id}", '
            f'fieldId: "{schema.status_field_id}", '
            f'value: {{singleSelectOptionId: "{schema.status_todo_id}"}}'
            f"}}) {{ clientMutationId }}"
        )

    # Director based on repo
    director = get_director(repo_name, default="TOOL")
    director_opt_id = schema.director_options.get(director)
    if schema.director_field_id and director_opt_id:
        mutations.append(
            f"director: updateProjectV2ItemFieldValue(input: {{"
            f'projectId: "{schema.project_id}", '
            f'itemId: "{item_id}", '
            f'fieldId: "{schema.director_field_id}", '
            f'value: {{singleSelectOptionId: "{director_opt_id}"}}'
            f"}}) {{ clientMutationId }}"
        )

    # Type = Task
    if schema.type_field_id and schema.type_task_id:
        mutations.append(
            f"type: updateProjectV2ItemFieldValue(input: {{"
            f'projectId: "{schema.project_id}", '
            f'itemId: "{item_id}", '
            f'fieldId: "{schema.type_field_id}", '
            f'value: {{singleSelectOptionId: "{schema.type_task_id}"}}'
            f"}}) {{ clientMutationId }}"
        )

    if not mutations:
        return

    query = f"mutation {{ {' '.join(mutations)} }}"
    result = subprocess.run(
        [real_gh, "api", "graphql", "-f", f"query={query}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        # Check for schema-related errors that indicate stale cache (#1209)
        # Check both stderr (human-readable) and stdout (JSON response) for patterns
        combined_output = (result.stderr + result.stdout).lower()
        schema_error_patterns = [
            "could not resolve",  # Invalid field/option ID
            "invalid field",
            "invalid option",
            "field not found",
            "option not found",
            "doesn't exist on type",  # GraphQL field doesn't exist error
            "undefinedfield",  # GraphQL error code (no space in JSON)
        ]
        is_schema_error = any(p in combined_output for p in schema_error_patterns)

        if is_schema_error and not _retry:
            # Invalidate cache and retry once with fresh schema (#1209)
            gh_post_module.PROJECT_SCHEMA_CACHE_FILE.unlink(missing_ok=True)
            print(
                "[gh_post] Schema error, retrying with fresh schema...",
                file=sys.stderr,
            )
            fresh_schema = gh_post_module._fetch_project_schema(real_gh)
            if fresh_schema:
                gh_post_module._save_schema_cache(fresh_schema)
                gh_post_module._set_project_fields(
                    real_gh, fresh_schema, item_id, repo_name, _retry=True
                )
                return
            # Fresh fetch failed, fall through to warning
            print(
                f"[gh_post] Warning: schema refresh failed: {result.stderr}",
                file=sys.stderr,
            )
        elif is_schema_error:
            # Already retried, give up
            print(
                f"[gh_post] Warning: schema error persists after refresh: {result.stderr}",
                file=sys.stderr,
            )
        else:
            print(
                f"[gh_post] Warning: failed to set project fields: {result.stderr}",
                file=sys.stderr,
            )


def _extract_issue_info(issue_url: str) -> tuple[str | None, str | None]:
    """Extract (issue_number, repo) from GitHub issue URL.

    Args:
        issue_url: GitHub issue URL, e.g., https://github.com/owner/repo/issues/123

    Returns:
        Tuple of (issue_number, repo) where repo is owner/repo format.
        issue_number is None if URL is invalid or doesn't contain /issues/.
        repo may be None if github.com/ not found (caller should use fallback).
    """
    issue_url = issue_url.strip().rstrip("/")
    if not issue_url:
        return None, None

    try:
        parsed = urlparse(issue_url)
    except Exception as e:
        debug_swallow("gh_post_project_extract_issue_info", e)
        return None, None

    # Validate domain is exactly github.com (#1216)
    # Reject: notgithub.com, github.com.evil.com, api.github.com
    if parsed.netloc != "github.com":
        return None, None

    # Parse path: /owner/repo/issues/123[/...]
    # Only use the path, not query or fragment (#1217)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 4 or path_parts[2] != "issues":
        return None, None

    owner, repo, _, issue_str = path_parts[:4]

    # Validate issue number is numeric
    if not issue_str.isdigit():
        return None, None

    # Validate owner/repo format: non-empty, no spaces
    if not owner or not repo or " " in owner or " " in repo:
        return None, None

    return issue_str, f"{owner}/{repo}"


def _add_issue_to_project_by_url(
    real_gh: str,
    issue_url: str,
    repo_hint: str | None,
    source: str = "issue",
) -> None:
    """Add issue to project from URL with optional repo hint.

    Common logic for both direct create and sync replay paths.

    Args:
        real_gh: Path to real gh binary
        issue_url: GitHub issue URL from create output
        repo_hint: Optional repo override (from parsed args or queue)
        source: Log prefix ("issue" or "synced issue")
    """
    gh_post_module = _gh_post()

    issue_number, url_repo = gh_post_module._extract_issue_info(issue_url)
    if not issue_number:
        return

    # Determine repo: hint > URL > current
    repo = (
        repo_hint
        or url_repo
        or f"{gh_post_module.PROJECT_OWNER}/{gh_post_module._get_current_repo_name()}"
    )
    repo_full = gh_post_module._ensure_full_repo_name(repo)
    repo_name = repo_full.split("/")[-1]

    # Fetch node_id and add to project
    node_id = gh_post_module._get_issue_node_id(real_gh, repo_full, issue_number)
    if not node_id:
        print(
            f"[gh_post] Warning: could not get node_id for {source} #{issue_number}",
            file=sys.stderr,
        )
        return

    success = gh_post_module._add_issue_to_project(real_gh, node_id, repo_name)
    if success:
        print(
            f"[gh_post] Added {source} #{issue_number} to Project #1",
            file=sys.stderr,
        )


def _auto_add_to_project_from_sync(
    real_gh: str, create_output: str, repo: str | None
) -> None:
    """Add issue to project when replayed from sync queue."""
    gh_post_module = _gh_post()

    gh_post_module._add_issue_to_project_by_url(
        real_gh, create_output, repo, "synced issue"
    )


def _auto_add_to_project(real_gh: str, create_output: str, parsed: dict) -> None:
    """Add newly created issue to GitHub Project #1.

    Delegates to _add_issue_to_project_by_url with repo from parsed args.
    Failures are logged but don't block issue creation.
    """
    gh_post_module = _gh_post()

    gh_post_module._add_issue_to_project_by_url(
        real_gh, create_output, parsed.get("repo"), "issue"
    )

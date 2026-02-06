# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/rest_fallback.py - REST API Fallback for GitHub CLI

Encapsulates REST preference logic and issue list/view conversion.
When GraphQL is rate-limited, converts gh commands to REST API calls.

Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Methods extracted from RateLimiter:
- _should_prefer_rest_for_quota, _prefer_issue_rest
- _issue_list_rest_fallback, _issue_view_rest_fallback
- _is_graphql_rate_limit_error, _is_silent_list_rate_limit
- _has_json_flag, _is_issue_list_command, _is_issue_view_command
- _extract_json_fields
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
from collections.abc import Callable
from typing import Any

from ai_template_scripts.gh_rate_limit.limiter import debug_log
from ai_template_scripts.gh_rate_limit.secondary_backoff import (
    is_secondary_rate_limit_error,
)
from ai_template_scripts.result import Result

# Default load balance threshold (70% of 5000)
DEFAULT_LOAD_BALANCE_THRESHOLD = 3500


def is_graphql_rate_limit_error(output: str) -> bool:
    """Check if output indicates GraphQL rate limiting (primary or secondary).

    Detects both primary rate limits (quota exceeded) and secondary rate limits
    (abuse detection). Both require fallback handling.
    """
    if not output:
        return False
    lower = output.lower()
    # Primary rate limit patterns
    if "rate limit" in lower or "api rate limit" in lower:
        return True
    # Secondary rate limit patterns (delegated to secondary_backoff module)
    if is_secondary_rate_limit_error(output):
        return True
    return False


def is_any_rate_limit_error(output: str) -> bool:
    """Check if output indicates any type of rate limiting.

    Alias for is_graphql_rate_limit_error that makes the combined check
    explicit in calling code.
    """
    return is_graphql_rate_limit_error(output)


def has_json_flag(args: list[str]) -> bool:
    """Check if args request JSON output."""
    return "--json" in args or any(arg.startswith("--json=") for arg in args)


def is_issue_list_command(args: list[str]) -> bool:
    """Check if args represent a gh issue list command."""
    return len(args) >= 2 and args[0] == "issue" and args[1] == "list"


def is_issue_view_command(args: list[str]) -> bool:
    """Check if args represent a gh issue view command."""
    return len(args) >= 2 and args[0] == "issue" and args[1] == "view"


def is_issue_search_command(args: list[str]) -> bool:
    """Check if args represent a gh issue list command with --search.

    The --search flag triggers GraphQL usage in gh CLI. When GraphQL
    is rate-limited, we can fall back to REST /search/issues endpoint.
    """
    if not is_issue_list_command(args):
        return False
    for arg in args:
        if arg in ("--search", "-S"):
            return True
        if arg.startswith("--search="):
            return True
    return False


def extract_json_fields(args: list[str]) -> list[str]:
    """Extract --json field list from command args.

    Handles both `--json fields` and `--json=fields` formats.
    Returns empty list if no --json flag present.
    """
    for j, arg in enumerate(args):
        if arg == "--json" and j + 1 < len(args):
            return args[j + 1].split(",")
        if arg.startswith("--json="):
            return arg.split("=", 1)[1].split(",")
    return []


def extract_repo_from_args(args: list[str]) -> str | None:
    """Extract repo override from command args like ['-R', 'owner/repo']."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-R", "--repo") and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--repo="):
            return arg.split("=", 1)[1]
        if arg.startswith("-R") and len(arg) > 2:
            # Handle -Rowner/repo (no space)
            return arg[2:]
        i += 1
    return None


# REST/GraphQL field name mapping for issue list/search
_ISSUE_FIELD_MAPPING = {
    "state": ".state",
    "labels": "[.labels[] | {name: .name}]",
    "title": ".title",
    "number": ".number",
    "body": ".body",
    "createdAt": ".created_at",
    "closedAt": ".closed_at",
    "author": ".user.login",
}


def _parse_issue_list_args(
    args: list[str],
) -> Result[dict[str, Any]]:
    """Parse gh issue list arguments for REST fallback.

    Returns Result with parsed params dict or failure/skip reason.
    Params dict contains: state, labels, limit, json_fields, repo_override, user_jq.
    """
    state = "open"
    labels: list[str] = []
    limit = 30
    json_fields: list[str] = []
    repo_override: str | None = None
    user_jq: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--state", "-s") and i + 1 < len(args):
            state = args[i + 1]
            i += 2
        elif arg.startswith("--state="):
            state = arg.split("=", 1)[1]
            i += 1
        elif arg in ("--label", "-l") and i + 1 < len(args):
            labels.append(args[i + 1])
            i += 2
        elif arg.startswith("--label="):
            labels.append(arg.split("=", 1)[1])
            i += 1
        elif arg in ("--limit", "-L") and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                return Result.failure("invalid_limit")
            i += 2
        elif arg.startswith("--limit="):
            try:
                limit = int(arg.split("=", 1)[1])
            except ValueError:
                return Result.failure("invalid_limit")
            i += 1
        elif arg in ("--repo", "-R") and i + 1 < len(args):
            repo_override = args[i + 1]
            i += 2
        elif arg.startswith("--repo="):
            repo_override = arg.split("=", 1)[1]
            i += 1
        elif arg == "--json" and i + 1 < len(args):
            json_fields = args[i + 1].split(",")
            i += 2
        elif arg.startswith("--json="):
            json_fields = arg.split("=", 1)[1].split(",")
            i += 1
        elif arg in ("--jq", "-q") and i + 1 < len(args):
            user_jq = args[i + 1]
            i += 2
        elif arg.startswith("--jq="):
            user_jq = arg.split("=", 1)[1]
            i += 1
        elif arg in ("issue", "list"):
            i += 1
        elif arg.startswith("-"):
            return Result.skip(f"unsupported_flag:{arg}")
        else:
            i += 1

    if limit < 1:
        return Result.failure("invalid_limit")
    if state not in ("open", "closed", "all"):
        return Result.failure(f"invalid_state:{state}")

    return Result.success({
        "state": state,
        "labels": labels,
        "limit": limit,
        "json_fields": json_fields,
        "repo_override": repo_override,
        "user_jq": user_jq,
    })


def _build_issue_list_jq_filter(
    json_fields: list[str], user_jq: str | None
) -> str:
    """Build jq filter for issue list REST response."""
    if json_fields:
        jq_parts = []
        for field in json_fields:
            if field in _ISSUE_FIELD_MAPPING:
                jq_parts.append(f"{field}: {_ISSUE_FIELD_MAPPING[field]}")
            else:
                jq_parts.append(f"{field}: .{field}")
        jq_filter = (
            f"[.[] | select(.pull_request == null) | {{{', '.join(jq_parts)}}}]"
        )
    else:
        jq_filter = "[.[] | select(.pull_request == null)]"
    if user_jq:
        jq_filter = f"({jq_filter}) | {user_jq}"
    return jq_filter


def _build_search_jq_filter(json_fields: list[str], user_jq: str | None) -> str:
    """Build jq filter for search REST response.

    Search API returns {items: [...], total_count: N} structure.
    """
    if json_fields:
        jq_parts = []
        for field in json_fields:
            if field in _ISSUE_FIELD_MAPPING:
                jq_parts.append(f"{field}: {_ISSUE_FIELD_MAPPING[field]}")
            else:
                jq_parts.append(f"{field}: .{field}")
        jq_filter = f"[.items[] | {{{', '.join(jq_parts)}}}]"
    else:
        jq_filter = ".items"
    if user_jq:
        jq_filter = f"({jq_filter}) | {user_jq}"
    return jq_filter


def _parse_issue_search_args(
    args: list[str],
) -> Result[dict[str, Any]]:
    """Parse gh issue list --search arguments for REST fallback.

    Extends issue_list parsing with search query extraction.
    Returns Result with parsed params dict or failure/skip reason.
    """
    search_query: str | None = None
    state = "open"
    labels: list[str] = []
    limit = 30
    json_fields: list[str] = []
    repo_override: str | None = None
    user_jq: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--search", "-S") and i + 1 < len(args):
            search_query = args[i + 1]
            i += 2
        elif arg.startswith("--search="):
            search_query = arg.split("=", 1)[1]
            i += 1
        elif arg in ("--state", "-s") and i + 1 < len(args):
            state = args[i + 1]
            i += 2
        elif arg.startswith("--state="):
            state = arg.split("=", 1)[1]
            i += 1
        elif arg in ("--label", "-l") and i + 1 < len(args):
            labels.append(args[i + 1])
            i += 2
        elif arg.startswith("--label="):
            labels.append(arg.split("=", 1)[1])
            i += 1
        elif arg in ("--limit", "-L") and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                return Result.failure("invalid_limit")
            i += 2
        elif arg.startswith("--limit="):
            try:
                limit = int(arg.split("=", 1)[1])
            except ValueError:
                return Result.failure("invalid_limit")
            i += 1
        elif arg in ("--repo", "-R") and i + 1 < len(args):
            repo_override = args[i + 1]
            i += 2
        elif arg.startswith("--repo="):
            repo_override = arg.split("=", 1)[1]
            i += 1
        elif arg == "--json" and i + 1 < len(args):
            json_fields = args[i + 1].split(",")
            i += 2
        elif arg.startswith("--json="):
            json_fields = arg.split("=", 1)[1].split(",")
            i += 1
        elif arg in ("--jq", "-q") and i + 1 < len(args):
            user_jq = args[i + 1]
            i += 2
        elif arg.startswith("--jq="):
            user_jq = arg.split("=", 1)[1]
            i += 1
        elif arg in ("issue", "list"):
            i += 1
        elif arg.startswith("-"):
            # Unknown flag - skip (may fail but better than rejecting)
            i += 1
        else:
            i += 1

    if not search_query:
        return Result.skip("missing_search_query")
    if limit < 1:
        return Result.failure("invalid_limit")
    if state not in ("open", "closed", "all"):
        return Result.failure(f"invalid_state:{state}")

    return Result.success({
        "search_query": search_query,
        "state": state,
        "labels": labels,
        "limit": limit,
        "json_fields": json_fields,
        "repo_override": repo_override,
        "user_jq": user_jq,
    })


def _build_search_query(
    owner_repo: str, search_query: str, state: str, labels: list[str]
) -> str:
    """Build full search query string with qualifiers."""
    query_parts = [f"repo:{owner_repo}", "is:issue", search_query]

    if state == "open":
        query_parts.append("is:open")
    elif state == "closed":
        query_parts.append("is:closed")

    for label in labels:
        query_parts.append(f'label:"{label}"')

    return " ".join(query_parts)


def _process_paginated_response(
    stdout: str, use_paginate: bool, limit: int
) -> tuple[str | None, bool]:
    """Process paginated REST response.

    Returns (processed_stdout, parse_error).
    processed_stdout is None if no processing needed.
    """
    stdout = stdout.strip()
    if not stdout:
        return None, False

    all_items: list | None = None
    parse_error = False

    if use_paginate and "\n" in stdout:
        all_items = []
        for line in stdout.split("\n"):
            line = line.strip()
            if line:
                try:
                    all_items.extend(json.loads(line))
                except json.JSONDecodeError:
                    parse_error = True
    else:
        try:
            all_items = json.loads(stdout)
        except json.JSONDecodeError:
            parse_error = True

    if parse_error:
        return None, True

    if isinstance(all_items, list):
        if len(all_items) > limit:
            all_items = all_items[:limit]
        return json.dumps(all_items) + "\n", False

    return None, False


class IssueRestFallback:
    """REST fallback for gh issue list/view commands (#1671, #1074).

    When GraphQL quota is low or exhausted, converts gh CLI commands
    to direct REST API calls.

    Args:
        get_real_gh: Callable returning path to real gh binary.
        get_owner_repo: Callable returning owner/repo from args or git origin.
        rate_cache: Dict mapping resource name to RateLimitInfo.
        load_balance_threshold: GraphQL remaining below this prefers REST.
    """

    def __init__(
        self,
        get_real_gh: Callable[[], str],
        get_owner_repo: Callable[[str | None], str | None],
        rate_cache: dict[str, Any],
        load_balance_threshold: int | None = None,
    ) -> None:
        self._get_real_gh = get_real_gh
        self._get_owner_repo = get_owner_repo
        self._rate_cache = rate_cache
        self._load_balance_threshold = load_balance_threshold or int(
            os.environ.get(
                "AIT_GH_LOAD_BALANCE_THRESHOLD", str(DEFAULT_LOAD_BALANCE_THRESHOLD)
            )
        )

    def should_prefer_rest_for_quota(self, log: bool = True) -> bool:
        """Check if REST should be preferred based on quota levels (#1143).

        Proactive load balancing: when GraphQL quota is running low but REST
        has more headroom, prefer REST to avoid exhausting GraphQL.

        Args:
            log: Whether to emit a log message when preferring REST.
                 Set to False when just checking status without acting on it.

        Returns True if REST should be preferred, False otherwise.
        """
        graphql_info = self._rate_cache.get("graphql")
        core_info = self._rate_cache.get("core")

        # Need both quotas to make a decision
        if not graphql_info or not core_info:
            return False

        graphql_remaining = graphql_info.remaining
        core_remaining = core_info.remaining

        # If GraphQL is getting low and REST has more headroom, prefer REST
        if graphql_remaining < self._load_balance_threshold:
            if core_remaining > graphql_remaining:
                if log:
                    print(
                        f"gh_rate_limit: REST preferred "
                        f"(GQL: {graphql_remaining}, REST: {core_remaining})",
                        file=sys.stderr,
                    )
                return True

        return False

    def has_search_quota(self, log: bool = True) -> bool:
        """Check if search API quota is available (#1869).

        The search API has its own quota (30 requests/minute).
        Returns True if we can make search requests, False otherwise.

        Args:
            log: Whether to emit a log message when quota is exhausted.
        """
        search_info = self._rate_cache.get("search")
        if not search_info:
            return True  # No data - assume OK

        # Search quota is much smaller (30/min vs 5000/hr)
        # Use lower threshold: block when < 2 remaining (from limiter.py THRESHOLDS)
        if search_info.remaining < 2:
            if log:
                print(
                    f"gh_rate_limit: search quota exhausted "
                    f"({search_info.remaining}/{search_info.limit})",
                    file=sys.stderr,
                )
            return False

        # Warn when getting low (< 10 remaining) to provide early visibility (#1869)
        if search_info.remaining < 10 and log:
            print(
                f"gh_rate_limit: search quota LOW "
                f"({search_info.remaining}/{search_info.limit})",
                file=sys.stderr,
            )

        return True

    def prefer_issue_rest(self, args: list[str]) -> bool:
        """Prefer REST for routine issue reads to reduce GraphQL usage (#1074).

        REST is preferred when:
        1. AIT_GH_ISSUE_REST env var is set
        2. Command has --json flag (REST can handle JSON output)
        3. GraphQL quota is running low (#1143 proactive load balancing)

        For search commands, also checks search quota is available (#1869).
        """
        # Search commands require search quota to be available (#1869)
        if is_issue_search_command(args):
            if not self.has_search_quota():
                return False
            # Search commands with JSON flag can use REST search API
            if has_json_flag(args):
                return True
            # Search commands without JSON can't be handled by REST
            return False

        if not (is_issue_list_command(args) or is_issue_view_command(args)):
            return False
        # Explicit env var override
        force = os.environ.get("AIT_GH_ISSUE_REST", "").lower()
        if force in ("1", "true", "yes"):
            return True
        # JSON flag means REST can handle it
        if has_json_flag(args):
            return True
        # Proactive load balancing (#1143)
        if self.should_prefer_rest_for_quota():
            return True
        return False

    def is_silent_list_rate_limit(
        self, result: subprocess.CompletedProcess, args: list[str]
    ) -> bool:
        """Detect silent rate limit failure for gh issue list (#1671).

        gh CLI may return exit 0 with rate limit error in stdout/stderr
        instead of a proper error exit code. This detects that case.

        Returns True if:
        - Command is gh issue list (with --json flag)
        - Exit code is 0
        - stdout is NOT valid JSON array (error replaced the output)
        - Output contains rate limit error message
        """
        if result.returncode != 0:
            return False  # Already detected as error

        if not is_issue_list_command(args):
            return False

        output = (result.stdout or "") + (result.stderr or "")
        if not output.strip():
            return False

        if has_json_flag(args):
            # If stdout is valid JSON array, the API call succeeded
            # This avoids false positives when issue content contains "rate limit"
            stdout = result.stdout or ""
            if stdout.strip().startswith("["):
                try:
                    json.loads(stdout)
                    return False  # Valid JSON array - API succeeded
                except json.JSONDecodeError:
                    pass  # Invalid JSON - check for rate limit message

            # Check for rate limit message in stdout or stderr
            if is_graphql_rate_limit_error(output):
                print(
                    "gh_rate_limit: detected silent rate limit in issue list output",
                    file=sys.stderr,
                )
                return True
            return False

        # Non-JSON: treat a standalone rate limit error as failure
        first_line = ""
        for line in output.splitlines():
            if line.strip():
                first_line = line.strip()
                break
        if not first_line:
            return False
        first_lower = first_line.lower()
        if first_lower.startswith(
            ("graphql:", "api rate limit")
        ) and is_graphql_rate_limit_error(first_line):
            print(
                "gh_rate_limit: detected silent rate limit in issue list output",
                file=sys.stderr,
            )
            return True

        return False

    def issue_list_rest_fallback(
        self, args: list[str], timeout: float
    ) -> Result[subprocess.CompletedProcess]:
        """Execute gh issue list via REST API fallback (#1671).

        Converts gh issue list args to REST API call.
        Returns Result with explicit skip/failure reasons.
        """
        # Parse args
        parse_result = _parse_issue_list_args(args)
        if not parse_result.ok:
            return parse_result  # type: ignore

        params = parse_result.value
        state = params["state"]
        labels = params["labels"]
        limit = params["limit"]
        json_fields = params["json_fields"]
        repo_override = params["repo_override"]
        user_jq = params["user_jq"]

        # Build REST API URL
        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return Result.skip("missing_repo")

        api_path = f"/repos/{owner_repo}/issues"
        url_params = [f"per_page={min(limit, 100)}"]
        if state != "open":
            url_params.append(f"state={state}")
        if labels:
            url_params.append(f"labels={','.join(labels)}")
        api_url = api_path + "?" + "&".join(url_params)

        # Build jq filter
        jq_filter = _build_issue_list_jq_filter(json_fields, user_jq)

        # Execute REST API call
        try:
            api_args = ["api"]
            use_paginate = limit > 100
            if use_paginate:
                api_args.append("--paginate")
            api_args.extend([api_url, "-q", jq_filter])
            result = subprocess.run(
                [self._get_real_gh()] + api_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                error_output = (result.stderr or "") + (result.stdout or "")
                if is_graphql_rate_limit_error(error_output):
                    return Result.failure("rate_limited", value=result)
                return Result.failure(f"rest_error:{result.returncode}", value=result)

            # Process paginated response
            processed_stdout, parse_error = _process_paginated_response(
                result.stdout or "", use_paginate, limit
            )
            if parse_error:
                return Result.failure("invalid_json", value=result)
            if processed_stdout is not None:
                result = subprocess.CompletedProcess(
                    args=result.args,
                    returncode=0,
                    stdout=processed_stdout,
                    stderr=result.stderr,
                )

            # Buffer diagnostic message into result.stderr to prevent interleaving (#1773)
            # gh_wrapper.py outputs stdout first, then stderr, ensuring clean separation
            rest_msg = "gh_rate_limit: GraphQL rate-limited, used REST fallback\n"
            result = subprocess.CompletedProcess(
                args=result.args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=(result.stderr or "") + rest_msg,
            )
            return Result.success(result)

        except subprocess.TimeoutExpired:
            timeout_result = subprocess.CompletedProcess(
                args=["gh"] + api_args,
                returncode=-1,
                stdout="",
                stderr=f"REST fallback timeout after {timeout}s",
            )
            return Result.failure("timeout", value=timeout_result)
        except Exception as e:
            debug_log(f"issue_list_rest_fallback failed: {e}")
            return Result.failure(f"exception:{type(e).__name__}")

    def issue_view_rest_fallback(
        self,
        args: list[str],
        timeout: float,
        parse_etag: Callable[[str], str | None] | None = None,
        parse_body: Callable[[str], str] | None = None,
        set_cached: Callable[[list[str], str, str | None], None] | None = None,
    ) -> Result[subprocess.CompletedProcess]:
        """Execute gh issue view via REST API fallback.

        Args:
            args: Command arguments
            timeout: Request timeout
            parse_etag: Optional callable to parse ETag from response headers
            parse_body: Optional callable to parse body from response with headers
            set_cached: Optional callable to cache result with ETag
        """
        issue_num: str | None = None
        repo_override: str | None = None
        json_fields: list[str] = []

        unsupported_flags = {"--comments", "--web", "--template"}

        i = 2  # Skip "issue view"
        while i < len(args):
            arg = args[i]
            if arg in ("--repo", "-R") and i + 1 < len(args):
                repo_override = args[i + 1]
                i += 2
            elif arg.startswith("--repo="):
                repo_override = arg.split("=", 1)[1]
                i += 1
            elif arg == "--json" and i + 1 < len(args):
                json_fields = args[i + 1].split(",")
                i += 2
            elif arg.startswith("--json="):
                json_fields = arg.split("=", 1)[1].split(",")
                i += 1
            elif arg in ("--jq", "-q") and i + 1 < len(args):
                return Result.skip("unsupported_flag:--jq")
            elif arg.startswith("--jq="):
                return Result.skip("unsupported_flag:--jq")
            elif arg in unsupported_flags:
                return Result.skip(f"unsupported_flag:{arg}")
            elif arg.startswith("-"):
                return Result.skip(f"unsupported_flag:{arg}")
            else:
                if issue_num is None:
                    issue_num = arg
                i += 1

        if not issue_num or not json_fields:
            if not issue_num:
                return Result.skip("missing_issue_num")
            return Result.skip("missing_json_fields")

        issue_num = issue_num.lstrip("#")
        if not issue_num.isdigit():
            # User error - invalid issue number (#1755)
            return Result.failure("invalid_issue_num")

        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return Result.skip("missing_repo")

        api_url = f"/repos/{owner_repo}/issues/{issue_num}"

        def _camel_to_snake(name: str) -> str:
            out = []
            for ch in name:
                if ch.isupper():
                    out.append("_")
                    out.append(ch.lower())
                else:
                    out.append(ch)
            return "".join(out)

        try:
            # Use -i to include headers for ETag capture (#1674)
            api_args = ["api", "-i", api_url]
            result = subprocess.run(
                [self._get_real_gh()] + api_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                error_output = (result.stderr or "") + (result.stdout or "")
                if is_graphql_rate_limit_error(error_output):
                    return Result.failure("rate_limited", value=result)
                return Result.failure(f"rest_error:{result.returncode}", value=result)

            # Extract ETag from headers (#1674)
            etag = parse_etag(result.stdout) if parse_etag else None
            body = parse_body(result.stdout) if parse_body else result.stdout

            try:
                data = json.loads(body or "{}")
            except json.JSONDecodeError:
                return Result.failure("invalid_json", value=result)

            output: dict[str, Any] = {}
            for field in json_fields:
                if field == "labels":
                    output[field] = data.get("labels", [])
                elif field == "author":
                    user = data.get("user") or {}
                    output[field] = {
                        "login": user.get("login"),
                        "id": user.get("node_id") or user.get("id"),
                        "is_bot": user.get("type") == "Bot",
                        "name": user.get("name"),
                    }
                elif field == "url":
                    output[field] = data.get("html_url") or data.get("url")
                elif field == "createdAt":
                    output[field] = data.get("created_at")
                elif field == "closedAt":
                    output[field] = data.get("closed_at")
                elif field == "updatedAt":
                    output[field] = data.get("updated_at")
                else:
                    value = data.get(field)
                    if value is None:
                        value = data.get(_camel_to_snake(field))
                    output[field] = value

            # Add trailing newline to prevent stderr interleaving (#1722)
            final_stdout = json.dumps(output) + "\n"

            # Cache with ETag for conditional requests (#1674)
            if set_cached:
                set_cached(args, final_stdout, etag)

            # Buffer diagnostic message into result.stderr to prevent interleaving (#1773)
            # gh_wrapper.py outputs stdout first, then stderr, ensuring clean separation
            rest_msg = "gh_rate_limit: GraphQL rate-limited, used REST fallback\n"
            result = subprocess.CompletedProcess(
                args=result.args,
                returncode=0,
                stdout=final_stdout,
                stderr=(result.stderr or "") + rest_msg,
            )
            return Result.success(result)
        except subprocess.TimeoutExpired:
            timeout_result = subprocess.CompletedProcess(
                args=["gh"] + api_args,
                returncode=-1,
                stdout="",
                stderr=f"REST fallback timeout after {timeout}s",
            )
            return Result.failure("timeout", value=timeout_result)
        except Exception as e:
            debug_log(f"issue_view_rest_fallback failed: {e}")
            return Result.failure(f"exception:{type(e).__name__}")

    def issue_search_rest_fallback(
        self, args: list[str], timeout: float
    ) -> Result[subprocess.CompletedProcess]:
        """Execute gh issue list --search via REST search API fallback (#1777).

        When GraphQL is rate-limited, falls back to GitHub REST search API.
        The search API has its own rate limit bucket (30 requests/minute).

        Args:
            args: Command arguments (should be gh issue list --search ...)
            timeout: Request timeout in seconds

        Returns:
            Result with CompletedProcess on success, failure/skip on error.

        Note: REST search API returns different structure than GraphQL:
        - Results wrapped in {"items": [...], "total_count": N}
        - Issues and PRs mixed (filter with is:issue in query)
        - Different field names (created_at vs createdAt)
        """
        # Parse args
        parse_result = _parse_issue_search_args(args)
        if not parse_result.ok:
            return parse_result  # type: ignore

        params = parse_result.value
        search_query = params["search_query"]
        state = params["state"]
        labels = params["labels"]
        limit = params["limit"]
        json_fields = params["json_fields"]
        repo_override = params["repo_override"]
        user_jq = params["user_jq"]

        # Build search query with repo qualifier
        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return Result.skip("missing_repo")

        # Build and encode query
        full_query = _build_search_query(owner_repo, search_query, state, labels)
        encoded_query = urllib.parse.quote(full_query, safe="")
        api_url = f"/search/issues?q={encoded_query}&per_page={min(limit, 100)}"

        # Build jq filter
        jq_filter = _build_search_jq_filter(json_fields, user_jq)

        # Execute REST API call
        api_args = ["api", api_url, "-q", jq_filter]
        try:
            result = subprocess.run(
                [self._get_real_gh()] + api_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                error_output = (result.stderr or "") + (result.stdout or "")
                if is_graphql_rate_limit_error(error_output):
                    return Result.failure("rate_limited", value=result)
                if "secondary rate limit" in error_output.lower():
                    return Result.failure("search_rate_limited", value=result)
                return Result.failure(f"rest_error:{result.returncode}", value=result)

            # Parse and limit results
            stdout = result.stdout.strip()
            if stdout:
                try:
                    all_items = json.loads(stdout)
                except json.JSONDecodeError:
                    return Result.failure("invalid_json", value=result)

                if isinstance(all_items, list):
                    if len(all_items) > limit:
                        all_items = all_items[:limit]
                    # Add trailing newline to prevent stderr interleaving (#1722)
                    list_stdout = json.dumps(all_items) + "\n"
                    result = subprocess.CompletedProcess(
                        args=result.args,
                        returncode=0,
                        stdout=list_stdout,
                        stderr=result.stderr,
                    )

            # Buffer diagnostic message into result.stderr (#1773)
            rest_msg = (
                "gh_rate_limit: GraphQL rate-limited, used REST search fallback\n"
            )
            result = subprocess.CompletedProcess(
                args=result.args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=(result.stderr or "") + rest_msg,
            )
            return Result.success(result)

        except subprocess.TimeoutExpired:
            timeout_result = subprocess.CompletedProcess(
                args=["gh"] + api_args,
                returncode=-1,
                stdout="",
                stderr=f"REST search fallback timeout after {timeout}s",
            )
            return Result.failure("timeout", value=timeout_result)
        except Exception as e:
            debug_log(f"issue_search_rest_fallback failed: {e}")
            return Result.failure(f"exception:{type(e).__name__}")

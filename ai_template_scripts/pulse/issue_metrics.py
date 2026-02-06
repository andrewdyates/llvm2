#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
pulse.issue_metrics - GitHub issue counts and velocity tracking.

Functions for querying GitHub API for issue counts, velocities, and blocked issue analysis.

Part of #404: pulse.py module split
"""

import json
from datetime import UTC, datetime, timedelta

try:
    from ai_template_scripts.gh_graphql import graphql_batch
    from ai_template_scripts.result import Result
    from ai_template_scripts.subprocess_utils import run_cmd, run_cmd_with_retry
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.gh_graphql import graphql_batch
    from ai_template_scripts.result import Result
    from ai_template_scripts.subprocess_utils import run_cmd, run_cmd_with_retry

from .constants import BLOCKER_PATTERN, ISSUE_REF_PATTERN


def _empty_issue_counts() -> dict[str, int]:
    """Return default issue counts with all states zeroed."""
    return {"open": 0, "in_progress": 0, "blocked": 0, "closed": 0}


def _get_repo_owner_and_name() -> tuple[str, str] | None:
    """Get owner and repo name from git remote.

    Returns (owner, repo) tuple or None if unable to determine.

    REQUIRES: Current directory is within a git repository
    ENSURES: Returns (owner, repo) tuple or None on failure
    ENSURES: Never raises (returns None on error)
    """
    result = run_cmd(["git", "remote", "get-url", "origin"])
    if not result.ok or not result.stdout.strip():
        return None
    url = result.stdout.strip().removesuffix(".git")
    # Handle both HTTPS and SSH formats:
    # - https://github.com/owner/repo
    # - git@github.com:owner/repo
    if "github.com:" in url:
        # SSH format
        parts = url.split("github.com:")[-1].split("/")
    else:
        # HTTPS format
        parts = url.split("/")[-2:]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None


def _has_in_progress_label(labels: list[str]) -> bool:
    """Return True if labels include in-progress or any role-specific claims.

    Handles: in-progress, legacy in-progress-W*, in-progress-P*, in-progress-R*, in-progress-M*

    REQUIRES: labels is a list of strings
    ENSURES: Returns True iff any label matches in-progress or in-progress-*
    ENSURES: Never raises
    """
    for label in labels:
        if label == "in-progress" or label.startswith("in-progress-"):
            return True
    return False


def _get_issue_counts_graphql(owner: str, repo: str) -> Result[dict[str, int]]:
    """Get issue counts using GraphQL for accurate totals (no pagination limit).

    This queries totalCount directly rather than fetching all issues,
    so it works correctly for repos with >500 issues.

    REQUIRES: owner and repo are non-empty strings
    ENSURES: Returns Result with 'open', 'closed', 'blocked', 'in_progress' counts
    ENSURES: On API failure, returns Result.failure with partial counts
    ENSURES: All count values are non-negative integers
    """
    counts = _empty_issue_counts()

    # GraphQL query for issue counts by state and label
    # Note: labels filter in GraphQL requires exact match, so we query
    # in-progress and legacy in-progress-* labels separately (W1-5, P/R/M1-3)
    query = """
    query($owner: String!, $repo: String!) {
      repository(owner: $owner, name: $repo) {
        open: issues(states: OPEN) { totalCount }
        closed: issues(states: CLOSED) { totalCount }
        blocked: issues(states: OPEN, labels: ["blocked"]) { totalCount }
        inProgress: issues(states: OPEN, labels: ["in-progress"]) { totalCount }
        inProgressW1: issues(states: OPEN, labels: ["in-progress-W1"]) { totalCount }
        inProgressW2: issues(states: OPEN, labels: ["in-progress-W2"]) { totalCount }
        inProgressW3: issues(states: OPEN, labels: ["in-progress-W3"]) { totalCount }
        inProgressW4: issues(states: OPEN, labels: ["in-progress-W4"]) { totalCount }
        inProgressW5: issues(states: OPEN, labels: ["in-progress-W5"]) { totalCount }
        inProgressP1: issues(states: OPEN, labels: ["in-progress-P1"]) { totalCount }
        inProgressP2: issues(states: OPEN, labels: ["in-progress-P2"]) { totalCount }
        inProgressP3: issues(states: OPEN, labels: ["in-progress-P3"]) { totalCount }
        inProgressR1: issues(states: OPEN, labels: ["in-progress-R1"]) { totalCount }
        inProgressR2: issues(states: OPEN, labels: ["in-progress-R2"]) { totalCount }
        inProgressR3: issues(states: OPEN, labels: ["in-progress-R3"]) { totalCount }
        inProgressM1: issues(states: OPEN, labels: ["in-progress-M1"]) { totalCount }
        inProgressM2: issues(states: OPEN, labels: ["in-progress-M2"]) { totalCount }
        inProgressM3: issues(states: OPEN, labels: ["in-progress-M3"]) { totalCount }
      }
    }
    """

    result = run_cmd_with_retry(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo}",
        ],
        timeout=60,
        retries=2,
    )

    if not result.ok:
        error_msg = (
            result.stderr.strip() if result.stderr.strip() else "GraphQL query failed"
        )
        return Result.failure(error_msg, value=counts)

    try:
        data = json.loads(result.stdout)
        repo_data = data.get("data", {}).get("repository", {})
        if not repo_data:
            return Result.failure("No repository data in response", value=counts)

        counts["open"] = repo_data.get("open", {}).get("totalCount", 0)
        counts["closed"] = repo_data.get("closed", {}).get("totalCount", 0)
        counts["blocked"] = repo_data.get("blocked", {}).get("totalCount", 0)

        # Sum all in-progress variants: in-progress + W1-5 + P/R/M1-3
        in_progress_total = repo_data.get("inProgress", {}).get("totalCount", 0)
        for role in ["W", "P", "R", "M"]:
            # W has 5 workers, others have 3
            max_id = 6 if role == "W" else 4
            for i in range(1, max_id):
                in_progress_total += repo_data.get(f"inProgress{role}{i}", {}).get(
                    "totalCount", 0
                )
        counts["in_progress"] = in_progress_total

        # Adjust open count: subtract blocked and in-progress from open total
        # (GraphQL 'open' includes all open issues including blocked/in-progress)
        counts["open"] = max(
            0, counts["open"] - counts["blocked"] - counts["in_progress"]
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return Result.failure(f"Failed to parse GraphQL response: {e}", value=counts)

    return Result.success(counts)


def _get_closed_issue_count_graphql(owner: str, repo: str) -> Result[int]:
    """Get *closed issue* count via GraphQL totalCount (cheap).

    Avoids enumerating closed issues via REST pagination (can be huge).

    REQUIRES: owner and repo are non-empty strings
    ENSURES: Returns Result with closed issue count
    ENSURES: On failure, returns Result.failure with value=0
    """
    query = """
    query($owner: String!, $repo: String!) {
      repository(owner: $owner, name: $repo) {
        closed: issues(states: CLOSED) { totalCount }
      }
    }
    """

    # No retries: when GraphQL is exhausted, retrying just burns time.
    result = run_cmd(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo}",
        ],
        timeout=30,
    )

    if not result.ok:
        error_msg = (
            result.stderr.strip() if result.stderr.strip() else "GraphQL query failed"
        )
        return Result.failure(error_msg, value=0)

    try:
        data = json.loads(result.stdout)
        repo_data = data.get("data", {}).get("repository", {})
        if not repo_data:
            return Result.failure("No repository data in response", value=0)
        return Result.success(repo_data.get("closed", {}).get("totalCount", 0))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return Result.failure(f"Failed to parse GraphQL response: {e}", value=0)


def _is_graphql_rate_limited(error_msg: str) -> bool:
    """Check if error message indicates GraphQL rate limiting.

    REQUIRES: error_msg is a string (may be empty)
    ENSURES: Returns True iff message indicates rate limiting
    ENSURES: Never raises
    """
    if not error_msg:
        return False
    lower = error_msg.lower()
    return "rate limit" in lower or "api rate limit" in lower


def _get_open_issue_counts_rest(owner: str, repo: str) -> Result[dict[str, int]]:
    """Compute open/blocked/in-progress counts by scanning *open* issues via REST.

    Rationale: scanning open issues is usually cheap, and avoids costly GraphQL
    label-count queries (which can quickly exhaust GraphQL quota).

    REQUIRES: owner and repo are non-empty strings
    ENSURES: Returns Result with 'open', 'blocked', 'in_progress' counts
    ENSURES: 'closed' is always 0 (not computed here)
    ENSURES: On API failure, returns Result.failure with partial counts
    """
    counts = _empty_issue_counts()

    result = run_cmd_with_retry(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=open&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {labels: [(.labels // [])[].name]}]",
        ],
        timeout=60,
        retries=2,
    )
    if not result.ok:
        error_msg = result.stderr.strip() if result.stderr else "REST API failed"
        return Result.failure(error_msg, value=counts)
    if not result.stdout.strip():
        return Result.success(counts)

    try:
        # gh api --paginate with jq outputs multiple JSON arrays, one per line
        all_issues: list[dict] = []
        stdout = result.stdout.strip()
        if "\n" in stdout:
            for line in stdout.split("\n"):
                line = line.strip()
                if line:
                    all_issues.extend(json.loads(line))
        else:
            all_issues = json.loads(stdout)

        for issue in all_issues:
            raw_labels = issue.get("labels", [])
            labels = [
                lbl["name"] if isinstance(lbl, dict) else lbl for lbl in raw_labels
            ]
            blocked = "blocked" in labels
            in_progress = _has_in_progress_label(labels)
            if blocked:
                counts["blocked"] += 1
            if in_progress:
                counts["in_progress"] += 1
            if not blocked and not in_progress:
                counts["open"] += 1
    except (json.JSONDecodeError, TypeError):
        return Result.failure("REST JSON parse error", value=counts)

    return Result.success(counts)


def _get_issue_counts_rest(owner: str, repo: str) -> Result[dict[str, int]]:
    """Get issue counts using TRUE REST API (not gh issue list which uses GraphQL).

    Scans *open* issues for open/blocked/in-progress counts.

    Closed issue totals are expensive to enumerate via REST, so those are
    fetched via GraphQL totalCount in get_issue_counts().

    REQUIRES: owner and repo are non-empty strings
    ENSURES: Returns Result with 'open', 'blocked', 'in_progress' counts
    ENSURES: 'closed' is always 0 (not computed here - see get_issue_counts)
    """
    return _get_open_issue_counts_rest(owner, repo)


def _should_prefer_rest_issue_counts(gh_limits: dict | None) -> bool:
    """Check if we should skip GraphQL closed-count due to quota exhaustion.

    When GraphQL quota is fully exhausted (remaining=0), we skip the closed
    count query and return open-issue counts only. This avoids a guaranteed
    failure when GraphQL is rate-limited.

    REQUIRES: gh_limits is dict from _get_gh_rate_limits() or None
    ENSURES: Returns True if GraphQL quota is fully exhausted (remaining <= 0)
    ENSURES: Returns False if quota info unavailable (conservative: try GraphQL)
    """
    if not gh_limits:
        return False
    graphql = gh_limits.get("resources", {}).get("graphql", {})
    remaining = graphql.get("remaining")
    return isinstance(remaining, (int, float)) and remaining <= 0


def get_issue_counts(gh_limits: dict | None = None) -> Result[dict[str, int]]:
    """Get GitHub issue counts by state.

    Uses REST to compute open/blocked/in-progress by scanning open issues.
    Uses GraphQL only for closed totalCount (cheap) when quota permits.

    Args:
        gh_limits: Optional pre-fetched rate limits from _get_gh_rate_limits().
            If None, will fetch rate limits internally.

    History:
    - Originally REST-first to avoid GraphQL exhaustion (#1029)
    - Changed to GraphQL-first (#1356) because REST --paginate hangs on large repos
    - GraphQL label-count query caused frequent quota exhaustion; switched to
      REST-open scan + GraphQL-closed totalCount (#1781, tla2#803)

    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns Result with dict containing 'open', 'closed', 'blocked', 'in_progress' counts
    ENSURES: On API failure, returns Result.failure with partial counts
    ENSURES: All count values are non-negative integers
    """
    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return Result.failure(
            "Cannot determine repo owner/name", value=_empty_issue_counts()
        )

    owner, repo = repo_info

    # Prefer REST open-issue scan to avoid burning GraphQL quota.
    rest_result = _get_issue_counts_rest(owner, repo)
    if rest_result.ok:
        counts = rest_result.value or _empty_issue_counts()
        if not _should_prefer_rest_issue_counts(gh_limits):
            closed_result = _get_closed_issue_count_graphql(owner, repo)
            if closed_result.ok:
                counts["closed"] = closed_result.value or 0
        return Result.success(counts)

    # REST failed - fall back to GraphQL (may still fail under rate limiting).
    return _get_issue_counts_graphql(owner, repo)


def get_blocked_missing_reason() -> list[int]:
    """Get blocked issues that are missing a 'Blocked:' reason in the body.

    Returns list of issue numbers that have 'blocked' label but no 'Blocked:' line.

    Uses REST API instead of gh issue list (GraphQL) to avoid rate limit exhaustion.
    See #1079.

    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns list of issue numbers (non-negative integers)
    ENSURES: On API failure, returns empty list (never raises)
    """
    missing: list[int] = []

    # Get repo info for API call
    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return missing

    owner, repo = repo_info

    # Use REST API: /repos/{owner}/{repo}/issues?labels=blocked&state=open
    # Filter out PRs (have pull_request field) and extract number+body
    result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=open&labels=blocked&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {number, body}]",
        ],
        timeout=60,
    )

    if not result.ok or not result.stdout.strip():
        return missing

    try:
        # gh api --paginate with jq outputs multiple JSON arrays, one per line
        all_issues: list[dict] = []
        stdout = result.stdout.strip()

        if "\n" in stdout:
            # Multiple arrays from pagination - parse each line and flatten
            for line in stdout.split("\n"):
                line = line.strip()
                if line:
                    all_issues.extend(json.loads(line))
        else:
            # Single array (small repo or single page)
            all_issues = json.loads(stdout)

        for issue in all_issues:
            body = issue.get("body", "") or ""
            # Check if body contains "Blocked:" (case-insensitive)
            if "blocked:" not in body.lower():
                missing.append(issue.get("number", 0))
    except json.JSONDecodeError:
        pass

    return missing


def get_blocked_issue_list() -> list[dict]:
    """Get list of blocked issues with number and title for flag output.

    Returns list of dicts with 'number' and 'title' keys.
    Added per #1354 - make blocked_issues flag actionable.

    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns list of dicts with 'number' (int) and 'title' (str) keys
    ENSURES: On API failure, returns empty list
    """
    issues: list[dict] = []

    # Get repo info for API call
    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return issues

    owner, repo = repo_info

    # Use REST API to fetch blocked issues with title
    result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=open&labels=blocked&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {number, title}]",
        ],
        timeout=60,
    )

    if not result.ok or not result.stdout.strip():
        return issues

    try:
        stdout = result.stdout.strip()
        if "\n" in stdout:
            for line in stdout.split("\n"):
                line = line.strip()
                if line:
                    issues.extend(json.loads(line))
        else:
            issues = json.loads(stdout)
    except json.JSONDecodeError:
        pass

    return issues


def get_stale_blockers() -> list[dict]:
    """Get blocked issues whose blockers are now closed (#1497).

    Parses 'Blocked: #N' patterns from blocked issue bodies,
    checks if referenced issues are closed via GitHub API.

    Returns:
        List of dicts with:
        - 'number': The blocked issue number
        - 'title': The blocked issue title
        - 'stale_blockers': List of issue numbers that are now closed

    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns list of dicts with 'number', 'title', 'stale_blockers' keys
    ENSURES: On API failure, returns empty list
    """
    results: list[dict] = []

    # Get repo info for API call
    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return results

    owner, repo = repo_info

    # Step 1: Fetch blocked issues with body (need to parse Blocked: #N)
    result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=open&labels=blocked&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {number, title, body}]",
        ],
        timeout=60,
    )

    if not result.ok or not result.stdout.strip():
        return results

    try:
        # Parse blocked issues
        all_issues: list[dict] = []
        stdout = result.stdout.strip()

        if "\n" in stdout:
            for line in stdout.split("\n"):
                line = line.strip()
                if line:
                    all_issues.extend(json.loads(line))
        else:
            all_issues = json.loads(stdout)

        # Step 2: Extract Blocked: #N patterns from each issue body
        # Uses module-level BLOCKER_PATTERN and ISSUE_REF_PATTERN (#1698)

        issues_with_blockers: list[tuple[dict, list[int]]] = []

        for issue in all_issues:
            body = issue.get("body", "") or ""
            match = BLOCKER_PATTERN.search(body)
            if match:
                # Extract all #N references (local repo only for now)
                refs_str = match.group(1)
                blocker_nums = [
                    int(m.group(1)) for m in ISSUE_REF_PATTERN.finditer(refs_str)
                ]
                if blocker_nums:
                    issues_with_blockers.append((issue, blocker_nums))

        if not issues_with_blockers:
            return results

        # Step 3: Batch check which referenced issues are closed
        # Collect all unique blocker issue numbers
        all_blocker_nums = set()
        for _, blockers in issues_with_blockers:
            all_blocker_nums.update(blockers)

        # Fetch state of all blocker issues via single batched GraphQL query
        # (#1697: Fixes N+1 API pattern - was O(n) REST calls, now O(1))
        closed_issues: set[int] = set()

        if all_blocker_nums:
            # Build batch query: one alias per issue number
            queries: list[tuple[str, str]] = []
            for num in all_blocker_nums:
                alias = f"issue{num}"
                frag = f'repository(owner: "{owner}", name: "{repo}") {{ '
                frag += f"issue(number: {num}) {{ state }} }}"
                queries.append((alias, frag))

            gql_result = graphql_batch(queries, timeout=60)
            if gql_result.ok and gql_result.data:
                for num in all_blocker_nums:
                    alias = f"issue{num}"
                    repo_data = gql_result.data.get(alias, {})
                    issue_data = repo_data.get("issue") if repo_data else None
                    if issue_data and issue_data.get("state") == "CLOSED":
                        closed_issues.add(num)

        # Step 4: Build results - issues with closed blockers
        for issue, blockers in issues_with_blockers:
            stale = [b for b in blockers if b in closed_issues]
            if stale:
                results.append(
                    {
                        "number": issue.get("number", 0),
                        "title": issue.get("title", ""),
                        "stale_blockers": stale,
                    }
                )

    except json.JSONDecodeError:
        pass

    return results


def get_long_blocked_issues(days_threshold: int = 7) -> list[dict]:
    """Get blocked issues with no progress for more than threshold days (#1497).

    Checks each blocked issue's updated_at timestamp. Issues not updated
    in >days_threshold days are considered long-blocked and need escalation.

    Args:
        days_threshold: Number of days without update to trigger (default 7).

    Returns:
        List of dicts with:
        - 'number': Issue number
        - 'title': Issue title
        - 'days_blocked': Days since last update

    REQUIRES: days_threshold >= 0
    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns list of dicts with 'number', 'title', 'days_blocked' keys
    ENSURES: On API failure, returns empty list
    """
    results: list[dict] = []

    # Get repo info for API call
    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return results

    owner, repo = repo_info

    # Fetch blocked issues with updated_at timestamp
    result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=open&labels=blocked&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {number, title, updated_at}]",
        ],
        timeout=60,
    )

    if not result.ok or not result.stdout.strip():
        return results

    try:
        all_issues: list[dict] = []
        stdout = result.stdout.strip()

        if "\n" in stdout:
            for line in stdout.split("\n"):
                line = line.strip()
                if line:
                    all_issues.extend(json.loads(line))
        else:
            all_issues = json.loads(stdout)

        now = datetime.now(UTC)
        threshold_date = now - timedelta(days=days_threshold)

        for issue in all_issues:
            updated_str = issue.get("updated_at", "")
            if not updated_str:
                continue

            try:
                # Parse ISO timestamp (GitHub format: 2026-01-24T10:30:00Z)
                # Note: .replace() needed for Python <3.11 compatibility (FURB162)
                updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))  # noqa: FURB162
                if updated_at < threshold_date:
                    days_blocked = (now - updated_at).days
                    results.append(
                        {
                            "number": issue.get("number", 0),
                            "title": issue.get("title", ""),
                            "days_blocked": days_blocked,
                        }
                    )
            except (ValueError, TypeError):
                # Skip issues with unparsable dates
                pass

    except json.JSONDecodeError:
        pass

    return results


def _unpack_issue_counts(
    issues: object,
) -> tuple[dict[str, int], str | None]:
    """Extract issue counts and error from various input formats.

    Handles Result objects, dicts with value/error keys, and plain dicts.
    Returns tuple of (counts_dict, error_string_or_none).

    Skipped results (e.g., --skip-gh-api) return None for error since they are
    intentional and should not trigger gh_error flag (#1902).
    """
    if isinstance(issues, Result):
        # Skipped results are intentional, not errors
        error = None if issues.skipped else issues.error
        return issues.value or {}, error
    if isinstance(issues, dict) and "value" in issues:
        counts = issues.get("value") or {}
        error = issues.get("error")
        ok = issues.get("ok")
        if ok is False and not error:
            error = "unknown error"
        return counts, error
    if isinstance(issues, dict):
        return issues, issues.get("_error")
    return {}, None


def _count_issues_within_days(gh_json_output: str, date_field: str, days: int) -> int:
    """Count issues where date_field is within N days of now."""
    try:
        issues = json.loads(gh_json_output)
    except json.JSONDecodeError:
        return 0

    # Use UTC time since GitHub timestamps are in UTC
    now = datetime.now(UTC)
    count = 0
    for issue in issues:
        date_str = issue.get(date_field, "")
        try:
            dt = datetime.fromisoformat(date_str)
            if (now - dt).days <= days:
                count += 1
        except (ValueError, TypeError):
            pass
    return count


def get_issue_velocity() -> dict:
    """Get issue velocity metrics (improvement over time).

    Uses REST API instead of gh issue list (GraphQL) to avoid rate limit exhaustion.
    See #1029.

    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns dict with 'created_7d', 'closed_7d' counts (non-negative integers)
    ENSURES: On API failure, returns empty dict (never raises)
    """
    velocity: dict[str, int] = {}

    # Get repo info for API call
    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return velocity

    owner, repo = repo_info

    # Use REST API with 'since' parameter to limit results
    # 'since' filters by updated_at, so we fetch recently updated issues
    # and count created/closed within 7 days client-side
    since_date = (datetime.now(UTC) - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch all issues updated in last 14 days (captures issues created or closed in 7)
    # Filter out PRs (have pull_request field) and extract dates
    result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=all&since={since_date}&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {created_at, closed_at}]",
        ],
        timeout=60,
    )

    if not result.ok or not result.stdout.strip():
        return velocity

    try:
        # gh api --paginate with jq outputs multiple JSON arrays, one per line
        all_issues: list[dict] = []
        stdout = result.stdout.strip()

        if "\n" in stdout:
            # Multiple arrays from pagination - parse each line and flatten
            for line in stdout.split("\n"):
                line = line.strip()
                if line:
                    all_issues.extend(json.loads(line))
        else:
            # Single array (small repo or single page)
            all_issues = json.loads(stdout)

        # Count issues created/closed in last 7 days
        # Use UTC time since GitHub timestamps are in UTC
        now = datetime.now(UTC)
        opened_7d = 0
        closed_7d = 0

        for issue in all_issues:
            # Check created_at
            created_str = issue.get("created_at", "")
            if created_str:
                try:
                    created_dt = datetime.fromisoformat(created_str)
                    if (now - created_dt).days <= 7:
                        opened_7d += 1
                except (ValueError, TypeError):
                    pass

            # Check closed_at
            closed_str = issue.get("closed_at", "")
            if closed_str:
                try:
                    closed_dt = datetime.fromisoformat(closed_str)
                    if (now - closed_dt).days <= 7:
                        closed_7d += 1
                except (ValueError, TypeError):
                    pass

        velocity["opened_7d"] = opened_7d
        velocity["closed_7d"] = closed_7d
        velocity["net_7d"] = closed_7d - opened_7d

    except (json.JSONDecodeError, TypeError):
        pass

    return velocity


def get_issues_reopened() -> dict:
    """Get issue reopen metrics for closure quality tracking.

    Uses REST API to fetch issue events and count reopens within 7 days.
    Part of #1937: Add issues_reopened metric to pulse.py telemetry.

    REQUIRES: Git repo with GitHub remote configured
    ENSURES: Returns dict with 'reopened_7d', 'reopen_rate', 'closed_7d' counts
    ENSURES: On API failure, returns empty dict (never raises)
    """
    result_dict: dict[str, int | float] = {}

    repo_info = _get_repo_owner_and_name()
    if not repo_info:
        return result_dict

    owner, repo = repo_info

    # Get issues closed in last 7 days first (for reopen rate calculation)
    since_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch repository events filtered to 'reopened' type
    # Events API returns events for all issues; we filter by date client-side
    result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues/events?per_page=100",
            "-q",
            '[.[] | select(.event == "reopened") | {created_at, issue: .issue.number}]',
        ],
        timeout=60,
    )

    reopened_7d = 0
    if result.ok and result.stdout.strip():
        try:
            # gh api --paginate with jq outputs multiple JSON arrays, one per line
            all_events: list[dict] = []
            stdout = result.stdout.strip()

            if "\n" in stdout:
                for line in stdout.split("\n"):
                    line = line.strip()
                    if line:
                        all_events.extend(json.loads(line))
            else:
                all_events = json.loads(stdout)

            now = datetime.now(UTC)
            for event in all_events:
                created_str = event.get("created_at", "")
                if created_str:
                    try:
                        created_dt = datetime.fromisoformat(created_str)
                        if (now - created_dt).days <= 7:
                            reopened_7d += 1
                    except (ValueError, TypeError):
                        pass

        except (json.JSONDecodeError, TypeError):
            pass

    # Get closed count for rate calculation (reuse velocity logic)
    closed_result = run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{owner}/{repo}/issues?state=closed&since={since_date}&per_page=100",
            "-q",
            "[.[] | select(.pull_request == null) | {closed_at}]",
        ],
        timeout=60,
    )

    closed_7d = 0
    if closed_result.ok and closed_result.stdout.strip():
        try:
            all_issues: list[dict] = []
            stdout = closed_result.stdout.strip()

            if "\n" in stdout:
                for line in stdout.split("\n"):
                    line = line.strip()
                    if line:
                        all_issues.extend(json.loads(line))
            else:
                all_issues = json.loads(stdout)

            now = datetime.now(UTC)
            for issue in all_issues:
                closed_str = issue.get("closed_at", "")
                if closed_str:
                    try:
                        closed_dt = datetime.fromisoformat(closed_str)
                        if (now - closed_dt).days <= 7:
                            closed_7d += 1
                    except (ValueError, TypeError):
                        pass

        except (json.JSONDecodeError, TypeError):
            pass

    result_dict["reopened_7d"] = reopened_7d
    result_dict["closed_7d"] = closed_7d

    # Calculate reopen rate as percentage (avoid division by zero)
    if closed_7d > 0:
        result_dict["reopen_rate"] = round((reopened_7d / closed_7d) * 100, 1)
    else:
        result_dict["reopen_rate"] = 0.0

    return result_dict

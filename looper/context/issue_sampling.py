# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/issue_sampling.py - Role-based issue sampling.

Functions for sampling GitHub issues based on role and priority tiers.
Worker sees full priority-ordered list, other roles see domain-specific subsets.
"""

__all__ = [
    "get_sampled_issues",
    "get_issue_by_number",
    "get_issues_by_numbers",
    "get_issues_structured",
]

import json
import os
import random
import re
import sys

from looper.config import load_project_config, load_timeout_config
from looper.context.helpers import (
    Issue,
    format_issue,
    get_p_level,
    has_in_progress_label,
    has_label,
    is_pending_issue,
    issue_number,
)
from looper.context.issue_cache import IterationIssueCache, is_feature_freeze
from looper.log import debug_swallow
from looper.result import Result
from looper.subprocess_utils import (
    get_github_repo,
    is_full_local_mode,
    is_local_mode,
    run_gh_command,
)

# Import batch_issue_view at module level for testability
# The function falls back to individual fetches if import fails
try:
    sys.path.insert(0, "ai_template_scripts")
    from gh_rate_limit import batch_issue_view  # type: ignore[import-not-found]
except ImportError:
    batch_issue_view = None  # type: ignore[misc, assignment]


_ISSUE_SAMPLING_DEFAULTS: dict[str, int] = {
    "do_audit": 5,
    "in_progress": 5,
    "urgent": 5,
    "P1": 3,
    "P2": 2,
    "P3": 1,
    "new": 2,
    "random": 1,
    "oldest": 1,
    "domain": 10,
}


def _load_issue_sampling_config() -> dict[str, int]:
    """Load issue sampling limits from .looper_config.json if present."""
    sampling = _ISSUE_SAMPLING_DEFAULTS.copy()
    project_config = load_project_config()
    raw = project_config.get("issue_sampling")
    if not isinstance(raw, dict):
        return sampling
    for key, value in raw.items():
        if key not in sampling:
            continue
        if isinstance(value, int) and value >= 0:
            sampling[key] = value
        elif isinstance(value, float) and value.is_integer() and value >= 0:
            sampling[key] = int(value)
    return sampling


def _issue_created_at(issue: Issue) -> str:
    """Get issue createdAt as str for type safety in sorting."""
    return str(issue["createdAt"])


def _format_pending_issue(issue: Issue) -> str:
    """Format a pending issue for display.

    Pending issues are shown with [PENDING] prefix and their synthetic ID.

    Contracts:
        REQUIRES: issue is a dict with optional "number", "title", "labels" fields
        ENSURES: returns string in format "[PENDING] {number}: {title} [{labels}]"
        ENSURES: missing number defaults to "?"
        ENSURES: missing title defaults to empty string
        ENSURES: missing labels defaults to "-"
    """
    number = issue.get("number")
    if isinstance(number, int):
        number_str = str(number)
    elif isinstance(number, str) and number.strip():
        number_str = number.strip()
    else:
        number_str = "?"

    title = issue.get("title")
    title_str = title if isinstance(title, str) else ""

    raw_labels = issue.get("labels")
    labels: list[str] = []
    if isinstance(raw_labels, list):
        for lbl in raw_labels:
            if isinstance(lbl, dict):
                name = lbl.get("name")
                if isinstance(name, str):
                    labels.append(name)
            elif isinstance(lbl, str):
                labels.append(lbl)
    label_str = ", ".join(labels) if labels else "-"
    return f"[PENDING] {number_str}: {title_str} [{label_str}]"


def get_sampled_issues(role: str = "worker") -> Result[str]:
    """Get role-filtered sampled issues.

    Contracts:
        REQUIRES: role is a string (unknown roles treated as 'worker')
        ENSURES: P0 issues ALWAYS appear first in output (if any exist)
        ENSURES: No issue appears more than once in output
        ENSURES: For worker, excludes tracking/deferred issues unless they are P0
        ENSURES: For non-worker, only shows P0 + domain-specific issues
        ENSURES: Never raises - catches all exceptions
        INVARIANT: Priority order: P0 > do-audit > in-progress > urgent > P1-3

    Worker sees all issues in priority order:
    - P0 (always first - system compromised)
    - do-audit (workflow gate)
    - in-progress (current work, includes worker-specific labels)
    - All urgent (sorted by P-level: urgent P1 > urgent P2 > urgent P3)
    - P1, P2, P3 (non-urgent by priority)
    - Newest, random, oldest (discovery)

    Non-Worker roles:
    - Manager: P0 + needs-review (closure workflow)
    - Prover: P0 only (rotation phases ARE the work)
    - Researcher: P0 only (rotation phases ARE the work)
    """
    # Regular local mode (AIT_LOCAL_MODE=1): return placeholder message
    # Full local mode (AIT_LOCAL_MODE=full): continue to process local issues
    if is_local_mode() and not is_full_local_mode():
        return Result.success("(local mode - GitHub API disabled)")

    # Manager needs "needs-review" for closure workflow
    # Prover/Researcher don't need domain filtering - rotation phases ARE their work
    if role == "manager":
        return _get_domain_issues_targeted(["needs-review"])

    # Prover/Researcher: just show P0s (emergencies only)
    # Their rotation phases provide the work, not issues
    if role in ("prover", "researcher"):
        return _get_p0_issues_only()

    # Worker: full priority sampling - use cache for single API call (#1676)
    worker_id = os.environ.get("AI_WORKER_ID")
    cache_result = IterationIssueCache.get_all()
    if not cache_result.ok:
        error = cache_result.error or "unknown error"
        return Result.failure(f"{error}. See docs/troubleshooting.md#github-issues")

    try:
        all_issues = list(cache_result.value or [])
        if not all_issues:
            return Result.success("(no open issues)")

        # Separate pending issues from real issues (#1854)
        # Pending issues have synthetic IDs like "pending-4a90ba36" and
        # cannot be processed by normal issue sampling (which expects int IDs)
        pending_issues = [i for i in all_issues if is_pending_issue(i)]
        issues = [i for i in all_issues if not is_pending_issue(i)]

        # Worker: full priority sampling
        # Exclude tracking/deferred issues - not actionable work
        # (except P0 tracking/deferred issues, which must still be visible)
        issues = [
            i
            for i in issues
            if not (
                (has_label(i, "tracking") or has_label(i, "deferred"))
                and not has_label(i, "P0")
            )
        ]

        # Feature freeze: exclude feature-labeled issues (bugs/docs/refactor only)
        # P0 features still visible (emergencies override freeze)
        if is_feature_freeze():
            issues = [
                i
                for i in issues
                if not (has_label(i, "feature") and not has_label(i, "P0"))
            ]

        # Worker 3 specialization: P3-only by default
        # Override with WORKER3_NORMAL_MODE=1 to use normal priority sampling
        if worker_id == "3" and os.environ.get("WORKER3_NORMAL_MODE") != "1":
            if not issues and pending_issues:
                pending_lines = [_format_pending_issue(p) for p in pending_issues]
                return Result.success("\n".join(pending_lines))
            result = _sample_worker3_issues(issues, worker_id)
            # Prepend pending issues if any (#1854)
            if pending_issues and result.ok:
                pending_lines = [_format_pending_issue(p) for p in pending_issues]
                return Result.success(
                    "\n".join(pending_lines) + "\n" + (result.value or "")
                )
            return result

        shown: set[int] = set()
        lines: list[str] = []
        sampling = _load_issue_sampling_config()

        # Show pending issues first (#1854) - these are local-only, not yet synced
        for pending in pending_issues:
            lines.append(_format_pending_issue(pending))

        # Sample by priority tiers (P0 > do-audit > in-progress > urgent > P1-3)
        _sample_priority_issues(issues, lines, shown, worker_id, sampling)

        # Sample for discovery (newest, random, oldest)
        _sample_discovery_issues(issues, lines, shown, sampling)

        if not lines:
            return Result.success("(no open issues)")

        return Result.success("\n".join(lines))

    except Exception as e:
        return Result.failure(f"gh issue list error: {e}")


def _sample_tier(
    issues: list[Issue],
    label: str,
    limit: int,
    shown: set[int],
    prefix: str,
) -> list[str]:
    """Sample up to `limit` issues matching `label`.

    Args:
        issues: List of issue dicts to sample from.
        label: Label to filter by.
        limit: Maximum number to sample (0 = unlimited for P0).
        shown: Set of issue numbers already shown (mutated in place).
        prefix: Prefix string for formatted output (e.g., "[P1]").

    Returns:
        List of formatted issue strings.
    """
    lines: list[str] = []
    for issue in issues:
        if limit > 0 and len(lines) >= limit:
            break
        if has_label(issue, label) and issue_number(issue) not in shown:
            lines.append(f"{prefix} {format_issue(issue)}")
            shown.add(issue_number(issue))
    return lines


def _sample_priority_issues(
    issues: list[Issue],
    lines: list[str],
    shown: set[int],
    worker_id: str | None,
    sampling: dict[str, int] | None = None,
) -> None:
    """Sample issues by priority tier, mutating lines and shown.

    Priority order: P0 > do-audit > in-progress > urgent > P1 > P2 > P3

    Args:
        issues: List of issue dicts to sample from.
        lines: Output list to append formatted issue strings.
        shown: Set of issue numbers already shown (mutated in place).
        worker_id: Current worker id string, if set (used for claim annotations).
        sampling: Optional sampling limits dict.
    """
    sampling = sampling or _ISSUE_SAMPLING_DEFAULTS

    # P0 first (always - system compromised, no limit)
    lines.extend(_sample_tier(issues, "P0", 0, shown, "[P0]"))

    # do-audit second (workflow gate)
    lines.extend(
        _sample_tier(
            issues, "do-audit", sampling.get("do_audit", 5), shown, "[DO-AUDIT]"
        )
    )

    # In-progress third (current work) - needs custom handling for claim suffix
    in_progress_limit = sampling.get("in_progress", 5)
    if in_progress_limit > 0:
        count = 0
        for issue in issues:
            if count >= in_progress_limit:
                break
            if has_in_progress_label(issue) and issue_number(issue) not in shown:
                line = f"[IN-PROGRESS] {format_issue(issue)}"
                if suffix := _claimed_by_suffix(issue, worker_id):
                    line = f"{line} {suffix}"
                lines.append(line)
                shown.add(issue_number(issue))
                count += 1

    # Urgent issues (sorted by P-level within)
    urgent_limit = sampling.get("urgent", 5)
    if urgent_limit > 0:
        urgent = [
            i
            for i in issues
            if has_label(i, "urgent") and issue_number(i) not in shown
        ]
        urgent.sort(key=get_p_level)
    else:
        urgent = []
    for issue in urgent[:urgent_limit]:
        p = get_p_level(issue)
        prefix = f"[URGENT P{p}]" if p < 4 else "[URGENT]"
        lines.append(f"{prefix} {format_issue(issue)}")
        shown.add(issue_number(issue))

    # P1, P2, P3 (non-urgent by priority)
    lines.extend(_sample_tier(issues, "P1", sampling.get("P1", 3), shown, "[P1]"))
    lines.extend(_sample_tier(issues, "P2", sampling.get("P2", 2), shown, "[P2]"))
    lines.extend(_sample_tier(issues, "P3", sampling.get("P3", 1), shown, "[P3]"))


def _claimed_by_suffix(issue: Issue, worker_id: str | None) -> str | None:
    """Return suffix indicating claim by another role, if applicable.

    Per ai_template.md: `in-progress` + `XN` labels = claimed (X=W/P/R/M, N=instance).
    Worker ownership labels are W1, W2, etc. P0-P3 are priority labels (NOT ownership).

    Returns:
        "+YOU" if claimed by current worker
        "+W{N}" if claimed by another worker
        None if not claimed by any worker
    """
    # Handle both GraphQL (dict) and REST (string) label formats
    raw_labels = issue.get("labels", [])
    labels = [lbl["name"] if isinstance(lbl, dict) else lbl for lbl in raw_labels]

    # Look for worker ownership labels: W1, W2, W3, etc.
    # Note: P0-P3 are priority labels, not Prover ownership
    for label in labels:
        match = re.match(r"^W(\d+)$", label)
        if match:
            instance = match.group(1)
            if worker_id and instance == worker_id:
                return "+YOU"  # Current worker
            return f"+W{instance}"  # Other worker
    return None


def _sample_discovery_issues(
    issues: list[Issue],
    lines: list[str],
    shown: set[int],
    sampling: dict[str, int] | None = None,
) -> None:
    """Sample discovery issues (newest, random, oldest), mutating lines and shown.

    Ensures issues that don't fit priority buckets still get visibility.

    Args:
        issues: List of issue dicts to sample from.
        lines: Output list to append formatted issue strings.
        shown: Set of issue numbers already shown (mutated in place).
    """
    sampling = sampling or _ISSUE_SAMPLING_DEFAULTS
    remaining = [i for i in issues if issue_number(i) not in shown]

    # Newest 2 not shown
    new_limit = sampling.get("new", 2)
    if new_limit > 0:
        newest = sorted(remaining, key=_issue_created_at, reverse=True)[:new_limit]
        for issue in newest:
            lines.append(f"[NEW] {format_issue(issue)}")
            shown.add(issue_number(issue))

    remaining = [i for i in remaining if issue_number(i) not in shown]

    # Random 1
    random_limit = sampling.get("random", 1)
    if remaining and random_limit > 0:
        sample_count = min(random_limit, len(remaining))
        random_issues = random.sample(remaining, sample_count)
        for issue in random_issues:
            lines.append(f"[RANDOM] {format_issue(issue)}")
            shown.add(issue_number(issue))
        remaining = [i for i in remaining if issue_number(i) not in shown]

    # Oldest 1
    oldest_limit = sampling.get("oldest", 1)
    if remaining and oldest_limit > 0:
        oldest = sorted(remaining, key=_issue_created_at)[:oldest_limit]
        for issue in oldest:
            lines.append(f"[OLDEST] {format_issue(issue)}")
            shown.add(issue_number(issue))


def _sample_worker3_issues(issues: list[Issue], worker_id: str | None) -> Result[str]:
    """Sample issues for Worker 3 with P3-only default behavior.

    Worker 3 specializes in P3 (maintenance/quality) work by default.
    If no P3 issues exist, falls back to normal priority sampling.

    Contracts:
        REQUIRES: issues is a list of issue dicts (excludes tracking/deferred except P0)
        ENSURES: P0 issues ALWAYS appear (even in P3-only mode)
        ENSURES: If P3 issues exist, only P3 issues shown (plus P0)
        ENSURES: If no P3 issues, falls back to normal priority sampling
        ENSURES: Never raises - returns Result.failure on errors

    Args:
        issues: List of issue dicts to sample from.
        worker_id: Worker ID (should be "3").

    Returns:
        Result with formatted issue list string.
    """
    shown: set[int] = set()
    lines: list[str] = []

    # P0 always shown (system compromised takes priority over specialization)
    for issue in issues:
        if has_label(issue, "P0"):
            lines.append(f"[P0] {format_issue(issue)}")
            shown.add(issue_number(issue))

    # Filter to P3-only
    p3_issues = [
        i for i in issues if has_label(i, "P3") and issue_number(i) not in shown
    ]

    # If no P3 issues, fall back to normal behavior for this iteration
    if not p3_issues:
        # Clear lines (keep P0) and do full sampling
        _sample_priority_issues(issues, lines, shown, worker_id)
        _sample_discovery_issues(issues, lines, shown)
        if not lines:
            return Result.success("(no P3 issues - using normal sampling)")
        return Result.success(
            "(no P3 issues - using normal sampling)\n" + "\n".join(lines)
        )

    # P3-only sampling: do-audit P3 first, then in-progress P3, then rest
    # do-audit P3 (workflow gate)
    count = 0
    for issue in p3_issues:
        if count >= 5:
            break
        if has_label(issue, "do-audit") and issue_number(issue) not in shown:
            lines.append(f"[DO-AUDIT P3] {format_issue(issue)}")
            shown.add(issue_number(issue))
            count += 1

    # in-progress P3
    count = 0
    for issue in p3_issues:
        if count >= 3:
            break
        if has_in_progress_label(issue) and issue_number(issue) not in shown:
            line = f"[IN-PROGRESS P3] {format_issue(issue)}"
            if suffix := _claimed_by_suffix(issue, worker_id):
                line = f"{line} {suffix}"
            lines.append(line)
            shown.add(issue_number(issue))
            count += 1

    # Remaining P3 (up to 10)
    count = 0
    for issue in p3_issues:
        if count >= 10:
            break
        if issue_number(issue) not in shown:
            lines.append(f"[P3] {format_issue(issue)}")
            shown.add(issue_number(issue))
            count += 1

    if not lines:
        return Result.success("(no open issues)")

    return Result.success("\n".join(lines))


def _get_p0_issues_only() -> Result[str]:
    """Fetch P0 issues only for rotation-based roles (Prover, Researcher).

    These roles have rotation phases as their primary work.
    Issues are only shown for emergencies (P0).

    Contracts:
        ENSURES: Only P0 issues are shown (including pending P0s)
        ENSURES: Pending P0 issues are shown first with [PENDING] marker
        ENSURES: Returns message about rotation being primary work if no P0s
        ENSURES: Never raises - catches all exceptions
    """
    # Regular local mode: return placeholder. Full local mode: continue processing.
    if is_local_mode() and not is_full_local_mode():
        return Result.success("(local mode - GitHub API disabled)")

    cache_result = IterationIssueCache.get_all()
    if not cache_result.ok:
        error = cache_result.error or "unknown error"
        return Result.failure(error)

    try:
        all_issues_raw = list(cache_result.value or [])
        pending_issues = [i for i in all_issues_raw if is_pending_issue(i)]
        all_issues = [i for i in all_issues_raw if not is_pending_issue(i)]
        pending_p0 = [i for i in pending_issues if has_label(i, "P0")]
        p0_issues = [i for i in all_issues if has_label(i, "P0")]

        if not pending_p0 and not p0_issues:
            return Result.success("(P0 only - rotation phases are your primary work)")

        lines = [_format_pending_issue(issue) for issue in pending_p0]
        lines.extend([f"[P0] {format_issue(issue)}" for issue in p0_issues])
        return Result.success("\n".join(lines))
    except Exception as e:
        return Result.failure(f"Failed to fetch P0 issues: {e}")


def _get_domain_issues_targeted(domain_labels: list[str]) -> Result[str]:
    """Fetch P0 + domain-specific issues using IterationIssueCache.

    Contracts:
        REQUIRES: domain_labels is non-empty list of label strings
        ENSURES: P0 issues appear first
        ENSURES: After P0, only issues matching domain_labels are shown
        ENSURES: Within domain, urgent issues before non-urgent
        ENSURES: No issue appears more than once in output
        ENSURES: Never raises - catches all exceptions

    Uses IterationIssueCache for client-side filtering (#1676):
    - Single API call fetches all issues
    - P0 filtered client-side (all roles must see)
    - Domain labels filtered client-side (manager: needs-review, etc.)
    """
    # Regular local mode: return placeholder. Full local mode: continue processing.
    if is_local_mode() and not is_full_local_mode():
        return Result.success("(local mode - GitHub API disabled)")

    lines: list[str] = []
    shown: set[int] = set()
    sampling = _load_issue_sampling_config()
    domain_limit = sampling.get("domain", 10)

    # Get all issues from cache (single API call shared across iteration)
    cache_result = IterationIssueCache.get_all()
    if not cache_result.ok:
        error = cache_result.error or "unknown error"
        return Result.failure(error)

    all_issues_raw = cache_result.value or []

    # Separate pending issues from real issues (#1854)
    pending_issues = [i for i in all_issues_raw if is_pending_issue(i)]
    all_issues = [i for i in all_issues_raw if not is_pending_issue(i)]

    # Show pending issues first (#1854)
    for pending in pending_issues:
        lines.append(_format_pending_issue(pending))

    # Filter P0 issues (all roles must see these)
    p0_issues = [i for i in all_issues if has_label(i, "P0")]
    for issue in p0_issues:
        lines.append(f"[P0] {format_issue(issue)}")
        shown.add(issue_number(issue))

    # Filter domain issues (OR logic - any of the domain labels)
    domain_issues: list[dict[str, object]] = []
    if domain_limit > 0:
        for issue in all_issues:
            if issue_number(issue) in shown:
                continue  # Skip P0 already shown
            if any(has_label(issue, lbl) for lbl in domain_labels):
                domain_issues.append(issue)
                shown.add(issue_number(issue))

    # Sort: urgent first, then by P-level
    def domain_sort_key(issue: dict[str, object]) -> tuple[int, int]:
        urgent = 0 if has_label(issue, "urgent") else 1
        return (urgent, get_p_level(issue))

    domain_issues.sort(key=domain_sort_key)

    for issue in domain_issues[:domain_limit]:
        p = get_p_level(issue)
        is_urgent = has_label(issue, "urgent")
        if is_urgent:
            prefix = f"[URGENT P{p}]" if p < 4 else "[URGENT]"
        else:
            prefix = f"[P{p}]" if p < 4 else "[DOMAIN]"
        lines.append(f"{prefix} {format_issue(issue)}")

    if not lines:
        return Result.success(
            "(no P0 or needs-review issues - audit phases are your primary work)"
        )

    return Result.success("\n".join(lines))


def get_issue_by_number(issue_num: int) -> str:
    """Get a single issue by number for focused audit context.

    Args:
        issue_num: The issue number to fetch.

    Returns:
        Formatted issue string, or empty string if not found.
    """
    # Use batch function for consistency
    results = get_issues_by_numbers([issue_num])
    return results.get(issue_num, "")


def get_issues_by_numbers(issue_nums: list[int]) -> dict[int, str]:
    """Get multiple issues by number in a single GraphQL query.

    Batches issue lookups using GraphQL aliases for efficiency (#1178).

    Args:
        issue_nums: List of issue numbers to fetch.

    Returns:
        Dict mapping issue number to formatted issue string.
        Missing/failed issues have empty string values.
    """
    if not issue_nums:
        return {}

    # Full local mode: return empty results (local issues use L-prefix, not numbers)
    # GitHub issue numbers cannot be fetched in full local mode
    if is_full_local_mode():
        return dict.fromkeys(issue_nums, "")

    # Regular local mode: also skip API calls
    if is_local_mode():
        return dict.fromkeys(issue_nums, "")

    project_config = load_project_config()
    gh_view_timeout = load_timeout_config(project_config).get("gh_view", 10)

    # Try batch GraphQL fetch (uses module-level import for testability)
    if batch_issue_view is not None:
        batch_results = batch_issue_view(
            issue_nums,
            ["number", "title", "labels", "state"],
            timeout=gh_view_timeout,
        )

        results: dict[int, str] = {}
        for num in issue_nums:
            issue = batch_results.get(num)
            if issue:
                # labels are already flattened by batch_issue_view
                raw_labels = issue.get("labels", [])
                labels = (
                    [
                        lbl["name"] if isinstance(lbl, dict) else lbl
                        for lbl in raw_labels
                    ]
                    if isinstance(raw_labels, list)
                    else []
                )
                label_str = ", ".join(labels) or "-"
                state = issue.get("state", "UNKNOWN")
                results[num] = (
                    f"[WORKING] #{issue['number']}: {issue['title']} [{label_str}] ({state})"
                )
            else:
                results[num] = ""

        return results

    # Fallback: individual fetches (old behavior, if batch_issue_view unavailable)
    results = {}
    repo = get_github_repo()  # Avoid cwd dependency (#2317)
    for num in issue_nums:
        result = run_gh_command(
            [
                "issue",
                "view",
                str(num),
                "--json",
                "number,title,labels,state",
            ],
            timeout=gh_view_timeout,
            repo=repo,
        )
        if not result.ok or not result.value:
            results[num] = ""
            continue

        try:
            issue = json.loads(result.value)
            raw_labels = issue.get("labels", [])
            labels = [
                lbl["name"] if isinstance(lbl, dict) else lbl for lbl in raw_labels
            ]
            label_str = ", ".join(labels) or "-"
            state = issue.get("state", "UNKNOWN")
            results[num] = (
                f"[WORKING] #{issue['number']}: {issue['title']} [{label_str}] ({state})"
            )
        except Exception as e:
            debug_swallow(f"get_issue_view:{num}", e)
            results[num] = ""

    return results


def get_issues_structured(role: str = "worker") -> Result[list[dict[str, object]]]:
    """Get role-filtered issues as structured data (list of dicts).

    This is the structured alternative to get_sampled_issues() which returns
    formatted strings. Use this when you need to filter, transform, or
    display issues in a custom format without writing inline Python.

    ANTI-PATTERN PREVENTION (#1987):
        Instead of piping gh output through python3 -c (shell escaping breaks
        operators like !=), use this function and process in Python directly.

    Each issue dict contains:
        - number: int | str | None - issue number (str for pending issues, None if missing)
        - title: str - issue title
        - labels: list[str] - label names
        - createdAt: str - ISO timestamp
        - body: str - issue body (may be truncated)
        - p_level: int - priority level (0-3, 4 if none)
        - is_urgent: bool - has 'urgent' label
        - is_in_progress: bool - has 'in-progress' label
        - is_pending: bool - is a pending (unsynced) issue

    Contracts:
        REQUIRES: role is a string (unknown roles treated as 'worker')
        ENSURES: Returns list of issue dicts with normalized fields
        ENSURES: P0 issues always included (even for non-worker roles)
        ENSURES: Never raises - catches all exceptions
        ENSURES: Same role-based filtering as get_sampled_issues() (but no priority ordering)

    Args:
        role: Role name (worker, manager, prover, researcher).

    Returns:
        Result with list of issue dicts on success.
    """
    # Regular local mode: return empty. Full local mode: continue processing.
    if is_local_mode() and not is_full_local_mode():
        return Result.success([])

    # Normalize role to lowercase for consistent matching
    role = role.lower()

    cache_result = IterationIssueCache.get_all()
    if not cache_result.ok:
        error = cache_result.error or "unknown error"
        return Result.failure(error, value=[])

    try:
        all_issues = list(cache_result.value or [])
        if not all_issues:
            return Result.success([])

        # Normalize issues to consistent structure
        def normalize_issue(issue: Issue) -> dict[str, object]:
            """Convert raw issue dict to normalized structure."""
            raw_labels = issue.get("labels", [])
            labels: list[str] = []
            for lbl in raw_labels:
                if isinstance(lbl, dict):
                    name = lbl.get("name")
                    if isinstance(name, str):
                        labels.append(name)
                elif isinstance(lbl, str):
                    labels.append(lbl)

            return {
                "number": issue.get("number"),
                "title": issue.get("title", ""),
                "labels": labels,
                "createdAt": issue.get("createdAt", ""),
                "body": issue.get("body", ""),
                "p_level": get_p_level(issue),
                "is_urgent": has_label(issue, "urgent"),
                "is_in_progress": has_in_progress_label(issue),
                "is_pending": is_pending_issue(issue),
            }

        # Apply role-based filtering (same logic as get_sampled_issues)
        if role in ("prover", "researcher"):
            # Only P0 issues for rotation-based roles
            p0_issues = [i for i in all_issues if has_label(i, "P0")]
            return Result.success([normalize_issue(i) for i in p0_issues])

        if role == "manager":
            # P0 + needs-review
            filtered = [
                i
                for i in all_issues
                if has_label(i, "P0") or has_label(i, "needs-review")
            ]
            return Result.success([normalize_issue(i) for i in filtered])

        # Worker: exclude tracking/deferred (except P0)
        filtered = [
            i
            for i in all_issues
            if not (
                (has_label(i, "tracking") or has_label(i, "deferred"))
                and not has_label(i, "P0")
            )
        ]

        # Feature freeze: exclude feature-labeled issues (except P0)
        if is_feature_freeze():
            filtered = [
                i
                for i in filtered
                if not (has_label(i, "feature") and not has_label(i, "P0"))
            ]

        return Result.success([normalize_issue(i) for i in filtered])

    except Exception as e:
        return Result.failure(f"get_issues_structured error: {e}", value=[])

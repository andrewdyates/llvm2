# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context - Session context gathering package.

Gathers context injected into AI prompts at session start:
- Git log (recent commits)
- GitHub issues (sampled by priority)
- Last directive (## Next from prior same-role commit)
- Other role feedback (recent commits from M > R/P > W)
- @ROLE mentions
- System status (memory/disk)
- Audit data (for manager)

Public API:
    run_session_start_commands - Main entry point for context gathering
    SessionContext - TypedDict schema for context injection
    validate_session_context - Validate context structure
    get_sampled_issues - Role-filtered issue sampling (returns formatted strings)
    get_issues_structured - Role-filtered issues as dicts (for programmatic use)
    get_issue_by_number - Single issue lookup
    get_do_audit_issues - Issues ready for audit
    transition_audit_to_review - Transition issue labels
    run_system_health_check - Run health check script

Error Handling:
    Functions in this package use two error handling patterns:

    1. Result[T] - For operations where callers need to distinguish success from error.
       Example: get_sampled_issues() -> Result[str]

    2. Silent fallback - For convenience helpers where errors map to safe defaults.
       These functions document their fallback behavior in docstrings:
       - get_issue_by_number() -> str ("" on not found/error)
       - get_uncommitted_changes_size() -> int (0 on error)
       - is_feature_freeze() -> bool (False on error)

    See designs/2026-02-01-looper-api-consistency.md for design rationale.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from typing import NotRequired


class SessionContext(TypedDict, total=False):
    """Schema for context injected into AI prompts at session start.

    Required fields are populated for all roles. Optional fields are
    role-specific or only present when tracking is enabled.

    Fields:
        git_log: Recent commits from `git log --oneline -10`
        gh_issues: Sampled issues formatted for prompt injection
        active_issue: Worker's own in-progress issue for prominent display
        last_directive: ## Next section from last same-role commit
        other_feedback: Recent commits from other roles
        role_mentions: @ROLE mentions directed at current role
        handoff_context: Structured ## Handoff blocks from recent commits
        system_status: Memory and disk usage stats
        audit_data: Pre-computed audit info (manager only, empty string for others)
        _context_sizes: JSON dict of byte counts per field (if track_sizes=True)
    """

    # Required fields (present for all roles)
    git_log: str
    gh_issues: str
    active_issue: str  # Worker's own in-progress issue (empty for non-worker)
    last_directive: str
    other_feedback: str
    role_mentions: str
    handoff_context: str
    system_status: str
    audit_data: str  # Present but empty for non-manager

    # Optional fields (only when requested)
    if TYPE_CHECKING:
        _context_sizes: NotRequired[str]


# Required fields that must be present in all SessionContext results
_REQUIRED_CONTEXT_FIELDS: frozenset[str] = frozenset(
    [
        "git_log",
        "gh_issues",
        "active_issue",
        "last_directive",
        "other_feedback",
        "role_mentions",
        "handoff_context",
        "system_status",
        "audit_data",
    ]
)


def validate_session_context(ctx: dict[str, str]) -> bool:
    """Validate that a context dict has all required SessionContext fields.

    REQUIRES: ctx is a dict (may have any keys/values)
    ENSURES: returns True if valid; raises ValueError if missing/non-string fields

    Args:
        ctx: Context dictionary to validate

    Returns:
        True if all required fields are present and are strings

    Raises:
        ValueError: If required fields are missing or have wrong type
    """
    missing = _REQUIRED_CONTEXT_FIELDS - ctx.keys()
    if missing:
        raise ValueError(f"Missing required SessionContext fields: {sorted(missing)}")

    non_string = [k for k in _REQUIRED_CONTEXT_FIELDS if not isinstance(ctx.get(k), str)]
    if non_string:
        raise ValueError(f"SessionContext fields must be strings: {sorted(non_string)}")

    return True

from looper.config import load_timeout_config
from looper.context.audit_context import get_audit_data
from looper.context.git_context import (
    get_handoff_context,
    get_last_directive,
    get_other_role_feedback,
    get_role_mentions,
    has_role_mention,
)
from looper.context.helpers import (
    format_issue,
    get_labels,
    get_p_level,
    has_label,
)
from looper.context.issue_context import (
    IterationIssueCache,
    get_active_issue,
    get_do_audit_issues,
    get_issue_by_number,
    get_issues_by_numbers,
    get_issues_structured,
    get_sampled_issues,
    is_feature_freeze,
    transition_audit_to_review,
)
from looper.context.system_context import (
    get_system_status,
    run_system_health_check,
    truncate_output,
)
from looper.context.uncommitted_warning import (
    get_uncommitted_changes_size,
    get_uncommitted_changes_warning,
)
from looper.log import debug_swallow
from looper.result import Result, format_result
from looper.subprocess_utils import is_local_mode, run_git_command

try:
    from ai_template_scripts.gh_issues_mirror import refresh_if_stale
except ModuleNotFoundError:
    refresh_if_stale = None  # type: ignore[assignment]


def _extract_next_section(commit_message: str) -> str:
    """Extract the body lines under a ## Next heading from a commit message."""
    if "## Next" not in commit_message:
        return ""

    lines = commit_message.splitlines()
    in_next = False
    next_lines: list[str] = []
    for line in lines:
        if line.startswith("## Next"):
            in_next = True
            continue
        if not in_next:
            continue
        if line.startswith("## ") or line.startswith("---"):
            break
        next_lines.append(line)
    return "\n".join(next_lines).strip()


def _format_rich_git_log(
    raw_log: str,
    max_next_lines: int = 3,
) -> tuple[str, set[str]]:
    """Format git log output with per-commit subject and optional ## Next snippet.

    Returns:
        Tuple of (formatted_log, shown_hashes) where shown_hashes contains the
        7-char abbreviated hashes of all commits included in the formatted output.
        Callers can pass shown_hashes to get_other_role_feedback() to avoid
        displaying the same commits in both sections. See #2688.
    """
    text = raw_log.strip()
    if not text:
        return "(no commits)", set()

    # Backward compatibility for tests/mocks that still provide --oneline style logs.
    if "---COMMIT_SEP---" not in text:
        return text, set()

    blocks: list[str] = []
    shown_hashes: set[str] = set()
    for chunk in text.split("---COMMIT_SEP---"):
        commit_block = chunk.strip()
        if not commit_block:
            continue

        lines = commit_block.splitlines()
        if not lines:
            continue

        header = lines[0].strip()
        if not header:
            continue
        body = "\n".join(lines[1:]).strip()

        header_parts = header.split(" ", 1)
        short_hash = header_parts[0][:7]
        subject = header_parts[1].strip() if len(header_parts) > 1 else ""

        shown_hashes.add(short_hash)

        formatted_lines: list[str] = []
        if subject:
            formatted_lines.append(f"{short_hash} {subject}")
        else:
            formatted_lines.append(short_hash)

        next_section = _extract_next_section(body)
        if next_section:
            snippet_lines = [
                line.strip()
                for line in next_section.splitlines()
                if line.strip()
            ][:max_next_lines]
            if snippet_lines:
                formatted_lines.append("  ## Next")
                formatted_lines.extend(f"  {line}" for line in snippet_lines)

        blocks.append("\n".join(formatted_lines))

    formatted = "\n\n".join(blocks) if blocks else "(no commits)"
    return formatted, shown_hashes


def _get_recent_same_role_handoffs(
    role: str,
    worker_id: int | None,
    timeout: int,
    limit: int = 3,
) -> str:
    """Return same-role commit subjects with ## Next snippets for prompt context."""
    normalized = role.strip().lower()
    if not normalized or limit <= 0:
        return ""

    role_prefix = normalized[0].upper()
    if worker_id is not None:
        pattern = rf"^\[[^\]]*-?{role_prefix}{worker_id}\]"
    else:
        pattern = rf"^\[[^\]]*-?{role_prefix}[0-9]*\]"
    scan_limit = max(limit * 5, limit)

    hashes_result = run_git_command(
        [
            "log",
            "--extended-regexp",
            f"--grep={pattern}",
            "--format=%H",
            "-n",
            str(scan_limit),
        ],
        timeout=timeout,
    )
    if not hashes_result.ok:
        return ""

    hashes = [line.strip() for line in (hashes_result.value or "").splitlines() if line.strip()]
    if not hashes:
        return ""

    handoff_blocks: list[str] = []
    for commit_hash in hashes:
        if len(handoff_blocks) >= limit:
            break

        msg_result = run_git_command(
            ["log", "-1", "--format=%s%n%b", commit_hash],
            timeout=timeout,
        )
        if not msg_result.ok:
            continue

        message = (msg_result.value or "").strip()
        if not message:
            continue

        subject = message.splitlines()[0].strip()
        next_section = _extract_next_section(message)
        if not next_section:
            continue

        # Avoid excessively long handoffs while still providing real context.
        next_section = next_section[:1200].rstrip()
        handoff_blocks.append(
            f"- {commit_hash[:7]} {subject}\n{next_section}"
        )

    if not handoff_blocks:
        return ""
    return "## Same-role commit handoffs\n" + "\n\n".join(handoff_blocks)


def run_session_start_commands(
    role: str,
    gh_issues_result: Result[str] | None = None,
    track_sizes: bool = False,
    worker_id: int | None = None,
) -> dict[str, str]:
    """Execute session start commands and capture output.

    Args:
        role: Current role (worker, manager, researcher, prover)
        gh_issues_result: Pre-fetched issues result (optional)
        track_sizes: If True, add _context_sizes dict with byte counts per field
        worker_id: Worker instance ID (1-5). When set, filters git context
            for this specific worker (e.g., W1 only sees W1's directives).

    Returns:
        Dict with injection content:
        - git_log: Recent commits
        - gh_issues: Sampled issues (important, newest, random, oldest)
        - last_directive: ## Next section from last same-role commit
        - other_feedback: Recent commits from other roles
        - role_mentions: @ROLE mentions directed at this role
        - system_status: Memory/disk usage
        - audit_data: Pre-computed audit info (manager only)
        - _context_sizes: (if track_sizes=True) dict of field sizes in bytes
    """
    results: dict[str, str] = {}
    sizes: dict[str, int] = {} if track_sizes else {}

    # Refresh local issue mirror if stale (best-effort, non-blocking).
    if refresh_if_stale is not None and not is_local_mode():
        try:
            refresh_if_stale()
        except Exception as e:
            debug_swallow("refresh_if_stale", e)

    # git log + same-role commit handoff context
    git_timeout = load_timeout_config().get("git_default", 5)
    result = run_git_command(
        ["log", "--format=%H %s%n%b%n---COMMIT_SEP---", "-10"],
        timeout=git_timeout,
    )
    git_log_hashes: set[str] = set()
    if result.ok:
        git_log, git_log_hashes = _format_rich_git_log(result.value or "")
        handoffs = _get_recent_same_role_handoffs(
            role=role,
            worker_id=worker_id,
            timeout=git_timeout,
        )
        results["git_log"] = f"{git_log}\n\n{handoffs}" if handoffs else git_log
    else:
        error = result.error or "unknown error"
        results["git_log"] = f"(git log failed: {error})"

    # Sampled issues (role-filtered: Worker sees all, others see P0 + domain)
    # Thread worker_id so issue sampling uses explicit param instead of env var (#2591)
    if gh_issues_result is None:
        gh_issues_result = get_sampled_issues(role, worker_id=worker_id)
    results["gh_issues"] = format_result(gh_issues_result)

    # Worker's own in-progress issue for prominent display (#2571)
    results["active_issue"] = format_result(get_active_issue(worker_id=worker_id))

    # Last directive from same role's ## Next section
    # Thread worker_id so W1 only gets W1's directive, not W2/W3's (#2563)
    results["last_directive"] = format_result(get_last_directive(role, worker_id=worker_id))

    # Feedback from other roles (worker_id enables cross-worker visibility #2566)
    # Pass git_log hashes to skip commits already shown in git_log (#2688)
    results["other_feedback"] = format_result(
        get_other_role_feedback(role, worker_id=worker_id, exclude_hashes=git_log_hashes)
    )

    # @ROLE mentions directed at this role (worker_id enables @W1 syntax)
    results["role_mentions"] = format_result(
        get_role_mentions(role, worker_id=worker_id)
    )

    # Structured handoff context (## Handoff blocks, worker_id for targeting #2568)
    results["handoff_context"] = format_result(
        get_handoff_context(role, worker_id=worker_id)
    )

    # System status (memory/disk)
    results["system_status"] = format_result(get_system_status())

    # Pre-computed audit data for manager
    if role == "manager":
        results["audit_data"] = format_result(get_audit_data())
    else:
        results["audit_data"] = ""

    # Add size tracking if requested
    if track_sizes:
        for key, value in results.items():
            sizes[key] = len(value.encode("utf-8")) if value else 0
        sizes["_total"] = sum(sizes.values())
        # Store as JSON string for consistency with other string values
        results["_context_sizes"] = json.dumps(sizes)

    # Validate context structure before returning
    validate_session_context(results)

    return results


__all__ = [
    # Main entry point
    "run_session_start_commands",
    # Schema and validation
    "SessionContext",
    "validate_session_context",
    # Issue context
    "IterationIssueCache",
    "get_active_issue",
    "get_sampled_issues",
    "get_issues_structured",
    "get_issue_by_number",
    "get_issues_by_numbers",
    "get_do_audit_issues",
    "is_feature_freeze",
    "transition_audit_to_review",
    # Git context
    "get_handoff_context",
    "get_last_directive",
    "get_other_role_feedback",
    "get_role_mentions",
    "has_role_mention",
    # System context
    "get_system_status",
    "get_uncommitted_changes_size",
    "get_uncommitted_changes_warning",
    "run_system_health_check",
    "truncate_output",
    # Audit context
    "get_audit_data",
    # Helpers
    "get_labels",
    "has_label",
    "format_issue",
    "get_p_level",
]

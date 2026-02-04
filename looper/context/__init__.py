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
    get_sampled_issues - Role-filtered issue sampling (returns formatted strings)
    get_issues_structured - Role-filtered issues as dicts (for programmatic use)
    get_issue_by_number - Single issue lookup
    get_do_audit_issues - Issues ready for audit
    transition_audit_to_review - Transition issue labels
    run_system_health_check - Run health check script
"""

import json

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
    check_urgent_handoff,
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


def run_session_start_commands(
    role: str,
    gh_issues_result: Result[str] | None = None,
    track_sizes: bool = False,
) -> dict[str, str]:
    """Execute session start commands and capture output.

    Args:
        role: Current role (worker, manager, researcher, prover)
        gh_issues_result: Pre-fetched issues result (optional)
        track_sizes: If True, add _context_sizes dict with byte counts per field

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

    # git log --oneline -10
    git_timeout = load_timeout_config().get("git_default", 5)
    result = run_git_command(["log", "--oneline", "-10"], timeout=git_timeout)
    if result.ok:
        results["git_log"] = (result.value or "").strip() or "(no commits)"
    else:
        error = result.error or "unknown error"
        results["git_log"] = f"(git log failed: {error})"

    # Sampled issues (role-filtered: Worker sees all, others see P0 + domain)
    if gh_issues_result is None:
        gh_issues_result = get_sampled_issues(role)
    results["gh_issues"] = format_result(gh_issues_result)

    # Last directive from same role's ## Next section
    results["last_directive"] = format_result(get_last_directive(role))

    # Feedback from other roles
    results["other_feedback"] = format_result(get_other_role_feedback(role))

    # @ROLE mentions directed at this role
    results["role_mentions"] = format_result(get_role_mentions(role))

    # Structured handoff context (## Handoff blocks)
    results["handoff_context"] = format_result(get_handoff_context(role))

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

    return results


__all__ = [
    # Main entry point
    "run_session_start_commands",
    # Issue context
    "IterationIssueCache",
    "check_urgent_handoff",
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

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/helpers.py - Shared helper functions for context modules.

Internal helpers for issue formatting and label extraction.
"""

__all__ = [
    "Issue",
    "LabelDict",
    "clean_issue_body",
    "enforce_prompt_budget",
    "get_labels",
    "has_label",
    "has_in_progress_label",
    "format_issue",
    "get_p_level",
    "is_pending_issue",
    "issue_number",
    "truncate_injection",
]

import re
from typing import Any, TypedDict

from looper.log import log_warning


class LabelDict(TypedDict):
    """GraphQL label format from GitHub API."""

    name: str


# Label can be either a dict (GraphQL) or string (REST/processed)
Label = LabelDict | str

# Type alias for issue dicts from GitHub API
Issue = dict[str, Any]


def get_labels(issue: Issue) -> list[str]:
    """Extract label names from issue dict.

    Contracts:
        REQUIRES: issue is a dict (may have optional 'labels' key)
        ENSURES: Returns list of label name strings
        ENSURES: Returns empty list if 'labels' key is missing or empty
        ENSURES: Never raises (handles both GraphQL and REST formats)

    Handles both formats:
    - GraphQL: [{"name": "P1"}, {"name": "bug"}]
    - REST/processed: ["P1", "bug"]
    """
    labels: list[Label] = issue.get("labels", [])
    result: list[str] = []
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name", "")
            if name:
                result.append(name)
        elif label:  # String label
            result.append(label)
    return result


def has_label(issue: Issue, label: str) -> bool:
    """Check if issue has a specific label.

    Contracts:
        REQUIRES: issue is a dict with optional 'labels' key
        REQUIRES: label is a string (label name to search for)
        ENSURES: Returns True if label found in issue's labels
        ENSURES: Returns False if labels missing or label not found
        ENSURES: Never raises

    Iterates directly to avoid creating a temporary list. Part of #820.
    """
    labels: list[Label] = issue.get("labels", [])
    for lbl in labels:
        name = lbl.get("name", "") if isinstance(lbl, dict) else (lbl or "")
        if name == label:
            return True
    return False


def has_in_progress_label(issue: Issue) -> bool:
    """Check if issue has in-progress or any role-specific in-progress label.

    Contracts:
        REQUIRES: issue is a dict with optional 'labels' key
        ENSURES: Returns True if 'in-progress' label or any 'in-progress-*' label found
        ENSURES: Returns False if labels missing or no in-progress labels found
        ENSURES: Never raises

    Handles: in-progress, in-progress-W*, in-progress-P*, in-progress-R*, in-progress-M*
    Iterates directly to avoid creating a temporary list. Part of #820.
    """
    labels: list[Label] = issue.get("labels", [])
    for lbl in labels:
        name = lbl.get("name", "") if isinstance(lbl, dict) else (lbl or "")
        if name == "in-progress" or name.startswith("in-progress-"):
            return True
    return False


def format_issue(issue: Issue) -> str:
    """Format issue for display in session context.

    Contracts:
        REQUIRES: issue is a dict (may have 'number', 'title', 'labels' keys)
        ENSURES: Returns formatted string "#N: title... [labels]"
        ENSURES: Title is truncated to 60 characters
        ENSURES: Labels displayed as comma-separated list, or "-" if none
        ENSURES: Never raises (uses defaults for missing keys)
    """
    labels = ", ".join(get_labels(issue)) or "-"
    number = issue.get("number", 0)
    title = issue.get("title", "(no title)")[:60]
    return f"#{number}: {title} [{labels}]"


def get_p_level(issue: Issue) -> int:
    """Return priority level (0-3), 4 for unlabeled.

    Contracts:
        REQUIRES: issue is a dict with optional 'labels' key
        ENSURES: Returns int in range 0-4
        ENSURES: Returns 0 for P0, 1 for P1, 2 for P2, 3 for P3, 4 for no P-label
        ENSURES: If multiple P-labels exist, returns lowest (highest priority)
        ENSURES: Never raises

    If multiple P-labels exist, returns the lowest (highest priority).
    Single pass through labels for efficiency. Part of #820.
    """
    labels: list[Label] = issue.get("labels", [])
    min_level = 4
    for lbl in labels:
        name = lbl.get("name", "") if isinstance(lbl, dict) else (lbl or "")
        if name in ("P0", "P1", "P2", "P3"):
            level = int(name[1])
            if level < min_level:
                min_level = level
                if level == 0:  # Can't get lower than P0
                    return 0
    return min_level


def is_pending_issue(issue: Issue) -> bool:
    """Check if issue is a pending (unsynced) issue from change_log.

    Pending issues have synthetic IDs like 'pending-4a90ba36'.

    Contracts:
        REQUIRES: issue is a dict-like object (may be empty)
        ENSURES: Returns True for missing/None numbers or non-digit strings
        ENSURES: Returns False for numeric IDs
        ENSURES: Never raises
    """
    number = issue.get("number")
    if number is None:
        return True
    if isinstance(number, int) and not isinstance(number, bool):
        return False
    if isinstance(number, str):
        stripped = number.strip()
        if not stripped:
            return True
        if stripped.startswith("pending-"):
            return True
        # Guard against any non-numeric synthetic IDs slipping into lists.
        if not stripped.isdigit():
            return True
        return False
    return True


def issue_number(issue: Issue) -> int:
    """Get issue number as int for type safety.

    Contracts:
        REQUIRES: issue is not a pending issue (use is_pending_issue() to filter)
        ENSURES: Returns int issue number
        ENSURES: Raises TypeError/ValueError if number is not convertible to int
    """
    return int(issue["number"])  # type: ignore[call-overload]


# Patterns for machine metadata that should not be injected into AI prompts
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_METADATA_FOOTER_RE = re.compile(
    r"^[a-z_][\w-]* \| (?:WORKER|PROVER|RESEARCHER|MANAGER) #\d+ \|.+$",
    re.MULTILINE,
)
_FROM_HEADER_RE = re.compile(r"^\*\*FROM:\*\* .+$", re.MULTILINE)
_GH_RATE_LIMIT_RE = re.compile(r"^gh_rate_limit: .+$", re.MULTILINE)


def clean_issue_body(body: str) -> str:
    """Strip machine metadata from issue body for clean prompt injection.

    Removes HTML comments, metadata footer lines, FROM headers,
    and gh_rate_limit messages that waste prompt tokens.

    Contracts:
        REQUIRES: body is a string
        ENSURES: Returns cleaned string with metadata removed
        ENSURES: Preserves all non-metadata content
        ENSURES: Collapses runs of blank lines to single blank line
        ENSURES: Never raises
    """
    text = _HTML_COMMENT_RE.sub("", body)
    text = _METADATA_FOOTER_RE.sub("", text)
    text = _FROM_HEADER_RE.sub("", text)
    text = _GH_RATE_LIMIT_RE.sub("", text)
    # Collapse runs of blank lines (from removed content) to single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing --- separators left after metadata removal
    text = re.sub(r"\n---\s*$", "", text)
    return text.strip()


# Default total budget for looper prompt injections (chars)
DEFAULT_PROMPT_BUDGET = 15000

# Truncation priority order: lowest priority truncated first.
# Each tuple is (injection_key, minimum_chars_to_preserve).
_TRUNCATION_ORDER = [
    ("audit_data", 3000),
    ("other_feedback", 1500),
    ("role_mentions", 500),
    ("git_log", 2000),
    ("gh_issues", 1000),
    ("handoff_context", 400),
    ("active_issue", 1000),
    ("last_directive", 500),
]


def truncate_injection(content: str, max_chars: int, label: str) -> str:
    """Truncate injection content to max_chars, preserving line boundaries.

    Contracts:
        REQUIRES: content is a string, max_chars >= 0
        ENSURES: len(result) <= max_chars + len(truncation_suffix)
        ENSURES: Never raises
    """
    if len(content) <= max_chars:
        return content
    if max_chars <= 0:
        return ""
    # Truncate at last newline before max_chars to avoid cutting mid-line
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl]
    return truncated + f"\n... [{label} truncated]"


def enforce_prompt_budget(
    replacements: dict[str, str], budget: int = DEFAULT_PROMPT_BUDGET
) -> dict[str, str]:
    """Progressively truncate injections to fit within total character budget.

    Truncates lowest-priority injections first. Small/fixed injections
    (system_status, rotation_focus, audit_min_issues, recovery_context,
    theme_context) are never truncated.

    Contracts:
        REQUIRES: replacements is a dict of string keys to string values
        REQUIRES: budget > 0
        ENSURES: Sum of non-metadata values in result <= budget (approximately)
        ENSURES: Never raises
    """
    # Exclude metadata keys from budget accounting
    prompt_keys = {k: v for k, v in replacements.items() if not k.startswith("_")}
    if not prompt_keys:
        return replacements
    total = sum(len(v) for v in prompt_keys.values())
    if total <= budget:
        result = dict(replacements)
        result["_budget_info"] = f"{total}/{budget}"
        return result

    result = dict(replacements)
    truncations: list[str] = []
    for key, min_size in _TRUNCATION_ORDER:
        if key not in result:
            continue
        current = len(result[key])
        if current <= min_size:
            continue
        excess = sum(len(v) for k, v in result.items() if not k.startswith("_")) - budget
        if excess <= 0:
            break
        # Account for truncation suffix overhead (~30 chars for "\n... [key truncated]")
        suffix_overhead = len(key) + 20
        cut = min(current - min_size, excess + suffix_overhead)
        result[key] = truncate_injection(result[key], current - cut, key)
        truncations.append(f"{key}: {current}->{len(result[key])}")

    final_total = sum(len(v) for k, v in result.items() if not k.startswith("_"))
    result["_budget_info"] = f"{final_total}/{budget}"
    if truncations:
        result["_budget_truncations"] = ", ".join(truncations)

    # Warn if still over budget after all truncation (#2713)
    if final_total > budget:
        truncatable_keys = {k for k, _ in _TRUNCATION_ORDER}
        non_truncatable = {
            k: len(v) for k, v in result.items()
            if not k.startswith("_") and k not in truncatable_keys and v
        }
        log_warning(
            f"Prompt budget exceeded after truncation: {final_total}/{budget} chars. "
            f"Non-truncatable keys: {non_truncatable}"
        )

    return result

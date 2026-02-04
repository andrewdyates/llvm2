# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/git_context.py - Git-based context gathering.

Functions for extracting context from git history:
- Last directive (## Next from prior same-role commit)
- Other role feedback (recent commits from other roles)
- @ROLE mentions
- Structured handoff context (## Handoff blocks: Markdown or JSON)
"""

from __future__ import annotations

__all__ = [
    "VALID_ROLES",
    "HANDOFF_MAX_CHARS",
    # Functions
    "get_last_directive",
    "get_other_role_feedback",
    "has_role_mention",
    "get_role_mentions",
    "get_handoff_context",
    # TypedDict schemas for IDE autocomplete and type checking
    "HandoffContext",
    "HandoffContextWithMeta",
    "validate_handoff_dict",
]

import json
import re
import warnings
from typing import NotRequired, TypedDict

from looper.config import load_timeout_config
from looper.result import Result
from looper.subprocess_utils import run_git_command

# -----------------------------------------------------------------------------
# Handoff Schema TypedDict Definitions
#
# These provide IDE autocomplete, static type checking, and self-documenting
# schema for structured handoff messages between roles.
#
# Usage:
#     from looper.context.git_context import HandoffContext, validate_handoff_dict
#
#     # Creating a handoff (type-checked by IDE):
#     handoff: HandoffContext = {
#         "target": "PROVER",
#         "issue": 42,
#         "state": "verification_needed",
#         "context": {"files_changed": ["foo.py", "bar.py"]},
#     }
#
#     # Validating a parsed dict:
#     if validate_handoff_dict(data):
#         # data conforms to HandoffContext schema
#         ...
# -----------------------------------------------------------------------------


class HandoffContext(TypedDict, total=False):
    """Schema for structured ## Handoff blocks in commit messages.

    Required fields:
        target: Role(s) to receive the handoff (e.g., "PROVER", ["WORKER", "MANAGER"])
        issue: GitHub issue number this handoff relates to
        state: Current state descriptor (e.g., "verification_needed", "blocked")

    Optional fields:
        context: Nested dict with additional context (e.g., files_changed, error_msg)
        priority: Priority hint (e.g., "urgent", "normal")

    Example commit message:
        ## Handoff
        - target: PROVER
        - issue: 42
        - state: verification_needed
        - context.files_changed: foo.py, bar.py
    """

    # Required fields (validated by _validate_handoff)
    target: str | list[str]
    issue: int
    state: str

    # Optional fields
    context: NotRequired[dict[str, object]]
    priority: NotRequired[str]


class HandoffContextWithMeta(HandoffContext, total=False):
    """HandoffContext with metadata added by get_handoff_context().

    Additional fields added during parsing:
        from_commit: Short commit hash where handoff was found
        from_role: Role that created the handoff (WORKER, PROVER, etc.)
        truncated: True if payload was truncated to fit HANDOFF_MAX_CHARS
    """

    from_commit: NotRequired[str]
    from_role: NotRequired[str]
    truncated: NotRequired[bool]


def validate_handoff_dict(data: dict) -> bool:
    """Validate that a dict conforms to HandoffContext schema.

    This is a runtime validation function for parsed dicts. Use TypedDict
    annotations for static type checking in your code.

    Args:
        data: Dictionary to validate

    Returns:
        True if data has required fields with correct types, False otherwise.

    Example:
        >>> validate_handoff_dict({"target": "PROVER", "issue": 42, "state": "ready"})
        True
        >>> validate_handoff_dict({"target": "PROVER"})  # Missing required fields
        False
    """
    return _validate_handoff(data)


# Valid roles for contract validation
VALID_ROLES = frozenset({"worker", "manager", "researcher", "prover"})


def _get_git_timeout() -> int:
    """Return configured git timeout (seconds)."""
    return load_timeout_config().get("git_default", 5)


def _validate_role(role: str, func_name: str) -> Result[str]:
    """Validate role parameter and return normalized role or failure.

    Contracts:
        REQUIRES: role is passed from caller
        ENSURES: Returns Result.success(normalized_role) if role is valid
        ENSURES: Returns Result.failure with descriptive error if invalid
        ENSURES: Unknown roles are warnings (logged) not errors for flexibility

    Args:
        role: Role string to validate
        func_name: Calling function name for error messages

    Returns:
        Result.success(normalized_role) if valid, Result.failure if invalid.
    """
    if not isinstance(role, str):
        return Result.failure(
            f"{func_name}: role must be string, got {type(role).__name__}"
        )
    normalized = role.strip().lower()
    if not normalized:
        return Result.failure(f"{func_name}: role must be non-empty string")
    # Log warning for unknown roles but don't fail - allows future role extensions
    if normalized not in VALID_ROLES:
        warnings.warn(
            f"{func_name}: unknown role '{normalized}'",
            RuntimeWarning,
            stacklevel=2,
        )
    return Result.success(normalized)


def _join_lines(lines: list[str], max_lines: int) -> str:
    """Join lines with validation and line count enforcement."""
    if not isinstance(lines, list) or max_lines <= 0:
        return ""
    cleaned = [line for line in lines if isinstance(line, str) and line.strip()]
    return "\n".join(cleaned[:max_lines])


def _role_tag_pattern(role_prefix: str) -> re.Pattern[str]:
    """Build regex to match role tags like [W], [W1], or [sat-W1]."""
    return re.compile(rf"\[(?:[^\]]+-)?{role_prefix}\d*\]")


def _extract_role_tag(title: str, role_prefix: str) -> str:
    """Extract role tag from commit title, falling back to base role tag."""
    match = re.match(rf"^\[(?:[^\]]+-)?{role_prefix}\d*\]", title)
    return match.group(0) if match else f"[{role_prefix}]"


def get_last_directive(role: str) -> Result[str]:
    """Extract ## Next section from last commit by same role.

    Contracts:
        REQUIRES: role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no prior commit found
        ENSURES: Returns Result.success with directive if found
        ENSURES: Never raises - all exceptions caught

    Args:
        role: Current role (worker, manager, etc.)

    Returns:
        Result with ## Next directive, or empty string if not found.
    """
    # Contract: validate role input
    validation = _validate_role(role, "get_last_directive")
    if not validation.ok:
        return validation
    normalized_role = validation.value or ""

    role_prefix = normalized_role[0].upper()  # W, M, R, P
    try:
        # Find last commit by this role
        pattern = rf"^\[[^\]]*-?{role_prefix}[0-9]*\]"
        result = run_git_command(
            ["log", "--extended-regexp", f"--grep={pattern}", "--format=%H", "-1"],
            timeout=_get_git_timeout(),
        )
        if not result.ok:
            error = result.error or "unknown error"
            return Result.failure(f"git log failed: {error}")
        commit_hash = (result.value or "").strip()
        if not commit_hash:
            return Result.success("")

        # Get full commit message
        result = run_git_command(
            ["log", "-1", "--format=%B", commit_hash], timeout=_get_git_timeout()
        )
        if not result.ok:
            error = result.error or "unknown error"
            return Result.failure(f"git log failed: {error}")

        msg = result.value or ""
        title = msg.split("\n", 1)[0] if msg else ""
        role_tag = _extract_role_tag(title, role_prefix)

        # Extract ## Next section
        if "## Next" not in msg:
            return Result.success("")

        lines = msg.split("\n")
        in_next = False
        next_lines: list[str] = []
        for line in lines:
            if line.startswith("## Next"):
                in_next = True
                continue
            if in_next:
                if line.startswith(("## ", "---")):
                    break
                next_lines.append(line)

        directive = "\n".join(next_lines).strip()
        if directive:
            return Result.success(
                f"From {role_tag} commit {commit_hash[:7]}:\n{directive}"
            )
        return Result.success("")

    except Exception as exc:
        return Result.failure(f"git log error: {exc}")


def get_other_role_feedback(current_role: str) -> Result[str]:
    """Get recent commits from other roles, prioritized by importance.

    Contracts:
        REQUIRES: current_role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no feedback found
        ENSURES: Excludes current role from output
        ENSURES: Max 5 commits total, max 2 per role
        ENSURES: Priority order preserved: M > R/P > W
        ENSURES: Never raises - catches all exceptions

    Priority order: Manager > Researcher/Prover > Worker
    Excludes current role from results.

    Returns:
        Result with formatted feedback from other roles.
    """
    # Contract: validate role
    validation = _validate_role(current_role, "get_other_role_feedback")
    if not validation.ok:
        return validation

    normalized_role = validation.value or ""
    role_prefix = normalized_role[0].upper()
    # Priority order: M first, then R and P, then W
    priority_order = ["M", "R", "P", "W"]
    other_roles = [r for r in priority_order if r != role_prefix]

    try:
        result = run_git_command(
            ["log", "--oneline", "-30"], timeout=_get_git_timeout()
        )
        if not result.ok:
            error = result.error or "unknown error"
            return Result.failure(f"git log failed: {error}")

        output = result.value or ""
        lines = output.strip().split("\n") if output.strip() else []

        # Collect commits by role, preserving recency within each role
        by_role: dict[str, list[str]] = {r: [] for r in other_roles}
        role_patterns = {role: _role_tag_pattern(role) for role in other_roles}
        for line in lines:
            for role in other_roles:
                if role_patterns[role].search(line):
                    if len(by_role[role]) < 2:  # Max 2 per role
                        by_role[role].append(line)
                    break

        # Build output in priority order
        feedback_lines: list[str] = []
        for role in other_roles:
            feedback_lines.extend(by_role[role])

        if not feedback_lines:
            return Result.success("")

        return Result.success(_join_lines(feedback_lines, 5))  # Max 5 total

    except Exception as exc:
        return Result.failure(f"git log error: {exc}")


def has_role_mention(line: str, tag: str) -> bool:
    """Check if line contains @TAG as a word (not substring).

    Uses word boundary to prevent @WORKER matching @WORKERS.
    Matches @TAG followed by colon, whitespace, or end of line.
    """
    if not isinstance(line, str) or not isinstance(tag, str):
        return False
    if not line or not tag:
        return False
    # Pattern: @TAG followed by : or whitespace or end of string
    pattern = re.escape(tag) + r"(?:[:\s]|$)"
    return bool(re.search(pattern, line))


def get_role_mentions(current_role: str) -> Result[str]:
    """Extract @ROLE mentions directed at the current role from recent commits.

    Contracts:
        REQUIRES: current_role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no mentions found
        ENSURES: Matches @ROLE and @ALL mentions only
        ENSURES: Max 5 mentions returned, deduplicated
        ENSURES: Never raises - catches all exceptions

    Searches recent commit messages for @WORKER, @PROVER, @RESEARCHER, @MANAGER, @ALL
    and returns lines directed at the current role.

    Args:
        current_role: Current role (worker, manager, researcher, prover)

    Returns:
        Result with formatted mentions directed at this role.
    """
    # Contract: validate role
    validation = _validate_role(current_role, "get_role_mentions")
    if not validation.ok:
        return validation

    normalized_role = validation.value or ""
    role_upper = normalized_role.upper()
    role_tag = f"@{role_upper}"

    try:
        # Get full commit messages from recent commits (not just titles)
        result = run_git_command(
            ["log", "--format=%B---COMMIT_SEP---", "-20"],
            timeout=_get_git_timeout(),
        )
        if not result.ok:
            error = result.error or "unknown error"
            return Result.failure(f"git log failed: {error}")

        commits = (result.value or "").split("---COMMIT_SEP---")
        mentions: list[str] = []
        seen: set[str] = set()  # Deduplicate mentions

        for commit in commits:
            if not commit.strip():
                continue

            for line in commit.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check for direct mention or @ALL using word boundary
                if has_role_mention(line, role_tag) or has_role_mention(line, "@ALL"):
                    # Clean up the line - remove the @TAG prefix if it starts with it
                    clean_line = line
                    if clean_line.startswith(role_tag + ":"):
                        clean_line = clean_line[len(role_tag) + 1 :].strip()
                    elif clean_line.startswith("@ALL:"):
                        clean_line = clean_line[5:].strip()

                    # Deduplicate
                    if clean_line in seen:
                        continue
                    seen.add(clean_line)

                    mentions.append(f"- {clean_line}")
                    if len(mentions) >= 5:  # Max 5 mentions
                        break

            if len(mentions) >= 5:
                break

        if not mentions:
            return Result.success("")

        return Result.success(_join_lines(mentions, 5))

    except Exception as exc:
        return Result.failure(f"git log error: {exc}")


# Max chars for handoff payload (approx 200 tokens)
HANDOFF_MAX_CHARS = 1200


def _extract_handoff_block(commit_body: str) -> str | None:
    """Extract ## Handoff block content from commit message.

    Returns the content between ## Handoff and the next ## or --- line,
    or None if no handoff block found.
    """
    if "## Handoff" not in commit_body:
        return None

    lines = commit_body.split("\n")
    in_handoff = False
    handoff_lines: list[str] = []

    for line in lines:
        if line.startswith("## Handoff"):
            in_handoff = True
            continue
        if in_handoff:
            if line.startswith(("## ", "---")):
                break
            handoff_lines.append(line)

    return "\n".join(handoff_lines).strip() or None


def _parse_handoff_json(block: str) -> dict | None:
    """Parse JSON or Markdown key/value entries from handoff block.

    Returns parsed dict or None if parsing fails.
    """
    # Try to extract fenced JSON first (```json ... ```)
    fenced_match = re.search(r"```json\s*([\s\S]*?)```", block)
    if fenced_match:
        json_str = fenced_match.group(1).strip()
    else:
        # Only treat the block as JSON when it looks like JSON
        stripped = block.strip()
        json_str = stripped if stripped.startswith("{") else None

    if json_str is not None:
        if not json_str:
            return None
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (json.JSONDecodeError, ValueError):
            return None

    return _parse_handoff_markdown(block)


def _parse_handoff_markdown(block: str) -> dict | None:
    """Parse Markdown key/value entries from a handoff block."""
    data: dict[str, object] = {}
    context: dict[str, object] = {}

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue

        key: str | None = None
        value: str | None = None

        if line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            if all(set(part) <= {"-", ":"} for part in parts):
                continue
            if parts[0].lower() == "key" and parts[1].lower() == "value":
                continue
            key, value = parts[0], parts[1]
        else:
            if line.startswith(("-", "*")):
                line = line.lstrip("-* ").strip()
            if ":" not in line:
                continue
            key, value = (segment.strip() for segment in line.split(":", 1))

        if not key or value is None or value == "":
            continue

        normalized_key = key.strip().lower()
        parsed_value = _parse_handoff_value(value)
        if normalized_key.startswith("context."):
            context[normalized_key[len("context.") :]] = parsed_value
        else:
            data[normalized_key] = parsed_value

    if context:
        data["context"] = context

    return data or None


def _parse_handoff_value(value: str) -> object:
    """Normalize Markdown values into ints/lists/strings."""
    cleaned = value.strip()
    if (
        (cleaned.startswith('"') and cleaned.endswith('"'))
        or (cleaned.startswith("'") and cleaned.endswith("'"))
    ) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        if inner:
            parts = [
                part.strip().strip('"').strip("'")
                for part in inner.split(",")
                if part.strip()
            ]
            if parts:
                return parts

    if cleaned.isdigit():
        return int(cleaned)

    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) > 1:
            return parts

    return cleaned


def _validate_handoff(data: dict) -> bool:
    """Validate handoff dict has required fields.

    Required: target (str or list), issue (int), state (str).
    """
    if "target" not in data or "issue" not in data or "state" not in data:
        return False

    # target must be string or list of strings
    target = data["target"]
    if not isinstance(target, (str, list)):
        return False
    if isinstance(target, list) and not all(isinstance(t, str) for t in target):
        return False

    # issue must be int
    if not isinstance(data["issue"], int):
        return False

    # state must be string
    if not isinstance(data["state"], str):
        return False

    return True


def _normalize_target(target: str | list) -> list[str]:
    """Normalize target to uppercase list."""
    if isinstance(target, str):
        return [target.upper()]
    return [t.upper() for t in target]


def _role_matches_target(role: str, targets: list[str]) -> bool:
    """Check if role matches any target (including ALL)."""
    role_upper = role.upper()
    return role_upper in targets or "ALL" in targets


def _extract_commit_role(title: str) -> str | None:
    """Extract role from commit title like [W]123 or [M]45."""
    match = re.match(r"^\[([WMRP])\d*\]", title)
    if match:
        role_map = {"W": "WORKER", "M": "MANAGER", "R": "RESEARCHER", "P": "PROVER"}
        return role_map.get(match.group(1))
    return None


def get_handoff_context(role: str) -> Result[str]:
    """Extract structured ## Handoff context targeted at the current role.

    Scans recent commits for ## Handoff blocks (Markdown or JSON), parses and
    validates them, and returns the most recent handoff targeted at this role or ALL.

    Contracts:
        REQUIRES: role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no matching handoff found
        ENSURES: Returns Result.success with JSON string if found
        ENSURES: Payload truncated to HANDOFF_MAX_CHARS with "truncated": true
        ENSURES: Never raises - all exceptions caught

    Args:
        role: Current role (worker, manager, researcher, prover)

    Returns:
        Result with JSON handoff context or empty string.
    """
    # Contract: validate role input
    validation = _validate_role(role, "get_handoff_context")
    if not validation.ok:
        return validation
    normalized_role = validation.value or ""

    try:
        # Get recent commit messages
        result = run_git_command(
            ["log", "--format=%H%n%B---COMMIT_SEP---", "-20"],
            timeout=_get_git_timeout(),
        )
        if not result.ok:
            error = result.error or "unknown error"
            return Result.failure(f"git log failed: {error}")

        commits = (result.value or "").split("---COMMIT_SEP---")

        for commit in commits:
            if not commit.strip():
                continue

            lines = commit.strip().split("\n", 2)
            if len(lines) < 2:
                continue

            commit_hash = lines[0]
            # Body is everything after hash line
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""
            title = lines[1] if len(lines) > 1 else ""

            # Extract handoff block
            block = _extract_handoff_block(body)
            if not block:
                continue

            # Parse JSON
            data = _parse_handoff_json(block)
            if not data:
                continue

            # Validate required fields
            if not _validate_handoff(data):
                continue

            # Normalize and check target
            targets = _normalize_target(data["target"])
            if not _role_matches_target(normalized_role, targets):
                continue

            # Found a matching handoff - add metadata
            from_role = _extract_commit_role(title)
            data["from_commit"] = commit_hash[:7]
            if from_role:
                data["from_role"] = from_role

            # Serialize and check size
            json_str = json.dumps(data, indent=2)
            if len(json_str) > HANDOFF_MAX_CHARS:
                # Truncate context if present
                if "context" in data and isinstance(data["context"], dict):
                    # Remove context items until under limit
                    context = data["context"]
                    keys = list(context.keys())
                    while keys and len(json.dumps(data, indent=2)) > HANDOFF_MAX_CHARS:
                        del context[keys.pop()]
                    # Remove empty context dict to save bytes
                    if not context:
                        del data["context"]
                data["truncated"] = True
                json_str = json.dumps(data, indent=2)
                # Final safety check - if still over limit after removing context,
                # the required fields themselves are too large. Skip this handoff.
                if len(json_str) > HANDOFF_MAX_CHARS:
                    continue  # Try next commit for a smaller handoff

            return Result.success(json_str)

        # No matching handoff found
        return Result.success("")

    except Exception as exc:
        return Result.failure(f"handoff parsing error: {exc}")

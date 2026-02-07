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
from typing import NotRequired

from looper.config import load_injection_caps, load_timeout_config
from looper.context.handoff_schema import HandoffContext
from looper.context.helpers import truncate_injection
from looper.result import Result
from looper.subprocess_utils import run_git_command

# -----------------------------------------------------------------------------
# Handoff Schema TypedDict Definitions
#
# These provide IDE autocomplete, static type checking, and self-documenting
# schema for structured handoff messages between roles. The TypedDicts live
# in looper.context.handoff_schema and are re-exported here for convenience.
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

# Per-injection character caps (Phase 2 of prompt budget design, #2733)
# Applied at source before assembly to prevent any single injection from
# crowding out others during Phase 4's progressive truncation.
# Configurable via .looper_config.json "injection_caps" key (#2745).
_injection_caps: dict[str, int] | None = None


def _get_injection_cap(name: str) -> int:
    """Return the configured per-injection cap for the given injection name."""
    global _injection_caps  # noqa: PLW0603
    if _injection_caps is None:
        _injection_caps = load_injection_caps()
    return _injection_caps[name]


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


def _role_tag_pattern(
    role_prefix: str, worker_id: int | None = None
) -> re.Pattern[str]:
    """Build regex to match role tags like [W], [W1], or [sat-W1].

    Args:
        role_prefix: Single-char role prefix (W, M, R, P)
        worker_id: If set, matches only that specific instance (e.g., W2).
                   If None, matches any instance (W, W1, W2, ...).
    """
    if worker_id is not None:
        return re.compile(rf"\[(?:[^\]]+-)?{role_prefix}{worker_id}\]")
    return re.compile(rf"\[(?:[^\]]+-)?{role_prefix}\d*\]")


def _parse_oneline_role_tag(line: str) -> tuple[str, str] | None:
    """Extract leading role tag from a `git log --oneline` entry.

    Returns:
        Tuple of (role_prefix, instance_suffix) if subject starts with a role tag.
        instance_suffix is empty string for bare tags like [W].
        None if line is malformed or subject has no leading role tag.
    """
    if not isinstance(line, str):
        return None
    parts = line.split(" ", 1)
    if len(parts) != 2:
        return None
    subject = parts[1]
    match = re.match(r"^\[(?:[^\]]+-)?([A-Z])(\d*)\]", subject)
    if not match:
        return None
    return (match.group(1), match.group(2))


def _parse_feedback_commits(raw_log: str) -> list[tuple[str, str]]:
    """Parse git log output into (oneline_header, commit_body) tuples."""
    text = raw_log.strip()
    if not text:
        return []

    # Backward compatibility: older callers/tests may still return --oneline output.
    if "---COMMIT_SEP---" not in text:
        return [(line.strip(), "") for line in text.splitlines() if line.strip()]

    commits: list[tuple[str, str]] = []
    for chunk in text.split("---COMMIT_SEP---"):
        block = chunk.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        if not header:
            continue
        body = "\n".join(lines[1:]).strip()
        commits.append((header, body))
    return commits


def _extract_section_lines(
    commit_body: str,
    section: str,
    max_lines: int = 3,
) -> list[str]:
    """Extract up to max_lines from a markdown section (e.g., ## Next).

    Lines are extracted from the first occurrence of `## {section}` until
    the next `## ` header or `---` separator. Blank lines are skipped.
    """
    if not commit_body or max_lines <= 0:
        return []
    header = f"## {section}"
    if header not in commit_body:
        return []

    in_section = False
    extracted: list[str] = []
    for line in commit_body.splitlines():
        if line.strip() == header:
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith(("## ", "---")):
            break
        stripped = line.strip()
        if not stripped:
            continue
        extracted.append(stripped)
        if len(extracted) >= max_lines:
            break
    return extracted


def _extract_directive_lines(
    commit_body: str, max_lines: int = 3, sections: tuple[str, ...] = ("Next",)
) -> list[str]:
    """Extract @ROLE/@ALL directive lines from specified sections.

    Only returns lines with a line-leading @TAG directive (after optional
    list marker). Mid-line mentions are excluded. See #2702, #2717, #2727.
    """
    all_lines: list[str] = []
    for section in sections:
        all_lines.extend(_extract_section_lines(commit_body, section, max_lines=10))
    directives: list[str] = []
    for line in all_lines:
        if is_directive_mention(line):
            directives.append(line)
            if len(directives) >= max_lines:
                break
    return directives


def _format_feedback_entry(oneline: str, commit_body: str) -> str:
    """Format other-role feedback with ## Learned and @ROLE directives.

    Full ## Next is omitted because git_log already includes it for recent
    commits. However, actionable @ROLE/@ALL directive lines from ## Next and
    ## Handoff are preserved so cross-role coordination is not lost for commits
    outside the git_log window (commits 11-30). See #2673, #2702, #2727.
    """
    learned_lines = _extract_section_lines(commit_body, "Learned", max_lines=3)
    directive_lines = _extract_directive_lines(
        commit_body, max_lines=3, sections=("Next", "Handoff")
    )

    if not learned_lines and not directive_lines:
        return oneline

    lines = [oneline]
    if directive_lines:
        lines.append("  ## Directives")
        lines.extend(f"  {line}" for line in directive_lines)
    if learned_lines:
        lines.append("  ## Learned")
        lines.extend(f"  {line}" for line in learned_lines)
    return "\n".join(lines)


def _extract_role_tag(title: str, role_prefix: str) -> str:
    """Extract role tag from commit title, falling back to base role tag."""
    match = re.match(rf"^\[(?:[^\]]+-)?{role_prefix}\d*\]", title)
    return match.group(0) if match else f"[{role_prefix}]"


def get_last_directive(role: str, worker_id: int | None = None) -> Result[str]:
    """Extract ## Next section from last commit by same role.

    Contracts:
        REQUIRES: role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no prior commit found
        ENSURES: Returns Result.success with directive if found
        ENSURES: Never raises - all exceptions caught

    Args:
        role: Current role (worker, manager, etc.)
        worker_id: If set, only match commits from this specific worker instance
                   (e.g., worker_id=1 matches [W1] but not [W2] or [W]).

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
        # Find last commit by this role (or specific instance)
        if worker_id is not None:
            pattern = rf"^\[([^]]+-)?{role_prefix}{worker_id}\]"
        else:
            pattern = rf"^\[([^]]+-)?{role_prefix}[0-9]*\]"
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

        # Extract ## Why, ## Learned, ## Next sections from commit body
        sections_to_extract = ("## Why", "## Learned", "## Next")
        lines = msg.split("\n")
        extracted: dict[str, list[str]] = {}
        current_section: str | None = None

        for line in lines:
            if any(line.startswith(s) for s in sections_to_extract):
                current_section = line.split(None, 2)[1] if len(line.split(None, 2)) > 1 else line
                # Use the header text as key (e.g., "Why", "Learned", "Next")
                current_section = line.strip()
                extracted[current_section] = []
                continue
            if current_section is not None:
                if line.startswith(("## ", "---")):
                    current_section = None
                    continue
                extracted[current_section].append(line)

        # Build output with all found sections
        parts: list[str] = [f"Last commit: {title}"]
        for section_header in sections_to_extract:
            if section_header in extracted:
                content = "\n".join(extracted[section_header]).strip()
                if content:
                    parts.append(f"> {section_header}\n> {content.replace(chr(10), chr(10) + '> ')}")

        if len(parts) > 1:
            directive = "\n\n".join(parts)
            return Result.success(
                truncate_injection(directive, _get_injection_cap("last_directive"), "last_directive")
            )
        return Result.success("")

    except Exception as exc:
        return Result.failure(f"git log error: {exc}")


def get_other_role_feedback(
    current_role: str,
    worker_id: int | None = None,
    exclude_hashes: set[str] | None = None,
) -> Result[str]:
    """Get recent commits from other roles, prioritized by importance.

    Contracts:
        REQUIRES: current_role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no feedback found
        ENSURES: Excludes current worker instance but includes other workers (#2566)
        ENSURES: Skips commits whose 7-char hash is in exclude_hashes (#2688)
        ENSURES: Max 7 commits total, max 2 per role/instance
        ENSURES: Priority order preserved: M > R/P > W (other instances)
        ENSURES: Classification is based on the leading role tag only
        ENSURES: Includes up to 3 ## Learned lines + up to 3 @ROLE directive lines from ## Next and ## Handoff
        ENSURES: Never raises - catches all exceptions

    Priority order: Manager > Researcher/Prover > Worker (other instances)
    When worker_id is set, excludes only that specific instance (e.g., W1)
    but includes other worker instances (W2, W3) as cross-worker feedback.

    Args:
        current_role: Current role (worker, manager, etc.)
        worker_id: If set, only exclude this specific worker instance.
                   Other worker instances are included in feedback.
        exclude_hashes: Set of 7-char abbreviated hashes to skip (already shown
                        in git_log). See #2688.

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
            ["log", "--format=%h %s%n%b%n---COMMIT_SEP---", "-30"],
            timeout=_get_git_timeout(),
        )
        if not result.ok:
            error = result.error or "unknown error"
            return Result.failure(f"git log failed: {error}")

        commits = _parse_feedback_commits(result.value or "")
        if not commits:
            return Result.success("")

        # Collect commits by role, preserving recency within each role
        by_role: dict[str, list[str]] = {r: [] for r in other_roles}

        # Include other instances of the same role when worker_id is set (#2566, #2589)
        # Applies to all roles (Worker, Prover, Researcher, Manager), not just Workers.
        # Only include explicit instance tags (e.g., [W2], [sat-W3]), so cross-instance
        # feedback is always clearly attributable to a specific peer instance.
        include_other_instances = worker_id is not None
        my_instance = str(worker_id) if worker_id is not None else None
        other_instance_lines: list[str] = []

        skip_hashes = exclude_hashes or set()

        for oneline, body in commits:
            # Skip commits already shown in git_log (#2688).
            # Both sides normalize to 7 chars: git_log uses %H[:7],
            # other_feedback uses %h[:7]. Requires core.abbrev >= 7 (git default).
            commit_hash = oneline.split(" ", 1)[0][:7] if oneline else ""
            if commit_hash and commit_hash in skip_hashes:
                continue

            parsed = _parse_oneline_role_tag(oneline)
            if parsed is None:
                continue
            line_role_prefix, line_instance = parsed
            formatted = _format_feedback_entry(oneline, body)

            # Check if it's from another instance of our role (cross-instance visibility)
            if line_role_prefix == role_prefix:
                if include_other_instances and line_instance:
                    if line_instance == my_instance:
                        continue
                    if len(other_instance_lines) < 2:  # Max 2 from other instances
                        other_instance_lines.append(formatted)
                continue

            if line_role_prefix in by_role and len(by_role[line_role_prefix]) < 2:
                by_role[line_role_prefix].append(formatted)

        # Build output in priority order
        feedback_lines: list[str] = []
        for role in other_roles:
            feedback_lines.extend(by_role[role])

        # Append other same-role instance commits at the end
        if other_instance_lines:
            feedback_lines.extend(other_instance_lines)

        if not feedback_lines:
            return Result.success("")

        feedback = "\n\n".join(feedback_lines[:7])  # Max 7 total
        return Result.success(
            truncate_injection(feedback, _get_injection_cap("other_feedback"), "other_feedback")
        )

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


# Compiled regex for directive-leading position (#2717).
# Matches @TAG at the start of the line, after optional list/checkbox markers.
_DIRECTIVE_LEADING_RE = re.compile(
    r"^(?:[-*+]\s+|\d+\.\s+)?"  # optional list marker (-, *, +, or numbered)
    r"(?:\[[ xX]\]\s+)?"  # optional markdown checkbox marker
    r"@(?:WORKER|PROVER|RESEARCHER|MANAGER|ALL|[WPRM]\d+)"
    r"(?:[:\s]|$)"  # followed by colon, whitespace, or end
)


def is_directive_mention(line: str) -> bool:
    """Check if line is a directive-form mention (line-leading @TAG).

    Unlike has_role_mention() which matches @TAG anywhere in the line,
    this requires @TAG at line-leading position (with optional list marker).
    Mid-line references like 'supports @ALL directives' do not match.
    See #2717.
    """
    if not isinstance(line, str) or not line:
        return False
    return bool(_DIRECTIVE_LEADING_RE.match(line.lstrip()))


def _is_mention_owned_by_other_worker(
    line: str, worker_id: int | None
) -> bool:
    """Check if a broadcast @WORKER mention references issues owned by another worker.

    When a mention line contains issue references like #123, checks whether those
    issues are owned by a different worker (via W<N> labels). If ALL referenced
    issues are owned by other workers, returns True (filter it out). If no issues
    are referenced, or any referenced issue has no worker label or matches the
    current worker, returns False (keep it).

    Uses IterationIssueCache (already populated during session start) to avoid
    additional API calls (#2562).
    """
    if worker_id is None:
        return False

    issue_refs = re.findall(r"#(\d+)", line)
    if not issue_refs:
        return False  # No issue references = true broadcast, keep it

    # Lazy import to avoid circular dependency (git_context <- issue_cache)
    from looper.context.issue_cache import IterationIssueCache
    from looper.context.helpers import has_label

    cache_result = IterationIssueCache.get_all()
    if not cache_result.ok:
        return False  # Can't check ownership, keep the mention

    issues_by_num: dict[int, dict[str, object]] = {}
    for issue in cache_result.value or []:
        num = issue.get("number")
        if num is not None:
            issues_by_num[int(num)] = issue  # type: ignore[arg-type]

    my_label = f"W{worker_id}"
    all_owned_by_others = True
    for ref in issue_refs:
        issue = issues_by_num.get(int(ref))
        if issue is None:
            # Issue not found in cache (closed/deleted), keep mention
            all_owned_by_others = False
            break
        # Check for any W<N> ownership label on this issue
        has_any_worker_label = any(
            re.match(r"^W\d+$", lbl.get("name", "") if isinstance(lbl, dict) else str(lbl))
            for lbl in issue.get("labels", [])  # type: ignore[union-attr]
        )
        if not has_any_worker_label:
            # No worker ownership label = unassigned, keep for all workers
            all_owned_by_others = False
            break
        if has_label(issue, my_label):
            # Owned by me, keep it
            all_owned_by_others = False
            break

    return all_owned_by_others


def get_role_mentions(
    current_role: str, worker_id: int | None = None
) -> Result[str]:
    """Extract @ROLE mentions directed at the current role from recent commits.

    Contracts:
        REQUIRES: current_role is a non-empty string
        ENSURES: Returns Result.failure if role validation fails
        ENSURES: Returns Result.success("") if no mentions found
        ENSURES: Matches @ROLE, @W<N> (instance-level), and @ALL mentions
        ENSURES: Broadcast @WORKER mentions for issues owned by other workers filtered
        ENSURES: Max 5 mentions returned, deduplicated
        ENSURES: Never raises - catches all exceptions

    Searches recent commit messages for @WORKER, @W1, @PROVER, @RESEARCHER,
    @MANAGER, @ALL and returns lines directed at the current role/instance.

    When worker_id is set, broadcast @WORKER mentions that reference issues
    owned by another worker (via W<N> labels) are filtered out (#2562).

    Args:
        current_role: Current role (worker, manager, researcher, prover)
        worker_id: If set, also matches instance-level mentions like @W1
            and filters broadcast mentions by issue ownership.

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

    # Build instance-level tag if worker_id is set (e.g., @W1)
    role_prefix = normalized_role[0].upper()
    instance_tag = f"@{role_prefix}{worker_id}" if worker_id is not None else None

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

            # Only scan directive sections (## Next, ## Handoff) — not
            # ## Changes, ## Why, ## Learned, ## Verified (#2716)
            directive_lines: list[str] = []
            for section in ("Next", "Handoff"):
                directive_lines.extend(
                    _extract_section_lines(commit, section, max_lines=20)
                )

            for line in directive_lines:
                if not is_directive_mention(line):
                    continue

                # Check for direct mention, instance mention, or @ALL
                is_instance_mention = (
                    instance_tag is not None and has_role_mention(line, instance_tag)
                )
                is_broadcast = (
                    has_role_mention(line, role_tag)
                    or has_role_mention(line, "@ALL")
                )
                matched = is_instance_mention or is_broadcast

                # Filter broadcast @WORKER mentions that reference issues
                # owned by another worker (#2562). Instance mentions (@W1)
                # bypass this filter - they were explicitly targeted.
                if matched and not is_instance_mention and worker_id is not None:
                    if _is_mention_owned_by_other_worker(line, worker_id):
                        continue

                if matched:
                    # Clean up: strip list/checkbox marker then @TAG prefix.
                    clean_line = re.sub(
                        r"^(?:[-*+]\s+|\d+\.\s+)?(?:\[[ xX]\]\s+)?",
                        "",
                        line.strip(),
                    )
                    for mention_tag in (instance_tag, role_tag, "@ALL"):
                        if not mention_tag:
                            continue
                        if clean_line.startswith(mention_tag):
                            clean_line = clean_line[len(mention_tag) :].lstrip()
                            if clean_line.startswith(":"):
                                clean_line = clean_line[1:].lstrip()
                            break

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


def _validate_handoff(data: dict, strict: bool | None = None) -> bool:
    """Validate handoff dict has required fields and schema conformance.

    Required: target (str or list), issue (int), state (str).

    Per designs/2026-02-04-structured-output-validation.md, validation modes:
    - strict=None (default): check AIT_HANDOFF_STRICT env var
    - strict=True: reject unknown states and invalid depends_on
    - strict=False: warn on unknown states but allow

    Args:
        data: Dictionary to validate
        strict: Strictness mode (default: check env var)

    Returns:
        True if valid, False if required fields missing or malformed.
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

    # Validate state and target against enums (warn-then-strict per design)
    try:
        from looper.context.handoff_schema import (
            validate_depends_on,
            validate_state,
            validate_target,
        )

        validate_state(data["state"], strict=strict)
        validate_target(data["target"], strict=strict)

        # Validate depends_on if present
        if "depends_on" in data:
            validate_depends_on(data["depends_on"], strict=strict)
    except ValueError:
        # Strict mode rejection
        return False
    except ImportError:
        # Schema module not available - continue with basic validation
        pass

    return True


def _normalize_target(target: str | list) -> list[str]:
    """Normalize target to uppercase list."""
    if isinstance(target, str):
        return [target.upper()]
    return [t.upper() for t in target]


def _role_matches_target(
    role: str, targets: list[str], worker_id: int | None = None
) -> bool:
    """Check if role matches any target (including ALL and instance-level).

    Supports instance-level targeting: if worker_id is set, matches targets
    like "W1" in addition to "WORKER" and "ALL".

    Args:
        role: Current role (e.g., "worker")
        targets: List of uppercase target strings (e.g., ["WORKER", "W1", "ALL"])
        worker_id: If set, also checks instance-level targets like "W1"
    """
    role_upper = role.upper()
    if role_upper in targets or "ALL" in targets:
        return True
    # Check instance-level targeting (e.g., target="W1" for worker_id=1)
    if worker_id is not None:
        role_prefix = role_upper[0]
        instance_tag = f"{role_prefix}{worker_id}"
        if instance_tag in targets:
            return True
    return False


def _extract_commit_role(title: str) -> tuple[str | None, int | None]:
    """Extract role and instance ID from commit title like [W]123 or [W1]45.

    Returns:
        Tuple of (role_name, worker_id). worker_id is None if no instance number.
        Examples: ("WORKER", 1), ("MANAGER", None), (None, None)
    """
    match = re.match(r"^\[([WMRP])(\d*)\]", title)
    if match:
        role_map = {"W": "WORKER", "M": "MANAGER", "R": "RESEARCHER", "P": "PROVER"}
        role_name = role_map.get(match.group(1))
        instance_id = int(match.group(2)) if match.group(2) else None
        return role_name, instance_id
    return None, None


def get_handoff_context(role: str, worker_id: int | None = None) -> Result[str]:
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
        worker_id: If set, matches instance-level targets like "W1" in addition
                   to role-level targets like "WORKER" and "ALL".

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
            if not _role_matches_target(normalized_role, targets, worker_id):
                continue

            # Found a matching handoff - add metadata
            from_role, from_instance = _extract_commit_role(title)
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

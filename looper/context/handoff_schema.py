# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Handoff schema definitions and validation.

Part of #2430: Implement handoff schema validation.
Design: designs/2026-02-04-structured-output-validation.md

This module centralizes:
- HandoffState enum with standard workflow states
- JSON Schema for handoff validation and future structured output support
- Validation helpers with warn-then-strict mode
"""

from __future__ import annotations

__all__ = [
    "HandoffState",
    "HandoffDependency",
    "HandoffContext",
    "HANDOFF_JSON_SCHEMA",
    "VALID_TARGETS",
    "validate_state",
    "validate_target",
    "validate_depends_on",
    "get_handoff_schema",
]

import json
import os
import warnings
from enum import Enum
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from typing import Any


class HandoffDependency(TypedDict, total=False):
    """Dependency reference for handoff depends_on entries."""

    type: str
    path: NotRequired[str]
    number: NotRequired[int]
    hash: NotRequired[str]
    output: NotRequired[str]


class HandoffContext(TypedDict, total=False):
    """Schema for structured ## Handoff blocks in commit messages."""

    target: str | list[str]
    issue: int
    state: str
    context: NotRequired[dict[str, object]]
    priority: NotRequired[str]
    depends_on: NotRequired[list[HandoffDependency]]


class HandoffState(Enum):
    """Standard handoff states with semantic meaning.

    Per designs/2026-02-02-handoff-schema-extensions.md, these states
    align with the role-issue workflow.
    """

    # Worker -> Prover flow
    IMPLEMENTATION_DONE = "implementation_done"  # Code written, needs verification
    VERIFICATION_NEEDED = "verification_needed"  # Explicit verification request

    # Prover -> Worker flow
    VERIFICATION_FAILED = "verification_failed"  # Tests/proofs failed
    VERIFICATION_PASSED = "verification_passed"  # All checks passed

    # Researcher -> Worker flow
    DESIGN_READY = "design_ready"  # Design doc complete, ready to implement
    DESIGN_REVIEW = "design_review"  # Design needs review before implementation

    # Worker -> Researcher flow
    RESEARCH_NEEDED = "research_needed"  # Need investigation before proceeding

    # Any role -> Manager flow
    AUDIT_NEEDED = "audit_needed"  # Needs manager review
    BLOCKED = "blocked"  # Cannot proceed, needs unblocking

    # Manager -> Any role flow
    PRIORITY_CHANGE = "priority_change"  # Priority has changed
    REDIRECT = "redirect"  # Work should shift to different focus

    # Cross-role
    HANDOFF_COMPLETE = "handoff_complete"  # Generic completion marker

    @classmethod
    def from_string(cls, value: str) -> "HandoffState | None":
        """Parse string to state, case-insensitive.

        Normalizes hyphens and spaces to underscores.

        Args:
            value: State string to parse

        Returns:
            HandoffState if valid, None otherwise.
        """
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        for state in cls:
            if state.value == normalized:
                return state
        return None

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid state values."""
        return [state.value for state in cls]


# Valid targets for handoff target field
# Includes ALL for broadcast handoffs (e.g., @ALL mentions)
VALID_TARGETS = frozenset({"WORKER", "PROVER", "RESEARCHER", "MANAGER", "ALL"})

# Valid dependency types for depends_on field
VALID_DEPENDENCY_TYPES = frozenset({"design", "issue", "commit", "file"})


# JSON Schema for handoff validation and structured output support
HANDOFF_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["target", "issue", "state"],
    "properties": {
        "target": {
            "oneOf": [
                {"type": "string", "enum": list(VALID_TARGETS)},
                {"type": "array", "items": {"type": "string", "enum": list(VALID_TARGETS)}},
            ],
            "description": "Role(s) to receive the handoff",
        },
        "issue": {
            "type": "integer",
            "description": "GitHub issue number this handoff relates to",
        },
        "state": {
            "type": "string",
            "enum": [state.value for state in HandoffState],
            "description": "Current workflow state",
        },
        "context": {
            "type": "object",
            "additionalProperties": True,
            "description": "Additional context (files_changed, error_msg, etc.)",
        },
        "priority": {
            "type": "string",
            "enum": ["urgent", "normal", "low"],
            "description": "Priority hint",
        },
        "depends_on": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(VALID_DEPENDENCY_TYPES),
                    },
                    "path": {"type": "string"},
                    "number": {"type": "integer"},
                    "hash": {"type": "string"},
                    "output": {"type": "string"},
                },
            },
            "description": "Explicit output dependencies",
        },
    },
    "additionalProperties": True,
}


def get_handoff_schema() -> str:
    """Return JSON Schema as string for prompt/tool integrations.

    This allows the schema to be exported for future provider-native
    structured output features.

    Returns:
        JSON Schema as formatted string.
    """
    return json.dumps(HANDOFF_JSON_SCHEMA, indent=2)


def validate_state(state: str, strict: bool | None = None) -> bool:
    """Validate handoff state against the enum.

    Args:
        state: State string to validate
        strict: If True, reject unknown states. If False, warn and allow.
                If None (default), check AIT_HANDOFF_STRICT env var.

    Returns:
        True if state is valid (or strict=False and state is unknown).

    Raises:
        ValueError: If strict=True and state is not in enum.
    """
    # Determine strictness from env var if not specified
    if strict is None:
        strict = os.environ.get("AIT_HANDOFF_STRICT", "").lower() in ("1", "true")

    parsed = HandoffState.from_string(state)

    if parsed is not None:
        return True

    if strict:
        valid_states = ", ".join(HandoffState.values())
        raise ValueError(
            f"Unknown handoff state '{state}'. Valid states: {valid_states}"
        )

    # Warn but allow in lenient mode
    warnings.warn(
        f"Unknown handoff state '{state}'. Consider using one of: "
        f"{', '.join(HandoffState.values()[:5])}...",
        UserWarning,
        stacklevel=2,
    )
    return True


def validate_target(target: str | list[str], strict: bool | None = None) -> bool:
    """Validate handoff target against valid roles.

    Args:
        target: Target role string or list of target roles
        strict: If True, reject unknown targets. If False, warn and allow.
                If None (default), check AIT_HANDOFF_STRICT env var.

    Returns:
        True if all targets are valid (or strict=False and targets have issues).

    Raises:
        ValueError: If strict=True and any target is not a valid role.
    """
    if strict is None:
        strict = os.environ.get("AIT_HANDOFF_STRICT", "").lower() in ("1", "true")

    # Normalize to list for uniform processing
    targets = [target] if isinstance(target, str) else target

    for t in targets:
        normalized = t.upper()
        if normalized in VALID_TARGETS:
            continue

        if strict:
            valid_targets = ", ".join(sorted(VALID_TARGETS))
            raise ValueError(
                f"Unknown handoff target '{t}'. Valid targets: {valid_targets}"
            )

        # Warn but allow in lenient mode
        warnings.warn(
            f"Unknown handoff target '{t}'. Valid targets: "
            f"{', '.join(sorted(VALID_TARGETS))}",
            UserWarning,
            stacklevel=2,
        )

    return True


def validate_depends_on(deps: object, strict: bool | None = None) -> bool:
    """Validate depends_on field structure.

    Args:
        deps: Value to validate (should be list of dicts)
        strict: If True, reject invalid entries. If False, warn and allow.
                If None, check AIT_HANDOFF_STRICT env var.

    Returns:
        True if deps is valid (or strict=False and deps has issues).

    Raises:
        ValueError: If strict=True and deps is malformed.
    """
    if strict is None:
        strict = os.environ.get("AIT_HANDOFF_STRICT", "").lower() in ("1", "true")

    if not isinstance(deps, list):
        if strict:
            raise ValueError("depends_on must be a list")
        return True  # Lenient: allow non-list

    for i, dep in enumerate(deps):
        if not isinstance(dep, dict):
            if strict:
                raise ValueError(f"depends_on[{i}] must be a dict")
            continue

        if "type" not in dep:
            if strict:
                raise ValueError(f"depends_on[{i}] missing required 'type' field")
            continue

        dep_type = dep["type"]
        if dep_type not in VALID_DEPENDENCY_TYPES:
            if strict:
                raise ValueError(
                    f"depends_on[{i}] has invalid type '{dep_type}'. "
                    f"Valid: {', '.join(VALID_DEPENDENCY_TYPES)}"
                )
            warnings.warn(
                f"Unknown dependency type '{dep_type}' in depends_on[{i}]",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Type-specific required fields
        required_field = {
            "design": "path",
            "issue": "number",
            "commit": "hash",
            "file": "path",
        }.get(dep_type)

        if required_field and required_field not in dep:
            if strict:
                raise ValueError(
                    f"depends_on[{i}] type '{dep_type}' requires '{required_field}'"
                )
            warnings.warn(
                f"depends_on[{i}] type '{dep_type}' missing '{required_field}'",
                UserWarning,
                stacklevel=2,
            )

    return True

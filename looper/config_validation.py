# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/config_validation.py - Shared config validation utilities.

Common patterns extracted from:
- looper/config.py (unknown key warnings, type/bounds validation)
- ai_template_scripts/pulse/config.py (unknown key warnings, type validation)
- ai_template_scripts/cargo_wrapper/timeouts.py (unknown key warnings)

Part of #2008: Extract shared config validation utility.
"""

from __future__ import annotations

__all__ = [
    "check_unknown_keys",
    "validate_type",
    "validate_bounds",
    "merge_configs",
]

from typing import Any, TypeVar

T = TypeVar("T")


def check_unknown_keys(
    config: dict[str, Any],
    known_keys: set[str],
    context: str,
    logger: callable | None = None,
) -> list[str]:
    """Check for unknown keys in a config dict.

    Args:
        config: Config dict to validate.
        known_keys: Set of known/expected keys.
        context: Context string for warning messages (e.g., "pulse.toml", "[thresholds]").
        logger: Optional callable to emit warnings. If None, warnings are returned only.

    Returns:
        List of unknown keys found (sorted for deterministic output).

    Example:
        >>> unknown = check_unknown_keys({"foo": 1, "bar": 2}, {"foo"}, "config.toml")
        >>> print(unknown)
        ['bar']
    """
    unknown = sorted(set(config.keys()) - known_keys)
    if unknown and logger:
        logger(f"Warning: {context} has unknown keys: {', '.join(unknown)}")
    return unknown


def validate_type(
    value: Any,
    expected_type: type | str,
    key: str,
    context: str,
) -> tuple[bool, str | None]:
    """Validate that a value has the expected type.

    Args:
        value: The value to validate.
        expected_type: Expected type (type object or "list[str]" string).
        key: Config key name for error messages.
        context: Context string for error messages.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.

    Example:
        >>> valid, msg = validate_type(42, int, "count", "config.toml")
        >>> print(valid, msg)
        True None

        >>> valid, msg = validate_type("hello", int, "count", "config.toml")
        >>> print(valid, msg)
        False Error [config.toml]: 'count' expected int, got str
    """
    # Handle special list types
    if expected_type == "list[str]":
        if not isinstance(value, list):
            return (
                False,
                f"Error [{context}]: '{key}' expected list, got {type(value).__name__}",
            )
        if value and not all(isinstance(v, str) for v in value):
            return False, f"Error [{context}]: '{key}' expected list of strings"
        return True, None

    # Handle simple types
    if isinstance(expected_type, type):
        if not isinstance(value, expected_type):
            # Allow int where float expected (e.g., 0 instead of 0.0)
            if expected_type is float and isinstance(value, int):
                return True, None
            exp = expected_type.__name__
            got = type(value).__name__
            return False, f"Error [{context}]: '{key}' expected {exp}, got {got}"

    return True, None


def validate_bounds(
    value: Any,
    key: str,
    context: str,
    min_val: float | None = None,
    max_val: float | None = None,
    allowed: list[str] | None = None,
) -> list[str]:
    """Validate that a value is within bounds or in allowed values.

    Args:
        value: The value to validate.
        key: Config key name for warning messages.
        context: Context string for warning messages.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        allowed: List of allowed string values.

    Returns:
        List of warning messages (empty if valid).

    Example:
        >>> msgs = validate_bounds(150, "timeout", "config", min_val=0, max_val=100)
        >>> print(msgs)
        ["Warning [config]: 'timeout' value 150 is above maximum 100"]
    """
    warnings: list[str] = []

    # Check min bound
    if min_val is not None and isinstance(value, (int, float)):
        if value < min_val:
            warnings.append(
                f"Warning [{context}]: '{key}' value {value} is below minimum {min_val}"
            )

    # Check max bound
    if max_val is not None and isinstance(value, (int, float)):
        if value > max_val:
            warnings.append(
                f"Warning [{context}]: '{key}' value {value} is above maximum {max_val}"
            )

    # Check allowed values
    if allowed is not None and isinstance(value, str):
        if value not in allowed:
            allowed_str = ", ".join(repr(v) for v in allowed)
            warnings.append(
                f"Warning [{context}]: '{key}' value {value!r} is not in allowed values: [{allowed_str}]"
            )

    return warnings


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple config dicts with later dicts taking precedence.

    Args:
        *configs: Config dicts to merge, in increasing priority order.

    Returns:
        Merged config dict.

    Example:
        >>> base = {"a": 1, "b": 2}
        >>> override = {"b": 3, "c": 4}
        >>> merged = merge_configs(base, override)
        >>> print(merged)
        {'a': 1, 'b': 3, 'c': 4}
    """
    result: dict[str, Any] = {}
    for config in configs:
        if config:
            result.update(config)
    return result

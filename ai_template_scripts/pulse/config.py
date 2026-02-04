#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Configuration loading and validation for pulse module.

Handles loading pulse.toml config, applying threshold overrides,
and managing mutable config state.

Part of #404: pulse.py module split
"""

import sys
import tomllib

from looper.config_validation import check_unknown_keys

from .constants import (
    CONFIG_PATHS,
    DEFAULT_THRESHOLDS,
    KNOWN_LARGE_FILES_KEYS,
    KNOWN_RUNTIME_KEYS,
    KNOWN_SECTIONS,
)

# Module-level mutable state (set by _apply_config at import time)
LARGE_FILE_EXCLUDE_PATTERNS: list[str] = []
SKIP_ORPHANED_TESTS: bool = False


def _warn_unknown_keys(config: dict, config_path: str) -> None:
    """Emit warnings for unknown keys in pulse.toml config.

    Per design in designs/2026-01-31-pulse-cargo-unknown-key-warnings.md:
    - Warn on unknown top-level sections
    - Warn on unknown keys within known sections
    - Sort keys for deterministic output

    Uses shared check_unknown_keys() utility from looper/config_validation.py.

    Args:
        config: Parsed config dict.
        config_path: Path to config file (for warning messages).
    """

    def stderr_warn(msg: str) -> None:
        """Print warning to stderr."""
        print(msg, file=sys.stderr)

    # Check for unknown top-level sections
    check_unknown_keys(config, KNOWN_SECTIONS, config_path, logger=stderr_warn)

    # Check for unknown keys in [thresholds]
    thresholds = config.get("thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            stderr_warn(f"Warning: {config_path} [thresholds] must be a table")
        else:
            check_unknown_keys(
                thresholds,
                set(DEFAULT_THRESHOLDS.keys()),
                f"{config_path} [thresholds]",
                logger=stderr_warn,
            )

    # Check for unknown keys in [large_files]
    large_files = config.get("large_files")
    if large_files is not None:
        if not isinstance(large_files, dict):
            stderr_warn(f"Warning: {config_path} [large_files] must be a table")
        else:
            check_unknown_keys(
                large_files,
                KNOWN_LARGE_FILES_KEYS,
                f"{config_path} [large_files]",
                logger=stderr_warn,
            )

    # Check for unknown keys in [runtime]
    runtime = config.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, dict):
            stderr_warn(f"Warning: {config_path} [runtime] must be a table")
        else:
            check_unknown_keys(
                runtime,
                KNOWN_RUNTIME_KEYS,
                f"{config_path} [runtime]",
                logger=stderr_warn,
            )


def _load_config() -> tuple[dict, str | None]:
    """Load pulse config from pulse.toml if present.

    Searches for config in: pulse.toml, ai_template_scripts/pulse.toml, .pulse.toml

    Config format (all keys optional):
        [thresholds]
        max_file_lines = 1000
        max_files_over_limit = 5

        [large_files]
        exclude_patterns = ["tests/", "crates/star/tests/"]

        [runtime]
        skip_orphaned_tests = true  # Skip orphan test detection entirely

    Returns:
        Tuple of (config dict, config path string). Path is None if no config found.
    """
    for config_path in CONFIG_PATHS:
        if not config_path.exists():
            continue
        try:
            content = config_path.read_text()
            config = tomllib.loads(content)
            path_str = str(config_path)
            _warn_unknown_keys(config, path_str)
            return config, path_str
        except (OSError, tomllib.TOMLDecodeError) as e:
            # Log but don't fail - default config still works
            print(f"Warning: Error reading {config_path}: {e}", file=sys.stderr)
    return {}, None


def _apply_config(config: dict, config_path: str | None = None) -> dict:
    """Apply config overrides to default thresholds.

    Args:
        config: Parsed config dict from _load_config().
        config_path: Path to config file (for warning messages).

    Returns:
        Dict with merged thresholds (defaults + overrides).
    """
    global LARGE_FILE_EXCLUDE_PATTERNS, SKIP_ORPHANED_TESTS

    # Start with defaults
    thresholds = DEFAULT_THRESHOLDS.copy()

    # Apply threshold overrides
    config_thresholds = config.get("thresholds", {})
    for key, value in config_thresholds.items():
        if key in thresholds:
            if isinstance(value, (int, float)):
                thresholds[key] = int(value)
            elif config_path:
                print(
                    f"Warning: {config_path} [thresholds] {key} must be numeric "
                    f"(got {type(value).__name__})",
                    file=sys.stderr,
                )

    # Load large file exclude patterns
    large_files_config = config.get("large_files", {})
    exclude_patterns = large_files_config.get("exclude_patterns", [])
    if not isinstance(exclude_patterns, list):
        if config_path and exclude_patterns is not None:
            print(
                f"Warning: {config_path} [large_files] exclude_patterns must be array "
                f"(got {type(exclude_patterns).__name__})",
                file=sys.stderr,
            )
    else:
        valid_patterns = []
        for i, p in enumerate(exclude_patterns):
            if isinstance(p, str):
                valid_patterns.append(p)
            elif config_path:
                print(
                    f"Warning: {config_path} [large_files] exclude_patterns[{i}] "
                    f"must be string (got {type(p).__name__})",
                    file=sys.stderr,
                )
        # Modify in-place to preserve references from other modules
        LARGE_FILE_EXCLUDE_PATTERNS.clear()
        LARGE_FILE_EXCLUDE_PATTERNS.extend(valid_patterns)

    # Load runtime config (#1238)
    runtime_config = config.get("runtime", {})
    skip_val = runtime_config.get("skip_orphaned_tests")
    if skip_val is not None:
        if isinstance(skip_val, bool):
            SKIP_ORPHANED_TESTS = skip_val
        elif config_path:
            print(
                f"Warning: {config_path} [runtime] skip_orphaned_tests must be bool "
                f"(got {type(skip_val).__name__})",
                file=sys.stderr,
            )

    return thresholds


# Load config and apply to THRESHOLDS at module import
_config, _config_path = _load_config()
THRESHOLDS = _apply_config(_config, _config_path)

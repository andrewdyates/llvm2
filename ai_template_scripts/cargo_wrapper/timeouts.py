# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Timeout configuration for cargo_wrapper."""

from __future__ import annotations

import tomllib

from looper.config_validation import check_unknown_keys

from . import _state
from .constants import (
    _CARGO_GLOBAL_FLAGS_WITH_VALUES,
    BUILD_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_CARGO,
    KANI_TIMEOUT,
    LOCK_KIND_BUILD,
    LOCK_KIND_TEST,
    TEST_LOCK_SUBCOMMANDS,
    TEST_TIMEOUT,
    TIMEOUT_CONFIG_PATHS,
)
from .logging import log_stderr

__all__ = [
    "get_cargo_subcommand",
    "get_lock_kind_for_command",
    "get_limits_config",
    "get_timeout_config",
    "select_cargo_timeout",
]


def get_cargo_subcommand(args: list[str]) -> str | None:
    """Best-effort extraction of cargo subcommand from args."""
    if not args:
        return None

    i = 0
    if args[0].startswith("+"):
        i += 1

    while i < len(args):
        arg = args[i]
        if arg == "--":
            i += 1
            break
        if arg.startswith("-"):
            if arg.startswith("-Z") and arg != "-Z":
                i += 1
                continue
            if "=" in arg:
                i += 1
                continue
            if arg in _CARGO_GLOBAL_FLAGS_WITH_VALUES:
                if i + 1 >= len(args):
                    return None
                i += 2
                continue
            i += 1
            continue
        return arg

    if i < len(args):
        return args[i]
    return None


def get_lock_kind_for_command(args: list[str]) -> str:
    """Return lock kind based on cargo subcommand."""
    subcommand = get_cargo_subcommand(args)
    if subcommand in TEST_LOCK_SUBCOMMANDS:
        return LOCK_KIND_TEST
    return LOCK_KIND_BUILD


# Known config keys for validation (#1525)
_KNOWN_SECTIONS = {"limits", "timeouts"}
_KNOWN_TIMEOUT_KEYS = {
    "build_sec",
    "build_timeout_sec",
    "test_sec",
    "test_timeout_sec",
    "kani_sec",
    "kani_timeout_sec",
}
_KNOWN_LIMIT_KEYS = {"max_concurrent_cargo"}


def _warn_unknown_keys(
    data: dict, timeouts: dict, limits: dict, source: str | object
) -> None:
    """Emit warnings for unknown keys in cargo_wrapper.toml config.

    Per design in designs/2026-01-31-pulse-cargo-unknown-key-warnings.md:
    - Warn on unknown top-level sections
    - Warn on unknown keys in [timeouts]
    - Warn on unknown keys in [limits]
    - Sort keys for deterministic output

    Uses shared check_unknown_keys() utility from looper/config_validation.py.

    Args:
        data: Full parsed config dict.
        timeouts: The [timeouts] table (already validated as dict).
        limits: The [limits] table (already validated as dict).
        source: Config file path (for warning messages).
    """

    def cargo_warn(msg: str) -> None:
        """Emit warning with [cargo] prefix."""
        log_stderr(f"[cargo] {msg.replace('Warning:', 'WARNING:')}")

    # Check for unknown top-level sections
    check_unknown_keys(data, _KNOWN_SECTIONS, str(source), logger=cargo_warn)

    # Check for unknown keys in [timeouts]
    check_unknown_keys(
        timeouts, _KNOWN_TIMEOUT_KEYS, f"{source} [timeouts]", logger=cargo_warn
    )

    # Check for unknown keys in [limits]
    check_unknown_keys(
        limits, _KNOWN_LIMIT_KEYS, f"{source} [limits]", logger=cargo_warn
    )


def _apply_timeout_config(
    defaults: dict[str, int],
    config: dict,
    source: str | object,
) -> dict[str, int]:
    """Apply timeout overrides from config to defaults."""
    mapping = {
        "build_sec": "build",
        "build_timeout_sec": "build",
        "test_sec": "test",
        "test_timeout_sec": "test",
        "kani_sec": "kani",
        "kani_timeout_sec": "kani",
    }
    for key, target in mapping.items():
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (int, float)) and value > 0:
            defaults[target] = int(value)
        else:
            log_stderr(
                f"[cargo] WARNING: {source} {key} must be a positive number (got {value!r})"
            )
    return defaults


def _apply_limit_config(
    defaults: dict[str, int],
    config: dict,
    source: str | object,
) -> dict[str, int]:
    """Apply limit overrides from config to defaults."""
    value = config.get("max_concurrent_cargo")
    if value is None:
        return defaults
    if isinstance(value, bool):
        log_stderr(
            "[cargo] WARNING: "
            f"{source} max_concurrent_cargo must be 1 or 2 (got {value!r})"
        )
        return defaults
    if isinstance(value, (int, float)) and float(value).is_integer():
        int_value = int(value)
        if int_value in (1, 2):
            defaults["max_concurrent_cargo"] = int_value
            return defaults
    log_stderr(
        f"[cargo] WARNING: {source} max_concurrent_cargo must be 1 or 2 (got {value!r})"
    )
    return defaults


def _load_configs() -> tuple[dict[str, int], dict[str, int]]:
    """Load timeout and limit config from cargo_wrapper.toml if present."""
    timeout_defaults = {
        "build": BUILD_TIMEOUT,
        "test": TEST_TIMEOUT,
        "kani": KANI_TIMEOUT,
    }
    limit_defaults = {"max_concurrent_cargo": DEFAULT_MAX_CONCURRENT_CARGO}
    for config_path in TIMEOUT_CONFIG_PATHS:
        if not config_path.exists():
            continue
        try:
            data = tomllib.loads(config_path.read_text())
            timeouts = data.get("timeouts", {})
            limits = data.get("limits", {})
            timeouts_table = timeouts if isinstance(timeouts, dict) else {}
            limits_table = limits if isinstance(limits, dict) else {}
            _warn_unknown_keys(data, timeouts_table, limits_table, config_path)
            if not isinstance(timeouts, dict):
                log_stderr(f"[cargo] WARNING: {config_path} [timeouts] must be a table")
            if "limits" in data and not isinstance(limits, dict):
                log_stderr(f"[cargo] WARNING: {config_path} [limits] must be a table")
            timeouts_config = _apply_timeout_config(
                timeout_defaults.copy(), timeouts_table, config_path
            )
            limits_config = _apply_limit_config(
                limit_defaults.copy(), limits_table, config_path
            )
            return timeouts_config, limits_config
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log_stderr(f"[cargo] WARNING: Failed to read {config_path}: {exc}")
            return timeout_defaults, limit_defaults
    return timeout_defaults, limit_defaults


def _ensure_configs_loaded() -> None:
    """Ensure timeout/limit configs are loaded without clobbering overrides."""
    if _state.TIMEOUT_CONFIG is not None and _state.LIMITS_CONFIG is not None:
        return
    timeouts, limits = _load_configs()
    if _state.TIMEOUT_CONFIG is None:
        _state.TIMEOUT_CONFIG = timeouts
    if _state.LIMITS_CONFIG is None:
        _state.LIMITS_CONFIG = limits


def get_timeout_config() -> dict[str, int]:
    """Return cached timeout configuration."""
    _ensure_configs_loaded()
    return _state.TIMEOUT_CONFIG


def get_limits_config() -> dict[str, int]:
    """Return cached limits configuration."""
    _ensure_configs_loaded()
    return _state.LIMITS_CONFIG


def select_cargo_timeout(
    args: list[str], timeouts: dict[str, int] | None = None
) -> int:
    """Choose timeout based on subcommand."""
    subcommand = get_cargo_subcommand(args)
    config = timeouts or get_timeout_config()
    if subcommand in ("kani", "zani"):
        return config["kani"]
    if subcommand in TEST_LOCK_SUBCOMMANDS:
        return config["test"]
    return config["build"]

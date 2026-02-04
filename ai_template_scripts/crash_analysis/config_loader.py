# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Configuration loading for crash analysis.

This module handles loading and validating crash_analysis.toml configuration files.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    tomllib = None  # Python < 3.11 fallback - config loading will use defaults

# Config file search order (first found wins)
CONFIG_PATHS = [
    Path("crash_analysis.toml"),
    Path("ai_template_scripts/crash_analysis.toml"),
    Path(".crash_analysis.toml"),
]

# Default adaptive threshold config
DEFAULT_CONFIG = {
    "baseline_window_hours": 168,  # 1 week
    "bucket_hours": 24,  # 24-hour buckets for baseline sampling
    "min_samples": 3,  # Minimum buckets required for adaptive thresholds
    "warning_threshold_rate": None,  # If set, uses fixed rate
    "critical_threshold_rate": None,  # If set, uses fixed rate
    "warning_threshold_sigma": 2.0,  # Standard deviations above mean
    "critical_threshold_sigma": 3.0,  # Standard deviations above mean
    "stddev_floor": 0.05,  # Minimum stddev to prevent overfitting
}

# Valid config keys for unknown-key warnings
VALID_CONFIG_KEYS = {"crash_analysis"}
VALID_CRASH_ANALYSIS_KEYS = set(DEFAULT_CONFIG.keys())


def _warn_unknown_keys(config: dict, config_path: str) -> None:
    """Emit warnings for unknown keys in crash_analysis.toml config."""
    for key in config:
        if key not in VALID_CONFIG_KEYS:
            print(
                f"Warning: Unknown section '{key}' in {config_path}",
                file=sys.stderr,
            )
        elif key == "crash_analysis":
            section = config[key]
            if isinstance(section, dict):
                for subkey in section:
                    if subkey not in VALID_CRASH_ANALYSIS_KEYS:
                        print(
                            f"Warning: Unknown key '{subkey}' in "
                            f"[crash_analysis] in {config_path}",
                            file=sys.stderr,
                        )


def load_config() -> tuple[dict, str | None]:
    """Load crash_analysis config from TOML file if present.

    Searches for config in: crash_analysis.toml, ai_template_scripts/crash_analysis.toml,
    .crash_analysis.toml

    Returns:
        (config_dict, config_path) - config_path is None if no config found
    """
    # Python < 3.11 doesn't have tomllib - fall back to defaults
    if tomllib is None:
        return {}, None

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
            print(f"Warning: Error reading {config_path}: {e}", file=sys.stderr)
    return {}, None


def get_config_value(config: dict, key: str):
    """Get a config value from [crash_analysis] section, with fallback to defaults."""
    section = config.get("crash_analysis", {})
    if key in section:
        return section[key]
    return DEFAULT_CONFIG.get(key)


__all__ = [
    "CONFIG_PATHS",
    "DEFAULT_CONFIG",
    "VALID_CONFIG_KEYS",
    "VALID_CRASH_ANALYSIS_KEYS",
    "load_config",
    "get_config_value",
]

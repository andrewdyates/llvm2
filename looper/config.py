# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
# Config validation: bounds/allowed-values added 2026-01-31 (#382)

"""
looper/config.py - Configuration and role config parsing
"""

__all__ = [
    "CONFIG_SCHEMA",
    "ConfigConstraints",
    "INJECTION_CAP_DEFAULTS",
    "ITERATION_FILE_TEMPLATE",
    "LOG_DIR",
    "LOG_RETENTION_HOURS",
    "MAX_CRASH_LOG_LINES",
    "MAX_LOG_FILES",
    "PID_FILE_TEMPLATE",
    "ROLES_DIR",
    "STATUS_FILE_TEMPLATE",
    "TIMEOUT_DEFAULTS",
    "ThemeConfig",
    "build_claude_autoload_context",
    "build_codex_context",
    "get_project_name",
    "get_theme_config",
    "inject_content",
    "load_injection_caps",
    "load_project_config",
    "load_role_config",
    "load_sync_config",
    "load_timeout_config",
    "parse_frontmatter",
    "parse_phase_blocks",
    "set_tab_title",
    "validate_config",
]

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, TypedDict

from looper.config_validation import get_unknown_keys, validate_bounds, validate_type
from looper.constants import (
    FLAGS_DIR,
    LOG_DIR,
    LOG_RETENTION_HOURS,
    MAX_CRASH_LOG_LINES,
    MAX_LOG_FILES,
    ROLES_DIR,
)
from looper.log import debug_swallow, log_info, log_warning
from looper.subprocess_utils import run_git_command, set_local_mode_from_config

# --- Constants ---

# File templates for per-mode state
# NOTE: {mode} is replaced with "worker_1", "worker_2" etc. for multi-worker support.
# See runner.py:162-166 where file_suffix = f"{mode}_{worker_id}" when --id=N is passed.
# Multiple workers per repo ARE supported - use spawn_session.sh --id=1 worker, --id=2 worker, etc.
ITERATION_FILE_TEMPLATE = ".iteration_{mode}"
PID_FILE_TEMPLATE = ".pid_{mode}"
STATUS_FILE_TEMPLATE = ".{mode}_status.json"

# --- Config Schema ---
# Defines expected config keys with their types and optional bounds.
# Schema format: {"type": type, "min": num, "max": num, "allowed": [values]}
# Simple types (str, bool) can be specified directly.
# Keys not in schema trigger warnings (possible typos).
# Bounds violations generate warnings (not errors) to avoid breaking existing configs.


class ConfigConstraints(TypedDict, total=False):
    """Type constraints for a config key."""

    type: type | str
    min: int | float
    max: int | float
    allowed: list[str]


# Schema with type and bounds information
CONFIG_SCHEMA: dict[str, type | str | ConfigConstraints] = {
    # Timing (seconds unless noted)
    "restart_delay": {"type": int, "min": 0, "max": 86400},  # 0 to 24 hours
    "error_delay": {"type": int, "min": 0, "max": 3600},  # 0 to 1 hour
    "iteration_timeout": {"type": int, "min": 60, "max": 7200},  # 1 min to 2 hours
    "silence_timeout": {"type": int, "min": 60, "max": 3600},  # 1 min to 1 hour
    "cleanup_closed_interval": {"type": int, "min": 1, "max": 100},
    "uncommitted_warn_threshold": {"type": int, "min": 0, "max": 1000},
    # Probabilities (0.0-1.0)
    "codex_probability": {"type": float, "min": 0.0, "max": 1.0},
    "dasher_probability": {"type": float, "min": 0.0, "max": 1.0},
    "gemini_probability": {"type": float, "min": 0.0, "max": 1.0},
    # Identity
    "git_author_name": str,
    # Rotation
    "rotation_type": {
        "type": str,
        "allowed": ["", "audit", "research", "verification", "work"],
    },
    "rotation_phases": "list[str]",
    "phase_weights": "list[str]",  # ["phase1:weight1", "phase2:weight2"] format
    "freeform_frequency": {"type": int, "min": 0, "max": 100},
    "force_phase": str,
    "starvation_hours": {"type": int, "min": 0, "max": 168},  # 0 to 1 week
    # Audit
    "auto_audit": bool,
    "audit_max_rounds": {"type": int, "min": 0, "max": 10},
    "audit_min_issues": {"type": int, "min": 0, "max": 20},
    # Priority-aware audit tuning (#2798)
    # Allows stricter audits for high-priority work and lighter audits for P3.
    "audit_max_rounds_p0": {"type": int, "min": 0, "max": 10},
    "audit_max_rounds_p1": {"type": int, "min": 0, "max": 10},
    "audit_max_rounds_p2": {"type": int, "min": 0, "max": 10},
    "audit_max_rounds_p3": {"type": int, "min": 0, "max": 10},
    "audit_min_issues_p0": {"type": int, "min": 0, "max": 20},
    "audit_min_issues_p1": {"type": int, "min": 0, "max": 20},
    "audit_min_issues_p2": {"type": int, "min": 0, "max": 20},
    "audit_min_issues_p3": {"type": int, "min": 0, "max": 20},
    "escalation_sla_days": {"type": int, "min": 1, "max": 30},
    # Models
    "claude_model": str,
    "codex_model": str,
    "codex_models": "list[str]",
    "dasher_model": str,
    # Status/monitoring
    "pulse_interval_minutes": {"type": int, "min": 1, "max": 60},
    "scrub_logs": bool,
    # Internal (set by load_role_config, not user-provided)
    "phase_data": dict,
    # Multi-machine sync settings
    "sync_strategy": {"type": str, "allowed": ["rebase", "merge"]},
    "sync_on_startup": bool,
    "sync_interval_iterations": {"type": int, "min": 0, "max": 100},  # 0=disabled
    "staged_check_abort": bool,
    "auto_pr": bool,
    # Checkpoint settings - core implementation in looper/checkpoint.py
    # NOTE: Tool-level checkpoint settings (checkpoint_on_tool_*) are RESERVED
    # for Phase 2 (#1625). See designs/2026-02-01-tool-call-checkpointing.md.
    # Do NOT use these keys until Phase 2 is implemented.
    "checkpoint_enabled": bool,
    "checkpoint_recovery_max_tokens": {"type": int, "min": 100, "max": 10000},
    "checkpoint_tool_output_truncate": {"type": int, "min": 100, "max": 50000},
    # RESERVED: Phase 2 - tool-level checkpointing (#1625)
    # "checkpoint_on_tool_complete": bool,
    # "checkpoint_on_tool_start": bool,
    # Memory watchdog settings (#1468)
    "memory_watchdog_enabled": bool,
    "memory_watchdog_threshold": {"type": str, "allowed": ["warn", "critical"]},
    # Token growth guardrails (#1881)
    # Warn when input_tokens exceeds threshold; abort if exceeds critical.
    # When abort threshold exceeded, session resume is skipped to prevent
    # unbounded context accumulation in audit rounds.
    "token_warn_threshold": {"type": int, "min": 100000, "max": 10000000},
    "token_abort_threshold": {"type": int, "min": 500000, "max": 50000000},
    # Local mode settings (#1592)
    "local_mode": bool,
    # Timeouts (configurable via .looper_config.json "timeouts")
    "timeouts": dict,
    # Per-role model routing (#1888)
    # Enables explicit model selection per role without frontmatter changes.
    # See designs/2026-02-02-per-role-model-routing.md for full specification.
    "model_routing": dict,
    # Model switching config for resumed sessions (#1888)
    # Controls what happens when model changes during a resumed session.
    "model_switching": dict,
    # Prompt budget (#2695) - total character budget for looper injections
    "prompt_budget_chars": {"type": int, "min": 5000, "max": 50000},
    # Per-injection caps (#2733 Phase 2, #2745) - override default caps per injection
    # Keys: "active_issue", "last_directive", "other_feedback"
    "injection_caps": dict,
    # AI Themes (#2478) - configurable focus via filtering and prompt injection
    # See designs/2026-02-05-ai-themes.md for full specification.
    # Per-instance or per-role themes filter issues by label/search.
    "theme": str,  # Theme name (e.g., "cleanup", "security")
    "theme_description": str,  # Human-readable description for prompt injection
    "issue_filter": dict,  # Filter config: {labels: [], exclude_labels: [], search: ""}
    # Team theme (applies to all roles on this machine)
    "team_theme": dict,  # {name: str, description: str, issue_filter: dict}
    # Audit-overhead circuit breaker (#2808)
    # See designs/2026-02-06-audit-overhead-circuit-breaker.md
    "audit_overhead_circuit": dict,
}

TIMEOUT_DEFAULTS: dict[str, int] = {
    "git_default": 5,
    "gh_list": 15,
    "gh_view": 10,
    "health_check": 120,  # Matches script's per-check timeout (#2000)
    "max_silence": 3600,
}


def _normalize_timeouts(raw: object) -> dict[str, int]:
    """Normalize timeout config, applying defaults and basic validation."""
    timeouts = TIMEOUT_DEFAULTS.copy()
    if not isinstance(raw, dict):
        return timeouts
    for key, value in raw.items():
        if key not in timeouts:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            normalized = int(value)
            if normalized <= 0:
                continue
            timeouts[key] = normalized
    return timeouts


def load_timeout_config(project_config: dict[str, Any] | None = None) -> dict[str, int]:
    """Load timeout overrides from .looper_config.json (top-level "timeouts")."""
    if project_config is None:
        project_config = load_project_config()
    raw = project_config.get("timeouts")
    return _normalize_timeouts(raw)


INJECTION_CAP_DEFAULTS: dict[str, int] = {
    "active_issue": 2000,
    "last_directive": 2000,
    "other_feedback": 3000,
}


def load_injection_caps(project_config: dict[str, Any] | None = None) -> dict[str, int]:
    """Load per-injection character caps from .looper_config.json.

    Reads the "injection_caps" key and merges with defaults. Only known
    keys are accepted; unknown keys are silently ignored.

    Returns:
        Dict mapping injection name to character cap.
    """
    caps = INJECTION_CAP_DEFAULTS.copy()
    if project_config is None:
        project_config = load_project_config()
    raw = project_config.get("injection_caps")
    if not isinstance(raw, dict):
        return caps
    for key, value in raw.items():
        if key not in caps:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            normalized = int(value)
            if normalized <= 0:
                continue
            caps[key] = normalized
    return caps


def _get_schema_type(schema_entry: type | str | ConfigConstraints) -> type | str:
    """Extract the type from a schema entry.

    Handles both simple types (int, str) and ConfigConstraints dicts.
    """
    if isinstance(schema_entry, dict):
        return schema_entry.get("type", str)
    return schema_entry


def validate_config(
    config: dict[str, Any], role: str, strict: bool = False
) -> list[str]:
    """Validate merged config against schema.

    Contracts:
        REQUIRES: config is a dict (type-hinted)
        REQUIRES: role is a string (role name for error context)
        ENSURES: Returns list of warning/error message strings
        ENSURES: Unknown keys generate "Warning" messages
        ENSURES: Type mismatches generate "Error" messages
        ENSURES: Bounds violations generate "Warning" messages
        ENSURES: Invalid allowed values generate "Warning" messages
        ENSURES: If strict=True and errors exist, raises ValueError
        ENSURES: If strict=False, never raises (returns messages only)

    Uses shared validation utilities from looper/config_validation.py.

    Args:
        config: Merged configuration dict
        role: Role name for error messages ('worker', 'manager', etc.)
        strict: If True, raise ValueError on errors; otherwise warn

    Returns:
        List of validation messages (warnings/errors)
    """
    messages: list[str] = []

    # Check for unknown keys using shared utility
    unknown = get_unknown_keys(config, set(CONFIG_SCHEMA.keys()), role)
    for key in unknown:
        messages.append(
            f"Warning [{role}]: Unknown config key '{key}' (possible typo). "
            "See docs/troubleshooting.md#config-issues"
        )

    # Validate known keys
    for key, value in config.items():
        if key not in CONFIG_SCHEMA:
            continue  # Already handled above

        schema_entry = CONFIG_SCHEMA[key]
        expected_type = _get_schema_type(schema_entry)

        # Type validation using shared utility
        valid, error_msg = validate_type(value, expected_type, key, role)
        if not valid:
            # validate_type returns "Error [context]: ..." format already
            messages.append(error_msg)
            continue  # Skip bounds check if type is wrong

        # Bounds and allowed value checks (only for ConfigConstraints dicts)
        if isinstance(schema_entry, dict):
            bounds_warnings = validate_bounds(
                value,
                key,
                role,
                min_val=schema_entry.get("min"),
                max_val=schema_entry.get("max"),
                allowed=schema_entry.get("allowed"),
            )
            messages.extend(bounds_warnings)

    if strict:
        errors = [m for m in messages if m.startswith("Error")]
        if errors:
            raise ValueError("\n".join(errors))

    return messages


def _write_startup_warnings(role: str, warnings: list[str]) -> None:
    """Write startup warnings to .flags/ for Manager visibility.

    Called during load_role_config() when configuration warnings occur.
    Manager sees these via the audit context (audit_context.py reads .flags/).

    Part of #2452: Surface startup warnings to Manager.

    Args:
        role: Role that encountered the warning (worker, prover, etc.)
        warnings: List of warning messages to record.
    """
    from datetime import datetime

    try:
        FLAGS_DIR.mkdir(exist_ok=True)
        flag_path = FLAGS_DIR / "startup_warnings"

        # Append to existing warnings if file exists (multiple roles may have warnings)
        existing = ""
        if flag_path.exists():
            try:
                existing = flag_path.read_text()
            except OSError:
                pass

        # Add timestamp and new warnings
        timestamp = datetime.now().isoformat()
        new_content = f"\n[{timestamp}] {role}:\n" + "\n".join(f"  {w}" for w in warnings)

        content = existing + new_content + "\n"
        flag_path.write_text(content)
    except OSError as e:
        # Don't fail config loading if flag writing fails
        debug_swallow("_write_startup_warnings", e)


# --- Frontmatter Parsing ---


def _is_float(value: str) -> bool:
    """Check if a string represents a float value.

    Handles simple decimals (0.5, -0.5) and scientific notation (1e-5, 1.5e10).
    Returns False for integers to preserve int type.
    """
    if not value:
        return False
    # Must contain decimal point or 'e' to be a float
    if "." not in value and "e" not in value.lower():
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-like frontmatter from role files.

    Contracts:
        REQUIRES: content is a string
        ENSURES: Returns (config_dict, remaining_content) tuple
        ENSURES: If no frontmatter (no leading ---), returns ({}, content)
        ENSURES: If malformed frontmatter (no closing ---), returns ({}, content)
        ENSURES: Values are auto-typed: bool, int, float, list[str], or str
        ENSURES: Never raises (graceful fallback to empty config)

    Format:
        ---
        key: value
        list_key: item1,item2,item3
        ---
        Rest of content...

    Returns:
        (config_dict, remaining_content)
    """
    config: dict[str, Any] = {}
    lines = content.split("\n")

    if not lines or lines[0].strip() != "---":
        return config, content

    # Find closing ---
    end_idx = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return config, content

    # Parse frontmatter
    for line in lines[1:end_idx]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Type conversion
            if value.lower() == "true":
                config[key] = True
            elif value.lower() == "false":
                config[key] = False
            elif "," in value:
                # Comma-separated list
                config[key] = [v.strip() for v in value.split(",")]
            elif _is_float(value):
                config[key] = float(value)
            else:
                # Try integer conversion (handles +42, -42, 42)
                try:
                    config[key] = int(value)
                except ValueError:
                    config[key] = value

    remaining = "\n".join(lines[end_idx + 1 :])
    return config, remaining


def parse_phase_blocks(content: str) -> dict[str, dict[str, Any]]:
    """Parse phase-specific content blocks from role files.

    Contracts:
        REQUIRES: content is a string
        ENSURES: Returns dict mapping phase names to config
        ENSURES: Each phase config has "content" key with stripped text
        ENSURES: Never raises (regex is fail-safe)

    Format (markdown headers):
        ### Phase: phase_name
        Instructions for this phase...

        ### Phase: another_phase
        Instructions for this phase...

    Content ends at next ### Phase: header or ## header.

    Weights are configured separately in frontmatter (phase_weights key)
    or .looper_config.json, not inline with content.

    Returns:
        Dict mapping phase names to their config:
        {
            "phase_name": {
                "content": "Instructions for this phase...",
            }
        }
    """
    phases: dict[str, dict[str, Any]] = {}

    # Pattern to find phase headers: ### Phase: name
    header_pattern = r"^###\s*Phase:\s*(\w+)\s*$"

    lines = content.split("\n")
    current_phase: str | None = None
    current_lines: list[str] = []

    def save_current_phase() -> None:
        """Save accumulated content for current phase."""
        nonlocal current_phase, current_lines
        if current_phase:
            phase_content = "\n".join(current_lines).strip()
            if phase_content:
                phases[current_phase] = {"content": phase_content}
        current_phase = None
        current_lines = []

    for line in lines:
        # Check for phase header
        match = re.match(header_pattern, line, re.IGNORECASE)
        if match:
            save_current_phase()
            current_phase = match.group(1)
            current_lines = []
            continue

        # Check for section end (## header or another ### non-phase header)
        if current_phase:
            if line.startswith("## ") or (
                line.startswith("### ")
                and not re.match(header_pattern, line, re.IGNORECASE)
            ):
                save_current_phase()
                continue
            current_lines.append(line)

    # Save final phase
    save_current_phase()

    return phases


# --- Project Config ---


def load_project_config() -> dict[str, Any]:
    """Load project-specific config overrides from .looper_config.json.

    This file is NOT synced by template - projects can tune intervals locally.

    Returns:
        Dict with per-role overrides, e.g.:
        {"prover": {"restart_delay": 300}, "worker": {"silence_timeout": 600}}
    """
    config_file = Path(".looper_config.json")
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text())
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            log_warning(f"Warning: Could not parse .looper_config.json: {e}")
    return {}


# --- Theme Config (#2478) ---


class ThemeConfig(TypedDict, total=False):
    """Configuration for an AI theme.

    Themes allow focusing AI work on specific issue subsets without new labels.
    See designs/2026-02-05-ai-themes.md for full specification.

    Fields:
        name: Theme name (e.g., "cleanup", "security")
        description: Human-readable description for prompt injection
        issue_filter: Filter config with labels, exclude_labels, search
    """

    name: str
    description: str
    issue_filter: dict[str, Any]


def get_theme_config(
    role: str, worker_id: int | None = None
) -> ThemeConfig | None:
    """Get theme configuration for the current role/instance.

    Contracts:
        REQUIRES: role is a string (worker, manager, prover, researcher)
        REQUIRES: worker_id is None or int >= 1
        ENSURES: Returns ThemeConfig if theme is configured, None otherwise
        ENSURES: Precedence: per-instance > per-role > team_theme
        ENSURES: Never raises - returns None on error

    Theme config is loaded from .looper_config.json with precedence:
    1. Per-instance theme (e.g., "worker_1.theme")
    2. Per-role theme (e.g., "worker.theme")
    3. Team theme ("team_theme")

    Example .looper_config.json:
        {
            "worker_3": {
                "theme": "cleanup",
                "theme_description": "P3 tech debt only",
                "issue_filter": {"labels": ["P3"]}
            },
            "team_theme": {
                "name": "security",
                "description": "Focus on security issues",
                "issue_filter": {"search": "security OR auth"}
            }
        }

    Args:
        role: Role name (worker, manager, prover, researcher)
        worker_id: Optional instance ID (1, 2, 3, etc.)

    Returns:
        ThemeConfig dict if theme configured, None otherwise.
    """
    project_config = load_project_config()

    # Try per-instance config first (e.g., "worker_1")
    if worker_id is not None:
        instance_key = f"{role}_{worker_id}"
        instance_config = project_config.get(instance_key, {})
        if isinstance(instance_config, dict) and instance_config.get("theme"):
            return ThemeConfig(
                name=str(instance_config.get("theme", "")),
                description=str(instance_config.get("theme_description", "")),
                issue_filter=instance_config.get("issue_filter", {}),
            )

    # Try per-role config (e.g., "worker")
    role_config = project_config.get(role, {})
    if isinstance(role_config, dict) and role_config.get("theme"):
        return ThemeConfig(
            name=str(role_config.get("theme", "")),
            description=str(role_config.get("theme_description", "")),
            issue_filter=role_config.get("issue_filter", {}),
        )

    # Try team theme (applies to all roles)
    team_theme = project_config.get("team_theme", {})
    if isinstance(team_theme, dict) and team_theme.get("name"):
        return ThemeConfig(
            name=str(team_theme.get("name", "")),
            description=str(team_theme.get("description", "")),
            issue_filter=team_theme.get("issue_filter", {}),
        )

    return None


# --- Role Config ---


def load_role_config(
    role: str, worker_id: int | None = None
) -> tuple[dict[str, Any], str]:
    """Load role configuration from .claude/roles/ markdown files.

    Contracts:
        REQUIRES: role in {'worker', 'manager', 'researcher', 'prover'}
        REQUIRES: worker_id is None or int in range 1-5
        ENSURES: Returns (merged_config_dict, prompt_template) tuple
        ENSURES: Config merges shared < role < project < instance (higher wins)
        ENSURES: Prompt template contains <!-- INJECT:... --> markers
        ENSURES: phase_data added to config if PHASE blocks present
        RAISES: FileNotFoundError if .claude/roles/{role}.md or shared.md missing

    Args:
        role: Role name ('worker', 'manager', 'researcher', 'prover')
        worker_id: Optional instance ID (1-5) for per-worker config (#1175)

    Returns:
        (config_dict, prompt_template) with <!-- INJECT:... --> markers.

    Config precedence (highest to lowest):
        1. .looper_config.json per-instance overrides (e.g., "worker_1")
        2. .looper_config.json role overrides (e.g., "worker")
        3. .claude/roles/{role}.md frontmatter (role-specific)
        4. .claude/roles/shared.md frontmatter (shared defaults)

    Example .looper_config.json for per-worker config:
        {
            "worker": {"restart_delay": 60},
            "worker_1": {"force_phase": "high_priority"},
            "worker_3": {"force_phase": "quality"}
        }

    Raises:
        FileNotFoundError: If required role files are missing.
    """
    shared_file = ROLES_DIR / "shared.md"
    role_file = ROLES_DIR / f"{role}.md"

    if not shared_file.exists():
        raise FileNotFoundError(f"Missing required file: {shared_file}")
    if not role_file.exists():
        raise FileNotFoundError(f"Missing required file: {role_file}")

    shared_config, shared_body = parse_frontmatter(shared_file.read_text())
    role_content = role_file.read_text()
    role_config, role_body = parse_frontmatter(role_content)

    # Parse phase blocks for rotation
    phase_data = parse_phase_blocks(role_content)

    # Load project overrides
    project_config = load_project_config()
    project_role_config = project_config.get(role, {})
    timeout_config = load_timeout_config(project_config)

    # Load per-instance overrides (e.g., "worker_1") if worker_id provided (#1175)
    project_instance_config: dict[str, Any] = {}
    if worker_id is not None:
        instance_key = f"{role}_{worker_id}"
        project_instance_config = project_config.get(instance_key, {})

    # Merge: shared < role < project < instance
    merged_config = {
        **shared_config,
        **role_config,
        **project_role_config,
        **project_instance_config,
    }
    merged_config["timeouts"] = timeout_config

    # Coerce list[str] config values from string when single value (no comma)
    for key in ("rotation_phases", "codex_models"):
        raw_value = merged_config.get(key)
        if isinstance(raw_value, str):
            if raw_value:
                merged_config[key] = [raw_value]
                log_warning(
                    f"Warning [{role}]: {key} '{raw_value}' was string, converted to list"
                )
            else:
                merged_config[key] = []

    # Add phase data to config
    if phase_data:
        merged_config["phase_data"] = phase_data

    # Merge phase_weights from frontmatter into phase_data
    # Format: "phase1:weight1,phase2:weight2" or list ["phase1:weight1", "phase2:weight2"]
    phase_weights_raw = merged_config.get("phase_weights")
    if phase_weights_raw and phase_data:
        weights_list: list[str] = []
        if isinstance(phase_weights_raw, str):
            weights_list = [w.strip() for w in phase_weights_raw.split(",")]
        elif isinstance(phase_weights_raw, list):
            weights_list = phase_weights_raw

        for item in weights_list:
            if ":" in item:
                phase_name, weight_str = item.split(":", 1)
                phase_name = phase_name.strip()
                if phase_name in phase_data:
                    try:
                        phase_data[phase_name]["weight"] = int(weight_str.strip())
                    except ValueError:
                        pass  # Invalid weight, skip

    # Validate merged config against schema
    validation_messages = validate_config(merged_config, role)
    for msg in validation_messages:
        log_info(msg)

    # Validate rotation_phases vs PHASE blocks
    rotation_phases = merged_config.get("rotation_phases", [])
    startup_warnings: list[str] = []
    if rotation_phases:
        config_phases = (
            set(rotation_phases) if isinstance(rotation_phases, list) else set()
        )
        block_phases = set(phase_data.keys()) if phase_data else set()

        missing_blocks = config_phases - block_phases
        extra_blocks = block_phases - config_phases

        if missing_blocks:
            # Actionable warning message (#2452 item 3)
            missing_str = ", ".join(sorted(missing_blocks))
            msg = (
                f"Warning [{role}]: rotation_phases references undefined phases: {missing_str}. "
                f"Each phase listed in 'rotation_phases' frontmatter needs a matching "
                f"'### Phase: <name>' block in .claude/roles/{role}.md"
            )
            log_warning(msg)
            startup_warnings.append(msg)
        if extra_blocks:
            extra_str = ", ".join(sorted(extra_blocks))
            msg = (
                f"Warning [{role}]: phase blocks not in rotation_phases: {extra_str}. "
                f"Either add these phases to 'rotation_phases' frontmatter or remove "
                f"the '### Phase: <name>' blocks from .claude/roles/{role}.md"
            )
            log_warning(msg)
            startup_warnings.append(msg)

    # Write startup warnings to .flags/ for Manager visibility (#2452 item 2)
    if startup_warnings:
        _write_startup_warnings(role, startup_warnings)

    set_local_mode_from_config(merged_config.get("local_mode"))

    prompt = f"{shared_body.strip()}\n\n{role_body.strip()}"

    return merged_config, prompt


def get_project_name() -> str:
    """Get the current project name from git remote or directory.

    Returns the repo name portion of the git remote URL, or falls back
    to the current directory name.
    """
    timeout_sec = load_timeout_config().get("git_default", 5)
    result = run_git_command(["remote", "get-url", "origin"], timeout=timeout_sec)
    if result.ok and result.value:
        url = result.value.strip()
        # Handle both SSH and HTTPS URLs
        # git@github.com:user/repo.git -> repo
        # https://github.com/user/repo.git -> repo
        return url.rstrip("/").split("/")[-1].removesuffix(".git")

    # Fallback to directory name
    return Path.cwd().name


def set_tab_title(
    role: str, project: str | None = None, worker_id: int | None = None
) -> bool:
    """Set the terminal tab title.

    REQUIRES: role in ('W', 'P', 'R', 'M', 'U')
    REQUIRES: worker_id is None or worker_id >= 1
    ENSURES: Returns True if at least one method succeeded
    ENSURES: Returns False only if all methods failed (no TTY, no iTerm2 session)

    Uses escape sequences to set the tab/window title. This is called at
    iteration start to ensure the title stays correct even for AI tools
    (like codex) that don't have MCP plugin support.

    Args:
        role: Role letter (W, P, R, M, U)
        project: Project name, defaults to get_project_name()
        worker_id: Worker ID for multi-worker mode (W1, W2, etc.)

    Returns:
        True if title was set via any method, False if all methods failed.
    """
    if project is None:
        project = get_project_name()

    # Build role display: W or W1 for multi-worker
    role_display = f"{role}{worker_id}" if worker_id is not None else role
    title = f"[{role_display}]{project}"

    success = False

    # Try escape sequences first (fast, works in most terminals)
    try:
        with open("/dev/tty", "w") as tty:
            # OSC 0: Set icon name and window title
            tty.write(f"\033]0;{title}\007")
            # OSC 1: Set icon name (tab title in iTerm2) - more persistent
            tty.write(f"\033]1;{title}\007")
            tty.flush()
        success = True
    except OSError as e:
        debug_swallow("set_tab_title_tty", e)

    # Also try AppleScript for iTerm2 (more reliable for "sticking")
    # Do this even if escape sequences succeeded, for better persistence
    session_id = os.environ.get("ITERM_SESSION_ID", "")
    if session_id:
        escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
        guid = session_id.split(":")[-1] if ":" in session_id else session_id
        script = f'''tell application "iTerm2"
    repeat with w in windows
        repeat with t in tabs of w
            repeat with s in sessions of t
                try
                    if unique ID of s contains "{guid}" then
                        set name of s to "{escaped_title}"
                        return true
                    end if
                end try
            end repeat
        end repeat
    end repeat
end tell'''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                success = True
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            debug_swallow("set_tab_title_osascript", e)

    return success


# --- Prompt Utilities ---


def inject_content(template: str, replacements: dict[str, str]) -> str:
    """Replace injection markers with actual content.

    Markers have format: <!-- INJECT:key -->

    Args:
        template: Template string with markers
        replacements: Dict mapping marker keys to replacement content

    Returns:
        Template with markers replaced.

    Note:
        Warns on stderr if any markers remain unreplaced (likely typos).
    """
    result = template
    for key, value in replacements.items():
        marker = f"<!-- INJECT:{key} -->"
        result = result.replace(marker, value)

    # Warn about unreplaced markers (likely typos in template) (#1995)
    remaining = re.findall(r"<!-- INJECT:(\w+) -->", result)
    for key in remaining:
        log_warning(f"Unknown injection key in template: {key}")
    return result


def load_sync_config() -> dict[str, Any]:
    """Load sync configuration from .looper_config.json.

    Returns the 'sync' section of the config, which maps to SyncConfig.
    Returns empty dict if not configured (use defaults).

    Example .looper_config.json:
        {
            "machine": "sat",
            "branch": "zone/sat",
            "sync": {
                "strategy": "rebase",
                "trigger": "iteration_start",
                "auto_stash": true,
                "conflict_action": "abort"
            }
        }
    """
    project_config = load_project_config()
    return project_config.get("sync", {})


def build_claude_autoload_context() -> str:
    """Build the content Claude auto-loads (CLAUDE.md + rules).

    Use this for display purposes - shows what Claude sees beyond the -p prompt.
    Claude CLI reads CLAUDE.md and .claude/rules/*.md automatically before
    receiving the looper prompt.

    Returns:
        String with [AUTO-LOADED BY CLAUDE] markers for each file.
    """
    parts: list[str] = []

    # Include CLAUDE.md
    claude_md = Path("CLAUDE.md")
    if claude_md.exists():
        try:
            content = claude_md.read_text().strip()
            if content:
                parts.append(
                    f"### [AUTO-LOADED BY CLAUDE] CLAUDE.md ###\n\n{content}"
                )
        except (OSError, UnicodeDecodeError) as e:
            log_warning(f"Warning: Could not read {claude_md}: {e}")

    # Include .claude/rules/*.md files
    rules_dir = Path(".claude/rules")
    if rules_dir.exists():
        for rules_file in sorted(rules_dir.glob("*.md")):
            try:
                content = rules_file.read_text().strip()
                if content:
                    parts.append(
                        f"### [AUTO-LOADED BY CLAUDE] {rules_file.name} ###\n\n{content}"
                    )
            except (OSError, UnicodeDecodeError) as e:
                log_warning(f"Warning: Could not read {rules_file}: {e}")

    return "\n\n".join(parts) if parts else ""


def build_codex_context() -> str:
    """Build Codex context by reading CLAUDE.md + rules + codex.md.

    This is the primary Codex context builder. The looper prepends this output
    to the Codex prompt so that Codex sees the same instructions as Claude.
    AGENTS.md is now a minimal stub; all rules delivery happens here.
    Gracefully handles file read errors to prevent crashes.

    NOTE: .claude/roles/*.md are NOT included here because they are already
    part of the prompt (via load_role_config() -> shared.md + role.md).
    The previous implementation (#2230 fix in 10c96d7a) incorrectly added
    ALL role files, causing:
    1. Duplicate content (roles appear twice)
    2. Role confusion (worker sees manager/prover/researcher instructions)
    This was identified as root cause of #2438 headless violations.
    """
    parts: list[str] = []

    # Include CLAUDE.md (project-specific config)
    claude_md = Path("CLAUDE.md")
    if claude_md.exists():
        try:
            content = claude_md.read_text().strip()
            if content:
                parts.append(f"# Project Instructions (CLAUDE.md)\n\n{content}")
        except (OSError, UnicodeDecodeError) as e:
            log_warning(f"Warning: Could not read {claude_md}: {e}")

    # Include .claude/rules/*.md files
    rules_dir = Path(".claude/rules")
    if rules_dir.exists():
        for rules_file in sorted(rules_dir.glob("*.md")):
            try:
                content = rules_file.read_text().strip()
                if content:
                    parts.append(content)
            except (OSError, UnicodeDecodeError) as e:
                # Log but continue - don't crash if one file is unreadable
                log_warning(f"Warning: Could not read {rules_file}: {e}")

    # Include .claude/codex.md (Codex-only overrides, if present)
    codex_md = Path(".claude/codex.md")
    if codex_md.exists():
        try:
            content = codex_md.read_text().strip()
            if content:
                parts.append(f"# Codex-Specific Instructions\n\n{content}")
        except (OSError, UnicodeDecodeError) as e:
            log_warning(f"Warning: Could not read {codex_md}: {e}")

    # NOTE: .claude/roles/*.md are NOT loaded here - they come from the prompt
    # (shared.md + role.md via load_role_config()). Loading them here would
    # duplicate content and include ALL roles, confusing the model.

    return "\n\n".join(parts) + "\n\n" if parts else ""

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
    "ITERATION_FILE_TEMPLATE",
    "LOG_DIR",
    "LOG_RETENTION_HOURS",
    "MAX_CRASH_LOG_LINES",
    "MAX_LOG_FILES",
    "PID_FILE_TEMPLATE",
    "ROLES_DIR",
    "STATUS_FILE_TEMPLATE",
    "TIMEOUT_DEFAULTS",
    "build_codex_context",
    "get_project_name",
    "inject_content",
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

from looper.config_validation import check_unknown_keys, validate_bounds, validate_type
from looper.constants import (
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
    "freeform_frequency": {"type": int, "min": 0, "max": 100},
    "force_phase": str,
    "starvation_hours": {"type": int, "min": 0, "max": 168},  # 0 to 1 week
    # Audit
    "auto_audit": bool,
    "audit_max_rounds": {"type": int, "min": 0, "max": 10},
    "audit_min_issues": {"type": int, "min": 0, "max": 20},
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
    unknown = check_unknown_keys(config, set(CONFIG_SCHEMA.keys()), role)
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
        ENSURES: If weight:N attribute present, includes "weight" as int
        ENSURES: Unmatched/malformed blocks are silently skipped
        ENSURES: Never raises (regex is fail-safe)

    Format:
        <!-- PHASE:phase_name [weight:N] -->
        Phase-specific content here...
        <!-- /PHASE:phase_name -->

    Returns:
        Dict mapping phase names to their config:
        {
            "phase_name": {
                "content": "Phase-specific content...",
                "weight": 3,  # optional, default 1
            }
        }
    """
    phases: dict[str, dict[str, Any]] = {}

    # Pattern to match phase blocks with optional weight
    # <!-- PHASE:name [weight:N] -->content<!-- /PHASE:name -->
    pattern = r"<!--\s*PHASE:(\w+)((?:\s+\w+:[^>]+)*)\s*-->(.*?)<!--\s*/PHASE:\1\s*-->"

    for match in re.finditer(pattern, content, re.DOTALL):
        phase_name = match.group(1)
        attrs_str = match.group(2).strip()
        phase_content = match.group(3).strip()

        phase_config: dict[str, Any] = {"content": phase_content}

        # Parse weight attribute
        if attrs_str:
            for attr in attrs_str.split():
                if ":" in attr:
                    key, value = attr.split(":", 1)
                    if key.strip() == "weight":
                        try:
                            phase_config["weight"] = int(value.strip())
                        except ValueError as e:
                            debug_swallow(
                                "parse_phase_weight", e
                            )  # Invalid weight, skip

        phases[phase_name] = phase_config

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

    # Validate merged config against schema
    validation_messages = validate_config(merged_config, role)
    for msg in validation_messages:
        log_info(msg)

    # Validate rotation_phases vs PHASE blocks
    rotation_phases = merged_config.get("rotation_phases", [])
    if rotation_phases:
        config_phases = (
            set(rotation_phases) if isinstance(rotation_phases, list) else set()
        )
        block_phases = set(phase_data.keys()) if phase_data else set()

        missing_blocks = config_phases - block_phases
        extra_blocks = block_phases - config_phases

        if missing_blocks:
            log_warning(f"Warning [{role}]: phases without blocks: {missing_blocks}")
        if extra_blocks:
            log_warning(f"Warning [{role}]: blocks not in phases: {extra_blocks}")

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


def build_codex_context() -> str:
    """Build context that Claude reads automatically but Codex doesn't.

    Claude CLI reads CLAUDE.md, .claude/rules/*.md, and .claude/roles/*.md
    automatically. Codex only reads AGENTS.md. This function builds the
    equivalent context. Gracefully handles file read errors to prevent crashes.
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

    # Include .claude/roles/*.md files
    # Critical: Claude CLI reads these automatically but Codex doesn't.
    # These contain the "YOU ARE HEADLESS" warning that prevents violations.
    roles_dir = Path(".claude/roles")
    if roles_dir.exists():
        for roles_file in sorted(roles_dir.glob("*.md")):
            try:
                content = roles_file.read_text().strip()
                if content:
                    parts.append(content)
            except (OSError, UnicodeDecodeError) as e:
                log_warning(f"Warning: Could not read {roles_file}: {e}")

    return "\n\n".join(parts) + "\n\n" if parts else ""

"""
looper/config.py - Configuration and role config parsing

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

# --- Constants ---

LOG_DIR = Path("worker_logs")
ROLES_DIR = Path(".claude/roles")

# File templates for per-mode state
ITERATION_FILE_TEMPLATE = ".iteration_{mode}"
PID_FILE_TEMPLATE = ".pid_{mode}"
STATUS_FILE_TEMPLATE = ".{mode}_status.json"
MAX_LOG_FILES = 50
MAX_CRASH_LOG_LINES = 500


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


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from role files.

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
            elif value.lstrip("-").isdigit():
                config[key] = int(value)
            elif _is_float(value):
                config[key] = float(value)
            else:
                config[key] = value

    remaining = "\n".join(lines[end_idx + 1 :])
    return config, remaining


def parse_phase_blocks(content: str) -> dict[str, dict]:
    """Parse phase-specific content blocks from role files.

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
    phases: dict[str, dict] = {}

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
                        except ValueError:
                            pass  # Invalid weight, skip

        phases[phase_name] = phase_config

    return phases


# --- Project Config ---


def load_project_config() -> dict:
    """Load project-specific config overrides from .looper_config.json.

    This file is NOT synced by template - projects can tune intervals locally.

    Returns:
        Dict with per-role overrides, e.g.:
        {"prover": {"restart_delay": 300}, "worker": {"silence_timeout": 600}}
    """
    config_file = Path(".looper_config.json")
    if config_file.exists():
        try:
            return json.loads(config_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not parse .looper_config.json: {e}")
    return {}


# --- Role Config ---


def load_role_config(mode: str) -> tuple[dict, str]:
    """Load role configuration from .claude/roles/ markdown files.

    Args:
        mode: Role name ('worker', 'manager', 'researcher', 'prover')

    Returns:
        (config_dict, prompt_template) with <!-- INJECT:... --> markers.

    Config precedence (highest to lowest):
        1. .looper_config.json (project-specific overrides)
        2. .claude/roles/{mode}.md frontmatter (role-specific)
        3. .claude/roles/shared.md frontmatter (shared defaults)

    Raises:
        FileNotFoundError: If required role files are missing.
    """
    shared_file = ROLES_DIR / "shared.md"
    role_file = ROLES_DIR / f"{mode}.md"

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
    project_role_config = project_config.get(mode, {})

    # Merge: shared < role < project
    merged_config = {**shared_config, **role_config, **project_role_config}

    # Add phase data to config
    if phase_data:
        merged_config["phase_data"] = phase_data

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
            print(
                f"Warning [{mode}]: rotation_phases lists phases without PHASE blocks: {missing_blocks}"
            )
        if extra_blocks:
            print(
                f"Warning [{mode}]: PHASE blocks exist but not in rotation_phases: {extra_blocks}"
            )

    prompt = f"{shared_body.strip()}\n\n{role_body.strip()}"

    return merged_config, prompt


def get_project_name() -> str:
    """Get the current project name from git remote or directory.

    Returns the repo name portion of the git remote URL, or falls back
    to the current directory name.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Handle both SSH and HTTPS URLs
            # git@github.com:user/repo.git -> repo
            # https://github.com/user/repo.git -> repo
            return url.rstrip("/").split("/")[-1].removesuffix(".git")
    except Exception:
        pass

    # Fallback to directory name
    return Path.cwd().name


# --- Prompt Utilities ---


def inject_content(template: str, replacements: dict[str, str]) -> str:
    """Replace injection markers with actual content.

    Markers have format: <!-- INJECT:key -->

    Args:
        template: Template string with markers
        replacements: Dict mapping marker keys to replacement content

    Returns:
        Template with markers replaced.
    """
    result = template
    for key, value in replacements.items():
        marker = f"<!-- INJECT:{key} -->"
        result = result.replace(marker, value)
    return result


def build_codex_context() -> str:
    """Build context that Claude reads automatically but Codex doesn't.

    Claude CLI reads CLAUDE.md and .claude/rules/*.md automatically.
    Codex only reads AGENTS.md. This function builds the equivalent context.
    Gracefully handles file read errors to prevent crashes.
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
            print(f"Warning: Could not read {claude_md}: {e}")

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
                print(f"Warning: Could not read {rules_file}: {e}")

    return "\n\n".join(parts) + "\n\n" if parts else ""

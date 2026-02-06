# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_prompt.py - Prompt debugging utility.

Provides show_prompt() for testing and inspection:
- Renders the full prompt that would be sent to an AI session
- Applies role configuration, frontmatter, and content injection
- Shows rotation focus and context without starting a session

Usage:
    python -m looper.runner_prompt worker  # Show worker prompt
    python -m looper.runner_prompt manager  # Show manager prompt

Useful for debugging prompt construction and context injection.
Not used during normal looper operation - purely a development tool.
"""

from __future__ import annotations

__all__ = ["show_prompt"]

from pathlib import Path

from looper.config import (
    ROLES_DIR,
    build_claude_autoload_context,
    inject_content,
    load_role_config,
    parse_frontmatter,
)
from looper.context import run_session_start_commands
from looper.rotation import get_rotation_focus


def show_prompt(mode: str) -> None:
    """Display the COMPLETE prompt that AI actually sees.

    For Claude, shows:
    - [AUTO-LOADED BY CLAUDE] CLAUDE.md + .claude/rules/*.md
    - [LOOPER INJECTED] shared.md + role.md with injection content

    For Codex, shows:
    - [LOOPER PREPENDED] CLAUDE.md + .claude/rules/*.md + .claude/codex.md
    - [LOOPER INJECTED] shared.md + role.md with injection content
    """
    print("=" * 70)
    print(f"FULL AI PROMPT FOR: {mode.upper()}")
    print("=" * 70)
    print()

    # Load role config
    config, prompt_template = load_role_config(mode)

    # Show source files
    print("### SOURCE FILES ###")
    print()
    idx = 1
    # Claude auto-loads these
    if Path("CLAUDE.md").exists():
        print(f"{idx}. CLAUDE.md [AUTO-LOADED BY CLAUDE]")
        idx += 1
    rules_dir = Path(".claude/rules")
    if rules_dir.exists():
        rules_files = sorted(rules_dir.glob("*.md"))
        if rules_files:
            print(f"{idx}. .claude/rules/*.md ({len(rules_files)} files) [AUTO-LOADED BY CLAUDE]")
            idx += 1
    shared_path = ROLES_DIR / "shared.md"
    role_path = ROLES_DIR / f"{mode}.md"
    print(f"{idx}. {shared_path} [LOOPER INJECTED]")
    idx += 1
    print(f"{idx}. {role_path} [LOOPER INJECTED]")
    idx += 1
    if Path(".looper_config.json").exists():
        print(f"{idx}. .looper_config.json (config overrides)")
    print()

    # Show config
    print("### PARSED CONFIG ###")
    print()
    for key, value in config.items():
        if key != "phase_data":  # Skip verbose phase data
            print(f"  {key}: {value}")
    print()

    # Show phase data if present
    phase_data = config.get("phase_data", {})
    if phase_data:
        print("### ROTATION PHASES ###")
        print()
        for phase_name, data in phase_data.items():
            print(f"  {phase_name}:")
            print(f"    weight: {data.get('weight', 1)}")
            print(f"    min_findings: {data.get('min_findings', 3)}")
            goals = data.get("goals", [])
            if goals:
                print(f"    goals: ({len(goals)} items)")
        print()

    # Run session start commands to get injection content
    print("### INJECTED CONTENT ###")
    print()
    session_results = run_session_start_commands(mode)

    for key, value in session_results.items():
        if value:
            lines = value.split("\n")
            preview = lines[0][:60] + "..." if len(lines[0]) > 60 else lines[0]
            if len(lines) > 1:
                preview += f" (... and {len(lines) - 1} more lines)"
            print(f"  {key}: {preview}")
        else:
            print(f"  {key}: (empty)")
    print()

    # Calculate rotation focus
    rotation_type = config.get("rotation_type", "")
    rotation_phases = config.get("rotation_phases", [])
    freeform_frequency = config.get("freeform_frequency", 3)
    force_phase = config.get("force_phase")
    starvation_hours = config.get("starvation_hours", 24)

    # Use iteration 1 for demo
    rotation_focus, selected_phase = get_rotation_focus(
        iteration=1,
        rotation_type=rotation_type,
        phases=rotation_phases,
        phase_data=phase_data,
        role=mode,
        freeform_frequency=freeform_frequency,
        force_phase=force_phase,
        starvation_hours=starvation_hours,
    )

    if rotation_focus:
        print("### ROTATION FOCUS (iteration 1) ###")
        print()
        print(f"  Selected phase: {selected_phase or 'freeform'}")
        print()

    # Build replacements
    audit_min_issues = config.get("audit_min_issues", 3)
    replacements = {
        "git_log": session_results.get("git_log", "(unavailable)"),
        "gh_issues": session_results.get("gh_issues", "(unavailable)"),
        "active_issue": session_results.get("active_issue", ""),
        "last_directive": session_results.get("last_directive", ""),
        "other_feedback": session_results.get("other_feedback", ""),
        "role_mentions": session_results.get("role_mentions", ""),
        "system_status": session_results.get("system_status", ""),
        "audit_data": session_results.get("audit_data", ""),
        "rotation_focus": rotation_focus,
        "audit_min_issues": str(audit_min_issues),
        "recovery_context": "",  # Only populated during crash recovery
        "theme_context": "",  # Not available in diagnostic mode (#2574)
        "handoff_context": session_results.get("handoff_context", ""),
    }

    # Apply injections to looper prompt
    looper_prompt = inject_content(prompt_template, replacements)

    # Build complete prompt (what AI actually sees)
    autoload_context = build_claude_autoload_context()
    autoload_lines = autoload_context.count("\n") + 1 if autoload_context else 0

    print("=" * 70)
    print("COMPLETE PROMPT (what AI actually sees)")
    print("=" * 70)
    print()

    # Show auto-loaded content
    if autoload_context:
        print("### [AUTO-LOADED BY CLAUDE] ###")
        print("-" * 70)
        print(autoload_context)
        print("-" * 70)
        print()

    # Show looper-injected content
    print("### [LOOPER INJECTED] ###")
    print("-" * 70)
    print(looper_prompt)
    print("-" * 70)
    print()

    # Summary
    full_prompt = f"{autoload_context}\n\n{looper_prompt}" if autoload_context else looper_prompt
    print("### SUMMARY ###")
    print()
    print(f"  Auto-loaded content: {len(autoload_context)} chars, {autoload_lines} lines")
    print(f"  Looper prompt: {len(looper_prompt)} chars, {looper_prompt.count(chr(10)) + 1} lines")
    print(f"  TOTAL: {len(full_prompt)} chars, {full_prompt.count(chr(10)) + 1} lines")
    replaced = len([k for k, v in replacements.items() if v])
    print(f"  Injection markers replaced: {replaced}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m looper.runner_prompt <role>")
        print("  Roles: worker, manager, prover, researcher")
        sys.exit(1)

    role = sys.argv[1].lower()
    show_prompt(role)

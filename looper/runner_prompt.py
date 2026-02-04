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

from looper.config import ROLES_DIR, inject_content, load_role_config, parse_frontmatter
from looper.context import run_session_start_commands
from looper.rotation import get_rotation_focus


def show_prompt(mode: str) -> None:
    """Display the full system prompt with source annotations.

    Shows how the prompt is assembled from:
    - .claude/roles/shared.md (shared template)
    - .claude/roles/{mode}.md (role-specific template)
    - Injected content (git_log, gh_issues, rotation_focus, etc.)
    """
    print("=" * 70)
    print(f"SYSTEM PROMPT FOR: {mode.upper()}")
    print("=" * 70)
    print()

    # Load role config
    config, prompt_template = load_role_config(mode)

    # Show source files
    print("### SOURCE FILES ###")
    print()
    shared_path = ROLES_DIR / "shared.md"
    role_path = ROLES_DIR / f"{mode}.md"
    print(f"1. {shared_path}")
    print(f"2. {role_path}")
    if Path(".looper_config.json").exists():
        print("3. .looper_config.json (config overrides)")
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
        "last_directive": session_results.get("last_directive", ""),
        "other_feedback": session_results.get("other_feedback", ""),
        "role_mentions": session_results.get("role_mentions", ""),
        "system_status": session_results.get("system_status", ""),
        "audit_data": session_results.get("audit_data", ""),
        "rotation_focus": rotation_focus,
        "audit_min_issues": str(audit_min_issues),
    }

    # Show template before injection
    print("### TEMPLATE (before injection) ###")
    print()
    print("--- shared.md ---")
    shared_content = (ROLES_DIR / "shared.md").read_text()
    _, shared_body = parse_frontmatter(shared_content)
    # Show first 20 lines
    for line in shared_body.strip().split("\n")[:20]:
        print(f"  {line}")
    if shared_body.count("\n") > 20:
        print(f"  ... ({shared_body.count(chr(10)) - 20} more lines)")
    print()

    print(f"--- {mode}.md ---")
    role_content = (ROLES_DIR / f"{mode}.md").read_text()
    _, role_body = parse_frontmatter(role_content)
    # Show first 30 lines
    for line in role_body.strip().split("\n")[:30]:
        print(f"  {line}")
    if role_body.count("\n") > 30:
        print(f"  ... ({role_body.count(chr(10)) - 30} more lines)")
    print()

    # Apply injections and show final prompt
    final_prompt = inject_content(prompt_template, replacements)

    print("### FINAL PROMPT (after injection) ###")
    print()
    print("-" * 70)
    print(final_prompt)
    print("-" * 70)
    print()

    # Summary
    print("### SUMMARY ###")
    print()
    print(f"  Total prompt length: {len(final_prompt)} chars")
    print(f"  Total prompt lines: {final_prompt.count(chr(10)) + 1}")
    replaced = len([k for k, v in replacements.items() if v])
    print(f"  Injection markers replaced: {replaced}")
    print()

"""
looper/rotation.py - Rotation state management

Manages phase rotation for roles that cycle through different focus areas:
- Manager: audit phases (priority_review, worker_health, issue_health, etc.)
- Researcher: research phases (external, internal, design, gap_analysis, etc.)
- Prover: verification phases (formal_proofs, tool_quality, claim_verification, etc.)
- Worker: priority tiers (high_priority, normal_work, quality)

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# fcntl is Unix-only for file locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

ROTATION_STATE_FILE = Path(".rotation_state.json")
ROTATION_LOCK_FILE = Path(".rotation_state.lock")


def load_rotation_state() -> dict:
    """Load rotation state from .rotation_state.json."""
    if ROTATION_STATE_FILE.exists():
        try:
            return json.loads(ROTATION_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_rotation_state(state: dict) -> None:
    """Save rotation state to .rotation_state.json."""
    try:
        ROTATION_STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    except OSError:
        pass


def update_rotation_state(role: str, phase: str) -> None:
    """Update rotation state after completing a phase.

    Uses file locking to prevent race conditions when multiple roles
    update state simultaneously. If locking fails, proceeds without lock.
    """
    lock_file = None
    try:
        # Try to acquire exclusive lock for read-modify-write
        if HAS_FCNTL:
            try:
                lock_file = open(ROTATION_LOCK_FILE, "w")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except OSError:
                # Lock failed - proceed without lock (better than skipping update)
                if lock_file:
                    lock_file.close()
                    lock_file = None

        # Read-modify-write (with or without lock)
        state = load_rotation_state()
        if role not in state:
            state[role] = {}
        state[role][phase] = {
            "last_run": datetime.now(timezone.utc).isoformat(),
        }
        save_rotation_state(state)

    finally:
        # Release lock if held
        if lock_file:
            if HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


def select_phase_by_priority(
    phases: list[str],
    phase_data: dict[str, dict],
    role_state: dict[str, dict],
    starvation_hours: int = 24,
) -> str:
    """Select next phase using priority queue with starvation prevention.

    Score = weight × hours_since_last_run + starvation_bonus

    Starvation prevention: After starvation_hours (default 24), tasks get
    a large bonus that overrides weight differences. This ensures even
    low-weight tasks run at least once per day (in AI time = hours).

    Degenerates to sequential when:
    - No state exists (all phases score equally high)
    - All weights equal and all run same hour

    Returns empty string if phases list is empty.
    """
    if not phases:
        return ""

    now = datetime.now(timezone.utc)
    scores: list[tuple[float, int, str]] = []

    for i, phase in enumerate(phases):
        weight = phase_data.get(phase, {}).get("weight", 1)
        last_run_str = role_state.get(phase, {}).get("last_run")

        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
                hours = (now - last_run).total_seconds() / 3600
            except ValueError:
                hours = 1000 - i  # Parse error = treat as never run
        else:
            # Never run = high priority, but use index for deterministic ordering
            hours = 1000 - i  # Ensures sequential order for fresh start

        # Base score: weight × hours
        score = weight * max(hours, 1)

        # Starvation prevention: after threshold, add large bonus
        # This ensures low-weight tasks don't starve indefinitely
        if hours > starvation_hours:
            # Bonus grows quadratically after threshold
            overtime = hours - starvation_hours
            score += overtime * overtime

        scores.append((score, i, phase))  # i for stable sort

    # Highest score wins; tie-break by original order
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][2]


def get_rotation_focus(
    iteration: int,
    rotation_type: str,
    phases: list[str],
    phase_data: Optional[dict[str, dict]] = None,
    role: str = "",
    freeform_frequency: int = 3,
    force_phase: Optional[str] = None,
    starvation_hours: int = 24,
) -> tuple[str, Optional[str]]:
    """Determine current rotation focus using priority queue.

    Args:
        iteration: Current iteration number (1-based)
        rotation_type: Type of rotation ('audit', 'research', 'verification', or empty)
        phases: List of phase names to cycle through
        phase_data: Optional dict of phase configs from parse_phase_blocks()
        role: Role name for state tracking
        freeform_frequency: Every Nth iteration is freeform (default 3)
        force_phase: Override to force specific phase
        starvation_hours: Hours before low-weight phases get bonus (default 24)

    Returns:
        (focus_string, selected_phase) - phase is None for freeform iterations

    Logic:
        - Every Nth iteration (configurable): Freeform
        - Other iterations: Priority queue (weight × hours_since_last_run)
    """
    if not rotation_type or not phases:
        return "", None

    # Check for freeform iteration
    if freeform_frequency > 0 and iteration % freeform_frequency == 0:
        # Worker follows issues, other roles follow directives
        if role == "worker":
            return "**Freeform** - Follow issues or your judgment", None
        return "**Freeform** - Follow directives or your judgment", None

    # Check for forced phase
    if force_phase and force_phase in phases:
        phase = force_phase
    else:
        # Select by priority queue
        state = load_rotation_state()
        role_state = state.get(role, {})
        phase = select_phase_by_priority(
            phases, phase_data or {}, role_state, starvation_hours
        )

    # Format phase name nicely
    phase_display = phase.replace("_", " ").title()

    # If we have phase data with content, use it directly
    if phase_data and phase in phase_data:
        content = phase_data[phase].get("content", "")
        if content:
            return content, phase

    # Fallback without phase data or content
    return f"**{phase_display}** - Focus on this area for this iteration", phase

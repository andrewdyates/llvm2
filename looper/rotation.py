# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/rotation.py - Rotation state management

Manages phase rotation for roles that cycle through different focus areas:
- Manager: audit phases (priority_review, worker_health, issue_health, etc.)
- Researcher: research phases (external, internal, design, gap_analysis, etc.)
- Prover: verification phases (formal_proofs, tool_quality, claim_verification, etc.)
- Worker: priority tiers (high_priority, normal_work, quality)
"""

__all__ = [
    "ROTATION_STATE_VERSION",
    "get_rotation_focus",
    "load_rotation_state",
    "migrate_rotation_state",
    "save_rotation_state",
    "select_phase_by_priority",
    "update_rotation_state",
]

import json
from datetime import UTC, datetime
from pathlib import Path

from looper.log import debug_swallow, log_info, log_warning

# fcntl is Unix-only for file locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

ROTATION_STATE_FILE = Path(".rotation_state.json")
ROTATION_LOCK_FILE = Path(".rotation_state.lock")

# Schema version for rotation state - increment when changing structure
# Version history:
# - 0: Implicit/missing - no version field (original schema)
# - 1: Added _version field
ROTATION_STATE_VERSION = 1

# Phase scoring constants
# NEVER_RUN_HOURS: Synthetic "hours since last run" for phases that have never run
# or have unparsable timestamps. Set high enough (1000 hours ~= 41 days) to ensure
# never-run phases are selected before recently-run phases, but low enough to avoid
# overflow concerns. Subtracted by index (0, 1, 2...) for deterministic ordering.
NEVER_RUN_HOURS = 1000.0

# MAX_OVERTIME_BONUS: Cap on the starvation prevention bonus to prevent unbounded
# score growth. 168 hours = 1 week of AI time, giving bonus of 168² = 28,224.
# This is large enough to override any reasonable weight difference while keeping
# scores in a predictable range.
MAX_OVERTIME_BONUS = 168.0

# MAX_PHASE_SCORE: Informational upper bound for typical scores.
# Calculated as: typical_max_weight * max_hours + max_bonus² = 10 * 1000 + 168² ≈ 38,224
# Not enforced in code - scores could exceed this with extreme weights.
# Used for documentation and test assertions only.
MAX_PHASE_SCORE = 50000.0


def migrate_rotation_state(state: dict) -> tuple[dict, bool]:
    """Migrate rotation state from older versions to current.

    Contracts:
        REQUIRES: state is a dict (may be empty or any version)
        ENSURES: Returns (state, migrated) where migrated is True if state was changed
        ENSURES: For versions < ROTATION_STATE_VERSION: _version is upgraded
        ENSURES: For versions >= ROTATION_STATE_VERSION: state unchanged
        ENSURES: Logs warning when migration is applied
        ENSURES: Never raises (defensive parsing)

    Migration history:
        v0 -> v1: Add _version field (no structural changes)
    """
    version = state.get("_version", 0)
    migrated = False

    if version < ROTATION_STATE_VERSION:
        # v0 -> v1: Add version field
        if version < 1:
            state["_version"] = 1
            migrated = True
            # Log that migration was applied (only if state had content)
            if any(k for k in state if not k.startswith("_")):
                log_warning(
                    f"Migrated .rotation_state.json from v{version} to "
                    f"v{ROTATION_STATE_VERSION}"
                )

    return state, migrated


def load_rotation_state() -> dict[str, dict[str, object]]:
    """Load rotation state from .rotation_state.json.

    Contracts:
        REQUIRES: Current directory is project root (contains .rotation_state.json)
        ENSURES: Returns dict mapping role -> {phase -> {last_run: iso_timestamp}}
        ENSURES: If file missing or invalid JSON, returns empty dict {}
        ENSURES: Migrates state if version < ROTATION_STATE_VERSION and persists
        ENSURES: Never raises (silent fallback to empty state)
    """
    if ROTATION_STATE_FILE.exists():
        try:
            data = json.loads(ROTATION_STATE_FILE.read_text())
            if isinstance(data, dict):
                # Migrate to current version if needed
                data, migrated = migrate_rotation_state(data)
                # Persist migration so it doesn't repeat on next load
                if migrated:
                    save_rotation_state(data)
                # Validate structure - return role->dict state only (exclude metadata)
                return {
                    k: v
                    for k, v in data.items()
                    if isinstance(k, str) and not k.startswith("_") and isinstance(v, dict)
                }
        except (json.JSONDecodeError, OSError) as e:
            debug_swallow("load_rotation_state", e)
    return {}


def save_rotation_state(state: dict[str, dict[str, object]]) -> None:
    """Save rotation state to .rotation_state.json.

    Contracts:
        REQUIRES: state is a dict (type-hinted)
        ENSURES: Writes JSON to .rotation_state.json with indent=2 and trailing newline
        ENSURES: State includes _version field
        ENSURES: If write fails (OSError), silently returns (no exception)
        ENSURES: Never raises (graceful degradation)
    """
    try:
        # Ensure version is set
        if "_version" not in state:
            state["_version"] = ROTATION_STATE_VERSION
        ROTATION_STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    except OSError as e:
        debug_swallow("save_rotation_state", e)


def update_rotation_state(role: str, phase: str) -> None:
    """Update rotation state after completing a phase.

    Uses file locking to prevent race conditions when multiple roles
    update state simultaneously. If locking fails, proceeds without lock.

    Contracts:
        REQUIRES: role is a non-empty string (role name)
        REQUIRES: phase is a non-empty string (phase name)
        ENSURES: Updates state[role][phase] = {last_run: current_iso_timestamp}
        ENSURES: Creates role entry in state if not present
        ENSURES: Acquires exclusive file lock if fcntl available (Unix)
        ENSURES: Releases lock in finally block (no lock leaks)
        ENSURES: If lock acquisition fails, proceeds without lock
        ENSURES: Never raises (graceful degradation)
    """
    lock_file = None
    try:
        # Try to acquire exclusive lock for read-modify-write
        if HAS_FCNTL:
            try:
                lock_file = open(ROTATION_LOCK_FILE, "w")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except OSError as e:
                # Lock failed - proceed without lock (better than skipping update)
                debug_swallow("rotation_state_lock", e)
                if lock_file:
                    lock_file.close()
                    lock_file = None

        # Read-modify-write (with or without lock)
        state = load_rotation_state()
        if role not in state:
            state[role] = {}
        state[role][phase] = {
            "last_run": datetime.now(UTC).isoformat(),
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
    phase_data: dict[str, dict[str, object]],
    role_state: dict[str, object],
    starvation_hours: int = 24,
) -> str:
    """Select next phase using priority queue with starvation prevention.

    Contracts:
        REQUIRES: phases is a list (may be empty)
        REQUIRES: phase_data values have optional 'weight' key (default 1)
        REQUIRES: role_state values have optional 'last_run' ISO timestamp
        REQUIRES: starvation_hours > 0
        ENSURES: If phases is empty, returns ""
        ENSURES: Otherwise, returns a value that is in phases
        ENSURES: Highest-scored phase wins; ties broken by list order
        ENSURES: Deterministic for same inputs and system time
        ENSURES: Score typically bounded by MAX_PHASE_SCORE under normal weights
                 (extreme weight values may exceed; see constant docstring)

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

    now = datetime.now(UTC)
    scores: list[tuple[float, int, str]] = []

    for i, phase in enumerate(phases):
        weight_val = phase_data.get(phase, {}).get("weight", 1)
        weight: float = (
            float(weight_val) if isinstance(weight_val, (int, float)) else 1.0
        )

        # Extract last_run from role_state[phase], with type narrowing
        phase_state = role_state.get(phase)
        last_run_str: str | None = None
        if isinstance(phase_state, dict):
            lr = phase_state.get("last_run")
            if isinstance(lr, str):
                last_run_str = lr

        hours: float
        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
                hours = (now - last_run).total_seconds() / 3600
            except ValueError:
                # Parse error = treat as never run with deterministic ordering
                hours = NEVER_RUN_HOURS - i
        else:
            # Never run = high priority, but use index for deterministic ordering
            hours = NEVER_RUN_HOURS - i

        # Base score: weight × hours
        score: float = weight * max(hours, 1.0)

        # Starvation prevention: after threshold, add large bonus
        # This ensures low-weight tasks don't starve indefinitely
        if hours > starvation_hours:
            # Bonus grows quadratically after threshold, capped to prevent unbounded growth
            overtime = min(hours - starvation_hours, MAX_OVERTIME_BONUS)
            score += overtime * overtime

        scores.append((score, i, phase))  # i for stable sort

    # Highest score wins; tie-break by original order
    if not scores:
        return ""  # Defensive: empty scores shouldn't happen but prevents IndexError
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][2]


def get_rotation_focus(
    iteration: int,
    rotation_type: str,
    phases: list[str],
    phase_data: dict[str, dict[str, object]] | None = None,
    role: str = "",
    freeform_frequency: int = 3,
    force_phase: str | None = None,
    starvation_hours: int = 24,
) -> tuple[str, str | None]:
    """Determine current rotation focus using priority queue.

    Contracts:
        REQUIRES: iteration >= 1 (1-based iteration counter)
        REQUIRES: freeform_frequency >= 0 (0 disables freeform)
        ENSURES: If rotation_type empty or phases empty, returns ("", None)
        ENSURES: If freeform iteration, returns freeform message with None phase
        ENSURES: Otherwise, returns (focus_string, phase) where phase in phases

    Args:
        iteration: Current iteration number (1-based)
        rotation_type: Type of rotation ('audit', 'research', 'verification', 'work', or empty)
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
        # Audit-overhead circuit breaker (#2808 §4): override researcher freeform
        # to "design" phase when breaker is active, producing design artifacts
        # instead of repeated observations.
        if role == "researcher" and "design" in phases:
            try:
                from looper.telemetry import get_audit_overhead_state

                overhead = get_audit_overhead_state()
                if overhead and overhead.state == "active":
                    log_info(
                        "Audit-overhead circuit active: overriding researcher "
                        "freeform to design phase"
                    )
                    # Fall through to normal phase selection with force_phase="design"
                    force_phase = "design"
                else:
                    return "**Freeform** - Follow directives or your judgment", None
            except Exception as e:
                debug_swallow("audit_overhead_freeform_override", e)
                return "**Freeform** - Follow directives or your judgment", None
        elif role == "worker":
            return "**Freeform** - Follow issues or your judgment", None
        else:
            return "**Freeform** - Follow directives or your judgment", None

    # Check for forced phase
    if force_phase and force_phase in phases:
        phase = force_phase
    else:
        # Select by priority queue
        state = load_rotation_state()
        # state[role] is dict[phase -> {last_run: ...}], same type as state values
        role_state = state.get(role, {})
        phase = select_phase_by_priority(
            phases, phase_data or {}, role_state, starvation_hours
        )

    # Format phase name nicely
    phase_display = phase.replace("_", " ").title()

    # If we have phase data with content, use it directly
    if phase_data and phase in phase_data:
        content = phase_data[phase].get("content", "")
        if content and isinstance(content, str):
            return content, phase

    # Fallback without phase data or content
    return f"**{phase_display}** - Focus on this area for this iteration", phase

# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner.py - Main loop runner class and CLI entrypoints.

Contains LoopRunner class and CLI helpers. Implementation is split across
runner_* modules to keep large files under the pulse threshold.

Architecture overview:
- LoopRunner composes RunnerBase with six mixins:
  RunnerLoopMixin, RunnerAuditMixin, RunnerIterationMixin,
  RunnerSyncMixin, RunnerGitMixin, RunnerControlMixin.
- Method resolution order (MRO):
  LoopRunner -> RunnerLoopMixin -> RunnerAuditMixin -> RunnerIterationMixin ->
  RunnerSyncMixin -> RunnerGitMixin -> RunnerControlMixin -> RunnerBase -> object
- Mixins share state via RunnerBase attributes; see mixin class docstrings
  for required attributes and assumptions.

Context Transfer Strategy (Claude vs Codex):
- Claude Code: Uses --resume <session_id> for full conversation context continuity
- Codex: No session resume support with --json output. Instead, injects last
  commit message into audit prompt for context. Limited to 1 audit round.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from looper.cleanup import cleanup_stale_state_files
from looper.config import LOG_DIR, PID_FILE_TEMPLATE
from looper.log import (
    build_log_path,
    log_error,
    log_info,
    log_warning,
    setup_logging,
)
from looper.runner_audit import RunnerAuditMixin
from looper.runner_base import RunnerBase
from looper.runner_control import RunnerControlMixin
from looper.runner_git import RunnerGitMixin
from looper.runner_iteration import RunnerIterationMixin
from looper.runner_loop import RunnerLoopMixin
from looper.runner_prompt import show_prompt
from looper.runner_sync import RunnerSyncMixin
from ai_template_scripts.subprocess_utils import is_process_alive
from looper.subprocess_utils import run_git_command

__all__ = [
    "LoopRunner",
    "show_prompt",
    "main",
    "setup_wrapper_path",
    "check_dirty_worktree",
    "detect_other_sessions",
    "check_concurrent_sessions",
]


def detect_other_sessions(
    current_mode: str | None = None, worker_id: int | None = None
) -> list[tuple[str, int]]:
    """Detect other running looper sessions sharing this worktree (#1482).

    Checks PID files in the coordination directory (AIT_COORD_DIR or current dir)
    to find other active sessions.

    Args:
        current_mode: The mode of the current session (to exclude from results).
        worker_id: The worker ID of the current session (to exclude from results).

    Returns:
        List of (mode_display, pid) tuples for other running sessions.
        mode_display is like "worker", "worker_1", "manager", etc.
    """
    coord_dir = Path(os.environ.get("AIT_COORD_DIR", "."))
    other_sessions: list[tuple[str, int]] = []

    # Build the current session's PID file name for exclusion
    if current_mode:
        if worker_id is not None:
            current_suffix = f"{current_mode}_{worker_id}"
        else:
            current_suffix = current_mode
        current_pid_name = PID_FILE_TEMPLATE.format(mode=current_suffix)
    else:
        current_pid_name = None

    # Check all PID files in coord_dir
    for pid_file in coord_dir.glob(".pid_*"):
        if pid_file.suffix == ".tmp":
            continue
        if pid_file.name == current_pid_name:
            continue  # Skip our own PID file

        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            continue  # Malformed or unreadable

        if not is_process_alive(pid):
            continue  # Process not running

        # Extract mode display from filename: .pid_worker_1 -> worker_1
        mode_display = pid_file.name.replace(".pid_", "")
        if not mode_display:
            continue  # Skip malformed .pid_ file without mode suffix
        other_sessions.append((mode_display, pid))

    return other_sessions


def check_concurrent_sessions(
    current_mode: str,
    worker_id: int | None = None,
) -> None:
    """Log other running sessions in this checkout for awareness.

    All roles share the same checkout and coordinate via file tracking
    (.worker_N_files.json). This function logs other active sessions
    so the user knows who else is working in the same repo.

    Run 'git status' to see worker file ownership context.

    Args:
        current_mode: The mode of the current session.
        worker_id: The worker ID of the current session.
    """
    other_sessions = detect_other_sessions(current_mode, worker_id)
    if not other_sessions:
        return  # No other sessions

    # Log other sessions for awareness (informational, not blocking)
    log_info("Other sessions active in this checkout:")
    for mode_display, pid in other_sessions:
        log_info(f"    - {mode_display} (PID {pid})")
    log_info("  Run 'git status' to see worker file ownership.")
    log_info("")


def check_dirty_worktree(
    allow_dirty: bool = False,
    current_mode: str | None = None,
    worker_id: int | None = None,
) -> None:
    """Guard against starting sessions with uncommitted changes (#983).

    Prevents confusing audits, accidental commits, and broken builds from
    dirty working trees left by previous sessions.

    Args:
        allow_dirty: If True, skip the blocking check (still warns if other
            sessions detected per #1482).
        current_mode: The mode of the current session (to exclude from
            other-session detection).
        worker_id: The worker ID of the current session (to exclude from
            other-session detection).

    Raises:
        SystemExit: If uncommitted changes are detected and not overridden.

    Behavior:
        1. Checks for uncommitted changes (staged or unstaged)
        2. If dirty and other sessions running: warns about cross-role interference
        3. If dirty and not overridden: saves diff snapshot and exits
        4. Snapshot saved to reports/YYYY-MM-DD-dirty-worktree-snapshot.diff
    """

    result = run_git_command(["status", "--porcelain"], timeout=10)
    if not result.ok:
        # Can't determine status - warn but continue
        log_warning("Could not check git status", stream="stderr")
        return

    status_output = (result.value or "").strip()
    if not status_output:
        return  # Clean worktree, proceed

    # Dirty worktree detected - count affected files
    dirty_files = status_output.split("\n")
    file_count = len(dirty_files)

    # Check for other running sessions (#1482) - warn even if allow_dirty
    other_sessions = detect_other_sessions(current_mode, worker_id)
    if allow_dirty:
        if other_sessions:
            # Warn but don't block - user explicitly requested --allow-dirty
            log_warning(
                f"Dirty worktree ({file_count} files) with other sessions running:",
                stream="stderr",
            )
            for mode_display, pid in other_sessions:
                log_warning(f"    - {mode_display} (PID {pid})", stream="stderr")
            log_warning(
                "  Changes may belong to another role.",
                stream="stderr",
            )
            log_info("", stream="stderr")
        return  # User explicitly allowed dirty

    # Save snapshot to reports/ for audit trail
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    snapshot_path = reports_dir / f"{timestamp}-dirty-worktree-snapshot.diff"

    # Capture full diff for audit trail
    diff_result = run_git_command(["diff", "HEAD"], timeout=30)
    diff_content = diff_result.value if diff_result.ok else "(diff unavailable)"

    # Also capture staged changes
    staged_result = run_git_command(["diff", "--cached"], timeout=30)
    staged_content = staged_result.value if staged_result.ok else ""

    # Write snapshot
    with open(snapshot_path, "w") as f:
        f.write(f"# Dirty worktree snapshot - {timestamp}\n")
        f.write(f"# {file_count} file(s) with uncommitted changes\n")
        f.write("#\n")
        f.write("# Files:\n")
        f.writelines(f"#   {line}\n" for line in dirty_files)
        f.write("#\n")
        if staged_content:
            f.write("# === STAGED CHANGES ===\n")
            f.write(staged_content)
            f.write("\n")
        f.write("# === UNSTAGED CHANGES ===\n")
        f.write(diff_content or "(no diff)")

    # other_sessions already detected above (#1482)

    # Error block
    log_info("", stream="stderr")
    log_error(
        f"Dirty working tree detected ({file_count} file(s) with uncommitted changes)",
        stream="stderr",
    )

    # Warn if another session may own the changes (#1482)
    if other_sessions:
        log_info("", stream="stderr")
        log_warning(
            "  Other looper sessions are running in this worktree:",
            stream="stderr",
        )
        for mode_display, pid in other_sessions:
            log_warning(f"      - {mode_display} (PID {pid})", stream="stderr")
        log_warning(
            "  These changes may belong to another session!",
            stream="stderr",
        )
        log_warning(
            "  Run 'git status' to see worker file ownership.",
            stream="stderr",
        )

    log_info("", stream="stderr")
    log_error(
        "  This can cause confusing audits and accidental commits.",
        stream="stderr",
    )
    log_error(f"  Snapshot saved to: {snapshot_path}", stream="stderr")
    log_info("", stream="stderr")
    log_error("  Options:", stream="stderr")
    log_error("    1. Commit your changes (WIP commit is fine)", stream="stderr")
    log_error("    2. Discard changes: git checkout -- .", stream="stderr")
    log_error(
        "    3. Override check: ./looper.py <mode> --allow-dirty",
        stream="stderr",
    )
    log_info("", stream="stderr")
    sys.exit(1)


class LoopRunner(
    RunnerLoopMixin,
    RunnerAuditMixin,
    RunnerIterationMixin,
    RunnerSyncMixin,
    RunnerGitMixin,
    RunnerControlMixin,
    RunnerBase,
):
    """Main loop runner for autonomous AI iterations.

    Composes RunnerBase with six mixins to orchestrate the AI iteration loop.
    Each mixin handles a specific responsibility:

    - RunnerLoopMixin: Main run() loop, delays, telemetry recording
    - RunnerAuditMixin: do-audit → needs-review workflow, audit rounds
    - RunnerIterationMixin: Single iteration execution, git commit/push
    - RunnerSyncMixin: Multi-machine zone branch synchronization
    - RunnerGitMixin: Git identity, environment variables, session ID
    - RunnerControlMixin: Signal handlers, graceful shutdown

    Usage:
        runner = LoopRunner(mode="worker", worker_id=1)
        runner.setup()  # Initialize environment
        runner.run()    # Start iteration loop

    The run() method loops indefinitely, executing AI iterations until
    a STOP file is detected, a signal is received, or an unrecoverable
    error occurs.

    See docs/looper.md for full architecture and configuration options.
    """


def setup_wrapper_path() -> None:
    """Prepend ai_template_scripts/bin to PATH for wrapper precedence.

    This ensures gh, cargo, etc. use the ai_template wrappers instead of
    system binaries. Must be called very early in main() before any gh/cargo
    calls. See #1690 (gh wrapper PATH precedence issue).
    """
    wrapper_bin = Path("ai_template_scripts/bin").resolve()
    if not wrapper_bin.exists():
        # Not in an ai_template repo - skip silently
        return

    current_path = os.environ.get("PATH", "")
    wrapper_str = str(wrapper_bin)

    # Check if wrapper is already first in PATH
    path_parts = current_path.split(os.pathsep) if current_path else []
    if path_parts and path_parts[0] == wrapper_str:
        return  # Already configured

    # Prepend wrapper bin to PATH
    os.environ["PATH"] = f"{wrapper_str}{os.pathsep}{current_path}"


def main() -> None:
    """Entry point for the looper runner.

    Parses CLI arguments, validates configuration, and starts the iteration loop
    for the specified role (WORKER, PROVER, RESEARCHER, MANAGER).
    """
    setup_logging()

    # Ensure ai_template wrappers precede system binaries (#1690)
    # Must happen before ANY gh/cargo calls including checks
    setup_wrapper_path()

    # Check for flags
    show_prompt_mode = "--show-prompt" in sys.argv
    validate_config_mode = "--validate-config" in sys.argv
    allow_dirty = "--allow-dirty" in sys.argv
    help_mode = "--help" in sys.argv or "-h" in sys.argv

    # Parse args first so we can pass mode/worker_id to check_dirty_worktree (#1482)
    # Parse --id=N for multi-worker support
    # Parse --machine=NAME and --branch=BRANCH for multi-machine support
    worker_id: int | None = None
    machine: str | None = None
    branch: str | None = None
    filtered_args = []
    for a in sys.argv[1:]:
        if a in (
            "--show-prompt",
            "--validate-config",
            "--allow-dirty",
            "--help",
            "-h",
        ):
            continue
        if a.startswith("--id="):
            try:
                worker_id = int(a[5:])
                # Must be 1-5 to match pulse.py GraphQL label counting
                # See #1080: prevents in-progress-W43 label sprawl
                if worker_id < 1 or worker_id > 5:
                    log_error("Error: --id must be 1-5 (e.g., --id=1)", stream="stderr")
                    sys.exit(1)
            except ValueError:
                log_error(
                    f"Error: Invalid --id value: {a[5:]!r}",
                    stream="stderr",
                )
                sys.exit(1)
            continue
        if a.startswith("--machine="):
            machine = a[10:].strip()
            if not machine or not machine.isidentifier():
                log_error(
                    "Error: --machine must be a valid identifier (e.g., --machine=sat)",
                    stream="stderr",
                )
                sys.exit(1)
            continue
        if a.startswith("--branch="):
            branch = a[9:].strip()
            if not branch:
                log_error("Error: --branch cannot be empty", stream="stderr")
                sys.exit(1)
            continue
        filtered_args.append(a)
    args = filtered_args

    # Validate machine/branch consistency
    if machine and not branch:
        # Auto-derive branch from machine name
        branch = f"zone/{machine}"
    if branch and not machine:
        log_error(
            "Error: --branch requires --machine (e.g., --machine=sat --branch=zone/sat)",
            stream="stderr",
        )
        sys.exit(1)

    # Determine mode and whether to show help
    # Check help_mode first - always show help if explicitly requested
    if help_mode:
        show_help = True
        mode = "worker"  # Dummy value, not used
    elif len(args) == 0:
        show_help = False
        mode = "worker"
    elif len(args) == 1 and args[0] in (
        "worker",
        "prover",
        "researcher",
        "manager",
        "cleanup",
    ):
        show_help = False
        mode = args[0]
    else:
        # Invalid args - show help with error exit
        show_help = True
        mode = "worker"  # Dummy value, not used

    if show_help:
        # Show help - exit 0 if explicitly requested, 1 if bad args
        log_info("Usage: ./looper.py [worker|prover|researcher|manager|cleanup] [opts]")
        log_info("")
        log_info("  worker     - Fast autonomous loop (no delay) [default]")
        log_info("  prover     - Proof loop (15-min intervals)")
        log_info("  researcher - Research loop (10-min intervals)")
        log_info("  manager    - Audit loop (5-min intervals)")
        log_info("  cleanup    - Remove stale state files and exit")
        log_info("")
        log_info("Options:")
        log_info(
            "  --id=N            Instance identity (1-5) for multi-instance mode (e.g., --id=1)"
        )
        log_info(
            "  --machine=NAME    Machine identity for multi-machine mode (e.g., --machine=sat)"
        )
        log_info(
            "  --branch=BRANCH   Git branch for this machine (default: zone/<machine>)"
        )
        log_info("  --show-prompt     Display the full system prompt and exit")
        log_info("  --validate-config Validate config schema and exit")
        log_info(
            "  --allow-dirty     Allow starting with uncommitted changes (not recommended)"
        )
        log_info("  -h, --help        Show this help message and exit")
        log_info("")
        log_info("Stop: touch STOP (all) or STOP_WORKER/STOP_MANAGER/etc.")
        log_info("")
        log_info("Spawn all 4 loops: ./ai_template_scripts/spawn_all.sh")
        sys.exit(0 if help_mode else 1)

    # Handle --validate-config (does not need clean worktree)
    if validate_config_mode:
        from looper.config import (  # noqa: PLC0415
            CONFIG_SCHEMA,
            load_role_config,
            validate_config,
        )

        log_info(f"Validating config for mode: {mode}")
        log_info(f"Known config keys: {sorted(CONFIG_SCHEMA.keys())}")
        log_info("")
        try:
            config, _ = load_role_config(mode)
            # validate_config already called during load_role_config, but call again
            # with strict=True to show all messages and exit with error code
            messages = validate_config(config, mode, strict=False)
            if messages:
                log_info("\nValidation issues found:")
                for msg in messages:
                    log_info(f"  {msg}")
                errors = [m for m in messages if m.startswith("Error")]
                if errors:
                    sys.exit(1)
            else:
                log_info("Config validation passed - no issues found")
            sys.exit(0)
        except Exception as e:
            # Catch-all: config load failure reported, exit non-zero
            log_error(f"Error loading config: {e}", stream="stderr")
            sys.exit(1)

    # Handle --show-prompt (does not need clean worktree)
    if show_prompt_mode:
        show_prompt(mode)
        sys.exit(0)

    # Handle cleanup command (no interactive terminal required)
    if mode == "cleanup":
        cleanup_stale_state_files()
        sys.exit(0)

    # Check for dirty worktree (#983) - hard fail unless overridden
    # Now we have mode and worker_id parsed, so we can detect cross-role interference (#1482)
    check_dirty_worktree(
        allow_dirty=allow_dirty, current_mode=mode, worker_id=worker_id
    )

    # Log other active sessions for awareness
    check_concurrent_sessions(current_mode=mode, worker_id=worker_id)

    # Require interactive terminal - running in background causes orphaned processes
    if not sys.stdin.isatty():
        log_error("ERROR: looper.py requires an interactive terminal", stream="stderr")
        log_error(
            "       Use spawn_session.sh to create a proper iTerm2 tab",
            stream="stderr",
        )
        sys.exit(1)

    log_file = build_log_path(LOG_DIR, mode, worker_id=worker_id, machine=machine)
    setup_logging(log_file=log_file)
    cleanup_stale_state_files()
    LoopRunner(mode, worker_id=worker_id, machine=machine, branch=branch).run()


if __name__ == "__main__":
    main()

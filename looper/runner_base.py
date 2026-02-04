# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_base.py - Base class for the LoopRunner.

Provides RunnerBase, the foundation class that handles:
- Configuration loading and validation
- Git identity and environment setup
- PID file management for process isolation
- Status file writing and crash log management
- Checkpoint manager initialization for crash recovery
- Multi-worker coordination (zones, file tracking)

This is part of the runner mixin architecture:
    LoopRunner(RunnerBase, RunnerLoopMixin, RunnerIterationMixin,
               RunnerAuditMixin, RunnerSyncMixin, RunnerGitMixin,
               RunnerControlMixin)

See docs/looper.md "Runner Architecture" for the full component diagram.
"""

from __future__ import annotations

__all__ = ["RunnerBase", "VALID_ROLES"]

import os
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from looper.checkpoint import (
    CheckpointManager,
    RecoveryContext,
    get_checkpoint_filename,
)
from looper.config import (
    ITERATION_FILE_TEMPLATE,
    LOG_DIR,
    PID_FILE_TEMPLATE,
    STATUS_FILE_TEMPLATE,
    load_role_config,
)
from looper.constants import FLAGS_DIR
from looper.file_tracker import (
    FileTracker,
    cleanup_stale_trackers,
    get_uncommitted_files,
)
from looper.hooks import install_hooks
from looper.issue_manager import IssueManager
from looper.iteration import IterationRunner
from looper.log import debug_swallow, log_error, log_info, log_warning
from looper.status import StatusManager
from looper.subprocess_utils import is_local_mode, run_gh_command
from looper.sync import check_stale_staged_files

VALID_ROLES = frozenset({"worker", "manager", "researcher", "prover"})


class RunnerBase:
    def __init__(
        self,
        mode: str,
        worker_id: int | None = None,
        machine: str | None = None,
        branch: str | None = None,
    ) -> None:
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a string, got {type(mode).__name__}")
        normalized_mode = mode.strip().lower()
        if not normalized_mode:
            raise ValueError("mode must be a non-empty string")
        if normalized_mode not in VALID_ROLES:
            raise ValueError(
                f"Invalid role: {mode!r}. Must be one of: {sorted(VALID_ROLES)}"
            )
        mode = normalized_mode
        self.mode = mode
        self.worker_id = worker_id
        self.machine = machine
        self.branch = branch
        self.iteration = 1
        self.running = True

        # For multi-machine mode, validate and setup branch
        if branch:
            self._setup_branch(branch)

        # Load role config from .claude/roles/ files
        # Pass worker_id for per-instance config support (#1175)
        self.config, self._prompt_template = load_role_config(mode, worker_id)

        # File paths - include worker_id suffix for multi-worker support
        # Format: worker -> worker_1, worker_2 when worker_id is specified
        file_suffix = f"{mode}_{worker_id}" if worker_id is not None else mode
        self.iteration_file = LOG_DIR / ITERATION_FILE_TEMPLATE.format(mode=file_suffix)
        # PID file location: use AIT_COORD_DIR if set, otherwise current directory.
        # This enables centralized PID management across multiple workers.
        pid_dir = Path(os.environ.get("AIT_COORD_DIR", "."))
        pid_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = pid_dir / PID_FILE_TEMPLATE.format(mode=file_suffix)
        # Status file format differs: .worker_1_status.json vs .worker_status.json
        if worker_id is not None:
            self.status_file = Path(f".{mode}_{worker_id}_status.json")
        else:
            self.status_file = Path(STATUS_FILE_TEMPLATE.format(mode=mode))
        self.crash_log = LOG_DIR / "crashes.log"

        # Managers for issue ops and status/logging
        repo_path = Path.cwd().resolve()
        self.issue_manager = IssueManager(repo_path=repo_path, role=mode)
        self.status_manager = StatusManager(
            repo_path=repo_path,
            mode=mode,
            status_file=self.status_file,
            crash_log=self.crash_log,
            config=self.config,
            get_ait_version=self.get_ait_version,
            log_dir=LOG_DIR,
        )

        # Check for alternative AI tool availability
        self.codex_available = shutil.which("codex") is not None
        self.dasher_available = shutil.which("dasher") is not None

        # Session identity
        self._session_id = uuid.uuid4().hex[:6]
        self._started_at = datetime.now(UTC).isoformat()
        self.status_manager.set_started_at(self._started_at)

        self.iteration_runner = IterationRunner(
            mode=mode,
            config=self.config,
            prompt_template=self._prompt_template,
            issue_manager=self.issue_manager,
            status_manager=self.status_manager,
            get_ait_version=self.get_ait_version,
            session_id=self._session_id,
            log_dir=LOG_DIR,
            codex_available=self.codex_available,
            dasher_available=self.dasher_available,
            worker_id=worker_id,
            machine=machine,
        )

        # Working issues from main iteration (for audit context)
        self._working_issues: list[int] = []

        # Per-role STOP file (e.g., STOP_WORKER, STOP_MANAGER)
        self._role_stop_file = f"STOP_{mode.upper()}"

        # STOP files are checked in current working directory
        self._stop_dir = Path(".")

        # Coordination directory for iteration locks and shared state
        self._coord_dir = LOG_DIR

        # Checkpoint manager for crash recovery
        checkpoint_filename = get_checkpoint_filename(mode, worker_id)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_file=Path(checkpoint_filename),
            mode=mode,
            session_id=self._session_id,
            worker_id=worker_id,
        )
        self._recovery_context: RecoveryContext | None = None

        # File tracker for multi-worker coordination (workers only)
        # Tracks which files each worker modifies to scope auto-fix tools
        self.file_tracker: FileTracker | None = None
        if mode == "worker" and worker_id is not None:
            self.file_tracker = FileTracker(
                repo_root=repo_path,
                worker_id=worker_id,
                session_id=self._session_id,
            )

        # Memory watchdog process (spawned in setup())
        self._watchdog_process: subprocess.Popen | None = None

    def setup(self) -> None:
        """Initialize environment, dependencies, and runtime state.

        Performs the following initialization steps in order:

        1. **Dependency checks** - Verifies claude, codex (optional), git, gh CLIs
        2. **Authentication** - Validates gh auth status and GitHub API connectivity
        3. **Directory setup** - Creates log directory (worker_logs/)
        4. **Git hooks** - Installs commit-msg-hook.sh and pre-commit-hook.sh
        5. **Git identity** - Configures GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL, etc.
        6. **Sync staleness** - Warns if repo is behind ai_template (100+ commits)
        7. **PID file** - Creates .pid_<mode> for process isolation
        8. **Iteration counter** - Restores from .iteration_<mode> if exists
        9. **Log rotation** - Rotates old logs via StatusManager
        10. **Signal handlers** - Installs SIGINT/SIGTERM handlers
        11. **Status tracking** - Initializes start time and status file
        12. **Multi-machine sync** - Syncs from origin/main if on zone branch
        13. **Staged file check** - Warns about stale staged files from prior sessions
        14. **Crash recovery** - Loads recovery context from checkpoint if available
        15. **File tracker setup** - Initializes multi-worker file tracking (workers only)
        16. **Memory watchdog** - Spawns watchdog process if enabled

        Exits with code 1 if any required dependency is missing or auth fails.
        """
        # Check core dependencies
        if not shutil.which("claude"):
            log_error("ERROR: claude CLI not found in PATH")
            log_error("  Install: npm install -g @anthropic-ai/claude-code")
            sys.exit(1)
        log_info(f"✓ Found claude: {shutil.which('claude')}")

        codex_prob = self.config.get("codex_probability", 0.0)
        if codex_prob > 0:
            if self.codex_available:
                codex_path = shutil.which("codex")
                log_info(f"✓ Found codex: {codex_path} (prob: {codex_prob:.0%})")
            else:
                log_info(f"  (codex not found, claude only - prob: {codex_prob:.0%})")

        json_to_text = Path("ai_template_scripts/json_to_text.py")
        if not json_to_text.exists():
            log_error(f"ERROR: {json_to_text} not found")
            sys.exit(1)
        log_info(f"✓ Found {json_to_text}")

        # Check git
        if not shutil.which("git"):
            log_error("ERROR: git not found in PATH")
            sys.exit(1)
        log_info(f"✓ Found git: {shutil.which('git')}")

        # Check GitHub CLI
        if not shutil.which("gh"):
            log_error("ERROR: gh (GitHub CLI) not found in PATH")
            log_error("  Install: brew install gh")
            sys.exit(1)
        log_info(f"✓ Found gh: {shutil.which('gh')}")

        # Check gh authentication
        if is_local_mode():
            log_warning("⚠ local mode enabled: skipping gh auth/network checks")
        else:
            auth_result = run_gh_command(["auth", "status"], timeout=10)
            if not auth_result.ok:
                log_error("ERROR: gh not authenticated")
                log_error("  Run: gh auth login")
                sys.exit(1)
            log_info("✓ gh authenticated")

            # Check network (non-blocking warning)
            net_result = run_gh_command(["api", "user", "--jq", ".login"], timeout=10)
            if net_result.ok and net_result.value:
                log_info(f"✓ GitHub connected as: {net_result.value.strip()}")
            elif net_result.error and "timeout" in net_result.error.lower():
                log_warning("⚠ GitHub API timeout (offline mode)")
            else:
                log_warning("⚠ GitHub API unreachable (offline mode)")

        # Create log directory
        LOG_DIR.mkdir(exist_ok=True)

        # Install git hooks
        install_hooks()

        # Set up git identity for commits
        self.setup_git_identity()

        # Check sync staleness (warn if behind ai_template)
        commits_behind = self.check_sync_staleness()
        if commits_behind is not None:
            if commits_behind == 0:
                log_info("✓ ai_template sync: current")
            elif commits_behind < 50:
                log_info(f"✓ ai_template sync: {commits_behind} commits behind")
            elif commits_behind < 100:
                log_warning(f"⚠ ai_template sync: {commits_behind} behind - sync?")
            else:
                # 100+ commits behind: create flag file for health monitoring
                msg = "sync recommended" if commits_behind < 200 else "STALE"
                log_warning(
                    f"⚠ ai_template sync: {commits_behind} commits behind - {msg}"
                )
                FLAGS_DIR.mkdir(exist_ok=True)
                (FLAGS_DIR / "sync_stale").write_text(f"{commits_behind}\n")

        # Check for existing instance
        if self.pid_file.exists():
            try:
                old_pid = int(self.pid_file.read_text().strip())
                # Check if process is still running
                os.kill(old_pid, 0)  # Doesn't kill, just checks
                log_error(f"ERROR: Another {self.mode} loop running (PID {old_pid})")
                log_error(f"  Stop it first or remove {self.pid_file}")
                sys.exit(1)
            except (ProcessLookupError, ValueError):
                # Process not running, clean up stale PID file
                self.pid_file.unlink()

        # Write our PID
        self.pid_file.write_text(str(os.getpid()))

        # Restore iteration counter
        if self.iteration_file.exists():
            try:
                self.iteration = int(self.iteration_file.read_text().strip())
                log_info(f"Resuming from iteration {self.iteration}")
            except ValueError:
                self.iteration = 1

        # Rotate logs
        self.status_manager.rotate_logs()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        # Track start time for status
        self._started_at = datetime.now(UTC).isoformat()
        self.status_manager.set_started_at(self._started_at)
        self.status_manager.write_status(self.iteration, "starting")

        # Sync from main on startup if enabled (multi-machine mode)
        # Only runs on zone branches, not on main
        sync_on_startup = self.config.get("sync_on_startup", True)
        if self.branch and sync_on_startup:
            sync_strategy = self.config.get("sync_strategy", "rebase")
            self._sync_from_main(sync_strategy)

        # Check for stale staged files from prior sessions (#995)
        # Default: warn only. Set staged_check_abort=True in config to block.
        staged_check_abort = self.config.get("staged_check_abort", False)
        check_stale_staged_files(abort=staged_check_abort)

        # Check for crash recovery
        self._recovery_context = self.checkpoint_manager.check_recovery()
        if self._recovery_context:
            self.iteration_runner.set_recovery_context(self._recovery_context)

        # Clean up stale file trackers from crashed workers (multi-worker mode)
        if self.worker_id is not None:
            repo_path = Path.cwd().resolve()
            cleaned = cleanup_stale_trackers(repo_path)
            if cleaned:
                log_info(f"✓ Cleaned {len(cleaned)} stale file tracker(s)")

            # Initialize file tracker with current uncommitted files
            if self.file_tracker:
                current_files = get_uncommitted_files()
                if current_files:
                    self.file_tracker.save(current_files)
                    log_info(
                        f"✓ File tracker initialized with {len(current_files)} file(s)"
                    )

        # Start memory watchdog if enabled (#1468)
        # Only on macOS (darwin) where memory_pressure command is available
        self._start_memory_watchdog()

        log_info("")
        log_info(f"Starting {self.mode} loop...")
        log_info("")

    def _start_memory_watchdog(self) -> None:
        """Start memory watchdog daemon for OOM prevention (#1468).

        Spawns memory_watchdog.py as a companion process that monitors system
        memory pressure and kills runaway AI-spawned processes before kernel panic.

        Only runs on macOS where memory_pressure command is available.
        Configurable via memory_watchdog_enabled (default: True) and
        memory_watchdog_threshold (default: "critical") in role config.

        Uses a PID file to ensure only one watchdog runs per machine.
        """
        # Only run on macOS
        if sys.platform != "darwin":
            return

        # Check if enabled in config (default: True)
        if not self.config.get("memory_watchdog_enabled", True):
            log_info("  Memory watchdog disabled in config")
            return

        # Find the watchdog script
        watchdog_script = Path("ai_template_scripts/memory_watchdog.py")
        if not watchdog_script.exists():
            log_warning("⚠ Memory watchdog script not found")
            return

        # Check if watchdog is already running (global PID file)
        watchdog_pid_file = Path.home() / ".ait_memory_watchdog.pid"
        if watchdog_pid_file.exists():
            try:
                existing_pid = int(watchdog_pid_file.read_text().strip())
                # Check if process is still running
                os.kill(existing_pid, 0)  # Just checks existence
                log_info(f"✓ Memory watchdog already running (PID {existing_pid})")
                return
            except (ProcessLookupError, ValueError, OSError):
                # Stale PID file, remove it
                try:
                    watchdog_pid_file.unlink()
                except OSError as e:
                    debug_swallow("unlink_watchdog_pid_stale", e)

        # Get threshold from config (default: critical)
        threshold = self.config.get("memory_watchdog_threshold", "critical")

        try:
            # Spawn watchdog in daemon mode
            # Redirect stdout/stderr to devnull to avoid cluttering terminal
            self._watchdog_process = subprocess.Popen(
                [
                    sys.executable,
                    str(watchdog_script),
                    "--daemon",
                    "--threshold",
                    threshold,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent process group
            )
            # Write PID file for other loopers to detect
            try:
                watchdog_pid_file.write_text(str(self._watchdog_process.pid))
            except OSError as e:
                debug_swallow("write_watchdog_pid", e)
            log_info(
                f"✓ Memory watchdog started (PID {self._watchdog_process.pid}, "
                f"threshold={threshold})"
            )
        except Exception as e:
            log_warning(f"⚠ Failed to start memory watchdog: {e}")
            self._watchdog_process = None

    def _stop_memory_watchdog(self) -> None:
        """Stop the memory watchdog daemon.

        Only stops the watchdog if this looper instance spawned it.
        Cleans up the global PID file to allow future loopers to spawn a new one.
        """
        if self._watchdog_process is None:
            return

        watchdog_pid_file = Path.home() / ".ait_memory_watchdog.pid"

        try:
            # Check if still running
            if self._watchdog_process.poll() is None:
                # Still running, terminate gracefully
                self._watchdog_process.terminate()
                try:
                    self._watchdog_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if doesn't respond
                    self._watchdog_process.kill()
                    self._watchdog_process.wait()
            # Clean up PID file
            if watchdog_pid_file.exists():
                try:
                    watchdog_pid_file.unlink()
                except OSError as e:
                    debug_swallow("unlink_watchdog_pid_stop", e)
            log_info("✓ Memory watchdog stopped")
        except Exception as e:
            log_warning(f"⚠ Error stopping memory watchdog: {e}")
        finally:
            self._watchdog_process = None

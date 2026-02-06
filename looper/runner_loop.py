# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_loop.py - Main run() loop orchestration.

Provides RunnerLoopMixin for LoopRunner:
- Main run() loop that iterates until STOP signal or error
- Iteration delay logic (restart_delay, error_delay)
- Legacy periodic sync handling (sync_interval_iterations)
- Checkpoint writes before/after iterations
- Uncommitted work size warnings
- Telemetry recording after each iteration

The run() method is the main entry point called by looper.py CLI.

See docs/looper.md "Execution Flow" for the iteration lifecycle diagram.
"""

from __future__ import annotations

__all__ = ["RunnerLoopMixin"]

from looper.config import load_role_config
from looper.file_tracker import get_uncommitted_files
from looper.iteration import extract_issue_numbers
from looper.log import log_info
from looper.sync import warn_uncommitted_work


class RunnerLoopMixin:
    """Main loop for autonomous AI iteration.

    Required attributes:
        running: bool
        config: dict
        mode: str
        branch: str | None
        iteration: int
        _working_issues: list[int]
        _prompt_template: str
        file_tracker: FileTracker | None
        iteration_file: Path
        checkpoint_manager: CheckpointManager
        status_manager: StatusManager
        issue_manager: IssueManager
        iteration_runner: IterationRunner
    """

    def run(self) -> None:
        """Main loop for autonomous AI iteration.

        REQUIRES:
        - self.mode in {"worker", "prover", "researcher", "manager"}
        - Config files exist: .claude/roles/shared.md, .claude/roles/{mode}.md
        - Git repository initialized in working directory
        - No other LoopRunner instance for same mode (PID file check in setup)

        ENSURES:
        - On return (normal or exception): cleanup() called
        - On return: subprocess termination attempted; warnings emitted if one persists
        - On return: PID file removed
        - On return: status written to worker_logs/{mode}_status.json
        - If self.running becomes False: graceful exit with best-effort cleanup
        - Each iteration increments self.iteration and persists to iteration file
        - Telemetry recorded for each iteration via record_iteration()
        """
        self.setup()

        try:
            while self.running:
                # Update heartbeat and detect wake from sleep
                # Must happen before STOP check so sleep detection informs expiry
                self._update_heartbeat()

                # Check for STOP file
                stop_file = self._check_stop_file()
                if stop_file:
                    log_info(f"\n*** {stop_file} detected - stopping ***")
                    # Extract reason from STOP file content before consuming
                    stop_path = self._stop_dir / stop_file
                    reason = self._get_stop_file_reason(stop_path)
                    if reason:
                        log_info(f"    Reason: {reason}")
                    # Log STOP event to session log (clean shutdown)
                    self._log_session_event("STOP", reason=reason, clean=True)
                    self._consume_stop_file(stop_file)
                    self.running = False
                    break

                # Reload config each iteration (allows live tuning)
                self.config, self._prompt_template = load_role_config(self.mode)
                self.status_manager.config = self.config
                self.iteration_runner.update_config(self.config, self._prompt_template)

                # Run pulse health check if due
                self.status_manager.run_pulse()

                # Periodic cleanup: remove workflow labels from closed issues (#911)
                # Runs every 10 iterations to catch issues auto-closed via Fixes commits
                cleanup_interval = self.config.get("cleanup_closed_interval", 10)
                if cleanup_interval > 0 and self.iteration % cleanup_interval == 0:
                    log_info("Pre-iteration: cleaning up closed issue labels...")
                    self.issue_manager.cleanup_closed_issues_labels()

                # Replay pending gh operations from change_log.json (#1846)
                # Ensures queued issues are synced even when no gh commands run
                try:
                    from ai_template_scripts.gh_post.queue import (
                        replay_pending_changes,
                    )

                    log_info("Pre-iteration: replaying pending changes...")
                    replay_pending_changes(max_per_call=3)
                except ImportError:
                    pass  # gh_post not available - skip replay

                # Sync from main before iteration (multi-machine mode)
                # Uses SyncConfig from .looper_config.json if trigger=iteration_start
                self._pre_iteration_sync()

                # Legacy periodic sync (for backwards compatibility)
                sync_interval = self.config.get("sync_interval_iterations", 0)
                if self.branch and sync_interval > 0:
                    if self.iteration % sync_interval == 0:
                        sync_strategy = self.config.get("sync_strategy", "rebase")
                        self._sync_from_main(sync_strategy)

                # Write checkpoint before iteration (for crash recovery)
                # Note: Tool call log NOT included - cleared when iteration starts.
                # Only the post-iteration checkpoint has valid tool call state.
                # See designs/2026-02-01-tool-call-checkpointing.md
                self.checkpoint_manager.write(
                    iteration=self.iteration,
                    phase=self.iteration_runner.current_phase,
                    working_issues=self._working_issues,
                )

                # Check for accumulated uncommitted work (#1015)
                # Warns if uncommitted changes exceed threshold (default 100 lines)
                uncommitted_threshold = self.config.get(
                    "uncommitted_warn_threshold", 100
                )
                if uncommitted_threshold > 0:
                    warn_uncommitted_work(threshold=uncommitted_threshold)

                # Run main iteration
                log_info("Pre-iteration: building iteration context...")
                result = self.run_iteration(audit_round=0)
                exit_code = result.exit_code
                start_time = result.start_time
                ai_tool = result.ai_tool
                session_id = result.session_id
                selected_codex_model = result.codex_model

                if not self.running:
                    break

                # Check if session committed
                session_committed = self.check_session_success(start_time)

                # Extract working issues from last commit
                self._working_issues = []
                if session_committed:
                    last_commit = (
                        self.iteration_runner._prompt_builder._get_last_commit_message()
                    )
                    if last_commit:
                        self._working_issues = extract_issue_numbers(last_commit)

                # Commit uncommitted changes as WIP if needed
                if not session_committed:
                    wip_committed = self.commit_uncommitted_changes()
                    if wip_committed:
                        session_committed = True

                # Update checkpoint with final tool call log state before clearing (#1625)
                # This captures tool calls for recovery if crash happens between iterations
                tool_log_path, tool_completed, tool_seq = (
                    self.iteration_runner.get_tool_call_log_info()
                )
                self.checkpoint_manager.write(
                    iteration=self.iteration,
                    phase=self.iteration_runner.current_phase,
                    working_issues=self._working_issues,
                    tool_call_log_path=tool_log_path,
                    tool_calls_completed=tool_completed,
                    tool_calls_last_seq=tool_seq,
                )

                # Clear checkpoint after successful commit (clean exit)
                if session_committed:
                    self.checkpoint_manager.clear()

                # Auto-create PR for zone branch if enabled (multi-machine mode)
                auto_pr = self.config.get("auto_pr", True)
                if self.branch and session_committed and auto_pr:
                    self._ensure_pr_exists()

                # Update pending metrics with main iteration outcome
                if self.iteration_runner.pending_metrics:
                    self.iteration_runner.pending_metrics.committed = session_committed
                    self.iteration_runner.pending_metrics.working_issues = (
                        self._working_issues.copy()
                    )

                # Determine if audit should run
                (
                    should_audit,
                    do_audit_issues,
                    audit_max_rounds,
                    audit_min_issues,
                ) = self._should_run_audit(exit_code, ai_tool)

                # Run audit rounds if needed
                audit_committed = False
                audit_start_time = 0.0
                audit_rounds_run = 0

                if should_audit:
                    (
                        audit_committed,
                        audit_start_time,
                        audit_start_times,
                    ) = self._run_audit_loop(
                        session_id,
                        ai_tool,
                        selected_codex_model,
                        do_audit_issues,
                        audit_max_rounds,
                        audit_min_issues,
                    )
                    audit_rounds_run = len(audit_start_times)

                # Calculate delay, handling crashes and [INCOMPLETE]
                delay = self._calculate_delay(
                    exit_code,
                    ai_tool,
                    session_committed,
                    start_time,
                    audit_committed,
                    audit_start_time,
                )

                # Finalize telemetry and process checkboxes
                self._finalize_iteration(
                    session_committed,
                    audit_committed,
                    audit_rounds_run,
                    start_time,
                    audit_start_time,
                )

                # Update file tracker with current uncommitted files (multi-worker mode)
                if self.file_tracker:
                    current_files = get_uncommitted_files()
                    self.file_tracker.save(current_files)

                # Wait before next iteration
                self._wait_interruptible(delay)

        finally:
            self.cleanup()

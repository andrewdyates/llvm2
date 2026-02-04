# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_audit.py - Audit workflow for do-audit → needs-review.

Provides RunnerAuditMixin for LoopRunner:
- run_audit_rounds() for self-audit passes after main iteration
- do-audit → needs-review label transitions
- System health check execution between rounds
- Urgent handoff detection for fast role spawning

Audit rounds resume the same AI session (--resume) to maintain context.
Maximum rounds controlled by audit_max_rounds config (default 5).

See docs/looper.md "Audit Configuration" for tuning parameters.
"""

from __future__ import annotations

__all__ = ["RunnerAuditMixin"]

from looper.config import load_timeout_config
from looper.context import (
    check_urgent_handoff,
    get_do_audit_issues,
    run_system_health_check,
    transition_audit_to_review,
)
from looper.context.issue_context import IterationIssueCache
from looper.log import log_info, log_warning
from looper.telemetry import check_consecutive_abort_alert, record_iteration


class RunnerAuditMixin:
    """Post-iteration audit logic.

    Required attributes:
        config: dict
        running: bool
        mode: str
        iteration: int
        iteration_file: Path
        status_manager: StatusManager
        issue_manager: IssueManager
        iteration_runner: IterationRunner
    """

    def _should_run_audit(
        self, exit_code: int, ai_tool: str
    ) -> tuple[bool, list[dict], int, int]:
        """Determine if audit should run and get audit configuration.

        Returns:
            (should_audit, do_audit_issues, audit_max_rounds, audit_min_issues)
        """
        auto_audit = self.config.get("auto_audit", True)
        audit_max_rounds = self.config.get("audit_max_rounds", 5)
        audit_min_issues = self.config.get("audit_min_issues", 3)

        if not auto_audit:
            return False, [], audit_max_rounds, audit_min_issues

        # Clear issue cache before checking do-audit issues (#1677)
        # Ensures label changes from main iteration are visible
        IterationIssueCache.clear()

        do_audit_result = get_do_audit_issues()
        if not do_audit_result.ok:
            error = do_audit_result.error or "unknown error"
            log_warning(f"  → get_do_audit_issues failed: {error}")
            return False, [], audit_max_rounds, audit_min_issues

        do_audit_issues = do_audit_result.value or []
        should_audit = (
            len(do_audit_issues) > 0
            and exit_code == 0
            and self.running
            and not self._check_stop_file()
        )

        # Codex: max 1 audit round
        if ai_tool == "codex":
            audit_max_rounds = min(audit_max_rounds, 1)

        return should_audit, do_audit_issues, audit_max_rounds, audit_min_issues

    def _run_single_audit_round(
        self,
        audit_round: int,
        session_id: str | None,
        ai_tool: str,
        selected_codex_model: str | None,
        audit_issue_list: str,
        audit_min_issues: int,
    ) -> tuple[int, float, bool, bool]:
        """Run a single audit round.

        Returns:
            (exit_code, start_time, round_committed, should_stop)
        """
        # Clear cache before re-checking do-audit (#1677)
        # Ensures label changes from previous audit round are visible
        IterationIssueCache.clear()

        # Re-check for do-audit (may have been removed)
        current_do_audit_result = get_do_audit_issues()
        if not current_do_audit_result.ok:
            error = current_do_audit_result.error or "unknown error"
            log_warning(f"  → get_do_audit_issues failed during audit: {error}")
            return 0, 0.0, False, True

        current_do_audit = current_do_audit_result.value or []
        if not current_do_audit:
            log_info(f"\n✓ {self.iteration}.{audit_round}: do-audit gone")
            return 0, 0.0, False, True

        # Print audit header
        log_info("")
        header = (
            f"--- {self.iteration}.{audit_round}: Audit "
            f"{audit_issue_list} - find {audit_min_issues}+ "
            "issues or [DONE] ---"
        )
        log_info(header)

        # Run audit iteration
        result = self.run_iteration(
            audit_round=audit_round,
            resume_session_id=session_id,
            force_ai_tool=ai_tool,
            force_codex_model=selected_codex_model,
        )
        audit_exit_code = result.exit_code
        audit_start_time = result.start_time
        audit_ai_tool = result.ai_tool
        session_id = result.session_id

        if not self.running:
            return audit_exit_code, audit_start_time, False, True

        # Check STOP file between audit rounds
        stop_file = self._check_stop_file()
        if stop_file:
            log_info(f"\n*** {stop_file} file detected - stopping audits ***")
            self._consume_stop_file(stop_file)
            self.running = False
            return audit_exit_code, audit_start_time, False, True

        # Check if this audit round committed
        round_committed = self.check_session_success(audit_start_time)

        # Log exit if non-zero
        # Pass graceful_stop=True if we're stopping due to STOP file (#1988)
        if audit_exit_code != 0:
            self.status_manager.log_exit(
                self.iteration,
                audit_exit_code,
                audit_ai_tool,
                round_committed,
                graceful_stop=not self.running,
            )

        return audit_exit_code, audit_start_time, round_committed, False

    def _handle_done_marker(
        self, audit_round: int, audit_start_time: float, audit_issue_nums: list[int]
    ) -> bool:
        """Handle [DONE] marker in audit commit.

        Returns:
            True if audit should stop (either DONE processed or health check failed).
        """
        log_info(f"✓ {self.iteration}.{audit_round}: [DONE]")

        health_timeout = load_timeout_config().get("health_check", 120)
        system_health = run_system_health_check(timeout_sec=health_timeout)
        if system_health.skipped:
            pass  # Script not present - proceed with audit transition
        elif not system_health.ok:
            error = system_health.error or "unknown error"
            log_warning("  → health error; keeping do-audit")
            log_warning(f"  → system_health_check.py error: {error}")
            return True

        if system_health.value and system_health.value[0] != 0:
            log_warning("  → health failed; keeping do-audit")
            output = system_health.value[1]
            if output:
                lines = output.splitlines()
                max_lines = 20
                trimmed = "\n".join(lines[:max_lines])
                if len(lines) > max_lines:
                    trimmed += "\n... (truncated)"
                log_warning("  → system_health_check.py output:\n" + trimmed)
            return True

        # Transition do-audit → needs-review for audited issues
        for issue_num in audit_issue_nums:
            if not isinstance(issue_num, int):
                continue
            transition_result = transition_audit_to_review(issue_num)
            if transition_result.ok and transition_result.value:
                log_info(f"  → #{issue_num}: do-audit → needs-review")
            else:
                err = transition_result.error or "unknown"
                log_warning(f"  → #{issue_num}: failed ({err})")

        return True  # Stop after successful DONE

    def _run_audit_loop(
        self,
        session_id: str | None,
        ai_tool: str,
        selected_codex_model: str | None,
        do_audit_issues: list[dict],
        audit_max_rounds: int,
        audit_min_issues: int,
    ) -> tuple[bool, float, list[float]]:
        """Run follow-up audit rounds.

        Returns:
            (audit_committed, earliest_audit_start_time, audit_start_times)
        """
        audit_committed = False
        audit_start_times: list[float] = []

        # Skip session resume if main iteration exceeded token limit (#1881)
        # This prevents accumulating unbounded context in audit rounds
        if self.iteration_runner.should_skip_resume():
            log_warning(
                "⚠️  Skipping audit session resume due to token growth (#1881). "
                "Audit rounds will start fresh sessions."
            )
            session_id = None

        # Build issue list for display
        audit_issue_nums = [i["number"] for i in do_audit_issues]
        audit_issue_list = ", ".join(f"#{n}" for n in audit_issue_nums[:3])
        if len(audit_issue_nums) > 3:
            audit_issue_list += f" (... and {len(audit_issue_nums) - 3} more)"

        for audit_round in range(1, audit_max_rounds + 1):
            (
                audit_exit_code,
                audit_start_time,
                round_committed,
                should_stop,
            ) = self._run_single_audit_round(
                audit_round,
                session_id,
                ai_tool,
                selected_codex_model,
                audit_issue_list,
                audit_min_issues,
            )

            if audit_start_time > 0:
                audit_start_times.append(audit_start_time)

            if should_stop:
                break

            if round_committed:
                audit_committed = True
                # Check for [DONE] marker
                if self.check_done_commit(audit_start_time):
                    if self._handle_done_marker(
                        audit_round, audit_start_time, audit_issue_nums
                    ):
                        break
                log_info(f"✓ {self.iteration}.{audit_round}: issues fixed")
            else:
                # No commit - session ended without completing
                log_info(f"✓ {self.iteration}.{audit_round}: no commit")
                break

        earliest_start = min(audit_start_times) if audit_start_times else 0.0
        return audit_committed, earliest_start, audit_start_times

    def _calculate_delay(
        self,
        exit_code: int,
        ai_tool: str,
        session_committed: bool,
        main_start_time: float,
        audit_committed: bool,
        audit_start_time: float,
    ) -> int:
        """Calculate delay before next iteration, handling errors and [INCOMPLETE].

        Returns:
            Delay in seconds.
        """
        if exit_code != 0:
            # Pass graceful_stop=True if we're stopping due to STOP file (#1988)
            self.status_manager.log_exit(
                self.iteration,
                exit_code,
                ai_tool,
                session_committed,
                graceful_stop=not self.running,
            )
            delay = (
                self.config["restart_delay"]
                if session_committed
                else self.config["error_delay"]
            )
        else:
            delay = self.config["restart_delay"]

        # Check [INCOMPLETE] marker - continue immediately
        main_incomplete = session_committed and self.check_incomplete_commit(
            main_start_time
        )
        audit_incomplete = audit_committed and self.check_incomplete_commit(
            audit_start_time
        )
        if main_incomplete or audit_incomplete:
            log_info("")
            log_info("📝 Session marked [INCOMPLETE] - continuing immediately")
            delay = 0

        # Check for urgent-handoff issues targeting this role (#1645)
        # Skip delay if another role urgently needs this role's attention
        if delay > 0:
            handoff_result = check_urgent_handoff(self.mode)
            if handoff_result.ok and handoff_result.value:
                issue_nums = handoff_result.value
                log_info("")
                log_info(
                    f"🚨 Urgent handoff detected (#{', #'.join(map(str, issue_nums))}) - "
                    "skipping delay"
                )
                delay = 0

        return delay

    def _finalize_iteration(
        self,
        session_committed: bool,
        audit_committed: bool,
        audit_rounds_run: int,
        main_start_time: float,
        audit_start_time: float,
    ) -> None:
        """Finalize telemetry and process checkboxes."""
        # Update pending metrics
        if self.iteration_runner.pending_metrics:
            self.iteration_runner.pending_metrics.audit_committed = audit_committed
            self.iteration_runner.pending_metrics.audit_rounds_run = audit_rounds_run
            main_incomplete = session_committed and self.check_incomplete_commit(
                main_start_time
            )
            audit_incomplete = audit_committed and self.check_incomplete_commit(
                audit_start_time
            )
            self.iteration_runner.pending_metrics.incomplete_marker = (
                main_incomplete or audit_incomplete
            )
            self.iteration_runner.pending_metrics.done_marker = (
                audit_committed and self.check_done_commit(audit_start_time)
                if audit_start_time > 0
                else False
            )
            # Collect tool call metrics after all rounds complete (#1630)
            # See: designs/2026-02-01-tool-call-metrics-format.md
            tool_count, tool_types, tool_durations = (
                self.iteration_runner.get_tool_call_stats()
            )
            self.iteration_runner.pending_metrics.tool_call_count = tool_count
            self.iteration_runner.pending_metrics.tool_call_types = (
                tool_types if tool_types else None
            )
            self.iteration_runner.pending_metrics.tool_call_duration_ms = (
                tool_durations if tool_durations else None
            )
            record_iteration(self.iteration_runner.pending_metrics)
            self.iteration_runner.pending_metrics = None

        # Check for consecutive early abort alert (#1644)
        check_consecutive_abort_alert()

        # Convert unchecked checkboxes to new issues (Worker only)
        if self.mode == "worker" and session_committed:
            self.issue_manager.process_unchecked_checkboxes()

        # Increment and persist iteration
        self.iteration += 1
        self.iteration_file.parent.mkdir(exist_ok=True)
        self.iteration_file.write_text(str(self.iteration))

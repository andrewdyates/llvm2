# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_iteration.py - Single iteration execution delegation.

Provides RunnerIterationMixin for LoopRunner:
- run_iteration() method that delegates to IterationRunner
- Git add/commit/push operations after AI completes
- Issue number extraction from commit messages
- Handling of [INCOMPLETE] and [DONE] markers

Parameters for run_iteration():
    audit_round: Current audit round (0 = main iteration, 1+ = audit passes)
    resume_session_id: Optional session ID to resume (for --resume support)
    force_ai_tool: Override AI tool selection (claude, codex, dasher)
    force_codex_model: Override Codex model when using codex tool
    audit_min_issues_override: Override audit min-issues prompt value
    audit_max_rounds_override: Override audit max-rounds prompt value

This mixin is the bridge between the loop orchestration (runner_loop.py)
and the actual AI subprocess execution (iteration.py).

See docs/looper.md "Execution Flow" for where this fits in the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["RunnerIterationMixin"]

from looper.log import log_info, log_warning
from looper.subprocess_utils import run_git_command

if TYPE_CHECKING:
    from looper.iteration import IterationResult


class RunnerIterationMixin:
    """Single-iteration execution helpers.

    Required attributes:
        iteration: int
        mode: str
        iteration_runner: IterationRunner
        issue_manager: IssueManager
        _working_issues: list[int]
    """

    def run_iteration(
        self,
        audit_round: int = 0,
        resume_session_id: str | None = None,
        force_ai_tool: str | None = None,
        force_codex_model: str | None = None,
        audit_min_issues_override: int | None = None,
        audit_max_rounds_override: int | None = None,
    ) -> IterationResult:
        """Run a single AI iteration via IterationRunner.

        Args:
            audit_round: Current audit round. 0 = main iteration, 1+ = audit
                passes that resume the session for self-review.
            resume_session_id: Optional session ID to resume. Used for audit
                rounds to maintain context from the main iteration.
            force_ai_tool: Override AI tool selection. One of "claude", "codex",
                or "dasher". If None, uses config default.
            force_codex_model: Override Codex model when using codex tool.
                If None, uses config default.
            audit_min_issues_override: Optional per-run override for audit
                min-issues prompt requirement.
            audit_max_rounds_override: Optional per-run override for audit
                max-rounds prompt display value.

        Returns:
            IterationResult with exit_code, start_time, ai_tool, session_id,
            and codex_model fields.
        """
        return self.iteration_runner.run_iteration(
            iteration=self.iteration,
            audit_round=audit_round,
            resume_session_id=resume_session_id,
            force_ai_tool=force_ai_tool,
            force_codex_model=force_codex_model,
            working_issues=self._working_issues if audit_round > 0 else None,
            audit_min_issues_override=audit_min_issues_override,
            audit_max_rounds_override=audit_max_rounds_override,
        )

    def check_session_success(self, start_time: float) -> bool:
        """Check if the AI session made a commit during this iteration.

        A session that committed is considered successful even if the exit code
        is non-zero (e.g., EPIPE at the end after work completed).
        """
        # Get commits from the last iteration window
        # Use --since with a timestamp slightly before start_time
        since_time = int(start_time) - 60  # 1 minute buffer
        # Match git author format: {project}-{role}-{iteration}
        result = run_git_command(
            [
                "log",
                "--oneline",
                f"--since={since_time}",
                "--author=-worker-",
                "--author=-prover-",
                "--author=-researcher-",
                "--author=-manager-",
                "-1",
            ],
            timeout=5,
        )
        # If we got a commit hash, the session was successful
        return result.ok and bool((result.value or "").strip())

    def check_incomplete_commit(self, start_time: float) -> bool:
        """Check if the last commit was marked [INCOMPLETE].

        If so, the next session should continue immediately (0 delay).
        """
        return self._check_commit_marker(start_time, "[INCOMPLETE]")

    def check_done_commit(self, start_time: float) -> bool:
        """Check if the last commit was marked [DONE].

        Used by audit loop to detect when AI signals no more issues to find.
        """
        return self._check_commit_marker(start_time, "[DONE]")

    def _check_commit_marker(self, start_time: float, marker: str) -> bool:
        """Check if the last commit contains a specific marker."""
        since_time = int(start_time) - 60
        result = run_git_command(
            [
                "log",
                "--format=%s",
                f"--since={since_time}",
                "--author=-worker-",
                "--author=-prover-",
                "--author=-researcher-",
                "--author=-manager-",
                "-1",
            ],
            timeout=5,
        )
        if result.ok and result.value and result.value.strip():
            return marker in result.value
        return False

    @staticmethod
    def _parse_manager_report_files(changes: str) -> list[str]:
        """Extract manager report filenames from git status --porcelain output.

        Returns filenames matching the reports/*manager-iter* pattern.
        Used by both _has_manager_iteration_reports and _commit_manager_reports.
        """
        files = []
        for line in changes.strip().split("\n"):
            if not line:
                continue
            # status --porcelain format: XY filename (or XY -> newname for renames)
            # Strip status indicators and check filename
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[-1]  # Last part is the filename
                if "reports/" in filename and "manager-iter" in filename:
                    files.append(filename)
        return files

    def _has_manager_iteration_reports(self, changes: str) -> bool:
        """Check if uncommitted changes include manager iteration reports.

        Manager iteration reports (reports/*manager-iter*.md) should be
        committed by the MANAGER loop, not as WIP or by another role (#1481).
        """
        return len(self._parse_manager_report_files(changes)) > 0

    def commit_uncommitted_changes(self) -> bool:
        """Check for and commit any uncommitted changes after session ends.

        Returns True if changes were committed, False otherwise.
        Prevents work from being lost between sessions.

        For manager role: If uncommitted changes include manager iteration
        reports, commits them with proper [M] attribution (#1481).
        """
        # Check for uncommitted changes
        result = run_git_command(["status", "--porcelain"], timeout=5)
        if not result.ok or not (result.value or "").strip():
            return False  # No changes or error

        changes = result.value or ""

        # Manager-specific: commit iteration reports with proper attribution (#1481)
        if self.mode == "manager" and self._has_manager_iteration_reports(changes):
            return self._commit_manager_reports(changes)

        # There are uncommitted changes - commit them as WIP
        log_info("\n⚠️  Uncommitted changes detected - auto-committing as WIP")

        # Create WIP commit
        commit_msg = (
            f"[{self.mode[0].upper()}]{self.iteration}: WIP - session ended "
            "with uncommitted changes\n\n## Note\n"
            "Auto-committed by looper.py to prevent work loss.\n"
            "Review and amend or continue in next session."
        )

        # Non-Worker roles must not stage files — they produce text, not code.
        # Using git add -A would sweep in Worker's staged-but-uncommitted files,
        # causing cross-role staging contamination (#2812, #2794, #2729, #2405).
        # Use --allow-empty --only to create the WIP commit without any files.
        if self.mode != "worker":
            log_info(
                "Non-Worker WIP: using --allow-empty --only to avoid "
                "staging contamination (#2812)"
            )
            commit_result = run_git_command(
                ["commit", "--allow-empty", "--only", "-m", commit_msg],
                timeout=30,
            )
        else:
            # Worker role: stage all changes as before
            add_result = run_git_command(["add", "-A"], timeout=10)
            if not add_result.ok:
                error = add_result.error or "unknown error"
                log_warning(f"Warning: git add failed: {error}")
                return False
            commit_result = run_git_command(
                ["commit", "-m", commit_msg], timeout=30
            )

        if commit_result.ok:
            log_info("✓ WIP commit created")

            # Get commit hash for issue
            hash_result = run_git_command(["rev-parse", "--short", "HEAD"], timeout=5)
            commit_hash = (
                (hash_result.value or "").strip() if hash_result.ok else "unknown"
            )

            # Push the commit
            push_result = run_git_command(["push"], timeout=60)
            if not push_result.ok:
                error = push_result.error or "unknown error"
                log_warning(f"Warning: push failed: {error}")

            # Create follow-up issue to track WIP (#266)
            self.issue_manager.create_wip_followup(commit_hash, self.iteration)

            return True  # Commit succeeded even if push failed
        error = commit_result.error or "unknown error"
        log_warning(f"⚠️  WIP commit failed: {error}")
        return False

    def _commit_manager_reports(self, changes: str) -> bool:
        """Commit manager iteration reports with proper [M] attribution (#1481).

        Manager iteration reports should be committed by the manager loop,
        not as WIP or by another role. This preserves process ownership.

        Args:
            changes: git status --porcelain output (reused from caller to
                avoid redundant git status call, #2812 self-audit).

        Returns True if commit succeeded, False otherwise.
        """
        log_info("\n📋 Manager iteration report detected - committing with [M] tag")

        # Stage ONLY manager report files, not all changes (#2812).
        # Using git add -A would sweep in Worker's staged-but-uncommitted files.
        report_files = self._parse_manager_report_files(changes)
        if not report_files:
            log_warning("Warning: no manager report files found to stage")
            return False

        add_result = run_git_command(["add"] + report_files, timeout=10)
        if not add_result.ok:
            error = add_result.error or "unknown error"
            log_warning(f"Warning: git add failed: {error}")
            return False

        # Create commit with proper [M] attribution (not WIP)
        # Use --only to commit ONLY the files we explicitly staged above,
        # excluding any other staged files in the worktree (#2812).
        commit_msg = (
            f"[M]{self.iteration}: Auto-commit manager iteration report\n\n"
            "## Note\n"
            "Manager session ended with uncommitted iteration report.\n"
            "Auto-committed by looper.py to preserve manager attribution (#1481)."
        )
        commit_result = run_git_command(
            ["commit", "--only"] + report_files + ["-m", commit_msg],
            timeout=30,
        )

        if commit_result.ok:
            log_info("✓ Manager report committed")

            # Push the commit
            push_result = run_git_command(["push"], timeout=60)
            if not push_result.ok:
                error = push_result.error or "unknown error"
                log_warning(f"Warning: push failed: {error}")

            return True  # Commit succeeded even if push failed

        error = commit_result.error or "unknown error"
        log_warning(f"⚠️  Manager report commit failed: {error}")
        return False

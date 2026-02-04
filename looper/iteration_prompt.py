# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Prompt building utilities for the looper iteration runner.

This module contains:
- Module-level helper functions for prompt construction
- Constants for issue pattern matching
- PromptBuilder class for assembling iteration prompts

Split from iteration.py for maintainability (#1804).
See: designs/2026-02-01-iteration-split.md
"""

import re
from pathlib import Path
from typing import Any

from looper.config import ROLES_DIR
from looper.context import (
    get_issues_by_numbers,
    get_sampled_issues,
    run_session_start_commands,
)
from looper.log import log_info
from looper.result import format_result
from looper.rotation import get_rotation_focus
from looper.subprocess_utils import run_git_command

__all__ = [
    "build_audit_prompt",
    "extract_issue_numbers",
    "PromptBuilder",
    "NO_ISSUES_PATTERNS",
    "WORKER_PHASE_PRIORITIES",
    "_has_no_issues",
    "_phase_has_matching_issues",
    "_get_fallback_phase",
]

# Patterns indicating no actionable issues in gh_issues output.
# Checked to trigger early abort for Worker (rotation-based roles skip this check).
# NOTE: Rotation roles (Manager, Prover, Researcher) skip abort even if gh_issues
# matches these patterns - their rotation phases ARE their work.
NO_ISSUES_PATTERNS = (
    "(no open issues)",
    "(local mode",  # local mode - GitHub API disabled
    "gh issue list failed:",  # API failure - no valid issue data
)

# Worker phase to issue priority mapping (#1643).
# Defines which P-levels each rotation phase should work on.
WORKER_PHASE_PRIORITIES: dict[str, tuple[str, ...]] = {
    "high_priority": ("[P0]", "[P1]"),
    "normal_work": ("[P2]",),
    "quality": ("[P3]",),
}


def build_audit_prompt(
    round_num: int,
    min_issues: int,
    max_rounds: int,
    last_commit: str | None = None,
) -> str:
    """Build audit prompt for a specific round.

    REQUIRES: round_num >= 1 (1-indexed round number)
    REQUIRES: min_issues >= 0
    REQUIRES: max_rounds >= round_num
    ENSURES: result contains round_num and max_rounds in header
    ENSURES: result contains min_issues requirement
    ENSURES: If last_commit provided, result includes commit context
    """
    context = ""
    if last_commit:
        # Provide context about what was just done (for tools without session resume)
        context = f"""
## Last Commit (your recent work)
```
{last_commit}
```
"""
    return f"""
{context}
## FOLLOW-UP AUDIT (Round {round_num}/{max_rounds})

Self-audit the work just completed. Find at least {min_issues} issues.

If you find issues, fix them and commit. If you find more, fix more.
If you cannot find at least {min_issues}, include [DONE] in your commit
message and explain why you cannot find more issues.
"""


def _has_no_issues(gh_issues: str) -> bool:
    """Check if gh_issues content indicates no actionable issues.

    Used for fail-fast early abort when there's no work to do (#1641).

    REQUIRES: gh_issues is a string
    ENSURES: Returns True if gh_issues matches any NO_ISSUES_PATTERNS
    ENSURES: Returns False if gh_issues has issue content to work on
    """
    return any(pattern in gh_issues for pattern in NO_ISSUES_PATTERNS)


def _phase_has_matching_issues(phase: str | None, gh_issues: str) -> bool:
    """Check if gh_issues contains issues matching the given Worker phase.

    Used for phase fallback when current phase has no actionable issues (#1643).

    REQUIRES: gh_issues is a string
    ENSURES: Returns True if gh_issues contains any priority matching the phase
    ENSURES: Returns True for None/unknown phases (no filtering)
    ENSURES: Returns True if gh_issues is empty/no-issues (let fail-fast handle)
    """
    if not phase or phase not in WORKER_PHASE_PRIORITIES:
        return True  # Unknown phase - don't filter

    if _has_no_issues(gh_issues):
        return True  # Let fail-fast handle empty issues

    priorities = WORKER_PHASE_PRIORITIES[phase]
    return any(p in gh_issues for p in priorities)


def _get_fallback_phase(
    current_phase: str | None,
    gh_issues: str,
    phases: list[str],
) -> tuple[str | None, str | None]:
    """Find a fallback phase with matching issues, or None if current is best.

    Returns (fallback_phase, fallback_reason) if fallback occurred,
    or (None, None) if current phase is fine.

    REQUIRES: phases is the ordered list of rotation phases
    ENSURES: If fallback needed, returns (phase, reason) where phase in phases
    ENSURES: If current phase has issues or no fallback possible, returns (None, None)
    """
    if not current_phase or current_phase not in WORKER_PHASE_PRIORITIES:
        return None, None  # Non-worker or unknown phase

    if _phase_has_matching_issues(current_phase, gh_issues):
        return None, None  # Current phase has issues

    # Try phases in order (respects weight ordering from rotation selection)
    for phase in phases:
        if phase == current_phase:
            continue  # Already tried
        if phase not in WORKER_PHASE_PRIORITIES:
            continue  # Not a Worker priority phase
        if _phase_has_matching_issues(phase, gh_issues):
            return phase, f"no {current_phase} issues, falling back to {phase}"

    # No phase has matching issues - let any issue be worked on
    # This handles edge cases like only having do-audit or in-progress issues
    if not _has_no_issues(gh_issues):
        return "any", f"no {current_phase} issues, working any available issue"

    return None, None  # No issues at all (fail-fast will handle)


def extract_issue_numbers(commit_msg: str) -> list[int]:
    """Extract issue numbers from a commit message.

    Extracts issue numbers from patterns indicating active work on an issue.
    Intentionally excludes Reopens/Unclaims - these are state changes, not work indicators.

    REQUIRES: commit_msg is a string (may be empty)
    ENSURES: All elements in result are positive integers
    ENSURES: Order matches appearance in commit_msg
    ENSURES: Only extracts from working patterns: Fixes/Part of/Re:/Claims #N
    ENSURES: Does NOT extract from state-change patterns: Reopens/Unclaims #N
    """
    # Case-insensitive to match auto-fixed "fixes" from commit-msg-hook
    return [
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:Fixes|Part of|Re:|Claims) #(\d+)", commit_msg, re.IGNORECASE
        )
    ]


class PromptBuilder:
    """Build prompt replacements for iteration context injection.

    Encapsulates the logic for building prompt context including:
    - Issue sampling and formatting
    - Rotation phase selection
    - Recovery context injection
    - Audit vs main iteration handling
    """

    def __init__(
        self,
        mode: str,
        config: dict[str, Any],
        worker_id: int | None = None,
    ) -> None:
        self.mode = mode
        self.config = config
        self.worker_id = worker_id
        self._recovery_context_str: str = ""

    def update_config(self, config: dict[str, Any]) -> None:
        """Refresh config for the next iteration."""
        self.config = config

    def set_recovery_context(self, context_str: str) -> None:
        """Store recovery context string for injection into next prompt.

        REQUIRES: context_str is a valid string (from RecoveryContext.to_prompt())
        ENSURES: _recovery_context_str is set for next build
        """
        self._recovery_context_str = context_str

    def build_replacements(
        self,
        iteration: int,
        audit_round: int,
        working_issues: list[int] | None,
        current_phase: str | None,
    ) -> tuple[dict[str, str], str | None]:
        """Build injection replacements and selected phase.

        REQUIRES: iteration >= 1
        REQUIRES: audit_round >= 0
        ENSURES: result[0] contains all required keys for prompt injection
        ENSURES: result[1] is selected phase string or None
        ENSURES: If audit_round > 0, minimal context (audit mode)
        ENSURES: If audit_round == 0, full context from run_session_start_commands
        """
        is_audit = audit_round > 0
        if is_audit:
            return self._build_audit_replacements(working_issues, current_phase)
        return self._build_main_replacements(iteration, current_phase)

    def _build_audit_replacements(
        self,
        working_issues: list[int] | None,
        current_phase: str | None,
    ) -> tuple[dict[str, str], str | None]:
        """Build minimal context for audit rounds."""
        # Minimal context for audit - just git_log and working issue
        git_log = "(see resumed session)"
        result = run_git_command(["log", "--oneline", "-3"], timeout=5)
        if result.ok and result.value:
            git_log = result.value.strip()

        # Get only the working issue(s), not all issues
        # Use batch fetch for efficiency (#1178)
        issues = working_issues or []
        if issues:
            issue_strs = get_issues_by_numbers(issues[:3])  # Max 3 working issues
            issue_lines = [s for s in issue_strs.values() if s]
            gh_issues = "\n".join(issue_lines) if issue_lines else "(no working issue)"
        else:
            gh_issues = "(no working issue tracked)"

        audit_min_issues = self.config.get("audit_min_issues", 3)
        replacements = {
            "git_log": git_log,
            "gh_issues": gh_issues,
            "last_directive": "",  # Session resume has full context
            "other_feedback": "",
            "role_mentions": "",
            "system_status": "",
            "audit_data": "",
            "rotation_focus": "",
            "audit_min_issues": str(audit_min_issues),
            "recovery_context": "",  # No recovery during audit (session resumed)
        }
        return replacements, current_phase

    def _build_main_replacements(
        self,
        iteration: int,
        current_phase: str | None,
    ) -> tuple[dict[str, str], str | None]:
        """Build full context for main iterations."""
        sampled_issues = get_sampled_issues(self.mode)
        gh_issues = format_result(sampled_issues)

        # Calculate rotation focus if this role has rotation
        rotation_type = self.config.get("rotation_type", "")
        rotation_phases = self.config.get("rotation_phases", [])
        phase_data = self.config.get("phase_data", {})
        freeform_frequency = self.config.get("freeform_frequency", 3)
        force_phase = self.config.get("force_phase")
        starvation_hours = self.config.get("starvation_hours", 24)
        rotation_focus, selected_phase = get_rotation_focus(
            iteration=iteration,
            rotation_type=rotation_type,
            phases=rotation_phases,
            phase_data=phase_data,
            role=self.mode,
            freeform_frequency=freeform_frequency,
            force_phase=force_phase,
            starvation_hours=starvation_hours,
        )

        # Build recovery context if present (from crash recovery)
        recovery_context_str = self._recovery_context_str
        self._recovery_context_str = ""  # Clear after use

        if _has_no_issues(gh_issues):
            audit_min_issues = self.config.get("audit_min_issues", 3)
            replacements = {
                "git_log": "(skipped: no issues assigned)",
                "gh_issues": gh_issues if gh_issues else "(no open issues)",
                "last_directive": "",
                "other_feedback": "",
                "role_mentions": "",
                "system_status": "",
                "audit_data": "",
                "rotation_focus": rotation_focus,
                "audit_min_issues": str(audit_min_issues),
                "recovery_context": recovery_context_str,
            }
            return replacements, selected_phase

        session_results = run_session_start_commands(self.mode, sampled_issues)

        # Phase fallback: if current phase has no matching issues, try others (#1643)
        # Only applies to Worker role (phases map to priority tiers)
        if self.mode == "worker" and selected_phase:
            fallback_phase, fallback_reason = _get_fallback_phase(
                selected_phase, gh_issues, rotation_phases
            )
            if fallback_phase and fallback_reason:
                log_info(f"Phase fallback: {fallback_reason}")
                # Update rotation focus to reflect fallback
                if fallback_phase == "any":
                    rotation_focus = (
                        f"**Fallback** - {fallback_reason}\n\n"
                        "Work on any available issue (do-audit, in-progress, etc.)"
                    )
                    # Don't update selected_phase - keep original for state tracking
                else:
                    # Get phase content from phase_data if available
                    selected_phase = fallback_phase
                    if phase_data and fallback_phase in phase_data:
                        content = phase_data[fallback_phase].get("content", "")
                        if content and isinstance(content, str):
                            rotation_focus = (
                                f"**Fallback** - {fallback_reason}\n\n{content}"
                            )
                        else:
                            phase_display = fallback_phase.replace("_", " ").title()
                            rotation_focus = (
                                f"**Fallback** - {fallback_reason}\n\n"
                                f"**{phase_display}** - Focus on this area"
                            )
                    else:
                        phase_display = fallback_phase.replace("_", " ").title()
                        rotation_focus = (
                            f"**Fallback** - {fallback_reason}\n\n"
                            f"**{phase_display}** - Focus on this area"
                        )

        audit_min_issues = self.config.get("audit_min_issues", 3)

        replacements = {
            "git_log": session_results.get("git_log", "(unavailable)"),
            "gh_issues": gh_issues if gh_issues else "(unavailable)",
            "last_directive": session_results.get("last_directive", ""),
            "other_feedback": session_results.get("other_feedback", ""),
            "role_mentions": session_results.get("role_mentions", ""),
            "system_status": session_results.get("system_status", ""),
            "audit_data": session_results.get("audit_data", ""),
            "rotation_focus": rotation_focus,
            "audit_min_issues": str(audit_min_issues),
            "recovery_context": recovery_context_str,
        }
        return replacements, selected_phase

    def display_prompt_info(
        self,
        replacements: dict[str, str],
        final_prompt: str,
        ai_tool: str = "claude",
        audit_round: int = 0,
    ) -> None:
        """Display prompt assembly information to console."""
        log_info("### PROMPT SOURCES ###")
        idx = 1
        # For Codex, show the additional sources it receives
        if ai_tool == "codex":
            if Path("CLAUDE.md").exists():
                log_info(f"  {idx}. CLAUDE.md (via build_codex_context)")
                idx += 1
            rules_dir = Path(".claude/rules")
            if rules_dir.exists():
                rules_files = sorted(rules_dir.glob("*.md"))
                if rules_files:
                    log_info(f"  {idx}. .claude/rules/*.md ({len(rules_files)} files)")
                    idx += 1
        log_info(f"  {idx}. {ROLES_DIR / 'shared.md'}")
        idx += 1
        log_info(f"  {idx}. {ROLES_DIR / f'{self.mode}.md'}")
        idx += 1
        if Path(".looper_config.json").exists():
            log_info(f"  {idx}. .looper_config.json (overrides)")
            idx += 1
        if audit_round > 0:
            max_rounds = self.config.get("audit_max_rounds", 5)
            log_info(f"  {idx}. AUDIT_PROMPT (round {audit_round}/{max_rounds})")
        log_info("")

        log_info("### INJECTED CONTENT ###")
        for key, value in replacements.items():
            if value:
                lines = value.split("\n")
                line_count = len(lines)
                preview = lines[0][:50]
                if len(lines[0]) > 50:
                    preview += "..."
                log_info(f"  {key}: {preview} ({line_count} lines)")
            else:
                log_info(f"  {key}: (empty)")
        log_info("")

        log_info("### FINAL PROMPT ###")
        log_info("-" * 70)
        log_info(final_prompt)
        log_info("-" * 70)
        log_info(
            f"Total: {len(final_prompt)} chars, {final_prompt.count(chr(10)) + 1} lines"
        )
        log_info("")

    def append_audit_prompt(self, prompt: str, ai_tool: str, audit_round: int) -> str:
        """Append audit prompt to the base prompt."""
        min_issues = self.config.get("audit_min_issues", 3)
        max_rounds = self.config.get("audit_max_rounds", 5)
        last_commit = None
        if ai_tool == "codex":
            last_commit = self._get_last_commit_message()
        return prompt + build_audit_prompt(
            audit_round, min_issues, max_rounds, last_commit
        )

    @staticmethod
    def _get_last_commit_message() -> str | None:
        """Get the most recent commit message for context injection."""
        result = run_git_command(["log", "-1", "--format=%B"], timeout=10)
        if result.ok and result.value and result.value.strip():
            return result.value.strip()
        return None

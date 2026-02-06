# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

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

from looper.config import ROLES_DIR, get_theme_config, inject_content, parse_frontmatter
from looper.context import (
    get_issues_by_numbers,
    get_sampled_issues,
    run_session_start_commands,
)
from looper.log import log_info, log_warning
from looper.result import format_result
from looper.rotation import get_rotation_focus
from looper.subprocess_utils import run_git_command
from looper.telemetry import get_prior_iteration_diagnostic

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
    "(all issues filtered:",  # Issues exist but all blocked/deferred/tracking
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
    if min_issues == 0:
        requirement_text = "Self-audit the work just completed. Find any issues you can."
        done_text = (
            "If you cannot find any issues, include [DONE] in your commit\n"
            "message and explain why you cannot find more issues."
        )
    else:
        requirement_text = (
            f"Self-audit the work just completed. Find at least {min_issues} issues."
        )
        done_text = (
            f"If you cannot find at least {min_issues}, include [DONE] in your commit\n"
            "message and explain why you cannot find more issues."
        )
    return f"""
{context}
## FOLLOW-UP AUDIT (Round {round_num}/{max_rounds})

{requirement_text}

If you find issues, fix them and commit. If you find more, fix more.
{done_text}
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

    def _build_theme_context(self) -> str:
        """Build theme context string for prompt injection (#2478).

        If a theme is configured, returns a markdown section describing the theme
        focus for the AI to follow. Returns empty string if no theme.

        ENSURES: Returns non-empty string only if theme is configured
        ENSURES: Returns markdown-formatted theme description
        ENSURES: Never raises - returns empty string on error
        """
        theme_config = get_theme_config(self.mode, self.worker_id)
        if not theme_config:
            return ""

        theme_name = theme_config.get("name", "")
        theme_description = theme_config.get("description", "")

        if not theme_name:
            return ""

        # Build theme context section
        lines = ["## Current Theme", ""]
        lines.append(f"**Theme:** {theme_name}")
        if theme_description:
            lines.append(f"**Focus:** {theme_description}")
        lines.append("")
        lines.append(
            "You are configured to focus on this theme. "
            "Only work on issues that match this theme. "
            "If no matching issues exist, enter Maintenance Mode."
        )
        return "\n".join(lines)

    def build_replacements(
        self,
        iteration: int,
        audit_round: int,
        working_issues: list[int] | None,
        current_phase: str | None,
        audit_min_issues_override: int | None = None,
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
            return self._build_audit_replacements(
                working_issues,
                current_phase,
                audit_min_issues_override=audit_min_issues_override,
            )
        return self._build_main_replacements(iteration, current_phase)

    def _build_audit_replacements(
        self,
        working_issues: list[int] | None,
        current_phase: str | None,
        audit_min_issues_override: int | None = None,
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

        if audit_min_issues_override is not None:
            audit_min_issues = audit_min_issues_override
        else:
            audit_min_issues = self.config.get("audit_min_issues", 3)
        replacements = {
            "git_log": git_log,
            "gh_issues": gh_issues,
            "active_issue": "",  # Active issue applies to main iteration (#2571)
            "last_directive": "",  # Session resume has full context
            "other_feedback": "",
            "role_mentions": "",
            "system_status": "",
            "audit_data": "",
            "rotation_focus": "",
            "audit_min_issues": str(audit_min_issues),
            "recovery_context": "",  # No recovery during audit (session resumed)
            "theme_context": "",  # Theme applies to main iteration, not audit
            "handoff_context": "",  # Handoff applies to session start (#2561)
        }
        return replacements, current_phase

    def _build_main_replacements(
        self,
        iteration: int,
        current_phase: str | None,
    ) -> tuple[dict[str, str], str | None]:
        """Build full context for main iterations."""
        log_info("Context: sampling issues for this iteration...")
        sampled_issues = get_sampled_issues(self.mode, worker_id=self.worker_id)
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
            theme_context = self._build_theme_context()
            replacements = {
                "git_log": "(skipped: no issues assigned)",
                "gh_issues": gh_issues if gh_issues else "(no open issues)",
                "active_issue": "",  # No active issue in no-issues path (#2571)
                "last_directive": "",
                "other_feedback": "",
                "role_mentions": "",
                "system_status": "",
                "audit_data": "",
                "rotation_focus": rotation_focus,
                "audit_min_issues": str(audit_min_issues),
                "recovery_context": recovery_context_str,
                "theme_context": theme_context,
                "handoff_context": "",  # No handoff in no-issues path (#2561)
            }
            return replacements, selected_phase

        log_info("Context: gathering git history and role feedback...")
        session_results = run_session_start_commands(
            self.mode, sampled_issues, worker_id=self.worker_id
        )

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

        # Build theme context for prompt injection (#2478)
        theme_context = self._build_theme_context()

        # Inject diagnostic when worker has zero prior commits (#2564)
        last_directive = session_results.get("last_directive", "")
        if (
            not last_directive
            and iteration > 1
            and self.mode == "worker"
        ):
            diag = get_prior_iteration_diagnostic(
                self.mode, self.worker_id, iteration
            )
            msg = (
                f"(No prior commits found after {iteration - 1} iteration(s). "
            )
            if diag:
                msg += f"{diag} "
            else:
                msg += "Prior iterations may have timed out or failed silently. "
            msg += "Focus on incremental progress and commit early.)"
            last_directive = msg

        replacements = {
            "git_log": session_results.get("git_log", "(unavailable)"),
            "gh_issues": gh_issues if gh_issues else "(unavailable)",
            "active_issue": session_results.get("active_issue", ""),
            "last_directive": last_directive,
            "other_feedback": session_results.get("other_feedback", ""),
            "role_mentions": session_results.get("role_mentions", ""),
            "system_status": session_results.get("system_status", ""),
            "audit_data": session_results.get("audit_data", ""),
            "rotation_focus": rotation_focus,
            "audit_min_issues": str(audit_min_issues),
            "recovery_context": recovery_context_str,
            "theme_context": theme_context,
            "handoff_context": session_results.get("handoff_context", ""),
        }
        return replacements, selected_phase

    def display_prompt_info(
        self,
        replacements: dict[str, str],
        final_prompt: str,
        ai_tool: str = "claude",
        audit_round: int = 0,
        audit_max_rounds_override: int | None = None,
    ) -> None:
        """Display COMPLETE prompt information showing what AI actually sees.

        For Claude: Shows auto-loaded content (CLAUDE.md + rules) + looper prompt.
        For Codex: Shows looper-prepended rules + looper prompt.
        """

        def read_text(path: Path) -> str | None:
            """Read text file for diagnostics; return None on read failure."""
            try:
                return path.read_text()
            except (OSError, UnicodeDecodeError) as e:
                log_warning(f"Warning: Could not read {path}: {e}")
                return None

        def format_block(source_name: str, content: str, end_source: str | None = None) -> str:
            """Wrap source content in explicit start/end markers."""
            end_name = end_source if end_source is not None else source_name
            if not content:
                return f"<!-- START {source_name} -->\n<!-- END {end_name} -->"
            if content.endswith("\n"):
                return f"<!-- START {source_name} -->\n{content}<!-- END {end_name} -->"
            return f"<!-- START {source_name} -->\n{content}\n<!-- END {end_name} -->"

        def count_lines(content: str) -> int:
            """Return line count for non-empty content."""
            return content.count("\n") + 1 if content else 0

        autoload_sections: list[tuple[str, str, str]] = []
        if ai_tool not in ("codex", "dasher"):
            # Claude auto-loads CLAUDE.md + .claude/rules/*.md
            claude_md = Path("CLAUDE.md")
            if claude_md.exists():
                claude_content = read_text(claude_md)
                if claude_content is not None:
                    autoload_sections.append(("CLAUDE.md", "CLAUDE.md", claude_content))

            rules_dir = Path(".claude/rules")
            if rules_dir.exists():
                for rules_file in sorted(rules_dir.glob("*.md")):
                    rules_content = read_text(rules_file)
                    if rules_content is not None:
                        rules_name = str(rules_file)
                        autoload_sections.append((rules_name, rules_name, rules_content))
        # For Codex/Dasher: rules are prepended to prompt by build_codex_context().
        # No auto-load sections to display - they are part of final_prompt itself.

        looper_sections: list[tuple[str, str, str]] = []
        audit_max_rounds = (
            audit_max_rounds_override
            if audit_max_rounds_override is not None
            else self.config.get("audit_max_rounds", 5)
        )
        shared_file = ROLES_DIR / "shared.md"
        role_file = ROLES_DIR / f"{self.mode}.md"
        shared_exists = shared_file.exists()
        role_exists = role_file.exists()
        if shared_exists and role_exists:
            shared_template = read_text(shared_file)
            role_template = read_text(role_file)
            if shared_template is not None and role_template is not None:
                _, shared_body = parse_frontmatter(shared_template)
                _, role_body = parse_frontmatter(role_template)
                shared_prompt = inject_content(shared_body.strip(), replacements)
                role_prompt = inject_content(role_body.strip(), replacements)
                base_prompt = f"{shared_prompt}\n\n{role_prompt}"

                if final_prompt.startswith(base_prompt):
                    shared_name = str(shared_file)
                    role_name = str(role_file)
                    looper_sections.append(
                        (f"{shared_name} (with injections)", shared_name, shared_prompt)
                    )
                    looper_sections.append(
                        (f"{role_name} (with injections)", role_name, role_prompt)
                    )
                    suffix = final_prompt[len(base_prompt):]
                    if suffix:
                        if audit_round > 0:
                            label = (
                                f"AUDIT_PROMPT (round {audit_round}/{audit_max_rounds})"
                            )
                        else:
                            label = "APPENDED_PROMPT"
                        looper_sections.append((label, label, suffix))
                else:
                    log_warning(
                        "Warning: Could not map final prompt to shared/role boundaries; "
                        "logging raw looper prompt."
                    )
                    looper_sections.append(("LOOPER_PROMPT (raw)", "LOOPER_PROMPT (raw)", final_prompt))
            else:
                looper_sections.append(("LOOPER_PROMPT (raw)", "LOOPER_PROMPT (raw)", final_prompt))
        else:
            looper_sections.append(("LOOPER_PROMPT (raw)", "LOOPER_PROMPT (raw)", final_prompt))

        sep = "=" * 72
        thin_sep = "-" * 72

        log_info("")
        log_info(sep)
        log_info("  PROMPT DIAGNOSTIC: What the AI receives this iteration")
        log_info(sep)

        # --- Section 1: Source files ---
        log_info("")
        log_info("  SOURCES")
        log_info(thin_sep)
        idx = 1
        if ai_tool in ("codex", "dasher"):
            log_info(f"  {idx}. CLAUDE.md + .claude/rules/*.md + .claude/codex.md  [LOOPER PREPENDED]")
            idx += 1
        else:
            if Path("CLAUDE.md").exists():
                log_info(f"  {idx}. CLAUDE.md  [AUTO-LOADED]")
                idx += 1
            rules_dir = Path(".claude/rules")
            if rules_dir.exists():
                for rules_file in sorted(rules_dir.glob("*.md")):
                    log_info(f"  {idx}. {rules_file}  [AUTO-LOADED]")
                    idx += 1
        shared_status = "LOOPER INJECTED" if shared_exists else "MISSING"
        role_status = "LOOPER INJECTED" if role_exists else "MISSING"
        log_info(f"  {idx}. {shared_file}  [{shared_status}]")
        idx += 1
        log_info(f"  {idx}. {role_file}  [{role_status}]")
        idx += 1
        if Path(".looper_config.json").exists():
            log_info(f"  {idx}. .looper_config.json  [OVERRIDES]")
            idx += 1
        if audit_round > 0:
            log_info(
                f"  {idx}. AUDIT_PROMPT  [ROUND {audit_round}/{audit_max_rounds}]"
            )

        # --- Section 2: Injected content summary ---
        log_info("")
        log_info("  INJECTED CONTENT")
        log_info(thin_sep)
        # Summary table first (skip metadata keys prefixed with _)
        total_inject_chars = 0
        for key, value in replacements.items():
            if key.startswith("_"):
                continue
            if value:
                line_count = value.count("\n") + 1
                total_inject_chars += len(value)
                log_info(f"  {key:30s}  {len(value):>6,} chars  {line_count:>4} lines")
            else:
                log_info(f"  {key:30s}  (empty)")
        log_info(f"  {'TOTAL INJECTED':30s}  {total_inject_chars:>6,} chars")
        # Budget utilization (#2695)
        budget_info = replacements.get("_budget_info", "")
        if budget_info:
            log_info(f"  {'INJECTION BUDGET':30s}  {budget_info} chars")
        budget_truncations = replacements.get("_budget_truncations", "")
        if budget_truncations:
            log_warning(f"  BUDGET TRUNCATED: {budget_truncations}")
        # Full content for each injection
        for key, value in replacements.items():
            if key.startswith("_"):
                continue
            if value:
                log_info("")
                log_info(f"  --- {key} ---")
                log_info(value)

        # --- Section 3: Auto-loaded content ---
        autoload_chars = sum(len(content) for _, _, content in autoload_sections)
        autoload_lines = sum(count_lines(content) for _, _, content in autoload_sections)

        if autoload_sections:
            log_info("")
            log_info("  AUTO-LOADED CONTENT (CLAUDE.md + rules)")
            log_info(thin_sep)
            for start_source, end_source, content in autoload_sections:
                lines = count_lines(content)
                log_info(f"  {start_source}: {len(content):,} chars, {lines} lines")
            log_info(f"  TOTAL AUTO-LOADED: {autoload_chars:,} chars, {autoload_lines} lines")
            log_info("")
            for start_source, end_source, content in autoload_sections:
                log_info(f"  --- {start_source} ---")
                log_info(format_block(start_source, content, end_source))

        # --- Section 4: Looper prompt ---
        log_info("")
        log_info("  LOOPER PROMPT (shared + role + injections)")
        log_info(thin_sep)
        for start_source, end_source, content in looper_sections:
            lines = count_lines(content)
            log_info(f"  {start_source}: {len(content):,} chars, {lines} lines")
        log_info("")
        for start_source, end_source, content in looper_sections:
            log_info(f"  --- {start_source} ---")
            log_info(format_block(start_source, content, end_source))

        # --- Section 5: Totals ---
        total_chars = autoload_chars + len(final_prompt)
        total_lines = autoload_lines + count_lines(final_prompt)
        log_info("")
        log_info("  TOTALS")
        log_info(thin_sep)
        if autoload_sections:
            log_info(f"  Auto-loaded:  {autoload_chars:>8,} chars  {autoload_lines:>5} lines")
        log_info(f"  Looper:       {len(final_prompt):>8,} chars  {final_prompt.count(chr(10)) + 1:>5} lines")
        log_info(f"  GRAND TOTAL:  {total_chars:>8,} chars  {total_lines:>5} lines")
        log_info(sep)
        log_info("")

    def append_audit_prompt(
        self,
        prompt: str,
        ai_tool: str,
        audit_round: int,
        audit_min_issues_override: int | None = None,
        audit_max_rounds_override: int | None = None,
    ) -> str:
        """Append audit prompt to the base prompt."""
        min_issues = (
            audit_min_issues_override
            if audit_min_issues_override is not None
            else self.config.get("audit_min_issues", 3)
        )
        max_rounds = (
            audit_max_rounds_override
            if audit_max_rounds_override is not None
            else self.config.get("audit_max_rounds", 5)
        )
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

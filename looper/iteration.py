# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Single-iteration execution for the looper runtime.

Responsibilities:
- Build prompt context (main vs audit) and inject it into templates.
- Choose AI tool/model and assemble the command line.
- Run the AI subprocess with logging, timeouts, and session tracking.
- Prepare iteration metrics for the outer loop to finalize.

Usage: LoopRunner owns the lifecycle and instantiates IterationRunner once per
session. This class is not thread-safe and must not be used concurrently.

Split into submodules for maintainability (#1804):
- iteration_prompt.py: Prompt building utilities
- iteration_tool_tracking.py: Tool call tracking
- iteration_process.py: AI subprocess execution
See: designs/2026-02-01-iteration-split.md
"""

import json
import os
import random
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from looper.checkpoint import RecoveryContext
from looper.config import (
    LOG_DIR,
    build_codex_context,
    get_project_name,
    get_theme_config,
    inject_content,
    set_tab_title,
)
from looper.constants import EXIT_NOT_INITIALIZED
from looper.context import IterationIssueCache
from looper.context.helpers import DEFAULT_PROMPT_BUDGET, enforce_prompt_budget
from looper.issue_manager import IssueManager
from looper.iteration_process import ProcessManager
from looper.iteration_prompt import (
    PromptBuilder,
    _has_no_issues,
    build_audit_prompt,
    extract_issue_numbers,
)
from looper.iteration_tool_tracking import ToolCallTracker
from looper.log import debug_swallow, log_info, log_warning
from looper.memory_logger import log_post_crash_memory, log_pre_command_memory
from looper.model_router import ModelRouter, ModelSwitchingPolicy
from looper.rotation import update_rotation_state
from looper.status import StatusManager
from looper.subprocess_utils import run_cmd
from looper.telemetry import (
    IterationMetrics,
    extract_token_usage,
)


@dataclass
class IterationResult:
    """Result of a single iteration execution.

    Replaces the 5-tuple return from run_iteration() for clarity (#1913).
    """

    exit_code: int
    """Exit code: 0=success, 124=timeout, 125=silence, other=error."""

    start_time: float
    """Unix timestamp when iteration began."""

    ai_tool: str
    """AI tool used: "claude", "codex", or "dasher"."""

    session_id: str | None
    """Session ID for resume support (claude --resume)."""

    codex_model: str | None
    """Selected Codex model name, if applicable."""


# Re-export public API from submodules for backwards compatibility
__all__ = [
    "IterationRunner",
    "IterationResult",
    "build_audit_prompt",
    "extract_issue_numbers",
    "EXIT_NOT_INITIALIZED",  # Re-exported from looper.constants
]


class IterationRunner:
    """Run a single iteration of the loop (main or audit).

    Uses composition with:
    - PromptBuilder for prompt context assembly
    - ToolCallTracker for tool call metrics and checkpointing
    """

    def __init__(
        self,
        mode: str,
        config: dict[str, Any],
        prompt_template: str,
        issue_manager: IssueManager,
        status_manager: StatusManager,
        get_ait_version: Callable[[], tuple[str, str] | None],
        session_id: str,
        log_dir: Path = LOG_DIR,
        codex_available: bool = False,
        dasher_available: bool = False,
        worker_id: int | None = None,
        machine: str | None = None,
    ) -> None:
        self.mode = mode
        self.worker_id = worker_id
        self.machine = machine
        self.config = config
        self.prompt_template = prompt_template
        self.issue_manager = issue_manager
        self.status_manager = status_manager
        self._get_ait_version = get_ait_version
        self._session_id = session_id
        self.log_dir = log_dir
        self.codex_available = codex_available
        self.dasher_available = dasher_available

        self.current_phase: str | None = None
        self.pending_metrics: IterationMetrics | None = None
        self._recovery_context: RecoveryContext | None = None
        self._used_recovery: bool = False  # Track if recovery context was used (#2073)
        # Log file prefix: worker_1 or worker (without suffix for single-worker mode)
        self._log_prefix = f"{mode}_{worker_id}" if worker_id is not None else mode
        # Display name for headers: "sat-Worker 1" or "Worker 1" or "Worker"
        if machine:
            base_name = (
                f"{mode.title()} {worker_id}" if worker_id is not None else mode.title()
            )
            self._display_name = f"{machine}-{base_name}"
        else:
            self._display_name = (
                f"{mode.title()} {worker_id}" if worker_id is not None else mode.title()
            )

        # Initialize PromptBuilder for prompt context assembly
        self._prompt_builder = PromptBuilder(mode, config, worker_id)
        # Initialize ToolCallTracker for tool call metrics (#1804)
        self._tool_tracker = ToolCallTracker(mode, worker_id)
        # Initialize ProcessManager for subprocess execution (#1804)
        self._process_manager = ProcessManager(config)
        # Initialize ModelRouter for per-role model selection (#1888)
        self._model_router = ModelRouter(config)
        self._model_switching_policy = ModelSwitchingPolicy.from_config(config)
        # Token growth tracking for resume skip decision (#1881)
        self._last_iteration_exceeded_token_limit = False
        # Track previous session model for switching decisions (#1888)
        self._previous_session_model: str | None = None

    def update_config(self, config: dict[str, Any], prompt_template: str) -> None:
        """Refresh config and prompt template for the next iteration."""
        self.config = config
        self.prompt_template = prompt_template
        self._prompt_builder.update_config(config)
        self._process_manager.update_config(config)
        self._model_router.update_config(config)
        self._model_switching_policy = ModelSwitchingPolicy.from_config(config)

    @property
    def current_process(self) -> subprocess.Popen[bytes] | None:
        """Get current running process for external termination access.

        Delegates to ProcessManager which owns the subprocess lifecycle.
        """
        return self._process_manager.current_process

    @current_process.setter
    def current_process(self, value: subprocess.Popen[bytes] | None) -> None:
        """Allow setting for test mocking and signal handler cleanup."""
        self._process_manager.current_process = value

    def set_recovery_context(self, context: RecoveryContext) -> None:
        """Store recovery context for injection into next prompt.

        REQUIRES: context is a valid RecoveryContext
        ENSURES: _recovery_context is set for next iteration
        ENSURES: _used_recovery is set to True for telemetry (#2073)
        """
        self._recovery_context = context
        self._used_recovery = True  # Track for telemetry (#2073)
        # Pass to PromptBuilder for prompt injection
        self._prompt_builder.set_recovery_context(context.to_prompt())

    def should_skip_resume(self) -> bool:
        """Check if session resume should be skipped due to token growth (#1881).

        Returns True if the last iteration exceeded the token abort threshold,
        indicating that resuming the session would continue accumulating an
        unbounded context.

        ENSURES: Returns bool indicating whether to skip resume
        """
        return self._last_iteration_exceeded_token_limit

    def get_tool_call_log_info(self) -> tuple[str | None, int, int]:
        """Get tool call log info for checkpoint updates.

        See: designs/2026-02-01-tool-call-checkpointing.md

        ENSURES: Returns (log_path, finished_count, last_seq)
        ENSURES: All values are valid even if no log exists
        """
        return self._tool_tracker.get_log_info()

    def get_tool_call_stats(self) -> tuple[int, dict[str, int], dict[str, int]]:
        """Get aggregated tool call statistics for telemetry.

        See: designs/2026-02-01-tool-call-metrics-format.md

        Returns:
            Tuple of (total_count, type_counts, duration_ms_per_type).
            - total_count: Total number of tool calls
            - type_counts: Count per tool type (e.g., {"Bash": 10, "Read": 15})
            - duration_ms_per_type: Duration in milliseconds per tool type

        ENSURES: Returns aggregates from all rounds (main + audits)
        ENSURES: Returns (0, {}, {}) if no tool call log exists
        """
        return self._tool_tracker.get_stats()

    def select_ai_tool(self) -> str:
        """Select which AI tool to use for this iteration.

        Selection priority: dasher > codex > claude (based on probabilities).
        Each tool is selected with its configured probability, falling through
        to the next tool if not selected or unavailable.

        REQUIRES: self.config is valid configuration dict
        REQUIRES: self.codex_available and self.dasher_available reflect actual availability
        ENSURES: result in ("claude", "codex", "dasher")
        ENSURES: result == "dasher" only if self.dasher_available and probability > 0
        ENSURES: result == "codex" only if self.codex_available and probability > 0
        ENSURES: result == "claude" as fallback
        """
        # Check dasher first (highest priority when available)
        dasher_prob = self.config.get("dasher_probability", 0.0)
        if self.dasher_available and dasher_prob > 0:
            if dasher_prob >= 1.0:
                return "dasher"
            if random.random() < dasher_prob:
                return "dasher"

        # Then check codex
        codex_prob = self.config.get("codex_probability", 0.0)
        if self.codex_available and codex_prob > 0:
            if codex_prob >= 1.0:
                return "codex"
            if random.random() < codex_prob:
                return "codex"

        # Fallback to claude
        return "claude"

    def _build_command(
        self,
        prompt: str,
        ai_tool: str,
        resume_session_id: str | None,
        force_codex_model: str | None,
        audit_round: int = 0,
    ) -> tuple[str, list[str], str | None]:
        """Build the final prompt and command for the selected tool.

        Uses ModelRouter for model selection with precedence:
        1. force_codex_model (explicit override)
        2. model_routing.audit (when audit_round > 0)
        3. model_routing.roles.<role>
        4. model_routing.default
        5. Legacy keys (claude_model, codex_model, codex_models, dasher_model)

        Returns (final_prompt, command, selected_model).
        The selected_model is used for logging.
        """
        # Use ModelRouter for model selection (#1888)
        tool_literal = cast(
            Literal["claude", "codex", "dasher"],
            ai_tool if ai_tool in ("claude", "codex", "dasher") else "claude",
        )

        if ai_tool == "claude":
            final_prompt = prompt
            cmd = [
                "claude",
                "--dangerously-skip-permissions",
                "-p",
                final_prompt,
                "--permission-mode",
                "acceptEdits",
                "--output-format",
                "stream-json",
                "--verbose",
            ]
            # Use router for model selection
            selection = self._model_router.select_model(
                self.mode, tool_literal, audit_round
            )
            claude_model = selection.model
            if claude_model:
                log_info(f"Model routing: {claude_model} (source: {selection.source})")
                cmd.extend(["--model", claude_model])
            if resume_session_id:
                cmd.extend(["--resume", resume_session_id])
            return final_prompt, cmd, claude_model

        # Codex/Dasher: rules are prepended to the prompt by the looper.
        # AGENTS.md is a minimal stub; build_codex_context() provides the rules.
        # Role prompt comes from `prompt` (shared.md + role.md via load_role_config).
        codex_context = build_codex_context()
        final_prompt = codex_context + prompt if codex_context else prompt

        # Select binary name based on tool
        binary = "dasher" if ai_tool == "dasher" else "codex"
        cmd = [
            binary,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            final_prompt,
        ]

        # Model selection using router
        selected_model = None
        if force_codex_model:
            # Explicit override takes precedence
            selected_model = force_codex_model
            log_info(f"Model routing: {selected_model} (source: force_codex_model)")
        elif ai_tool == "codex":
            # Try codex_models list first (for random selection)
            codex_models, models_source = self._model_router.select_codex_models(
                self.mode, audit_round
            )
            if codex_models:
                selected_model = random.choice(codex_models)
                log_info(
                    f"Model routing: {selected_model} (source: {models_source}, "
                    f"random from {codex_models})"
                )
            else:
                # Fall back to single model selection
                selection = self._model_router.select_model(
                    self.mode, tool_literal, audit_round
                )
                selected_model = selection.model
                if selected_model:
                    log_info(
                        f"Model routing: {selected_model} (source: {selection.source})"
                    )
        else:
            # Dasher uses its own model config
            selection = self._model_router.select_model(
                self.mode, tool_literal, audit_round
            )
            selected_model = selection.model
            if selected_model:
                log_info(
                    f"Model routing: {selected_model} (source: {selection.source})"
                )

        if selected_model:
            cmd.extend(["--model", selected_model])
        return final_prompt, cmd, selected_model

    @staticmethod
    def _sanitize_command_for_memory_log(cmd: list[str]) -> str:
        """Return a safe, concise command string for memory logging."""
        if not cmd:
            return ""

        sanitized: list[str] = []
        skip_next = False
        for token in cmd:
            if skip_next:
                skip_next = False
                continue
            if token in ("-p", "--prompt", "--json"):
                sanitized.append(token)
                sanitized.append("<prompt omitted>")
                skip_next = True
                continue
            sanitized.append(token)

        return " ".join(sanitized)

    def _print_iteration_header(
        self,
        iter_display: str,
        ai_tool: str,
        codex_model: str | None,
        replacements: dict[str, str],
        final_prompt: str,
        audit_round: int,
        audit_max_rounds_override: int | None = None,
    ) -> None:
        """Print iteration header and prompt diagnostics."""
        log_info("")
        log_info("=" * 70)
        log_info(f"=== {self._display_name} {iter_display}")
        log_info(f"=== Started at {datetime.now()}")
        log_info(f"=== Tool: {ai_tool}")
        if ai_tool == "claude":
            model = self.config.get("claude_model", "default")
            log_info(f"=== Model: {model}")
        elif ai_tool in ("codex", "dasher"):
            # codex_model parameter holds the selected model for both codex and dasher
            log_info(f"=== Model: {codex_model or 'default'}")
        ait_version = self._get_ait_version()
        if ait_version:
            log_info(f"=== AIT: {ait_version[0]} (synced {ait_version[1]})")
        log_info("=" * 70)
        log_info("")
        # Display prompt assembly info (shows the actual prompt being sent)
        self._prompt_builder.display_prompt_info(
            replacements,
            final_prompt,
            ai_tool,
            audit_round,
            audit_max_rounds_override=audit_max_rounds_override,
        )

        # Report stuck issues (manager only)
        sla_days = self.config.get("escalation_sla_days")
        self.issue_manager.report_stuck_issues(escalation_sla_days=sla_days)

    def _prepare_log_file(
        self,
        iteration: int,
        audit_round: int,
        ai_tool: str,
        model: str | None = None,
    ) -> Path:
        """Prepare log file path, create tombstone, and update status.

        Creates an empty tombstone file immediately so that even if the AI
        process crashes before producing any output, pulse.py can find the
        log file. This addresses #1373 (untraceable_failures when log not found).

        ENSURES: log_file exists on disk (may be empty)
        ENSURES: status is updated to "working"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iter_display = (
            f"{iteration}.{audit_round}" if audit_round > 0 else str(iteration)
        )
        log_file = (
            self.log_dir
            / f"{self._log_prefix}_iter_{iter_display}_{ai_tool}_{timestamp}.jsonl"
        )

        # Create tombstone file so pulse.py can always find it (#1373)
        # The AI process will append to this file during execution
        try:
            log_file.touch()
        except OSError as e:
            # Non-fatal: pulse.py will report "log not found" instead of "log empty"
            # This is acceptable - we continue with iteration even if tombstone fails
            log_warning(
                f"Warning: Failed to create tombstone log file: {e}", stream="stderr"
            )

        status_extra: dict[str, Any] = {"ai_tool": ai_tool, "audit_round": audit_round}
        if model:
            status_extra["model"] = model
        self.status_manager.write_status(
            iteration, "working", log_file=log_file, extra=status_extra
        )
        return log_file

    def _set_coder_info(self, ai_tool: str) -> None:
        """Set coder info for commit signatures (Codex, Dasher, or Claude)."""
        if ai_tool == "codex":
            os.environ["AI_CODER"] = "codex"
            os.environ.pop("CLAUDE_CODE_VERSION", None)
            os.environ.pop("DASHER_VERSION", None)
            result = run_cmd(["codex", "--version"], timeout=5)
            if result.ok and result.value and result.value.returncode == 0:
                stdout = result.value.stdout or ""
                if stdout.strip():
                    parts = stdout.strip().split()
                    if parts:
                        os.environ["CODEX_CLI_VERSION"] = parts[-1]
        elif ai_tool == "dasher":
            os.environ["AI_CODER"] = "dasher"
            os.environ.pop("CLAUDE_CODE_VERSION", None)
            os.environ.pop("CODEX_CLI_VERSION", None)
            result = run_cmd(["dasher", "--version"], timeout=5)
            if result.ok and result.value and result.value.returncode == 0:
                stdout = result.value.stdout or ""
                if stdout.strip():
                    parts = stdout.strip().split()
                    if parts:
                        os.environ["DASHER_VERSION"] = parts[-1]
        else:
            os.environ["AI_CODER"] = "claude-code"
            os.environ.pop("CODEX_CLI_VERSION", None)
            os.environ.pop("DASHER_VERSION", None)

    def _extract_session_id(self, log_file: Path, ai_tool: str) -> str | None:
        """Extract session/thread ID from the JSON log file.

        Optimized to avoid reading entire file into memory (#1744):
        - codex: streams line-by-line from start (thread_id appears early)
        - claude: reads only tail of file (session_id appears in later entries)
        """
        try:
            if not log_file.exists():
                return None

            if ai_tool == "codex":
                # Stream line-by-line for forward search (thread_id appears early)
                with log_file.open("r") as f:
                    for line in f:
                        if '"thread_id"' in line:
                            data = json.loads(line)
                            thread_id = data.get("thread_id")
                            return str(thread_id) if thread_id is not None else None
            else:
                # Read only tail of file for reverse search (session_id in later entries)
                # 64KB is sufficient for ~500 log lines which covers typical session metadata
                tail_size = 64 * 1024
                file_size = log_file.stat().st_size
                with log_file.open("r") as f:
                    if file_size > tail_size:
                        f.seek(file_size - tail_size)
                        f.readline()  # Skip partial line after seek
                    lines = f.readlines()
                for line in reversed(lines):
                    if '"session_id"' in line:
                        data = json.loads(line)
                        session_id = data.get("session_id")
                        return str(session_id) if session_id is not None else None
        except Exception as e:
            debug_swallow(
                "extract_session_id", e
            )  # Best-effort: session ID extraction for observability only
        return None

    def _record_pending_metrics(
        self,
        start_time: float,
        ai_tool: str,
        exit_code: int,
        codex_model: str | None,
        log_file: Path,
        is_audit: bool,
        iteration: int,
    ) -> None:
        """Create pending metrics for main iteration only."""
        if is_audit:
            return

        token_usage = extract_token_usage(log_file, ai_tool)
        input_tokens = int(token_usage.get("input_tokens", 0))
        output_tokens = int(token_usage.get("output_tokens", 0))
        cache_read_tokens = int(token_usage.get("cache_read_tokens", 0))
        cache_creation_tokens = int(token_usage.get("cache_creation_tokens", 0))

        # Token growth guardrails (#1881)
        # Warn when input_tokens exceeds threshold to alert about context growth
        warn_threshold = self.config.get("token_warn_threshold", 500_000)
        abort_threshold = self.config.get("token_abort_threshold", 2_000_000)
        if input_tokens >= abort_threshold:
            log_warning(
                f"⚠️  CRITICAL: input_tokens={input_tokens:,} exceeds abort threshold "
                f"({abort_threshold:,}). Consider reducing context or starting fresh session."
            )
            log_warning(
                "See: #1881 - Runaway prompt growth detected. "
                "Check for unbounded tool output or excessive session resume."
            )
            # Store flag for runner to potentially skip resume on next iteration
            self._last_iteration_exceeded_token_limit = True
        elif input_tokens >= warn_threshold:
            log_warning(
                f"⚠️  WARNING: input_tokens={input_tokens:,} exceeds warning threshold "
                f"({warn_threshold:,}). Context growth may cause issues."
            )
            self._last_iteration_exceeded_token_limit = False
        else:
            self._last_iteration_exceeded_token_limit = False

        end_time = time.time()
        self.pending_metrics = IterationMetrics(
            project=get_project_name(),
            role=self.mode,
            iteration=iteration,
            session_id=self._session_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=end_time - start_time,
            ai_tool=ai_tool,
            ai_model=(
                codex_model
                if ai_tool in ("codex", "dasher")
                else self.config.get("claude_model")
            ),
            exit_code=exit_code,
            committed=False,
            incomplete_marker=False,
            done_marker=False,
            audit_round=0,
            audit_committed=False,
            audit_rounds_run=0,
            rotation_phase=self.current_phase,
            working_issues=[],
            worker_id=self.worker_id,  # Multi-worker log file matching (#1373)
            log_file=str(log_file),  # Canonical log path for traceability (#1463)
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            recovered=self._used_recovery,  # Crash recovery tracking (#2073)
        )
        # Reset recovery flag after recording metrics
        self._used_recovery = False

    def run_iteration(
        self,
        iteration: int,
        audit_round: int = 0,
        resume_session_id: str | None = None,
        force_ai_tool: str | None = None,
        force_codex_model: str | None = None,
        working_issues: list[int] | None = None,
        audit_min_issues_override: int | None = None,
        audit_max_rounds_override: int | None = None,
    ) -> IterationResult:
        """Run a single AI iteration.

        REQUIRES: iteration >= 1 (1-based iteration counter)
        REQUIRES: audit_round >= 0 (0 = main iteration, 1+ = audit rounds)
        REQUIRES: If resume_session_id provided, must be valid session ID
        REQUIRES: If force_ai_tool provided, must be in ("claude", "codex", "dasher")
        ENSURES: Returns IterationResult with exit_code, start_time, ai_tool, session_id, codex_model
        ENSURES: exit_code is int (0=success, 124=timeout, 125=silence, other=error)
        ENSURES: start_time is Unix timestamp when iteration began
        ENSURES: ai_tool in ("claude", "codex", "dasher")
        ENSURES: Creates log file in self.log_dir
        ENSURES: Updates status via self.status_manager
        ENSURES: If not is_audit, updates rotation state
        ENSURES: If not is_audit, self.pending_metrics is populated
        """
        is_audit = audit_round > 0

        # Clear issue cache at start of main iteration (#1676)
        # This ensures fresh data for each iteration while sharing cache
        # within an iteration (main + audit rounds use same cache).
        if not is_audit:
            IterationIssueCache.clear()

        replacements, selected_phase = self._prompt_builder.build_replacements(
            iteration,
            audit_round,
            working_issues,
            self.current_phase,
            audit_min_issues_override=audit_min_issues_override,
        )
        self.current_phase = selected_phase

        # Export input provenance to env vars for commit trailers (#2473)
        # Phase: what rotation phase the AI is in (or "freeform"/"none")
        if selected_phase:
            os.environ["AI_PHASE"] = selected_phase
        else:
            os.environ["AI_PHASE"] = "freeform" if self.mode != "user" else "none"

        # Input-Issues: issues shown in the prompt (only capture on main iteration)
        if not is_audit:
            gh_issues = replacements.get("gh_issues", "")
            issue_numbers: list[str] = []
            seen: set[str] = set()
            for match in re.finditer(r"#(\d+):", gh_issues):
                number = match.group(1)
                if number not in seen:
                    issue_numbers.append(number)
                    seen.add(number)
            os.environ["AI_INPUT_ISSUES"] = ",".join(issue_numbers)

        # Theme: what theme the AI is configured with (#2478)
        # Exported for commit trailers and prompt context
        theme_config = get_theme_config(self.mode, self.worker_id)
        if theme_config and theme_config.get("name"):
            os.environ["AI_THEME"] = theme_config["name"]
        elif "AI_THEME" in os.environ:
            del os.environ["AI_THEME"]  # Clear stale theme if no longer configured

        # Check for uninitialized repo (no VISION.md + no issues) (#2410)
        # This is a project init failure, not a work scenario.
        # Workers with no issues enter Maintenance Mode (see worker.md).
        # But no VISION.md means the project was never initialized.
        gh_issues = replacements.get("gh_issues", "")
        if (
            not is_audit
            and _has_no_issues(gh_issues)
            and not Path("VISION.md").exists()
        ):
            start_time = time.time()
            log_warning("\n⚠️  Repo not initialized - cannot start autonomous work")
            log_info("")
            log_info("This repo is missing VISION.md and has no issues.")
            log_info("Autonomous roles need direction to work on.")
            log_info("")
            log_info("To initialize (see CLAUDE.md 'Project init workflow'):")
            log_info("  1. Write VISION.md with project direction")
            log_info("  2. Create initial issues: gh issue create --title '...' --label P1")
            log_info("  3. Then start autonomous roles")
            log_info("")
            ai_tool = force_ai_tool if force_ai_tool else self.select_ai_tool()
            return IterationResult(
                exit_code=EXIT_NOT_INITIALIZED,
                start_time=start_time,
                ai_tool=ai_tool,
                session_id=None,
                codex_model=force_codex_model,
            )

        budget = self.config.get("prompt_budget_chars", DEFAULT_PROMPT_BUDGET)
        replacements = enforce_prompt_budget(replacements, budget=budget)
        prompt = inject_content(self.prompt_template, replacements)
        ai_tool = force_ai_tool if force_ai_tool else self.select_ai_tool()

        # Codex rules are prepended in _build_command() via build_codex_context().
        # AGENTS.md is a minimal stub.

        # Model switching logic for resumed sessions (#1888)
        # Determine selected model before building command to check switching policy
        effective_resume_session_id = resume_session_id
        if resume_session_id and ai_tool == "claude":
            # Pre-compute what model would be selected
            tool_literal = cast(Literal["claude", "codex", "dasher"], "claude")
            selection = self._model_router.select_model(
                self.mode, tool_literal, audit_round
            )
            new_model = selection.model

            # Check if model changed and whether to restart session
            if self._previous_session_model is not None:
                if self._model_switching_policy.should_restart_session(
                    ai_tool, self._previous_session_model, new_model
                ):
                    # Drop resume_session_id to start fresh with new model
                    effective_resume_session_id = None
                else:
                    # Pin to previous model if switching disabled
                    pinned_model = self._model_switching_policy.should_pin_model(
                        ai_tool, self._previous_session_model, new_model
                    )
                    if pinned_model != new_model:
                        # Update router to use pinned model (affects _build_command)
                        # We do this by temporarily overriding the config
                        # This is safe because config gets refreshed each iteration
                        self.config["claude_model"] = pinned_model
                        self._model_router.update_config(self.config)

        if is_audit:
            prompt = self._prompt_builder.append_audit_prompt(
                prompt,
                ai_tool,
                audit_round,
                audit_min_issues_override=audit_min_issues_override,
                audit_max_rounds_override=audit_max_rounds_override,
            )

        final_prompt, cmd, codex_model = self._build_command(
            prompt, ai_tool, effective_resume_session_id, force_codex_model, audit_round
        )

        # Track model for next iteration's switching decision
        if ai_tool == "claude":
            self._previous_session_model = (
                codex_model  # codex_model holds claude model here
            )

        iter_display = f"{iteration}.{audit_round}" if is_audit else str(iteration)

        # Set tab title to ensure it stays correct (especially for codex which lacks MCP)
        role_letter = {
            "worker": "W",
            "prover": "P",
            "researcher": "R",
            "manager": "M",
        }.get(self.mode, "U")
        set_tab_title(role_letter, worker_id=self.worker_id)

        self._print_iteration_header(
            iter_display,
            ai_tool,
            codex_model,
            replacements,
            final_prompt,
            audit_round,
            audit_max_rounds_override=audit_max_rounds_override,
        )

        log_file = self._prepare_log_file(iteration, audit_round, ai_tool, model=codex_model)
        self._set_coder_info(ai_tool)

        # Initialize tool call tracker for fine-grained recovery (#1625, #1804)
        # See: designs/2026-02-01-tool-call-checkpointing.md
        source: Literal["claude", "codex", "dasher"] = (
            cast("Literal['claude', 'codex', 'dasher']", ai_tool)
            if ai_tool in ("claude", "codex", "dasher")
            else "claude"
        )
        self._tool_tracker.initialize(source)
        # Only clear on main iteration (audit_round == 0) to accumulate stats
        # across audit rounds for telemetry (#1630)
        # See: designs/2026-02-01-tool-call-metrics-format.md
        if not is_audit:
            self._tool_tracker.clear()

        # Log memory state before running AI process (#1470)
        cmd_str = self._sanitize_command_for_memory_log(cmd)
        log_pre_command_memory(cmd_str, log_dir=self.log_dir)

        start_time = time.time()
        exit_code = self._process_manager.run_ai_process(
            cmd, log_file, ai_tool, start_time, self._tool_tracker
        )
        # Note: current_process is now a property delegating to ProcessManager,
        # so external code can access it during execution for termination.

        # Log memory state after crash/abnormal exit (#1470)
        if exit_code != 0:
            log_post_crash_memory(cmd_str, exit_code, log_dir=self.log_dir)

        log_info("")
        log_info(f"=== {self._display_name} {iter_display} ({ai_tool}) completed ===")
        log_info(f"=== Exit code: {exit_code} ===")
        log_info(f"=== Log saved to: {log_file} ===")

        self.status_manager.scrub_log_file(log_file)

        # Check for headless violations (AI asking for user direction)
        self.status_manager.check_headless_violation(log_file, self.mode)

        session_id = self._extract_session_id(log_file, ai_tool)
        if session_id:
            log_info(f"=== Session: {session_id[:8]}... ===")
        log_info("")

        if self.current_phase and not is_audit:
            update_rotation_state(self.mode, self.current_phase)

        self._record_pending_metrics(
            start_time, ai_tool, exit_code, codex_model, log_file, is_audit, iteration
        )

        return IterationResult(
            exit_code=exit_code,
            start_time=start_time,
            ai_tool=ai_tool,
            session_id=session_id,
            codex_model=codex_model,
        )

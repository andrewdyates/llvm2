# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Checkpoint state dataclasses for crash recovery.

See: designs/2026-01-26-checkpoint-recovery.md

This module defines the data structures used to capture and restore
session state for crash recovery.
"""

from dataclasses import asdict, dataclass, field

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointState",
    "CheckpointContext",
    "CrashSignature",
    "LooperCheckpoint",
    "RecoveryContext",
]

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class CheckpointState:
    """Current work state at checkpoint time."""

    working_issues: list[int] = field(default_factory=list)
    current_phase: str | None = None
    todo_list: list[dict] = field(default_factory=list)
    last_tool: str | None = None
    last_tool_input: str | None = None
    last_tool_output_truncated: str | None = None
    # Tool-call log tracking (see designs/2026-02-01-tool-call-checkpointing.md)
    tool_call_log_path: str | None = None
    tool_calls_completed: int = 0
    tool_calls_last_seq: int = 0


@dataclass
class CheckpointContext:
    """Git context at checkpoint time."""

    files_modified: list[str] = field(default_factory=list)
    uncommitted_changes: bool = False
    last_commit_before_session: str | None = None


@dataclass
class CrashSignature:
    """Process identity for crash detection."""

    pid: int
    hostname: str
    started_at: str
    worker_id: int | None = None


@dataclass
class LooperCheckpoint:
    """Complete checkpoint data structure."""

    schema_version: int
    session_id: str
    mode: str
    worker_id: int | None
    iteration: int
    timestamp: str
    state: CheckpointState
    context: CheckpointContext
    crash_signature: CrashSignature

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LooperCheckpoint":
        """Create from dict (JSON deserialization)."""
        return cls(
            schema_version=data["schema_version"],
            session_id=data["session_id"],
            mode=data["mode"],
            worker_id=data.get("worker_id"),
            iteration=data["iteration"],
            timestamp=data["timestamp"],
            state=CheckpointState(**data["state"]),
            context=CheckpointContext(**data["context"]),
            crash_signature=CrashSignature(**data["crash_signature"]),
        )


@dataclass
class RecoveryContext:
    """Context to inject into prompt for crash recovery.

    ENSURES: to_prompt() returns markdown formatted recovery context
    ENSURES: to_prompt() output is typically ~500 chars for typical recovery (no hard limit)
    """

    working_issues: list[int]
    current_phase: str | None
    todo_progress: list[dict]
    last_tool: str | None
    files_modified: list[str]
    uncommitted_changes: bool
    crashed_at_iteration: int
    crashed_at: str
    # Tool call log fields (see designs/2026-02-01-tool-call-checkpointing.md)
    tool_call_log_path: str | None = None
    tool_calls_completed: int = 0
    tool_call_summary: str = ""  # Markdown summary of completed tools

    def to_prompt(self, max_tokens: int = 500) -> str:
        """Format recovery context for prompt injection.

        Note: max_tokens is reserved for future truncation support.
        Currently outputs complete context (~500 chars for typical recovery).

        REQUIRES: max_tokens > 0 (unused, reserved)
        ENSURES: Result is valid markdown
        """
        iter_num = self.crashed_at_iteration
        lines: list[str] = [
            "## RECOVERY: Resuming from crash",
            "",
            f"The previous session crashed at iteration {iter_num}. Recovery context:",
            "",
            "**Last known state:**",
        ]

        # Working issues
        if self.working_issues:
            issues_str = ", ".join(f"#{n}" for n in self.working_issues[:5])
            lines.append(f"- Working on issues: {issues_str}")
        else:
            lines.append("- Working on issues: (none tracked)")

        # Phase
        if self.current_phase:
            lines.append(f"- Phase: {self.current_phase}")

        # Last tool
        if self.last_tool:
            lines.append(f"- Last tool: `{self.last_tool}`")

        # Progress section
        if self.todo_progress:
            lines.append("")
            lines.append("**Progress before crash:**")
            for todo in self.todo_progress[:10]:
                status = todo.get("status", "pending")
                content = todo.get("content", "")[:60]
                if status == "completed":
                    lines.append(f"- [x] {content}")
                elif status == "in_progress":
                    lines.append(f"- [ ] {content} (IN PROGRESS when crashed)")
                else:
                    lines.append(f"- [ ] {content}")

        # Files modified
        if self.files_modified:
            lines.append("")
            lines.append("**Files modified (uncommitted):**")
            lines.extend(f"- {f}" for f in self.files_modified[:10])

        # Tool call history (see designs/2026-02-01-tool-call-checkpointing.md)
        if self.tool_calls_completed > 0 and self.tool_call_summary:
            lines.append("")
            lines.append(
                f"**Tool calls completed before crash:** {self.tool_calls_completed}"
            )
            lines.append("")
            lines.append(self.tool_call_summary)

        # Instruction
        lines.append("")
        if self.uncommitted_changes:
            lines.append(
                "**Instruction:** Review the uncommitted changes and continue from where the "
                "previous session left off. Avoid re-running expensive operations that may have "
                "already completed."
            )
        else:
            lines.append(
                "**Instruction:** Continue from where the previous session left off. "
                "The previous session crashed before making changes."
            )

        lines.append("")
        return "\n".join(lines)

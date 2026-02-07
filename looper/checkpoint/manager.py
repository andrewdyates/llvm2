# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Checkpoint manager for crash recovery.

See: designs/2026-01-26-checkpoint-recovery.md

This module manages the checkpoint lifecycle: writing checkpoints atomically,
detecting crashed sessions, and providing recovery context.

REQUIRES: checkpoint_file is a valid Path
ENSURES: Atomic writes via temp file + rename
ENSURES: Recovery detection via session_id mismatch + dead PID
"""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from socket import gethostname

from ai_template_scripts.subprocess_utils import is_process_alive
from looper.log import log_info, log_warning
from looper.subprocess_utils import run_git_command

from .state import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointContext,
    CheckpointState,
    CrashSignature,
    LooperCheckpoint,
    RecoveryContext,
)
from .tool_call_log import ToolCallLog

__all__ = [
    "CheckpointManager",
    "get_checkpoint_filename",
    "get_tool_call_log_filename",
    # Private helpers exported for testing
    "_is_pid_dead",
    "_get_uncommitted_files",
    "_get_last_commit_hash",
]


def _is_pid_dead(pid: int) -> bool:
    """Check if a process with given PID is no longer running.

    Thin wrapper around is_process_alive for backward compatibility.
    Exported in __all__ for tests that reference _is_pid_dead.
    """
    return not is_process_alive(pid)


def _get_uncommitted_files() -> list[str]:
    """Get list of modified/untracked files from git status.

    ENSURES: Returns list of file paths (may be empty)
    ENSURES: Never raises - returns empty list on error
    """
    result = run_git_command(["status", "--porcelain"], timeout=5)
    if not result.ok or not result.value:
        return []

    files = []
    for line in result.value.rstrip("\n").split("\n"):
        if not line:
            continue
        # Format: XY filename or XY -> newname (for renames)
        parts = line[3:].split(" -> ")
        filename = parts[-1].strip()
        if filename:
            files.append(filename)
    return files


def _get_last_commit_hash() -> str | None:
    """Get the HEAD commit hash.

    ENSURES: Returns short commit hash or None on error
    """
    result = run_git_command(["rev-parse", "--short", "HEAD"], timeout=5)
    if result.ok and result.value:
        return result.value.strip()
    return None


class CheckpointManager:
    """Manages checkpoint lifecycle for crash recovery.

    REQUIRES: checkpoint_file is a valid Path
    ENSURES: Atomic writes via temp file + rename
    ENSURES: Recovery detection via session_id mismatch + dead PID
    """

    def __init__(
        self,
        checkpoint_file: Path,
        mode: str,
        session_id: str,
        worker_id: int | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file (e.g., .looper_checkpoint_worker.json)
            mode: Role name (worker, manager, etc.)
            session_id: Unique session identifier
            worker_id: Optional worker ID for multi-worker mode
        """
        self.checkpoint_file = checkpoint_file
        self.mode = mode
        self.session_id = session_id
        self.worker_id = worker_id
        self._started_at = datetime.now(UTC).isoformat()
        self._hostname = gethostname().split(".")[0]

    def check_recovery(self) -> RecoveryContext | None:
        """Check for crashed session and return recovery context if found.

        REQUIRES: Called at session start before any work
        ENSURES: Returns None if no checkpoint or checkpoint is from this session
        ENSURES: Returns RecoveryContext if checkpoint from crashed session
        ENSURES: Crashed = PID dead AND session_id mismatch
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            data = json.loads(self.checkpoint_file.read_text())
            checkpoint = LooperCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted checkpoint - remove it
            log_warning(
                f"Warning: corrupted checkpoint, removing: {e}. "
                "See docs/troubleshooting.md#checkpoint-issues"
            )
            self.clear()
            return None

        # Check if this is our own session (shouldn't happen - same session resumed)
        if checkpoint.session_id == self.session_id:
            return None

        # Check if checkpoint is from same mode/worker
        if checkpoint.mode != self.mode:
            return None
        if checkpoint.worker_id != self.worker_id:
            return None

        # Check if the crashed process is still running
        crashed_pid = checkpoint.crash_signature.pid
        if not _is_pid_dead(crashed_pid):
            # Process still running - don't recover
            return None

        # Process is dead and session_id differs - this is a crash recovery
        log_info(f"*** RECOVERY: Found crashed session {checkpoint.session_id[:6]} ***")
        log_info(f"    Crashed at iteration {checkpoint.iteration}")
        log_info(f"    PID {crashed_pid} is no longer running")

        # Load tool call summary if log exists (designs/2026-02-01-tool-call-checkpointing.md)
        tool_call_summary = ""
        tool_call_log_path = checkpoint.state.tool_call_log_path
        tool_calls_completed = checkpoint.state.tool_calls_completed
        if tool_call_log_path:
            log_path = Path(tool_call_log_path)
            if log_path.exists():
                log_info(f"    Tool calls completed: {tool_calls_completed}")
                # Create temporary ToolCallLog to load and summarize
                temp_log = ToolCallLog(
                    log_path,
                    source="claude",  # Source doesn't matter for loading
                )
                tool_call_summary = temp_log.summarize_for_recovery()

        # Build recovery context
        recovery = RecoveryContext(
            working_issues=checkpoint.state.working_issues,
            current_phase=checkpoint.state.current_phase,
            todo_progress=checkpoint.state.todo_list,
            last_tool=checkpoint.state.last_tool,
            files_modified=checkpoint.context.files_modified,
            uncommitted_changes=checkpoint.context.uncommitted_changes,
            crashed_at_iteration=checkpoint.iteration,
            crashed_at=checkpoint.timestamp,
            tool_call_log_path=tool_call_log_path,
            tool_calls_completed=tool_calls_completed,
            tool_call_summary=tool_call_summary,
        )

        # Clear the old checkpoint - we're now the new session
        self.clear()

        return recovery

    def write(
        self,
        iteration: int,
        phase: str | None = None,
        working_issues: list[int] | None = None,
        todo_list: list[dict] | None = None,
        last_tool: str | None = None,
        last_tool_input: str | None = None,
        last_tool_output: str | None = None,
        tool_output_truncate_chars: int = 2000,
        tool_call_log_path: str | None = None,
        tool_calls_completed: int = 0,
        tool_calls_last_seq: int = 0,
    ) -> bool:
        """Write checkpoint atomically.

        ENSURES: Atomic write via temp + rename
        ENSURES: Returns True on success, False on error
        """
        # Get current git context
        files_modified = _get_uncommitted_files()
        uncommitted_changes = len(files_modified) > 0
        last_commit = _get_last_commit_hash()

        # Truncate tool output if needed
        tool_output_truncated = None
        if last_tool_output:
            if len(last_tool_output) > tool_output_truncate_chars:
                tool_output_truncated = (
                    last_tool_output[:tool_output_truncate_chars]
                    + f"...[truncated at {tool_output_truncate_chars} chars]"
                )
            else:
                tool_output_truncated = last_tool_output

        checkpoint = LooperCheckpoint(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            session_id=self.session_id,
            mode=self.mode,
            worker_id=self.worker_id,
            iteration=iteration,
            timestamp=datetime.now(UTC).isoformat(),
            state=CheckpointState(
                working_issues=working_issues or [],
                current_phase=phase,
                todo_list=todo_list or [],
                last_tool=last_tool,
                last_tool_input=last_tool_input,
                last_tool_output_truncated=tool_output_truncated,
                tool_call_log_path=tool_call_log_path,
                tool_calls_completed=tool_calls_completed,
                tool_calls_last_seq=tool_calls_last_seq,
            ),
            context=CheckpointContext(
                files_modified=files_modified,
                uncommitted_changes=uncommitted_changes,
                last_commit_before_session=last_commit,
            ),
            crash_signature=CrashSignature(
                pid=os.getpid(),
                hostname=self._hostname,
                started_at=self._started_at,
                worker_id=self.worker_id,
            ),
        )

        try:
            # Atomic write: write to temp file, then rename
            parent_dir = self.checkpoint_file.parent
            parent_dir.mkdir(exist_ok=True)

            # Use PID in temp file name to avoid collisions
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=parent_dir,
                prefix=".checkpoint_tmp_",
                suffix=".json",
                delete=False,
            ) as tmp_f:
                json.dump(checkpoint.to_dict(), tmp_f, indent=2)
                tmp_path = Path(tmp_f.name)

            # Atomic rename - clean up temp file on failure
            try:
                tmp_path.rename(self.checkpoint_file)
            except OSError:
                tmp_path.unlink(missing_ok=True)
                raise
            return True

        except OSError as e:
            log_warning(f"Warning: checkpoint write failed: {e}")
            return False

    def clear(self) -> bool:
        """Clear checkpoint file on clean exit.

        ENSURES: Checkpoint file deleted if exists
        ENSURES: Returns True on success or if file didn't exist
        """
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            return True
        except OSError as e:
            log_warning(f"Warning: checkpoint clear failed: {e}")
            return False


def get_checkpoint_filename(role: str, worker_id: int | None = None) -> str:
    """Generate checkpoint filename for a given role and worker ID.

    Args:
        role: Role name ('worker', 'manager', 'researcher', 'prover')
        worker_id: Optional worker instance ID for multi-worker support

    ENSURES: Returns filename like .looper_checkpoint_worker.json
             or .looper_checkpoint_worker_1.json
    """
    if worker_id is not None:
        return f".looper_checkpoint_{role}_{worker_id}.json"
    return f".looper_checkpoint_{role}.json"


def get_tool_call_log_filename(role: str, worker_id: int | None = None) -> str:
    """Generate tool call log filename for a given role and worker ID.

    Args:
        role: Role name ('worker', 'manager', 'researcher', 'prover')
        worker_id: Optional worker instance ID for multi-worker support

    ENSURES: Returns filename like .looper_tool_calls_worker.jsonl
             or .looper_tool_calls_worker_1.jsonl
    """
    if worker_id is not None:
        return f".looper_tool_calls_{role}_{worker_id}.jsonl"
    return f".looper_tool_calls_{role}.jsonl"

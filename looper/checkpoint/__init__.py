# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Execution state checkpointing for crash recovery.

Enables automatic recovery from crashes by persisting session state to disk.
Sessions that crash resume from their last checkpoint rather than starting
fresh with only git history as context.

See: designs/2026-01-26-checkpoint-recovery.md

REQUIRES: checkpoint_file is a valid Path
ENSURES: Atomic writes via temp file + rename
ENSURES: Recovery detection via session_id mismatch + dead PID

This package is organized into three modules:
- state: Checkpoint state dataclasses (CheckpointState, RecoveryContext, etc.)
- manager: CheckpointManager class and filename helpers
- tool_call_log: ToolCallRecord, ToolCallLog for fine-grained recovery
"""

# Re-export from state module
# Re-export from manager module
from .manager import (
    CheckpointManager,
    _get_last_commit_hash,
    _get_uncommitted_files,
    # Private helpers exported for testing
    _is_pid_dead,
    get_checkpoint_filename,
    get_tool_call_log_filename,
)
from .state import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointContext,
    CheckpointState,
    CrashSignature,
    LooperCheckpoint,
    RecoveryContext,
)

# Re-export from tool_call_log module
from .tool_call_log import (
    TOOL_CALL_LOG_VERSION,
    ToolCallLog,
    ToolCallLogConfig,
    ToolCallRecord,
)

__all__ = [
    # Constants
    "CHECKPOINT_SCHEMA_VERSION",
    "TOOL_CALL_LOG_VERSION",
    # State dataclasses
    "CheckpointState",
    "CheckpointContext",
    "CrashSignature",
    "LooperCheckpoint",
    "RecoveryContext",
    # Manager
    "CheckpointManager",
    # Tool call log
    "ToolCallRecord",
    "ToolCallLogConfig",
    "ToolCallLog",
    # Filename helpers
    "get_checkpoint_filename",
    "get_tool_call_log_filename",
    # Private helpers (exported for testing)
    "_is_pid_dead",
    "_get_uncommitted_files",
    "_get_last_commit_hash",
]

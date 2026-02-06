# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper - Autonomous continuous loop for AI workers and managers

API Stability:
    - STABLE: Classes, constants, and main() are stable public API.
    - INTERNAL: Functions marked "testing/debugging" in __all__ are implementation
      details. They may change without notice between versions.
    - EXPERIMENTAL: Zone and sync functions are subject to change as multi-machine
      coordination evolves.

Public API (stable):
    Classes: LoopRunner, IterationRunner, IterationResult, IssueManager, StatusManager,
             CheckpointManager, CheckpointState, RecoveryContext
    Functions: load_role_config, load_project_config, main
    Constants: ROLES_DIR, LOG_DIR, LOG_RETENTION_HOURS,
               EXIT_NOT_INITIALIZED, EXIT_SILENCE, EXIT_TIMEOUT

Internal Helpers (may change):
    build_codex_context, inject_content, select_phase_by_priority, show_prompt,
    parse_frontmatter, parse_phase_blocks, validate_config

These internal functions are exported for testing and debugging but should not be
relied upon by external code. They may be moved to private modules in future versions.

Modular package structure:
- checkpoint: Execution state checkpointing for crash recovery
- config: Configuration loading, frontmatter parsing, role config
- context/: Subpackage for session context builders (git_context, issue_context,
  audit_context, system_context, helpers)
- hooks: Git hook installation and management
- issue_manager: Issue operations (gh wrapper, sampling)
- iteration: Iteration execution (prompt build, AI run, metrics)
- log: Structured logging (JSON files, console routing)
- result: Result type for iteration outcomes
- rotation: Phase rotation state management
- runner: LoopRunner class and main loop logic
- status: Status tracking (metrics, logs)
- subprocess_utils: Result[T] wrappers for git/gh subprocess commands
- sync: Multi-machine zone branch syncing with origin/main
- telemetry: Metrics collection
- zones: Zone-based file locking for multi-worker coordination

The main LoopRunner class is in looper/runner.py.
"""

from looper.checkpoint import (
    CheckpointManager,
    CheckpointState,
    RecoveryContext,
    get_checkpoint_filename,
)
from looper.config import (
    LOG_DIR,
    LOG_RETENTION_HOURS,
    ROLES_DIR,
    build_codex_context,
    get_project_name,
    inject_content,
    load_project_config,
    load_role_config,
    parse_frontmatter,
    parse_phase_blocks,
    validate_config,
)
from looper.constants import (
    EXIT_NOT_INITIALIZED,
    EXIT_SILENCE,
    EXIT_TIMEOUT,
)
from looper.context import (
    run_session_start_commands,
)
from looper.hooks import (
    install_hooks,
)
from looper.issue_manager import IssueManager
from looper.iteration import (
    IterationResult,
    IterationRunner,
    build_audit_prompt,
    extract_issue_numbers,
)
from looper.log import (
    build_log_path,
    get_logger,
    log_debug,
    log_error,
    log_info,
    log_warning,
    setup_logging,
)
from looper.model_router import (
    AiTool,
    ModelRouter,
    ModelSelection,
    ModelSwitchingPolicy,
)
from looper.rotation import (
    get_rotation_focus,
    load_rotation_state,
    save_rotation_state,
    select_phase_by_priority,
    update_rotation_state,
)
from looper.runner import (
    LoopRunner,
    check_concurrent_sessions,
    main,
    show_prompt,
)
from looper.status import StatusManager
from looper.sync import (
    SyncConfig,
    SyncResult,
    SyncStatus,
    check_stale_staged_files,
    check_uncommitted_work_size,
    get_commits_behind,
    get_conflict_files,
    get_current_branch,
    get_staged_files,
    get_uncommitted_changes_result,
    get_uncommitted_line_count,
    has_uncommitted_changes,
    sync_from_main,
)
from looper.zones import (
    WorkerInfo,
    ZoneLock,
    ZoneStatus,
    can_edit_file,
    check_files_in_zone,
    file_in_zone,
    get_worker_zone_patterns,
    get_zone_status,
    get_zone_status_line,
    load_zone_config,
)

__all__ = [
    # ===== STABLE API =====
    # Classes - primary entry points
    "LoopRunner",
    "IterationRunner",
    "IterationResult",
    "IssueManager",
    "StatusManager",
    # Functions - stable public interface
    "load_role_config",
    "load_project_config",
    "main",
    # Constants - used by external scripts (stable)
    "ROLES_DIR",
    "LOG_DIR",
    "LOG_RETENTION_HOURS",
    # Exit codes - iteration termination reasons (#1972, stable)
    "EXIT_NOT_INITIALIZED",
    "EXIT_SILENCE",
    "EXIT_TIMEOUT",
    # Logging - structured logging support (stable)
    "build_log_path",
    "get_logger",
    "log_debug",
    "log_error",
    "log_info",
    "log_warning",
    "setup_logging",
    # ===== INTERNAL HELPERS =====
    # These are exported for testing/debugging but may change without notice.
    # See docstring "Internal Helpers" section.
    "run_session_start_commands",
    "get_rotation_focus",
    "parse_frontmatter",
    "parse_phase_blocks",
    "inject_content",
    "build_codex_context",
    "install_hooks",
    "show_prompt",
    "check_concurrent_sessions",
    "get_project_name",
    "validate_config",
    "build_audit_prompt",
    "extract_issue_numbers",
    "load_rotation_state",
    "save_rotation_state",
    "update_rotation_state",
    "select_phase_by_priority",
    # ===== DEPRECATED (scheduled for removal in v2.0) =====
    # Use replacements documented in docs/deprecations.md
    "check_stale_staged_files",  # → get_staged_files() + warn_stale_staged_files()
    "check_uncommitted_work_size",  # → warn_uncommitted_work()
    # ===== EXPERIMENTAL =====
    # Multi-machine sync - subject to change as coordination evolves
    "SyncConfig",
    "SyncResult",
    "SyncStatus",
    "get_commits_behind",
    "get_conflict_files",
    "get_current_branch",
    "get_staged_files",
    "get_uncommitted_changes_result",
    "get_uncommitted_line_count",
    "has_uncommitted_changes",
    "sync_from_main",
    # Zone management - multi-worker file coordination
    "WorkerInfo",
    "ZoneLock",
    "ZoneStatus",
    "can_edit_file",
    "check_files_in_zone",
    "file_in_zone",
    "get_worker_zone_patterns",
    "get_zone_status",
    "get_zone_status_line",
    "load_zone_config",
    # Checkpoint - crash recovery
    "CheckpointManager",
    "CheckpointState",
    "RecoveryContext",
    "get_checkpoint_filename",
    # Model routing - per-role model selection (#1888)
    "AiTool",
    "ModelRouter",
    "ModelSelection",
    "ModelSwitchingPolicy",
]

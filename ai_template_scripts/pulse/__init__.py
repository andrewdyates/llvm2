#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
pulse - Programmatic stats collection and threshold checking.

CANONICAL SOURCE: dropbox-ai-prototypes/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

This package collects metrics, detects issues, and writes flags for managers.

Usage:
    ./ai_template_scripts/pulse.py              # Run once, show markdown output
    python -m ai_template_scripts.pulse         # Module entrypoint

Public API (re-exported from submodules):
    Constants: METRICS_DIR, FLAGS_DIR, THRESHOLDS, etc.
    Functions: collect_metrics, check_thresholds, pulse_once, pulse_watch, etc.

Part of #404: pulse.py module split
"""

# Standard library imports needed for test monkeypatching
import os

# Import code metrics module
from .code_metrics import (
    _matches_exclude_pattern,
    _path_is_excluded,
    _relative_path,
    _resolve_root,
    count_lines_by_type,
    find_forbidden_ci,
    find_large_files,
)

# Import config first to trigger config loading
from .config import (
    LARGE_FILE_EXCLUDE_PATTERNS,
    SKIP_ORPHANED_TESTS,
    THRESHOLDS,
    _apply_config,
    _load_config,
    _warn_unknown_keys,
)

# Import constants
from .constants import (
    BLOCKER_PATTERN,
    CONFIG_PATHS,
    CRASH_LOG_PATTERN,
    DEFAULT_THRESHOLDS,
    EXCLUDE_DIRS,
    EXCLUDE_GLOB_PATTERNS,
    FIND_EXCLUDE,
    FIND_PRUNE,
    FLAGS_DIR,
    GIT_DEP_PATTERN,
    GREP_EXCLUDE,
    GREP_EXCLUDE_DIRS,
    ISSUE_REF_PATTERN,
    KNOWN_LARGE_FILES_KEYS,
    KNOWN_RUNTIME_KEYS,
    KNOWN_SECTIONS,
    LONG_RUNNING_EXCLUDE_PATTERNS,
    LONG_RUNNING_PROCESS_NAMES,
    MAX_METRICS_PER_DAY,
    METRICS_ARCHIVE_DIR,
    METRICS_DIR,
    METRICS_RETENTION_DAYS,
    WORKER_LOG_PATTERN,
    _dir_to_regex,
)

# Import dirty tracking module
from .dirty_tracking import (
    _detect_dirty_file_changes,
    _format_dirty_diagnostics,
    _get_dirty_file_fingerprints,
    _get_dirty_source_files,
    _get_file_mtimes,
    _get_porcelain_status,
    _identify_file_modifiers,
    _is_pulse_output_path,
    _is_rename_status,
    _normalize_status_path,
    _save_dirty_snapshot,
    _snapshot_repo_root,
)

# Import flags module
from .flags import (
    _flag_summary,
    check_thresholds,
    write_flags,
)

# Import git metrics module
from .git_metrics import (
    _compute_python_type_coverage,
    _compute_test_code_ratio,
    _count_todo_comments,
    _detect_git_operation_in_progress,
    _format_drift_reason,
    _get_commit_staleness,
    _get_repo_head_rev,
    count_template_lines,
    get_code_quality,
    get_consolidation_debt,
    get_doc_claim_status,
    get_git_status,
    get_outdated_git_deps,
)

# Import issue metrics module
from .issue_metrics import (
    _count_issues_within_days,
    _empty_issue_counts,
    _get_closed_issue_count_graphql,
    _get_issue_counts_graphql,
    _get_issue_counts_rest,
    _get_open_issue_counts_rest,
    _get_repo_owner_and_name,
    _has_in_progress_label,
    _is_graphql_rate_limited,
    _should_prefer_rest_issue_counts,
    _unpack_issue_counts,
    get_blocked_issue_list,
    get_blocked_missing_reason,
    get_issue_counts,
    get_issue_velocity,
    get_issues_reopened,
    get_long_blocked_issues,
    get_stale_blockers,
)

# Import output module
from .output import (
    _print_code_status,
    _print_complexity_status,
    _print_issue_status,
    _print_proof_status,
    _print_system_status,
    _print_test_status,
)

# Import process metrics module
from .process_metrics import (
    _args_contain_repo_path,
    _classify_crash,
    _get_process_cwd,
    _infer_repo_from_path,
    _parse_etime,
    get_long_running_processes,
    get_recent_crashes,
)

# Import session metrics module
from .session_metrics import (
    _STATUS_OPTIONAL_FIELDS,
    INFRASTRUCTURE_ERROR_PATTERNS,
    INFRASTRUCTURE_EXIT_CODES,
    _get_worker_log_info,
    _is_infrastructure_failure,
    _iter_role_status_files,
    _parse_pid,
    _pid_command_line,
    _pid_looks_like_looper,
    _read_pid_file,
    _read_status_data,
    _read_status_pid,
    get_active_session_details,
    get_active_sessions,
    get_untraceable_failures,
)

# Import storage module
from .storage import (
    _compact_metrics_files,
    _rotate_old_metrics,
    _trim_current_metrics,
    get_repo_name,
    write_metrics,
)

# Import system resources module
from .system import (
    BLOAT_ARTIFACT_EXTENSIONS,
    ArtifactFileEntry,
    DiskBloatResult,
    LargeFileEntry,
    StatesDirEntry,
    _get_build_artifact_sizes,
    _get_disk_usage,
    _get_gh_rate_limits,
    _get_memory_usage_linux,
    _get_memory_usage_macos,
    detect_disk_bloat,
    get_system_resources,
)

# Import test metrics module
from .test_metrics import (
    _count_pattern_in_paths,
    _count_tests_by_framework,
    _detect_kani_coverage,
    _detect_lean_coverage,
    _detect_nn_verification,
    _detect_orphaned_python_tests,
    _detect_smt_coverage,
    _detect_tla_coverage,
    _get_recent_test_results,
    _get_workspace_member_dirs,
    _is_cargo_invocation_error,
    _load_pulse_ignore,
    get_proof_coverage,
    get_test_coverage,
    get_test_status,
)

# Re-export Result and subprocess utilities for backwards compatibility
try:
    from ai_template_scripts.result import Result
    from ai_template_scripts.subprocess_utils import run_cmd, run_cmd_with_retry
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.result import Result
    from ai_template_scripts.subprocess_utils import run_cmd, run_cmd_with_retry

# Backward compatibility aliases for underscore-prefixed names
# Tests reference these with the leading underscore
_DEFAULT_THRESHOLDS = DEFAULT_THRESHOLDS
_BLOCKER_PATTERN = BLOCKER_PATTERN
_ISSUE_REF_PATTERN = ISSUE_REF_PATTERN
_CRASH_LOG_PATTERN = CRASH_LOG_PATTERN
_WORKER_LOG_PATTERN = WORKER_LOG_PATTERN
_GIT_DEP_PATTERN = GIT_DEP_PATTERN
_CONFIG_PATHS = CONFIG_PATHS
_KNOWN_SECTIONS = KNOWN_SECTIONS
_KNOWN_LARGE_FILES_KEYS = KNOWN_LARGE_FILES_KEYS
_KNOWN_RUNTIME_KEYS = KNOWN_RUNTIME_KEYS

# Import core orchestration functions
from .core import (
    collect_metrics,
    main,
    metrics_to_broadcast,
    pulse_once,
    pulse_watch,
)

# Public API - this list must match the original pulse.py __all__
# Functions will be added as modules are extracted
__all__ = [
    # Constants
    "METRICS_DIR",
    "FLAGS_DIR",
    "MAX_METRICS_PER_DAY",
    "METRICS_RETENTION_DAYS",
    "METRICS_ARCHIVE_DIR",
    "THRESHOLDS",
    "LONG_RUNNING_PROCESS_NAMES",
    # Re-exports from other modules
    "Result",
    "run_cmd",
    "run_cmd_with_retry",
    # Internal constants (for test compatibility)
    "EXCLUDE_DIRS",
    "EXCLUDE_GLOB_PATTERNS",
    "FIND_EXCLUDE",
    "FIND_PRUNE",
    "GREP_EXCLUDE",
    "GREP_EXCLUDE_DIRS",
    "BLOCKER_PATTERN",
    "ISSUE_REF_PATTERN",
    "CRASH_LOG_PATTERN",
    "WORKER_LOG_PATTERN",
    "GIT_DEP_PATTERN",
    "DEFAULT_THRESHOLDS",
    "LONG_RUNNING_EXCLUDE_PATTERNS",
    "CONFIG_PATHS",
    "KNOWN_SECTIONS",
    "KNOWN_LARGE_FILES_KEYS",
    "KNOWN_RUNTIME_KEYS",
    # Config
    "LARGE_FILE_EXCLUDE_PATTERNS",
    "SKIP_ORPHANED_TESTS",
    "_dir_to_regex",
    "_warn_unknown_keys",
    "_load_config",
    "_apply_config",
    # System resources (from system.py)
    "BLOAT_ARTIFACT_EXTENSIONS",
    "LargeFileEntry",
    "ArtifactFileEntry",
    "StatesDirEntry",
    "DiskBloatResult",
    "get_system_resources",
    "detect_disk_bloat",
    "_get_memory_usage_macos",
    "_get_memory_usage_linux",
    "_get_disk_usage",
    "_get_build_artifact_sizes",
    "_get_gh_rate_limits",
    # Code metrics (from code_metrics.py)
    "count_lines_by_type",
    "find_large_files",
    "find_forbidden_ci",
    "_resolve_root",
    "_relative_path",
    "_path_is_excluded",
    "_matches_exclude_pattern",
    # Test metrics (from test_metrics.py)
    "get_test_status",
    "get_test_coverage",
    "get_proof_coverage",
    "_count_tests_by_framework",
    "_count_pattern_in_paths",
    "_detect_orphaned_python_tests",
    "_load_pulse_ignore",
    "_is_cargo_invocation_error",
    "_get_workspace_member_dirs",
    "_detect_kani_coverage",
    "_detect_tla_coverage",
    "_detect_lean_coverage",
    "_detect_smt_coverage",
    "_detect_nn_verification",
    # Issue metrics (from issue_metrics.py)
    "get_issue_counts",
    "get_issue_velocity",
    "get_issues_reopened",
    "get_blocked_missing_reason",
    "get_blocked_issue_list",
    "get_stale_blockers",
    "get_long_blocked_issues",
    "_empty_issue_counts",
    "_get_repo_owner_and_name",
    "_has_in_progress_label",
    "_get_issue_counts_graphql",
    "_get_closed_issue_count_graphql",
    "_is_graphql_rate_limited",
    "_get_open_issue_counts_rest",
    "_get_issue_counts_rest",
    "_should_prefer_rest_issue_counts",
    "_unpack_issue_counts",
    "_count_issues_within_days",
    # Git metrics (from git_metrics.py)
    "get_git_status",
    "get_code_quality",
    "get_consolidation_debt",
    "get_doc_claim_status",
    "get_outdated_git_deps",
    "count_template_lines",
    "_detect_git_operation_in_progress",
    "_count_todo_comments",
    "_compute_test_code_ratio",
    "_compute_python_type_coverage",
    "_format_drift_reason",
    "_get_repo_head_rev",
    "_get_commit_staleness",
    # Session metrics (from session_metrics.py)
    "get_active_sessions",
    "get_active_session_details",
    "get_untraceable_failures",
    "_STATUS_OPTIONAL_FIELDS",
    "INFRASTRUCTURE_EXIT_CODES",
    "INFRASTRUCTURE_ERROR_PATTERNS",
    "_parse_pid",
    "_read_pid_file",
    "_read_status_pid",
    "_read_status_data",
    "_pid_command_line",
    "_pid_looks_like_looper",
    "_iter_role_status_files",
    "_get_worker_log_info",
    "_is_infrastructure_failure",
    # Storage (from storage.py)
    "write_metrics",
    "get_repo_name",
    "_rotate_old_metrics",
    "_trim_current_metrics",
    "_compact_metrics_files",
    # Flags (from flags.py)
    "check_thresholds",
    "write_flags",
    "_flag_summary",
    # Process metrics (from process_metrics.py)
    "get_long_running_processes",
    "get_recent_crashes",
    "_args_contain_repo_path",
    "_get_process_cwd",
    "_infer_repo_from_path",
    "_parse_etime",
    "_classify_crash",
    # Output (from output.py)
    "_print_code_status",
    "_print_complexity_status",
    "_print_issue_status",
    "_print_proof_status",
    "_print_system_status",
    "_print_test_status",
    # Dirty tracking (from dirty_tracking.py)
    "_is_rename_status",
    "_normalize_status_path",
    "_is_pulse_output_path",
    "_identify_file_modifiers",
    "_get_file_mtimes",
    "_save_dirty_snapshot",
    "_format_dirty_diagnostics",
    "_get_porcelain_status",
    "_get_dirty_source_files",
    "_get_dirty_file_fingerprints",
    "_detect_dirty_file_changes",
    "_snapshot_repo_root",
]

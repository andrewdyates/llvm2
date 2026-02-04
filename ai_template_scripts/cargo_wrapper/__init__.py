# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Cargo wrapper - Serializes cargo builds per-repo with mutex and orphan cleanup.

Features:
- Per-repo serialization prevents concurrent cargo builds within a workspace (OOM, lock deadlocks)
- Different repos can build concurrently - isolation at repo level, not machine level
- Separate build/test locks by default; set [limits].max_concurrent_cargo=1 to share
- Configurable timeouts per command type (build=60m, test=10m, kani=30m by default)
- Re-entrant for nested calls: if lock is held by an ancestor process, bypasses locking
- Orphan process cleanup for stale cargo/rustc processes
- PID reuse detection via process start time comparison
- Retry loop detection warns on repeated failures without code changes

Lock separation (see #937, default when max_concurrent_cargo=2):
- Build lock: cargo build/check/clippy/doc/run
- Test lock: cargo test/bench/kani/zani/miri
- Workers can compile while Prover runs long test suites

Shared lock (max_concurrent_cargo=1):
- Build and test commands share the same lock file

Public API (library usage):
    from ai_template_scripts.cargo_wrapper import (
        LOCK_ACQUIRE_TIMEOUT,      # Lock acquisition timeout
        BUILD_TIMEOUT,             # Build command timeout (1 hour)
        TEST_TIMEOUT,              # Test command timeout (10 minutes)
        KANI_TIMEOUT,              # Shorter timeout for cargo kani/zani (30 min)
        STALE_PROCESS_AGE,         # Threshold for stale process detection
        LOCK_KIND_BUILD,           # Lock kind for build commands
        LOCK_KIND_TEST,            # Lock kind for test commands
        get_lock_kind_for_command, # Determine lock kind for a command
        get_timeout_config,        # Get timeout configuration
        is_lock_stale,             # Check if lock is stale
        acquire_lock,              # Acquire the build lock
        release_lock,              # Release the build lock
        force_release_stale_lock,  # Force release a stale lock
        cleanup_orphans,           # Kill stale cargo/rustc processes
        find_real_cargo,           # Find the real cargo binary
        get_lock_holder_info,      # Get info about current lock holder
    )

CLI usage:
    This script replaces 'cargo' in PATH. All cargo commands pass through it.

CANONICAL SOURCE: ayates_dbx/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Andrew Yates <ayates@dropbox.com>
Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

# Import _state for attribute proxying
from . import _state

# Re-export public API from submodules
from .cli import _has_test_filter_arg, main
from .constants import (
    BUILD_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_CARGO,
    KANI_TIMEOUT,
    LOCK_ACQUIRE_TIMEOUT,
    LOCK_BASENAMES,
    LOCK_KIND_BUILD,
    LOCK_KIND_TEST,
    LOCK_KINDS,
    MAX_LOG_LINES,
    MAX_STALE_RELEASE_ATTEMPTS,
    STALE_PROCESS_AGE,
    STATUS_INTERVAL,
    TEST_LOCK_SUBCOMMANDS,
    TEST_TIMEOUT,
    TIMEOUT_CONFIG_PATHS,
)
from .env import get_env_context, get_git_commit, get_repo_identifier
from .executor import (
    _make_process_group,
    atexit_handler,
    find_built_binary,
    find_real_cargo,
    kill_child,
    parse_cargo_run_args,
    run_binary,
    run_cargo,
    signal_handler,
)
from .lock import (
    acquire_lock,
    cleanup_stale_temp_files,
    force_release_stale_lock,
    get_builds_log,
    get_lock_dir,
    get_lock_file,
    get_lock_holder_info,
    get_lock_meta,
    get_orphans_log,
    init_lock_paths,
    is_lock_stale,
    release_lock,
    set_lock_kind,
)
from .logging import (
    check_retry_loop,
    log_build,
    log_orphan,
    log_stderr,
    now_iso,
    rotate_log_file,
)
from .processes import (
    cleanup_orphans,
    find_cargo_processes,
    get_process_parent,
    get_process_start_time,
    is_ancestor_of_self,
    parse_etime,
)
from .timeouts import (
    get_cargo_subcommand,
    get_limits_config,
    get_lock_kind_for_command,
    get_timeout_config,
    select_cargo_timeout,
)

__all__ = [
    # Internal (exposed for testing)
    "_has_test_filter_arg",
    "_make_process_group",
    # Constants
    "LOCK_ACQUIRE_TIMEOUT",
    "BUILD_TIMEOUT",
    "TEST_TIMEOUT",
    "KANI_TIMEOUT",
    "DEFAULT_MAX_CONCURRENT_CARGO",
    "STALE_PROCESS_AGE",
    "STATUS_INTERVAL",
    "MAX_LOG_LINES",
    "MAX_STALE_RELEASE_ATTEMPTS",
    "LOCK_KIND_BUILD",
    "LOCK_KIND_TEST",
    "LOCK_KINDS",
    "LOCK_BASENAMES",
    "TEST_LOCK_SUBCOMMANDS",
    "TIMEOUT_CONFIG_PATHS",
    # Environment
    "get_repo_identifier",
    "get_env_context",
    "get_git_commit",
    # Timeouts
    "get_cargo_subcommand",
    "get_lock_kind_for_command",
    "get_limits_config",
    "get_timeout_config",
    "select_cargo_timeout",
    # Lock management
    "init_lock_paths",
    "set_lock_kind",
    "get_lock_dir",
    "get_lock_file",
    "get_lock_meta",
    "get_builds_log",
    "get_orphans_log",
    "is_lock_stale",
    "acquire_lock",
    "release_lock",
    "force_release_stale_lock",
    "cleanup_stale_temp_files",
    "get_lock_holder_info",
    # Processes
    "get_process_start_time",
    "get_process_parent",
    "is_ancestor_of_self",
    "find_cargo_processes",
    "parse_etime",
    "cleanup_orphans",
    # Logging
    "now_iso",
    "log_stderr",
    "rotate_log_file",
    "log_orphan",
    "log_build",
    "check_retry_loop",
    # Executor
    "find_real_cargo",
    "run_cargo",
    "run_binary",
    "kill_child",
    "signal_handler",
    "atexit_handler",
    "parse_cargo_run_args",
    "find_built_binary",
    # CLI
    "main",
]

# Module wrapper to support mutable state attribute assignment.
# Python's PEP 562 only supports __getattr__, not __setattr__ for modules.
# We use a ModuleWrapper class to intercept attribute assignment.

import sys
import types

# State attributes that can be read/written via module attribute access
_STATE_ATTRS = {
    "LOCK_DIR",
    "LOCK_FILE",
    "LOCK_META",
    "BUILDS_LOG",
    "ORPHANS_LOG",
    "LOCK_KIND",
    "TIMEOUT_CONFIG",
    "LIMITS_CONFIG",
    "_lock_held",
    "_child_process",
    "_child_pgid",
}


class _ModuleWrapper(types.ModuleType):
    """Module wrapper that intercepts attribute access to _state."""

    def __init__(self, wrapped: types.ModuleType) -> None:
        # Copy essential module attributes
        for attr in (
            "__name__",
            "__doc__",
            "__file__",
            "__loader__",
            "__package__",
            "__path__",
            "__spec__",
        ):
            try:
                setattr(self, attr, getattr(wrapped, attr))
            except AttributeError:
                pass
        self._wrapped = wrapped
        self._state = _state  # Keep reference to _state module

    def __getattr__(self, name: str) -> object:
        if name in _STATE_ATTRS:
            return getattr(self._state, name)
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in (
            "_wrapped",
            "_state",
            "__name__",
            "__doc__",
            "__file__",
            "__loader__",
            "__package__",
            "__path__",
            "__spec__",
        ):
            super().__setattr__(name, value)
        elif name in _STATE_ATTRS:
            setattr(self._state, name, value)
        else:
            setattr(self._wrapped, name, value)

    def __delattr__(self, name: str) -> None:
        # Support unittest.mock.patch cleanup
        if name in _STATE_ATTRS:
            # Can't delete state attrs - just reset to None
            setattr(self._state, name, None)
        elif hasattr(self._wrapped, name):
            delattr(self._wrapped, name)
        else:
            # Ignore delattr for attributes that don't exist
            pass

    def __dir__(self) -> list[str]:
        return list(set(dir(self._wrapped)) | _STATE_ATTRS)


# Replace this module with a wrapper instance
_current_module = sys.modules[__name__]
sys.modules[__name__] = _ModuleWrapper(_current_module)

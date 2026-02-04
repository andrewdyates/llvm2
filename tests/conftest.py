# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Pytest configuration and shared fixtures.

Patterns documented in designs/2026-02-02-test-infrastructure.md:
- Autouse fixtures with nodeid filtering for module-level state
- Before/after cleanup (clear before yield AND after for failure safety)
- Save/restore for global state isolation
- Subprocess mocking via ai_template_scripts.test_utils.make_subprocess_result

== Autospec vs Plain Mock (#2051) ==

Use `autospec=True` to catch API drift: if the real function's signature changes,
tests fail immediately rather than silently accepting invalid calls.

When to use autospec:
  1. Mocking functions with known signatures (run_cmd, subprocess.run)
  2. Tests that verify call arguments
  3. Integration points where API stability matters

When NOT to use autospec:
  1. Mocking constants/paths (just use monkeypatch.setattr directly)
  2. Replacing modules with test doubles
  3. Complex conditional behavior that can't use side_effect

Examples:

  # BAD - won't catch if run_cmd adds required parameter
  monkeypatch.setattr(module, "run_cmd", lambda cmd, **kw: CmdResult(...))

  # GOOD - validates signature matches real function
  mock = create_autospec(run_cmd, return_value=CmdResult(...))
  monkeypatch.setattr(module, "run_cmd", mock)

  # GOOD - custom logic with signature validation
  mock = create_autospec(run_cmd, side_effect=lambda cmd, **kw: ...)
  monkeypatch.setattr(module, "run_cmd", mock)

Helper: make_run_cmd_mock() creates autospec mocks for run_cmd patterns.

== Monkeypatch Location (#2048) ==

Patch on the module under test, not on the stdlib module. This survives refactoring.

  # BAD - fragile, breaks when module structure changes
  monkeypatch.setattr(subprocess, "run", mock_run)

  # GOOD - robust, survives module refactoring
  monkeypatch.setattr(my_module.subprocess, "run", mock_run)
  # Or for modules that import the function:
  monkeypatch.setattr(my_module, "run", mock_run)

Why: When you patch `subprocess.run`, you're patching the original. If the module
under test gets refactored (split, renamed), the patch still targets the original
but the code may be looking elsewhere. Patching on the module under test ensures
the mock is where the code actually looks.

See: designs/2026-02-02-monkeypatch-best-practices.md
"""

import contextlib
import subprocess
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, create_autospec

import pytest

import ai_template_scripts.cargo_wrapper as cargo_wrapper
from ai_template_scripts.subprocess_utils import CmdResult, run_cmd
from looper.issue_manager import IssueManager

# ==== Test Environment Isolation (#2184) ====
# Clear AI-specific env vars to prevent execution context leaking into tests.
# This fixes issues where tests fail in AI sessions because AI_PROJECT, AI_ROLE,
# or PROJECT_ROOT override mocked values.

# Module-level constant for AI environment variables that must be cleared.
# Exported for documentation and potential reuse in test utilities.
AI_ENVIRONMENT_VARS = [
    "AI_PROJECT",
    "AI_ROLE",
    "AI_ITERATION",
    "AI_SESSION",
    "AI_WORKER_ID",
    "AI_MACHINE",
    "AI_MACHINE_PREFIX",
    "AI_CODER",
    "AI_FLEET_CONFIG_DIR",
    "AIT_VERSION",
    "PROJECT_ROOT",
]


@pytest.fixture(autouse=True)
def clean_ai_environment(monkeypatch):
    """Clear AI-specific env vars to prevent execution context leaking into tests.

    Fixes #2179, #2181, #2182: Tests that mock git remote, PROJECT_ROOT, or
    other configuration were failing because env vars from the AI execution
    context took precedence over the mocks.

    Uses the module-level AI_ENVIRONMENT_VARS constant.
    """
    for var in AI_ENVIRONMENT_VARS:
        monkeypatch.delenv(var, raising=False)


# Skip deprecated re-export files that cause tests to run twice (#1375)
# These files import and re-export test classes for backwards compatibility,
# but pytest collects the tests from both the source and re-export files.
collect_ignore = [
    # test_looper/test_runner.py removed in #1787 - split into test_runner_*.py
    # test_looper/test_issue_manager.py removed in #1719 - split into test_issue_manager_*.py
    # test_context.py removed in #1721 - split into context/ package
]


# ==== Autospec helpers (#2051) ====


def make_run_cmd_mock(
    handler: Callable[[list[str]], CmdResult] | None = None,
    default_result: CmdResult | None = None,
) -> MagicMock:
    """Create an autospec mock for run_cmd with optional custom handler.

    This creates a signature-validated mock that catches API drift. If run_cmd's
    signature changes, tests using this mock will fail immediately.

    Args:
        handler: Optional function(cmd) -> CmdResult for custom behavior.
                 Receives only the cmd argument for simplicity.
        default_result: Default CmdResult when handler is None or returns None.

    Returns:
        MagicMock with autospec of run_cmd, ready for monkeypatch.setattr.

    Examples:
        # Simple fixed return
        mock = make_run_cmd_mock(default_result=CmdResult(0, "output", ""))
        monkeypatch.setattr(module, "run_cmd", mock)

        # Conditional behavior
        def handler(cmd):
            if cmd[0] == "git":
                return CmdResult(0, "main", "")
            return CmdResult(1, "", "unknown command")
        mock = make_run_cmd_mock(handler=handler)
    """
    if default_result is None:
        default_result = CmdResult(returncode=0, stdout="", stderr="")

    def side_effect(cmd, timeout=30, cwd=None):
        if handler is not None:
            result = handler(cmd)
            if result is not None:
                return result
        return default_result

    return create_autospec(run_cmd, side_effect=side_effect)


def make_subprocess_run_mock(
    handler: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    default_result: subprocess.CompletedProcess | None = None,
) -> MagicMock:
    """Create an autospec mock for subprocess.run with optional custom handler.

    Args:
        handler: Optional function(cmd) -> CompletedProcess for custom behavior.
        default_result: Default CompletedProcess when handler is None.

    Returns:
        MagicMock with autospec of subprocess.run, ready for monkeypatch.setattr.
    """
    if default_result is None:
        default_result = subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def side_effect(cmd, *args, **kwargs):
        if handler is not None:
            result = handler(cmd)
            if result is not None:
                return result
        return default_result

    return create_autospec(subprocess.run, side_effect=side_effect)


@pytest.fixture(autouse=True)
def force_json_to_text_no_color(request, monkeypatch):
    """Ensure json_to_text uses FORCE_COLOR=0 for deterministic output."""
    if "test_json_to_text_" not in request.node.nodeid:
        yield
        return
    monkeypatch.setenv("FORCE_COLOR", "0")
    import ai_template_scripts.json_to_text as json_to_text  # noqa: PLC0415 - dynamic import after env setup

    original_state = {
        "USE_COLORS": json_to_text.USE_COLORS,
        "BLUE": json_to_text.BLUE,
        "GREEN": json_to_text.GREEN,
        "YELLOW": json_to_text.YELLOW,
        "RED": json_to_text.RED,
        "CYAN": json_to_text.CYAN,
        "MAGENTA": json_to_text.MAGENTA,
        "BOLD": json_to_text.BOLD,
        "DIM": json_to_text.DIM,
        "RESET": json_to_text.RESET,
    }

    json_to_text.USE_COLORS = False
    json_to_text.BLUE = ""
    json_to_text.GREEN = ""
    json_to_text.YELLOW = ""
    json_to_text.RED = ""
    json_to_text.CYAN = ""
    json_to_text.MAGENTA = ""
    json_to_text.BOLD = ""
    json_to_text.DIM = ""
    json_to_text.RESET = ""
    try:
        yield
    finally:
        for name, value in original_state.items():
            setattr(json_to_text, name, value)


@pytest.fixture(autouse=True)
def clear_pending_tool_uses(request):
    """Clear pending_tool_uses dict before each json_to_text test (#2059).

    Prevents test pollution from module-level state that persists across tests.
    Only applies to test_json_to_text_* test files.
    """
    if "test_json_to_text_" not in request.node.nodeid:
        yield
        return
    from ai_template_scripts.json_to_text import pending_tool_uses

    pending_tool_uses.clear()
    yield
    pending_tool_uses.clear()


@pytest.fixture(autouse=True)
def clear_iteration_issue_cache(request):
    """Clear IterationIssueCache before each issue_context test (#2058).

    The cache is a class-level singleton that persists across tests.
    Without clearing, mocked run_gh_command calls won't be invoked
    because the cache returns previously cached data.
    Only applies to test_issue_context* test files.
    """
    if "test_issue_context" not in request.node.nodeid:
        yield
        return
    from looper.context.issue_context import IterationIssueCache

    IterationIssueCache.clear()
    yield
    IterationIssueCache.clear()


# Default timeout for run_subprocess calls in tests (seconds)
# Prevents tests from hanging indefinitely; defense in depth beyond pytest timeout
TEST_SUBPROCESS_TIMEOUT = 30


def run_subprocess(*args, **kwargs):
    """Run subprocess with a default timeout unless overridden."""
    kwargs.setdefault("timeout", TEST_SUBPROCESS_TIMEOUT)
    return subprocess.run(*args, **kwargs)


@pytest.fixture
def lock_env(tmp_path):
    """Set up isolated lock environment for cargo_wrapper tests.

    Saves and restores all cargo_wrapper global state:
    - LOCK_DIR, LOCK_FILE, LOCK_META
    - BUILDS_LOG, ORPHANS_LOG
    - _lock_held flag

    Yields:
        tmp_path: Temporary directory for test lock files
    """
    # Save original values
    orig_lock_dir = cargo_wrapper.LOCK_DIR
    orig_lock_file = cargo_wrapper.LOCK_FILE
    orig_lock_meta = cargo_wrapper.LOCK_META
    orig_builds_log = cargo_wrapper.BUILDS_LOG
    orig_orphans_log = cargo_wrapper.ORPHANS_LOG
    orig_lock_held = cargo_wrapper._lock_held
    orig_lock_kind = cargo_wrapper.LOCK_KIND

    # Set test paths
    cargo_wrapper.LOCK_DIR = tmp_path
    cargo_wrapper.set_lock_kind(cargo_wrapper.LOCK_KIND_BUILD)
    cargo_wrapper.BUILDS_LOG = tmp_path / "builds.log"
    cargo_wrapper.ORPHANS_LOG = tmp_path / "orphans.log"
    cargo_wrapper._lock_held = False

    yield tmp_path

    # Restore original values
    cargo_wrapper.LOCK_DIR = orig_lock_dir
    cargo_wrapper.LOCK_FILE = orig_lock_file
    cargo_wrapper.LOCK_META = orig_lock_meta
    cargo_wrapper.BUILDS_LOG = orig_builds_log
    cargo_wrapper.ORPHANS_LOG = orig_orphans_log
    cargo_wrapper._lock_held = orig_lock_held
    cargo_wrapper.LOCK_KIND = orig_lock_kind


@contextlib.contextmanager
def lock_env_context(tmp_path: Path):
    """Context manager for isolated lock environment (for hypothesis tests).

    Same as lock_env fixture but as a context manager for use with
    hypothesis property tests that can't use fixtures directly.

    Args:
        tmp_path: Temporary directory for test lock files

    Yields:
        tmp_path: The same temporary directory
    """
    # Save original values
    orig_lock_dir = cargo_wrapper.LOCK_DIR
    orig_lock_file = cargo_wrapper.LOCK_FILE
    orig_lock_meta = cargo_wrapper.LOCK_META
    orig_builds_log = cargo_wrapper.BUILDS_LOG
    orig_orphans_log = cargo_wrapper.ORPHANS_LOG
    orig_lock_held = cargo_wrapper._lock_held
    orig_lock_kind = cargo_wrapper.LOCK_KIND

    # Set test paths
    cargo_wrapper.LOCK_DIR = tmp_path
    cargo_wrapper.set_lock_kind(cargo_wrapper.LOCK_KIND_BUILD)
    cargo_wrapper.BUILDS_LOG = tmp_path / "builds.log"
    cargo_wrapper.ORPHANS_LOG = tmp_path / "orphans.log"
    cargo_wrapper._lock_held = False

    try:
        yield tmp_path
    finally:
        # Restore original values
        cargo_wrapper.LOCK_DIR = orig_lock_dir
        cargo_wrapper.LOCK_FILE = orig_lock_file
        cargo_wrapper.LOCK_META = orig_lock_meta
        cargo_wrapper.BUILDS_LOG = orig_builds_log
        cargo_wrapper.ORPHANS_LOG = orig_orphans_log
        cargo_wrapper._lock_held = orig_lock_held
        cargo_wrapper.LOCK_KIND = orig_lock_kind


# ==== IssueManager fixtures ====


@pytest.fixture
def worker_issue_manager() -> IssueManager:
    """Create IssueManager with worker role in current directory."""
    return IssueManager(Path.cwd(), "worker")


@pytest.fixture
def manager_issue_manager() -> IssueManager:
    """Create IssueManager with manager role in current directory."""
    return IssueManager(Path.cwd(), "manager")


@pytest.fixture
def tmp_issue_manager(tmp_path: Path) -> IssueManager:
    """Create IssueManager with worker role in temp directory."""
    return IssueManager(tmp_path, "worker")


# ==== Crash analysis fixtures ====


@pytest.fixture
def use_fixed_thresholds(monkeypatch):
    """Force crash_analysis tests to use fixed thresholds.

    The adaptive threshold feature computes dynamic thresholds from git history.
    This causes tests to fail when mocked git output produces unexpected
    baseline samples. By patching load_config to return explicit fixed thresholds,
    we preserve the original test behavior.

    Usage: Include this fixture in test functions that depend on fixed thresholds.
    """
    import ai_template_scripts.crash_analysis as crash_analysis

    config_with_fixed = {
        "crash_analysis": {
            "warning_threshold_rate": crash_analysis.FAILURE_RATE_WARNING,
            "critical_threshold_rate": crash_analysis.FAILURE_RATE_CRITICAL,
        }
    }
    # Patch load_config (imported from config_loader into crash_analysis.__init__)
    monkeypatch.setattr(
        crash_analysis, "load_config", lambda: (config_with_fixed, None)
    )

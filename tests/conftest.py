"""
Pytest configuration and shared fixtures.

Andrew Yates <ayates@dropbox.com>
Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import contextlib
import sys
from pathlib import Path

import pytest

# Add scripts directory to path for cargo_wrapper imports
sys.path.insert(0, str(Path(__file__).parent.parent / "ai_template_scripts"))

import cargo_wrapper  # noqa: E402


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

    # Set test paths
    cargo_wrapper.LOCK_DIR = tmp_path
    cargo_wrapper.LOCK_FILE = tmp_path / "lock.pid"
    cargo_wrapper.LOCK_META = tmp_path / "lock.json"
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

    # Set test paths
    cargo_wrapper.LOCK_DIR = tmp_path
    cargo_wrapper.LOCK_FILE = tmp_path / "lock.pid"
    cargo_wrapper.LOCK_META = tmp_path / "lock.json"
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

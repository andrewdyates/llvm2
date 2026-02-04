# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Constants for cargo_wrapper package."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "LOCK_ACQUIRE_TIMEOUT",
    "BUILD_TIMEOUT",
    "TEST_TIMEOUT",
    "KANI_TIMEOUT",
    "STALE_PROCESS_AGE",
    "STATUS_INTERVAL",
    "MAX_LOG_LINES",
    "MAX_STALE_RELEASE_ATTEMPTS",
    "DEFAULT_MAX_CONCURRENT_CARGO",
    "LOCK_KIND_BUILD",
    "LOCK_KIND_TEST",
    "LOCK_KINDS",
    "LOCK_BASENAMES",
    "TEST_LOCK_SUBCOMMANDS",
    "TIMEOUT_CONFIG_PATHS",
    "_CARGO_GLOBAL_FLAGS_WITH_VALUES",
    "_ARGS_WITH_VALUES",
]

# Timeouts in seconds
LOCK_ACQUIRE_TIMEOUT = 30 * 60  # 30 minutes to acquire lock
BUILD_TIMEOUT = 60 * 60  # 1 hour (3600s) per build; exit code 124 on timeout
TEST_TIMEOUT = 10 * 60  # 10 minutes per test/bench/miri by default
KANI_TIMEOUT = 30 * 60  # 30 minutes for cargo kani/zani (proofs can hang indefinitely)
STALE_PROCESS_AGE = 2 * 60 * 60  # 2 hours = stale lock or orphan process

STATUS_INTERVAL = 60  # Print status every 60 seconds while waiting
MAX_LOG_LINES = 1000  # Rotate logs at this size
MAX_STALE_RELEASE_ATTEMPTS = 5  # Limit retries when releasing stale locks
DEFAULT_MAX_CONCURRENT_CARGO = 2  # Max concurrent cargo operations per repo

_CARGO_GLOBAL_FLAGS_WITH_VALUES = {
    "-Z",
    "--color",
    "--config",
    "--manifest-path",
    "--message-format",
    "--target-dir",
}

# Lock kinds
LOCK_KIND_BUILD = "build"
LOCK_KIND_TEST = "test"
LOCK_KINDS = (LOCK_KIND_BUILD, LOCK_KIND_TEST)

LOCK_BASENAMES = {
    LOCK_KIND_BUILD: "lock",
    LOCK_KIND_TEST: "lock.test",
}

TEST_LOCK_SUBCOMMANDS = {
    "bench",
    "kani",
    "miri",
    "test",
    "zani",
}

TIMEOUT_CONFIG_PATHS = [
    Path("cargo_wrapper.toml"),
    Path(".cargo_wrapper.toml"),
]

# Args that consume the next positional value in _has_test_filter_arg
_ARGS_WITH_VALUES = {
    "-p",
    "--package",
    "--features",
    "-j",
    "--jobs",
    "--target",
    "--test",
    "--bench",
}

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Shared test utilities for pytest-based test suites.

This module provides common utilities used across test suites:
- Subprocess execution with default timeouts
- Mock subprocess result factory
- Color stripping helpers for deterministic output

Usage in conftest.py:
    from ai_template_scripts.test_utils import (
        TEST_SUBPROCESS_TIMEOUT,
        run_subprocess,
        make_subprocess_result,
    )

Synced from ai_template - do not edit in target repos.
"""

import subprocess
from unittest.mock import MagicMock

__all__ = [
    "TEST_SUBPROCESS_TIMEOUT",
    "run_subprocess",
    "make_subprocess_result",
]

# Default timeout for subprocess calls in tests (seconds)
# Prevents tests from hanging indefinitely; defense in depth beyond pytest timeout
TEST_SUBPROCESS_TIMEOUT = 30


def run_subprocess(*args, **kwargs) -> subprocess.CompletedProcess:
    """Run subprocess with a default timeout unless overridden.

    Wrapper around subprocess.run() that applies a default timeout to prevent
    tests from hanging indefinitely.

    Contracts:
        REQUIRES: args[0] is a valid subprocess command (list[str] or str)
        ENSURES: Returns subprocess.CompletedProcess on success
        RAISES: subprocess.TimeoutExpired if timeout exceeded
        RAISES: FileNotFoundError if command not found

    Args:
        *args: Passed to subprocess.run()
        **kwargs: Passed to subprocess.run(); timeout defaults to TEST_SUBPROCESS_TIMEOUT

    Returns:
        subprocess.CompletedProcess result
    """
    kwargs.setdefault("timeout", TEST_SUBPROCESS_TIMEOUT)
    return subprocess.run(*args, **kwargs)


def make_subprocess_result(
    returncode: int = 0,
    stdout: str | bytes = "",
    stderr: str | bytes = "",
) -> MagicMock:
    """Create a mock subprocess result for testing.

    Reduces test boilerplate when mocking subprocess.run() calls.

    Contracts:
        REQUIRES: returncode is an int
        REQUIRES: stdout is str or bytes
        REQUIRES: stderr is str or bytes
        ENSURES: Returns MagicMock with returncode/stdout/stderr attributes

    Args:
        returncode: Exit code (default 0 for success)
        stdout: Standard output content (str or bytes)
        stderr: Standard error content (str or bytes)

    Returns:
        MagicMock configured as subprocess.CompletedProcess

    Example:
        >>> with patch('subprocess.run') as mock_run:
        ...     mock_run.return_value = make_subprocess_result(returncode=0, stdout="ok")
        ...     result = subprocess.run(['some', 'command'])
        ...     assert result.returncode == 0
    """
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Version helper for CLI scripts.

Provides consistent version output format across ai_template scripts.

Format: `{script_name} {git_hash} ({date})`

Example: `pulse.py abc1234 (2026-01-30)`

Design: designs/2026-01-30-cli-version-flag.md

Note: Intentionally single-line format, deviating from GNU standards
which specify multi-line output with copyright/license. This is documented
as an intentional simplification for ai_template.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

__all__ = ["get_version"]


def get_version(script_name: str) -> str:
    """Get version string for CLI scripts.

    Args:
        script_name: Canonical name of the script (not argv[0]).
                    Should be the filename without path.

    Returns:
        Version string in format: `{script_name} {git_hash} ({date})`

    Example:
        >>> get_version("pulse.py")
        'pulse.py abc1234 (2026-01-30)'
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_hash = result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        git_hash = "unknown"

    date = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{script_name} {git_hash} ({date})"

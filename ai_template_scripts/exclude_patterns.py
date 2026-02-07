#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Shared directory exclusion patterns for codebase analysis tools.

Single source of truth for directories to skip during filesystem traversal.
Used by: code_stats/, integration_audit.py, check_doc_claims.py, check_deps.py,
pulse/constants.py

Fixes #1798: Inconsistent SKIP_DIRS definitions across modules.

Pattern types:
- SKIP_DIRS: Exact directory name matches (fast, most common case)
- SKIP_GLOBS: fnmatch-style patterns for variable suffixes (e.g., .venv*)
- VENDORED_DIRS: Subset for external/vendored code detection only
"""

import fnmatch
import re
from typing import Iterable

# Directories to skip during codebase analysis (exact name match)
# This is the canonical list - all analysis tools should use this.
SKIP_DIRS: set[str] = {
    # Version control
    ".git",
    # Build artifacts
    "target",  # Rust
    "build",
    ".build",  # Some Rust toolchains
    "dist",
    "node_modules",  # Node.js
    # Python environments
    ".venv",
    "venv",
    "env",
    ".env",  # Environment dirs (not .env files)
    "site-packages",  # Python packages inside venvs
    # Caches
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    # Vendored/external code (see also VENDORED_DIRS below)
    "reference",
    "vendor",
    "third_party",
    "external",
    # IDE/editor
    ".idea",
    ".vscode",
    # ai_template specific
    ".claude",  # Claude Code configuration and plugins
    ".ai_template_self",  # ai_template management files
}

# Glob patterns for directories that may have variable suffixes
# Use with fnmatch.fnmatch() or find's -name option
SKIP_GLOBS: list[str] = [
    ".venv*",  # Matches .venv, .venv-py313, .venv-dev, etc.
    "*.egg-info",  # Python egg metadata directories
]

# Subset: directories containing vendored/external code
# Use when you only want to exclude external code, not build artifacts
VENDORED_DIRS: set[str] = {
    "reference",
    "vendor",
    "third_party",
    "external",
}


def should_skip_dir(name: str) -> bool:
    """Check if a directory name should be skipped (exact or glob match).

    Args:
        name: Directory name (not full path)

    Returns:
        True if directory should be skipped
    """
    if name in SKIP_DIRS:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in SKIP_GLOBS)


def should_skip_path(path_parts: Iterable[str]) -> bool:
    """Check if any path component should be skipped.

    Args:
        path_parts: Iterable of path components (e.g., Path.parts)

    Returns:
        True if path should be skipped
    """
    return any(should_skip_dir(part) for part in path_parts)


def get_find_exclude() -> str:
    """Get find(1) exclusion flags for root-level directories.

    Returns:
        String like '-not -path "./target/*" -not -path "./.venv/*" ...'
    """
    return " ".join(f'-not -path "./{d}/*"' for d in SKIP_DIRS)


def get_find_prune() -> str:
    """Get find(1) -prune expression for excluding directories by name anywhere.

    Returns:
        String like '-type d -name "target" -prune -o -type d -name ".venv*" -prune'
    """
    all_patterns = list(SKIP_DIRS) + SKIP_GLOBS
    return " -o ".join(f'-type d -name "{d}" -prune' for d in all_patterns)


def get_grep_exclude_dirs() -> str:
    """Get grep --exclude-dir flags.

    Returns:
        String like '--exclude-dir=target --exclude-dir=.venv* ...'
    """
    all_patterns = list(SKIP_DIRS) + SKIP_GLOBS
    return " ".join(f"--exclude-dir={d}" for d in all_patterns)


def _dir_to_regex(d: str) -> str:
    """Convert directory pattern to regex, escaping . and converting * to .*."""
    return d.replace(".", r"\.").replace("*", ".*")


def get_grep_exclude_regex() -> str:
    """Get regex pattern for filtering paths (e.g., with grep -v).

    Returns:
        Regex pattern matching excluded directory paths
    """
    all_patterns = list(SKIP_DIRS) + SKIP_GLOBS
    return "|".join(f"(^|/){_dir_to_regex(d)}/" for d in all_patterns)


# Pre-compiled regex for performance (use in loops)
GREP_EXCLUDE_REGEX = re.compile(get_grep_exclude_regex())


# Public API
__all__ = [
    # Core sets
    "SKIP_DIRS",
    "SKIP_GLOBS",
    "VENDORED_DIRS",
    # Helper functions
    "should_skip_dir",
    "should_skip_path",
    # Shell command builders
    "get_find_exclude",
    "get_find_prune",
    "get_grep_exclude_dirs",
    "get_grep_exclude_regex",
    # Pre-compiled patterns
    "GREP_EXCLUDE_REGEX",
]

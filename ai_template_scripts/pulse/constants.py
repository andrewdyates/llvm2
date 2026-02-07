#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Constants and default thresholds for pulse module.

This module provides path constants, default threshold values, exclusion patterns,
and pre-compiled regex patterns used throughout pulse.

Part of #404: pulse.py module split
"""

import re
import sys
from pathlib import Path

# Support import when running as part of package
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.exclude_patterns import (  # noqa: E402
    SKIP_DIRS as _BASE_SKIP_DIRS,
)
from ai_template_scripts.exclude_patterns import (
    SKIP_GLOBS as _BASE_SKIP_GLOBS,
)

# Directories
METRICS_DIR = Path("metrics")
FLAGS_DIR = Path(".flags")

# Metrics retention settings
MAX_METRICS_PER_DAY = 200  # ~5 per hour for 40 hours of operation
METRICS_RETENTION_DAYS = 7  # Compress/delete files older than this
METRICS_ARCHIVE_DIR = METRICS_DIR / "archive"

# Directories to exclude from analysis (from exclude_patterns.py, #1798)
# Converted to list for backwards compatibility with pulse consumers
EXCLUDE_DIRS = sorted(_BASE_SKIP_DIRS - {".git"})  # .git handled separately in grep

# Glob patterns for directories that may have suffixes (from exclude_patterns.py, #1798)
EXCLUDE_GLOB_PATTERNS = list(_BASE_SKIP_GLOBS)

# Pre-built exclusion patterns for find and grep
# FIND_EXCLUDE uses exact paths at root level
FIND_EXCLUDE = " ".join([f'-not -path "./{d}/*"' for d in EXCLUDE_DIRS])


# GREP_EXCLUDE uses regex - convert globs (* -> .*) for regex matching
# Match both start-of-path (^dir/) and nested (/dir/) patterns (#1338 fix)
# Escape . for regex (otherwise .venv matches avenv, etc.)
def _dir_to_regex(d: str) -> str:
    """Convert directory pattern to regex, escaping . and converting * to .*."""
    return d.replace(".", r"\.").replace("*", ".*")


GREP_EXCLUDE = "|".join(
    [f"(^|/){_dir_to_regex(d)}/" for d in EXCLUDE_DIRS + EXCLUDE_GLOB_PATTERNS]
)
# Upfront exclusion for grep (much faster than post-filtering with grep -v)
# Includes glob patterns for venv variants (#1233)
GREP_EXCLUDE_DIRS = " ".join(
    [f"--exclude-dir={d}" for d in EXCLUDE_DIRS + EXCLUDE_GLOB_PATTERNS + [".git"]]
)

# Prune-style exclusion for find: matches dir names ANYWHERE in the tree
# Unlike FIND_EXCLUDE which only matches at root level (e.g., ./.venv/*),
# this prunes directories by name wherever they appear (e.g., tests/foo/.venv/)
# Includes both exact matches (EXCLUDE_DIRS) and glob patterns (EXCLUDE_GLOB_PATTERNS)
FIND_PRUNE = " -o ".join(
    [f'-type d -name "{d}" -prune' for d in EXCLUDE_DIRS + EXCLUDE_GLOB_PATTERNS]
)

# Pre-compiled regex patterns (#1698: avoid re.compile() inside functions)
# Blocker pattern: Blocked: #123, Blocked: #123, #456, Blocked: dropbox-ai-prototypes/repo#123
BLOCKER_PATTERN = re.compile(
    r"[Bb]locked:\s*"  # Case-insensitive "Blocked:"
    r"((?:#\d+|[\w-]+/[\w-]+#\d+)(?:\s*,\s*(?:#\d+|[\w-]+/[\w-]+#\d+))*)"
)
# Sub-pattern to extract individual issue refs (#N)
ISSUE_REF_PATTERN = re.compile(r"#(\d+)")
# Failure log entry pattern (consistent with health_check.py)
CRASH_LOG_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Iteration \d+: (.+)",
)
# Worker log filename pattern
WORKER_LOG_PATTERN = re.compile(
    r"^worker_(?:(?P<worker_id>\d+)_)?iter_(?P<iteration>\d+)"
    r"(?:\.(?P<audit_round>\d+))?_.*\.jsonl$"
)
# Git dependency pattern for Cargo.toml - matches git deps with rev pinning
GIT_DEP_PATTERN = re.compile(
    r'git\s*=\s*"((?:https://|ssh://git@)github\.com/[^"]+)"[^}]*rev\s*=\s*"([a-f0-9]+)"',
    re.IGNORECASE | re.DOTALL,
)

# Default thresholds (can be overridden via pulse.toml)
DEFAULT_THRESHOLDS = {
    "max_file_lines": 1000,  # Notice tier - files getting large (Part of #2358)
    "max_file_lines_warning": 5000,  # Warning tier - files too large (Part of #2358)
    "max_complexity": 15,
    "max_files_over_limit": 3,  # Flag when this many files exceed warning tier
    "stale_issue_days": 7,
    "memory_warning_percent": 80,  # Flag when memory usage exceeds this
    "memory_critical_percent": 90,  # Critical flag
    "disk_warning_percent": 80,  # Flag when disk usage exceeds this
    "disk_critical_percent": 90,  # Critical flag
    "long_running_process_minutes": 120,  # Flag processes running > this (#922)
    "large_file_size_gb": 1,  # Flag individual files larger than this (#1477)
    "tests_dir_size_gb": 10,  # Flag tests/ directory larger than this (#1477)
    "states_dir_size_gb": 10,  # Flag TLA+ states/ directories larger than this (#1551)
    "reports_dir_size_mb": 50,  # Flag reports/ directory larger than this (#1695)
    # Git dependency staleness threshold (#1876)
    # Only flag deps as outdated if they exceed staleness criteria.
    # Threshold behavior: flag if days_old >= N OR commits_behind >= N.
    # Set BOTH to 0 to disable staleness filter (flag any outdated dep).
    "git_dep_stale_days": 3,  # Flag if pinned rev is >=N days old
    "git_dep_stale_commits": 10,  # Flag if pinned rev is >=N commits behind
    # GitHub API quota overflow threshold (#2362)
    # Flag when overflow events in last hour exceed this count
    # Set to 0 to flag on any overflow
    "gh_quota_overflow_threshold": 10,
}

# Processes to monitor for long-running detection (#922)
# These are known long-running tools used in verification/compilation
# NOTE: python removed (#2139) - too many false positives from daemons
# (looper.py, tab-title server, memory_watchdog). Monitor specific scripts
# via their tool names instead (z3, kani, etc.)
LONG_RUNNING_PROCESS_NAMES = [
    "cargo",
    "rustc",
    "cbmc",
    "kani",
    "z3",
    "cvc5",
    "lean",
]

# Exclude patterns for long-running process detection (#1288)
# These are known long-running but expected processes that should not trigger stuck_process
# Each pattern is matched against the full command line (case-insensitive)
# NOTE: Python-specific patterns removed in #2139 (python removed from LONG_RUNNING_PROCESS_NAMES).
LONG_RUNNING_EXCLUDE_PATTERNS: list[str] = [
    # Memory watchdog process (from homebrew) - expected to run continuously (#2146)
    "/opt/homebrew/Cellar/",
]

# Config file locations (checked in order, first found wins)
CONFIG_PATHS = [
    Path("pulse.toml"),
    Path("ai_template_scripts/pulse.toml"),
    Path(".pulse.toml"),
]

# Known config keys for validation (#1525)
KNOWN_SECTIONS = {"thresholds", "large_files", "runtime"}
KNOWN_LARGE_FILES_KEYS = {"exclude_patterns"}
KNOWN_RUNTIME_KEYS = {"skip_orphaned_tests"}

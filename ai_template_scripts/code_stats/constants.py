# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Constants for code complexity analysis."""

from ai_template_scripts.exclude_patterns import SKIP_DIRS

# Complexity thresholds
THRESHOLD_CYCLOMATIC = 10
THRESHOLD_COGNITIVE = 15
THRESHOLD_HIGH = 20  # "high" severity above this

# File extensions by language
LANG_EXTENSIONS = {
    "rust": [".rs"],
    "python": [".py"],
    "go": [".go"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
    "c": [".c", ".h"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "swift": [".swift"],
    "objc": [".m", ".mm"],
    "bash": [".sh", ".bash"],
}

__all__ = [
    "THRESHOLD_CYCLOMATIC",
    "THRESHOLD_COGNITIVE",
    "THRESHOLD_HIGH",
    "SKIP_DIRS",
    "LANG_EXTENSIONS",
]

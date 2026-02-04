# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
code_stats - Code complexity analysis using best-in-class tools.

This package analyzes cyclomatic/cognitive complexity to identify problematic code.
Uses an ensemble of language-specific tools:
- Rust: lizard
- Python: radon (best-in-class)
- Go: gocyclo (best-in-class)
- C/C++: pmccabe (preferred) or lizard
- TypeScript/JS: lizard
- Swift/Objective-C/Bash: lizard or line count

Install required tools: ./ai_template_scripts/install_dev_tools.sh
"""

from ai_template_scripts.code_stats.analyzers import (
    analyze_bash,
    analyze_c_cpp,
    analyze_go,
    analyze_objc,
    analyze_python,
    analyze_rust,
    analyze_swift,
    analyze_typescript,
)

# Base analyzer utilities
from ai_template_scripts.code_stats.analyzers.base import (
    _get_relative_path,
    _maybe_add_complexity_warning,
)

# C/C++ analyzer (pmccabe)
from ai_template_scripts.code_stats.analyzers.cpp import (
    _analyze_with_pmccabe,
    _count_c_sloc,
)

# Go analyzer (gocyclo)
from ai_template_scripts.code_stats.analyzers.go import (
    GocycloParsedLine,
    _count_go_sloc,
    _get_go_tools_string,
    _parse_gocyclo_line,
    _process_gocyclo_output,
)

# Re-export internal functions for test access
# Lizard-based analyzers
from ai_template_scripts.code_stats.analyzers.lizard import (
    LizardParsedLine,
    _analyze_with_lizard,
    _build_lizard_lang_args,
    _determine_lang_from_extension,
    _parse_lizard_csv_line,
)
from ai_template_scripts.code_stats.cli import main, print_summary
from ai_template_scripts.code_stats.constants import (
    LANG_EXTENSIONS,
    SKIP_DIRS,
    THRESHOLD_COGNITIVE,
    THRESHOLD_CYCLOMATIC,
    THRESHOLD_HIGH,
)
from ai_template_scripts.code_stats.core import analyze
from ai_template_scripts.code_stats.models import (
    AnalysisResult,
    FunctionMetric,
    LanguageSummary,
)
from ai_template_scripts.code_stats.utils import find_files, get_git_info, has_tool

# Re-export run_cmd for public API compatibility
from ai_template_scripts.subprocess_utils import run_cmd

__all__ = [
    # Constants
    "THRESHOLD_CYCLOMATIC",
    "THRESHOLD_COGNITIVE",
    "THRESHOLD_HIGH",
    "SKIP_DIRS",
    "LANG_EXTENSIONS",
    # Data classes
    "FunctionMetric",
    "LanguageSummary",
    "AnalysisResult",
    # Utilities
    "run_cmd",
    "get_git_info",
    "find_files",
    "has_tool",
    # Analyzers
    "analyze_rust",
    "analyze_python",
    "analyze_go",
    "analyze_c_cpp",
    "analyze_typescript",
    "analyze_swift",
    "analyze_objc",
    "analyze_bash",
    # Main functions
    "analyze",
    "print_summary",
    "main",
]

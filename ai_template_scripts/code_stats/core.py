# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Core analysis orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from ai_template_scripts.code_stats.models import AnalysisResult
from ai_template_scripts.code_stats.utils import get_git_info
from ai_template_scripts.path_utils import resolve_path_alias


def analyze(dir_path: Path | None = None, **kwargs: Any) -> AnalysisResult:
    """Run full analysis on codebase.

    Args:
        dir_path: Directory to analyze.
        **kwargs: Accepts deprecated 'root' alias for dir_path.

    Returns:
        AnalysisResult with all findings.
    """
    resolved_path = resolve_path_alias("dir_path", "root", dir_path, kwargs, "analyze")
    project, commit = get_git_info(resolved_path)
    result = AnalysisResult(
        project=project,
        commit=commit,
        timestamp=datetime.now(UTC).isoformat(),
    )

    # Run all analyzers
    analyze_rust(resolved_path, result)
    analyze_python(resolved_path, result)
    analyze_go(resolved_path, result)
    analyze_c_cpp(resolved_path, result)
    analyze_typescript(resolved_path, result)
    analyze_swift(resolved_path, result)
    analyze_objc(resolved_path, result)
    analyze_bash(resolved_path, result)

    # Sort warnings by severity and complexity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    result.warnings.sort(
        key=lambda w: (
            severity_order.get(w.get("severity", "low"), 2),
            -w.get("complexity", 0),
        )
    )

    return result


__all__ = ["analyze"]

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Base utilities for code analyzers."""

from __future__ import annotations

from pathlib import Path

from ai_template_scripts.code_stats.constants import (
    THRESHOLD_CYCLOMATIC,
    THRESHOLD_HIGH,
)
from ai_template_scripts.code_stats.models import AnalysisResult


def _get_relative_path(filepath: str, root: Path) -> str:
    """Get relative path from root, or filepath if not under root."""
    if filepath.startswith(str(root)):
        return str(Path(filepath).relative_to(root))
    return filepath


def _maybe_add_complexity_warning(
    result: AnalysisResult,
    rel_path: str,
    name: str,
    line: int,
    complexity: int,
) -> None:
    """Add a complexity warning if threshold exceeded."""
    if complexity > THRESHOLD_CYCLOMATIC:
        result.warnings.append(
            {
                "file": rel_path,
                "name": name,
                "line": line,
                "complexity": complexity,
                "threshold": THRESHOLD_CYCLOMATIC,
                "type": "cyclomatic",
                "severity": "high" if complexity > THRESHOLD_HIGH else "medium",
            }
        )


__all__ = ["_get_relative_path", "_maybe_add_complexity_warning"]

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Python code analysis using radon."""

from __future__ import annotations

import json
from pathlib import Path

from ai_template_scripts.code_stats.constants import (
    LANG_EXTENSIONS,
    THRESHOLD_CYCLOMATIC,
    THRESHOLD_HIGH,
)
from ai_template_scripts.code_stats.models import (
    AnalysisResult,
    FunctionMetric,
    LanguageSummary,
)
from ai_template_scripts.code_stats.utils import find_files, has_tool
from ai_template_scripts.subprocess_utils import run_cmd


def analyze_python(root: Path, result: AnalysisResult) -> None:
    """Analyze Python code using radon."""
    if not has_tool("radon"):
        result.errors.append("radon not installed (pip install radon)")
        return

    files = find_files(root, LANG_EXTENSIONS["python"])
    if not files:
        return

    result.tools["python"] = "radon"
    lang_summary = LanguageSummary()
    lang_summary.files = len(files)

    # Get cyclomatic complexity
    cmd_result = run_cmd(["radon", "cc", "-j", "-a", str(root)])
    if cmd_result.ok and cmd_result.stdout.strip():
        try:
            data = json.loads(cmd_result.stdout)
            for filepath, functions in data.items():
                if isinstance(functions, list):
                    rel_path = (
                        str(Path(filepath).relative_to(root))
                        if filepath.startswith(str(root))
                        else filepath
                    )
                    for func in functions:
                        name = func.get("name", "")
                        line = func.get("lineno", 0)
                        complexity = func.get("complexity", 0)

                        lang_summary.functions += 1
                        lang_summary.total_complexity += complexity
                        lang_summary.max_complexity = max(
                            lang_summary.max_complexity, complexity
                        )

                        fm = FunctionMetric(
                            file=rel_path,
                            name=name,
                            line=line,
                            lang="python",
                            complexity=complexity,
                            complexity_type="cyclomatic",
                        )
                        result.functions.append(fm)

                        if complexity > THRESHOLD_CYCLOMATIC:
                            result.warnings.append(
                                {
                                    "file": rel_path,
                                    "name": name,
                                    "line": line,
                                    "complexity": complexity,
                                    "threshold": THRESHOLD_CYCLOMATIC,
                                    "type": "cyclomatic",
                                    "severity": "high"
                                    if complexity > THRESHOLD_HIGH
                                    else "medium",
                                }
                            )
        except json.JSONDecodeError:
            result.errors.append("Failed to parse radon cc output")

    # Get raw metrics (SLOC)
    cmd_result = run_cmd(["radon", "raw", "-j", str(root)])
    if cmd_result.ok and cmd_result.stdout.strip():
        try:
            data = json.loads(cmd_result.stdout)
            for metrics in data.values():
                if isinstance(metrics, dict):
                    lang_summary.code_lines += metrics.get("sloc", 0)
        except json.JSONDecodeError:
            pass

    if lang_summary.files > 0:
        result.by_language["python"] = lang_summary


__all__ = ["analyze_python"]

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Go code analysis using gocyclo and gocognit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_template_scripts.code_stats.analyzers.base import _maybe_add_complexity_warning
from ai_template_scripts.code_stats.constants import LANG_EXTENSIONS
from ai_template_scripts.code_stats.models import (
    AnalysisResult,
    FunctionMetric,
    LanguageSummary,
)
from ai_template_scripts.code_stats.utils import find_files, has_tool
from ai_template_scripts.subprocess_utils import run_cmd


@dataclass
class GocycloParsedLine:
    """Parsed data from a gocyclo output line."""

    complexity: int
    filepath: str
    name: str
    lineno: int


def _parse_gocyclo_line(line: str) -> GocycloParsedLine | None:
    """Parse a single gocyclo output line, return None if invalid.

    gocyclo output format: "complexity package/path/file.go:line:col func_name"
    """
    if not line or line.startswith("Average"):
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        complexity = int(parts[0])
        location = parts[1]  # file:line:col
        name = parts[2]

        loc_parts = location.split(":")
        filepath = loc_parts[0]
        lineno = int(loc_parts[1]) if len(loc_parts) > 1 else 0

        return GocycloParsedLine(
            complexity=complexity,
            filepath=filepath,
            name=name,
            lineno=lineno,
        )
    except (ValueError, IndexError):
        return None


def _get_go_tools_string(has_gocyclo: bool, has_gocognit: bool) -> str:
    """Build the tools string for Go analysis."""
    tools = []
    if has_gocyclo:
        tools.append("gocyclo")
    if has_gocognit:
        tools.append("gocognit")
    return "+".join(tools)


def _process_gocyclo_output(
    stdout: str,
    result: AnalysisResult,
    lang_summary: LanguageSummary,
) -> None:
    """Process gocyclo output and populate results."""
    seen_functions: set[tuple[str, str, int]] = set()

    for line in stdout.strip().split("\n"):
        parsed = _parse_gocyclo_line(line)
        if not parsed:
            continue

        key = (parsed.filepath, parsed.name, parsed.lineno)
        if key in seen_functions:
            continue
        seen_functions.add(key)

        lang_summary.functions += 1
        lang_summary.total_complexity += parsed.complexity
        lang_summary.max_complexity = max(
            lang_summary.max_complexity, parsed.complexity
        )

        result.functions.append(
            FunctionMetric(
                file=parsed.filepath,
                name=parsed.name,
                line=parsed.lineno,
                lang="go",
                complexity=parsed.complexity,
                complexity_type="cyclomatic",
            )
        )

        _maybe_add_complexity_warning(
            result,
            parsed.filepath,
            parsed.name,
            parsed.lineno,
            parsed.complexity,
        )


def _count_go_sloc(files: list[Path]) -> int:
    """Count source lines of code for Go files (excluding comments)."""
    total_sloc = 0
    for filepath in files:
        try:
            content = filepath.read_text(errors="ignore")
            sloc = sum(
                1
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("//")
            )
            total_sloc += sloc
        except Exception:
            pass  # Best-effort: file read for SLOC count, skip unreadable files
    return total_sloc


def analyze_go(root: Path, result: AnalysisResult) -> None:
    """Analyze Go code using gocyclo and gocognit."""
    has_gocyclo = has_tool("gocyclo")
    has_gocognit = has_tool("gocognit")

    if not has_gocyclo and not has_gocognit:
        result.errors.append(
            "Go tools not installed "
            "(go install github.com/fzipp/gocyclo/cmd/gocyclo@latest)"
        )
        return

    files = find_files(root, LANG_EXTENSIONS["go"])
    if not files:
        return

    result.tools["go"] = _get_go_tools_string(has_gocyclo, has_gocognit)

    lang_summary = LanguageSummary()
    lang_summary.files = len(files)

    if has_gocyclo:
        cmd_result = run_cmd(["gocyclo", "-avg", "."], cwd=root)
        if cmd_result.ok:
            _process_gocyclo_output(cmd_result.stdout, result, lang_summary)

    lang_summary.code_lines = _count_go_sloc(files)

    if lang_summary.files > 0:
        result.by_language["go"] = lang_summary


__all__ = ["analyze_go", "GocycloParsedLine"]

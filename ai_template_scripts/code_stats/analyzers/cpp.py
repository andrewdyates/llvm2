# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""C/C++ code analysis using pmccabe or lizard."""

from __future__ import annotations

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


def _count_c_sloc(content: str) -> int:
    """Count source lines of code for C/C++, excluding comments.

    Handles both // line comments and /* */ block comments.

    Known limitation: Does not handle comment delimiters inside string literals.
    A string like 'char* s = "/* text */"' will be miscounted. This is acceptable
    for rough SLOC metrics since proper handling would require a full C lexer.
    """
    sloc = 0
    in_block_comment = False

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Track block comment state and check for code
        has_code = False
        i = 0
        while i < len(stripped):
            if in_block_comment:
                # Look for end of block comment
                end_idx = stripped.find("*/", i)
                if end_idx >= 0:
                    in_block_comment = False
                    i = end_idx + 2
                else:
                    break  # Rest of line is comment
            else:
                # Look for start of comments or code
                line_comment = stripped.find("//", i)
                block_start = stripped.find("/*", i)

                if line_comment >= 0 and (
                    block_start < 0 or line_comment < block_start
                ):
                    # Line comment - anything before it is code
                    if line_comment > i:
                        has_code = True
                    break  # Rest of line is comment
                if block_start >= 0:
                    # Block comment starts
                    if block_start > i:
                        has_code = True
                    in_block_comment = True
                    i = block_start + 2
                else:
                    # No comments, rest of line is code
                    has_code = True
                    break

        if has_code:
            sloc += 1

    return sloc


def _analyze_with_pmccabe(
    root: Path, files: list[Path], result: AnalysisResult
) -> None:
    """Run pmccabe on files.

    Batches pmccabe calls (up to 100 files per call) for performance.
    SLOC counting is still done per-file since it requires reading file content.
    """
    c_summary = LanguageSummary()
    cpp_summary = LanguageSummary()

    # Build mapping from filepath to (summary, lang) for output parsing
    file_info: dict[str, tuple[LanguageSummary, str]] = {}
    for filepath in files:
        is_cpp = filepath.suffix.lower() in LANG_EXTENSIONS["cpp"]
        summary = cpp_summary if is_cpp else c_summary
        lang = "cpp" if is_cpp else "c"
        summary.files += 1
        file_info[str(filepath)] = (summary, lang)

        # Count SLOC (per-file, cannot batch)
        try:
            content = filepath.read_text(errors="ignore")
            sloc = _count_c_sloc(content)
            summary.code_lines += sloc
        except Exception:
            pass  # Best-effort: file read for SLOC count, skip unreadable files

    # Batch pmccabe calls (100 files per call to avoid arg limit)
    batch_size = 100
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        cmd_result = run_cmd(["pmccabe"] + [str(f) for f in batch])
        if not cmd_result.ok:
            continue

        # pmccabe output columns (tab-separated):
        # 0: modified complexity, 1: traditional complexity, 2: statements
        # 3: start line, 4: lines in function, 5: "file(line): function_name"
        for line in cmd_result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                try:
                    complexity = int(parts[0])
                    lineno = int(parts[3])
                    # Parse "file(line): function_name" from last column
                    file_func = parts[5]
                    # Format: "/path/file.c(42): function_name"
                    # Note: function name may contain () (e.g., C++ operator())
                    # so we find '): ' first, then find '(' before it
                    colon_space_idx = file_func.find("): ")
                    if colon_space_idx < 0:
                        continue
                    paren_idx = file_func.rfind("(", 0, colon_space_idx)
                    if paren_idx < 0:
                        continue
                    file_path = file_func[:paren_idx]
                    name = file_func[colon_space_idx + 3 :]

                    # Look up summary by file path
                    if file_path not in file_info:
                        continue
                    summary, lang = file_info[file_path]
                    rel_path = str(Path(file_path).relative_to(root))

                    summary.functions += 1
                    summary.total_complexity += complexity
                    summary.max_complexity = max(summary.max_complexity, complexity)

                    fm = FunctionMetric(
                        file=rel_path,
                        name=name,
                        line=lineno,
                        lang=lang,
                        complexity=complexity,
                        complexity_type="cyclomatic",
                    )
                    result.functions.append(fm)

                    if complexity > THRESHOLD_CYCLOMATIC:
                        result.warnings.append(
                            {
                                "file": rel_path,
                                "name": name,
                                "line": lineno,
                                "complexity": complexity,
                                "threshold": THRESHOLD_CYCLOMATIC,
                                "type": "cyclomatic",
                                "severity": "high"
                                if complexity > THRESHOLD_HIGH
                                else "medium",
                            }
                        )
                except (ValueError, IndexError):
                    continue

    if c_summary.files > 0:
        result.by_language["c"] = c_summary
    if cpp_summary.files > 0:
        result.by_language["cpp"] = cpp_summary


def analyze_c_cpp(root: Path, result: AnalysisResult) -> None:
    """Analyze C/C++ code using pmccabe or lizard."""
    # Import here to avoid circular import
    from ai_template_scripts.code_stats.analyzers.lizard import _analyze_with_lizard

    has_pmccabe = has_tool("pmccabe")
    has_lizard = has_tool("lizard")

    if not has_pmccabe and not has_lizard:
        result.errors.append(
            "C/C++ tools not installed (apt install pmccabe or pip install lizard)"
        )
        return

    c_files = find_files(root, LANG_EXTENSIONS["c"])
    cpp_files = find_files(root, LANG_EXTENSIONS["cpp"])
    all_files = c_files + cpp_files

    if not all_files:
        return

    # Prefer pmccabe for C/C++
    if has_pmccabe:
        result.tools["c"] = "pmccabe"
        result.tools["cpp"] = "pmccabe"
        _analyze_with_pmccabe(root, all_files, result)
    else:
        result.tools["c"] = "lizard"
        result.tools["cpp"] = "lizard"
        _analyze_with_lizard(root, all_files, ["c", "cpp"], result)


__all__ = ["analyze_c_cpp"]

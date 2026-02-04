# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Lizard-based code analysis for multiple languages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_template_scripts.code_stats.analyzers.base import (
    _get_relative_path,
    _maybe_add_complexity_warning,
)
from ai_template_scripts.code_stats.constants import LANG_EXTENSIONS
from ai_template_scripts.code_stats.models import (
    AnalysisResult,
    FunctionMetric,
    LanguageSummary,
)
from ai_template_scripts.code_stats.utils import find_files, has_tool
from ai_template_scripts.subprocess_utils import run_cmd

# Map our language names to lizard's expected names
LIZARD_LANG_MAP = {
    "typescript": "javascript",  # lizard uses 'javascript' for both
    "javascript": "javascript",
    "cpp": "cpp",
    "c": "c",
    "objc": "objectivec",
    "swift": "swift",
    "rust": "rust",
}


@dataclass
class LizardParsedLine:
    """Parsed data from a lizard CSV line."""

    sloc: int
    complexity: int
    filepath: str
    name: str
    start_line: int


def _build_lizard_lang_args(langs: list[str]) -> list[str]:
    """Build lizard -l arguments for the given languages."""
    lang_args = []
    for lang in langs:
        if lang in LIZARD_LANG_MAP:
            lang_args.extend(["-l", LIZARD_LANG_MAP[lang]])
    return lang_args


def _parse_lizard_csv_line(line: str) -> LizardParsedLine | None:
    """Parse a single lizard CSV line, return None if invalid."""
    if not line or line.startswith("NLOC"):
        return None
    parts = line.split(",")
    if len(parts) < 10:
        return None
    try:
        return LizardParsedLine(
            sloc=int(parts[0]),
            complexity=int(parts[1]),
            filepath=parts[6],
            name=parts[7],
            start_line=int(parts[8]),
        )
    except (ValueError, IndexError):
        return None


def _determine_lang_from_extension(filepath: str, langs: list[str]) -> str | None:
    """Determine language from file extension, constrained to given langs."""
    ext = Path(filepath).suffix.lower()
    for lang_key, exts in LANG_EXTENSIONS.items():
        if ext in exts and lang_key in langs:
            return lang_key
    return None


def _analyze_with_lizard(
    root: Path,
    files: list[Path],
    langs: list[str],
    result: AnalysisResult,
) -> None:
    """Use lizard for languages without better tools."""
    if not files:
        return

    lang_args = _build_lizard_lang_args(langs)
    file_args = [str(f) for f in files]
    cmd_result = run_cmd(["lizard", "--csv"] + lang_args + file_args)

    if not cmd_result.ok:
        return

    summaries: dict[str, LanguageSummary] = {lang: LanguageSummary() for lang in langs}
    seen_files: dict[str, set[str]] = {lang: set() for lang in langs}

    for line in cmd_result.stdout.strip().split("\n"):
        parsed = _parse_lizard_csv_line(line)
        if not parsed:
            continue

        lang = _determine_lang_from_extension(parsed.filepath, langs)
        if not lang:
            continue

        summary = summaries[lang]
        if parsed.filepath not in seen_files[lang]:
            seen_files[lang].add(parsed.filepath)
            summary.files += 1

        summary.functions += 1
        summary.total_complexity += parsed.complexity
        summary.max_complexity = max(summary.max_complexity, parsed.complexity)
        summary.code_lines += parsed.sloc

        rel_path = _get_relative_path(parsed.filepath, root)
        result.functions.append(
            FunctionMetric(
                file=rel_path,
                name=parsed.name,
                line=parsed.start_line,
                lang=lang,
                complexity=parsed.complexity,
                complexity_type="cyclomatic",
                sloc=parsed.sloc,
            )
        )
        _maybe_add_complexity_warning(
            result, rel_path, parsed.name, parsed.start_line, parsed.complexity
        )

    for lang, summary in summaries.items():
        if summary.files > 0:
            result.by_language[lang] = summary


def analyze_rust(root: Path, result: AnalysisResult) -> None:
    """Analyze Rust code using lizard."""
    if not has_tool("lizard"):
        result.errors.append("lizard not installed (pip install lizard)")
        return

    files = find_files(root, LANG_EXTENSIONS["rust"])
    if not files:
        return

    result.tools["rust"] = "lizard"
    _analyze_with_lizard(root, files, ["rust"], result)


def analyze_typescript(root: Path, result: AnalysisResult) -> None:
    """Analyze TypeScript/JavaScript using lizard."""
    if not has_tool("lizard"):
        result.errors.append("lizard not installed (pip install lizard)")
        return

    ts_files = find_files(root, LANG_EXTENSIONS["typescript"])
    js_files = find_files(root, LANG_EXTENSIONS["javascript"])
    all_files = ts_files + js_files

    if not all_files:
        return

    result.tools["typescript"] = "lizard"
    result.tools["javascript"] = "lizard"
    _analyze_with_lizard(root, all_files, ["typescript", "javascript"], result)


def analyze_swift(root: Path, result: AnalysisResult) -> None:
    """Analyze Swift using lizard."""
    if not has_tool("lizard"):
        result.errors.append("lizard not installed for Swift (pip install lizard)")
        return

    files = find_files(root, LANG_EXTENSIONS["swift"])
    if not files:
        return

    result.tools["swift"] = "lizard"
    _analyze_with_lizard(root, files, ["swift"], result)


def analyze_objc(root: Path, result: AnalysisResult) -> None:
    """Analyze Objective-C using lizard."""
    if not has_tool("lizard"):
        result.errors.append(
            "lizard not installed for Objective-C (pip install lizard)"
        )
        return

    files = find_files(root, LANG_EXTENSIONS["objc"])
    if not files:
        return

    result.tools["objc"] = "lizard"
    _analyze_with_lizard(root, files, ["objc"], result)


def analyze_bash(root: Path, result: AnalysisResult) -> None:
    """Analyze Bash (LOC only, no complexity tool exists)."""
    files = find_files(root, LANG_EXTENSIONS["bash"])
    if not files:
        return

    result.tools["bash"] = "line-count"  # No real complexity tool
    summary = LanguageSummary()
    summary.files = len(files)

    for filepath in files:
        try:
            content = filepath.read_text(errors="ignore")
            sloc = sum(
                1
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            )
            summary.code_lines += sloc
        except Exception:
            pass  # Best-effort: bash file read for SLOC count, skip unreadable files

    if summary.files > 0:
        result.by_language["bash"] = summary


__all__ = [
    "LIZARD_LANG_MAP",
    "LizardParsedLine",
    "_analyze_with_lizard",
    "analyze_rust",
    "analyze_typescript",
    "analyze_swift",
    "analyze_objc",
    "analyze_bash",
]

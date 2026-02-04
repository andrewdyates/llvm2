# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Data models for code complexity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionMetric:
    """Metrics for a single function."""

    file: str
    name: str
    line: int
    lang: str
    complexity: int
    complexity_type: str  # cyclomatic, cognitive
    sloc: int = 0


@dataclass
class LanguageSummary:
    """Summary stats for a language."""

    files: int = 0
    code_lines: int = 0
    functions: int = 0
    total_complexity: int = 0
    max_complexity: int = 0

    @property
    def avg_complexity(self) -> float:
        return self.total_complexity / self.functions if self.functions else 0.0


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    project: str
    commit: str
    timestamp: str
    tools: dict[str, str] = field(default_factory=dict)
    by_language: dict[str, LanguageSummary] = field(default_factory=dict)
    functions: list[FunctionMetric] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        summary = {
            "total_files": sum(ls.files for ls in self.by_language.values()),
            "total_code_lines": sum(ls.code_lines for ls in self.by_language.values()),
            "total_functions": sum(ls.functions for ls in self.by_language.values()),
            "by_language": {
                lang: {
                    "files": ls.files,
                    "code_lines": ls.code_lines,
                    "functions": ls.functions,
                    "avg_complexity": round(ls.avg_complexity, 2),
                    "max_complexity": ls.max_complexity,
                }
                for lang, ls in self.by_language.items()
            },
        }
        return {
            "version": "1.0",
            "timestamp": self.timestamp,
            "project": self.project,
            "commit": self.commit,
            "tools": self.tools,
            "summary": summary,
            "functions": [
                {
                    "file": f.file,
                    "name": f.name,
                    "line": f.line,
                    "lang": f.lang,
                    "complexity": f.complexity,
                    "type": f.complexity_type,
                    "sloc": f.sloc,
                }
                for f in self.functions
            ],
            "warnings": self.warnings,
            "errors": self.errors,
        }


__all__ = ["FunctionMetric", "LanguageSummary", "AnalysisResult"]

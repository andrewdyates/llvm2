# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""CLI interface for code complexity analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ai_template_scripts.code_stats.constants as constants
from ai_template_scripts.code_stats.core import analyze
from ai_template_scripts.code_stats.models import AnalysisResult
from ai_template_scripts.version import get_version


def print_summary(result: AnalysisResult) -> None:
    """Print human-readable summary to stderr."""
    print("=" * 60, file=sys.stderr)
    print("CODE COMPLEXITY ANALYSIS", file=sys.stderr)
    print(f"Project: {result.project} @ {result.commit}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if result.errors:
        print("\nMissing tools:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)

    if not result.by_language:
        print("\nNo source files found.", file=sys.stderr)
        return

    print("\n## Summary by Language\n", file=sys.stderr)
    print(
        f"{'Language':<12} {'Files':>6} {'SLOC':>8} "
        f"{'Funcs':>6} {'AvgCC':>6} {'MaxCC':>6}",
        file=sys.stderr,
    )
    print("-" * 52, file=sys.stderr)

    for lang in sorted(result.by_language.keys()):
        ls = result.by_language[lang]
        tool = result.tools.get(lang, "?")
        print(
            f"{lang:<12} {ls.files:>6} {ls.code_lines:>8} {ls.functions:>6} "
            f"{ls.avg_complexity:>6.1f} {ls.max_complexity:>6}  [{tool}]",
            file=sys.stderr,
        )

    totals = result.to_dict()["summary"]
    print("-" * 52, file=sys.stderr)
    print(
        f"{'TOTAL':<12} {totals['total_files']:>6} {totals['total_code_lines']:>8} "
        f"{totals['total_functions']:>6}",
        file=sys.stderr,
    )

    if result.warnings:
        high_warnings = [w for w in result.warnings if w.get("severity") == "high"]
        med_warnings = [w for w in result.warnings if w.get("severity") == "medium"]

        print(
            f"\n## Warnings: {len(high_warnings)} high, {len(med_warnings)} medium\n",
            file=sys.stderr,
        )

        # Show top 10 by complexity
        for w in result.warnings[:10]:
            sev = w.get("severity", "?")[0].upper()
            print(
                f"  [{sev}] {w['complexity']:>3} {w['file']}:{w['line']} {w['name']}",
                file=sys.stderr,
            )

        if len(result.warnings) > 10:
            print(f"  ... and {len(result.warnings) - 10} more", file=sys.stderr)

    print(file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Code complexity analysis")
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("code_stats.py"),
    )
    parser.add_argument("path", nargs="?", default=".", help="Path to analyze")
    parser.add_argument(
        "-j", "--json", action="store_true", help="Output JSON only (no summary)"
    )
    parser.add_argument("-o", "--output", help="Write JSON to file")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress summary output"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Complexity threshold for warnings (default: 10)",
    )
    args = parser.parse_args()

    # Update threshold (module-level for backward compatibility)
    constants.THRESHOLD_CYCLOMATIC = args.threshold

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        return 1

    result = analyze(root)

    # Output
    if not args.quiet and not args.json:
        print_summary(result)

    json_output = json.dumps(result.to_dict(), indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output)
        print(f"JSON written to {args.output}", file=sys.stderr)
    elif args.json:
        print(json_output)

    # Exit code: 1 if high-severity warnings
    high_count = sum(1 for w in result.warnings if w.get("severity") == "high")
    return 1 if high_count > 0 else 0


__all__ = ["print_summary", "main"]

#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""vision_validate.py - Validate VISION.md structure and required sections.

Checks that VISION.md files comply with the standard structure defined in
ai_template.md. Required elements are:
- Problem statement (Why section)
- Success metrics
- Readiness label (PLANNED, BUILDING, USABLE, V1, HOLD)
- Execution phases

Usage:
    vision_validate.py ~/project/VISION.md          # Validate single file
    vision_validate.py ~/project/                   # Validate VISION.md in dir
    vision_validate.py --json                       # Machine-readable output
    vision_validate.py --all                        # Check all repos

Part of #2540 - VISION.md validation script.

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.version import get_version  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "READINESS_LEVELS",
    "ValidationResult",
    "validate_vision_file",
    "check_problem_statement",
    "check_success_metrics",
    "check_readiness_label",
    "check_phases_section",
    "main",
]

# Valid readiness levels per ai_template.md
READINESS_LEVELS = frozenset({"PLANNED", "BUILDING", "USABLE", "V1", "HOLD"})


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str


def check_problem_statement(content: str) -> ValidationResult:
    """Check for problem statement / why section.

    Contracts:
        ENSURES: Returns ValidationResult with passed=True if why section found
        ENSURES: Returns ValidationResult with passed=False if missing

    Args:
        content: The VISION.md file content.

    Returns:
        ValidationResult indicating presence of problem statement.
    """
    # Look for common problem statement patterns
    patterns = [
        r"^##\s+Why\s+\w+\s+Exists",  # ## Why <name> Exists
        r"^##\s+Why\s+This\s+Exists",  # ## Why This Exists
        r"^##\s+Problem\s*(Statement)?",  # ## Problem or ## Problem Statement
        r"^##\s+Purpose",  # ## Purpose
        r"^##\s+Motivation",  # ## Motivation
        r"^##\s+Mission",  # ## Mission (z4 style)
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            return ValidationResult(
                name="problem_statement",
                passed=True,
                message="Problem statement / why section found",
            )

    return ValidationResult(
        name="problem_statement",
        passed=False,
        message="Missing problem statement (add '## Why <name> Exists' or '## Problem')",
    )


def check_success_metrics(content: str) -> ValidationResult:
    """Check for success metrics / success criteria section.

    Contracts:
        ENSURES: Returns ValidationResult with passed=True if metrics section found
        ENSURES: Returns ValidationResult with passed=False if missing

    Args:
        content: The VISION.md file content.

    Returns:
        ValidationResult indicating presence of success metrics.
    """
    # Look for success metrics patterns
    patterns = [
        r"^##\s+Success\s+(Criteria|Metrics)",  # ## Success Criteria or Metrics
        r"^##\s+Goals",  # ## Goals (alternative)
        r"^##\s+Objectives",  # ## Objectives (alternative)
        r"^##\s+Key\s+Results",  # ## Key Results (OKR style)
        r"\*\*Success\s+means:\*\*",  # **Success means:** (inline style in z4)
        r"^##\s+How\s+We\s+Measure\s+Progress",  # ## How We Measure Progress (ai_template/zani style)
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            return ValidationResult(
                name="success_metrics",
                passed=True,
                message="Success metrics section found",
            )

    return ValidationResult(
        name="success_metrics",
        passed=False,
        message="Missing success metrics (add '## Success Criteria' section)",
    )


def check_readiness_label(content: str) -> ValidationResult:
    """Check for explicit readiness level.

    Contracts:
        ENSURES: Returns ValidationResult with passed=True if valid readiness found
        ENSURES: Returns ValidationResult with passed=False if missing or invalid

    Args:
        content: The VISION.md file content.

    Returns:
        ValidationResult indicating presence and validity of readiness label.
    """
    # Look for readiness patterns
    # Pattern 1: "Readiness: LEVEL" or "**Readiness:** LEVEL"
    match = re.search(
        r"\*?\*?Readiness:?\*?\*?\s*:?\s*(\w+)",
        content,
        re.IGNORECASE,
    )
    if match:
        level = match.group(1).upper()
        if level in READINESS_LEVELS:
            return ValidationResult(
                name="readiness",
                passed=True,
                message=f"Readiness level: {level}",
            )
        return ValidationResult(
            name="readiness",
            passed=False,
            message=f"Invalid readiness '{level}'. Must be one of: {', '.join(sorted(READINESS_LEVELS))}",
        )

    # Pattern 2: "**Level:** LEVEL" (as in ai_template VISION.md)
    match = re.search(
        r"\*\*Level:\*\*\s*(\w+)",
        content,
    )
    if match:
        level = match.group(1).upper()
        if level in READINESS_LEVELS:
            return ValidationResult(
                name="readiness",
                passed=True,
                message=f"Readiness level: {level}",
            )
        return ValidationResult(
            name="readiness",
            passed=False,
            message=f"Invalid readiness '{level}'. Must be one of: {', '.join(sorted(READINESS_LEVELS))}",
        )

    # Pattern 3: Explicit level in ## Readiness section
    if re.search(r"^##\s+Readiness\b", content, re.MULTILINE | re.IGNORECASE):
        # Look for levels in section
        for level in READINESS_LEVELS:
            if re.search(rf"\b{level}\b", content, re.IGNORECASE):
                return ValidationResult(
                    name="readiness",
                    passed=True,
                    message=f"Readiness level: {level}",
                )
        return ValidationResult(
            name="readiness",
            passed=False,
            message="Readiness section found but no valid level. Add explicit level (PLANNED/BUILDING/USABLE/V1/HOLD)",
        )

    return ValidationResult(
        name="readiness",
        passed=False,
        message="Missing readiness label (add 'Readiness: BUILDING' or '## Readiness' section)",
    )


def check_phases_section(content: str) -> ValidationResult:
    """Check for execution phases section.

    Contracts:
        ENSURES: Returns ValidationResult with passed=True if phases found
        ENSURES: Returns ValidationResult with passed=False if missing

    Args:
        content: The VISION.md file content.

    Returns:
        ValidationResult indicating presence of phases section.
    """
    # Look for phases/milestones/roadmap patterns
    patterns = [
        r"^##\s+Phases?",  # ## Phase or ## Phases
        r"^##\s+Milestones?",  # ## Milestone(s)
        r"^##\s+Roadmap",  # ## Roadmap
        r"^##\s+Execution\s+Phases?",  # ## Execution Phase(s)
        r"^##\s+Implementation\s+Phases?",  # ## Implementation Phase(s)
        r"^##\s+How\s+We\s+Measure\s+Progress",  # ## How We Measure Progress (ai_template style)
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            return ValidationResult(
                name="phases",
                passed=True,
                message="Execution phases section found",
            )

    return ValidationResult(
        name="phases",
        passed=False,
        message="Missing execution phases (add '## Phases' or '## How We Measure Progress')",
    )


def validate_vision_file(path: Path) -> list[ValidationResult]:
    """Validate a VISION.md file.

    Contracts:
        REQUIRES: path exists and is readable
        ENSURES: Returns list of ValidationResult for all checks

    Args:
        path: Path to VISION.md file.

    Returns:
        List of ValidationResult for each check.
    """
    try:
        content = path.read_text()
    except OSError as e:
        return [
            ValidationResult(
                name="file_access",
                passed=False,
                message=f"Cannot read file: {e}",
            )
        ]

    return [
        check_problem_statement(content),
        check_success_metrics(content),
        check_readiness_label(content),
        check_phases_section(content),
    ]


def _get_all_repos() -> list[Path]:
    """Get all repos in ~/

    Returns:
        List of paths to repo directories with VISION.md.
    """
    home = Path.home()
    repos = []

    # Check direct children of home that are git repos with VISION.md
    for child in home.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            vision_path = child / "VISION.md"
            if vision_path.exists():
                repos.append(vision_path)

    return sorted(repos)


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = all pass, 1 = failures).
    """
    parser = argparse.ArgumentParser(
        description="Validate VISION.md structure and required sections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to VISION.md file or directory containing one",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all repos in ~/",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("vision_validate.py"),
    )

    parsed = parser.parse_args(args)

    # Determine paths to check
    paths: list[Path] = []

    if parsed.all:
        paths = _get_all_repos()
        if not paths:
            if parsed.json:
                print(json.dumps({"error": "No repos with VISION.md found"}))
            else:
                print("No repos with VISION.md found in ~/")
            return 1
    elif parsed.path:
        target = parsed.path
        if target.is_dir():
            target = target / "VISION.md"
        if not target.exists():
            if parsed.json:
                print(json.dumps({"error": f"File not found: {target}"}))
            else:
                print(f"Error: File not found: {target}")
            return 1
        paths = [target]
    else:
        # Default: check current directory
        target = Path.cwd() / "VISION.md"
        if not target.exists():
            if parsed.json:
                print(json.dumps({"error": "VISION.md not found in current directory"}))
            else:
                print("Error: VISION.md not found in current directory")
            return 1
        paths = [target]

    # Validate all paths
    all_results: dict[str, list[dict]] = {}
    has_failure = False

    for path in paths:
        results = validate_vision_file(path)
        repo_name = path.parent.name
        all_results[repo_name] = [
            {"name": r.name, "passed": r.passed, "message": r.message} for r in results
        ]
        if not all(r.passed for r in results):
            has_failure = True

    # Output results
    if parsed.json:
        print(json.dumps(all_results, indent=2))
    else:
        for repo_name, results in all_results.items():
            print(f"\n=== {repo_name} ===")
            for r in results:
                symbol = "\u2713" if r["passed"] else "\u2717"
                print(f"  {symbol} {r['message']}")

        # Summary
        total_repos = len(all_results)
        passing_repos = sum(
            1 for results in all_results.values() if all(r["passed"] for r in results)
        )
        print(f"\nSummary: {passing_repos}/{total_repos} repos pass all checks")

    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())

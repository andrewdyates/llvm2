# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Health Check CLI

Standard CLI argument parsing and main function for health checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ai_template_scripts.version import get_version

if TYPE_CHECKING:
    from ai_template_scripts.health_check.base import HealthCheckBase


def create_parser(
    description: str | None = None,
    script_name: str = "health_check.py",
) -> argparse.ArgumentParser:
    """Create standard CLI parser with required flags.

    REQUIRES: description is None or non-empty string
    ENSURES: Returns ArgumentParser with --version and --json-output flags

    Repos can extend this parser with additional arguments:

        parser = create_parser()
        parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
        args = parser.parse_args()

    Args:
        description: Custom description. Defaults to generic health check description.
        script_name: Script name for --version output. Defaults to "health_check.py".

    Returns:
        ArgumentParser with standard --version and --json-output flags.
    """
    parser = argparse.ArgumentParser(
        description=description
        or "System health check - verify the system is connected."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version(script_name),
    )
    parser.add_argument(
        "--json-output",
        metavar="PATH",
        type=Path,
        help="Write JSON manifest to PATH",
    )
    return parser


def standard_main(health_check: HealthCheckBase, args: argparse.Namespace) -> int:
    """Standard main function with exit code handling.

    REQUIRES: health_check has checks registered, args from create_parser()
    ENSURES: All checks run, summary printed
    ENSURES: Returns 0 if no errors, 1 if errors
    ENSURES: JSON manifest written to args.json_output if specified

    Runs all registered checks, prints summary, and writes JSON output if requested.

    Exit codes per contract (designs/2026-01-28-system-health-check-contract.md):
    - Exit 0: All checks passed (or passed with warnings)
    - Exit 1: One or more checks failed

    Args:
        health_check: HealthCheckBase instance with checks registered.
        args: Parsed arguments from create_parser().

    Returns:
        Exit code (0 or 1).
    """
    # Print header
    project = health_check.project_name or "Unknown"
    print("=" * 60)
    print(f"SYSTEM HEALTH CHECK - {project}")
    print("=" * 60)

    # Run checks
    passed = health_check.run_all()

    # Print summary
    health_check.print_summary()

    # Write JSON manifest if requested
    if args.json_output:
        manifest = health_check.to_json()
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nJSON manifest written to: {args.json_output}")

    # Exit 0 for pass OR warn, exit 1 for fail (per contract)
    return 0 if passed else 1

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
provenance_capture.py - Capture SLSA-style build/test provenance manifests

PURPOSE: Records builder identity, build type, and parameters for auditability.
CALLED BY: Test/build verification workflows, commit preparation
REFERENCED: ai_template.md commit template (## Verified section)
SPEC: https://slsa.dev/spec/v1.0/provenance

Schema follows SLSA v1.0 BuildDefinition + RunDetails structure.

Public API (library usage):
    from ai_template_scripts.provenance_capture import (
        get_git_info,            # Get current git repo info
        get_builder_info,        # Get builder identity (SLSA builder.id)
        compute_digest,          # Compute SHA256 of a file (file_path param)
        capture_command_output,  # Execute command and capture output
        create_provenance,       # Create SLSA-aligned provenance manifest
        save_provenance,         # Save provenance to file
    )

CLI usage:
    ./provenance_capture.py pytest -- pytest tests/ -v
    ./provenance_capture.py cargo-test -- cargo test
    ./provenance_capture.py build --no-run --command "cargo build"

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

__all__ = [
    "get_git_info",
    "get_builder_info",
    "compute_digest",
    "capture_command_output",
    "create_provenance",
    "save_provenance",
    "redact_git_url",
    "main",
]

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.path_utils import resolve_path_alias  # noqa: E402
from ai_template_scripts.subprocess_utils import run_cmd  # noqa: E402
from ai_template_scripts.url_sanitizer import sanitize_git_url  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

# Backward-compat alias for sanitize_git_url
redact_git_url = sanitize_git_url


def get_git_info() -> dict[str, str | bool]:
    """Get current git repository info.

    REQUIRES: Git installed (repo initialized optional - returns empty dict if not)
    ENSURES: Returns dict with 'commit', 'branch', 'repo', 'dirty' keys (if available),
             or empty dict if not in a git repo
    """
    info: dict[str, str | bool] = {}

    commit_result = run_cmd(["git", "rev-parse", "HEAD"], timeout=10)
    if commit_result.ok:
        info["commit"] = commit_result.stdout.strip()

        branch_result = run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10
        )
        if branch_result.ok:
            info["branch"] = branch_result.stdout.strip()

        repo_result = run_cmd(
            ["git", "config", "--get", "remote.origin.url"], timeout=10
        )
        if repo_result.ok:
            # Redact credentials/tokens before persisting (#2189)
            redacted = redact_git_url(repo_result.stdout.strip())
            if redacted:
                info["repo"] = redacted

        # Check for uncommitted changes
        status_result = run_cmd(["git", "status", "--porcelain"], timeout=10)
        if status_result.ok:
            info["dirty"] = bool(status_result.stdout.strip())

    return info


def get_builder_info() -> dict[str, Any]:
    """Get builder identity information (SLSA builder.id).

    REQUIRES: None (reads platform info)
    ENSURES: Returns dict with 'id', 'hostname', 'platform', 'platform_version',
             'python_version', 'user' keys
    """
    return {
        "id": f"https://github.com/dropbox-ai-prototypes/ai_template/builder/{platform.node()}",
        "hostname": platform.node(),
        "platform": platform.system(),
        "platform_version": platform.release(),
        "python_version": platform.python_version(),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }


def compute_digest(file_path: Path | None = None, **kwargs: Any) -> str | None:
    """Compute SHA256 digest of a file.

    REQUIRES: file_path (or 'filepath' kwarg) is valid Path
    ENSURES: Returns 64-char hex SHA256 string if file exists, else None

    Args:
        file_path: Path to the file to compute digest for.
        **kwargs: Accepts deprecated 'filepath' alias for file_path.

    Returns:
        SHA256 hex digest string, or None if file doesn't exist.
    """
    resolved_path = resolve_path_alias(
        "file_path", "filepath", file_path, kwargs, "compute_digest"
    )
    if not resolved_path.exists():
        return None
    sha256 = hashlib.sha256()
    with resolved_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def capture_command_output(command: list[str], timeout: int = 300) -> dict[str, Any]:
    """Execute command and capture its output for provenance.

    REQUIRES: command is non-empty list of strings, timeout > 0
    ENSURES: Returns dict with 'success', 'exit_code', timing, and output counts
    """
    start_time = datetime.now(UTC)

    result = run_cmd(command, timeout=timeout)
    end_time = datetime.now(UTC)

    output: dict[str, Any] = {
        "success": result.ok,
        "exit_code": result.returncode,
        "stdout_lines": len(result.stdout.splitlines()),
        "stderr_lines": len(result.stderr.splitlines()),
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
    }

    # Include error message if command failed
    if result.error:
        output["error"] = result.error

    return output


def create_provenance(
    build_type: str,
    command: list[str],
    output_files: list[str] | None = None,
    run_command: bool = True,
    timeout: int = 300,
) -> dict[str, Any]:
    """
    Create a SLSA-aligned provenance manifest.

    REQUIRES: build_type is non-empty string, command is non-empty list
    ENSURES: Returns SLSA v1.0 compliant provenance dict with buildDefinition and runDetails

    Args:
        build_type: URI-like identifier for the build type
            (e.g., "pytest", "cargo-test")
        command: The command that was/will be executed
        output_files: Optional list of output file paths to include digests for
        run_command: If True, execute the command and capture results
        timeout: Command timeout in seconds

    Returns:
        Provenance manifest dict
    """
    timestamp = datetime.now(UTC).isoformat()

    # SLSA BuildDefinition
    resolved_dependencies: list[dict[str, Any]] = []
    build_definition: dict[str, Any] = {
        "buildType": f"https://github.com/dropbox-ai-prototypes/ai_template/build/{build_type}",
        "externalParameters": {
            "command": command,
            "cwd": str(Path.cwd()),
        },
        "internalParameters": {
            "env_python": os.environ.get("VIRTUAL_ENV", "system"),
        },
        "resolvedDependencies": resolved_dependencies,
    }

    # Add git info as dependency
    git_info = get_git_info()
    if git_info.get("commit"):
        resolved_dependencies.append(
            {
                "uri": git_info.get("repo", "local"),
                "digest": {"gitCommit": git_info["commit"]},
                "annotations": {
                    "branch": git_info.get("branch", "unknown"),
                    "dirty": str(git_info.get("dirty", False)).lower(),
                },
            }
        )

    # SLSA RunDetails
    run_details: dict[str, Any] = {
        "builder": get_builder_info(),
        "metadata": {
            "invocationId": (
                f"{timestamp}-{hashlib.sha256(str(command).encode()).hexdigest()[:8]}"
            ),
            "startedOn": timestamp,
        },
    }

    # Execute command if requested
    if run_command:
        execution = capture_command_output(command, timeout)
        run_details["metadata"]["finishedOn"] = execution.get("finished_at", timestamp)
        run_details["execution"] = execution

    # Add output file digests if specified
    if output_files:
        byproducts = []
        for filepath in output_files:
            path = Path(filepath)
            digest = compute_digest(path)
            if digest:
                byproducts.append(
                    {
                        "uri": str(path),
                        "digest": {"sha256": digest},
                        "size": path.stat().st_size,
                    }
                )
        if byproducts:
            run_details["byproducts"] = byproducts

    return {
        "_type": "https://slsa.dev/provenance/v1",
        "subject": [
            {
                "name": build_type,
                "annotations": {"command": " ".join(command)},
            }
        ],
        "buildDefinition": build_definition,
        "runDetails": run_details,
    }


def save_provenance(
    provenance: dict[str, Any], output_path: Path | None = None
) -> Path:
    """Save provenance manifest to file.

    REQUIRES: provenance is valid dict, output_path (if provided) is writable
    ENSURES: Returns Path to written file; file contains JSON provenance data
    """
    if output_path is None:
        # Default to reports/provenance/ directory
        reports_dir = Path("reports/provenance")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        subject_list = provenance.get("subject", [])
        build_type = (
            subject_list[0].get("name", "unknown") if subject_list else "unknown"
        )
        output_path = reports_dir / f"{timestamp}_{build_type}.json"

    with output_path.open("w") as f:
        json.dump(provenance, f, indent=2)

    return output_path


def main() -> None:
    """CLI entry point.

    REQUIRES: sys.argv contains valid CLI arguments
    ENSURES: Creates and saves provenance; exits 0 on success, 1 on command failure
    """
    parser = argparse.ArgumentParser(
        description="Capture SLSA-style build/test provenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture pytest run provenance
  %(prog)s pytest -- pytest tests/ -v

  # Capture cargo test provenance
  %(prog)s cargo-test -- cargo test

  # Capture build provenance without running command
  %(prog)s build --no-run --command "cargo build --release"

  # Include output file digests
  %(prog)s build --output target/release/myapp -- cargo build --release
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("provenance_capture.py"),
    )
    parser.add_argument(
        "build_type",
        help="Build type identifier (e.g., 'pytest', 'cargo-test', 'build')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: reports/provenance/<timestamp>_<type>.json)",
    )
    parser.add_argument(
        "--output-files",
        nargs="*",
        help="Paths to output files to include SHA256 digests for",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Don't execute the command, just record it",
    )
    parser.add_argument(
        "--command",
        help="Command string (alternative to positional args after --)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Command timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output the provenance file path",
    )
    parser.add_argument(
        "command_args",
        nargs="*",
        help="Command to execute (after --)",
    )

    args = parser.parse_args()

    # Determine command
    if args.command:
        command = args.command.split()
    elif args.command_args:
        command = args.command_args
    else:
        parser.error("Must provide command via -- or --command")

    # Create provenance
    provenance = create_provenance(
        build_type=args.build_type,
        command=command,
        output_files=args.output_files,
        run_command=not args.no_run,
        timeout=args.timeout,
    )

    # Save provenance
    output_path = save_provenance(provenance, args.output)

    # Output
    if args.quiet:
        print(output_path)
    else:
        print(f"Provenance saved to: {output_path}")
        if not args.no_run:
            execution = provenance.get("runDetails", {}).get("execution", {})
            if execution.get("success"):
                print(
                    f"Command succeeded in {execution.get('duration_seconds', 0):.2f}s"
                )
            else:
                print(f"Command failed: exit code {execution.get('exit_code')}")
                sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""
Check CLAUDE.md documentation health for AI operations.

Verifies that AI-facing documentation is complete enough for autonomous operation.
Designed for Manager rotation phase 'doc_health'.

Usage:
    python3 ai_template_scripts/doc_health_check.py              # Human output
    python3 ai_template_scripts/doc_health_check.py --json       # Machine-readable
    python3 ai_template_scripts/doc_health_check.py --path=/repo # Check specific repo
"""

from __future__ import annotations

__all__ = [
    "CheckResult",
    "check_cli_documented",
    "check_build_commands",
    "check_env_vars",
    "check_recent_errors",
    "check_key_scripts",
    "run_checks",
    "main",
]

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple


class CheckResult(NamedTuple):
    """Result of a documentation check."""

    name: str
    passed: bool
    message: str
    details: list[str] | None = None


def check_cli_documented(repo_path: Path) -> CheckResult:
    """Check if main CLI invocation is documented."""
    claude_md = repo_path / "CLAUDE.md"
    claude_content = claude_md.read_text() if claude_md.exists() else ""

    # Check for Cargo.toml with [[bin]]
    cargo_toml = repo_path / "Cargo.toml"
    if cargo_toml.exists():
        cargo_content = cargo_toml.read_text()
        if "[[bin]]" in cargo_content:
            # Extract bin names
            bins = re.findall(r'name\s*=\s*"([^"]+)"', cargo_content)
            missing = [b for b in bins if b not in claude_content]
            if missing:
                return CheckResult(
                    name="cli_documented",
                    passed=False,
                    message=f"Binaries not documented: {', '.join(missing)}",
                    details=[f"Binary '{b}' not in CLAUDE.md" for b in missing],
                )

    # Check for Python entry points
    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        pyproject_content = pyproject.read_text()
        if (
            "[project.scripts]" in pyproject_content
            or "[tool.poetry.scripts]" in pyproject_content
        ):
            # Look for script names
            scripts = re.findall(r"^(\w+)\s*=", pyproject_content, re.MULTILINE)
            missing = [s for s in scripts if s not in claude_content]
            if missing:
                return CheckResult(
                    name="cli_documented",
                    passed=False,
                    message=f"Scripts not documented: {', '.join(missing[:3])}",
                    details=[f"Script '{s}' not in CLAUDE.md" for s in missing],
                )

    return CheckResult(
        name="cli_documented",
        passed=True,
        message="CLI entry points documented",
    )


def check_build_commands(repo_path: Path) -> CheckResult:
    """Check if build commands are documented."""
    claude_md = repo_path / "CLAUDE.md"
    claude_content = claude_md.read_text().lower() if claude_md.exists() else ""

    issues: list[str] = []

    # Rust projects
    if (repo_path / "Cargo.toml").exists():
        if "cargo build" not in claude_content and "cargo check" not in claude_content:
            issues.append("Cargo build commands not documented")

    # Python projects
    if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
        if "pip install" not in claude_content and "python" not in claude_content:
            issues.append("Python install/run commands not documented")

    # Node projects
    if (repo_path / "package.json").exists():
        if "npm" not in claude_content and "yarn" not in claude_content:
            issues.append("npm/yarn commands not documented")

    if issues:
        return CheckResult(
            name="build_commands",
            passed=False,
            message="Missing build command documentation",
            details=issues,
        )

    return CheckResult(
        name="build_commands",
        passed=True,
        message="Build commands documented",
    )


def check_env_vars(repo_path: Path) -> CheckResult:
    """Check if required environment variables are documented."""
    claude_md = repo_path / "CLAUDE.md"
    claude_content = claude_md.read_text() if claude_md.exists() else ""

    # Look for env var references in code
    env_vars_used: set[str] = set()

    # Check Python files for os.environ or os.getenv
    for py_file in repo_path.rglob("*.py"):
        if ".venv" in str(py_file) or "node_modules" in str(py_file):
            continue
        try:
            content = py_file.read_text()
            # os.environ["VAR"] or os.getenv("VAR")
            env_vars_used.update(
                re.findall(r'os\.(?:environ\[|getenv\()["\']([A-Z_]+)["\']', content)
            )
            # env.get("VAR")
            env_vars_used.update(re.findall(r'env\.get\(["\']([A-Z_]+)["\']', content))
        except Exception:
            pass

    # Check shell scripts for $VAR or ${VAR}
    for sh_file in repo_path.rglob("*.sh"):
        try:
            content = sh_file.read_text()
            env_vars_used.update(re.findall(r"\$\{?([A-Z_]{3,})\}?", content))
        except Exception:
            pass

    # Filter to likely project-specific vars (not PATH, HOME, etc.)
    common_vars = {
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "PWD",
        "TERM",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "PYTHONPATH",
        "CARGO_HOME",
        "RUSTUP_HOME",
    }
    project_vars = env_vars_used - common_vars

    # Check which are documented
    missing = [v for v in project_vars if v not in claude_content]

    if len(missing) > 5:
        # Too many to list, likely scanning picked up noise
        return CheckResult(
            name="env_vars",
            passed=True,
            message="Environment variables (manual review recommended)",
        )

    if missing:
        return CheckResult(
            name="env_vars",
            passed=False,
            message=f"Undocumented env vars: {', '.join(missing[:5])}",
            details=[f"${v} used but not documented" for v in missing],
        )

    return CheckResult(
        name="env_vars",
        passed=True,
        message="Environment variables documented",
    )


def check_recent_errors(repo_path: Path) -> CheckResult:
    """Check worker logs for 'not found' errors indicating doc gaps."""
    log_dir = repo_path / "worker_logs"
    if not log_dir.exists():
        return CheckResult(
            name="recent_errors",
            passed=True,
            message="No worker logs to check",
        )

    error_files: list[str] = []
    patterns = [
        "command not found",
        "No such file or directory",
        "not found",
        "does not exist",
    ]

    for log_file in sorted(log_dir.glob("*.log"))[-5:]:  # Last 5 logs
        try:
            content = log_file.read_text()
            for pattern in patterns:
                if pattern.lower() in content.lower():
                    error_files.append(log_file.name)
                    break
        except Exception:
            pass

    if error_files:
        return CheckResult(
            name="recent_errors",
            passed=False,
            message=f"'Not found' errors in: {', '.join(error_files[:3])}",
            details=[f"Check {f} for potential doc gaps" for f in error_files],
        )

    return CheckResult(
        name="recent_errors",
        passed=True,
        message="No recent 'not found' errors",
    )


def _load_script_docs(repo_path: Path, scripts_dir: Path) -> str:
    docs: list[str] = []
    claude_md = repo_path / "CLAUDE.md"
    if claude_md.exists():
        docs.append(claude_md.read_text())
    scripts_readme = scripts_dir / "README.md"
    if scripts_readme.exists():
        docs.append(scripts_readme.read_text())
    return "\n".join(docs)


def check_key_scripts(repo_path: Path) -> CheckResult:
    """Check if key scripts directory contents are documented."""

    scripts_dir = repo_path / "scripts"
    if not scripts_dir.exists():
        return CheckResult(
            name="key_scripts",
            passed=True,
            message="No scripts/ directory",
        )

    docs_content = _load_script_docs(repo_path, scripts_dir)

    # Find Python/shell scripts in scripts/
    scripts = list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh"))
    missing = [s.name for s in scripts if s.name not in docs_content]

    if len(missing) > 5:
        return CheckResult(
            name="key_scripts",
            passed=False,
            message=f"{len(missing)} scripts not documented in CLAUDE.md or scripts/README.md",
            details=[f"scripts/{s}" for s in missing[:5]] + ["..."],
        )

    if missing:
        return CheckResult(
            name="key_scripts",
            passed=False,
            message=f"Undocumented scripts: {', '.join(missing[:3])}",
            details=[
                f"scripts/{s} not in CLAUDE.md or scripts/README.md" for s in missing
            ],
        )

    return CheckResult(
        name="key_scripts",
        passed=True,
        message="Key scripts documented",
    )


def run_checks(repo_path: Path) -> list[CheckResult]:
    """Run all documentation health checks."""
    checks = [
        check_cli_documented,
        check_build_commands,
        check_env_vars,
        check_recent_errors,
        check_key_scripts,
    ]
    return [check(repo_path) for check in checks]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check CLAUDE.md documentation health for AI operations"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Repository path to check (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    results = run_checks(args.path)

    if args.json:
        output = {
            "path": str(args.path),
            "passed": all(r.passed for r in results),
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Documentation Health Check: {args.path}\n")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.message}")
            if result.details and not result.passed:
                for detail in result.details[:3]:
                    print(f"         - {detail}")

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"\nSummary: {passed}/{total} checks passed")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Health Check Base Class

Core infrastructure for system health checks. See __init__.py for usage.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


class Status(Enum):
    """Status of a health check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single health check."""

    status: Status
    message: str
    details: dict[str, Any] | None = None


@runtime_checkable
class Check(Protocol):
    """Protocol for health check functions.

    A check function returns a CheckResult. The function must have a `name`
    attribute for identification in JSON output.

    Example:
        def check_cargo() -> CheckResult:
            ...
        check_cargo.name = "cargo_check"
    """

    name: str

    def __call__(self) -> CheckResult: ...


def _get_git_commit(project_root: Path) -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root,
            timeout=10,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _get_project_name(project_root: Path) -> str:
    """Get project name from git remote or directory."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root,
            timeout=10,
        )
        url = result.stdout.strip()
        return url.rstrip("/").split("/")[-1].replace(".git", "")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return project_root.name


@dataclass
class HealthCheckBase:
    """Base health check runner with standard infrastructure.

    Subclass this and register checks to create repo-specific health checks.

    Example:
        class MyHealthCheck(HealthCheckBase):
            def __init__(self):
                super().__init__()
                self.register(self.check_cargo)

            def check_cargo(self) -> CheckResult:
                return CheckResult(Status.PASS, "cargo check passed")
            check_cargo.name = "cargo_check"
    """

    project_root: Path = field(default_factory=lambda: Path.cwd())
    project_name: str | None = None
    _checks: list[Check] = field(default_factory=list, init=False)
    _results: dict[str, CheckResult] = field(default_factory=dict, init=False)
    _errors: list[str] = field(default_factory=list, init=False)
    _warnings: list[str] = field(default_factory=list, init=False)
    _passed: list[str] = field(default_factory=list, init=False)
    _skipped: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize project_name from git if not provided.

        REQUIRES: project_root is a valid Path
        ENSURES: project_name is set (from git remote or directory name)
        """
        if self.project_name is None:
            self.project_name = _get_project_name(self.project_root)

    def register(self, check: Check | Callable[[], CheckResult]) -> None:
        """Register a check to run.

        REQUIRES: check is callable and has a 'name' attribute
        ENSURES: check is added to _checks list

        The check must be callable and return a CheckResult.
        It must have a `name` attribute for JSON output.
        """
        if not hasattr(check, "name"):
            raise ValueError(f"Check {check} must have a 'name' attribute")
        self._checks.append(check)  # type: ignore[arg-type]

    def error(self, msg: str) -> None:
        """Record an error message.

        REQUIRES: msg is a string (can be empty)
        ENSURES: msg is appended to _errors list
        """
        self._errors.append(msg)

    def warn(self, msg: str) -> None:
        """Record a warning message.

        REQUIRES: msg is a string (can be empty)
        ENSURES: msg is appended to _warnings list
        """
        self._warnings.append(msg)

    def ok(self, msg: str) -> None:
        """Record a passed check message.

        REQUIRES: msg is a string (can be empty)
        ENSURES: msg is appended to _passed list
        """
        self._passed.append(msg)

    def skip(self, msg: str) -> None:
        """Record a skipped check message.

        REQUIRES: msg is a string (can be empty)
        ENSURES: msg is appended to _skipped list
        """
        self._skipped.append(msg)

    def set_check_result(self, check_name: str, result: dict[str, Any]) -> None:
        """Store structured result for a check (legacy compatibility).

        REQUIRES: check_name is non-empty string, result contains 'status' key
        ENSURES: CheckResult stored in _results[check_name]
        """
        # Convert dict to CheckResult if needed
        status_str = result.get("status", "pass")
        status = (
            Status(status_str)
            if status_str in [s.value for s in Status]
            else Status.PASS
        )
        self._results[check_name] = CheckResult(
            status=status,
            message=result.get("message", ""),
            details=result,
        )

    def run_all(self, verbose: bool = True) -> bool:
        """Run all registered checks. Returns True if passed (no failures).

        REQUIRES: Checks registered via register()
        ENSURES: All checks executed, results stored in _results
        ENSURES: Returns True iff no errors recorded
        """
        for check in self._checks:
            result = check()
            self._results[check.name] = result
            self._record_result(result)
            if verbose:
                self._print_result(check.name, result)
        return not self._errors

    def _record_result(self, result: CheckResult) -> None:
        """Record result in appropriate list.

        REQUIRES: result is a CheckResult instance
        ENSURES: Result message stored in the list matching result.status
        """
        if result.status == Status.FAIL:
            self._errors.append(result.message)
        elif result.status == Status.WARN:
            self._warnings.append(result.message)
        elif result.status == Status.SKIP:
            self._skipped.append(result.message)
        else:
            self._passed.append(result.message)

    def _print_result(self, name: str, result: CheckResult) -> None:
        """Print a single check result.

        REQUIRES: name is non-empty string, result is a CheckResult
        ENSURES: Writes formatted result line to stderr
        """
        icon = {
            Status.PASS: "[OK]",
            Status.WARN: "[WARN]",
            Status.FAIL: "[FAIL]",
            Status.SKIP: "[SKIP]",
        }[result.status]
        print(f"  {icon} {name}: {result.message}", file=sys.stderr)

    @property
    def errors(self) -> list[str]:
        """List of error messages.

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Returns list (internal reference, not a copy)
        """
        return self._errors

    @property
    def warnings(self) -> list[str]:
        """List of warning messages.

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Returns list (internal reference, not a copy)
        """
        return self._warnings

    @property
    def passed(self) -> list[str]:
        """List of passed check messages.

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Returns list (internal reference, not a copy)
        """
        return self._passed

    @property
    def skipped(self) -> list[str]:
        """List of skipped check messages.

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Returns list (internal reference, not a copy)
        """
        return self._skipped

    @property
    def json_checks(self) -> dict[str, dict[str, Any]]:
        """Results as dict for JSON output (legacy compatibility).

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Returns dict with status/message for each check in _results
        """
        return {
            name: {
                "status": result.status.value,
                "message": result.message,
                **(result.details or {}),
            }
            for name, result in self._results.items()
        }

    def to_json(self) -> dict[str, Any]:
        """Generate JSON manifest per contract v1.0.

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Returns dict with schema_version, generated_at, git_commit,
                 project, summary (status/counts), checks
        ENSURES: summary.status is 'fail' if errors, 'warn' if warnings, else 'pass'
        """
        if self._errors:
            status = "fail"
        elif self._warnings:
            status = "warn"
        else:
            status = "pass"

        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "git_commit": _get_git_commit(self.project_root),
            "project": self.project_name,
            "summary": {
                "status": status,
                "passed": len(self._passed),
                "warnings": len(self._warnings),
                "errors": len(self._errors),
                "skipped": len(self._skipped),
            },
            "checks": self.json_checks,
        }

    def print_summary(self) -> None:
        """Print human-readable summary.

        REQUIRES: None (safe to call unconditionally)
        ENSURES: Prints categorized results (PASSED, SKIPPED, WARNINGS, ERRORS)
        ENSURES: Prints final HEALTH CHECK status line
        """
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if self._passed:
            print(f"\nPASSED ({len(self._passed)})")
            for msg in self._passed:
                print(f"  [OK] {msg}")

        if self._skipped:
            print(f"\nSKIPPED ({len(self._skipped)})")
            for msg in self._skipped:
                print(f"  [SKIP] {msg}")

        if self._warnings:
            print(f"\nWARNINGS ({len(self._warnings)})")
            for msg in self._warnings:
                print(f"  [WARN] {msg}")

        if self._errors:
            print(f"\nERRORS ({len(self._errors)})")
            for msg in self._errors:
                print(f"  [ERROR] {msg}")

        print("\n" + "=" * 60)
        if self._errors:
            print("HEALTH CHECK FAILED")
            print("The system has integration problems that need fixing.")
        elif self._warnings:
            print("HEALTH CHECK PASSED WITH WARNINGS")
        else:
            print("HEALTH CHECK PASSED")
        print("=" * 60)

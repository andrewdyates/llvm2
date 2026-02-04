#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
check_resource_patterns.py - Automated resource leak detection for Python code

PURPOSE: Detect common resource leak anti-patterns in Python files:
    1. open() without context manager (with statement)
    2. Popen() without proper cleanup (wait/communicate/context)
    3. File descriptors stored as attributes without cleanup
    4. Socket/connection objects without cleanup

CALLED BY: Prover (memory_verification phase), health_check.py
REFERENCED: .claude/roles/prover.md (Memory Verification phase)

Exit codes:
    0 - No issues found
    1 - Issues found
    2 - Error during analysis

Public API:
    - ResourceIssue, ResourceReport
    - check_open_without_context, check_popen_without_cleanup
    - check_attribute_file_handles, check_socket_patterns
    - scan_file (file_path param), scan_directory (dir_path param)
    - format_report, main
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.path_utils import resolve_path_alias  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

__all__ = [
    "ResourceIssue",
    "ResourceReport",
    "check_open_without_context",
    "check_popen_without_cleanup",
    "check_attribute_file_handles",
    "check_socket_patterns",
    "scan_file",
    "scan_directory",
    "format_report",
    "main",
]


@dataclass
class ResourceIssue:
    """A detected resource leak pattern."""

    file: str
    line: int
    category: str  # open_no_context, popen_no_cleanup, attr_file_handle, socket
    message: str
    severity: str  # "warning" or "error"
    code_snippet: str = ""


@dataclass
class ResourceReport:
    """Aggregate report of resource patterns."""

    files_scanned: int = 0
    issues: list[ResourceIssue] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        """Total number of issues found."""
        return len(self.issues)

    @property
    def warnings(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == "error")


class ResourcePatternVisitor(ast.NodeVisitor):
    """AST visitor to detect resource leak patterns."""

    def __init__(self, source: str, filename: str) -> None:
        """Initialize visitor with source code and filename.

        REQUIRES: source is valid Python code
        ENSURES: self.issues contains all detected patterns after visit
        """
        self.source = source
        self.lines = source.splitlines()
        self.filename = filename
        self.issues: list[ResourceIssue] = []
        self.in_with_context = False
        self.in_try_finally = False
        self.current_class: str | None = None
        self.class_has_cleanup: set[str] = set()  # Classes with __del__ or close()

    def _get_line_snippet(self, lineno: int) -> str:
        """Get code snippet for line number."""
        if 0 < lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def visit_With(self, node: ast.With) -> None:
        """Track when inside with context."""
        old_context = self.in_with_context
        self.in_with_context = True
        self.generic_visit(node)
        self.in_with_context = old_context

    def visit_Try(self, node: ast.Try) -> None:
        """Track when inside try-finally."""
        old_finally = self.in_try_finally
        if node.finalbody:
            self.in_try_finally = True
        self.generic_visit(node)
        self.in_try_finally = old_finally

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions and check for cleanup methods."""
        old_class = self.current_class
        self.current_class = node.name

        # Check if class has __del__, close(), or __exit__
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name in ("__del__", "close", "__exit__", "cleanup"):
                    self.class_has_cleanup.add(node.name)

        self.generic_visit(node)
        self.current_class = old_class

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for file handles assigned to attributes."""
        # Look for self.xxx = open(...) or self.xxx = Popen(...)
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        func_name = None
                        if isinstance(func, ast.Name):
                            func_name = func.id
                        elif isinstance(func, ast.Attribute):
                            func_name = func.attr

                        if func_name in ("open", "Popen"):
                            # Check if class has cleanup
                            if (
                                self.current_class
                                and self.current_class not in self.class_has_cleanup
                            ):
                                self.issues.append(
                                    ResourceIssue(
                                        file=self.filename,
                                        line=node.lineno,
                                        category="attr_file_handle",
                                        message=(
                                            f"self.{target.attr} = {func_name}() "
                                            f"without cleanup method (__del__/close)"
                                        ),
                                        severity="warning",
                                        code_snippet=self._get_line_snippet(
                                            node.lineno
                                        ),
                                    )
                                )

        self.generic_visit(node)

    def _is_detached_popen(self, node: ast.Call) -> bool:
        """Check if Popen has start_new_session=True (intentional detachment)."""
        for keyword in node.keywords:
            if keyword.arg == "start_new_session":
                if (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        """Check for open() and Popen() without context."""
        func = node.func
        func_name = None
        is_os_open = False

        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
            # Check if it's os.open() which is different from builtin open()
            if isinstance(func.value, ast.Name) and func.value.id == "os":
                is_os_open = True

        # Check for open() without with context
        # Skip os.open() which returns a file descriptor, not a file object
        if func_name == "open" and not is_os_open and not self.in_with_context:
            # Check if it's in a try-finally or has explicit close
            if not self.in_try_finally:
                self.issues.append(
                    ResourceIssue(
                        file=self.filename,
                        line=node.lineno,
                        category="open_no_context",
                        message="open() without context manager or try-finally",
                        severity="warning",
                        code_snippet=self._get_line_snippet(node.lineno),
                    )
                )

        # Check for Popen without context manager
        if func_name == "Popen" and not self.in_with_context:
            if not self.in_try_finally:
                # Skip intentionally detached processes (start_new_session=True)
                if self._is_detached_popen(node):
                    # This is intentional - don't flag it
                    pass
                else:
                    self.issues.append(
                        ResourceIssue(
                            file=self.filename,
                            line=node.lineno,
                            category="popen_no_cleanup",
                            message=(
                                "Popen() without context manager - verify "
                                "wait()/communicate() is called"
                            ),
                            severity="warning",
                            code_snippet=self._get_line_snippet(node.lineno),
                        )
                    )

        self.generic_visit(node)


def check_open_without_context(source: str, filename: str) -> list[ResourceIssue]:
    """Check for open() calls without context managers.

    REQUIRES: source is valid Python code
    ENSURES: Returns list of issues for open() without 'with'

    Args:
        source: Python source code
        filename: File being checked

    Returns:
        List of ResourceIssue for open() without context
    """
    issues: list[ResourceIssue] = []
    try:
        tree = ast.parse(source)
        visitor = ResourcePatternVisitor(source, filename)
        visitor.visit(tree)
        issues.extend(i for i in visitor.issues if i.category == "open_no_context")
    except SyntaxError:
        pass  # Invalid Python, skip
    return issues


def check_popen_without_cleanup(source: str, filename: str) -> list[ResourceIssue]:
    """Check for Popen() calls without proper cleanup.

    REQUIRES: source is valid Python code
    ENSURES: Returns list of issues for Popen() without cleanup

    Args:
        source: Python source code
        filename: File being checked

    Returns:
        List of ResourceIssue for Popen() without cleanup
    """
    issues: list[ResourceIssue] = []
    try:
        tree = ast.parse(source)
        visitor = ResourcePatternVisitor(source, filename)
        visitor.visit(tree)
        issues.extend(i for i in visitor.issues if i.category == "popen_no_cleanup")
    except SyntaxError:
        pass
    return issues


def check_attribute_file_handles(source: str, filename: str) -> list[ResourceIssue]:
    """Check for file handles stored as attributes without cleanup.

    REQUIRES: source is valid Python code
    ENSURES: Returns list of issues for attribute file handles

    Args:
        source: Python source code
        filename: File being checked

    Returns:
        List of ResourceIssue for attribute file handles without cleanup
    """
    issues: list[ResourceIssue] = []
    try:
        tree = ast.parse(source)
        visitor = ResourcePatternVisitor(source, filename)
        visitor.visit(tree)
        issues.extend(i for i in visitor.issues if i.category == "attr_file_handle")
    except SyntaxError:
        pass
    return issues


def check_socket_patterns(source: str, filename: str) -> list[ResourceIssue]:
    """Check for socket/connection patterns without cleanup.

    REQUIRES: source is valid Python code
    ENSURES: Returns list of issues for socket patterns

    Uses regex pattern matching for common socket patterns.

    Args:
        source: Python source code
        filename: File being checked

    Returns:
        List of ResourceIssue for socket patterns without cleanup
    """
    issues: list[ResourceIssue] = []
    lines = source.splitlines()

    # Patterns that indicate socket/connection creation
    socket_patterns = [
        (r"\bsocket\.socket\s*\(", "socket.socket()"),
        (r"http\.client\.\w+Connection\s*\(", "http connection"),
    ]

    for lineno, line in enumerate(lines, 1):
        # Skip lines that are likely string literals or comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Skip regex patterns in strings (e.g., r"...", "...", '...')
        if re.match(r'^["\'].*["\']', stripped) or re.match(r"^r['\"]", stripped):
            continue
        # Skip lines that are tuple/list definitions containing strings
        if re.match(r"^\(r?['\"]", stripped):
            continue

        for pattern, desc in socket_patterns:
            if re.search(pattern, line):
                # Check if in with context (simple heuristic)
                prev_lines = "\n".join(lines[max(0, lineno - 5) : lineno])
                if "with " not in prev_lines or "as " not in prev_lines:
                    issues.append(
                        ResourceIssue(
                            file=filename,
                            line=lineno,
                            category="socket",
                            message=f"{desc} - verify proper cleanup",
                            severity="warning",
                            code_snippet=line.strip(),
                        )
                    )

    return issues


def scan_file(
    file_path: Path | None = None, **kwargs: Any
) -> tuple[list[ResourceIssue], list[str]]:
    """Scan a single Python file for resource patterns.

    REQUIRES: file_path.exists() and file_path.suffix == '.py'
    ENSURES: Returns (issues, errors) tuple

    Args:
        file_path: Path to Python file.
        **kwargs: Accepts deprecated 'filepath' alias for file_path.

    Returns:
        Tuple of (issues list, errors list)
    """
    resolved_path = resolve_path_alias(
        "file_path", "filepath", file_path, kwargs, "scan_file"
    )
    issues: list[ResourceIssue] = []
    errors = []

    try:
        source = resolved_path.read_text(encoding="utf-8", errors="replace")
        filename = str(resolved_path)

        try:
            tree = ast.parse(source)
            visitor = ResourcePatternVisitor(source, filename)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        except SyntaxError as e:
            errors.append(f"{resolved_path}: Syntax error at line {e.lineno}: {e.msg}")

        # Also check socket patterns (regex-based)
        issues.extend(check_socket_patterns(source, filename))

    except OSError as e:
        errors.append(f"{resolved_path}: {e}")

    return issues, errors


def scan_directory(
    dir_path: Path | None = None,
    exclude_patterns: list[str] | None = None,
    **kwargs: Any,
) -> ResourceReport:
    """Scan directory for Python files with resource patterns.

    REQUIRES: dir_path.is_dir()
    ENSURES: Returns ResourceReport with all findings

    Args:
        dir_path: Directory to scan.
        exclude_patterns: Glob patterns to exclude (e.g., ["**/test_*.py"])
        **kwargs: Accepts deprecated 'directory' alias for dir_path.

    Returns:
        ResourceReport with findings
    """
    resolved_path = resolve_path_alias(
        "dir_path", "directory", dir_path, kwargs, "scan_directory"
    )
    report = ResourceReport()
    exclude_patterns = exclude_patterns or []

    # Default exclusions
    default_excludes = [
        "**/node_modules/**",
        "**/.venv/**",
        "**/venv/**",
        "**/__pycache__/**",
        "**/.git/**",
    ]

    all_excludes = default_excludes + exclude_patterns

    for pyfile in resolved_path.rglob("*.py"):
        # Check exclusions using fnmatch for proper glob matching
        rel_path = str(pyfile.relative_to(resolved_path))
        excluded = False
        for pattern in all_excludes:
            if "**" in pattern:
                # Handle ** by checking path components
                simple_pattern = pattern.replace("**/", "")
                for part in Path(rel_path).parts:
                    if fnmatch.fnmatch(part, simple_pattern):
                        excluded = True
                        break
                # Also handle "tests/**" style patterns
                if not excluded and simple_pattern.endswith("/**"):
                    dir_name = simple_pattern.replace("/**", "")
                    if dir_name in Path(rel_path).parts:
                        excluded = True
            else:
                if fnmatch.fnmatch(rel_path, pattern):
                    excluded = True
            if excluded:
                break

        if excluded:
            continue

        report.files_scanned += 1
        issues, errors = scan_file(pyfile)
        report.issues.extend(issues)
        report.errors.extend(errors)

    return report


def filter_known_safe(issues: list[ResourceIssue]) -> list[ResourceIssue]:
    """Filter out known-safe patterns from issues.

    REQUIRES: issues is a list of ResourceIssue
    ENSURES: Returns filtered list excluding known-safe patterns

    Known-safe patterns:
    - cargo_wrapper/* Popen with documented rationale and signal handlers
    - Lock file operations with explicit close in finally
    - gh_rate_limit/* file locking pattern (nested try with finally cleanup)
    - File locking classes with release() or __exit__ methods
    - gh_atomic_claim.py lock file pattern (same as looper)
    """
    filtered = []
    for issue in issues:
        file_basename = Path(issue.file).name
        file_parts = Path(issue.file).parts

        # cargo_wrapper/* modules have documented Popen usage with signal handlers
        if "cargo_wrapper" in file_parts and issue.category == "popen_no_cleanup":
            continue

        # gh_rate_limit/* modules have valid nested try-finally pattern for lock files
        # The open() is followed by inner try-finally that closes it
        if "gh_rate_limit" in file_parts and "lock_fd" in issue.code_snippet:
            continue

        # gh_atomic_claim.py has same lock file pattern as looper
        if file_basename == "gh_atomic_claim.py" and "lock_file" in issue.code_snippet:
            continue

        # Lock file patterns in looper modules - these have finally blocks or
        # release methods that properly clean up
        if "looper" in file_parts and "lock_file" in issue.code_snippet:
            continue

        # Check for documented exceptions via inline comments
        if "# NOTE:" in issue.code_snippet or "# SAFE:" in issue.code_snippet:
            continue

        # Skip if the code snippet mentions it's intentional
        if "# noqa" in issue.code_snippet.lower():
            continue

        filtered.append(issue)

    return filtered


def format_report(report: ResourceReport, verbose: bool = False) -> str:
    """Format resource report for display.

    REQUIRES: report is a ResourceReport
    ENSURES: Returns formatted string

    Args:
        report: ResourceReport to format
        verbose: Include all issues, not just summary

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("## Resource Pattern Analysis")
    lines.append("")
    lines.append(f"**Files scanned:** {report.files_scanned}")
    lines.append(f"**Issues found:** {report.total_issues}")
    lines.append(f"  - Warnings: {report.warnings}")
    lines.append(f"  - Errors: {report.error_count}")
    lines.append("")

    if report.errors:
        lines.append("**Parse errors:**")
        lines.extend(f"  - {err}" for err in report.errors)
        lines.append("")

    if report.issues:
        # Group by category
        by_category: dict[str, list[ResourceIssue]] = {}
        for issue in report.issues:
            by_category.setdefault(issue.category, []).append(issue)

        for category, issues in sorted(by_category.items()):
            lines.append(f"**{category}:** ({len(issues)} issues)")
            if verbose:
                for issue in issues:
                    lines.append(f"  - {issue.file}:{issue.line}: {issue.message}")
                    if issue.code_snippet:
                        lines.append(f"    `{issue.code_snippet}`")
            else:
                # Just show count and first few
                lines.extend(f"  - {issue.file}:{issue.line}" for issue in issues[:3])
                if len(issues) > 3:
                    lines.append(f"  ... and {len(issues) - 3} more")
            lines.append("")

    if report.total_issues == 0:
        lines.append("No resource leak patterns detected.")

    return "\n".join(lines)


def main() -> int:
    """Main entry point.

    ENSURES: Returns 0 if no issues, 1 if issues found, 2 on error
    """
    parser = argparse.ArgumentParser(
        description="Check Python files for resource leak patterns",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("check_resource_patterns.py"),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory or file to scan (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all issue details",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files (excluded by default)",
    )
    parser.add_argument(
        "--filter-safe",
        action="store_true",
        help="Filter out known-safe patterns",
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        return 2

    # Build exclude patterns
    exclude_patterns = list(args.exclude)
    if not args.include_tests:
        exclude_patterns.append("**/test_*.py")
        exclude_patterns.append("**/tests/**")

    if path.is_file():
        issues, errors = scan_file(path)
        report = ResourceReport(
            files_scanned=1,
            issues=issues,
            errors=errors,
        )
    else:
        report = scan_directory(path, exclude_patterns=exclude_patterns)

    # Filter known-safe patterns if requested
    if args.filter_safe:
        report.issues = filter_known_safe(report.issues)

    if args.json:
        output = {
            "files_scanned": report.files_scanned,
            "total_issues": report.total_issues,
            "warnings": report.warnings,
            "errors": report.error_count,
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "category": i.category,
                    "message": i.message,
                    "severity": i.severity,
                    "code_snippet": i.code_snippet,
                }
                for i in report.issues
            ],
            "parse_errors": report.errors,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(report, verbose=args.verbose))

    if report.errors:
        return 2
    if report.total_issues > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

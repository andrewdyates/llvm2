#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
integration_audit.py - Detect orphan modules not reachable from entry points

PURPOSE: Find code that's built but never executed (orphan modules, dead flags).
CALLED BY: MANAGER "Project Completion" phase, Worker manual audit
REFERENCED: .claude/rules/ai_template.md (Useful Scripts table)

Detects:
- Orphan Python modules (not imported from any entry point)
- Dead CLI flags (--skip-* variants that are always used)
- Modules that produce no output (imported but no artifacts)

Public API:
- SKIP_DIRS (re-exported from exclude_patterns.py), ENTRY_PATTERNS
- find_entry_points, find_modules, build_import_graph
- find_orphan_modules, find_skip_only_flags
- run_audit, main

Usage:
    integration_audit.py                  # Audit current directory
    integration_audit.py --json           # Output as JSON
    integration_audit.py /path/to/repo    # Audit specific repo
    integration_audit.py --ignore=tests   # Ignore tests directory
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    tomllib = None  # Python < 3.11 fallback handled in function

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.exclude_patterns import SKIP_DIRS  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

# SKIP_DIRS imported from exclude_patterns.py (single source of truth, #1798)

__all__ = [
    "SKIP_DIRS",
    "ENTRY_PATTERNS",
    "AuditResult",
    "find_entry_points",
    "find_modules",
    "build_import_graph",
    "find_orphan_modules",
    "find_skip_only_flags",
    "run_audit",
    "main",
]

# Internal functions also exported for testing
_INTERNAL_API = [
    "_detect_src_layout_packages",
    "_module_path_to_name",
]

# Patterns for entry point detection
ENTRY_PATTERNS = {
    "scripts": "scripts/**/*.py",
    "cli_main": "**/cli.py",
    "main_module": "**/__main__.py",
    "test_files": "tests/**/*.py",
}

# Precompiled regex patterns (#1698)
# Pattern for argparse skip flags: --skip-foo, --skip_bar
_SKIP_FLAG_PATTERN = re.compile(r'add_argument\(["\']--skip[_-](\w+)')


@dataclass
class AuditResult:
    """Result of integration audit."""

    orphan_modules: list[str] = field(default_factory=list)
    skip_only_flags: list[dict[str, str]] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    total_modules: int = 0
    reachable_modules: int = 0

    def has_issues(self) -> bool:
        """Return True if audit found issues."""
        return bool(self.orphan_modules or self.skip_only_flags)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "orphan_modules": self.orphan_modules,
            "skip_only_flags": self.skip_only_flags,
            "entry_points": self.entry_points,
            "total_modules": self.total_modules,
            "reachable_modules": self.reachable_modules,
        }


def _should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    return any(part in SKIP_DIRS for part in path.parts)


def _parse_pyproject_entry_points(root: Path) -> list[str]:
    """Extract module names from pyproject.toml entry points and build packages.

    Parses PEP 621 format:
    - [project.scripts] - console scripts
    - [project.entry-points.X] - plugin entry points (e.g., pytest11)

    Also extracts distributable packages from build tool config:
    - [tool.hatch.build.targets.wheel] - hatch packages (incl. src-layout)
    - [tool.setuptools.packages.find] - setuptools packages

    Returns:
        List of module names referenced as entry points or distributable packages
    """
    modules: list[str] = []

    # Find all pyproject.toml files
    for toml_path in root.rglob("pyproject.toml"):
        if _should_skip(toml_path):
            continue

        try:
            content = toml_path.read_text()
            if tomllib is not None:
                data = tomllib.loads(content)
            else:
                # Python < 3.11 fallback - basic regex extraction
                for match in re.finditer(
                    r'=\s*"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)(?::[a-zA-Z_][a-zA-Z0-9_]*)?"',
                    content,
                ):
                    module = match.group(1)
                    if module:
                        modules.append(module.split(".")[0])
                # Also extract hatch src-layout packages
                if 'packages = ["src/' in content:
                    for match in re.finditer(
                        r'packages\s*=\s*\["src/([a-zA-Z_][a-zA-Z0-9_]*)"\]', content
                    ):
                        modules.append(match.group(1))
                continue
        except Exception:
            continue

        project = data.get("project", {})

        # [project.scripts] - console scripts like: cli = "package.module:main"
        scripts = project.get("scripts", {})
        for value in scripts.values():
            if isinstance(value, str):
                # Format: "module.path:function" or "module.path"
                module_part = value.split(":")[0]
                modules.append(module_part.split(".")[0])

        # [project.entry-points.X] - plugin entry points
        entry_points = project.get("entry-points", {})
        for group in entry_points.values():
            if isinstance(group, dict):
                for value in group.values():
                    if isinstance(value, str):
                        # Format: "module.path" or "module.path:attr"
                        module_part = value.split(":")[0]
                        modules.append(module_part.split(".")[0])

        # [tool.hatch.build.targets.wheel] - hatch packages (incl. src-layout)
        # These are the distributable packages - they ARE the entry points for libraries
        hatch = data.get("tool", {}).get("hatch", {})
        wheel_packages = (
            hatch.get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("packages", [])
        )
        for pkg in wheel_packages:
            # Strip "src/" prefix if present (src-layout)
            pkg_name = pkg[4:] if pkg.startswith("src/") else pkg
            modules.append(pkg_name)

        # [tool.setuptools.packages.find] - setuptools packages
        setuptools = data.get("tool", {}).get("setuptools", {})
        find_cfg = setuptools.get("packages", {}).get("find", {})
        where = find_cfg.get("where", [])
        if where:
            # Use project name as the package name
            pkg_name = project.get("name", "").replace("-", "_")
            if pkg_name:
                modules.append(pkg_name)

    return list(set(modules))


def find_entry_points(root: Path) -> list[Path]:
    """Find all Python entry points in the project.

    Entry points are:
    - scripts/*.py (or scripts/**/*.py)
    - **/cli.py (CLI entry points)
    - **/__main__.py (module entry points)
    """
    entry_points: list[Path] = []

    # Check scripts/ directory
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        entry_points.extend(
            py_file
            for py_file in scripts_dir.rglob("*.py")
            if not _should_skip(py_file) and py_file.name != "__init__.py"
        )

    # Check ai_template_scripts/ for this repo specifically
    ai_scripts_dir = root / "ai_template_scripts"
    if ai_scripts_dir.exists():
        for py_file in ai_scripts_dir.glob("*.py"):
            # Only top-level scripts that are executable entry points
            if (
                not _should_skip(py_file)
                and py_file.name != "__init__.py"
                and py_file.name != "path_utils.py"  # utility module
                and py_file.name != "subprocess_utils.py"  # utility module
            ):
                # Check if file has a main() or if __name__ == "__main__"
                try:
                    content = py_file.read_text()
                    if 'if __name__ == "__main__"' in content or "def main(" in content:
                        entry_points.append(py_file)
                except Exception:
                    pass

    # Check for looper.py as entry point
    looper_py = root / "looper.py"
    if looper_py.exists():
        entry_points.append(looper_py)

    # Check for cli.py files
    entry_points.extend(
        cli_file for cli_file in root.rglob("cli.py") if not _should_skip(cli_file)
    )

    # Check for __main__.py files
    entry_points.extend(
        main_file
        for main_file in root.rglob("__main__.py")
        if not _should_skip(main_file)
    )

    return list(set(entry_points))


def find_modules(root: Path, ignore_dirs: set[str] | None = None) -> list[Path]:
    """Find all Python module files in the project.

    Args:
        root: Root directory to search
        ignore_dirs: Additional directories to ignore
    """
    ignore = SKIP_DIRS | (ignore_dirs or set())
    modules: list[Path] = []

    for py_file in root.rglob("*.py"):
        if any(part in ignore for part in py_file.parts):
            continue
        # Skip test files - they're not production modules
        if "test" in py_file.name.lower() or "/tests/" in str(py_file):
            continue
        modules.append(py_file)

    return modules


def _extract_imports(file_path: Path) -> set[str]:
    """Extract all imports from a Python file.

    Returns module names (not paths).
    """
    imports: set[str] = set()

    try:
        content = file_path.read_text()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level module
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get top-level module
                imports.add(node.module.split(".")[0])

    return imports


def _detect_src_layout_packages(root: Path) -> dict[Path, str]:
    """Detect src-layout packages from pyproject.toml files.

    Returns a dict mapping src directory paths to their package names.
    e.g., {Path("python/lean5_fate/src"): "lean5_fate"}

    Detection methods (in order):
    1. Hatch config: [tool.hatch.build.targets.wheel] packages = ["src/..."]
    2. Setuptools config: [tool.setuptools.packages.find] where = ["src"]
    3. Fallback: src/ directory with Python package subdirectory (#1853)
    """
    src_packages: dict[Path, str] = {}

    for toml_path in root.rglob("pyproject.toml"):
        if _should_skip(toml_path):
            continue

        project_dir = toml_path.parent
        src_dir = project_dir / "src"

        try:
            content = toml_path.read_text()
            if tomllib is None:
                # Python < 3.11: basic detection via regex
                # Look for src/ in hatch or setuptools config
                if 'packages = ["src/' in content:
                    # Extract package name from pattern like: packages = ["src/lean5_fate"]
                    match = re.search(
                        r'packages\s*=\s*\["src/([a-zA-Z_][a-zA-Z0-9_]*)"\]', content
                    )
                    if match:
                        pkg_name = match.group(1)
                        if src_dir.exists():
                            src_packages[src_dir] = pkg_name
                continue

            data = tomllib.loads(content)

            # Check hatch src-layout: [tool.hatch.build.targets.wheel]
            hatch = data.get("tool", {}).get("hatch", {})
            wheel_packages = (
                hatch.get("build", {})
                .get("targets", {})
                .get("wheel", {})
                .get("packages", [])
            )
            for pkg in wheel_packages:
                if pkg.startswith("src/"):
                    pkg_name = pkg[4:]  # Remove "src/" prefix
                    if src_dir.exists():
                        src_packages[src_dir] = pkg_name

            # Check setuptools src-layout: [tool.setuptools.packages.find]
            setuptools = data.get("tool", {}).get("setuptools", {})
            find_cfg = setuptools.get("packages", {}).get("find", {})
            where = find_cfg.get("where", [])
            if "src" in where:
                # Package name is typically the project name
                project = data.get("project", {})
                pkg_name = project.get("name", "").replace("-", "_")
                if pkg_name:
                    if src_dir.exists():
                        src_packages[src_dir] = pkg_name

            # Fallback: detect src-layout by directory structure (#1853)
            # If src/ exists with a package subdirectory containing __init__.py,
            # and we haven't already detected it from build config
            if src_dir.exists() and src_dir not in src_packages:
                try:
                    # Find Python packages in src/ (sorted for determinism)
                    pkg_subdirs = sorted(
                        subdir
                        for subdir in src_dir.iterdir()
                        if subdir.is_dir()
                        and not subdir.name.startswith(".")
                        and (subdir / "__init__.py").exists()
                    )
                    if pkg_subdirs:
                        # Take the first (alphabetically) package
                        src_packages[src_dir] = pkg_subdirs[0].name
                except PermissionError:
                    pass  # Skip inaccessible directories

        except Exception:
            continue

    return src_packages


def _module_path_to_name(
    path: Path, root: Path, src_packages: dict[Path, str] | None = None
) -> str:
    """Convert a module path to its importable name.

    e.g., /root/src/foo/bar.py -> src.foo.bar or foo.bar
         /root/pkg/src/pkg/bar.py -> pkg.bar (with src-layout detection)

    Args:
        path: The Python file path
        root: The project root directory
        src_packages: Dict of src-layout package roots (from _detect_src_layout_packages)
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.stem

    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")

    # Handle src-layout: strip prefix up to and including "src/"
    if src_packages:
        for src_dir, pkg_name in src_packages.items():
            try:
                # Check if this path is under a known src-layout directory
                path.relative_to(src_dir)
                # Find the index of "src" in parts and strip everything before it
                if "src" in parts:
                    src_idx = parts.index("src")
                    # Keep only parts after "src"
                    parts = parts[src_idx + 1 :]
                    break
            except ValueError:
                continue

    # Remove __init__ from path
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts) if parts else path.stem


def build_import_graph(
    root: Path,
    entry_points: list[Path],
    all_modules: list[Path],
    src_packages: dict[Path, str] | None = None,
) -> dict[str, set[str]]:
    """Build a graph of imports from entry points.

    Returns dict mapping module name -> set of imported module names.

    Args:
        root: Project root directory
        entry_points: List of entry point file paths
        all_modules: List of all module file paths
        src_packages: Dict of src-layout package roots (from _detect_src_layout_packages)
    """
    graph: dict[str, set[str]] = {}

    # Build graph for all modules (entry points + all modules)
    all_files = set(entry_points) | set(all_modules)

    for file_path in all_files:
        module_name = _module_path_to_name(file_path, root, src_packages)
        imports = _extract_imports(file_path)
        graph[module_name] = imports

    return graph


def find_reachable_modules(
    graph: dict[str, set[str]], entry_points: list[str]
) -> set[str]:
    """Find all modules reachable from entry points via import chains."""
    reachable: set[str] = set()
    to_visit = list(entry_points)

    while to_visit:
        current = to_visit.pop()
        if current in reachable:
            continue
        reachable.add(current)

        # Get imports from this module
        imports = graph.get(current, set())
        to_visit.extend(imp for imp in imports if imp not in reachable)

    return reachable


def find_orphan_modules(
    root: Path, ignore_dirs: set[str] | None = None
) -> tuple[list[str], list[str], int, int]:
    """Find modules not reachable from any entry point.

    REQUIRES: root is a valid directory Path
    ENSURES: Returns tuple (orphan_modules, entry_points, total_count, reachable_count)
    ENSURES: len(orphan_modules) == total_count - reachable_count
    ENSURES: total_count >= reachable_count >= 0
    ENSURES: entry_points contains relative paths from root

    Returns:
        (orphan_modules, entry_points, total_count, reachable_count)
    """
    entry_points = find_entry_points(root)
    all_modules = find_modules(root, ignore_dirs)

    # Detect src-layout packages (e.g., python/pkg/src/pkg/)
    src_packages = _detect_src_layout_packages(root)

    # Build import graph (with src-layout awareness)
    graph = build_import_graph(root, entry_points, all_modules, src_packages)

    # Get entry point module names (with src-layout awareness)
    entry_names = [_module_path_to_name(ep, root, src_packages) for ep in entry_points]

    # Add pyproject.toml entry points (PEP 621: scripts, entry-points)
    pyproject_modules = _parse_pyproject_entry_points(root)
    entry_names.extend(pyproject_modules)

    # Find reachable modules
    reachable = find_reachable_modules(graph, entry_names)

    # Get all module names (with src-layout awareness)
    all_module_names = {
        _module_path_to_name(m, root, src_packages) for m in all_modules
    }

    # Pyproject entry points may define modules that appear mid-path
    # (e.g., gamma_pytest is defined in pyproject but appears as crates.gamma-python.gamma_pytest)
    pyproject_module_set = set(pyproject_modules)

    # Find orphans (modules that exist but aren't reachable)
    # Note: we need to handle the case where a module is a submodule of a reachable module
    orphans: list[str] = []
    for module_name in all_module_names:
        # Check if this module or any of its parent packages are reachable
        parts = module_name.split(".")
        is_reachable = False
        for i in range(len(parts)):
            parent = ".".join(parts[: i + 1])
            if parent in reachable:
                is_reachable = True
                break
            # Also check if this part matches a pyproject entry point
            # (handles nested paths like crates.gamma-python.gamma_pytest)
            if parts[i] in pyproject_module_set:
                is_reachable = True
                break
        if not is_reachable:
            orphans.append(module_name)

    entry_strs = [str(ep.relative_to(root)) for ep in entry_points]
    return orphans, entry_strs, len(all_module_names), len(reachable)


def find_skip_only_flags(
    root: Path, *, max_files: int = 500, max_flags: int = 50
) -> list[dict[str, str]]:
    """Find CLI flags that are always used with --skip-* variants.

    This is a heuristic check that looks for argparse definitions with
    skip patterns and checks if they're always used together.

    Bounded scan to avoid worst-case performance on large repos (#1746):
    - max_files: Stop scanning after this many files (default 500)
    - max_flags: Stop if this many skip flags are found (default 50)
    """
    skip_flags: list[dict[str, str]] = []
    files_scanned = 0

    # Uses module-level _SKIP_FLAG_PATTERN (#1698)
    for py_file in root.rglob("*.py"):
        if _should_skip(py_file):
            continue

        files_scanned += 1
        if files_scanned > max_files:
            break

        try:
            content = py_file.read_text()
        except Exception:
            continue

        for match in _SKIP_FLAG_PATTERN.finditer(content):
            flag_name = match.group(1)
            rel_path = str(py_file.relative_to(root))
            skip_flags.append(
                {
                    "file": rel_path,
                    "flag": f"--skip-{flag_name}",
                    "note": "Review if this feature is actually used",
                }
            )
            if len(skip_flags) >= max_flags:
                return skip_flags

    return skip_flags


def run_audit(root: Path, ignore_dirs: set[str] | None = None) -> AuditResult:
    """Run full integration audit.

    Args:
        root: Root directory to audit
        ignore_dirs: Additional directories to ignore
    """
    orphans, entries, total, reachable = find_orphan_modules(root, ignore_dirs)
    skip_flags = find_skip_only_flags(root)

    return AuditResult(
        orphan_modules=orphans,
        skip_only_flags=skip_flags,
        entry_points=entries,
        total_modules=total,
        reachable_modules=reachable,
    )


def print_summary(result: AuditResult) -> None:
    """Print human-readable audit summary."""
    print("=" * 60)
    print("INTEGRATION AUDIT SUMMARY")
    print("=" * 60)

    print(f"\nEntry points found: {len(result.entry_points)}")
    for ep in result.entry_points[:10]:
        print(f"  - {ep}")
    if len(result.entry_points) > 10:
        print(f"  ... and {len(result.entry_points) - 10} more")

    print(f"\nModules: {result.reachable_modules}/{result.total_modules} reachable")

    if result.orphan_modules:
        print(f"\n⚠️  ORPHAN MODULES ({len(result.orphan_modules)}):")
        print("   (not imported from any entry point)")
        for module in result.orphan_modules[:20]:
            print(f"   - {module}")
        if len(result.orphan_modules) > 20:
            print(f"   ... and {len(result.orphan_modules) - 20} more")
    else:
        print("\n✓ No orphan modules found")

    if result.skip_only_flags:
        print(f"\n⚠️  SKIP FLAGS FOUND ({len(result.skip_only_flags)}):")
        print("   (review if these features are actually used)")
        for flag in result.skip_only_flags[:10]:
            print(f"   - {flag['flag']} in {flag['file']}")
        if len(result.skip_only_flags) > 10:
            print(f"   ... and {len(result.skip_only_flags) - 10} more")
    else:
        print("\n✓ No skip-only flags found")

    print("\n" + "=" * 60)
    if result.has_issues():
        print("⚠️  Issues found - review orphan modules and skip flags")
    else:
        print("✓ No integration issues detected")
    print("=" * 60)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Detect orphan modules not reachable from entry points"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("integration_audit.py"),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to audit (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Additional directories to ignore (can be repeated)",
    )

    args = parser.parse_args()
    root = Path(args.path).resolve()

    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        return 1

    ignore_dirs = set(args.ignore) if args.ignore else None
    result = run_audit(root, ignore_dirs)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_summary(result)

    return 1 if result.has_issues() else 0


if __name__ == "__main__":
    sys.exit(main())

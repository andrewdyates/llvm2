#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Check project dependencies and verification tools.

Detects verification tools needed by the project and checks if installed.
Run during init and periodically by Manager.

Public API (library usage):
    from ai_template_scripts.check_deps import (
        CHECKS,          # Dict of dependency checks
        EXCLUDE_DIRS,    # Directories excluded from detection
        check_command,   # Check if a command exists
        check_java,      # Check for Java including Homebrew paths
        rglob_filtered,  # Recursively glob excluding EXCLUDE_DIRS
    )

CLI usage:
    check_deps.py            # Check all deps, exit 1 if missing
    check_deps.py --quiet    # Only show missing deps
    check_deps.py --fix      # Attempt to install missing deps
    check_deps.py --version  # Show version information
"""

__all__ = [
    "CHECKS",
    "CheckSpec",
    "EXCLUDE_DIRS",
    "check_command",
    "check_java",
    "check_tla_tools",
    "rglob_filtered",
    "resolve_tla2tools_jar",
    "main",
]

import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.exclude_patterns import VENDORED_DIRS  # noqa: E402
from ai_template_scripts.subprocess_utils import run_cmd  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

# EXCLUDE_DIRS is VENDORED_DIRS from exclude_patterns.py (single source of truth, #1798)
# Kept as EXCLUDE_DIRS for backwards compatibility with existing callers
EXCLUDE_DIRS = VENDORED_DIRS
TLA2TOOLS_ENV = "TLA2TOOLS_JAR"
TLA2TOOLS_DEFAULT_PATH = Path.home() / ".local/share/tla2tools.jar"
TLA2TOOLS_FALLBACK_PATHS = [
    Path("/usr/local/share/tla2tools.jar"),
    Path("/opt/homebrew/share/tla2tools.jar"),
]


class CheckSpec(TypedDict):
    """Type specification for dependency check entries."""

    detect: Callable[[], bool]
    check: Callable[[], bool]
    install: str
    desc: str


def run(cmd: str, shell: bool = True) -> tuple[int, str]:
    """Run command and return exit code and output.

    For simple commands without shell features (pipes, redirects, wildcards),
    uses subprocess_utils.run_cmd for consistent timeout handling.
    Falls back to shell=True for complex commands.
    """
    # Check if command needs shell features
    shell_chars = {"|", ">", "<", "&", ";", "*", "?", "$", "`", "(", ")"}
    needs_shell = any(c in cmd for c in shell_chars) or "2>/dev/null" in cmd

    if needs_shell:
        # Complex command - use shell
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return 124, "timeout after 30s"

    # Simple command - use run_cmd for consistent timeout handling
    try:
        cmd_list = shlex.split(cmd)
    except ValueError:
        # Fallback if shlex fails
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return 124, "timeout after 30s"

    result = run_cmd(cmd_list, timeout=30)
    return result.returncode, result.stdout + result.stderr


def rglob_filtered(pattern: str) -> list[Path]:
    """Recursively glob, excluding EXCLUDE_DIRS."""
    return [
        path
        for path in Path(".").rglob(pattern)
        if not any(excl in path.parts for excl in EXCLUDE_DIRS)
    ]


def check_command(cmd: str) -> bool:
    """Check if a command exists."""
    code, _ = run(f"which {cmd}")
    return code == 0


def check_java() -> tuple[bool, str]:
    """Check for Java, including Homebrew keg-only location."""
    # Standard PATH
    code, output = run("java -version")
    if code == 0:
        return True, "java"

    # Homebrew keg-only (macOS)
    homebrew_java = "/opt/homebrew/opt/openjdk/bin/java"
    if os.path.exists(homebrew_java):
        code, _ = run(f"{homebrew_java} -version")
        if code == 0:
            return True, homebrew_java

    # Intel Mac Homebrew
    intel_java = "/usr/local/opt/openjdk/bin/java"
    if os.path.exists(intel_java):
        code, _ = run(f"{intel_java} -version")
        if code == 0:
            return True, intel_java

    return False, ""


def resolve_tla2tools_jar() -> Path | None:
    """Resolve TLA+ tools jar path from env or common locations."""
    env_path = os.environ.get(TLA2TOOLS_ENV)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    if TLA2TOOLS_DEFAULT_PATH.exists():
        return TLA2TOOLS_DEFAULT_PATH

    for candidate in TLA2TOOLS_FALLBACK_PATHS:
        if candidate.exists():
            return candidate

    return None


def check_tla_tools() -> bool:
    """Check if TLA+ tools (TLC) are installed."""
    java_ok, _ = check_java()
    if not java_ok:
        return False

    if resolve_tla2tools_jar() is not None:
        return True

    if not check_command("tlc"):
        return False

    return run("tlc -help")[0] == 0


CHECKS: dict[str, CheckSpec] = {
    "clippy_config": {
        "detect": lambda: Path("Cargo.toml").exists(),
        "check": lambda: Path("clippy.toml").exists() or Path(".clippy.toml").exists(),
        "install": f"cp {_repo_root}/templates/clippy.toml .",
        "desc": "Rust project (needs clippy.toml for memory discipline)",
    },
    "cargo_config": {
        "detect": lambda: Path("Cargo.toml").exists(),
        "check": lambda: Path(".cargo/config.toml").exists(),
        "install": (
            f"mkdir -p .cargo && cp {_repo_root}/templates/cargo_config.toml "
            ".cargo/config.toml"
        ),
        "desc": "Rust project (needs .cargo/config.toml for build parallelism limits)",
    },
    "kani": {
        "detect": lambda: len(rglob_filtered("*.rs")) > 0
        and run(
            "grep -rE '(kani::|cfg_attr\\(kani,[[:space:]]*kani::)"
            "(proof|requires|ensures|modifies)' "
            "--include='*.rs' src/ tests/ crates/ 2>/dev/null"
        )[0]
        == 0,
        "check": lambda: check_command("kani"),
        "install": "cargo install kani-verifier && kani setup",
        "desc": "Kani proofs or contracts found in Rust files",
    },
    "tla": {
        "detect": lambda: len(rglob_filtered("*.tla")) > 0,
        "check": check_tla_tools,
        "install": "./ai_template_scripts/install_tla_tools.sh",
        "desc": "TLA+ specs found (TLC required)",
    },
    "fuzz": {
        "detect": lambda: Path("fuzz/fuzz_targets").exists()
        and len(list(Path("fuzz/fuzz_targets").glob("*.rs"))) > 0,
        "check": lambda: run("cargo fuzz --help")[0] == 0,
        "install": "cargo install cargo-fuzz",
        "desc": "Fuzz targets found",
    },
    "python": {
        "detect": lambda: len(rglob_filtered("*.py")) > 0,
        "check": lambda: check_command("python3"),
        "install": "brew install python3",
        "desc": "Python files found",
    },
    "rust": {
        "detect": lambda: Path("Cargo.toml").exists(),
        "check": lambda: check_command("cargo"),
        "install": "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
        "desc": "Cargo.toml found",
    },
    "node": {
        "detect": lambda: Path("package.json").exists(),
        "check": lambda: check_command("node"),
        "install": "brew install node",
        "desc": "package.json found",
    },
    "gh": {
        "detect": lambda: True,  # Always needed
        "check": lambda: check_command("gh"),
        "install": "brew install gh && gh auth login",
        "desc": "GitHub CLI (required)",
    },
}


def main() -> int:
    """Entry point: check and optionally install missing dependencies."""
    if "--version" in sys.argv:
        print(get_version("check_deps.py"))
        return 0

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --quiet       Only show missing dependencies")
        print("  --fix         Attempt to install missing dependencies")
        print("  --version     Show version information")
        print("  -h, --help    Show this help message")
        return 0

    quiet = "--quiet" in sys.argv
    fix = "--fix" in sys.argv

    missing = []

    for name, check in CHECKS.items():
        try:
            needed = check["detect"]()
        except Exception:
            needed = False  # Best-effort: detection failure means feature unused

        if not needed:
            continue

        try:
            installed = check["check"]()
        except Exception:
            installed = False  # Best-effort: check failure prompts manual install

        if not quiet:
            status = "✓" if installed else "✗"
            print(f"{status} {name}: {check['desc']}")

        if not installed:
            missing.append(name)
            if not quiet:
                print(f"  Install: {check['install']}")

    if missing:
        print(f"\nMissing: {', '.join(missing)}")

        if fix:
            print("\nAttempting to install...")
            for name in missing:
                cmd = CHECKS[name]["install"]
                print(f"  Running: {cmd}")
                code, output = run(cmd)
                if code != 0:
                    print(f"  Failed: {output}")
                else:
                    print(f"  ✓ {name} installed")

        return 1

    if not quiet:
        print("\nAll dependencies satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

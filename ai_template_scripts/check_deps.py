#!/usr/bin/env python3
"""Check project dependencies and verification tools.

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
License: Apache-2.0

Detects verification tools needed by the project and checks if installed.
Run during init and periodically by Manager.

Usage:
    check_deps.py          # Check all deps, exit 1 if missing
    check_deps.py --quiet  # Only show missing deps
    check_deps.py --fix    # Attempt to install missing deps
"""

import os
import subprocess
import sys
from pathlib import Path

# Directories to exclude from detection (cloned reference code, vendored deps)
EXCLUDE_DIRS = {"reference", "vendor", "third_party", "external"}


def run(cmd: str, shell: bool = True) -> tuple[int, str]:
    """Run command and return exit code and output."""
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
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


CHECKS = {
    "clippy_config": {
        "detect": lambda: Path("Cargo.toml").exists(),
        "check": lambda: Path("clippy.toml").exists() or Path(".clippy.toml").exists(),
        "install": "cp ~/ai_template/templates/clippy.toml .",
        "desc": "Rust project (needs clippy.toml for memory discipline)",
    },
    "cargo_config": {
        "detect": lambda: Path("Cargo.toml").exists(),
        "check": lambda: Path(".cargo/config.toml").exists(),
        "install": "mkdir -p .cargo && cp ~/ai_template/templates/cargo_config.toml .cargo/config.toml",
        "desc": "Rust project (needs .cargo/config.toml for build parallelism limits)",
    },
    "kani": {
        "detect": lambda: len(rglob_filtered("*.rs")) > 0
        and run(
            "grep -rE 'kani::(proof|requires|ensures|modifies)' --include='*.rs' src/ tests/ crates/ 2>/dev/null"
        )[0]
        == 0,
        "check": lambda: check_command("kani"),
        "install": "cargo install kani-verifier && kani setup",
        "desc": "Kani proofs or contracts found in Rust files",
    },
    "tla": {
        "detect": lambda: len(list(Path(".").glob("tla/*.tla"))) > 0
        or len(list(Path(".").glob("*.tla"))) > 0,
        "check": lambda: check_java()[0],
        "install": "brew install openjdk",
        "desc": "TLA+ specs found",
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


def main():
    quiet = "--quiet" in sys.argv
    fix = "--fix" in sys.argv

    missing = []

    for name, check in CHECKS.items():
        try:
            needed = check["detect"]()
        except Exception:
            needed = False

        if not needed:
            continue

        try:
            installed = check["check"]()
        except Exception:
            installed = False

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

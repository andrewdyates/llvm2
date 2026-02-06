#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Kani tiered proof runner - enumerate, tier, and run Kani proofs.

Consolidated from z4/lean5 implementations. Part of #2538.

Features:
- Enumerate all #[kani::proof] harnesses across crates
- Tier proofs by historical runtime (A=fast, B=medium, C=slow, D=intractable)
- Run proofs with per-tier timeouts
- Output JSON summary for tracking/CI integration
- Progress output every 60s (looper silence timeout compliance)
- Config-driven tier assignments via kani_tiers.toml

Tiers:
- Tier A (fast): <30s expected, run in bulk
- Tier B (medium): 30s-5m expected, run serially with moderate timeout
- Tier C (slow): 5m-30m expected, dedicated resource budget
- Tier D (intractable): >30m or known stuck, requires redesign

Usage:
    python3 -m ai_template_scripts.kani_tiered_runner --enumerate  # List all proofs
    python3 -m ai_template_scripts.kani_tiered_runner --tier-a     # Run fast proofs
    python3 -m ai_template_scripts.kani_tiered_runner --tier-b     # Run medium proofs
    python3 -m ai_template_scripts.kani_tiered_runner --all        # Run all tiers
    python3 -m ai_template_scripts.kani_tiered_runner --status     # Show status

Config file (kani_tiers.toml):
    [tiers]
    timeout_a = 30     # seconds
    timeout_b = 300
    timeout_c = 1800
    timeout_d = 0      # 0 = skip

    [intractable]
    proofs = [
        "crate::module::proof_name",
    ]

    [assignments]
    # Override default tier for specific proofs
    "crate::module::proof_name" = "C"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

# Default tier timeouts in seconds
DEFAULT_TIER_TIMEOUTS = {
    "A": 30,      # Fast proofs
    "B": 300,     # Medium proofs (5 min)
    "C": 1800,    # Slow proofs (30 min)
    "D": 0,       # Intractable (skip by default)
}

# Default tier assignment for unknown proofs
DEFAULT_TIER = "B"

# Config file name (searched in repo root, then proofs/)
CONFIG_FILE_NAME = "kani_tiers.toml"

# Status file location (relative to repo root)
STATUS_FILE = Path("proofs/kani_status.json")

# Alternate status file (for repos without proofs/ directory)
ALT_STATUS_FILE = Path("kani_status.json")


@dataclass
class KaniProof:
    """Represents a Kani proof harness."""
    crate: str
    module: str
    name: str
    file: str
    line: int
    tier: str = DEFAULT_TIER
    last_result: str | None = None  # "pass", "fail", "timeout", "skip"
    last_runtime_ms: int | None = None
    last_run: str | None = None  # ISO timestamp

    @property
    def full_name(self) -> str:
        """Full qualified name for the proof."""
        return f"{self.crate}::{self.module}::{self.name}"

    def to_dict(self) -> dict:
        return {
            "crate": self.crate,
            "module": self.module,
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "tier": self.tier,
            "last_result": self.last_result,
            "last_runtime_ms": self.last_runtime_ms,
            "last_run": self.last_run,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KaniProof:
        return cls(
            crate=d["crate"],
            module=d["module"],
            name=d["name"],
            file=d["file"],
            line=d["line"],
            tier=d.get("tier", DEFAULT_TIER),
            last_result=d.get("last_result"),
            last_runtime_ms=d.get("last_runtime_ms"),
            last_run=d.get("last_run"),
        )


@dataclass
class TieredConfig:
    """Configuration for tiered proof runner."""
    tier_timeouts: dict[str, int]
    intractable: set[str]
    assignments: dict[str, str]

    @classmethod
    def default(cls) -> TieredConfig:
        return cls(
            tier_timeouts=dict(DEFAULT_TIER_TIMEOUTS),
            intractable=set(),
            assignments={},
        )

    @classmethod
    def from_toml(cls, path: Path) -> TieredConfig:
        """Load configuration from TOML file."""
        if tomllib is None:
            print(f"Warning: tomllib not available, using defaults", file=sys.stderr)
            return cls.default()

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (IOError, tomllib.TOMLDecodeError) as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
            return cls.default()

        config = cls.default()

        # Load tier timeouts
        if "tiers" in data:
            tiers = data["tiers"]
            if "timeout_a" in tiers:
                config.tier_timeouts["A"] = tiers["timeout_a"]
            if "timeout_b" in tiers:
                config.tier_timeouts["B"] = tiers["timeout_b"]
            if "timeout_c" in tiers:
                config.tier_timeouts["C"] = tiers["timeout_c"]
            if "timeout_d" in tiers:
                config.tier_timeouts["D"] = tiers["timeout_d"]

        # Load intractable list
        if "intractable" in data and "proofs" in data["intractable"]:
            config.intractable = set(data["intractable"]["proofs"])

        # Load tier assignments
        if "assignments" in data:
            config.assignments = dict(data["assignments"])

        return config


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # Try to find repo root by looking for Cargo.toml with workspace
    current = Path.cwd()
    while current != current.parent:
        if (current / "Cargo.toml").exists():
            return current
        if (current / ".git").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


def find_config_file(repo_root: Path) -> Path | None:
    """Find the config file in standard locations."""
    # Check repo root first
    root_config = repo_root / CONFIG_FILE_NAME
    if root_config.exists():
        return root_config

    # Check proofs/ directory
    proofs_config = repo_root / "proofs" / CONFIG_FILE_NAME
    if proofs_config.exists():
        return proofs_config

    return None


def get_status_file(repo_root: Path) -> Path:
    """Get absolute path to status file."""
    # Prefer proofs/ directory if it exists
    proofs_dir = repo_root / "proofs"
    if proofs_dir.exists():
        return repo_root / STATUS_FILE
    return repo_root / ALT_STATUS_FILE


def get_crate_name_from_path(crates_dir: Path, file_path: str) -> str:
    """Get the actual crate name from the file path.

    Walks up the directory tree to find Cargo.toml and reads package name.
    """
    path = Path(file_path)
    # Walk up to find Cargo.toml
    for parent in path.parents:
        cargo_toml = parent / "Cargo.toml"
        if cargo_toml.exists():
            try:
                with open(cargo_toml) as f:
                    for line in f:
                        if line.startswith("name ="):
                            match = re.search(r'name\s*=\s*"([^"]+)"', line)
                            if match:
                                return match.group(1)
            except IOError:
                pass
            break
    # Fallback: use directory structure
    try:
        rel = path.relative_to(crates_dir)
        return str(rel.parts[0])
    except ValueError:
        return "unknown"


def enumerate_proofs(
    repo_root: Path,
    config: TieredConfig,
) -> list[KaniProof]:
    """Enumerate all Kani proofs in the repository."""
    proofs = []

    # Find crates directory (common patterns)
    crates_dir = None
    for candidate in ["crates", "src", "."]:
        candidate_path = repo_root / candidate
        if candidate_path.exists():
            crates_dir = candidate_path
            break
    if crates_dir is None:
        crates_dir = repo_root

    # Use grep to find all #[kani::proof] annotations
    result = subprocess.run(
        ["grep", "-rn", r"#\[kani::proof\]", str(crates_dir), "--include=*.rs"],
        capture_output=True,
        text=True,
    )

    if result.returncode not in (0, 1):  # 1 = no matches
        print(f"Error running grep: {result.stderr}", file=sys.stderr)
        return proofs

    # Parse grep output
    proof_pattern = re.compile(r"fn\s+(\w+)\s*\(")

    for match_line in result.stdout.strip().split("\n"):
        if not match_line:
            continue

        parts = match_line.split(":", 2)
        if len(parts) < 2:
            continue

        file_path = parts[0]
        line_num = int(parts[1])

        # Read the next few lines to get the function name
        try:
            with open(file_path) as f:
                lines = f.readlines()
                # Look for fn declaration in next 5 lines
                for i in range(line_num - 1, min(line_num + 5, len(lines))):
                    fn_match = proof_pattern.search(lines[i])
                    if fn_match:
                        fn_name = fn_match.group(1)
                        break
                else:
                    fn_name = f"unknown_proof_line_{line_num}"
        except (IOError, IndexError):
            fn_name = f"unknown_proof_line_{line_num}"

        # Get actual crate name from Cargo.toml
        crate = get_crate_name_from_path(crates_dir, file_path)

        # Get module from file path
        rel_path = Path(file_path)
        module = rel_path.stem

        proof = KaniProof(
            crate=crate,
            module=module,
            name=fn_name,
            file=file_path,
            line=line_num,
        )

        # Apply tier from config
        if proof.full_name in config.intractable:
            proof.tier = "D"
        elif proof.full_name in config.assignments:
            proof.tier = config.assignments[proof.full_name]

        proofs.append(proof)

    return proofs


def load_status(repo_root: Path) -> dict:
    """Load status from last run."""
    status_file = get_status_file(repo_root)
    if status_file.exists():
        try:
            with open(status_file) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
    return {"proofs": [], "last_run": None, "summary": {}}


def save_status(repo_root: Path, proofs: list[KaniProof], summary: dict):
    """Save status to file."""
    status_file = get_status_file(repo_root)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "proofs": [p.to_dict() for p in proofs],
        "last_run": datetime.now().isoformat(),
        "summary": summary,
    }
    with open(status_file, "w") as f:
        json.dump(data, f, indent=2)


def run_proof(
    proof: KaniProof,
    timeout: int,
    repo_root: Path,
) -> tuple[str, int]:
    """Run a single Kani proof.

    Returns:
        (result, runtime_ms) where result is "pass", "fail", "timeout", or "error"
    """
    if timeout == 0:
        return "skip", 0

    start = time.time()

    try:
        result = subprocess.run(
            [
                "cargo", "kani",
                "-p", proof.crate,
                "--harness", proof.name,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
        )

        elapsed_ms = int((time.time() - start) * 1000)

        if result.returncode == 0:
            return "pass", elapsed_ms
        else:
            return "fail", elapsed_ms

    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.time() - start) * 1000)
        return "timeout", elapsed_ms
    except (subprocess.SubprocessError, OSError) as e:
        elapsed_ms = int((time.time() - start) * 1000)
        print(f"Error running {proof.full_name}: {e}", file=sys.stderr)
        return "error", elapsed_ms


def run_tier(
    proofs: list[KaniProof],
    tier: str,
    config: TieredConfig,
    repo_root: Path,
    verbose: bool = False,
) -> dict:
    """Run all proofs in a specific tier.

    Returns summary dict with counts.
    """
    tier_proofs = [p for p in proofs if p.tier == tier]
    timeout = config.tier_timeouts.get(tier, DEFAULT_TIER_TIMEOUTS[DEFAULT_TIER])

    summary = {
        "total": len(tier_proofs),
        "pass": 0,
        "fail": 0,
        "timeout": 0,
        "skip": 0,
        "error": 0,
    }
    last_progress = time.time()

    print(f"\n=== Running Tier {tier} ({len(tier_proofs)} proofs, {timeout}s timeout) ===")

    for i, proof in enumerate(tier_proofs):
        # Progress output every 60s (looper silence timeout compliance)
        now = time.time()
        if now - last_progress >= 60:
            print(f"[Progress] Tier {tier}: {i}/{len(tier_proofs)} complete", file=sys.stderr)
            last_progress = now

        if verbose:
            print(f"  Running {proof.full_name}...", end=" ", flush=True)

        result, runtime_ms = run_proof(proof, timeout, repo_root)

        proof.last_result = result
        proof.last_runtime_ms = runtime_ms
        proof.last_run = datetime.now().isoformat()
        summary[result] += 1

        if verbose:
            print(f"{result} ({runtime_ms}ms)")

    print(f"\nTier {tier} summary: {summary['pass']} pass, {summary['fail']} fail, "
          f"{summary['timeout']} timeout, {summary['skip']} skip, {summary['error']} error")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Kani tiered proof runner (ai_template consolidated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--enumerate", action="store_true",
                       help="Enumerate all proofs without running")
    parser.add_argument("--tier-a", action="store_true",
                       help="Run tier A (fast) proofs only")
    parser.add_argument("--tier-b", action="store_true",
                       help="Run tier B (medium) proofs only")
    parser.add_argument("--tier-c", action="store_true",
                       help="Run tier C (slow) proofs only")
    parser.add_argument("--tier-d", action="store_true",
                       help="Run tier D (intractable) proofs (use with caution)")
    parser.add_argument("--all", action="store_true",
                       help="Run all tiers (A, B, C - not D)")
    parser.add_argument("--status", action="store_true",
                       help="Show status from last run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    parser.add_argument("--config", type=Path,
                       help="Path to config file (default: auto-discover)")
    parser.add_argument("--repo-root", type=Path,
                       help="Repository root (default: auto-discover)")

    args = parser.parse_args()

    # Find repo root
    repo_root = args.repo_root or get_repo_root()

    # Load config
    config_path = args.config or find_config_file(repo_root)
    if config_path and config_path.exists():
        config = TieredConfig.from_toml(config_path)
        print(f"Loaded config from {config_path}", file=sys.stderr)
    else:
        config = TieredConfig.default()

    if args.status:
        status = load_status(repo_root)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Last run: {status.get('last_run') or 'never'}")
            print(f"Total proofs: {len(status.get('proofs', []))}")
            if status.get("summary"):
                for tier, s in status["summary"].items():
                    print(f"  Tier {tier}: {s}")
        return

    # Enumerate proofs
    proofs = enumerate_proofs(repo_root, config)

    if args.enumerate:
        # Group by tier
        by_tier: dict[str, list[KaniProof]] = {}
        for proof in proofs:
            by_tier.setdefault(proof.tier, []).append(proof)

        if args.json:
            print(json.dumps([p.to_dict() for p in proofs], indent=2))
        else:
            print(f"Total proofs: {len(proofs)}")
            for tier in sorted(by_tier.keys()):
                print(f"\nTier {tier} ({len(by_tier[tier])} proofs):")
                for proof in by_tier[tier][:10]:  # Show first 10
                    print(f"  {proof.full_name} ({proof.file}:{proof.line})")
                if len(by_tier[tier]) > 10:
                    print(f"  ... and {len(by_tier[tier]) - 10} more")
        return

    # Determine which tiers to run
    tiers_to_run = []
    if args.tier_a:
        tiers_to_run.append("A")
    if args.tier_b:
        tiers_to_run.append("B")
    if args.tier_c:
        tiers_to_run.append("C")
    if args.tier_d:
        tiers_to_run.append("D")
    if args.all:
        tiers_to_run = ["A", "B", "C"]  # Not D by default

    if not tiers_to_run:
        parser.print_help()
        return

    # Run selected tiers
    all_summary: dict[str, dict] = {}
    for tier in tiers_to_run:
        summary = run_tier(proofs, tier, config, repo_root, verbose=args.verbose)
        all_summary[tier] = summary

    # Save status
    save_status(repo_root, proofs, all_summary)

    # Final summary
    print("\n=== Final Summary ===")
    total_pass = sum(s["pass"] for s in all_summary.values())
    total_fail = sum(s["fail"] for s in all_summary.values())
    total_timeout = sum(s["timeout"] for s in all_summary.values())
    print(f"Total: {total_pass} pass, {total_fail} fail, {total_timeout} timeout")

    # Exit code
    if total_fail > 0 or total_timeout > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

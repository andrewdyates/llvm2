#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Kani proof runner with timeout management and status tracking.

Runs Kani proofs with configurable timeouts, tracks results to kani_status.json,
and integrates with the looper 60s progress requirement.

Part of #2263: Kani proofs written but not executed - need tracking and enforcement

Usage:
    # Run all proofs with 5-minute default timeout
    python3 -m ai_template_scripts.kani_runner --timeout 300

    # Run specific harness with custom timeout
    python3 -m ai_template_scripts.kani_runner --harness proof_foo --timeout 600

    # Run only not_run proofs
    python3 -m ai_template_scripts.kani_runner --filter not_run

    # Audit: compare tracking file to actual harnesses
    python3 -m ai_template_scripts.kani_runner --audit

    # Dry run: discover harnesses without executing
    python3 -m ai_template_scripts.kani_runner --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Default timeout for individual harnesses (5 minutes)
DEFAULT_TIMEOUT_SEC = 300

# Progress output interval (looper requires output every 60s)
PROGRESS_INTERVAL_SEC = 60

# Status file location
STATUS_FILE = "kani_status.json"

# OOM exit codes (SIGKILL from OOM killer)
OOM_EXIT_CODES = {137, 139}

# Timeout exit code (from subprocess.TimeoutExpired)
TIMEOUT_EXIT_CODE = 124


@dataclass
class HarnessResult:
    """Result of running a single Kani harness."""

    status: str  # passed, failed, timeout, oom, not_run, error
    duration_sec: float | None = None
    verified_at: str | None = None
    commit: str | None = None
    error_message: str | None = None
    last_attempt: str | None = None
    timeout_sec: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result: dict = {"status": self.status}
        if self.duration_sec is not None:
            result["duration_sec"] = round(self.duration_sec, 1)
        if self.verified_at is not None:
            result["verified_at"] = self.verified_at
        if self.commit is not None:
            result["commit"] = self.commit
        if self.error_message is not None:
            result["error_message"] = self.error_message
        if self.last_attempt is not None:
            result["last_attempt"] = self.last_attempt
        if self.timeout_sec is not None:
            result["timeout_sec"] = self.timeout_sec
        return result


@dataclass
class KaniStatus:
    """Kani verification status for a repository."""

    harnesses: dict[str, HarnessResult] = field(default_factory=dict)
    last_updated: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Calculate summary
        summary = {
            "passed": 0,
            "failed": 0,
            "timeout": 0,
            "oom": 0,
            "not_run": 0,
            "error": 0,
        }
        for result in self.harnesses.values():
            if result.status in summary:
                summary[result.status] += 1

        return {
            "harnesses": {name: r.to_dict() for name, r in self.harnesses.items()},
            "summary": summary,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KaniStatus:
        """Load from dictionary."""
        status = cls()
        status.last_updated = data.get("last_updated", "")
        for name, harness_data in data.get("harnesses", {}).items():
            status.harnesses[name] = HarnessResult(
                status=harness_data.get("status", "not_run"),
                duration_sec=harness_data.get("duration_sec"),
                verified_at=harness_data.get("verified_at"),
                commit=harness_data.get("commit"),
                error_message=harness_data.get("error_message"),
                last_attempt=harness_data.get("last_attempt"),
                timeout_sec=harness_data.get("timeout_sec"),
            )
        return status


def get_current_commit() -> str | None:
    """Get current git commit hash.

    REQUIRES: git is available in PATH (or returns None)
    ENSURES: returns short commit hash string or None if unavailable
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def discover_harnesses(cwd: Path | None = None) -> list[str]:
    """Discover all Kani proof harnesses in the repository.

    Looks for #[kani::proof] and cfg_attr(kani, kani::proof) patterns.

    REQUIRES: cwd is None (current dir) or a valid Path to a Rust project
    ENSURES: returns sorted list of harness names in format "crate::harness_name"

    Returns:
        List of harness names in format "crate::harness_name".
    """
    root = cwd or Path(".")
    harnesses: list[str] = []

    # Pattern to match kani proof attributes and extract function name
    # Handles both #[kani::proof] and #[cfg_attr(kani, kani::proof)]
    proof_pattern = re.compile(
        r"#\[(kani::|cfg_attr\(kani,\s*kani::)proof\]"
        r"[\s\S]*?"  # Non-greedy match until fn
        r"(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.MULTILINE,
    )

    # Search in standard Rust source locations
    search_patterns = [
        root / "src" / "**" / "*.rs",
        root / "tests" / "**" / "*.rs",
        root / "crates" / "**" / "*.rs",
    ]

    seen_files: set[Path] = set()
    for pattern in search_patterns:
        for rust_file in root.glob(str(pattern.relative_to(root))):
            if rust_file in seen_files:
                continue
            seen_files.add(rust_file)

            try:
                content = rust_file.read_text(encoding="utf-8", errors="ignore")
                for match in proof_pattern.finditer(content):
                    fn_name = match.group(2)
                    # Derive crate name from path
                    rel_path = rust_file.relative_to(root)
                    parts = rel_path.parts
                    if "crates" in parts:
                        crate_idx = parts.index("crates")
                        if crate_idx + 1 < len(parts):
                            crate_name = parts[crate_idx + 1]
                        else:
                            crate_name = "unknown"
                    elif "src" in parts or "tests" in parts:
                        # Root crate - try to get name from Cargo.toml
                        crate_name = _get_crate_name(root)
                    else:
                        crate_name = "unknown"
                    harness_id = f"{crate_name}::{fn_name}"
                    if harness_id not in harnesses:
                        harnesses.append(harness_id)
            except OSError:
                continue

    return sorted(harnesses)


def _get_crate_name(root: Path) -> str:
    """Get crate name from Cargo.toml."""
    cargo_toml = root / "Cargo.toml"
    if cargo_toml.exists():
        try:
            import tomllib

            content = cargo_toml.read_text()
            data = tomllib.loads(content)
            if "package" in data and "name" in data["package"]:
                return data["package"]["name"]
        except (OSError, tomllib.TOMLDecodeError, KeyError):
            pass
    return "unknown"


def load_status(status_file: Path) -> KaniStatus:
    """Load status from file or return empty status.

    REQUIRES: status_file is a valid Path (file may or may not exist)
    ENSURES: returns KaniStatus (empty if file missing or invalid)
    """
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text())
            return KaniStatus.from_dict(data)
        except (json.JSONDecodeError, OSError):
            pass
    return KaniStatus()


def save_status(status: KaniStatus, status_file: Path) -> None:
    """Save status to file.

    REQUIRES: status is a valid KaniStatus, status_file is a writable Path
    ENSURES: status_file contains JSON with updated last_updated timestamp
    """
    status.last_updated = datetime.now().isoformat()
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(status.to_dict(), indent=2) + "\n")


def run_harness(
    harness_name: str,
    timeout_sec: int,
    cwd: Path | None = None,
) -> HarnessResult:
    """Run a single Kani harness with timeout.

    REQUIRES: harness_name is non-empty, timeout_sec > 0, cargo kani available
    ENSURES: returns HarnessResult with status in {passed, failed, timeout, oom, error}

    Args:
        harness_name: Harness name in format "crate::harness_name"
        timeout_sec: Maximum seconds to allow
        cwd: Working directory

    Returns:
        HarnessResult with status and timing information
    """
    root = cwd or Path(".")
    start_time = time.monotonic()
    timestamp = datetime.now().isoformat()
    commit = get_current_commit()

    # Parse harness name to get crate and function
    if "::" in harness_name:
        crate_name, fn_name = harness_name.rsplit("::", 1)
    else:
        crate_name = None
        fn_name = harness_name

    # Build cargo kani command
    cmd = ["cargo", "kani", "--harness", fn_name]
    if crate_name and crate_name != "unknown":
        cmd.extend(["-p", crate_name])

    # Progress reporter for long runs
    last_progress = [start_time]

    def progress_check() -> None:
        """Print progress if PROGRESS_INTERVAL_SEC has elapsed."""
        now = time.monotonic()
        if now - last_progress[0] >= PROGRESS_INTERVAL_SEC:
            elapsed = now - start_time
            remaining = timeout_sec - elapsed
            print(
                f"  kani_runner: {harness_name} running... "
                f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining",
                file=sys.stderr,
                flush=True,
            )
            last_progress[0] = now

    try:
        # Run with timeout
        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines: list[str] = []
        deadline = time.monotonic() + timeout_sec

        while True:
            progress_check()

            # Check if process has finished
            try:
                remaining = max(0.1, deadline - time.monotonic())
                # Poll for output with short timeout
                if process.stdout:
                    process.stdout.flush()
                    # Use select on Unix, or just poll on all platforms
                    import select

                    readable, _, _ = select.select([process.stdout], [], [], 1.0)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
            except (OSError, ValueError):
                # select may not work on some file descriptors
                time.sleep(1.0)

            # Check if process is done
            retcode = process.poll()
            if retcode is not None:
                # Read any remaining output
                if process.stdout:
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        output_lines.append(remaining_output)
                break

            # Check timeout
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                duration = time.monotonic() - start_time
                return HarnessResult(
                    status="timeout",
                    duration_sec=duration,
                    timeout_sec=timeout_sec,
                    last_attempt=timestamp,
                    commit=commit,
                    error_message=f"Timeout after {timeout_sec}s",
                )

        duration = time.monotonic() - start_time

        # Check exit code
        if retcode == 0:
            return HarnessResult(
                status="passed",
                duration_sec=duration,
                verified_at=timestamp,
                commit=commit,
            )
        elif retcode in OOM_EXIT_CODES:
            return HarnessResult(
                status="oom",
                duration_sec=duration,
                last_attempt=timestamp,
                commit=commit,
                error_message=f"Out of memory (exit code {retcode})",
            )
        else:
            # Extract error from output
            output = "".join(output_lines)
            error_msg = _extract_error(output) or f"Exit code {retcode}"
            return HarnessResult(
                status="failed",
                duration_sec=duration,
                last_attempt=timestamp,
                commit=commit,
                error_message=error_msg[:200],  # Truncate
            )

    except FileNotFoundError:
        return HarnessResult(
            status="error",
            last_attempt=timestamp,
            error_message="cargo kani not found",
        )
    except Exception as e:
        return HarnessResult(
            status="error",
            last_attempt=timestamp,
            error_message=str(e)[:200],
        )


def _extract_error(output: str) -> str | None:
    """Extract meaningful error message from Kani output."""
    # Look for verification failure lines
    for line in output.split("\n"):
        if "VERIFICATION FAILED" in line:
            return "Verification failed"
        if "error[" in line.lower():
            return line.strip()[:200]
        if "CBMC" in line and "error" in line.lower():
            return line.strip()[:200]
    return None


def audit_harnesses(status: KaniStatus, discovered: list[str]) -> dict:
    """Compare tracked harnesses against discovered ones.

    REQUIRES: status is valid KaniStatus, discovered is list of harness names
    ENSURES: returns dict with keys {missing, orphaned, matched} containing sorted lists

    Returns:
        Dict with 'missing' (in code but not tracked),
        'orphaned' (tracked but not in code), 'matched' (both).
    """
    tracked = set(status.harnesses.keys())
    found = set(discovered)

    return {
        "missing": sorted(found - tracked),
        "orphaned": sorted(tracked - found),
        "matched": sorted(found & tracked),
    }


def run_filtered_harnesses(
    status: KaniStatus,
    harnesses: list[str],
    filter_status: str | None,
    timeout_sec: int,
    cwd: Path | None = None,
) -> KaniStatus:
    """Run harnesses matching the filter.

    REQUIRES: status is valid KaniStatus, harnesses is list, timeout_sec > 0
    ENSURES: returns updated KaniStatus with results for matching harnesses

    Args:
        status: Current status
        harnesses: All discovered harnesses
        filter_status: Only run harnesses with this status (or None for all)
        timeout_sec: Timeout per harness
        cwd: Working directory

    Returns:
        Updated status
    """
    to_run: list[str] = []

    for harness in harnesses:
        current = status.harnesses.get(harness)
        if filter_status is None:
            to_run.append(harness)
        elif current is None and filter_status == "not_run":
            to_run.append(harness)
        elif current is not None and current.status == filter_status:
            to_run.append(harness)

    if not to_run:
        print(f"No harnesses match filter '{filter_status}'", file=sys.stderr)
        return status

    print(f"Running {len(to_run)} harnesses...", file=sys.stderr, flush=True)

    for i, harness in enumerate(to_run):
        print(
            f"[{i+1}/{len(to_run)}] {harness}...", file=sys.stderr, flush=True
        )
        result = run_harness(harness, timeout_sec, cwd)
        status.harnesses[harness] = result
        print(
            f"  -> {result.status}"
            + (f" ({result.duration_sec:.1f}s)" if result.duration_sec else ""),
            file=sys.stderr,
            flush=True,
        )

    return status


def main() -> int:
    """CLI entry point.

    REQUIRES: called as script or module entry point
    ENSURES: returns 0 on success, 1 on failures/errors or audit mismatches
    """
    parser = argparse.ArgumentParser(
        description="Run Kani proofs with timeout management and status tracking"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Timeout per harness in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--harness",
        type=str,
        help="Run specific harness by name",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["not_run", "passed", "failed", "timeout", "oom"],
        help="Only run harnesses with this status",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Compare tracking file to actual harnesses (no execution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover harnesses without executing",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path(STATUS_FILE),
        help=f"Status file path (default: {STATUS_FILE})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()
    cwd = Path(".")

    # Load existing status
    status = load_status(args.status_file)

    # Discover harnesses
    print("Discovering Kani harnesses...", file=sys.stderr, flush=True)
    discovered = discover_harnesses(cwd)
    print(f"Found {len(discovered)} harnesses", file=sys.stderr, flush=True)

    if args.dry_run:
        # Just list discovered harnesses
        if args.json:
            print(json.dumps({"harnesses": discovered}, indent=2))
        else:
            for h in discovered:
                current = status.harnesses.get(h)
                status_str = current.status if current else "not_run"
                print(f"  {h}: {status_str}")
        return 0

    if args.audit:
        # Compare tracking vs discovered
        audit_result = audit_harnesses(status, discovered)
        if args.json:
            print(json.dumps(audit_result, indent=2))
        else:
            print(f"\nMatched: {len(audit_result['matched'])}")
            print(f"Missing (in code, not tracked): {len(audit_result['missing'])}")
            for h in audit_result["missing"][:10]:
                print(f"  + {h}")
            if len(audit_result["missing"]) > 10:
                print(f"  ... and {len(audit_result['missing']) - 10} more")
            print(f"Orphaned (tracked, not in code): {len(audit_result['orphaned'])}")
            for h in audit_result["orphaned"][:10]:
                print(f"  - {h}")
            if len(audit_result["orphaned"]) > 10:
                print(f"  ... and {len(audit_result['orphaned']) - 10} more")
        return 0 if not audit_result["missing"] else 1

    # Run harnesses
    if args.harness:
        # Run specific harness
        if args.harness not in discovered:
            print(f"Harness '{args.harness}' not found", file=sys.stderr)
            print(f"Available: {', '.join(discovered[:5])}...", file=sys.stderr)
            return 1
        print(f"Running {args.harness}...", file=sys.stderr, flush=True)
        result = run_harness(args.harness, args.timeout, cwd)
        status.harnesses[args.harness] = result
        print(f"Result: {result.status}", file=sys.stderr, flush=True)
        if result.duration_sec:
            print(f"Duration: {result.duration_sec:.1f}s", file=sys.stderr)
        if result.error_message:
            print(f"Error: {result.error_message}", file=sys.stderr)
    else:
        # Run all (or filtered)
        status = run_filtered_harnesses(
            status, discovered, args.filter, args.timeout, cwd
        )

    # Save updated status
    save_status(status, args.status_file)
    print(f"Status saved to {args.status_file}", file=sys.stderr, flush=True)

    # Print summary
    summary = status.to_dict()["summary"]
    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        print(f"\nSummary: ", end="")
        parts = []
        for k, v in summary.items():
            if v > 0:
                parts.append(f"{v} {k}")
        print(", ".join(parts) if parts else "no results")

    # Return non-zero if any failures
    if summary.get("failed", 0) > 0 or summary.get("error", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

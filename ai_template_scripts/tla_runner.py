#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""TLA+ spec runner with timeout management and status tracking.

Runs TLA+ specs via TLC with configurable timeouts, tracks results to tla_status.json,
and integrates with the looper 60s progress requirement.

Part of #2370: Standardize verification status tracking across repos

Usage:
    # Run all specs with 5-minute default timeout
    python3 -m ai_template_scripts.tla_runner --timeout 300

    # Run specific spec with custom timeout
    python3 -m ai_template_scripts.tla_runner --spec IssueStateMachine --timeout 600

    # Run only not_run specs
    python3 -m ai_template_scripts.tla_runner --filter not_run

    # Audit: compare tracking file to actual specs
    python3 -m ai_template_scripts.tla_runner --audit

    # Dry run: discover specs without executing
    python3 -m ai_template_scripts.tla_runner --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Default timeout for individual specs (5 minutes)
DEFAULT_TIMEOUT_SEC = 300

# Progress output interval (looper requires output every 60s)
PROGRESS_INTERVAL_SEC = 60

# Status file location
STATUS_FILE = "tla_status.json"

# OOM exit codes (SIGKILL from OOM killer)
OOM_EXIT_CODES = {137, 139}

# Spec discovery directories (in order of preference)
SPEC_DIRS = ["specs", "tla"]


@dataclass
class SpecResult:
    """Result of running a single TLA+ spec."""

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
class TlaStatus:
    """TLA+ verification status for a repository."""

    specs: dict[str, SpecResult] = field(default_factory=dict)
    last_updated: str = ""
    tlc_version: str | None = None

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
        for result in self.specs.values():
            if result.status in summary:
                summary[result.status] += 1

        result: dict = {
            "specs": {name: r.to_dict() for name, r in self.specs.items()},
            "summary": summary,
            "last_updated": self.last_updated,
        }
        if self.tlc_version:
            result["tlc_version"] = self.tlc_version
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TlaStatus:
        """Load from dictionary."""
        status = cls()
        status.last_updated = data.get("last_updated", "")
        status.tlc_version = data.get("tlc_version")
        for name, spec_data in data.get("specs", {}).items():
            status.specs[name] = SpecResult(
                status=spec_data.get("status", "not_run"),
                duration_sec=spec_data.get("duration_sec"),
                verified_at=spec_data.get("verified_at"),
                commit=spec_data.get("commit"),
                error_message=spec_data.get("error_message"),
                last_attempt=spec_data.get("last_attempt"),
                timeout_sec=spec_data.get("timeout_sec"),
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


def find_tlc() -> str | None:
    """Find TLC command.

    Checks:
    1. 'tlc' in PATH
    2. 'tlc2.TLC' via java -jar (TLA+ Toolbox installation)
    3. ai_template_scripts/bin/tlc (our wrapper)

    ENSURES: returns command string or None if not found
    """
    # Check for tlc in PATH
    if shutil.which("tlc"):
        return "tlc"

    # Check for Java (required for TLC)
    if not shutil.which("java"):
        return None

    # Check for our wrapper
    wrapper = Path(__file__).parent / "bin" / "tlc"
    if wrapper.exists() and wrapper.is_file():
        return str(wrapper)

    return None


def get_tlc_version(tlc_cmd: str) -> str | None:
    """Get TLC version string.

    REQUIRES: tlc_cmd is a valid TLC command
    ENSURES: returns version string or None if unavailable
    """
    try:
        # TLC outputs version to stderr
        result = subprocess.run(
            [tlc_cmd, "-version"] if tlc_cmd != "java" else ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Parse version from output
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "TLC" in line or "version" in line.lower():
                return line.strip()[:50]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def discover_specs(cwd: Path | None = None) -> list[str]:
    """Discover all TLA+ specs in the repository.

    Looks for *.tla files in specs/ and tla/ directories.
    Excludes *_MC.tla (model config files).

    REQUIRES: cwd is None (current dir) or a valid Path
    ENSURES: returns sorted list of spec names (without .tla extension)

    Returns:
        List of spec names (e.g., "IssueStateMachine", "LockProtocol")
    """
    root = cwd or Path(".")
    specs: set[str] = set()

    for spec_dir in SPEC_DIRS:
        spec_path = root / spec_dir
        if not spec_path.exists():
            continue

        for tla_file in spec_path.glob("*.tla"):
            name = tla_file.stem
            # Exclude model config files (*_MC.tla)
            if name.endswith("_MC"):
                continue
            specs.add(name)

    return sorted(specs)


def load_status(status_file: Path) -> TlaStatus:
    """Load status from file or return empty status.

    REQUIRES: status_file is a valid Path (file may or may not exist)
    ENSURES: returns TlaStatus (empty if file missing or invalid)
    """
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text())
            return TlaStatus.from_dict(data)
        except (json.JSONDecodeError, OSError):
            pass
    return TlaStatus()


def save_status(status: TlaStatus, status_file: Path) -> None:
    """Save status to file.

    REQUIRES: status is a valid TlaStatus, status_file is a writable Path
    ENSURES: status_file contains JSON with updated last_updated timestamp
    """
    status.last_updated = datetime.now().isoformat()
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(status.to_dict(), indent=2) + "\n")


def _close_stdout(process: subprocess.Popen[str]) -> None:
    """Best-effort close for subprocess stdout."""
    if not process.stdout or process.stdout.closed:
        return
    with contextlib.suppress(OSError, ValueError):
        process.stdout.close()


def _drain_stdout(process: subprocess.Popen[str], output_lines: list[str]) -> None:
    """Read remaining stdout output then close."""
    if not process.stdout or process.stdout.closed:
        return
    try:
        remaining_output = process.stdout.read()
        if remaining_output:
            output_lines.append(remaining_output)
    except (OSError, ValueError):
        pass
    finally:
        _close_stdout(process)


def _format_timeout_error(timeout_sec: int, output_lines: list[str]) -> str:
    """Build a short timeout message with last output excerpt."""
    output = "".join(output_lines).strip()
    if not output:
        return f"Timeout after {timeout_sec}s"
    tail_lines = output.splitlines()[-5:]
    tail = " | ".join(line.strip() for line in tail_lines if line.strip())
    if not tail:
        return f"Timeout after {timeout_sec}s"
    return f"Timeout after {timeout_sec}s (last output: {tail})"[:200]


def find_spec_file(spec_name: str, cwd: Path | None = None) -> Path | None:
    """Find the .tla file for a spec by name.

    REQUIRES: spec_name is non-empty string
    ENSURES: returns Path to .tla file or None if not found
    """
    root = cwd or Path(".")
    for spec_dir in SPEC_DIRS:
        spec_file = root / spec_dir / f"{spec_name}.tla"
        if spec_file.exists():
            return spec_file
    return None


def run_spec(
    spec_name: str,
    timeout_sec: int,
    cwd: Path | None = None,
    tlc_cmd: str | None = None,
) -> SpecResult:
    """Run a single TLA+ spec with timeout.

    REQUIRES: spec_name is non-empty, timeout_sec > 0, TLC available
    ENSURES: returns SpecResult with status in {passed, failed, timeout, oom, error}

    Args:
        spec_name: Spec name (without .tla extension)
        timeout_sec: Maximum seconds to allow
        cwd: Working directory
        tlc_cmd: TLC command (discovered if None)

    Returns:
        SpecResult with status and timing information
    """
    root = cwd or Path(".")
    start_time = time.monotonic()
    timestamp = datetime.now().isoformat()
    commit = get_current_commit()

    # Find spec file
    spec_file = find_spec_file(spec_name, root)
    if not spec_file:
        return SpecResult(
            status="error",
            last_attempt=timestamp,
            error_message=f"Spec file not found: {spec_name}.tla",
        )

    # Find TLC
    tlc = tlc_cmd or find_tlc()
    if not tlc:
        return SpecResult(
            status="error",
            last_attempt=timestamp,
            error_message="TLC not found. Install with: brew install tla-plus-toolbox",
        )

    # Build TLC command
    # tlc -workers auto -cleanup spec.tla
    cmd = [tlc, "-workers", "auto", "-cleanup", str(spec_file)]

    # Progress reporter for long runs
    last_progress = [start_time]

    def progress_check() -> None:
        """Print progress if PROGRESS_INTERVAL_SEC has elapsed."""
        now = time.monotonic()
        if now - last_progress[0] >= PROGRESS_INTERVAL_SEC:
            elapsed = now - start_time
            remaining = timeout_sec - elapsed
            print(
                f"  tla_runner: {spec_name} running... "
                f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining",
                file=sys.stderr,
                flush=True,
            )
            last_progress[0] = now

    process = None
    output_lines: list[str] = []
    try:
        # Run with timeout
        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.monotonic() + timeout_sec

        while True:
            progress_check()

            # Check if process has finished
            try:
                # Poll for output with short timeout
                if process.stdout:
                    process.stdout.flush()
                    readable, _, _ = select.select([process.stdout], [], [], 1.0)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
            except (OSError, ValueError):
                time.sleep(1.0)

            # Check if process is done
            retcode = process.poll()
            if retcode is not None:
                _drain_stdout(process, output_lines)
                break

            # Check timeout
            if time.monotonic() >= deadline:
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(
                        "WARNING: TLC process did not exit after SIGKILL "
                        "(uninterruptible?), abandoning wait",
                        file=sys.stderr,
                    )
                _drain_stdout(process, output_lines)
                duration = time.monotonic() - start_time
                return SpecResult(
                    status="timeout",
                    duration_sec=duration,
                    timeout_sec=timeout_sec,
                    last_attempt=timestamp,
                    commit=commit,
                    error_message=_format_timeout_error(timeout_sec, output_lines),
                )

        duration = time.monotonic() - start_time
        output = "".join(output_lines)

        # Check exit code and output for success/failure
        if retcode == 0:
            # TLC returns 0 even on some failures, check output
            if "Error:" in output or "error:" in output:
                error_msg = _extract_error(output)
                return SpecResult(
                    status="failed",
                    duration_sec=duration,
                    last_attempt=timestamp,
                    commit=commit,
                    error_message=error_msg,
                )
            return SpecResult(
                status="passed",
                duration_sec=duration,
                verified_at=timestamp,
                commit=commit,
            )
        elif retcode in OOM_EXIT_CODES:
            return SpecResult(
                status="oom",
                duration_sec=duration,
                last_attempt=timestamp,
                commit=commit,
                error_message=f"Out of memory (exit code {retcode})",
            )
        else:
            error_msg = _extract_error(output) or f"Exit code {retcode}"
            return SpecResult(
                status="failed",
                duration_sec=duration,
                last_attempt=timestamp,
                commit=commit,
                error_message=error_msg[:200],
            )

    except FileNotFoundError:
        return SpecResult(
            status="error",
            last_attempt=timestamp,
            error_message=f"TLC command not found: {tlc}",
        )
    except Exception as e:
        return SpecResult(
            status="error",
            last_attempt=timestamp,
            error_message=str(e)[:200],
        )
    finally:
        if process is not None and process.poll() is None:
            with contextlib.suppress(OSError, ValueError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
                process.wait(timeout=1)
        if process is not None:
            _close_stdout(process)


def _extract_error(output: str) -> str | None:
    """Extract meaningful error message from TLC output."""
    for line in output.split("\n"):
        # TLC error patterns
        if "Error:" in line:
            return line.strip()[:200]
        if "Invariant" in line and "violated" in line.lower():
            return line.strip()[:200]
        if "Deadlock" in line:
            return line.strip()[:200]
        if "assert" in line.lower() and "failed" in line.lower():
            return line.strip()[:200]
    return None


def audit_specs(status: TlaStatus, discovered: list[str]) -> dict:
    """Compare tracked specs against discovered ones.

    REQUIRES: status is valid TlaStatus, discovered is list of spec names
    ENSURES: returns dict with keys {missing, orphaned, matched} containing sorted lists

    Returns:
        Dict with 'missing' (in repo but not tracked),
        'orphaned' (tracked but not in repo), 'matched' (both).
    """
    tracked = set(status.specs.keys())
    found = set(discovered)

    return {
        "missing": sorted(found - tracked),
        "orphaned": sorted(tracked - found),
        "matched": sorted(found & tracked),
    }


def run_filtered_specs(
    status: TlaStatus,
    specs: list[str],
    filter_status: str | None,
    timeout_sec: int,
    cwd: Path | None = None,
    tlc_cmd: str | None = None,
) -> TlaStatus:
    """Run specs matching the filter.

    REQUIRES: status is valid TlaStatus, specs is list, timeout_sec > 0
    ENSURES: returns updated TlaStatus with results for matching specs

    Args:
        status: Current status
        specs: All discovered specs
        filter_status: Only run specs with this status (or None for all)
        timeout_sec: Timeout per spec
        cwd: Working directory
        tlc_cmd: TLC command

    Returns:
        Updated status
    """
    to_run: list[str] = []

    for spec in specs:
        current = status.specs.get(spec)
        if filter_status is None:
            to_run.append(spec)
        elif current is None and filter_status == "not_run":
            to_run.append(spec)
        elif current is not None and current.status == filter_status:
            to_run.append(spec)

    if not to_run:
        print(f"No specs match filter '{filter_status}'", file=sys.stderr)
        return status

    print(f"Running {len(to_run)} specs...", file=sys.stderr, flush=True)

    for i, spec in enumerate(to_run):
        print(f"[{i+1}/{len(to_run)}] {spec}...", file=sys.stderr, flush=True)
        result = run_spec(spec, timeout_sec, cwd, tlc_cmd)
        status.specs[spec] = result
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
        description="Run TLA+ specs with timeout management and status tracking"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Timeout per spec in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="Run specific spec by name",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["not_run", "passed", "failed", "timeout", "oom"],
        help="Only run specs with this status",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Compare tracking file to actual specs (no execution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover specs without executing",
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

    # Find TLC
    tlc_cmd = find_tlc()
    if not tlc_cmd and not args.dry_run and not args.audit:
        print(
            "Error: TLC not found. Install with: brew install tla-plus-toolbox",
            file=sys.stderr,
        )
        print("Or ensure Java is installed: brew install openjdk", file=sys.stderr)
        return 1

    # Load existing status
    status = load_status(args.status_file)

    # Update TLC version if available
    if tlc_cmd:
        version = get_tlc_version(tlc_cmd)
        if version:
            status.tlc_version = version

    # Discover specs
    print("Discovering TLA+ specs...", file=sys.stderr, flush=True)
    discovered = discover_specs(cwd)
    print(f"Found {len(discovered)} specs", file=sys.stderr, flush=True)

    if args.dry_run:
        # Just list discovered specs
        if args.json:
            print(json.dumps({"specs": discovered}, indent=2))
        else:
            for s in discovered:
                current = status.specs.get(s)
                status_str = current.status if current else "not_run"
                print(f"  {s}: {status_str}")
        return 0

    if args.audit:
        # Compare tracking vs discovered
        audit_result = audit_specs(status, discovered)
        if args.json:
            print(json.dumps(audit_result, indent=2))
        else:
            print(f"\nMatched: {len(audit_result['matched'])}")
            print(f"Missing (in repo, not tracked): {len(audit_result['missing'])}")
            for s in audit_result["missing"][:10]:
                print(f"  + {s}")
            if len(audit_result["missing"]) > 10:
                print(f"  ... and {len(audit_result['missing']) - 10} more")
            print(f"Orphaned (tracked, not in repo): {len(audit_result['orphaned'])}")
            for s in audit_result["orphaned"][:10]:
                print(f"  - {s}")
            if len(audit_result["orphaned"]) > 10:
                print(f"  ... and {len(audit_result['orphaned']) - 10} more")
        return 0 if not audit_result["missing"] else 1

    # Run specs
    if args.spec:
        # Run specific spec
        if args.spec not in discovered:
            print(f"Spec '{args.spec}' not found", file=sys.stderr)
            print(f"Available: {', '.join(discovered[:5])}...", file=sys.stderr)
            return 1
        print(f"Running {args.spec}...", file=sys.stderr, flush=True)
        result = run_spec(args.spec, args.timeout, cwd, tlc_cmd)
        status.specs[args.spec] = result
        print(f"Result: {result.status}", file=sys.stderr, flush=True)
        if result.duration_sec:
            print(f"Duration: {result.duration_sec:.1f}s", file=sys.stderr)
        if result.error_message:
            print(f"Error: {result.error_message}", file=sys.stderr)
    else:
        # Run all (or filtered)
        status = run_filtered_specs(
            status, discovered, args.filter, args.timeout, cwd, tlc_cmd
        )

    # Save updated status
    save_status(status, args.status_file)
    print(f"Status saved to {args.status_file}", file=sys.stderr, flush=True)

    # Print summary
    summary = status.to_dict()["summary"]
    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        print("\nSummary: ", end="")
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

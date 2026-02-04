#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# Andrew Yates <ayates@dropbox.com>

"""
validate_claim.py - Validate benchmark claims in commit messages.

Purpose:
  Parse commit message claims like "15/20 eval_name" and verify them against
  eval results stored under a results directory (default: evals/results).

Public API (library usage):
    from ai_template_scripts.validate_claim import (
        Claim,
        ClaimResult,
        parse_claims,
        validate_claim,
    )

CLI usage:
    ./validate_claim.py --commit HEAD
    ./validate_claim.py --results-root evals/results --strict

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
License: Apache-2.0
"""

from __future__ import annotations

__all__ = [
    "Claim",
    "ClaimResult",
    "NumericClaim",
    "NumericClaimResult",
    "VerificationMatch",
    "parse_claims",
    "parse_numeric_claims",
    "validate_claim",
    "validate_numeric_claim",
    "detect_verification_output",
    "main",
]

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.version import get_version  # noqa: E402

CLAIM_PATTERNS = [
    re.compile(r"(?P<passed>\d+)\s*/\s*(?P<total>\d+)\s+(?P<eval_id>[A-Za-z0-9_-]+)\b"),
    re.compile(
        r"(?P<eval_id>[A-Za-z0-9_-]+)\b\s*:\s*(?P<passed>\d+)\s*/\s*(?P<total>\d+)"
    ),
]

# Numeric metric claim patterns (Part of #1900)
# Examples: "1.22x eval_id", "eval_id: ratio=1.22", "eval_id: speedup=1.5"
NUMERIC_CLAIM_PATTERNS = [
    # Multiplier format: "1.22x eval_id"
    re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*x\s+(?P<eval_id>[A-Za-z0-9_-]+)\b"
    ),
    # Metric=value format: "eval_id: metric_name=1.22"
    re.compile(
        r"(?P<eval_id>[A-Za-z0-9_-]+)\b\s*:\s*(?P<metric>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\d+(?:\.\d+)?)"
    ),
]

# Default tolerance for numeric comparisons (10%)
DEFAULT_NUMERIC_TOLERANCE = 0.1

# Patterns for detecting verification output (Part of #1938)
# These detect common test/build output patterns without full validation
VERIFICATION_PATTERNS = {
    "pytest_passed": re.compile(
        r"(\d+)\s+passed(?:\s*,\s*(\d+)\s+(?:skipped|warnings?))?\s+in\s+[\d.]+s",
        re.IGNORECASE,
    ),
    "pytest_failed": re.compile(
        r"(\d+)\s+failed(?:\s*,\s*\d+\s+passed)?\s+in\s+[\d.]+s", re.IGNORECASE
    ),
    "cargo_check_finished": re.compile(
        r"Finished\s+(?:dev|release)\s+.*?in\s+[\d.]+s", re.IGNORECASE
    ),
    "cargo_test_passed": re.compile(
        r"test\s+result:\s+ok\.\s+(\d+)\s+passed;\s+(\d+)\s+failed", re.IGNORECASE
    ),
    "cargo_test_failed": re.compile(
        r"test\s+result:\s+FAILED\.\s+\d+\s+passed;\s+(\d+)\s+failed", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class Claim:
    """Benchmark claim parsed from a commit message."""

    eval_id: str
    passed: int
    total: int
    raw: str


@dataclass(frozen=True)
class ClaimResult:
    """Validation outcome for a benchmark claim."""

    claim: Claim
    status: str
    reason: str


@dataclass(frozen=True)
class VerificationMatch:
    """Detected verification output pattern (Part of #1938)."""

    pattern_name: str  # e.g., "pytest_passed", "cargo_check_finished"
    raw: str  # The matched text
    passed: int | None = None  # For test results, number passed
    failed: int | None = None  # For test results, number failed


@dataclass(frozen=True)
class NumericClaim:
    """Numeric metric claim parsed from a commit message (Part of #1900).

    Examples:
        - "1.22x benchmark" -> NumericClaim(eval_id="benchmark", metric="ratio", value=1.22)
        - "benchmark: speedup=1.5" -> NumericClaim(eval_id="benchmark", metric="speedup", value=1.5)
    """

    eval_id: str
    metric: str  # "ratio" for multiplier format, or explicit metric name
    value: float
    raw: str


@dataclass(frozen=True)
class NumericClaimResult:
    """Validation outcome for a numeric metric claim (Part of #1900)."""

    claim: NumericClaim
    status: str  # "VALID", "INVALID", "UNKNOWN"
    reason: str
    actual_value: float | None = None  # The actual value found in results


def parse_claims(message: str) -> list[Claim]:
    """Extract benchmark claims (e.g. '15/20 eval_name') from text.

    REQUIRES: message is a string (may be empty)
    ENSURES: Returns list of Claim objects (may be empty)
    ENSURES: Each claim has valid passed/total integers extracted from text
    ENSURES: Deduplicates claims with same (eval_id, passed, total) case-insensitively
    """
    claims: list[Claim] = []
    seen: set[tuple[str, int, int]] = set()
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(message):
            eval_id = match.group("eval_id")
            passed = int(match.group("passed"))
            total = int(match.group("total"))
            raw = match.group(0)
            key = (eval_id.lower(), passed, total)
            if key in seen:
                continue
            seen.add(key)
            claims.append(Claim(eval_id=eval_id, passed=passed, total=total, raw=raw))
    return claims


def parse_numeric_claims(message: str) -> list[NumericClaim]:
    """Extract numeric metric claims from text (Part of #1900).

    Supports formats:
        - "1.22x eval_id" -> metric="ratio", value=1.22
        - "eval_id: metric_name=1.5" -> metric=metric_name, value=1.5

    REQUIRES: message is a string (may be empty)
    ENSURES: Returns list of NumericClaim objects (may be empty)
    ENSURES: Each claim has valid float value extracted from text
    ENSURES: Deduplicates claims with same (eval_id, metric, value) case-insensitively
    """
    claims: list[NumericClaim] = []
    seen: set[tuple[str, str, float]] = set()

    for pattern in NUMERIC_CLAIM_PATTERNS:
        for match in pattern.finditer(message):
            eval_id = match.group("eval_id")
            value = float(match.group("value"))
            raw = match.group(0)

            # Multiplier format uses "ratio" as metric name
            try:
                metric = match.group("metric")
            except IndexError:
                metric = "ratio"

            key = (eval_id.lower(), metric.lower(), value)
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                NumericClaim(eval_id=eval_id, metric=metric, value=value, raw=raw)
            )
    return claims


def detect_verification_output(text: str) -> list[VerificationMatch]:
    """Detect common verification output patterns in text (Part of #1938).

    Recognizes:
    - pytest output: "X passed in Y.YYs" or "X failed, Y passed"
    - cargo check: "Finished dev in X.XXs"
    - cargo test: "test result: ok. X passed; 0 failed"

    Returns list of detected patterns. Does NOT validate against stored results.

    REQUIRES: text is a string (may be empty)
    ENSURES: Returns list of VerificationMatch objects (may be empty)
    ENSURES: Each match has pattern_name from VERIFICATION_PATTERNS keys
    ENSURES: passed/failed fields populated for test result patterns
    """
    matches: list[VerificationMatch] = []

    for name, pattern in VERIFICATION_PATTERNS.items():
        for match in pattern.finditer(text):
            raw = match.group(0)
            passed: int | None = None
            failed: int | None = None

            if name == "pytest_passed":
                passed = int(match.group(1))
            elif name == "pytest_failed":
                failed = int(match.group(1))
            elif name == "cargo_test_passed":
                passed = int(match.group(1))
                failed = int(match.group(2))
            elif name == "cargo_test_failed":
                failed = int(match.group(1))

            matches.append(
                VerificationMatch(
                    pattern_name=name,
                    raw=raw,
                    passed=passed,
                    failed=failed,
                )
            )

    return matches


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON from path, returning None on error."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def resolve_commit(commit: str) -> str:
    """Resolve commit ref to full SHA via git rev-parse."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", commit],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return commit
    return result.stdout.strip()


def load_commit_message(commit: str) -> str:
    """Get commit message body via git show."""
    result = subprocess.run(
        ["git", "show", "-s", "--format=%B", commit],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = stderr or f"Failed to read commit message for {commit}"
        raise RuntimeError(message)
    return result.stdout


def commit_matches(metadata_commit: str, claimed_commit: str) -> bool:
    """Check if commit SHAs match (prefix comparison)."""
    meta = metadata_commit.strip().lower()
    claim = claimed_commit.strip().lower()
    if not meta or not claim:
        return False
    return meta.startswith(claim) or claim.startswith(meta)


def parse_score(obj: dict[str, Any]) -> tuple[int, int] | None:
    """Extract (passed, total) from dict, validating types and bounds.

    REQUIRES: obj is a dict (may be empty or missing keys)
    ENSURES: Returns None if 'passed' or 'total' keys missing
    ENSURES: Returns None if values are booleans (explicit rejection)
    ENSURES: Returns None if values are not int or float
    ENSURES: Returns None if float values have fractional parts
    ENSURES: Returns None if bounds invalid (passed < 0, total <= 0, passed > total)
    ENSURES: Returns (passed_int, total_int) tuple with 0 <= passed <= total, total > 0
    """
    passed = obj.get("passed")
    total = obj.get("total")
    if passed is None or total is None:
        return None
    if isinstance(passed, bool) or isinstance(total, bool):
        return None
    if not isinstance(passed, (int, float)) or not isinstance(total, (int, float)):
        return None
    try:
        passed_int = int(passed)
        total_int = int(total)
    except (OverflowError, ValueError):
        return None
    if passed_int != passed or total_int != total:
        return None
    if passed_int < 0 or total_int <= 0 or passed_int > total_int:
        return None
    return passed_int, total_int


def extract_score(results: dict[str, Any]) -> tuple[int, int] | None:
    """Extract score from results dict, checking metrics sub-object first."""
    metrics = results.get("metrics")
    if isinstance(metrics, dict):
        score = parse_score(metrics)
        if score is not None:
            return score
    return parse_score(results)


def resolve_eval_dir(results_root: Path, eval_id: str) -> Path | None:
    """Find eval directory by ID, case-insensitive fallback."""
    candidate = results_root / eval_id
    if candidate.is_dir():
        return candidate
    lower = results_root / eval_id.lower()
    if lower.is_dir():
        return lower
    if results_root.is_dir():
        for child in results_root.iterdir():
            if child.is_dir() and child.name.lower() == eval_id.lower():
                return child
    return None


def validate_claim(
    claim: Claim,
    commit: str,
    results_root: Path,
) -> ClaimResult:
    """Validate a benchmark claim against stored eval results.

    REQUIRES: claim is a valid Claim object
    REQUIRES: commit is a git commit ref (SHA or ref name)
    REQUIRES: results_root is a Path (may not exist)
    ENSURES: Returns ClaimResult with status in {"VALID", "INVALID", "UNKNOWN"}
    ENSURES: VALID only if score matches stored results for commit
    ENSURES: INVALID if claim score is invalid or mismatches stored results
    ENSURES: UNKNOWN if results directory not found or no matching runs
    """
    if claim.passed < 0 or claim.total <= 0 or claim.passed > claim.total:
        return ClaimResult(claim, "INVALID", "invalid claim score")
    eval_dir = resolve_eval_dir(results_root, claim.eval_id)
    if eval_dir is None:
        return ClaimResult(claim, "UNKNOWN", "results dir not found")

    mismatched = 0
    unknown = 0
    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        results_path = run_dir / "results.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        if metadata is None:
            unknown += 1
            continue
        metadata_commit = str(metadata.get("git_commit", ""))
        if not commit_matches(metadata_commit, commit):
            continue
        results = load_json(results_path) if results_path.exists() else None
        if results is None:
            unknown += 1
            continue
        score = extract_score(results)
        if score is None:
            unknown += 1
            continue
        if score == (claim.passed, claim.total):
            return ClaimResult(claim, "VALID", f"run_id={run_dir.name}")
        mismatched += 1

    if mismatched:
        return ClaimResult(claim, "INVALID", "score mismatch")
    if unknown:
        return ClaimResult(claim, "UNKNOWN", "missing metrics for commit")
    return ClaimResult(claim, "UNKNOWN", "no runs for commit")


def validate_numeric_claim(
    claim: NumericClaim,
    commit: str,
    results_root: Path,
    tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> NumericClaimResult:
    """Validate a numeric metric claim against stored eval results (Part of #1900).

    REQUIRES: claim is a valid NumericClaim object
    REQUIRES: commit is a git commit ref (SHA or ref name)
    REQUIRES: results_root is a Path (may not exist)
    REQUIRES: 0 <= tolerance <= 1 (fraction, e.g., 0.1 = 10%)
    ENSURES: Returns NumericClaimResult with status in {"VALID", "INVALID", "UNKNOWN"}
    ENSURES: VALID if metric value within tolerance of stored value
    ENSURES: INVALID if value outside tolerance or claim invalid
    ENSURES: UNKNOWN if results directory not found or metric not present
    """
    if not (0 <= tolerance <= 1):
        raise ValueError(f"tolerance must be in [0, 1], got {tolerance}")
    if claim.value <= 0:
        return NumericClaimResult(claim, "INVALID", "value must be positive")

    eval_dir = resolve_eval_dir(results_root, claim.eval_id)
    if eval_dir is None:
        return NumericClaimResult(claim, "UNKNOWN", "results dir not found")

    mismatched = 0
    unknown = 0
    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        results_path = run_dir / "results.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        if metadata is None:
            unknown += 1
            continue
        metadata_commit = str(metadata.get("git_commit", ""))
        if not commit_matches(metadata_commit, commit):
            continue
        results = load_json(results_path) if results_path.exists() else None
        if results is None:
            unknown += 1
            continue

        # Look for metric in results.metrics[metric_name]
        metrics = results.get("metrics", {})
        if not isinstance(metrics, dict):
            unknown += 1
            continue

        actual_value = metrics.get(claim.metric)
        if actual_value is None:
            unknown += 1
            continue

        try:
            actual_value = float(actual_value)
        except (TypeError, ValueError):
            unknown += 1
            continue

        # Check within tolerance
        diff = abs(actual_value - claim.value)
        threshold = claim.value * tolerance
        if diff <= threshold:
            return NumericClaimResult(
                claim, "VALID", f"run_id={run_dir.name}", actual_value
            )
        mismatched += 1

    if mismatched:
        return NumericClaimResult(claim, "INVALID", "value outside tolerance")
    if unknown:
        return NumericClaimResult(claim, "UNKNOWN", "metric not found for commit")
    return NumericClaimResult(claim, "UNKNOWN", "no runs for commit")


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for validate_claim CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate benchmark-style claims in a commit message against eval results."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("validate_claim.py"),
    )
    parser.add_argument(
        "--commit",
        default="HEAD",
        help="Git commit or ref whose message is checked (default: HEAD).",
    )
    parser.add_argument(
        "--message",
        help="Override commit message text (skips git lookup).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("evals/results"),
        help="Root directory containing eval results (default: evals/results).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on UNKNOWN claims.",
    )
    return parser


def main() -> int:
    """CLI entry point: validate benchmark claims in commit messages.

    REQUIRES: sys.argv contains valid CLI arguments
    ENSURES: Returns 0 if no claims or all claims valid
    ENSURES: Returns 1 if any claims invalid or error occurred
    ENSURES: Returns 2 if --strict and any claims unknown
    ENSURES: Prints validation results for each claim to stdout
    """
    parser = build_parser()
    args = parser.parse_args()

    commit_ref = args.commit
    commit = resolve_commit(commit_ref)

    try:
        if args.message is not None:
            message = args.message
        else:
            message = load_commit_message(commit_ref)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    claims = parse_claims(message)
    if not claims:
        print("No claims found in commit message.")
        return 0

    results_root = args.results_root
    results = [validate_claim(claim, commit, results_root) for claim in claims]

    invalid = 0
    unknown = 0
    for result in results:
        status = result.status
        claim = result.claim
        print(
            f"{status} {claim.eval_id} {claim.passed}/{claim.total} "
            f"(commit {commit[:7]}) - {result.reason}"
        )
        if status == "INVALID":
            invalid += 1
        elif status == "UNKNOWN":
            unknown += 1

    summary = f"Summary: {len(results)} claims, {invalid} invalid, {unknown} unknown"
    print(summary)

    if invalid:
        return 1
    if args.strict and unknown:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

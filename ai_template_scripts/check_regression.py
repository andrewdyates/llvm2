#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Regression checker template for benchmark evals.

Compares latest eval results against committed baselines to detect:
- Correctness regression (wrong answers increase, passed rate drops)
- Performance regression (ratio drops beyond threshold)

Usage:
    python3 scripts/check_regression.py <eval_id>
    python3 scripts/check_regression.py <eval_id> --save-snapshot

Example:
    python3 scripts/check_regression.py smt-local-suite
    python3 scripts/check_regression.py smt-local-suite --save-snapshot

Directory structure expected:
    evals/results/<eval_id>/<run_id>/results.json   # Eval run results
    evals/results/<eval_id>/<run_id>/metadata.json  # Run metadata
    metrics/benchmarks/<eval_id>/<date>.json        # Baseline snapshots

Results JSON format:
    {"metrics": {"passed": N, "total": N, "wrong": N, "ratio": F}}

Metadata JSON format:
    {"run_id": "...", "git_commit": "..."}

Part of #1969
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Configurable paths - adjust for your project
RESULTS_DIR = Path("evals/results")
SNAPSHOTS_DIR = Path("metrics/benchmarks")


def find_latest_run(eval_id: str) -> Path | None:
    """Find the most recent run directory for an eval.

    Args:
        eval_id: The eval identifier (e.g., "smt-local-suite")

    Returns:
        Path to the run directory, or None if no results exist.
    """
    results_dir = RESULTS_DIR / eval_id
    if not results_dir.exists():
        return None

    runs = sorted(results_dir.iterdir(), reverse=True)
    for run_dir in runs:
        if (run_dir / "results.json").exists():
            return run_dir
    return None


def find_baseline(eval_id: str) -> dict | None:
    """Find the most recent committed baseline snapshot for an eval.

    Args:
        eval_id: The eval identifier

    Returns:
        Parsed baseline JSON, or None if no baseline exists.
    """
    snapshots_dir = SNAPSHOTS_DIR / eval_id
    if not snapshots_dir.exists():
        return None

    snapshots = sorted(snapshots_dir.glob("*.json"), reverse=True)
    if not snapshots:
        return None

    with open(snapshots[0]) as f:
        return json.load(f)


def check_regression(
    eval_id: str,
    threshold_ratio: float = 0.9,
    threshold_passed_pct: float = 0.95,
) -> tuple[bool, str]:
    """Check for regression in the latest eval run.

    Args:
        eval_id: The eval identifier to check
        threshold_ratio: Min acceptable ratio of current/baseline performance (default 0.9 = 10% regression allowed)
        threshold_passed_pct: Min acceptable ratio of current/baseline passed rate (default 0.95 = 5% regression allowed)

    Returns:
        Tuple of (passed: bool, report: str)
    """
    latest = find_latest_run(eval_id)
    if not latest:
        return (
            False,
            f"No eval results found for {eval_id}. Run eval first.",
        )

    with open(latest / "results.json") as f:
        current = json.load(f)

    metadata_path = latest / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"run_id": latest.name, "git_commit": "unknown"}

    current_metrics = current["metrics"]
    baseline = find_baseline(eval_id)

    report = []
    report.append(f"Eval: {eval_id}")
    report.append(f"Run: {metadata.get('run_id', 'unknown')}")
    report.append(f"Commit: {metadata.get('git_commit', 'unknown')}")
    report.append("")

    report.append("Current metrics:")
    report.append(f"  passed: {current_metrics['passed']}/{current_metrics['total']}")
    if "wrong" in current_metrics:
        report.append(f"  wrong: {current_metrics['wrong']}")
    if "ratio" in current_metrics:
        report.append(f"  ratio: {current_metrics['ratio']}x")

    issues = []

    if not baseline:
        report.append("")
        report.append("No baseline snapshot found. First run - no regression check.")
        return True, "\n".join(report)

    baseline_metrics = baseline["metrics"]
    report.append("")
    report.append("Baseline metrics:")
    report.append(f"  passed: {baseline_metrics['passed']}/{baseline_metrics['total']}")
    if "wrong" in baseline_metrics:
        report.append(f"  wrong: {baseline_metrics['wrong']}")
    if "ratio" in baseline_metrics:
        report.append(f"  ratio: {baseline_metrics['ratio']}x")

    # Check correctness regression (wrong answers)
    if current_metrics.get("wrong", 0) > baseline_metrics.get("wrong", 0):
        issues.append(
            f"REGRESSION: wrong answers increased "
            f"({baseline_metrics.get('wrong', 0)} -> {current_metrics['wrong']})"
        )

    # Check passed rate regression
    if current_metrics["total"] > 0 and baseline_metrics["total"] > 0:
        passed_ratio = current_metrics["passed"] / current_metrics["total"]
        baseline_passed_ratio = baseline_metrics["passed"] / baseline_metrics["total"]
        if passed_ratio < baseline_passed_ratio * threshold_passed_pct:
            issues.append(
                f"REGRESSION: passed ratio dropped "
                f"({baseline_passed_ratio:.2%} -> {passed_ratio:.2%})"
            )

    # Check performance regression (if ratio metric exists)
    if "ratio" in current_metrics and "ratio" in baseline_metrics:
        if baseline_metrics["ratio"] > 0:
            ratio_delta = current_metrics["ratio"] / baseline_metrics["ratio"]
            if ratio_delta < threshold_ratio:
                issues.append(
                    f"REGRESSION: performance ratio dropped "
                    f"({baseline_metrics['ratio']}x -> {current_metrics['ratio']}x)"
                )

    report.append("")
    if issues:
        report.append("FAILED:")
        for issue in issues:
            report.append(f"  {issue}")
        return False, "\n".join(report)
    report.append("PASSED: No regressions detected.")
    return True, "\n".join(report)


def save_snapshot(eval_id: str) -> str:
    """Save current eval results as a committed baseline snapshot.

    Args:
        eval_id: The eval identifier

    Returns:
        Status message
    """
    latest = find_latest_run(eval_id)
    if not latest:
        return f"No eval results found for {eval_id}"

    with open(latest / "results.json") as f:
        results = json.load(f)

    metadata_path = latest / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"run_id": latest.name, "git_commit": "unknown"}

    snapshots_dir = SNAPSHOTS_DIR / eval_id
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y-%m-%d")
    snapshot_path = snapshots_dir / f"{date}.json"

    snapshot = {
        "metrics": results["metrics"],
        "git_commit": metadata.get("git_commit", "unknown"),
        "run_id": metadata.get("run_id", "unknown"),
        "saved_at": datetime.now().isoformat(),
    }

    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return f"Snapshot saved to {snapshot_path}"


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Check for benchmark regressions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s smt-local-suite                  # Check for regressions
    %(prog)s smt-local-suite --save-snapshot  # Save new baseline
    %(prog)s eval-name --threshold-ratio 0.95 # Stricter performance check
        """,
    )
    parser.add_argument("eval_id", help="Eval identifier to check")
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Save current results as new baseline",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.9,
        help="Min acceptable ratio of current/baseline performance (default: 0.9)",
    )
    parser.add_argument(
        "--threshold-passed",
        type=float,
        default=0.95,
        help="Min acceptable ratio of current/baseline passed rate (default: 0.95)",
    )
    args = parser.parse_args()

    if args.save_snapshot:
        msg = save_snapshot(args.eval_id)
        print(msg)
        return 0

    passed, message = check_regression(
        args.eval_id,
        threshold_ratio=args.threshold_ratio,
        threshold_passed_pct=args.threshold_passed,
    )

    print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

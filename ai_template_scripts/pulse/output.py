#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Output formatting functions for pulse metrics display.

Part of #404: pulse.py module split.
"""

from .issue_metrics import _unpack_issue_counts


def _print_code_status(metrics: dict) -> None:
    """Print code status: LOC, large files, consolidation debt."""
    loc = metrics.get("loc", {})
    if loc:
        loc_str = ", ".join(f"{k}:{v}" for k, v in loc.items())
        print(f"**LOC:** {loc_str}")

    large = metrics.get("large_files", [])
    print(f"**Large files:** {len(large)}")

    consolidation = metrics.get("consolidation", {})
    if consolidation.get("total", 0) > 0:
        total = consolidation.get("total", 0)
        target = consolidation.get("target", 0)
        debt = consolidation.get("debt", 0)
        print(f"**Template Lines:** {total} (target: {target}, debt: {debt})")


def _print_test_status(metrics: dict) -> None:
    """Print test status: count, recent runs, orphaned tests."""
    tests = metrics.get("tests", {})
    by_framework = tests.get("by_framework", {})
    if len(by_framework) > 1:
        breakdown = ", ".join(f"{fw}:{n}" for fw, n in sorted(by_framework.items()))
        print(f"**Tests:** {tests.get('count', 0)} (mixed: {breakdown})")
    else:
        print(
            f"**Tests:** {tests.get('count', 0)} ({tests.get('framework', 'unknown')})"
        )
    recent = tests.get("recent_24h", {})
    if recent:
        parts = [
            f"{recent.get('runs', 0)} runs",
            f"{recent.get('passed', 0)} passed",
            f"{recent.get('failed', 0)} failed",
        ]
        # Show command_errors separately if present (#929)
        if recent.get("command_errors", 0) > 0:
            parts.append(f"{recent.get('command_errors')} command errors")
        print("**Test runs (24h):** " + ", ".join(parts))
    orphaned = tests.get("orphaned_tests", [])
    if orphaned:
        print(f"**Orphaned tests:** {', '.join(orphaned[:5])}")
    # Test coverage display (#2116)
    coverage = tests.get("coverage", {})
    if coverage:
        cov_pct = coverage.get("coverage_pct", 0)
        lines_covered = coverage.get("lines_covered", 0)
        lines_total = coverage.get("lines_total", 0)
        source = coverage.get("source", "")
        parts = [f"{cov_pct:.1f}%"]
        if lines_total > 0:
            parts.append(f"{lines_covered}/{lines_total} lines")
        # Only show source if meaningful (not empty or "unknown")
        if source and source != "unknown":
            parts.append(f"from {source}")
        print("**Test coverage:** " + " ".join(parts))


def _print_issue_status(metrics: dict) -> None:
    """Print issue status: counts and blocked issues missing reason."""
    issues = metrics.get("issues", {})
    issue_counts, issue_error = _unpack_issue_counts(issues)
    if issue_error:
        print(f"**Issues:** ⚠️ {issue_error}")
    else:
        print(
            "**Issues:** "
            f"{issue_counts.get('open', 0)} open, "
            f"{issue_counts.get('in_progress', 0)} in-progress, "
            f"{issue_counts.get('blocked', 0)} blocked"
        )

    blocked_missing = metrics.get("blocked_missing_reason", [])
    if blocked_missing:
        if blocked_missing == "skipped":
            print("**Blocked missing reason:** [skipped]")
        else:
            issue_list = ", ".join(f"#{n}" for n in blocked_missing[:5])
            print(f"**Blocked missing reason:** {issue_list}")


def _print_system_status(metrics: dict) -> None:
    """Print system status: memory, disk, artifacts, sessions, crashes, GH rate limits."""
    system = metrics.get("system", {})
    memory = system.get("memory", {})
    disk = system.get("disk", {})
    artifacts = system.get("build_artifacts", {})
    gh_limits = system.get("gh_rate_limits", {})

    if memory:
        print(
            "**Memory:** "
            f"{memory.get('percent_used', '?')}% "
            f"({memory.get('used_gb', '?')}GB / {memory.get('total_gb', '?')}GB)"
        )
    if disk:
        print(
            f"**Disk:** {disk.get('percent_used', '?')} "
            f"({disk.get('used', '?')} / {disk.get('total', '?')})"
        )
    if artifacts:
        artifact_str = ", ".join(f"{k}={v}" for k, v in artifacts.items())
        print(f"**Build artifacts:** {artifact_str}")

    # GitHub API rate limits (from cache)
    if gh_limits:
        parts = []
        for name in ["core", "graphql", "search"]:
            if name in gh_limits and isinstance(gh_limits[name], dict):
                info = gh_limits[name]
                pct = info.get("pct", 0)
                remaining = info.get("remaining", 0)
                limit = info.get("limit", 0)
                velocity = info.get("velocity")
                exhaust_min = info.get("exhaust_min")
                reset_min = info.get("reset_min", 0)

                # Status indicators
                status = ""
                if pct < 2:
                    status = " [CRITICAL]"
                elif pct < 10:
                    status = " [LOW]"

                # Velocity string (if available)
                vel_str = ""
                if velocity is not None and velocity != 0:
                    vel_str = f" {velocity:+.0f}/m"
                    if exhaust_min is not None and exhaust_min < reset_min:
                        vel_str += f" EXHAUST:{exhaust_min:.0f}m!"

                parts.append(f"{name}:{remaining}/{limit}{vel_str}{status}")

        if parts:
            age_sec = gh_limits.get("cache_age_sec", 0)
            stale = gh_limits.get("stale", False)
            commit = gh_limits.get("commit")
            pending = gh_limits.get("pending_sync", 0)

            age_str = f"{age_sec}s" if age_sec < 120 else f"{age_sec // 60}m"
            stale_marker = " STALE" if stale else ""
            commit_str = f"@{commit}" if commit else ""
            pending_str = f" +{pending}pending" if pending else ""

            print(
                f"**GH API ({age_str}{commit_str}{stale_marker}{pending_str}):** {', '.join(parts)}"
            )

    active = metrics.get("active_sessions", 0)
    if active > 0:
        print(f"**Active AI sessions:** {active}")

    crashes_data = metrics.get("crashes_24h", {})
    if isinstance(crashes_data, dict):
        real = crashes_data.get("real", 0)
        stale = crashes_data.get("stale_connection", 0)
        if real > 0 or stale > 0:
            if real > 0 and stale > 0:
                print(f"**Crashes (24h):** {real} real, {stale} restarts")
            elif real > 0:
                print(f"**Crashes (24h):** {real}")
            else:
                print(f"**Restarts (24h):** {stale} (stale_connection)")
    elif crashes_data > 0:
        # Legacy: int value
        print(f"**Crashes (24h):** {crashes_data}")


def _print_proof_status(metrics: dict) -> None:
    """Print proof coverage status: Kani, TLA+, Lean, SMT (#2234)."""
    proofs = metrics.get("proofs", {})
    if not proofs or not isinstance(proofs, dict):
        return

    parts = []

    # Kani (Rust bounded model checking)
    kani = proofs.get("kani", {})
    if kani:
        proof_count = kani.get("proofs", 0)
        # Build status string with verification breakdown if available (#2232, #2263)
        passing = kani.get("passing")
        failing = kani.get("failing")
        unexecuted = kani.get("unexecuted")
        timeout = kani.get("timeout")
        oom = kani.get("oom")
        has_status = any(
            x is not None for x in [passing, failing, unexecuted, timeout, oom]
        )
        if has_status:
            # Has verification status - show detailed breakdown
            status_parts = []
            if passing is not None and passing > 0:
                status_parts.append(f"{passing} passing")
            if failing is not None and failing > 0:
                status_parts.append(f"{failing} failing")
            if unexecuted is not None and unexecuted > 0:
                status_parts.append(f"{unexecuted} not_run")
            if timeout is not None and timeout > 0:
                status_parts.append(f"{timeout} timeout")
            if oom is not None and oom > 0:
                status_parts.append(f"{oom} oom")
            if status_parts:
                parts.append(f"Kani:{proof_count} ({', '.join(status_parts)})")
            else:
                parts.append(f"Kani:{proof_count}")
        else:
            # No verification status - just show count
            parts.append(f"Kani:{proof_count}")

    # TLA+ (distributed system specs)
    tla = proofs.get("tla2", {})
    if tla:
        specs = tla.get("specs", 0)
        invariants = tla.get("invariants", 0)
        parts.append(f"TLA+:{specs} specs, {invariants} inv")

    # Lean (theorem proving)
    lean = proofs.get("lean", {})
    if lean:
        theorems = lean.get("theorems", 0)
        parts.append(f"Lean:{theorems} theorems")

    # SMT (satisfiability)
    smt = proofs.get("smt", {})
    if smt:
        smt_files = smt.get("smt2_files", 0)
        z4_usage = smt.get("z4_usage", 0)
        if z4_usage > 0:
            parts.append(f"SMT:{smt_files} files, {z4_usage} z4")
        else:
            parts.append(f"SMT:{smt_files} files")

    # NN verification
    nn = proofs.get("nn_verification", {})
    if nn:
        onnx = nn.get("onnx_models", 0)
        vnnlib = nn.get("vnnlib_specs", 0)
        parts.append(f"NN:{onnx} models, {vnnlib} specs")

    if parts:
        print(f"**Proofs:** {', '.join(parts)}")


def _complexity_grade(avg: float) -> str:
    """Map average complexity to letter grade (radon scale).

    A: 1-5, B: 6-10, C: 11-20, D: 21-30, E: 31-40, F: 41+
    """
    if avg <= 5:
        return "A"
    if avg <= 10:
        return "B"
    if avg <= 20:
        return "C"
    if avg <= 30:
        return "D"
    if avg <= 40:
        return "E"
    return "F"


def _print_complexity_status(metrics: dict) -> None:
    """Print code complexity status: avg complexity, high-complexity functions."""
    complexity = metrics.get("complexity", {})
    if not complexity or not isinstance(complexity, dict):
        return

    # Check for error (tools not installed)
    if "error" in complexity:
        return  # Silently skip - tools not installed

    total_funcs = complexity.get("total_functions", 0)
    if total_funcs == 0:
        return  # No functions analyzed

    avg = complexity.get("avg_complexity", 0)
    high_count = complexity.get("high_warning_count", 0)  # functions > 20
    grade = _complexity_grade(avg)

    # Format: "Complexity: 5 high-complexity, avg grade C (15.2)"
    parts = []
    if high_count > 0:
        parts.append(f"{high_count} high-complexity")
    parts.append(f"avg grade {grade} ({avg:.1f})")

    print(f"**Complexity:** {', '.join(parts)}")

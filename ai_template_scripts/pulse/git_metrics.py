#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
pulse.git_metrics - Git repository status and code quality metrics.

Functions for git status, code quality metrics, dependency checking, and template tracking.

Part of #404: pulse.py module split
"""

import re
from datetime import datetime
from pathlib import Path

try:
    from ai_template_scripts.subprocess_utils import run_cmd
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import run_cmd

from .config import THRESHOLDS
from .constants import GIT_DEP_PATTERN, GREP_EXCLUDE_DIRS


def _resolve_root(repo_root: Path | None) -> Path:
    """Resolve repo root path, defaulting to cwd."""
    if repo_root is None:
        return Path.cwd()
    return repo_root


def _relative_path(path: Path, root: Path) -> str:
    """Get relative path string from root, with fallback."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _detect_git_operation_in_progress() -> str | None:
    """Detect if a git operation (rebase/merge/cherry-pick/revert/bisect) is in progress.

    Checks for sentinel files in .git/ that indicate an incomplete operation.
    These operations block commits/pushes and require manual resolution.

    Returns:
        Operation name (e.g., "rebase", "merge", "bisect") if one is in progress, else None.

    Per #1899: Detects stalled git state that can block AI roles.
    Recovery commands:
        - rebase: git rebase --abort
        - merge: git merge --abort
        - cherry-pick: git cherry-pick --abort
        - revert: git revert --abort
        - bisect: git bisect reset
    """
    # Get .git directory (handles worktrees)
    result = run_cmd(["git", "rev-parse", "--git-dir"])
    if not result.ok:
        return None
    git_dir = Path(result.stdout.strip())

    # Check for operation markers in priority order
    # Rebase markers (interactive or regular)
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        return "rebase"

    # Merge marker
    if (git_dir / "MERGE_HEAD").exists():
        return "merge"

    # Cherry-pick marker
    if (git_dir / "CHERRY_PICK_HEAD").exists():
        return "cherry-pick"

    # Revert marker
    if (git_dir / "REVERT_HEAD").exists():
        return "revert"

    # Bisect marker (uses BISECT_LOG for tracking)
    if (git_dir / "BISECT_LOG").exists():
        return "bisect"

    return None


def get_git_status(porcelain_lines: list[str] | None = None) -> dict:
    """Get git repository status.

    Args:
        porcelain_lines: Optional pre-fetched git status --porcelain output lines.
                         If None, will fetch fresh. Pass to avoid redundant calls (#1340).

    REQUIRES: Current directory is within a git repository
    ENSURES: Returns dict with 'dirty' (bool), 'uncommitted_files' (int)
    ENSURES: Returns dict with optional 'branch', 'head', 'commits_7d'
    ENSURES: Returns dict with 'operation_in_progress' (str|None) if git op active (#1899)
    ENSURES: Never raises (returns partial dict on error)
    """
    status: dict[str, object] = {}

    # Check if dirty (uncommitted changes)
    if porcelain_lines is None:
        result = run_cmd(["git", "status", "--porcelain"])
        if result.ok:
            stdout = result.stdout.strip()
            porcelain_lines = (
                [line for line in stdout.split("\n") if line.strip()] if stdout else []
            )
        else:
            porcelain_lines = []

    status["dirty"] = len(porcelain_lines) > 0
    status["uncommitted_files"] = len(porcelain_lines)

    # Get current branch
    result = run_cmd(["git", "branch", "--show-current"])
    if result.ok and result.stdout.strip():
        status["branch"] = result.stdout.strip()

    # Get last commit hash
    result = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    if result.ok and result.stdout.strip():
        status["head"] = result.stdout.strip()

    # Commit velocity (last 7 days)
    result = run_cmd(["git", "rev-list", "--count", "--since=7.days", "HEAD"])
    if result.ok and result.stdout.strip():
        try:
            status["commits_7d"] = int(result.stdout.strip())
        except ValueError:
            pass

    # Check for git operation in progress (#1899)
    # These operations block commits/pushes and can silently stall AIs
    status["operation_in_progress"] = _detect_git_operation_in_progress()

    return status


def _count_todo_comments(repo_root: Path | None = None) -> int | None:
    """Count TODO/FIXME/XXX/HACK comments as tech debt indicator.

    Returns:
        Count of TODO-like comments, or None if count failed.

    Optimized: Uses --exclude-dir for upfront filtering instead of post-grep filtering.
    """
    root = _resolve_root(repo_root)

    # Use --exclude-dir upfront (fast) instead of post-filtering with grep -v (slow)
    result = run_cmd(
        [
            "bash",
            "-c",
            (
                f"grep -rEc '(TODO|FIXME|XXX|HACK):?' . "
                f"--include='*.py' --include='*.rs' --include='*.go' "
                f"--include='*.js' --include='*.ts' "
                f"{GREP_EXCLUDE_DIRS} 2>/dev/null | "
                "awk -F: '{sum+=$2} END {print sum+0}'"
            ),
        ],
        timeout=30,
        cwd=root,
    )
    if result.ok and result.stdout.strip():
        try:
            return int(result.stdout.strip())
        except ValueError:
            pass
    return None


def _compute_test_code_ratio(repo_root: Path | None = None) -> float | None:
    """Compute ratio of test lines to source lines.

    Returns:
        Ratio rounded to 2 decimal places, or None if no source code.
    """
    root = _resolve_root(repo_root)

    # Count test lines (with timeout to prevent hang on large repos - #956)
    result = run_cmd(
        [
            "bash",
            "-c",
            (
                "find tests/ test/ -name '*.py' -o -name '*.rs' 2>/dev/null | "
                "xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}'"
            ),
        ],
        timeout=30,
        cwd=root,
    )
    test_lines = (
        int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0
    )

    # Count source lines (with timeout to prevent hang on large repos - #956)
    result = run_cmd(
        [
            "bash",
            "-c",
            (
                "find src/ lib/ -name '*.py' -o -name '*.rs' 2>/dev/null | "
                "xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}'"
            ),
        ],
        timeout=30,
        cwd=root,
    )
    src_lines = int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0

    if src_lines > 0:
        return round(test_lines / src_lines, 2)
    return None


def _compute_python_type_coverage(repo_root: Path | None = None) -> float | None:
    """Compute percentage of Python functions with type annotations.

    Returns:
        Coverage percentage rounded to 1 decimal, or None if no Python project.

    Optimized: Uses --exclude-dir for upfront filtering.

    Fix for #1069: Improved regex to handle:
    - async def functions
    - All Python identifiers (not just lowercase)
    - Return-only annotations (-> type)
    - Dunder methods (__init__, etc.)
    """
    root = _resolve_root(repo_root)
    if not (root / "pyproject.toml").exists() and not list(root.glob("*.py")):
        return None

    # Count functions with type annotations (params OR return type) (#1069)
    # Matches: def foo(x: int), def foo() -> int, async def bar(x: str) -> bool
    # Use --exclude-dir upfront for speed, -c with awk for accurate count
    result = run_cmd(
        [
            "bash",
            "-c",
            (
                f"grep -rEc '(async\\s+)?def\\s+[a-zA-Z_][a-zA-Z0-9_]*\\([^)]*:|"
                f"(async\\s+)?def\\s+[a-zA-Z_][a-zA-Z0-9_]*\\(.*\\)\\s*->' "
                f". --include='*.py' {GREP_EXCLUDE_DIRS} 2>/dev/null | "
                "awk -F: '{sum+=$2} END {print sum+0}'"
            ),
        ],
        timeout=30,
        cwd=root,
    )
    typed_fns = int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0

    # Count all functions (including async def)
    result = run_cmd(
        [
            "bash",
            "-c",
            (
                f"grep -rEc '^[[:space:]]*(async\\s+)?def\\s+[a-zA-Z_][a-zA-Z0-9_]*\\(' "
                f". --include='*.py' {GREP_EXCLUDE_DIRS} 2>/dev/null | "
                "awk -F: '{sum+=$2} END {print sum+0}'"
            ),
        ],
        timeout=30,
        cwd=root,
    )
    total_fns = int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0

    if total_fns > 0:
        return round(100 * typed_fns / total_fns, 1)
    return None


def get_code_quality(repo_root: Path | None = None) -> dict:
    """Get code quality metrics that should improve over time.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with optional 'todo_count', 'test_code_ratio', 'python_type_coverage_pct'
    ENSURES: Never raises (returns partial dict on error)
    """
    quality: dict[str, float | int] = {}

    if (todo_count := _count_todo_comments(repo_root)) is not None:
        quality["todo_count"] = todo_count

    if (ratio := _compute_test_code_ratio(repo_root)) is not None:
        quality["test_code_ratio"] = ratio

    if (type_cov := _compute_python_type_coverage(repo_root)) is not None:
        quality["python_type_coverage_pct"] = type_cov

    return quality


def count_template_lines(repo_root: Path | None = None) -> dict:
    """Count total template lines that sync to all repos.

    Template files are in .claude/rules/ and .claude/roles/ - these get synced
    to all projects in the org. Line count tracks consolidation effort.

    Returns dict with per-file counts, total, and target from VISION.md.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with 'total', 'target', and per-file line counts
    ENSURES: Never raises (returns partial dict on error)
    """
    template_patterns = [
        ".claude/rules/ai_template.md",
        ".claude/rules/org_chart.md",
        ".claude/roles/*.md",
    ]

    root = _resolve_root(repo_root)
    counts = {}
    total = 0

    for pattern in template_patterns:
        files = list(root.glob(pattern))
        for f in files:
            try:
                lines = len(f.read_text().splitlines())
                counts[_relative_path(f, root)] = lines
                total += lines
            except (FileNotFoundError, UnicodeDecodeError):
                pass

    return {
        "files": counts,
        "total": total,
        "target": 1200,  # From VISION.md consolidation phase goal
    }


def get_consolidation_debt(repo_root: Path | None = None) -> dict:
    """Calculate consolidation debt (lines over target).

    Returns dict with current total, target, and debt.
    Debt = max(0, actual - target) so negative values aren't shown.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with 'total', 'target', 'debt', 'files' keys
    ENSURES: debt is non-negative integer
    """
    template = count_template_lines(repo_root=repo_root)
    actual = template["total"]
    target = template["target"]
    debt = max(0, actual - target)

    return {
        "total": actual,
        "target": target,
        "debt": debt,
        "files": template["files"],
    }


def _format_drift_reason(details: dict) -> str:
    """Format drift details into human-readable reason string."""
    reasons = []
    if details.get("code_path_verified") is False:
        reasons.append(f"Path not found: {details.get('code_path', '?')}")
    if details.get("code_marker_verified") is False:
        reasons.append(f"Marker not found: {details.get('code_marker', '?')}")
    return "; ".join(reasons) if reasons else "Unknown reason"


def get_doc_claim_status(repo_root: Path | None = None) -> dict:
    """Check documentation claims against code reality.

    Uses check_doc_claims.py to verify YAML frontmatter claims in CLAUDE.md
    and VISION.md match the actual codebase state.

    Returns dict with verified/unverified counts and drift status.
    See #1494 for design.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with claim counts or an 'error' key on failure
    ENSURES: Never raises (returns error dict on failure)
    """
    root = _resolve_root(repo_root)

    try:
        # Import check_doc_claims lazily to avoid startup cost
        from ai_template_scripts.check_doc_claims import check_doc_claims  # noqa: PLC0415, I001

        results = check_doc_claims(root)
        return {
            "total_claims": results["total_claims"],
            "verified": results["verified_claims"],
            "unverified": results["unverified_claims"],
            "drift_detected": results["unverified_claims"] > 0,
            "unverified_claims": [
                {
                    "name": d["claim"],
                    "type": d["type"],
                    "reason": _format_drift_reason(d["details"]),
                }
                for d in results["drift"]
            ],
        }
    except ImportError:
        # Module not available (shouldn't happen in ai_template)
        return {"error": "check_doc_claims module not available"}
    except Exception as e:
        # Don't let doc claim failures break pulse
        return {"error": str(e)}


def _get_repo_head_rev(repo_url: str) -> str | None:
    """Get HEAD revision for a GitHub repository.

    Uses gh api when available, falls back to git ls-remote.
    Returns full SHA or None on failure.

    REQUIRES: repo_url is a GitHub repository URL
    ENSURES: Returns lowercase full SHA or None on failure
    ENSURES: Never raises (returns None on error)
    """
    # Extract owner/repo from URL
    match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
    if not match:
        return None

    owner, repo = match.groups()

    # Try gh api first (faster, uses cache)
    gh_result = run_cmd(
        ["gh", "api", f"repos/{owner}/{repo}/commits/HEAD", "--jq", ".sha"],
        timeout=10,
    )
    if gh_result.ok and gh_result.stdout.strip():
        return gh_result.stdout.strip().lower()

    # Fallback to git ls-remote
    ls_result = run_cmd(["git", "ls-remote", repo_url, "HEAD"], timeout=15)
    if ls_result.ok and ls_result.stdout.strip():
        # Format: "sha\tHEAD"
        parts = ls_result.stdout.strip().split()
        if parts:
            return parts[0].lower()

    return None


def _get_commit_staleness(repo_url: str, pinned_rev: str, head_rev: str) -> dict | None:
    """Get staleness info for a git dependency (#1876).

    Fetches commit date and counts commits between pinned rev and HEAD.
    Uses gh api for efficiency (with cache).

    Args:
        repo_url: GitHub repository URL
        pinned_rev: Currently pinned revision (short or full SHA)
        head_rev: HEAD revision (full SHA)

    Returns dict with:
        days_old: Days since pinned commit was authored
        commits_behind: Number of commits between pinned and HEAD
    Or None on failure.

    REQUIRES: repo_url is a GitHub repository URL
    REQUIRES: pinned_rev and head_rev are valid commit SHAs
    ENSURES: Returns dict with days_old (int) and commits_behind (int), or None
    ENSURES: Never raises (returns None on error)
    """
    match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
    if not match:
        return None

    owner, repo = match.groups()

    # Get commit date for pinned rev
    date_result = run_cmd(
        [
            "gh",
            "api",
            f"repos/{owner}/{repo}/commits/{pinned_rev}",
            "--jq",
            ".commit.author.date",
        ],
        timeout=10,
    )
    if not date_result.ok or not date_result.stdout.strip():
        return None

    try:
        commit_date_str = date_result.stdout.strip()
        # Parse ISO format date (e.g., "2026-01-29T10:00:00Z")
        commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
        now = datetime.now(commit_date.tzinfo)
        days_old = (now - commit_date).days
    except (ValueError, TypeError):
        days_old = 0

    # Count commits between pinned and HEAD using compare API
    # This returns commits ahead/behind between two refs
    compare_result = run_cmd(
        [
            "gh",
            "api",
            f"repos/{owner}/{repo}/compare/{pinned_rev}...{head_rev}",
            "--jq",
            ".ahead_by",
        ],
        timeout=10,
    )
    if compare_result.ok and compare_result.stdout.strip().isdigit():
        commits_behind = int(compare_result.stdout.strip())
    else:
        commits_behind = 0

    return {"days_old": days_old, "commits_behind": commits_behind}


def get_outdated_git_deps(repo_root: Path | None = None) -> dict:
    """Check for outdated git dependencies in Cargo.toml files.

    Scans Cargo.toml files for git dependencies with pinned revisions and
    compares them to the HEAD of each dependency's repository.

    Returns dict with:
        outdated: List of {repo: str, current_rev: str, head_rev: str, file: str}
        checked: Total dependencies checked
        error: Error message if any (graceful degradation)

    Part of #1553.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with 'outdated' (list) and 'checked' (int) keys
    ENSURES: Each outdated entry has repo, repo_url, current_rev, head_rev, file keys
    """
    root = _resolve_root(repo_root)
    result: dict = {
        "outdated": [],
        "checked": 0,
    }

    # Find all Cargo.toml files
    cargo_files = list(root.glob("**/Cargo.toml"))
    if not cargo_files:
        return result

    # Uses module-level GIT_DEP_PATTERN (#1698)
    # Track unique repos to avoid checking same dep multiple times
    checked_repos: dict[str, str] = {}  # repo_url -> head_rev (or "error")
    # Cache staleness info per (repo_url, pinned_rev) to avoid redundant API calls
    staleness_cache: dict[tuple[str, str], dict | None] = {}
    # Dedupe outdated entries by (repo_url, current_rev) (#1876)
    # Key: (repo_url, current_rev) -> entry dict with files list
    outdated_map: dict[tuple[str, str], dict] = {}

    # Get staleness thresholds (#1876)
    stale_days = THRESHOLDS.get("git_dep_stale_days", 3)
    stale_commits = THRESHOLDS.get("git_dep_stale_commits", 10)

    for cargo_file in cargo_files:
        try:
            content = cargo_file.read_text()
        except OSError:
            continue

        for match in GIT_DEP_PATTERN.finditer(content):
            repo_url = match.group(1).rstrip(".git")
            current_rev = match.group(2).lower()
            result["checked"] += 1

            # Get HEAD rev (cached per repo)
            if repo_url not in checked_repos:
                head_rev = _get_repo_head_rev(repo_url)
                checked_repos[repo_url] = head_rev or "error"

            head_rev = checked_repos[repo_url]
            if head_rev == "error":
                continue

            # Compare (prefix match for short revs)
            if not head_rev.startswith(current_rev) and not current_rev.startswith(
                head_rev[: len(current_rev)]
            ):
                rel_path = str(cargo_file.relative_to(root))
                key = (repo_url, current_rev[:12])
                if key in outdated_map:
                    # Add file to existing entry (#1876)
                    outdated_map[key]["files"].append(rel_path)
                else:
                    # Get staleness info for this pinned rev (cached)
                    staleness_key = (repo_url, current_rev[:12])
                    if staleness_key not in staleness_cache:
                        staleness_cache[staleness_key] = _get_commit_staleness(
                            repo_url, current_rev, head_rev
                        )
                    staleness = staleness_cache[staleness_key]

                    # Apply staleness threshold (#1876)
                    # Only flag if dep exceeds staleness criteria (or thresholds are 0)
                    days_old = staleness.get("days_old", 0) if staleness else 0
                    commits_behind = (
                        staleness.get("commits_behind", 0) if staleness else 0
                    )
                    is_stale = (stale_days == 0 and stale_commits == 0) or (
                        (stale_days > 0 and days_old >= stale_days)
                        or (stale_commits > 0 and commits_behind >= stale_commits)
                    )
                    if not is_stale:
                        continue  # Skip non-stale deps

                    # New outdated entry
                    outdated_map[key] = {
                        "repo": repo_url.split("/")[-1],  # Just repo name
                        "repo_url": repo_url,
                        "current_rev": current_rev[:12],
                        "head_rev": head_rev[:12],
                        "files": [rel_path],
                        "days_old": days_old,
                        "commits_behind": commits_behind,
                    }

    # Convert map to list, preserving backwards compatibility for single-file case
    for entry in outdated_map.values():
        files = entry["files"]
        entry["files_count"] = len(files)
        # Keep "file" for backwards compat (first file), add "files" for full list
        entry["file"] = files[0] if files else ""
        result["outdated"].append(entry)

    return result

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
System resource metrics collection.

Functions for memory, disk, and build artifact monitoring.
Part of #404: pulse.py module split.
"""

import json
import os
import time
from pathlib import Path
from typing import NotRequired, TypedDict

try:
    from ai_template_scripts.subprocess_utils import run_cmd
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import run_cmd

from .config import THRESHOLDS

# Generated artifact extensions that may bloat disk (#1477)
BLOAT_ARTIFACT_EXTENSIONS = [
    ".smt2",  # SMT solver artifacts (can be huge)
    ".symtab",  # Symbol tables
    ".profraw",  # LLVM profile data
    ".rlib",  # Rust libraries (can accumulate)
    ".rmeta",  # Rust metadata
    ".ll",  # LLVM IR
    ".bc",  # LLVM bitcode
]


class LargeFileEntry(TypedDict):
    """Entry for large file detection."""

    file: str
    size_gb: float


class ArtifactFileEntry(TypedDict):
    """Entry for artifact file detection."""

    file: str
    size_mb: float


class StatesDirEntry(TypedDict):
    """Entry for TLA+ states directory detection (#1551)."""

    path: str
    size_gb: float


class DiskBloatResult(TypedDict):
    """Type for detect_disk_bloat return value."""

    large_files: list[LargeFileEntry]
    tests_size_gb: float | None
    reports_size_mb: NotRequired[float]  # Part of #1695
    artifact_files: list[ArtifactFileEntry]
    states_dirs: NotRequired[list[StatesDirEntry]]
    bloat_detected: bool


def _resolve_root(repo_root: Path | None) -> Path:
    """Resolve repo root path, defaulting to cwd."""
    if repo_root is None:
        return Path.cwd()
    return repo_root


def _get_memory_usage_macos() -> dict | None:
    """Get memory usage on macOS via vm_stat.

    Returns:
        Dict with used_gb, free_gb, total_gb, percent_used, or None if not macOS.
    """
    result = run_cmd(["bash", "-c", "vm_stat 2>/dev/null | head -10"], timeout=5)
    if not result.ok or not result.stdout.strip() or "page size" not in result.stdout:
        return None

    page_size = 16384  # Default, usually 16KB on Apple Silicon
    pages_free = pages_active = pages_inactive = pages_wired = pages_compressed = 0

    for line in result.stdout.strip().split("\n"):
        try:
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif "Pages free:" in line:
                pages_free = int(line.split()[-1].rstrip("."))
            elif "Pages active:" in line:
                pages_active = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split()[-1].rstrip("."))
            elif "Pages wired down:" in line:
                pages_wired = int(line.split()[-1].rstrip("."))
            elif "Pages occupied by compressor:" in line:
                pages_compressed = int(line.split()[-1].rstrip("."))
        except (ValueError, IndexError):
            pass

    used_bytes = (pages_active + pages_wired + pages_compressed) * page_size
    free_bytes = (pages_free + pages_inactive) * page_size
    total_bytes = used_bytes + free_bytes

    if total_bytes > 0:
        return {
            "used_gb": round(used_bytes / (1024**3), 1),
            "free_gb": round(free_bytes / (1024**3), 1),
            "total_gb": round(total_bytes / (1024**3), 1),
            "percent_used": round(100 * used_bytes / total_bytes, 1),
        }
    return None


def _get_memory_usage_linux() -> dict | None:
    """Get memory usage on Linux via /proc/meminfo.

    Returns:
        Dict with used_gb, free_gb, total_gb, percent_used, or None if not Linux.
    """
    try:
        meminfo = Path("/proc/meminfo").read_text()
        mem_total = mem_available = 0
        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024  # KB to bytes
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) * 1024
        if mem_total > 0:
            used_bytes = mem_total - mem_available
            return {
                "used_gb": round(used_bytes / (1024**3), 1),
                "free_gb": round(mem_available / (1024**3), 1),
                "total_gb": round(mem_total / (1024**3), 1),
                "percent_used": round(100 * used_bytes / mem_total, 1),
            }
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return None


def _get_disk_usage() -> dict | None:
    """Get disk usage for current volume via df.

    Returns:
        Dict with total, used, available, percent_used, or None if unavailable.
    """
    result = run_cmd(["df", "-h", "."])
    if not result.ok or not result.stdout.strip():
        return None

    lines = result.stdout.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 5:
            return {
                "total": parts[1],
                "used": parts[2],
                "available": parts[3],
                "percent_used": parts[4],
            }
    return None


def _get_build_artifact_sizes(fast: bool = False) -> dict | None:
    """Get sizes of build artifact directories.

    Args:
        fast: If True, use faster estimation method (file count) instead of du.

    Returns:
        Dict mapping dir name to size string, or None if no artifacts.
    """
    artifact_dirs = ["target", "build", "node_modules", ".build"]
    artifacts = {}

    for dirname in artifact_dirs:
        dirpath = Path(dirname)
        if not dirpath.is_dir():
            continue

        if fast:
            # Fast mode: count files instead of computing exact size
            # This is O(readdir) instead of O(stat all files)
            result = run_cmd(
                ["find", dirname, "-maxdepth", "3", "-type", "f"],
                timeout=5,
            )
            if result.ok:
                file_count = (
                    len(result.stdout.strip().split("\n"))
                    if result.stdout.strip()
                    else 0
                )
                artifacts[dirname] = f"~{file_count} files"
        else:
            # Full mode: get exact size with reasonable timeout
            result = run_cmd(["du", "-sh", dirname], timeout=15)
            if result.ok and result.stdout.strip():
                size = result.stdout.split()[0]
                artifacts[dirname] = size
            elif not result.ok:
                # Timeout or error - fall back to file count estimate
                result = run_cmd(
                    ["find", dirname, "-maxdepth", "2", "-type", "f"],
                    timeout=5,
                )
                if result.ok:
                    file_count = (
                        len(result.stdout.strip().split("\n"))
                        if result.stdout.strip()
                        else 0
                    )
                    artifacts[dirname] = f"~{file_count}+ files"

    return artifacts if artifacts else None


def detect_disk_bloat(repo_root: Path | None = None) -> DiskBloatResult:
    """Detect disk bloat: large files, large tests/, reports/, large states/, and artifacts.

    Returns dict with:
        large_files: List of files >THRESHOLDS["large_file_size_gb"] GB
        tests_size_gb: Size of tests/ directory in GB (if >threshold)
        reports_size_mb: Size of reports/ directory in MB (if >threshold, #1695)
        artifact_files: List of large generated artifacts (>100MB)
        states_dirs: List of large TLA+ states/ directories (if >threshold, #1551)
        bloat_detected: True if any bloat found

    Part of #1477.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns DiskBloatResult typed dict
    ENSURES: bloat_detected is True iff any detected fields (large_files, tests_size_gb, etc.)
    ENSURES: Never raises (returns empty result on error)
    """
    root = _resolve_root(repo_root)
    result: DiskBloatResult = {
        "large_files": [],
        "tests_size_gb": None,
        "artifact_files": [],
        "bloat_detected": False,
    }

    size_threshold = THRESHOLDS.get("large_file_size_gb", 1)
    tests_threshold = THRESHOLDS.get("tests_dir_size_gb", 10)

    # 1. Find individual files > size_threshold GB
    # Use find with -size flag for efficiency
    size_bytes = int(size_threshold * 1024 * 1024 * 1024)
    find_result = run_cmd(
        [
            "find",
            ".",
            "-type",
            "f",
            "-size",
            f"+{size_bytes}c",
            "-not",
            "-path",
            "./.git/*",
            "-not",
            "-path",
            "./target/*",
            "-not",
            "-path",
            "./build/*",
            "-not",
            "-path",
            "./node_modules/*",
        ],
        timeout=60,
        cwd=root,
    )
    if find_result.ok and find_result.stdout.strip():
        for filepath in find_result.stdout.strip().split("\n"):
            filepath = filepath.removeprefix("./")
            if filepath:
                # Get actual size - use os.stat for cross-platform compatibility
                try:
                    file_stat = os.stat(root / filepath)
                    size_gb = file_stat.st_size / (1024**3)
                    result["large_files"].append(
                        {
                            "file": filepath,
                            "size_gb": round(size_gb, 2),
                        }
                    )
                except OSError:
                    pass  # Skip files we can't stat

    # 2. Check tests/ directory size
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        du_result = run_cmd(["du", "-sk", "tests"], timeout=30, cwd=root)
        if du_result.ok and du_result.stdout.strip():
            try:
                size_kb = int(du_result.stdout.split()[0])
                size_gb = size_kb / (1024 * 1024)  # KB to GB
                if size_gb >= tests_threshold:
                    result["tests_size_gb"] = round(size_gb, 2)
            except (ValueError, IndexError):
                pass

    # 2a. Check reports/ directory size (#1695)
    # Ephemeral reports accumulate indefinitely; use cleanup_old_reports.py to clean
    reports_dir = root / "reports"
    reports_threshold_mb = THRESHOLDS.get("reports_dir_size_mb", 50)
    if reports_dir.is_dir():
        du_result = run_cmd(["du", "-sk", "reports"], timeout=30, cwd=root)
        if du_result.ok and du_result.stdout.strip():
            try:
                size_kb = int(du_result.stdout.split()[0])
                size_mb = size_kb / 1024  # KB to MB
                if size_mb >= reports_threshold_mb:
                    result["reports_size_mb"] = round(size_mb, 1)
            except (ValueError, IndexError):
                pass

    # 2b. Check TLA+ states/ directories (#1551)
    # These grow unbounded from model checking and have caused disk issues
    states_threshold = THRESHOLDS.get("states_dir_size_gb", 10)
    states_dirs: list[StatesDirEntry] = []
    for states_dir in root.glob("**/states"):
        if (
            states_dir.is_dir()
            and ".git" not in states_dir.parts
            and "target" not in states_dir.parts
            and "node_modules" not in states_dir.parts
            and "build" not in states_dir.parts
            and "__pycache__" not in states_dir.parts
        ):
            du_result = run_cmd(
                ["du", "-sk", str(states_dir.relative_to(root))],
                timeout=30,
                cwd=root,
            )
            if du_result.ok and du_result.stdout.strip():
                try:
                    size_kb = int(du_result.stdout.split()[0])
                    size_gb = size_kb / (1024 * 1024)
                    if size_gb >= states_threshold:
                        states_dirs.append(
                            {
                                "path": str(states_dir.relative_to(root)),
                                "size_gb": round(size_gb, 2),
                            }
                        )
                except (ValueError, IndexError):
                    pass
    if states_dirs:
        result["states_dirs"] = states_dirs

    # 3. Find large generated artifacts (*.smt2, *.symtab, etc.)
    # Look for files with artifact extensions > 100MB
    artifact_size_bytes = 100 * 1024 * 1024  # 100MB
    ext_pattern = " -o ".join([f'-name "*{ext}"' for ext in BLOAT_ARTIFACT_EXTENSIONS])
    artifact_result = run_cmd(
        [
            "bash",
            "-c",
            f"find . \\( {ext_pattern} \\) -type f -size +{artifact_size_bytes}c "
            f'-not -path "./.git/*" -not -path "./target/*" -not -path "./build/*" '
            f'-not -path "./node_modules/*" 2>/dev/null',
        ],
        timeout=60,
        cwd=root,
    )
    if artifact_result.ok and artifact_result.stdout.strip():
        for filepath in artifact_result.stdout.strip().split("\n"):
            filepath = filepath.removeprefix("./")
            if filepath:
                # Use os.stat for cross-platform compatibility
                try:
                    file_stat = os.stat(root / filepath)
                    size_mb = file_stat.st_size / (1024**2)
                    result["artifact_files"].append(
                        {
                            "file": filepath,
                            "size_mb": round(size_mb, 1),
                        }
                    )
                except OSError:
                    pass  # Skip files we can't stat

    # Set bloat_detected flag
    result["bloat_detected"] = bool(
        result["large_files"]
        or result["tests_size_gb"]
        or result.get("reports_size_mb")  # Per #1695
        or result["artifact_files"]
        or result.get("states_dirs")  # Per #1551
    )

    return result


def _get_gh_rate_limits() -> dict | None:
    """Read GitHub rate limits from cache file (no API call).

    Returns dict with core/graphql/search limits, cache age, commit, velocity, and pending sync.
    Returns None if no cache file.
    """
    cache_file = Path.home() / ".ait_gh_cache" / "rate_state.json"
    change_log_file = Path.home() / ".ait_gh_cache" / "change_log.json"

    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        timestamp = data.get("timestamp", 0)
        age_seconds = time.time() - timestamp
        resources = data.get("resources", {})
        usage_log = data.get("usage_log", [])

        result: dict[str, object] = {
            "cache_age_sec": round(age_seconds),
            "stale": age_seconds > 300,  # > 5 min = stale
            "commit": data.get("commit"),
        }

        # Calculate velocity for each resource
        for name in ["core", "graphql", "search"]:
            if name in resources:
                info = resources[name]
                remaining = info.get("remaining", 0)
                limit = info.get("limit", 1)
                reset_ts = info.get("reset", 0)
                pct = (remaining / limit * 100) if limit > 0 else 0
                reset_min = max(0, (reset_ts - time.time()) / 60)

                resource_data: dict[str, object] = {
                    "remaining": remaining,
                    "limit": limit,
                    "pct": round(pct, 1),
                    "reset_min": round(reset_min, 1),
                }

                # Calculate velocity from usage log
                relevant = [e for e in usage_log if name in e]
                if len(relevant) >= 2:
                    # Use last 10 min or last 2 points
                    now = time.time()
                    recent = [e for e in relevant if now - e["t"] < 600]
                    if len(recent) < 2:
                        recent = relevant[-2:]
                    oldest, newest = recent[0], recent[-1]
                    time_delta_min = (newest["t"] - oldest["t"]) / 60
                    if time_delta_min >= 0.1:
                        velocity = (newest[name] - oldest[name]) / time_delta_min
                        resource_data["velocity"] = round(velocity, 1)
                        if velocity < 0 and remaining > 0:
                            exhaust_min = -remaining / velocity
                            resource_data["exhaust_min"] = round(exhaust_min, 1)

                result[name] = resource_data

        # Count pending sync changes
        if change_log_file.exists():
            try:
                cl_data = json.loads(change_log_file.read_text())
                pending = sum(
                    1 for c in cl_data.get("changes", []) if not c.get("synced")
                )
                if pending > 0:
                    result["pending_sync"] = pending
            except Exception:
                pass

        return result if len(result) > 3 else None  # Must have at least one resource
    except Exception:
        return None


def get_system_resources(fast: bool = False) -> dict:
    """Get system resource usage (memory, disk, build artifacts) for trend tracking.

    Args:
        fast: If True, use faster estimation for build artifact sizes.

    REQUIRES: None (platform-agnostic)
    ENSURES: Returns dict with optional memory/disk/build_artifacts/disk_bloat/gh_rate_limits
    ENSURES: Memory/disk values include percent_used as percentage
    ENSURES: Never raises (returns partial dict on error)
    """
    resources: dict[str, object] = {}

    # Memory usage - try macOS first, then Linux
    if mem := _get_memory_usage_macos():
        resources["memory"] = mem
    elif mem := _get_memory_usage_linux():
        resources["memory"] = mem

    # Disk usage
    if disk := _get_disk_usage():
        resources["disk"] = disk

    # Build artifact sizes
    if artifacts := _get_build_artifact_sizes(fast=fast):
        resources["build_artifacts"] = artifacts

    # GitHub API rate limits (from cache, no API call)
    if gh_limits := _get_gh_rate_limits():
        resources["gh_rate_limits"] = gh_limits

    # Disk bloat detection (#1477)
    if not fast:
        bloat = detect_disk_bloat()
        if bloat.get("bloat_detected"):
            resources["disk_bloat"] = bloat

    return resources


__all__ = [
    # TypedDicts
    "LargeFileEntry",
    "ArtifactFileEntry",
    "StatesDirEntry",
    "DiskBloatResult",
    # Constants
    "BLOAT_ARTIFACT_EXTENSIONS",
    # Public functions
    "get_system_resources",
    "detect_disk_bloat",
    # Internal functions (exported for testing)
    "_get_memory_usage_macos",
    "_get_memory_usage_linux",
    "_get_disk_usage",
    "_get_build_artifact_sizes",
    "_get_gh_rate_limits",
    "_resolve_root",
]

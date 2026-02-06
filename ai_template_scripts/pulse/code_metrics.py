#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Code metrics collection for pulse module.

Functions for counting lines of code, finding large files,
and detecting forbidden CI workflows.

Part of #404: pulse.py module split
"""

import fnmatch
import json
from pathlib import Path

try:
    from ai_template_scripts.subprocess_utils import run_cmd
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import run_cmd

from .config import LARGE_FILE_EXCLUDE_PATTERNS
from .constants import EXCLUDE_DIRS, EXCLUDE_GLOB_PATTERNS, FIND_PRUNE, GREP_EXCLUDE


def _resolve_root(repo_root: Path | None) -> Path:
    """Return repo root or current directory when unset."""
    return repo_root if repo_root is not None else Path(".")


def _relative_path(path: Path, root: Path) -> str:
    """Return path relative to root when possible, else stringified path."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _path_is_excluded(path: Path) -> bool:
    """Return True if path includes excluded directories."""
    for part in path.parts:
        if part in EXCLUDE_DIRS or part == ".git":
            return True
        for pattern in EXCLUDE_GLOB_PATTERNS:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _matches_exclude_pattern(filepath: str, patterns: list[str]) -> bool:
    """Check if filepath matches any exclude pattern.

    Patterns can be:
    - Directory prefix: "tests/" matches "tests/foo.py"
    - Glob-like: "crates/*/tests/" matches "crates/z4-chc/tests/foo.rs"

    Args:
        filepath: File path to check (with or without "./" prefix - both handled).
        patterns: List of exclude patterns.

    Returns:
        True if filepath matches any pattern.
    """
    # Normalize: ensure starts with "./"
    if not filepath.startswith("./"):
        filepath = "./" + filepath

    for pattern in patterns:
        # Normalize pattern to match find output format
        if not pattern.startswith("./"):
            pattern = "./" + pattern

        # Check for glob pattern with *
        if "*" in pattern:
            # Simple glob expansion: "crates/*/tests/" matches "crates/foo/tests/"
            if fnmatch.fnmatch(filepath, pattern + "*") or fnmatch.fnmatch(
                filepath, pattern.rstrip("/") + "/*"
            ):
                return True
        else:
            # Prefix match for directory patterns
            if pattern.endswith("/"):
                if filepath.startswith((pattern, pattern.rstrip("/") + "/")):
                    return True
            else:
                # Exact match or prefix
                if filepath == pattern or filepath.startswith(pattern + "/"):
                    return True

    return False


def count_lines_by_type(repo_root: Path | None = None) -> dict[str, int]:
    """Count lines of code by file type. Tries tokei first, falls back to find+wc.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict mapping language names to non-negative line counts
    ENSURES: On error, returns empty dict (never raises)
    ENSURES: Excludes EXCLUDE_DIRS from all counts
    """
    root = _resolve_root(repo_root)
    # Try tokei first (fastest - single pass, all languages)
    # Exclude reference/vendor directories
    tokei_cmd = ["tokei", "-o", "json"] + [
        arg for d in EXCLUDE_DIRS for arg in ["-e", d]
    ]
    result = run_cmd(tokei_cmd, timeout=60, cwd=root)
    if result.ok and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            counts = {}
            lang_map = {
                "Python": "python",
                "Rust": "rust",
                "Go": "go",
                "C++": "cpp",
                "C": "c",
                "JavaScript": "javascript",
                "TypeScript": "typescript",
            }
            for lang, stats in data.items():
                if lang in lang_map and isinstance(stats, dict):
                    lines = stats.get("code", 0)
                    if lines > 0:
                        counts[lang_map[lang]] = lines
            if counts:
                return counts
        except (json.JSONDecodeError, KeyError):
            pass

    # Fall back to find+wc (works everywhere, parallel across languages)
    counts = {}
    extensions = {
        ".py": "python",
        ".rs": "rust",
        ".go": "go",
        ".cpp": "cpp",
        ".c": "c",
        ".js": "javascript",
        ".ts": "typescript",
    }

    # Run all find+wc in parallel using shell backgrounding
    # Use prune-style exclusion for speed on large repos
    result = run_cmd(
        [
            "bash",
            "-c",
            f"""
for ext in .py .rs .go .cpp .c .js .ts; do
    (count=$(
        find . -type d -name .git -prune -o {FIND_PRUNE} -o \\
            -name "*$ext" -type f -print0 2>/dev/null \\
            | xargs -0 wc -l 2>/dev/null \\
            | tail -1 \\
            | awk '{{print $1}}'
    )
     [ -n "$count" ] && [ "$count" -gt 0 ] && echo "$ext:$count") &
done
wait
""",
        ],
        timeout=30,
        cwd=root,
    )

    if result.ok and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                ext, count_str = line.split(":", 1)
                ext = ext.strip()
                if ext in extensions:
                    try:
                        counts[extensions[ext]] = int(count_str.strip())
                    except ValueError:
                        pass

    return counts


def find_large_files(
    max_lines: int,
    repo_root: Path | None = None,
    use_git: bool = True,
    warning_lines: int | None = None,
) -> list[dict]:
    """Find tracked files exceeding line limit using git ls-files+wc.

    Only considers files tracked by git (ignores gitignored/untracked files).
    This prevents flagging generated code, vendored deps, or temp files.

    Respects LARGE_FILE_EXCLUDE_PATTERNS from pulse.toml config.

    Args:
        max_lines: Minimum line count for "notice" tier (Part of #2358)
        repo_root: Repository root directory
        use_git: Whether to use git ls-files (vs find)
        warning_lines: Minimum line count for "warning" tier. If set, adds
            'tier' field to each result ('notice' or 'warning')

    REQUIRES: max_lines > 0
    REQUIRES: warning_lines is None or warning_lines > max_lines
    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns list of dicts with 'file' (str), 'lines' (int), optionally 'tier' (str)
    ENSURES: All returned files have lines > max_lines
    ENSURES: On error, returns empty list (never raises)
    ENSURES: If use_git=True, only includes git-tracked files
    """
    # Validate warning_lines makes sense (Part of #2358 audit)
    if warning_lines is not None and warning_lines <= max_lines:
        import sys

        print(
            f"Warning: warning_lines ({warning_lines}) should be > max_lines ({max_lines})",
            file=sys.stderr,
        )
        # Continue anyway, but tiers may be confusing

    root = _resolve_root(repo_root)
    large = []
    extensions = [".py", ".rs", ".go", ".cpp", ".c", ".js", ".ts"]

    # Build grep pattern for extensions: \.(py|rs|go|cpp|c|js|ts)$
    ext_pattern = r"\.(" + "|".join(e.lstrip(".") for e in extensions) + r")$"

    if use_git:
        # Use git ls-files to get only tracked files, filter by extension, count lines
        # This excludes gitignored and untracked files automatically (#971)
        # Also exclude EXCLUDE_DIRS for consistency with find mode (#1338)
        exclude_grep = f"grep -vE '{GREP_EXCLUDE}'"
        result = run_cmd(
            [
                "bash",
                "-c",
                (
                    f"git ls-files | grep -E '{ext_pattern}' | {exclude_grep} | "
                    "tr '\\n' '\\0' | xargs -0 wc -l 2>/dev/null | "
                    f"awk -v max={max_lines} '$1>max && $2!=\"total\" {{print $1, $2}}'"
                ),
            ],
            timeout=60,
            cwd=root,
        )
    else:
        name_expr = " -o ".join([f"-name '*{ext}'" for ext in extensions])
        result = run_cmd(
            [
                "bash",
                "-c",
                (
                    f"find . -type d -name .git -prune -o {FIND_PRUNE} -o "
                    f"\\( {name_expr} \\) -type f -print0 2>/dev/null | "
                    "xargs -0 wc -l 2>/dev/null | "
                    f"awk -v max={max_lines} '$1>max && $2!=\"total\" {{print $1, $2}}'"
                ),
            ],
            timeout=60,
            cwd=root,
        )

    if result.ok and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    lines = int(parts[0])
                    filepath = parts[1]
                    # Normalize path format (find returns ./path, git returns path)
                    filepath = filepath.removeprefix("./")
                    # Apply config-based exclusions
                    if LARGE_FILE_EXCLUDE_PATTERNS:
                        if _matches_exclude_pattern(
                            filepath, LARGE_FILE_EXCLUDE_PATTERNS
                        ):
                            continue
                    entry: dict = {"file": filepath, "lines": lines}
                    # Add tier info if warning_lines specified (Part of #2358)
                    if warning_lines is not None:
                        entry["tier"] = "warning" if lines >= warning_lines else "notice"
                    large.append(entry)
                except ValueError:
                    pass

    return sorted(large, key=lambda x: -x["lines"])


def find_forbidden_ci(repo_root: Path | None = None) -> list[str]:
    """Check for forbidden GitHub Actions CI workflows.

    Per ai_template rules: "No GitHub CI/CD: Don't create .github/workflows/.
    Not supported in dropbox-ai-prototypes GitHub."

    Returns list of forbidden workflow files found.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns list of workflow file paths relative to repo root
    ENSURES: Returns empty list when no forbidden workflows are found
    """
    root = _resolve_root(repo_root)
    workflows_dir = root / ".github/workflows"

    if not workflows_dir.exists():
        return []

    workflows: list[str] = []
    for pattern in ["*.yml", "*.yaml"]:
        workflows.extend(_relative_path(f, root) for f in workflows_dir.glob(pattern))

    return workflows


def get_complexity_metrics(repo_root: Path | None = None) -> dict:
    """Get code complexity metrics using code_stats analyzer.

    Returns summary of cyclomatic complexity across the codebase:
    - total_functions: Number of functions analyzed
    - avg_complexity: Average cyclomatic complexity
    - max_complexity: Maximum complexity found
    - warning_count: Functions exceeding threshold (default 10)
    - high_warning_count: Functions exceeding high threshold (default 20)

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with complexity summary
    ENSURES: On error, returns dict with "error" key
    """
    root = _resolve_root(repo_root)

    try:
        # Import here to avoid circular dependency
        from ai_template_scripts.code_stats import analyze
    except ImportError:
        return {"error": "code_stats not available"}

    try:
        result = analyze(dir_path=root)
        data = result.to_dict()

        # Aggregate across all languages
        summary = data.get("summary", {})
        total_funcs = summary.get("total_functions", 0)

        # Calculate aggregate avg/max from by_language
        by_lang = summary.get("by_language", {})
        total_complexity = 0
        max_complexity = 0
        for lang_stats in by_lang.values():
            avg = lang_stats.get("avg_complexity", 0)
            funcs = lang_stats.get("functions", 0)
            total_complexity += avg * funcs
            max_complexity = max(max_complexity, lang_stats.get("max_complexity", 0))

        avg_complexity = total_complexity / total_funcs if total_funcs > 0 else 0

        warnings = data.get("warnings", [])
        high_warnings = [w for w in warnings if w.get("severity") == "high"]

        return {
            "total_functions": total_funcs,
            "avg_complexity": round(avg_complexity, 2),
            "max_complexity": max_complexity,
            "warning_count": len(warnings),
            "high_warning_count": len(high_warnings),
            "by_language": {
                lang: {
                    "functions": stats.get("functions", 0),
                    "avg_complexity": stats.get("avg_complexity", 0),
                    "max_complexity": stats.get("max_complexity", 0),
                }
                for lang, stats in by_lang.items()
            },
        }
    except Exception as e:
        return {"error": str(e)}


def get_previous_metrics(days_back: int = 1) -> dict | None:
    """Read metrics from a previous day for trend comparison.

    Args:
        days_back: Number of days to look back (default 1 = yesterday).

    Returns:
        Most recent metrics dict from that day, or None if unavailable.

    REQUIRES: days_back >= 1
    ENSURES: Returns dict with metrics or None
    ENSURES: Never raises (returns None on error)
    """
    from datetime import datetime, timedelta

    from .constants import METRICS_DIR

    target_date = datetime.now() - timedelta(days=days_back)
    date_str = target_date.strftime("%Y-%m-%d")
    metrics_file = METRICS_DIR / f"{date_str}.json"

    if not metrics_file.exists():
        return None

    try:
        data = json.loads(metrics_file.read_text())
        if isinstance(data, list) and data:
            return data[-1]  # Most recent entry for that day
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass

    return None


def get_trend_metrics(days: int = 7) -> dict:
    """Get multi-day trend metrics for gradual degradation detection.

    Collects metrics across multiple days to identify trends that single-day
    comparison would miss (e.g., gradual LOC growth, increasing complexity).

    Args:
        days: Number of days to analyze (default 7, max 90).

    Returns:
        Dict with:
        - data_points: List of {date, metrics} for each available day
        - days_available: Count of days with data
        - trend: Dict with trend analysis (direction, change_pct) for key metrics

    REQUIRES: 1 <= days <= 90
    ENSURES: Returns dict with data_points list (may be empty)
    ENSURES: Returns dict with days_available >= 0
    ENSURES: Returns dict with trend analysis when >= 2 data points
    ENSURES: Never raises (returns empty results on error)
    """
    from datetime import datetime, timedelta

    from .constants import METRICS_DIR

    # Clamp days to valid range
    days = max(1, min(days, 90))

    data_points = []
    for i in range(days):
        target_date = datetime.now() - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        metrics_file = METRICS_DIR / f"{date_str}.json"

        if not metrics_file.exists():
            continue

        try:
            data = json.loads(metrics_file.read_text())
            if isinstance(data, list) and data:
                metrics = data[-1]
            elif isinstance(data, dict):
                metrics = data
            else:
                continue
            data_points.append({"date": date_str, "metrics": metrics})
        except (json.JSONDecodeError, OSError):
            continue

    # Sort chronologically (oldest first)
    data_points.sort(key=lambda x: x["date"])

    result = {
        "data_points": data_points,
        "days_available": len(data_points),
        "days_requested": days,
        "trend": {},
    }

    # Calculate trends if we have at least 2 data points
    if len(data_points) >= 2:
        oldest = data_points[0]["metrics"]
        newest = data_points[-1]["metrics"]

        # Analyze key metrics for trends
        # Format: (metric_key, trend_name, negative_when_increasing)
        trend_keys = [
            ("loc_total", "code", False),  # Total lines of code
            ("test_count", "tests", False),  # Test count
            ("issues_open", "issues", True),  # Open issues - increase is negative
            ("complexity_avg", "complexity", True),  # Avg complexity - increase is negative
        ]

        for metric_key, trend_name, is_negative_when_increasing in trend_keys:
            old_val = _extract_metric(oldest, metric_key)
            new_val = _extract_metric(newest, metric_key)

            if old_val is not None and new_val is not None and old_val > 0:
                change = new_val - old_val
                change_pct = (change / old_val) * 100
                direction = "increasing" if change > 0 else "decreasing" if change < 0 else "stable"

                # Check for sustained trend (consecutive days moving in same direction)
                sustained = _is_sustained_trend(
                    data_points, metric_key, direction, min_consecutive=3
                )

                result["trend"][trend_name] = {
                    "direction": direction,
                    "change": change,
                    "change_pct": round(change_pct, 1),
                    "start_value": old_val,
                    "end_value": new_val,
                    "sustained_increase": sustained and direction == "increasing",
                }

        # Add top-level sustained_negative_trend flag
        result["sustained_negative_trend"] = any(
            trend.get("sustained_increase", False) and is_neg
            for (_, trend_name, is_neg) in trend_keys
            if trend_name in result["trend"]
            for trend in [result["trend"][trend_name]]
        )

    return result


def _is_sustained_trend(
    data_points: list[dict], metric_key: str, direction: str, min_consecutive: int = 3
) -> bool:
    """Check if a metric shows sustained consecutive trend in given direction.

    REQUIRES: data_points is a list of {date, metrics} sorted chronologically
    REQUIRES: min_consecutive >= 2
    ENSURES: Returns True if at least min_consecutive consecutive days show same direction
    """
    if len(data_points) < min_consecutive or direction == "stable":
        return False

    # Extract values for each day
    values = []
    for dp in data_points:
        val = _extract_metric(dp["metrics"], metric_key)
        if val is not None:
            values.append(val)

    if len(values) < min_consecutive:
        return False

    # Count consecutive increases/decreases
    consecutive = 1
    for i in range(1, len(values)):
        if direction == "increasing" and values[i] > values[i - 1]:
            consecutive += 1
        elif direction == "decreasing" and values[i] < values[i - 1]:
            consecutive += 1
        else:
            consecutive = 1

        if consecutive >= min_consecutive:
            return True

    return False


def _extract_metric(metrics: dict, key: str) -> int | None:
    """Extract a metric value, handling nested structures.

    REQUIRES: metrics is a dict
    ENSURES: Returns int value or None if not found/invalid
    """
    if key == "loc_total":
        # Sum LOC across languages
        loc = metrics.get("loc", {})
        if isinstance(loc, dict):
            return sum(v for v in loc.values() if isinstance(v, (int, float)))
        return None
    if key == "test_count":
        tests = metrics.get("tests", {})
        if isinstance(tests, dict):
            return tests.get("count")
        return None
    if key == "issues_open":
        issues = metrics.get("issues", {})
        if isinstance(issues, dict):
            return issues.get("open")
        return None
    if key == "complexity_avg":
        complexity = metrics.get("complexity", {})
        if isinstance(complexity, dict):
            avg = complexity.get("avg_complexity")
            if isinstance(avg, (int, float)):
                return avg
        return None
    return metrics.get(key)


__all__ = [
    # Public functions
    "count_lines_by_type",
    "find_large_files",
    "find_forbidden_ci",
    "get_complexity_metrics",
    "get_previous_metrics",
    "get_trend_metrics",
    # Internal functions (exported for testing and internal use)
    "_resolve_root",
    "_relative_path",
    "_path_is_excluded",
    "_matches_exclude_pattern",
    "_extract_metric",
    "_is_sustained_trend",
]

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
pulse.py - Programmatic stats collection and threshold checking

CANONICAL SOURCE: ayates_dbx/ai_template
DO NOT EDIT in other repos - file issues to ai_template for changes.

Collects metrics, detects issues, writes flags for manager to act on.

Usage:
    ./ai_template_scripts/pulse.py              # Run once, show markdown output
    ./ai_template_scripts/pulse.py --broadcast  # Single line for org collection
    ./ai_template_scripts/pulse.py --watch      # Run continuously (silent)

Metrics collected:
    - LOC by language (Python, Rust, Go, C++, C, JS, TS)
    - Test count and recent results (from log_test.py)
    - Proof coverage (Kani, TLA+, Lean, SMT, NN verification)
    - Code quality (TODO count, test/code ratio, type coverage)
    - Issue counts and velocity (7-day open/close)
    - Git status (branch, commits, dirty state)
    - System resources (memory, disk, build artifacts)
    - Active AI sessions
    - Template consolidation debt (lines in .claude/ over target)

Flags set in .flags/ when thresholds exceeded:
    - large_files, crashes, blocked_issues, no_work, gh_error
    - orphaned_tests, test_failures, low_proof_coverage
    - memory_warning, memory_critical, disk_warning, disk_critical

Broadcast format (for org collection):
    repo|branch|head|loc:N|tests:N|proofs:N|issues:N|mem:N%|disk:N%|flags:...

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import argparse
import glob as glob_module
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import tomllib

# Directories
METRICS_DIR = Path("metrics")
FLAGS_DIR = Path(".flags")

# Directories to exclude from analysis (cloned reference code, vendored deps)
EXCLUDE_DIRS = ["reference", "vendor", "third_party", "external"]

# Pre-built exclusion patterns for find and grep
FIND_EXCLUDE = " ".join([f'-not -path "./{d}/*"' for d in EXCLUDE_DIRS])
GREP_EXCLUDE = "|".join([f"/{d}/" for d in EXCLUDE_DIRS])

# Thresholds (customize per project)
THRESHOLDS = {
    "max_file_lines": 500,
    "max_complexity": 15,
    "max_files_over_limit": 3,
    "stale_issue_days": 7,
    "memory_warning_percent": 80,  # Flag when memory usage exceeds this
    "memory_critical_percent": 90,  # Critical flag
    "disk_warning_percent": 80,  # Flag when disk usage exceeds this
    "disk_critical_percent": 90,  # Critical flag
}


def run_cmd(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run command and return (exit_code, stdout)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def run_cmd_with_retry(
    cmd: list[str], timeout: int = 60, retries: int = 2, retry_delay: float = 1.0
) -> tuple[int, str, str]:
    """Run command with retries. Returns (exit_code, stdout, stderr).

    Used for network-dependent commands like `gh` where transient failures are common.
    """
    last_stderr = ""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return result.returncode, result.stdout, result.stderr
            last_stderr = result.stderr
        except subprocess.TimeoutExpired:
            last_stderr = f"Command timed out after {timeout}s"
        except FileNotFoundError:
            return 1, "", "Command not found"

        if attempt < retries:
            time.sleep(retry_delay)

    return 1, "", last_stderr


def count_lines_by_type() -> dict[str, int]:
    """Count lines of code by file type. Tries tokei first, falls back to find+wc."""
    # Try tokei first (fastest - single pass, all languages)
    # Exclude reference/vendor directories
    tokei_cmd = ["tokei", "-o", "json"] + [
        arg for d in EXCLUDE_DIRS for arg in ["-e", d]
    ]
    code, output = run_cmd(tokei_cmd, timeout=60)
    if code == 0 and output.strip():
        try:
            data = json.loads(output)
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
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"""
for ext in .py .rs .go .cpp .c .js .ts; do
    (count=$(find . -name "*$ext" -type f -not -path './.git/*' {FIND_EXCLUDE} -print0 2>/dev/null | xargs -0 wc -l 2>/dev/null | tail -1 | awk '{{print $1}}')
     [ -n "$count" ] && [ "$count" -gt 0 ] && echo "$ext:$count") &
done
wait
""",
        ],
        timeout=60,
    )

    if code == 0 and output.strip():
        for line in output.strip().split("\n"):
            if ":" in line:
                ext, count_str = line.split(":", 1)
                ext = ext.strip()
                if ext in extensions:
                    try:
                        counts[extensions[ext]] = int(count_str.strip())
                    except ValueError:
                        pass

    return counts


def find_large_files(max_lines: int) -> list[dict]:
    """Find files exceeding line limit using find+wc (works everywhere)."""
    large = []
    extensions = [".py", ".rs", ".go", ".cpp", ".c", ".js", ".ts"]

    # Build find pattern for all extensions
    name_args = " -o ".join([f'-name "*{ext}"' for ext in extensions])

    # Single find+wc call, filter large files with awk
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f'find . \\( {name_args} \\) -type f -not -path "./.git/*" {FIND_EXCLUDE} -print0 2>/dev/null | xargs -0 wc -l 2>/dev/null | awk -v max={max_lines} \'$1>max && $2!="total" {{print $1, $2}}\'',
        ],
        timeout=60,
    )

    if code == 0 and output.strip():
        for line in output.strip().split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    lines = int(parts[0])
                    filepath = parts[1]
                    large.append({"file": filepath, "lines": lines})
                except ValueError:
                    pass

    return sorted(large, key=lambda x: -x["lines"])


def get_test_status() -> dict:
    """Get test counts (fast grep-based) and recent results from log_test.py.

    Supports mixed repos with both Python and Rust tests.
    Returns per-framework counts and total.
    """
    status = {
        "count": 0,
        "framework": "unknown",
        "orphaned_tests": [],
        "by_framework": {},  # Per-framework counts
    }

    # Fast test counting via grep (no compilation)
    # Count BOTH Python and Rust tests, don't stop after finding one

    # Python: count "def test_" patterns in standard locations
    pytest_count = 0
    code, output = run_cmd(
        ["bash", "-c", "grep -r 'def test_' tests/ test/ 2>/dev/null | wc -l"]
    )
    if code == 0 and output.strip():
        try:
            pytest_count = int(output.strip())
        except ValueError:
            pass

    # Rust: count #[test] attributes in tests/ AND src/ (inline tests are valid)
    cargo_count = 0
    code, output = run_cmd(
        ["bash", "-c", "grep -rE '#\\[test\\]' tests/ src/ crates/ 2>/dev/null | wc -l"]
    )
    if code == 0 and output.strip():
        try:
            cargo_count = int(output.strip())
        except ValueError:
            pass

    # Build status based on what we found
    if pytest_count > 0:
        status["by_framework"]["pytest"] = pytest_count
    if cargo_count > 0:
        status["by_framework"]["cargo"] = cargo_count

    total_count = pytest_count + cargo_count
    status["count"] = total_count

    # Determine framework label
    if pytest_count > 0 and cargo_count > 0:
        status["framework"] = "mixed"
    elif pytest_count > 0:
        status["framework"] = "pytest"
    elif cargo_count > 0:
        status["framework"] = "cargo"
    else:
        status["framework"] = "unknown"

    # Python: detect orphaned tests (def test_ outside any tests/ or test/ directory)
    # Use stricter pattern: must start line with optional whitespace + "def test_"
    # Exclude:
    #   - /tests/, /test/ (standard test directories)
    #   - __pycache__, .venv/, venv/, env/, site-packages/ (virtual envs)
    #   - reference/ (cloned external repos for reference)
    #   - scripts/ (utility scripts, not test files)
    #   - examples/ (example code, not project tests)
    if (
        pytest_count > 0
        or Path("pytest.ini").exists()
        or Path("pyproject.toml").exists()
    ):
        # Build exclusion pattern from EXCLUDE_DIRS + standard exclusions
        exclude_pattern = "|".join(
            [f"/{d}/" for d in EXCLUDE_DIRS]
            + [
                "/tests/",
                "/test/",
                "__pycache__",
                "/.venv/",
                "/venv/",
                "/env/",
                "/site-packages/",
                "/scripts/",
                "/examples/",
            ]
        )
        code, output = run_cmd(
            [
                "bash",
                "-c",
                f"grep -rlE '^[[:space:]]*def test_' . --include='*.py' 2>/dev/null | "
                f"grep -v -E '{exclude_pattern}'",
            ]
        )
        if code == 0 and output.strip():
            orphaned = [f.strip() for f in output.strip().split("\n") if f.strip()]
            if orphaned:
                status["orphaned_tests"] = orphaned[:10]  # Limit to 10

    # Note: Rust inline tests in src/ with #[cfg(test)] are valid, not orphaned.
    # We count them above but don't flag them as orphaned.

    # Read recent results from log_test.py
    test_log = Path("logs/tests.jsonl")
    if test_log.exists():
        try:
            lines = test_log.read_text().strip().split("\n")
            # Get last 24h of results
            now = datetime.now()
            recent_runs = []
            for line in lines[-100:]:  # Check last 100 entries
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("event") == "end":
                    ts = entry.get("timestamp", "")
                    try:
                        entry_time = datetime.fromisoformat(ts)
                        if (now - entry_time).total_seconds() < 86400:
                            recent_runs.append(entry)
                    except ValueError:
                        pass

            if recent_runs:
                passed = sum(1 for r in recent_runs if r.get("exit_code") == 0)
                failed = len(recent_runs) - passed
                total_duration = sum(r.get("duration_s", 0) or 0 for r in recent_runs)
                status["recent_24h"] = {
                    "runs": len(recent_runs),
                    "passed": passed,
                    "failed": failed,
                    "total_duration_s": round(total_duration, 1),
                }
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    return status


def get_issue_counts() -> dict:
    """Get GitHub issue counts by state (single API call with retry).

    Returns dict with counts + "_error" key if API failed.
    The "_error" key allows callers to distinguish between:
    - Actually having 0 issues (no error)
    - Failed to fetch issues (error present)
    """
    counts = {"open": 0, "in_progress": 0, "blocked": 0, "closed": 0}

    # Single call for all issues with retry (network can be flaky)
    code, output, stderr = run_cmd_with_retry(
        [
            "gh",
            "issue",
            "list",
            "--state",
            "all",
            "--json",
            "state,labels",
            "--limit",
            "500",
        ],
        timeout=60,
        retries=2,
    )

    if code != 0:
        # API failed - mark error so callers don't assume 0 issues
        error_msg = stderr.strip() if stderr.strip() else "gh command failed"
        counts["_error"] = error_msg
        return counts

    if not output.strip():
        # Empty output but success - could be no issues or API issue
        # Be conservative: mark as potential error
        counts["_error"] = "Empty response from gh"
        return counts

    try:
        issues = json.loads(output)
        for issue in issues:
            state = issue.get("state", "").upper()
            labels = [lbl.get("name", "") for lbl in issue.get("labels", [])]

            if state == "CLOSED":
                counts["closed"] += 1
            elif "blocked" in labels:
                counts["blocked"] += 1
            elif "in-progress" in labels:
                counts["in_progress"] += 1
            else:
                counts["open"] += 1
    except json.JSONDecodeError as e:
        counts["_error"] = f"JSON parse error: {e}"

    return counts


def _get_workspace_member_dirs(cargo_toml_path: Path) -> list[str]:
    """Get workspace member directories from Cargo.toml.

    Parses [workspace] members array and expands glob patterns.
    Returns list of existing directories containing src/ or tests/.
    """
    try:
        content = cargo_toml_path.read_text()
        cargo = tomllib.loads(content)
    except (OSError, tomllib.TOMLDecodeError):
        return []

    members = cargo.get("workspace", {}).get("members", [])
    if not members:
        return []

    member_dirs = []
    for pattern in members:
        # Expand glob patterns (e.g., "crates/*", "libs/*")
        matches = glob_module.glob(pattern)
        if matches:
            member_dirs.extend(matches)
        elif Path(pattern).exists():
            # Literal path exists
            member_dirs.append(pattern)

    # For each member, add src/ and tests/ subdirs if they exist
    search_dirs = []
    for member in member_dirs:
        member_path = Path(member)
        if member_path.is_dir():
            for sub in ["src", "tests"]:
                sub_path = member_path / sub
                if sub_path.exists():
                    search_dirs.append(str(sub_path))

    return search_dirs


def get_proof_coverage() -> dict:
    """Detect formal verification coverage across proof systems.

    Systems detected:
    - Kani: Rust bounded model checking (#[kani::proof], requires/ensures/modifies)
    - TLA+: Distributed system specs (.tla files)
    - Lean: Theorem proving (.lean files)
    - SMT: SAT/SMT solving (.smt2 files, z4:: usage)
    - NN verification: Neural network proofs (.onnx + vnnlib)
    """
    coverage = {}

    # Kani (Rust bounded model checking)
    cargo_toml = Path("Cargo.toml")
    if cargo_toml.exists():
        # Build search paths: root src/tests, workspace member src/tests
        search_dirs = [d for d in ["src", "tests"] if Path(d).exists()]

        # Add workspace members (parses [workspace] members and expands globs)
        search_dirs.extend(_get_workspace_member_dirs(cargo_toml))

        # Legacy fallback: include crates/ even if not in workspace.members
        crates_dir = Path("crates")
        if crates_dir.exists():
            for crate_dir in crates_dir.iterdir():
                if crate_dir.is_dir():
                    for sub in ["src", "tests"]:
                        sub_path = str(crate_dir / sub)
                        if Path(sub_path).exists() and sub_path not in search_dirs:
                            search_dirs.append(sub_path)

        if search_dirs:
            search_path = " ".join(search_dirs)

            # Count kani proof harnesses
            code, output = run_cmd(
                [
                    "bash",
                    "-c",
                    f"grep -r '#\\[kani::proof\\]' {search_path} 2>/dev/null | wc -l",
                ]
            )
            proofs = int(output.strip()) if code == 0 and output.strip() else 0

            # Count kani contract attributes (requires/ensures/modifies)
            code, output = run_cmd(
                [
                    "bash",
                    "-c",
                    f"grep -rE '#\\[kani::(requires|ensures|modifies)' {search_path} 2>/dev/null | wc -l",
                ]
            )
            contract_attrs = int(output.strip()) if code == 0 and output.strip() else 0

            # Count unique functions with contracts (for accurate coverage_pct)
            # A function with both requires and ensures should count as 1, not 2
            # Note: no ^ anchor because grep -r prefixes filename to each line
            code, output = run_cmd(
                [
                    "bash",
                    "-c",
                    f"grep -rE -A1 '#\\[kani::(requires|ensures|modifies)' {search_path} 2>/dev/null "
                    "| grep -E '(pub\\s+)?(async\\s+)?fn\\s+' "
                    "| sed -E 's/.*fn[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*).*/\\1/' "
                    "| sort -u | wc -l",
                ]
            )
            contracted_fns = int(output.strip()) if code == 0 and output.strip() else 0

            # Count total functions (approximate) - only in src dirs for accuracy
            src_dirs = " ".join(
                d for d in search_dirs if d.endswith("src") or "/src" in d
            )
            if src_dirs:
                code, output = run_cmd(
                    [
                        "bash",
                        "-c",
                        f"grep -rE '^\\s*(pub )?fn ' {src_dirs} 2>/dev/null | wc -l",
                    ]
                )
                total_fns = int(output.strip()) if code == 0 and output.strip() else 0
            else:
                total_fns = 0

            if proofs > 0 or contract_attrs > 0:
                coverage["kani"] = {
                    "proofs": proofs,
                    "contract_attrs": contract_attrs,  # Total contract attributes
                    "contracted_fns": contracted_fns,  # Unique functions with contracts
                    "total_functions": total_fns,
                    "coverage_pct": round(100 * contracted_fns / total_fns, 1)
                    if total_fns > 0
                    else 0,
                }

    # TLA+ specifications
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"find . -name '*.tla' -not -path './.git/*' {FIND_EXCLUDE} 2>/dev/null",
        ]
    )
    if code == 0 and output.strip():
        tla_files = [f.strip() for f in output.strip().split("\n") if f.strip()]
        if tla_files:
            specs = invariants = 0
            for f in tla_files[:20]:  # Limit scanning
                try:
                    content = Path(f).read_text()
                    specs += content.count("SPECIFICATION")
                    invariants += content.count("INVARIANT") + content.count("PROPERTY")
                except (FileNotFoundError, UnicodeDecodeError):
                    pass
            coverage["tla2"] = {
                "files": len(tla_files),
                "specs": specs,
                "invariants": invariants,
            }

    # Lean proofs
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"find . -name '*.lean' -not -path './.git/*' {FIND_EXCLUDE} 2>/dev/null | head -100",
        ]
    )
    if code == 0 and output.strip():
        lean_files = [f.strip() for f in output.strip().split("\n") if f.strip()]
        if lean_files:
            theorems = 0
            for f in lean_files[:30]:  # Limit scanning
                try:
                    content = Path(f).read_text()
                    theorems += content.count("theorem ") + content.count("lemma ")
                except (FileNotFoundError, UnicodeDecodeError):
                    pass
            coverage["lean"] = {"files": len(lean_files), "theorems": theorems}

    # SMT-LIB files (z4/Z3)
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"find . -name '*.smt2' -not -path './.git/*' {FIND_EXCLUDE} 2>/dev/null | wc -l",
        ]
    )
    smt_count = int(output.strip()) if code == 0 and output.strip() else 0

    # Also check for z4 usage in Rust
    code, output = run_cmd(["bash", "-c", "grep -r 'z4::' src/ 2>/dev/null | wc -l"])
    z4_usage = int(output.strip()) if code == 0 and output.strip() else 0

    if smt_count > 0 or z4_usage > 0:
        coverage["smt"] = {"smt2_files": smt_count, "z4_usage": z4_usage}

    # Neural network verification (gamma-crown style)
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"find . -name '*.onnx' -not -path './.git/*' {FIND_EXCLUDE} 2>/dev/null | wc -l",
        ]
    )
    onnx_count = int(output.strip()) if code == 0 and output.strip() else 0

    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"find . -name '*.vnnlib' -not -path './.git/*' {FIND_EXCLUDE} 2>/dev/null | wc -l",
        ]
    )
    vnnlib_count = int(output.strip()) if code == 0 and output.strip() else 0

    if onnx_count > 0 or vnnlib_count > 0:
        coverage["nn_verification"] = {
            "onnx_models": onnx_count,
            "vnnlib_specs": vnnlib_count,
        }

    return coverage


def get_git_status() -> dict:
    """Get git repository status."""
    status = {}

    # Check if dirty (uncommitted changes)
    code, output = run_cmd(["git", "status", "--porcelain"])
    if code == 0:
        lines = [line for line in output.strip().split("\n") if line.strip()]
        status["dirty"] = len(lines) > 0
        status["uncommitted_files"] = len(lines)

    # Get current branch
    code, output = run_cmd(["git", "branch", "--show-current"])
    if code == 0 and output.strip():
        status["branch"] = output.strip()

    # Get last commit hash
    code, output = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    if code == 0 and output.strip():
        status["head"] = output.strip()

    # Commit velocity (last 7 days)
    code, output = run_cmd(["git", "rev-list", "--count", "--since=7.days", "HEAD"])
    if code == 0 and output.strip():
        try:
            status["commits_7d"] = int(output.strip())
        except ValueError:
            pass

    return status


def get_code_quality() -> dict:
    """Get code quality metrics that should improve over time."""
    quality = {}

    # Count TODO/FIXME comments (tech debt indicator)
    code, output = run_cmd(
        [
            "bash",
            "-c",
            f"grep -rE '(TODO|FIXME|XXX|HACK):?' . --include='*.py' --include='*.rs' --include='*.go' --include='*.js' --include='*.ts' 2>/dev/null | grep -v node_modules | grep -v target | grep -vE '{GREP_EXCLUDE}' | wc -l",
        ]
    )
    if code == 0 and output.strip():
        try:
            quality["todo_count"] = int(output.strip())
        except ValueError:
            pass

    # Test/code ratio
    code, output = run_cmd(
        [
            "bash",
            "-c",
            "find tests/ test/ -name '*.py' -o -name '*.rs' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}'",
        ]
    )
    test_lines = int(output.strip()) if code == 0 and output.strip() else 0

    code, output = run_cmd(
        [
            "bash",
            "-c",
            "find src/ lib/ -name '*.py' -o -name '*.rs' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}'",
        ]
    )
    src_lines = int(output.strip()) if code == 0 and output.strip() else 0

    if src_lines > 0:
        quality["test_code_ratio"] = round(test_lines / src_lines, 2)

    # Python type coverage (count typed functions vs total)
    if Path("pyproject.toml").exists() or list(Path(".").glob("*.py")):
        # Count functions with type annotations
        code, output = run_cmd(
            [
                "bash",
                "-c",
                f"grep -rE 'def [a-z_]+\\([^)]*:' . --include='*.py' 2>/dev/null | grep -v __pycache__ | grep -vE '{GREP_EXCLUDE}' | wc -l",
            ]
        )
        typed_fns = int(output.strip()) if code == 0 and output.strip() else 0

        # Count all functions
        code, output = run_cmd(
            [
                "bash",
                "-c",
                f"grep -rE '^\\s*def [a-z_]+\\(' . --include='*.py' 2>/dev/null | grep -v __pycache__ | grep -vE '{GREP_EXCLUDE}' | wc -l",
            ]
        )
        total_fns = int(output.strip()) if code == 0 and output.strip() else 0

        if total_fns > 0:
            quality["python_type_coverage_pct"] = round(100 * typed_fns / total_fns, 1)

    # Documentation coverage (Python docstrings)
    if Path("pyproject.toml").exists() or list(Path(".").glob("*.py")):
        # Count functions with docstrings (triple quotes after def line)
        code, output = run_cmd(
            [
                "bash",
                "-c",
                f"grep -rPzo 'def [^:]+:\\s*\\n\\s*\"\"\"' . --include='*.py' 2>/dev/null | grep -vE '{GREP_EXCLUDE}' | grep -c 'def ' || echo 0",
            ]
        )
        if code == 0 and output.strip():
            try:
                # This is approximate - pcregrep would be better
                pass  # Skip for now - complex to detect reliably
            except ValueError:
                pass

    return quality


def get_issue_velocity() -> dict:
    """Get issue velocity metrics (improvement over time)."""
    velocity = {}

    # Issues opened in last 7 days
    code, output = run_cmd(
        [
            "bash",
            "-c",
            "gh issue list --state all --json createdAt --limit 200 2>/dev/null",
        ]
    )
    if code == 0 and output.strip():
        try:
            issues = json.loads(output)
            now = datetime.now()
            opened_7d = 0
            for issue in issues:
                created = issue.get("createdAt", "")
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if (now - created_dt.replace(tzinfo=None)).days <= 7:
                        opened_7d += 1
                except (ValueError, TypeError):
                    pass
            velocity["opened_7d"] = opened_7d
        except json.JSONDecodeError:
            pass

    # Issues closed in last 7 days
    code, output = run_cmd(
        [
            "bash",
            "-c",
            "gh issue list --state closed --json closedAt --limit 200 2>/dev/null",
        ]
    )
    if code == 0 and output.strip():
        try:
            issues = json.loads(output)
            now = datetime.now()
            closed_7d = 0
            for issue in issues:
                closed = issue.get("closedAt", "")
                try:
                    closed_dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
                    if (now - closed_dt.replace(tzinfo=None)).days <= 7:
                        closed_7d += 1
                except (ValueError, TypeError):
                    pass
            velocity["closed_7d"] = closed_7d
        except json.JSONDecodeError:
            pass

    # Net velocity
    if "opened_7d" in velocity and "closed_7d" in velocity:
        velocity["net_7d"] = velocity["closed_7d"] - velocity["opened_7d"]

    return velocity


def count_template_lines() -> dict:
    """Count total template lines that sync to all repos.

    Template files are in .claude/rules/ and .claude/roles/ - these get synced
    to all projects in the org. Line count tracks consolidation effort.

    Returns dict with per-file counts, total, and target from VISION.md.
    """
    template_patterns = [
        ".claude/rules/ai_template.md",
        ".claude/rules/org_chart.md",
        ".claude/roles/*.md",
    ]

    counts = {}
    total = 0

    for pattern in template_patterns:
        files = list(Path(".").glob(pattern))
        for f in files:
            try:
                lines = len(f.read_text().splitlines())
                counts[str(f)] = lines
                total += lines
            except (FileNotFoundError, UnicodeDecodeError):
                pass

    return {
        "files": counts,
        "total": total,
        "target": 1200,  # From VISION.md consolidation phase goal
    }


def get_consolidation_debt() -> dict:
    """Calculate consolidation debt (lines over target).

    Returns dict with current total, target, and debt.
    Debt = max(0, actual - target) so negative values aren't shown.
    """
    template = count_template_lines()
    actual = template["total"]
    target = template["target"]
    debt = max(0, actual - target)

    return {
        "total": actual,
        "target": target,
        "debt": debt,
        "files": template["files"],
    }


def get_active_sessions() -> int:
    """Count active AI sessions (by .pid_* files or looper processes)."""
    # Count .pid_* files
    pid_files = list(Path(".").glob(".pid_*"))
    active = 0
    for pf in pid_files:
        try:
            pid = int(pf.read_text().strip())
            # Check if process is running
            code, _ = run_cmd(["kill", "-0", str(pid)])
            if code == 0:
                active += 1
        except (ValueError, FileNotFoundError):
            pass
    return active


def get_recent_crashes() -> int:
    """Count crashes in last 24 hours from crashes.log."""
    crash_log = Path("worker_logs/crashes.log")
    if not crash_log.exists():
        return 0

    count = 0
    now = datetime.now()
    try:
        for line in crash_log.read_text().splitlines():
            # Format: [2026-01-09 12:00:00] ...
            if line.startswith("["):
                try:
                    timestamp_str = line[1:20]
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    if (now - timestamp).total_seconds() < 86400:
                        count += 1
                except ValueError:
                    pass
    except Exception:
        pass
    return count


def get_system_resources() -> dict:
    """Get system resource usage (memory, disk, build artifacts) for trend tracking."""
    resources = {}

    # Memory usage - try macOS first, then Linux
    code, output = run_cmd(["bash", "-c", "vm_stat 2>/dev/null | head -10"])
    if code == 0 and output.strip() and "page size" in output:
        # Parse macOS vm_stat output
        page_size = 16384  # Default, usually 16KB on Apple Silicon
        pages_free = 0
        pages_active = 0
        pages_inactive = 0
        pages_wired = 0
        pages_compressed = 0

        for line in output.strip().split("\n"):
            if "page size of" in line:
                try:
                    page_size = int(line.split()[-2])
                except (ValueError, IndexError):
                    pass
            elif "Pages free:" in line:
                try:
                    pages_free = int(line.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass
            elif "Pages active:" in line:
                try:
                    pages_active = int(line.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass
            elif "Pages inactive:" in line:
                try:
                    pages_inactive = int(line.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass
            elif "Pages wired down:" in line:
                try:
                    pages_wired = int(line.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass
            elif "Pages occupied by compressor:" in line:
                try:
                    pages_compressed = int(line.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass

        used_bytes = (pages_active + pages_wired + pages_compressed) * page_size
        free_bytes = (pages_free + pages_inactive) * page_size
        total_bytes = used_bytes + free_bytes

        if total_bytes > 0:
            resources["memory"] = {
                "used_gb": round(used_bytes / (1024**3), 1),
                "free_gb": round(free_bytes / (1024**3), 1),
                "total_gb": round(total_bytes / (1024**3), 1),
                "percent_used": round(100 * used_bytes / total_bytes, 1),
            }
    else:
        # Try Linux /proc/meminfo
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
                resources["memory"] = {
                    "used_gb": round(used_bytes / (1024**3), 1),
                    "free_gb": round(mem_available / (1024**3), 1),
                    "total_gb": round(mem_total / (1024**3), 1),
                    "percent_used": round(100 * used_bytes / mem_total, 1),
                }
        except (FileNotFoundError, ValueError, IndexError):
            pass

    # Disk usage for current volume
    code, output = run_cmd(["df", "-h", "."])
    if code == 0 and output.strip():
        lines = output.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                try:
                    # df -h output: Filesystem Size Used Avail Capacity
                    resources["disk"] = {
                        "total": parts[1],
                        "used": parts[2],
                        "available": parts[3],
                        "percent_used": parts[4],
                    }
                except (ValueError, IndexError):
                    pass

    # Build artifact sizes (target/, build/, node_modules/)
    artifact_dirs = ["target", "build", "node_modules", ".build"]
    artifacts = {}
    for dirname in artifact_dirs:
        if Path(dirname).is_dir():
            code, output = run_cmd(["du", "-sh", dirname], timeout=10)
            if code == 0 and output.strip():
                size = output.split()[0]
                artifacts[dirname] = size
    if artifacts:
        resources["build_artifacts"] = artifacts

    return resources


def get_repo_name() -> str:
    """Get current repo name from git remote."""
    code, output = run_cmd(["git", "remote", "get-url", "origin"])
    if code == 0 and output.strip():
        # Extract repo name from URL (handles both HTTPS and SSH)
        url = output.strip()
        url = url.removesuffix(".git")
        return url.split("/")[-1]
    # Fallback to directory name
    return Path.cwd().name


def collect_metrics() -> dict:
    """Collect all metrics."""
    return {
        "timestamp": datetime.now().isoformat(),
        "repo": get_repo_name(),
        "loc": count_lines_by_type(),
        "large_files": find_large_files(THRESHOLDS["max_file_lines"]),
        "tests": get_test_status(),
        "proofs": get_proof_coverage(),
        "issues": get_issue_counts(),
        "crashes_24h": get_recent_crashes(),
        "system": get_system_resources(),
        "git": get_git_status(),
        "active_sessions": get_active_sessions(),
        "quality": get_code_quality(),
        "velocity": get_issue_velocity(),
        "consolidation": get_consolidation_debt(),
    }


def check_thresholds(metrics: dict) -> list[str]:
    """Check thresholds, return list of triggered flags."""
    flags = []

    # Large files
    if len(metrics.get("large_files", [])) >= THRESHOLDS["max_files_over_limit"]:
        flags.append("large_files")

    # Crashes
    if metrics.get("crashes_24h", 0) > 0:
        flags.append("crashes")

    # Blocked issues
    issues = metrics.get("issues", {})
    if issues.get("blocked", 0) > 0:
        flags.append("blocked_issues")

    # GitHub API error (couldn't fetch issues - don't assume no_work)
    if issues.get("_error"):
        flags.append("gh_error")
    # No open issues (might need roadmap refresh) - only if we successfully fetched
    elif issues.get("open", 0) == 0 and issues.get("in_progress", 0) == 0:
        flags.append("no_work")

    # Orphaned tests (code smell)
    tests = metrics.get("tests", {})
    if tests.get("orphaned_tests"):
        flags.append("orphaned_tests")

    # Test failures in last 24h
    recent = tests.get("recent_24h", {})
    if recent.get("failed", 0) > 0:
        flags.append("test_failures")

    # Low proof coverage (Rust project with Kani but <10% coverage)
    # Don't flag if project has proof harnesses - those provide verification too
    proofs = metrics.get("proofs", {})
    kani = proofs.get("kani", {})
    has_proofs = kani.get("proofs", 0) > 0
    if (
        kani.get("total_functions", 0) > 10
        and kani.get("coverage_pct", 0) < 10
        and not has_proofs
    ):
        flags.append("low_proof_coverage")

    # System resource checks
    system = metrics.get("system", {})

    # Memory usage
    memory = system.get("memory", {})
    mem_percent = memory.get("percent_used", 0)
    if mem_percent >= THRESHOLDS["memory_critical_percent"]:
        flags.append("memory_critical")
    elif mem_percent >= THRESHOLDS["memory_warning_percent"]:
        flags.append("memory_warning")

    # Disk usage
    disk = system.get("disk", {})
    disk_percent_str = disk.get("percent_used", "0%")
    try:
        disk_percent = int(disk_percent_str.rstrip("%"))
        if disk_percent >= THRESHOLDS["disk_critical_percent"]:
            flags.append("disk_critical")
        elif disk_percent >= THRESHOLDS["disk_warning_percent"]:
            flags.append("disk_warning")
    except ValueError:
        pass

    return flags


def write_metrics(metrics: dict):
    """Write metrics to file."""
    METRICS_DIR.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    metrics_file = METRICS_DIR / f"{date_str}.json"

    # Append to daily file
    existing = []
    if metrics_file.exists():
        try:
            existing = json.loads(metrics_file.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []

    existing.append(metrics)
    metrics_file.write_text(json.dumps(existing, indent=2))

    # Also write latest.json for easy access
    (METRICS_DIR / "latest.json").write_text(json.dumps(metrics, indent=2))


def write_flags(flags: list[str]):
    """Write flag files."""
    FLAGS_DIR.mkdir(exist_ok=True)

    # Clear old flags
    for f in FLAGS_DIR.glob("*"):
        f.unlink()

    # Write new flags
    for flag in flags:
        (FLAGS_DIR / flag).write_text(datetime.now().isoformat())


def pulse_once(quiet: bool = False):
    """Run pulse once. Set quiet=True for watch mode (no output)."""
    metrics = collect_metrics()
    flags = check_thresholds(metrics)

    write_metrics(metrics)
    write_flags(flags)

    if quiet:
        return

    # Output as readable markdown-style
    ts = datetime.now().strftime("%H:%M:%S")
    git = metrics.get("git", {})
    dirty = "dirty" if git.get("dirty") else "clean"

    print(f"## Pulse {ts}")
    print(f"**Git:** {git.get('branch', '?')}@{git.get('head', '?')} ({dirty})")
    print()

    # Code
    loc = metrics.get("loc", {})
    if loc:
        loc_str = ", ".join(f"{k}:{v}" for k, v in loc.items())
        print(f"**LOC:** {loc_str}")

    large = metrics.get("large_files", [])
    print(f"**Large files:** {len(large)}")

    # Consolidation debt (only in ai_template repo)
    consolidation = metrics.get("consolidation", {})
    if consolidation.get("total", 0) > 0:
        total = consolidation.get("total", 0)
        target = consolidation.get("target", 0)
        debt = consolidation.get("debt", 0)
        print(f"**Template Lines:** {total} (target: {target}, debt: {debt})")

    # Tests
    tests = metrics.get("tests", {})
    by_framework = tests.get("by_framework", {})
    if len(by_framework) > 1:
        # Mixed frameworks - show breakdown
        breakdown = ", ".join(f"{fw}:{n}" for fw, n in sorted(by_framework.items()))
        print(f"**Tests:** {tests.get('count', 0)} (mixed: {breakdown})")
    else:
        print(
            f"**Tests:** {tests.get('count', 0)} ({tests.get('framework', 'unknown')})"
        )
    recent = tests.get("recent_24h", {})
    if recent:
        print(
            f"**Test runs (24h):** {recent.get('runs', 0)} runs, {recent.get('passed', 0)} passed, {recent.get('failed', 0)} failed"
        )
    orphaned = tests.get("orphaned_tests", [])
    if orphaned:
        print(f"**Orphaned tests:** {', '.join(orphaned[:5])}")

    # Issues
    issues = metrics.get("issues", {})
    if issues.get("_error"):
        print(f"**Issues:** ⚠️ {issues.get('_error')}")
    else:
        print(
            f"**Issues:** {issues.get('open', 0)} open, {issues.get('in_progress', 0)} in-progress, {issues.get('blocked', 0)} blocked"
        )

    # System
    system = metrics.get("system", {})
    memory = system.get("memory", {})
    disk = system.get("disk", {})
    artifacts = system.get("build_artifacts", {})

    if memory:
        print(
            f"**Memory:** {memory.get('percent_used', '?')}% ({memory.get('used_gb', '?')}GB / {memory.get('total_gb', '?')}GB)"
        )
    if disk:
        print(
            f"**Disk:** {disk.get('percent_used', '?')} ({disk.get('used', '?')} / {disk.get('total', '?')})"
        )
    if artifacts:
        artifact_str = ", ".join(f"{k}={v}" for k, v in artifacts.items())
        print(f"**Build artifacts:** {artifact_str}")

    # Sessions
    active = metrics.get("active_sessions", 0)
    if active > 0:
        print(f"**Active AI sessions:** {active}")

    # Crashes
    crashes = metrics.get("crashes_24h", 0)
    if crashes > 0:
        print(f"**Crashes (24h):** {crashes}")

    # Flags
    print()
    if flags:
        print(f"**Flags:** {', '.join(flags)}")
    else:
        print("**Flags:** none")


def metrics_to_broadcast(metrics: dict) -> str:
    """Convert metrics to single-line broadcast format for org collection.

    Format: repo|branch|head|loc:N|tests:N|proofs:N|issues:N|mem:N%|disk:N%|flags:...
    """
    repo = metrics.get("repo", "unknown")
    git = metrics.get("git", {})
    branch = git.get("branch", "?")
    head = git.get("head", "?")

    loc_total = sum(metrics.get("loc", {}).values())
    tests = metrics.get("tests", {}).get("count", 0)

    # Sum all proof types
    proofs = metrics.get("proofs", {})
    proof_count = sum(
        [
            proofs.get("kani", {}).get("proofs", 0),
            proofs.get("tla2", {}).get("specs", 0),
            proofs.get("lean", {}).get("theorems", 0),
        ]
    )

    issues = metrics.get("issues", {}).get("open", 0)
    mem = metrics.get("system", {}).get("memory", {}).get("percent_used", 0)
    disk_str = metrics.get("system", {}).get("disk", {}).get("percent_used", "0%")

    flags = check_thresholds(metrics)
    flags_str = ",".join(flags) if flags else "none"

    return f"{repo}|{branch}|{head}|loc:{loc_total}|tests:{tests}|proofs:{proof_count}|issues:{issues}|mem:{mem}%|disk:{disk_str}|flags:{flags_str}"


def pulse_watch(interval: int = 300):
    """Run pulse continuously, writing metrics silently."""
    while True:
        metrics = collect_metrics()
        flags = check_thresholds(metrics)
        write_metrics(metrics)
        write_flags(flags)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Collect project metrics and detect issues"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously (silent, writes to metrics/)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Watch interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--broadcast", action="store_true", help="Output single line for org collection"
    )
    args = parser.parse_args()

    if args.broadcast:
        metrics = collect_metrics()
        write_metrics(metrics)
        write_flags(check_thresholds(metrics))
        print(metrics_to_broadcast(metrics))
    elif args.watch:
        pulse_watch(interval=args.interval)
    else:
        pulse_once()


if __name__ == "__main__":
    main()

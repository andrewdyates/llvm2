#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Test and proof coverage metrics collection for pulse module.

Functions for counting tests, detecting orphaned tests,
and measuring formal verification coverage.

Part of #404: pulse.py module split
"""

import glob as glob_module
import json
import shlex
import tomllib
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

from .config import SKIP_ORPHANED_TESTS
from .constants import (
    EXCLUDE_DIRS,
    EXCLUDE_GLOB_PATTERNS,
    FIND_PRUNE,
    GREP_EXCLUDE_DIRS,
)


def _resolve_root(repo_root: Path | None) -> Path:
    """Return repo root or current directory when unset."""
    return repo_root if repo_root is not None else Path(".")


def _parse_count_from_result(result) -> int:
    """Parse integer count from command result, returning 0 on bad output."""
    if not result.ok:
        return 0
    text = result.stdout.strip()
    if not text:
        return 0
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def _relative_path(path: Path, root: Path) -> str:
    """Return path relative to root when possible, else stringified path."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _path_is_excluded(path: Path) -> bool:
    """Return True if path includes excluded directories."""
    import fnmatch

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
    - Simple directory names (e.g., "tests") - matches if filepath starts with dir/ or contains /dir/
    - Glob patterns with ** (e.g., "**/tests/**") - uses glob-style matching
    """
    import fnmatch

    for pattern in patterns:
        if "**" in pattern:
            # Glob-style pattern
            if fnmatch.fnmatch(filepath, pattern):
                return True
        else:
            # Simple directory name
            if filepath.startswith(f"{pattern}/") or f"/{pattern}/" in filepath:
                return True
    return False


def _count_pattern_in_paths(
    root: Path,
    rel_dirs: list[str],
    file_glob: str,
    needle: str,
) -> int:
    """Count occurrences of needle in files under rel_dirs."""
    count = 0
    for rel_dir in rel_dirs:
        dir_path = root / rel_dir
        if not dir_path.is_dir():
            continue
        for path in dir_path.rglob(file_glob):
            if _path_is_excluded(path):
                continue
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        if needle in line:
                            count += line.count(needle)
            except OSError:
                continue
    return count


def _count_tests_by_framework(
    repo_root: Path | None = None,
    use_git: bool = True,
) -> tuple[int, int, dict[str, int]]:
    """Count tests by framework using fast grep.

    Returns:
        Tuple of (pytest_count, cargo_count, by_framework_dict).
    """
    by_framework: dict[str, int] = {}
    root = _resolve_root(repo_root)
    pytest_count = 0
    cargo_count = 0

    if use_git:
        # Python: count "def test_" patterns in standard locations
        # Use git grep for speed (#1035) - respects .gitignore, much faster than grep
        # Timeout prevents hang on very large repos (#956)
        result = run_cmd(
            [
                "bash",
                "-c",
                'git grep -c "def test_" -- "tests/*.py" "tests/**/*.py" '
                '"test/*.py" "test/**/*.py" 2>/dev/null | '
                "cut -d: -f2 | awk '{s+=$1} END {print s+0}'",
            ],
            timeout=30,
            cwd=root,
        )
        if result.ok and result.stdout.strip():
            try:
                pytest_count = int(result.stdout.strip())
            except ValueError:
                pass

        # Rust: count #[test] attributes in tests/ AND src/ (inline tests are valid)
        # Use git grep for speed on large codebases (#1035)
        # Timeout prevents hang on large repos (#956)
        result = run_cmd(
            [
                "bash",
                "-c",
                r'git grep -c "#\[test\]" -- "tests/*.rs" "tests/**/*.rs" '
                '"src/*.rs" "src/**/*.rs" "crates/**/*.rs" 2>/dev/null | '
                "cut -d: -f2 | awk '{s+=$1} END {print s+0}'",
            ],
            timeout=30,
            cwd=root,
        )
        if result.ok and result.stdout.strip():
            try:
                cargo_count = int(result.stdout.strip())
            except ValueError:
                pass
    else:
        pytest_count = _count_pattern_in_paths(
            root=root,
            rel_dirs=["tests", "test"],
            file_glob="*.py",
            needle="def test_",
        )
        cargo_count = _count_pattern_in_paths(
            root=root,
            rel_dirs=["tests", "src", "crates"],
            file_glob="*.rs",
            needle="#[test]",
        )

    if pytest_count > 0:
        by_framework["pytest"] = pytest_count
    if cargo_count > 0:
        by_framework["cargo"] = cargo_count

    return pytest_count, cargo_count, by_framework


def _load_pulse_ignore(repo_root: Path | None = None) -> list[str]:
    """Load exclusion patterns from .pulse_ignore file (#982).

    Returns:
        List of pathspec patterns to exclude from orphan test detection.
        Lines starting with # are comments. Empty lines are skipped.
        Use **/ glob syntax to match directories at any depth (#1238).

    Example .pulse_ignore:
        # Project-specific exclusions
        **/benchmarks/**  # VNN-COMP integration tests at any depth
        **/integration_tests/**
    """
    root = _resolve_root(repo_root)
    ignore_file = root / ".pulse_ignore"
    if not ignore_file.exists():
        return []

    patterns = []
    for line in ignore_file.read_text().splitlines():
        line = line.split("#", 1)[0].strip()  # Remove comments
        if line:
            patterns.append(line)
    return patterns


def _detect_orphaned_python_tests(
    has_pytest: bool,
    repo_root: Path | None = None,
    use_git: bool = True,
) -> list[str]:
    """Find Python test functions outside standard test directories.

    Args:
        has_pytest: Whether pytest tests were found (skip detection if no Python tests).

    Returns:
        List of file paths containing orphaned tests (max 10).
    """
    # Check runtime config first (#1238)
    if SKIP_ORPHANED_TESTS:
        return []

    root = _resolve_root(repo_root)

    if not has_pytest:
        # Only check if we have Python test infrastructure
        if (
            not (root / "pytest.ini").exists()
            and not (root / "pyproject.toml").exists()
        ):
            return []

    # Base exclusions (common non-source directories)
    # Need BOTH root patterns AND **/ patterns since git **/ doesn't match root (#1238)
    # Root patterns: match top-level directories (tests/, venv/)
    # **/ patterns: match nested directories (crates/foo/tests/)
    base_excludes = [
        # Root-level directories (existing behavior)
        "tests",
        "test",
        ".venv*",  # Glob pattern matches .venv, .venv-py313, etc. (#1233)
        "venv",
        "env",
        "scripts",
        "examples",
        "benchmarks",  # Integration/perf benchmarks (#998)
        ".ai_template_self",
        "site-packages",  # Python packages inside venvs (#1233)
        # Nested directories at any depth (#1238, #1242)
        # **/ prefix + /** suffix required for recursive matching
        # These REQUIRE :(exclude,glob) magic for ** patterns to work
        "**/tests/**",
        "**/test/**",
        "**/.venv*/**",
        "**/venv/**",
        "**/env/**",
        "**/scripts/**",
        "**/examples/**",
        "**/benchmarks/**",
        "**/.ai_template_self/**",
        "**/site-packages/**",
    ]

    # Add user-specified exclusions from .pulse_ignore (#982)
    user_excludes = _load_pulse_ignore(repo_root)
    all_excludes = base_excludes + user_excludes

    if use_git:
        # Use git grep for speed (#1035) - lists files with def test_ outside test dirs
        # Exclude standard test directories and common non-source directories
        # Use :(exclude,glob) for ** patterns, :(exclude) for simple patterns (#1242)
        def _make_exclude(pattern: str) -> str:
            if "**" in pattern:
                return f'":(exclude,glob){pattern}"'
            return f'":(exclude){pattern}"'

        exclude_patterns = " ".join([_make_exclude(d) for d in all_excludes])
        # Use pattern that matches function definitions (leading whitespace or line start)
        # This avoids matching strings like '"def test_"' in source code
        # Timeout prevents hang on very large repos (#956)
        grep_cmd = (
            f'git grep -lE "^[[:space:]]*def test_" -- "*.py" '
            f"{exclude_patterns} 2>/dev/null || true"
        )
        result = run_cmd(["bash", "-c", grep_cmd], timeout=30, cwd=root)
        if result.ok and result.stdout.strip():
            git_orphaned = [
                f.strip() for f in result.stdout.strip().split("\n") if f.strip()
            ]
            return git_orphaned[:10]  # Limit to 10
        return []

    orphaned: list[str] = []
    for path in root.rglob("*.py"):
        if _path_is_excluded(path):
            continue
        rel_path = path.relative_to(root).as_posix()
        if _matches_exclude_pattern(rel_path, all_excludes):
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lstrip().startswith("def test_"):
                        orphaned.append(rel_path)
                        break
        except OSError:
            continue
        if len(orphaned) >= 10:
            break
    return orphaned


def _is_cargo_invocation_error(command: str) -> bool:
    """Detect cargo commands likely to fail with invocation errors (#929).

    Cargo invocation errors occur when test harness arguments (like --test-threads)
    are passed to non-libtest binaries (like criterion benches).

    Returns True if the command pattern is known to cause invocation errors.
    """
    if not command:
        return False
    # Pattern 1: cargo test --benches with libtest args after --
    # Pattern 2: cargo bench with libtest args after --
    # criterion benches don't support libtest arguments
    is_bench_command = "cargo test" in command and "--benches" in command
    is_cargo_bench = command.strip().startswith("cargo bench")
    if (is_bench_command or is_cargo_bench) and " -- " in command:
        # Check for libtest-specific args that criterion doesn't understand
        after_separator = command.split(" -- ", 1)[1]
        libtest_args = ["--test-threads", "--nocapture", "--test", "--bench"]
        return any(arg in after_separator for arg in libtest_args)
    return False


def _get_recent_test_results(repo_root: Path | None = None) -> dict | None:
    """Read recent test results from log_test.py logs.

    Returns:
        Dict with runs, passed, failed, command_errors, total_duration_s for last 24h,
        or None if no log exists.

    Note: command_errors are cargo invocation errors (e.g., passing --test-threads
    to criterion benches) which should not be counted as test failures (#929).
    Current malformed-log behavior: a JSONDecodeError on any line in the recent
    window aborts parsing and returns None.
    """
    root = _resolve_root(repo_root)
    test_log = root / "logs/tests.jsonl"
    if not test_log.exists():
        return None

    try:
        lines = test_log.read_text().strip().split("\n")
        now = datetime.now()
        recent_runs = []
        for line in lines[-100:]:  # Check last 100 entries
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                continue
            if entry.get("event") == "end":
                ts = entry.get("timestamp", "")
                try:
                    entry_time = datetime.fromisoformat(ts)
                    if (now - entry_time).total_seconds() < 86400:
                        recent_runs.append(entry)
                except ValueError:
                    pass

        if recent_runs:
            passed = 0
            failed = 0
            command_errors = 0
            for r in recent_runs:
                exit_code = r.get("exit_code", 0)
                command = r.get("command", "")
                if exit_code == 0:
                    passed += 1
                elif _is_cargo_invocation_error(command):
                    # Cargo invocation error, not a real test failure (#929)
                    command_errors += 1
                else:
                    failed += 1
            total_duration = sum(r.get("duration_s", 0) or 0 for r in recent_runs)
            result: dict[str, int | float] = {
                "runs": len(recent_runs),
                "passed": passed,
                "failed": failed,
                "total_duration_s": round(total_duration, 1),
            }
            if command_errors > 0:
                result["command_errors"] = command_errors
            return result
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return None


def get_test_status(
    repo_root: Path | None = None,
    use_git: bool = True,
    log_root: Path | None = None,
) -> dict:
    """Get test counts (fast grep-based) and recent results from log_test.py.

    Supports mixed repos with both Python and Rust tests.
    Returns per-framework counts and total.

    REQUIRES: repo_root is None or a valid directory Path
    REQUIRES: log_root is None or a valid directory Path
    ENSURES: Returns dict with 'count' (int >= 0), 'framework' (str), 'by_framework' (dict)
    ENSURES: Returns dict with optional 'recent_results' if log files exist
    ENSURES: Never raises (returns partial results on error)
    """
    root = _resolve_root(repo_root)
    if log_root is None:
        log_root = root

    # Count tests by framework
    pytest_count, cargo_count, by_framework = _count_tests_by_framework(
        repo_root=root, use_git=use_git
    )
    total_count = pytest_count + cargo_count

    # Determine framework label
    if pytest_count > 0 and cargo_count > 0:
        framework = "mixed"
    elif pytest_count > 0:
        framework = "pytest"
    elif cargo_count > 0:
        framework = "cargo"
    else:
        framework = "unknown"

    status: dict[str, object] = {
        "count": total_count,
        "framework": framework,
        "by_framework": by_framework,
        "orphaned_tests": _detect_orphaned_python_tests(
            pytest_count > 0, repo_root=root, use_git=use_git
        ),
    }

    # Note: Rust inline tests in src/ with #[cfg(test)] are valid, not orphaned.

    # Add recent test results if available
    if recent := _get_recent_test_results(log_root):
        status["recent_24h"] = recent

    return status


# --- Proof Coverage Functions ---


def _get_workspace_member_dirs(
    cargo_toml_path: Path,
    repo_root: Path | None = None,
) -> list[str]:
    """Get workspace member directories from Cargo.toml.

    Parses [workspace] members array and expands glob patterns.
    Returns list of existing directories containing src/ or tests/.
    """
    try:
        content = cargo_toml_path.read_text()
        cargo = tomllib.loads(content)
    except (OSError, tomllib.TOMLDecodeError):
        return []

    root = _resolve_root(repo_root)
    members = cargo.get("workspace", {}).get("members", [])
    if not members:
        return []

    member_dirs: list[Path] = []
    for pattern in members:
        # Expand glob patterns (e.g., "crates/*", "libs/*")
        matches = [Path(p) for p in glob_module.glob(str(root / pattern))]
        if matches:
            member_dirs.extend(matches)
        else:
            # Literal path exists
            literal = root / pattern
            if literal.exists():
                member_dirs.append(literal)

    # For each member, add src/ and tests/ subdirs if they exist
    search_dirs = []
    for member_path in member_dirs:
        if member_path.is_dir():
            for sub in ["src", "tests"]:
                sub_path = member_path / sub
                if sub_path.exists():
                    search_dirs.append(_relative_path(sub_path, root))

    return search_dirs


def _detect_kani_coverage(
    search_dirs: list[str],
    repo_root: Path | None = None,
) -> dict | None:
    """Detect Kani bounded model checking coverage in Rust code.

    Priority order for verification status:
    1. kani_status.json (from kani_runner.py - includes actual run results)
    2. metrics/kani/latest.json (from kani_metrics_snapshot.py)
    3. grep-based counting (fallback)

    Args:
        search_dirs: List of directories to search for Rust files.
        repo_root: Repository root path (defaults to cwd).

    Returns:
        Dict with proofs, contract_attrs, contracted_fns, total_functions,
        coverage_pct if any Kani usage found, else None.
        Optional fields when verification status available (#2232, #2263):
        passing, failing, unexecuted, timeout, oom.
    """
    if not search_dirs:
        return None

    root = _resolve_root(repo_root)

    # Priority 1: kani_status.json from kani_runner.py (#2263)
    # This is the most authoritative source as it contains actual run results
    kani_status_file = root / "kani_status.json"
    canonical_proofs: int | None = None
    canonical_passing: int | None = None
    canonical_failing: int | None = None
    canonical_unexecuted: int | None = None
    canonical_timeout: int | None = None
    canonical_oom: int | None = None

    if kani_status_file.exists():
        try:
            data = json.loads(kani_status_file.read_text())
            summary = data.get("summary", {})
            if summary:
                # Total proofs = sum of all statuses
                canonical_proofs = sum(summary.values())
                canonical_passing = summary.get("passed", 0)
                canonical_failing = summary.get("failed", 0) + summary.get("error", 0)
                canonical_unexecuted = summary.get("not_run", 0)
                canonical_timeout = summary.get("timeout", 0)
                canonical_oom = summary.get("oom", 0)
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to other sources

    # Priority 2: metrics/kani/latest.json from kani_metrics_snapshot.py (#1723)
    # Only use if kani_status.json didn't provide data
    if canonical_proofs is None:
        canonical_metrics = root / "metrics" / "kani" / "latest.json"
        if canonical_metrics.exists():
            try:
                data = json.loads(canonical_metrics.read_text())
                proofs_data = data.get("proofs", {})
                if isinstance(proofs_data, dict) and "total" in proofs_data:
                    canonical_proofs = proofs_data["total"]
                    # Extract verification status if available (#2232)
                    if "passing" in proofs_data:
                        canonical_passing = proofs_data["passing"]
                    if "failing" in proofs_data:
                        canonical_failing = proofs_data["failing"]
                    if "unexecuted" in proofs_data:
                        canonical_unexecuted = proofs_data["unexecuted"]
            except (json.JSONDecodeError, OSError):
                pass  # Fall through to grep-based counting

    # Quote paths properly for shell to handle spaces (#1070)
    search_path = shlex.join(search_dirs)

    # Use canonical proof count if available, otherwise grep (#1723)
    if canonical_proofs is not None:
        proofs = canonical_proofs
    else:
        # Count kani proof harnesses (timeout prevents hang on large repos - #956)
        result = run_cmd(
            [
                "bash",
                "-c",
                f"grep -rE '#\\[(kani::|cfg_attr\\(kani,[[:space:]]*kani::)proof' "
                f"{search_path} 2>/dev/null | wc -l",
            ],
            timeout=30,
            cwd=root,
        )
        proofs = (
            int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0
        )

    # Count kani contract attributes (requires/ensures/modifies)
    # Also match cfg_attr(kani, kani::...) form for non-Kani build compatibility (#1018)
    # Timeout prevents hang on large repos (#956)
    pattern = (
        r"#\[(kani::|cfg_attr\(kani,[[:space:]]*kani::)(requires|ensures|modifies)"
    )
    grep_cmd = f"grep -rE '{pattern}' {search_path} 2>/dev/null | wc -l"
    result = run_cmd(["bash", "-c", grep_cmd], timeout=30, cwd=root)
    contract_attrs = (
        int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0
    )

    if proofs == 0 and contract_attrs == 0:
        return None

    # Count unique functions with contracts (for accurate coverage_pct)
    # A function with both requires and ensures should count as 1, not 2
    # Also match cfg_attr(kani, kani::...) form for non-Kani build compatibility (#1018)
    # Timeout prevents hang on large repos (#956)
    fn_grep_cmd = (
        f"grep -rE -A2 '{pattern}' {search_path} 2>/dev/null "
        "| grep -E '(pub\\s+)?(async\\s+)?fn\\s+' "
        "| sed -E 's/.*fn[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*).*/\\1/' "
        "| sort -u | wc -l"
    )
    result = run_cmd(["bash", "-c", fn_grep_cmd], timeout=30, cwd=root)
    contracted_fns = (
        int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0
    )

    # Count total functions (approximate) - only in src dirs for accuracy
    # Quote paths properly for shell to handle spaces (#1070)
    src_dir_list = [d for d in search_dirs if d.endswith("src") or "/src" in d]
    src_dirs_quoted = shlex.join(src_dir_list)
    total_fns = 0
    if src_dir_list:
        # Timeout prevents hang on large repos (#956)
        result = run_cmd(
            [
                "bash",
                "-c",
                # Use -h to suppress filename prefix so ^ anchor matches line content
                f"grep -rhE '^\\s*(pub )?fn ' {src_dirs_quoted} 2>/dev/null | wc -l",
            ],
            timeout=30,
            cwd=root,
        )
        total_fns = (
            int(result.stdout.strip()) if result.ok and result.stdout.strip() else 0
        )

    result_dict: dict[str, int | float] = {
        "proofs": proofs,
        "contract_attrs": contract_attrs,
        "contracted_fns": contracted_fns,
        "total_functions": total_fns,
        "coverage_pct": round(100 * contracted_fns / total_fns, 1)
        if total_fns > 0
        else 0,
    }
    # Add verification status if available from canonical metrics (#2232, #2263)
    if canonical_passing is not None:
        result_dict["passing"] = canonical_passing
    if canonical_failing is not None:
        result_dict["failing"] = canonical_failing
    if canonical_unexecuted is not None:
        result_dict["unexecuted"] = canonical_unexecuted
    if canonical_timeout is not None:
        result_dict["timeout"] = canonical_timeout
    if canonical_oom is not None:
        result_dict["oom"] = canonical_oom
    return result_dict


def _detect_tla_coverage(repo_root: Path | None = None) -> dict | None:
    """Detect TLA+ specification coverage.

    Returns:
        Dict with files, specs, invariants if any TLA+ files found, else None.
    """
    root = _resolve_root(repo_root)

    # Use prune-style exclusion for speed on large repos
    result = run_cmd(
        [
            "bash",
            "-c",
            f"find . -type d -name .git -prune -o {FIND_PRUNE} -o "
            f"-name '*.tla' -type f -print 2>/dev/null",
        ],
        timeout=10,
        cwd=root,
    )
    if not result.ok or not result.stdout.strip():
        return None

    tla_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    if not tla_files:
        return None

    specs = invariants = 0
    for f in tla_files[:20]:  # Limit scanning
        try:
            content = (root / f).read_text()
            specs += content.count("SPECIFICATION")
            invariants += content.count("INVARIANT") + content.count("PROPERTY")
        except (FileNotFoundError, UnicodeDecodeError):
            pass

    return {"files": len(tla_files), "specs": specs, "invariants": invariants}


def _detect_lean_coverage(repo_root: Path | None = None) -> dict | None:
    """Detect Lean theorem prover coverage.

    Returns:
        Dict with files, theorems if any Lean files found, else None.
    """
    root = _resolve_root(repo_root)

    # Use prune-style exclusion for speed on large repos
    result = run_cmd(
        [
            "bash",
            "-c",
            f"find . -type d -name .git -prune -o {FIND_PRUNE} -o "
            f"-name '*.lean' -type f -print 2>/dev/null | head -100",
        ],
        timeout=10,
        cwd=root,
    )
    if not result.ok or not result.stdout.strip():
        return None

    lean_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    if not lean_files:
        return None

    theorems = 0
    for f in lean_files[:30]:  # Limit scanning
        try:
            content = (root / f).read_text()
            theorems += content.count("theorem ") + content.count("lemma ")
        except (FileNotFoundError, UnicodeDecodeError):
            pass

    return {"files": len(lean_files), "theorems": theorems}


def _detect_smt_coverage(repo_root: Path | None = None) -> dict | None:
    """Detect SMT-LIB/Z3/z4 coverage.

    Returns:
        Dict with smt2_files, z4_usage if any SMT usage found, else None.
    """
    root = _resolve_root(repo_root)

    # Use prune-style exclusion for speed on large repos
    result = run_cmd(
        [
            "bash",
            "-c",
            f"find . -type d -name .git -prune -o {FIND_PRUNE} -o "
            f"-name '*.smt2' -type f -print 2>/dev/null | wc -l",
        ],
        timeout=10,
        cwd=root,
    )
    smt_count = _parse_count_from_result(result)

    # Also check for z4 usage in Rust (use --exclude-dir for speed)
    result = run_cmd(
        [
            "bash",
            "-c",
            f"grep -rc 'z4::' src/ {GREP_EXCLUDE_DIRS} 2>/dev/null | "
            "awk -F: '{sum+=$2} END {print sum+0}'",
        ],
        timeout=10,
        cwd=root,
    )
    z4_usage = _parse_count_from_result(result)

    if smt_count == 0 and z4_usage == 0:
        return None

    return {"smt2_files": smt_count, "z4_usage": z4_usage}


def _detect_nn_verification(repo_root: Path | None = None) -> dict | None:
    """Detect neural network verification coverage (ONNX + VNNLib).

    Returns:
        Dict with onnx_models, vnnlib_specs if any NN verification found, else None.
    """
    root = _resolve_root(repo_root)

    # Use prune-style exclusion for speed on large repos
    result = run_cmd(
        [
            "bash",
            "-c",
            f"find . -type d -name .git -prune -o {FIND_PRUNE} -o "
            f"-name '*.onnx' -type f -print 2>/dev/null | wc -l",
        ],
        timeout=10,
        cwd=root,
    )
    onnx_count = _parse_count_from_result(result)

    result = run_cmd(
        [
            "bash",
            "-c",
            f"find . -type d -name .git -prune -o {FIND_PRUNE} -o "
            f"-name '*.vnnlib' -type f -print 2>/dev/null | wc -l",
        ],
        timeout=10,
        cwd=root,
    )
    vnnlib_count = _parse_count_from_result(result)

    if onnx_count == 0 and vnnlib_count == 0:
        return None

    return {"onnx_models": onnx_count, "vnnlib_specs": vnnlib_count}


def get_proof_coverage(repo_root: Path | None = None) -> dict:
    """Detect formal verification coverage across proof systems.

    Systems detected:
    - Kani: Rust bounded model checking (#[kani::proof], requires/ensures/modifies)
    - TLA+: Distributed system specs (.tla files)
    - Lean: Theorem proving (.lean files)
    - SMT: SAT/SMT solving (.smt2 files, z4:: usage)
    - NN verification: Neural network proofs (.onnx + vnnlib)

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with proof system names as keys
    ENSURES: Each system value is a dict with system-specific metrics
    ENSURES: Never raises (returns empty dict on error)
    """
    coverage: dict[str, dict] = {}
    root = _resolve_root(repo_root)

    # Build Kani search paths from Cargo.toml workspace
    cargo_toml = root / "Cargo.toml"
    if cargo_toml.exists():
        search_dirs = [d for d in ["src", "tests"] if (root / d).exists()]
        search_dirs.extend(_get_workspace_member_dirs(cargo_toml, repo_root=root))

        # Legacy fallback: include crates/ even if not in workspace.members
        crates_dir = root / "crates"
        if crates_dir.exists():
            for crate_dir in crates_dir.iterdir():
                if crate_dir.is_dir():
                    for sub in ["src", "tests"]:
                        sub_path = crate_dir / sub
                        if sub_path.exists():
                            rel_sub_path = _relative_path(sub_path, root)
                            if rel_sub_path not in search_dirs:
                                search_dirs.append(rel_sub_path)

        if kani := _detect_kani_coverage(search_dirs, repo_root=root):
            coverage["kani"] = kani

    # Detect each proof system
    if tla := _detect_tla_coverage(repo_root=root):
        coverage["tla2"] = tla

    if lean := _detect_lean_coverage(repo_root=root):
        coverage["lean"] = lean

    if smt := _detect_smt_coverage(repo_root=root):
        coverage["smt"] = smt

    if nn := _detect_nn_verification(repo_root=root):
        coverage["nn_verification"] = nn

    return coverage


# --- Test Coverage Functions ---


def _parse_coverage_json(coverage_json: Path) -> dict | None:
    """Parse coverage.json format from coverage.py.

    Args:
        coverage_json: Path to coverage.json file.

    Returns:
        Dict with coverage_pct, lines_total, lines_covered, etc.
        Or None if parsing fails.
    """
    try:
        data = json.loads(coverage_json.read_text())
        totals = data.get("totals", {})
        if not totals:
            return None

        return {
            "coverage_pct": totals.get("percent_covered", 0.0),
            "lines_total": totals.get("num_statements", 0),
            "lines_covered": totals.get("covered_lines", 0),
            "lines_missing": totals.get("missing_lines", 0),
            "branches_total": totals.get("num_branches", 0),
            "branches_covered": totals.get("covered_branches", 0),
            "source": "coverage.json",
            "timestamp": coverage_json.stat().st_mtime,
        }
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _parse_coverage_xml(coverage_xml: Path) -> dict | None:
    """Parse coverage.xml (Cobertura format).

    Args:
        coverage_xml: Path to coverage.xml file.

    Returns:
        Dict with coverage_pct, lines_total, lines_covered, etc.
        Or None if parsing fails.
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(coverage_xml)
        root = tree.getroot()

        # Cobertura format has line-rate and branch-rate as percentages (0.0-1.0)
        line_rate = float(root.get("line-rate", 0))
        branch_rate = float(root.get("branch-rate", 0))
        lines_valid = int(root.get("lines-valid", 0))
        lines_covered = int(root.get("lines-covered", 0))
        branches_valid = int(root.get("branches-valid", 0))
        branches_covered = int(root.get("branches-covered", 0))

        return {
            "coverage_pct": round(line_rate * 100, 1),
            "lines_total": lines_valid,
            "lines_covered": lines_covered,
            "lines_missing": lines_valid - lines_covered,
            "branches_total": branches_valid,
            "branches_covered": branches_covered,
            "branch_coverage_pct": round(branch_rate * 100, 1),
            "source": "coverage.xml",
            "timestamp": coverage_xml.stat().st_mtime,
        }
    except (ET.ParseError, OSError, ValueError):
        return None


def get_test_coverage(repo_root: Path | None = None) -> dict | None:
    """Get test coverage metrics from existing coverage data.

    Parses coverage data from previous test runs. Does NOT run tests.
    Checks for coverage files in order of preference:
    1. coverage.json (most complete data)
    2. coverage.xml (Cobertura format)

    Args:
        repo_root: Repository root path (defaults to cwd).

    Returns:
        Dict with coverage metrics if coverage data exists, else None.
        Includes: coverage_pct, lines_total, lines_covered, lines_missing,
                  branches_total, branches_covered, source, timestamp.

    REQUIRES: repo_root is None or a valid directory Path
    ENSURES: Returns dict with coverage data or None if no data available
    ENSURES: Never raises (returns None on error)
    """
    root = _resolve_root(repo_root)

    # Try coverage formats in order of preference
    # coverage.json has the most complete data
    coverage_json = root / "coverage.json"
    if coverage_json.exists():
        if result := _parse_coverage_json(coverage_json):
            return result

    # Try coverage.xml (Cobertura format)
    coverage_xml = root / "coverage.xml"
    if coverage_xml.exists():
        if result := _parse_coverage_xml(coverage_xml):
            return result

    return None

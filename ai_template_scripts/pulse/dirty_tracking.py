#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Dirty file tracking and snapshot functions for pulse read-only invariant.

These functions detect when files are modified during pulse runs, helping
diagnose concurrent modifications from other AI sessions, formatters, etc.

Part of #404: pulse.py module split.
"""

import hashlib
import json
import os
import shlex
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    from ai_template_scripts.subprocess_utils import run_cmd
except ModuleNotFoundError:
    import sys as sys_inner

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys_inner.path:
        sys_inner.path.insert(0, str(repo_root))
    from ai_template_scripts.subprocess_utils import run_cmd

from .session_metrics import get_active_session_details


def _is_rename_status(status: str) -> bool:
    """Return True if status indicates a rename or copy."""
    return "R" in status or "C" in status


def _normalize_status_path(path: str, status: str) -> str:
    """Normalize git status paths, handling rename syntax."""
    if _is_rename_status(status) and " -> " in path:
        return path.split(" -> ", 1)[1].strip()
    return path


def _is_pulse_output_path(path: str) -> bool:
    """Return True if path is expected pulse output or session artifact.

    Excludes:
    - metrics/ - pulse's own output
    - .flags/ - pulse's flag files
    - reports/ - session artifacts from looper (concurrent sessions may modify)
    - worker_logs/ - worker session logs
    """
    clean_path = path.removeprefix("./")
    # Pre-computed tuples for efficient matching
    excluded_dirs = ("metrics", ".flags", "reports", "worker_logs")
    excluded_with_slash = ("metrics/", ".flags/", "reports/", "worker_logs/")
    if clean_path in excluded_dirs or clean_path in excluded_with_slash:
        return True
    return clean_path.startswith(excluded_with_slash)


def _identify_file_modifiers() -> list[str]:
    """Identify processes that could be modifying source files.

    Returns list of potential culprits when pulse detects unexpected file changes.
    Helps diagnose read-only violations that aren't caused by pulse itself.
    """
    culprits = []

    # Check for rust-analyzer (can auto-apply code actions)
    result = run_cmd(["pgrep", "-x", "rust-analyzer"], timeout=5)
    if result.ok and result.stdout.strip():
        culprits.append("rust-analyzer (may auto-apply fixes)")

    # Check for cargo clippy/fix processes
    result = run_cmd(
        ["bash", "-c", "pgrep -f 'cargo.*--fix' || pgrep -f 'cargo.*clippy'"],
        timeout=5,
    )
    if result.ok and result.stdout.strip():
        culprits.append("cargo clippy/fix (auto-fix in progress)")

    # Check for rustfmt
    result = run_cmd(["pgrep", "-x", "rustfmt"], timeout=5)
    if result.ok and result.stdout.strip():
        culprits.append("rustfmt (formatting in progress)")

    # Check for other formatters (black, prettier, etc.)
    result = run_cmd(
        ["bash", "-c", "pgrep -f 'black.*\\.py' || pgrep -f prettier"],
        timeout=5,
    )
    if result.ok and result.stdout.strip():
        culprits.append("code formatter (black/prettier)")

    # Check for active looper sessions (another AI might be editing)
    result = run_cmd(["bash", "-c", "pgrep -f looper.py"], timeout=5)
    if result.ok and result.stdout.strip():
        pids = result.stdout.strip().split("\n")
        if len(pids) > 1:  # More than just the current session
            culprits.append(f"other AI sessions ({len(pids)} looper processes)")

    # Check for recent commits (workers that committed and exited #1073)
    # If a commit happened in the last 2 minutes, dirty repo is likely from that
    result = run_cmd(
        ["git", "log", "--oneline", "--since=2 minutes ago", "-1"], timeout=5
    )
    if result.ok and result.stdout.strip():
        # A commit happened very recently - likely explains dirty files
        culprits.append("recent git commit (worker likely just committed)")

    return culprits


def _get_file_mtimes(files: set[str]) -> dict[str, float]:
    """Get modification times for a set of files.

    Used to detect file changes by comparing mtimes before/after pulse run.
    Returns dict mapping file path to mtime (Unix timestamp).
    """
    mtimes: dict[str, float] = {}
    for path in files:
        try:
            stat = os.stat(path)
            mtimes[path] = stat.st_mtime
        except OSError:
            pass
    return mtimes


def _save_dirty_snapshot(
    new_dirty: set[str],
    modified_dirty: set[str],
    porcelain_before: list[str] | None = None,
    porcelain_after: list[str] | None = None,
) -> Path | None:
    """Save debug snapshot when repo becomes dirty during pulse (#1858).

    Saves evidence to temp dir outside the repo (preserves pulse read-only guarantee):
    - head.txt: Current HEAD short SHA
    - status_before.txt / status_after.txt: Git porcelain status
    - modified_files.txt: List of files modified during pulse
    - diff.patch: Unstaged changes
    - diff_cached.patch: Staged changes
    - ps.txt: Process list (pid, etime, command)
    - active_sessions.json: Active looper sessions with optional fields

    REQUIRES: new_dirty or modified_dirty is non-empty
    ENSURES: Returns snapshot dir Path or None on failure
    ENSURES: Never raises
    """
    all_modified = new_dirty | modified_dirty
    if not all_modified:
        return None

    try:
        # Create snapshot in /tmp (outside repo to preserve read-only guarantee)
        repo_name = Path.cwd().name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path(f"/tmp/pulse_dirty_{repo_name}_{ts}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # head.txt - short SHA
        head_result = run_cmd(["git", "rev-parse", "--short", "HEAD"], timeout=5)
        if head_result.ok:
            (snapshot_dir / "head.txt").write_text(head_result.stdout.strip())

        # status_before.txt / status_after.txt
        if porcelain_before is not None:
            (snapshot_dir / "status_before.txt").write_text("\n".join(porcelain_before))
        if porcelain_after is not None:
            (snapshot_dir / "status_after.txt").write_text("\n".join(porcelain_after))

        # modified_files.txt
        (snapshot_dir / "modified_files.txt").write_text(
            "\n".join(sorted(all_modified))
        )

        # diff.patch - unstaged changes
        diff_result = run_cmd(["git", "diff"], timeout=30)
        if diff_result.ok and diff_result.stdout:
            (snapshot_dir / "diff.patch").write_text(diff_result.stdout)

        # diff_cached.patch - staged changes
        diff_cached_result = run_cmd(["git", "diff", "--cached"], timeout=30)
        if diff_cached_result.ok and diff_cached_result.stdout:
            (snapshot_dir / "diff_cached.patch").write_text(diff_cached_result.stdout)

        # ps.txt - process list (find potential culprits)
        # macOS ps doesn't support --sort, so use portable format
        ps_result = run_cmd(["ps", "-eo", "pid,etime,command"], timeout=10)
        if ps_result.ok:
            (snapshot_dir / "ps.txt").write_text(ps_result.stdout)

        # active_sessions.json - looper sessions with optional fields
        try:
            sessions = get_active_session_details()
            (snapshot_dir / "active_sessions.json").write_text(
                json.dumps(sessions, indent=2, default=str)
            )
        except Exception:
            pass

        print(
            f"    📸 Debug snapshot saved to: {snapshot_dir}",
            file=sys.stderr,
        )
        return snapshot_dir

    except Exception as e:
        print(f"    ⚠️  Failed to save dirty snapshot: {e}", file=sys.stderr)
        return None


def _format_dirty_diagnostics(
    new_dirty: set[str],
    modified_dirty: set[str],
    culprits: list[str],
    mtimes_before: dict[str, float] | None = None,
    mtimes_after: dict[str, float] | None = None,
    porcelain_before: list[str] | None = None,
    porcelain_after: list[str] | None = None,
) -> None:
    """Format and print enhanced repo-dirty diagnostics (#1583, #1858).

    Prints:
    - Summary of modified files
    - File mtimes if available (shows exactly when files changed)
    - Git status diff (porcelain before/after)
    - Culprit analysis with actionable next steps

    Also saves debug snapshot to temp dir for post-incident analysis (#1858).
    """
    all_modified = new_dirty | modified_dirty

    # Save debug snapshot before printing diagnostics (#1858)
    _save_dirty_snapshot(
        new_dirty=new_dirty,
        modified_dirty=modified_dirty,
        porcelain_before=porcelain_before,
        porcelain_after=porcelain_after,
    )

    # Header
    print(
        f"\n⚠️  REPO DIRTY: Files modified during pulse run: {', '.join(sorted(all_modified))}",
        file=sys.stderr,
    )

    if modified_dirty:
        print(
            f"    (Already dirty but content changed: {', '.join(sorted(modified_dirty))})",
            file=sys.stderr,
        )

    print(
        "    NOTE: Pulse is read-only. External processes modified these files.",
        file=sys.stderr,
    )

    # File mtime details (if available)
    if mtimes_before and mtimes_after:
        print("\n    File modification times:", file=sys.stderr)
        for path in sorted(all_modified):
            before = mtimes_before.get(path)
            after = mtimes_after.get(path)
            if path in new_dirty and after:
                # New file appeared - show creation time
                ts = time.strftime("%H:%M:%S", time.localtime(after))
                print(f"      {path}: created at {ts}", file=sys.stderr)
            elif before and after and before != after:
                # File was modified - show both times
                ts_before = time.strftime("%H:%M:%S", time.localtime(before))
                ts_after = time.strftime("%H:%M:%S", time.localtime(after))
                print(
                    f"      {path}: {ts_before} → {ts_after} (modified)",
                    file=sys.stderr,
                )

    # Git status diff (if available)
    if porcelain_before is not None and porcelain_after is not None:
        # Filter to source files only
        def filter_status(lines: list[str]) -> set[str]:
            result = set()
            for line in lines:
                if len(line) >= 4:
                    path = line[3:].strip()
                    if not _is_pulse_output_path(path):
                        result.add(line)
            return result

        before_set = filter_status(porcelain_before)
        after_set = filter_status(porcelain_after)
        new_status = after_set - before_set

        if new_status:
            print("\n    New git status entries:", file=sys.stderr)
            for line in sorted(new_status)[:5]:  # Limit to 5 lines
                print(f"      {line}", file=sys.stderr)
            if len(new_status) > 5:
                print(f"      ... and {len(new_status) - 5} more", file=sys.stderr)

    # Culprit analysis
    if culprits:
        print(f"\n    Possible culprits: {', '.join(culprits)}", file=sys.stderr)
    else:
        print(
            "\n    Possible causes: concurrent AI session, rust-analyzer, cargo clippy --fix.",
            file=sys.stderr,
        )

    # Actionable next steps for multi-session repos
    print("\n    📋 Next steps:", file=sys.stderr)
    if any("other ai sessions" in c.lower() for c in culprits):
        print(
            "      - Multiple looper sessions detected. This is expected behavior.",
            file=sys.stderr,
        )
        print(
            "      - Check: pgrep -f looper.py | xargs -I{} ps -p {} -o pid,etime,args",
            file=sys.stderr,
        )
    if any("recent git commit" in c.lower() for c in culprits):
        print(
            "      - Recent commit detected. Worker likely just finished.",
            file=sys.stderr,
        )
        print(
            "      - Check: git log --oneline -3",
            file=sys.stderr,
        )
    if not culprits:
        print(
            "      - Check for background processes: lsof +D . 2>/dev/null | head -20",
            file=sys.stderr,
        )
        print(
            "      - Check active sessions: pgrep -f looper.py",
            file=sys.stderr,
        )


def _get_porcelain_status() -> list[str]:
    """Get raw git status --porcelain output lines.

    Returns list of non-empty porcelain lines. Used as shared data source
    for _get_dirty_source_files() and get_git_status() to avoid redundant
    git status calls (#1340).
    """
    result = run_cmd(["git", "status", "--porcelain"])
    if not result.ok or not result.stdout.strip():
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _get_dirty_source_files(porcelain_lines: list[str] | None = None) -> set[str]:
    """Get set of dirty source files (excluding expected pulse outputs).

    Args:
        porcelain_lines: Optional pre-fetched git status --porcelain output lines.
                         If None, will fetch fresh. Pass to avoid redundant calls (#1340).

    Returns set of modified file paths, excluding metrics/ and .flags/ directories
    which pulse is expected to write to.
    """
    if porcelain_lines is None:
        porcelain_lines = _get_porcelain_status()

    dirty = set()
    for line in porcelain_lines:
        if len(line) < 4:
            continue
        # Format: "XY filename" where XY is status flags
        status = line[:2]
        path = _normalize_status_path(line[3:].strip(), status)
        # Skip expected pulse output directories
        if not path or _is_pulse_output_path(path):
            continue
        dirty.add(path)
    return dirty


def _get_dirty_file_fingerprints(files: set[str]) -> dict[str, str]:
    """Get fingerprints for dirty files to detect modifications (#1339).

    Uses `git diff <file>` hash to detect if file content changed, even if
    it was already dirty before the run. This catches pulse regressions that
    modify already-dirty files.

    Args:
        files: Set of dirty file paths to fingerprint.

    Returns:
        Dict mapping file path to hash of its diff output.
    """

    fingerprints: dict[str, str] = {}
    for path in files:
        # Get diff for this specific file
        result = run_cmd(["git", "diff", path], timeout=5)
        if result.ok:
            # Hash the diff output - changes if file content changes
            diff_hash = hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
            fingerprints[path] = diff_hash
    return fingerprints


def _detect_dirty_file_changes(
    before: dict[str, str], after: dict[str, str]
) -> set[str]:
    """Detect files that were already dirty but got modified (#1339).

    Args:
        before: Fingerprints from before pulse run.
        after: Fingerprints from after pulse run.

    Returns:
        Set of files that were dirty before AND had their content changed.
    """
    changed = set()
    for path, before_hash in before.items():
        if path in after and before_hash != after[path]:
            changed.add(path)
    return changed


@contextmanager
def _snapshot_repo_root() -> Generator[Path | None, None, None]:
    """Create a git snapshot of HEAD in a temp directory for stable scanning."""
    with tempfile.TemporaryDirectory(prefix="pulse_snapshot_") as tmpdir:
        snapshot_root = Path(tmpdir)
        # Use pipefail to catch git archive failures even if tar succeeds (#1336)
        archive_cmd = (
            f"set -o pipefail; git archive HEAD | tar -x -C {shlex.quote(tmpdir)}"
        )
        result = run_cmd(["bash", "-c", archive_cmd], timeout=60)
        if not result.ok:
            prefix = str(snapshot_root).rstrip("/") + "/"
            checkout_result = run_cmd(
                ["git", "checkout-index", "-a", "-f", f"--prefix={prefix}"],
                timeout=60,
            )
            if not checkout_result.ok:
                archive_error = result.error or result.stderr.strip() or "unknown error"
                checkout_error = (
                    checkout_result.error
                    or checkout_result.stderr.strip()
                    or "unknown error"
                )
                print(
                    "pulse: snapshot failed (archive/checkout-index): "
                    f"{archive_error}; {checkout_error}",
                    file=sys.stderr,
                )
                yield None
                return
        yield snapshot_root

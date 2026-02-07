#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Worker file tracker utilities for git hooks.

Provides functions for checking tracker file status and ownership.
Used by pre-commit-hook.sh and post-commit-hook.sh.

Usage from shell:
    python3 -m ai_template_scripts.tracker_utils check_alive <tracker_file>
    python3 -m ai_template_scripts.tracker_utils check_ownership <repo_root> <worker_id> <staged_files...>
    python3 -m ai_template_scripts.tracker_utils clear_committed <tracker_file> <committed_files...>
    python3 -m ai_template_scripts.tracker_utils clear_committed --stdin <tracker_file>
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    from ai_template_scripts.atomic_write import atomic_write_json
    from ai_template_scripts.subprocess_utils import is_process_alive
except ImportError:
    # Fallback when run as standalone script by pre-commit hook.
    # Intentionally simpler than atomic_write.py: no fsync, no mkstemp,
    # no directory fsync. Tracker files are ephemeral diagnostics that can
    # be rebuilt on next iteration — crash durability is not required, and
    # adding fsync would slow the commit-hook path. PID suffix prevents
    # cross-worker collisions; thread collisions within a single hook
    # invocation are not a concern for these sequential operations. (#2942)

    def atomic_write_json(path: Path, data: dict) -> None:  # type: ignore[misc]
        """Inline fallback for standalone mode (#2862).

        Simplified vs atomic_write.py: no fsync (ephemeral data),
        no mkstemp (PID suffix adequate for hook context). See #2942.
        """
        path.parent.mkdir(exist_ok=True)
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
        try:
            tmp_path.write_text(json.dumps(data, indent=2) + "\n")
            tmp_path.replace(path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def is_process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except (OSError, TypeError, ValueError):
            return False


def check_tracker_alive(tracker_file: str) -> str:
    """Check if tracker file exists and has alive PID.

    Returns:
        "alive" - tracker exists and PID is running
        "dead" - tracker exists but PID is not running
        "missing" - tracker file doesn't exist
        "invalid" - tracker file is corrupted/unreadable
    """
    if not os.path.exists(tracker_file):
        return "missing"

    try:
        with open(tracker_file) as f:
            data = json.load(f)

        # Handle malformed tracker files (e.g., empty array []) - Part of #2383
        if not isinstance(data, dict):
            return "invalid"

        pid = data.get("pid", 0)
        if pid <= 0:
            return "invalid"

        return "alive" if is_process_alive(pid) else "dead"
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return "invalid"


def _parse_worker_id(tracker_file: Path) -> str | None:
    match = re.match(r"\.worker_(\d+)_files\.json$", tracker_file.name)
    if not match:
        return None
    return match.group(1)


def _extract_worker_id(tracker_file: Path) -> str | None:
    try:
        with open(tracker_file) as f:
            data = json.load(f)
        # Handle malformed tracker files (e.g., empty array []) - Part of #2383
        if not isinstance(data, dict):
            return None
        worker_id = data.get("worker_id")
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return None

    if worker_id is None:
        return None
    return str(worker_id)


def _is_valid_tracked_file(entry: str) -> bool:
    path = Path(entry)
    if path.is_absolute():
        return False
    if ".." in path.parts:
        return False
    return True


def _load_tracker_files(tracker_file: Path, expected_worker_id: str) -> tuple[bool, list[str]]:
    try:
        with open(tracker_file) as f:
            data = json.load(f)

        # Handle malformed tracker files (e.g., empty array []) - Part of #2383
        if not isinstance(data, dict):
            return False, []

        stored_worker_id = data.get("worker_id")
        if stored_worker_id is None:
            return False, []
        # Normalize worker IDs: "W2" and "2" should match (looper stores
        # different formats depending on version/configuration)
        stored_num = re.sub(r"^[Ww]", "", str(stored_worker_id))
        expected_num = re.sub(r"^[Ww]", "", expected_worker_id)
        if stored_num != expected_num:
            return False, []

        files = data.get("files", [])
        if not isinstance(files, list) or any(not isinstance(entry, str) for entry in files):
            return False, []
        if any(not _is_valid_tracked_file(entry) for entry in files):
            return False, []

        return True, files
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return False, []


def check_ownership(repo_root: str, worker_id: str, staged_files: list[str]) -> dict:
    """Check if staged files are owned by other workers.

    Returns dict with:
        "conflicts": list of (file, owner_worker_id) tuples
        "invalid_trackers": list of (worker_id, tracker_file) tuples
        "ok": True if no conflicts or invalid trackers
    """
    repo_path = Path(repo_root)
    conflicts = []
    invalid_trackers = []
    staged_set = set(staged_files)

    # Find all OTHER workers' tracker files
    for tracker_file in repo_path.glob(".worker_*_files.json"):
        # Extract worker ID from filename
        other_id = _parse_worker_id(tracker_file)
        if other_id is None:
            stored_id = _extract_worker_id(tracker_file)
            invalid_trackers.append((stored_id or "unknown", str(tracker_file)))
            continue
        if other_id == worker_id:
            continue  # Skip our own tracker

        # Check if the other worker's process is alive
        status = check_tracker_alive(str(tracker_file))
        if status == "invalid":
            invalid_trackers.append((other_id, str(tracker_file)))
            continue
        if status != "alive":
            continue  # Skip dead/invalid trackers

        ok, files = _load_tracker_files(tracker_file, other_id)
        if not ok:
            invalid_trackers.append((other_id, str(tracker_file)))
            continue

        overlapping = sorted(staged_set & set(files))

        for f in overlapping:
            conflicts.append((f, other_id))

    conflicts_sorted = sorted(conflicts, key=lambda item: (item[1], item[0]))
    invalid_sorted = sorted(invalid_trackers, key=lambda item: (item[0], item[1]))
    return {
        "conflicts": conflicts_sorted,
        "invalid_trackers": invalid_sorted,
        "ok": len(conflicts_sorted) == 0 and len(invalid_sorted) == 0,
    }


def clear_committed(tracker_file: str, committed_files: list[str]) -> int:
    """Remove committed files from tracker.

    Also increments commit_count (#3202) to track session productivity.

    Returns number of files removed.
    """
    if not os.path.exists(tracker_file):
        return 0

    try:
        with open(tracker_file) as f:
            data = json.load(f)

        # Handle malformed tracker files (e.g., empty array []) - Part of #2383
        if not isinstance(data, dict):
            return 0

        original_files = set(data.get("files", []))
        committed_set = set(f.strip() for f in committed_files if f.strip())
        remaining_files = sorted(original_files - committed_set)

        removed = len(original_files) - len(remaining_files)
        if removed > 0:
            data["files"] = remaining_files
            # Increment commit_count (#3202)
            data["commit_count"] = data.get("commit_count", 0) + 1
            atomic_write_json(Path(tracker_file), data)

        return removed
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return 0


def main() -> None:
    """CLI interface for shell scripts."""
    if len(sys.argv) < 2:
        print("Usage: tracker_utils.py <command> [args...]", file=sys.stderr)
        print("Commands: check_alive, check_ownership, clear_committed", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "check_alive":
        if len(sys.argv) != 3:
            print("Usage: check_alive <tracker_file>", file=sys.stderr)
            sys.exit(1)
        result = check_tracker_alive(sys.argv[2])
        print(result)

    elif command == "check_ownership":
        if len(sys.argv) < 4:
            print("Usage: check_ownership <repo_root> <worker_id> [staged_files...]", file=sys.stderr)
            sys.exit(1)
        repo_root = sys.argv[2]
        worker_id = sys.argv[3]
        staged_files = sys.argv[4:] if len(sys.argv) > 4 else []

        result = check_ownership(repo_root, worker_id, staged_files)
        if result["ok"]:
            print("OK")
        else:
            for wid, tracker_file in result.get("invalid_trackers", []):
                print(f"INVALID:{wid}:{tracker_file}")
            for f, owner in result["conflicts"]:
                print(f"CONFLICT:{owner}:{f}")
            sys.exit(1)

    elif command == "clear_committed":
        if len(sys.argv) < 3:
            print(
                "Usage: clear_committed <tracker_file> [committed_files...] "
                "| clear_committed --stdin <tracker_file>",
                file=sys.stderr,
            )
            sys.exit(1)
        if sys.argv[2] == "--stdin":
            if len(sys.argv) != 4:
                print("Usage: clear_committed --stdin <tracker_file>", file=sys.stderr)
                sys.exit(1)
            tracker_file = sys.argv[3]
            committed_files = [line.rstrip("\n") for line in sys.stdin if line.strip()]
        else:
            tracker_file = sys.argv[2]
            committed_files = sys.argv[3:] if len(sys.argv) > 3 else []

        removed = clear_committed(tracker_file, committed_files)
        if removed > 0:
            print(f"Cleared {removed} file(s)", file=sys.stderr)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

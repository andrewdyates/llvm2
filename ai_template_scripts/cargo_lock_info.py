#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""cargo_lock_info.py - Diagnose the org-wide serialized cargo lock.

Purpose:
  Print who holds a `~/.ait_cargo_lock` lock (PID, command, start time, etc) and
  whether the lock appears stale. Intended as an AI-safe diagnostic tool for #433.

Public API (library usage):
    from ai_template_scripts.cargo_lock_info import (
        STALE_LOCK_AGE_SEC,  # Threshold for stale lock detection
        pid_alive,           # Check if process is running
        parse_etime,         # Parse ps etime format to seconds
        run_ps,              # Get ps fields for a PID
        format_duration,     # Format seconds as human-readable
    )

CLI usage:
    ./cargo_lock_info.py                 # Print build lock holder info
    ./cargo_lock_info.py --kind test     # Inspect test lock
    ./cargo_lock_info.py --json          # Emit JSON and exit

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
License: Apache-2.0
"""

from __future__ import annotations

__all__ = [
    "STALE_LOCK_AGE_SEC",
    "pid_alive",
    "parse_etime",
    "run_ps",
    "format_duration",
    "main",
]

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.version import get_version  # noqa: E402

STALE_LOCK_AGE_SEC = 2 * 60 * 60


def log(msg: str) -> None:
    """Print message to stdout."""
    print(msg)


def eprint(msg: str) -> None:
    """Print message to stderr."""
    print(msg, file=sys.stderr)


def read_text(path: Path) -> str | None:
    """Read file as stripped text, or None on error."""
    try:
        return path.read_text().strip()
    except Exception:
        return None  # Best-effort: file read for diagnostics, None is safe default


def read_json(path: Path) -> dict | None:
    """Load JSON from file, or None on error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None  # Best-effort: JSON parse for diagnostics, None is safe default


def parse_iso8601(ts: str) -> datetime | None:
    """Parse ISO8601 timestamp to datetime, or None on error."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return (
            None  # Best-effort: timestamp parse for diagnostics, None is safe default
        )


def format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration (e.g. 1h23m45s)."""
    seconds = max(0, int(seconds))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d{hrs:02}h{mins:02}m{sec:02}s"
    if hrs:
        return f"{hrs}h{mins:02}m{sec:02}s"
    if mins:
        return f"{mins}m{sec:02}s"
    return f"{sec}s"


def pid_alive(pid: int) -> bool:
    """Check if process with PID is alive using kill(0)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def parse_etime(etime: str) -> int:
    """Parse ps etime format to seconds (MM:SS, HH:MM:SS, or DD-HH:MM:SS)."""
    try:
        etime = etime.strip()
        days = 0
        if "-" in etime:
            days_s, etime = etime.split("-", 1)
            days = int(days_s)
        parts = etime.split(":")
        if len(parts) == 2:
            mm, ss = (int(x) for x in parts)
            return days * 86400 + mm * 60 + ss
        if len(parts) == 3:
            hh, mm, ss = (int(x) for x in parts)
            return days * 86400 + hh * 3600 + mm * 60 + ss
    except Exception:
        pass  # Best-effort: etime parse for diagnostics, 0 is safe default
    return 0


def run_ps(pid: int) -> dict | None:
    """Return a dict with basic ps fields for pid, or None if unavailable."""
    # Keep to fields supported by macOS + Linux ps.
    fmt = "pid=,ppid=,pgid=,stat=,etime=,command="
    cmd = ["ps", "-p", str(pid), "-o", fmt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        line = result.stdout.strip()
        if not line:
            return None
        parts = line.split(None, 5)
        if len(parts) < 6:
            return None
        pid_s, ppid_s, pgid_s, stat, etime, command = parts
        return {
            "pid": int(pid_s),
            "ppid": int(ppid_s),
            "pgid": int(pgid_s),
            "stat": stat,
            "etime": etime,
            "etime_sec": parse_etime(etime),
            "command": command,
        }
    except Exception:
        return None  # Best-effort: ps command for diagnostics, None is safe default


def summarize_lock_meta(meta: dict) -> str:
    """Format lock metadata dict as key=value summary string."""
    fields = []
    for key in ("project", "role", "session", "iteration", "commit"):
        val = meta.get(key)
        if val:
            fields.append(f"{key}={val}")
    return " ".join(fields) if fields else "(no metadata)"


def lock_basename(lock_kind: str) -> str:
    """Return lock filename prefix for given lock kind."""
    if lock_kind == "build":
        return "lock"
    if lock_kind == "test":
        return "lock.test"
    return f"lock.{lock_kind}"


def main() -> int:
    """CLI entry point: diagnose cargo lock holder status.

    REQUIRES: called as script or module entry point
    ENSURES: returns 0 (no lock/success), 2 (lock held by alive process),
             or 3 (stale lock detected)
    """
    parser = argparse.ArgumentParser(
        description="Inspect ~/.ait_cargo_lock lock holder"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("cargo_lock_info.py"),
    )
    parser.add_argument(
        "--lock-dir",
        default=str(Path.home() / ".ait_cargo_lock"),
        help="Lock directory (default: ~/.ait_cargo_lock)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON and exit")
    parser.add_argument(
        "--kind",
        choices=["build", "test"],
        default="build",
        help="Lock kind to inspect (default: build)",
    )
    args = parser.parse_args()

    lock_dir = Path(args.lock_dir).expanduser()
    basename = lock_basename(args.kind)
    lock_pid_path = lock_dir / f"{basename}.pid"
    lock_meta_path = lock_dir / f"{basename}.json"

    meta = read_json(lock_meta_path) or {}

    pid: int | None = None
    if isinstance(meta.get("pid"), int):
        pid = meta["pid"]
    else:
        pid_txt = read_text(lock_pid_path)
        if pid_txt and pid_txt.isdigit():
            pid = int(pid_txt)

    acquired_at = meta.get("acquired_at")
    acquired_dt = parse_iso8601(acquired_at) if isinstance(acquired_at, str) else None
    now = datetime.now(UTC)
    lock_age_sec = int((now - acquired_dt).total_seconds()) if acquired_dt else None

    alive = pid_alive(pid) if pid is not None else False
    ps_info = run_ps(pid) if (pid is not None and alive) else None

    stale = False
    if pid is None:
        stale = False
    elif not alive:
        stale = True
    elif lock_age_sec is not None and lock_age_sec >= STALE_LOCK_AGE_SEC:
        stale = True

    out = {
        "lock_dir": str(lock_dir),
        "lock_kind": args.kind,
        "lock_pid_path": str(lock_pid_path),
        "lock_meta_path": str(lock_meta_path),
        "pid": pid,
        "alive": alive,
        "stale": stale,
        "stale_after_sec": STALE_LOCK_AGE_SEC,
        "meta": meta,
        "ps": ps_info,
        "lock_age_sec": lock_age_sec,
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        log(f"[cargo-lock] dir: {lock_dir}")
        log(f"[cargo-lock] kind: {args.kind}")
        if pid is None:
            log("[cargo-lock] status: no lock.pid/lock.json PID found")
            return 0

        log(f"[cargo-lock] pid: {pid} (alive={alive})")
        if acquired_dt is not None and lock_age_sec is not None:
            log(
                "[cargo-lock] acquired: "
                f"{acquired_dt.isoformat(timespec='seconds')} "
                f"(age={format_duration(lock_age_sec)})"
            )
        if meta:
            log(f"[cargo-lock] meta: {summarize_lock_meta(meta)}")

        if ps_info is not None:
            ps = ps_info
            log(
                f"[cargo-lock] ps: pid={ps['pid']} ppid={ps['ppid']} "
                f"pgid={ps['pgid']} stat={ps['stat']} "
                f"etime={ps['etime']} cmd={ps['command']}"
            )

        if stale:
            log("[cargo-lock] stale: yes")
            log(
                "[cargo-lock] hint: if PID is dead, removing lock files is safe; "
                "otherwise the holder must exit/timeout."
            )
            return 3

        log("[cargo-lock] stale: no")
        return 2 if alive else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

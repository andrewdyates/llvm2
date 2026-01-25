#!/usr/bin/env python3
"""cargo_lock_info.py - Diagnose the org-wide serialized cargo lock.

Purpose:
  Print who holds `~/.ait_cargo_lock` (PID, command, start time, etc) and whether
  the lock appears stale. Intended as an AI-safe diagnostic tool for #433.

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
License: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STALE_LOCK_AGE_SEC = 2 * 60 * 60


def log(msg: str) -> None:
    print(msg)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except Exception:
        return None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_iso8601(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def format_duration(seconds: int) -> str:
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
        pass
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
        return None


def summarize_lock_meta(meta: dict) -> str:
    fields = []
    for key in ("project", "role", "session", "iteration", "commit"):
        val = meta.get(key)
        if val:
            fields.append(f"{key}={val}")
    return " ".join(fields) if fields else "(no metadata)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect ~/.ait_cargo_lock lock holder"
    )
    parser.add_argument(
        "--lock-dir",
        default=str(Path.home() / ".ait_cargo_lock"),
        help="Lock directory (default: ~/.ait_cargo_lock)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON and exit")
    args = parser.parse_args()

    lock_dir = Path(args.lock_dir).expanduser()
    lock_pid_path = lock_dir / "lock.pid"
    lock_meta_path = lock_dir / "lock.json"

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
    now = datetime.now(timezone.utc)
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
        if pid is None:
            log("[cargo-lock] status: no lock.pid/lock.json PID found")
            return 0

        log(f"[cargo-lock] pid: {pid} (alive={alive})")
        if acquired_dt is not None and lock_age_sec is not None:
            log(
                f"[cargo-lock] acquired: {acquired_dt.isoformat(timespec='seconds')} (age={format_duration(lock_age_sec)})"
            )
        if meta:
            log(f"[cargo-lock] meta: {summarize_lock_meta(meta)}")

        if ps_info is not None:
            log(
                f"[cargo-lock] ps: pid={ps_info['pid']} ppid={ps_info['ppid']} pgid={ps_info['pgid']} "
                f"stat={ps_info['stat']} etime={ps_info['etime']} cmd={ps_info['command']}"
            )

        if stale:
            log("[cargo-lock] stale: yes")
            log(
                "[cargo-lock] hint: if PID is dead, removing lock files is safe; otherwise the holder must exit/timeout."
            )
            return 3

        log("[cargo-lock] stale: no")
        return 2 if alive else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

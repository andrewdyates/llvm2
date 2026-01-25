# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Status, logging, and pulse management for looper."""

import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from looper.config import LOG_DIR, MAX_CRASH_LOG_LINES, MAX_LOG_FILES, get_project_name


class StatusManager:
    """Handle status files, log rotation, pulse checks, and crash logging."""

    def __init__(
        self,
        repo_path: Path,
        mode: str,
        status_file: Path,
        crash_log: Path,
        config: dict[str, Any],
        get_ait_version: Callable[[], tuple[str, str] | None],
        log_dir: Path = LOG_DIR,
    ):
        self.repo_path = repo_path.resolve()
        self.mode = mode
        self.status_file = self._anchor_path(status_file)
        self.crash_log = self._anchor_path(crash_log)
        self.config = config
        self.log_dir = self._anchor_path(log_dir)
        self._get_ait_version = get_ait_version
        self._last_pulse_time = 0.0
        self._started_at: str | None = None

    def _anchor_path(self, path: Path) -> Path:
        """Anchor relative paths to the repo root."""
        if path.is_absolute():
            return path
        return self.repo_path / path

    def set_started_at(self, started_at: str) -> None:
        """Store session start time for status updates."""
        self._started_at = started_at

    def run_pulse(self) -> bool:
        """Run pulse.py to update health metrics and flags."""
        interval_minutes = self.config.get("pulse_interval_minutes", 30)
        now = time.time()

        # Check if enough time has passed
        elapsed_minutes = (now - self._last_pulse_time) / 60
        if self._last_pulse_time > 0 and elapsed_minutes < interval_minutes:
            return False

        # Run pulse.py as subprocess (quiet mode - just writes files)
        pulse_script = self._anchor_path(Path("ai_template_scripts/pulse.py"))
        if not pulse_script.exists():
            return False

        try:
            result = subprocess.run(
                ["python3", str(pulse_script)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._last_pulse_time = now

            if result.returncode == 0:
                # Check for flags set
                flags_dir = self._anchor_path(Path(".flags"))
                flags = list(flags_dir.glob("*")) if flags_dir.exists() else []
                if flags:
                    flag_names = ", ".join(f.name for f in flags[:5])
                    print(f"⚡ Pulse: {flag_names}")
                else:
                    print("⚡ Pulse: OK (no flags)")
                return True
            print(
                f"⚠️  Pulse error: {result.stderr[:100] if result.stderr else 'unknown'}"
            )
            return False
        except subprocess.TimeoutExpired:
            print("⚠️  Pulse timeout")
            return False
        except Exception as e:
            print(f"⚠️  Pulse failed: {e}")
            return False

    def rotate_logs(self) -> None:
        """Remove old log files to prevent unbounded growth."""
        log_files = sorted(
            self.log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime
        )
        if len(log_files) > MAX_LOG_FILES:
            for f in log_files[:-MAX_LOG_FILES]:
                f.unlink()
            print(f"Rotated logs: removed {len(log_files) - MAX_LOG_FILES} old files")

        # Rotate crash log
        if self.crash_log.exists():
            lines = self.crash_log.read_text().splitlines()
            if len(lines) > MAX_CRASH_LOG_LINES:
                self.crash_log.write_text(
                    "\n".join(lines[-MAX_CRASH_LOG_LINES:]) + "\n"
                )

    def scrub_log_file(self, log_file: Path) -> bool:
        """Scrub secrets from log file if scrub_logs config is enabled."""
        if not self.config.get("scrub_logs", False):
            return False

        scrubber = self._anchor_path(Path("ai_template_scripts/log_scrubber.py"))
        if not scrubber.exists():
            print("⚠️  Log scrubber not found, skipping scrub")
            return False

        try:
            result = subprocess.run(
                ["python3", str(scrubber), str(log_file)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                print(f"=== Log scrubbed: {log_file.name} ===")
                return True
            print(f"⚠️  Log scrub error: {result.stderr[:100]}")
            return False
        except subprocess.TimeoutExpired:
            print("⚠️  Log scrub timeout")
            return False
        except Exception as e:
            print(f"⚠️  Log scrub failed: {e}")
            return False

    def write_status(
        self,
        iteration: int,
        status: str,
        log_file: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write worker status to a JSON file for manager visibility."""
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "pid": os.getpid(),
            "mode": self.mode,
            "project": get_project_name(),
            "iteration": iteration,
            "status": status,
            "updated_at": now,
            "started_at": self._started_at or now,
        }
        # Include AIT version for tracking
        ait_version = self._get_ait_version()
        if ait_version:
            data["ait_version"] = ait_version[0]
            data["ait_synced"] = ait_version[1]
        if log_file:
            data["log_file"] = str(log_file)
        if extra:
            data.update(extra)

        # Atomic write
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self.status_file)

    def clear_status(self) -> None:
        """Remove status file on clean exit."""
        if self.status_file.exists():
            self.status_file.unlink()

    def log_crash(
        self,
        iteration: int,
        exit_code: int,
        ai_tool: str,
        session_committed: bool = False,
    ) -> None:
        """Log crash or exit details."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if exit_code > 128:
            signal_num = exit_code - 128
            error_msg = f"{ai_tool} killed by signal {signal_num}"
            is_real_crash = True
        elif exit_code == 124:
            error_msg = f"{ai_tool} timed out"
            is_real_crash = not session_committed  # Timeout after commit = not a crash
        elif exit_code == 125:
            error_msg = f"{ai_tool} killed due to silence (stale connection)"
            is_real_crash = False  # Expected after sleep/resume, will restart cleanly
        else:
            error_msg = f"{ai_tool} exited with code {exit_code}"
            # Exit code 1 after successful commit = likely EPIPE or graceful exit
            is_real_crash = not session_committed

        if session_committed and not is_real_crash:
            # Session completed work - this is not a crash
            print()
            print(
                f"Note: {ai_tool} exited with code {exit_code} but session committed successfully"
            )
            print()
            return  # Don't log to crashes.log

        with open(self.crash_log, "a") as f:
            f.write(f"[{timestamp}] Iteration {iteration}: {error_msg}\n")

        print()
        print(f"*** {self.mode.upper()} CRASH: {error_msg}")
        print(f"    Log: {self.crash_log}")
        print()

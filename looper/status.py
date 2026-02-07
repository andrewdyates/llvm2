# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Status, logging, and pulse management for looper.

Use via LoopRunner/StatusManager for loop lifecycle status and log handling.
Direct use is intended for tests or tools that need status updates without
starting the full loop. Thread safety: not thread-safe; single-process usage.

Module contracts:
    ENSURES: Status file writes are atomic (temp + rename) when they succeed
    ENSURES: Log rotation attempts to prevent unbounded growth
    ENSURES: All methods handle filesystem errors gracefully
"""

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from looper.config import (
    LOG_DIR,
    LOG_RETENTION_HOURS,
    MAX_CRASH_LOG_LINES,
    MAX_LOG_FILES,
    get_project_name,
)
from looper.constants import FLAGS_DIR
from looper.log import debug_swallow, log_error, log_info, log_warning
from looper.memory_logger import get_memory_state, is_oom_signal

__all__ = [
    "StatusManager",
    "StatusManagerConfig",
]


class StatusManagerConfig(TypedDict, total=False):
    """Configuration options for StatusManager.

    All fields are optional with defaults in StatusManager methods.
    See StatusManager class docstring for contract details.
    """

    pulse_interval_minutes: int  # Default: 30
    scrub_logs: bool  # Default: False

# --- Diagnostic flag enhancement constants (backported from dasher #1638) ---

_DEFAULT_STUCK_PROCESS_THRESHOLD_MINUTES = 120

_STUCK_PROCESS_WATCH_KEYWORDS = (
    "cargo",
    "rustc",
    "cbmc",
    "kani",
    "z3",
    "cvc5",
    "lean",
    "python",
)

_STUCK_PROCESS_EXCLUDE_SUBSTRINGS = (
    "looper.py",
    "looper.issue_manager",
    "looper.telemetry",
    "spawn_session",
    "spawn_all",
    "pytest",
    "multiprocessing.resource_tracker",
    "multiprocessing.spawn",
    ".claude/plugins",
)

# --- Pulse coordination constants (#2444) ---
# When multiple sessions start simultaneously, only one should run pulse.
# Others wait for lock, then use cached results if fresh enough.

PULSE_CACHE_FRESHNESS_SEC = 60  # Cache valid for startup coordination
PULSE_LOCK_TIMEOUT_SEC = 180  # Max wait for lock (same as pulse subprocess timeout)
PULSE_LOCK_FILENAME = ".pulse.lock"  # Lock file in metrics/ directory

# Cross-platform file locking for pulse coordination
try:
    import fcntl

    def _pulse_lock_file(fd: int) -> None:
        """Acquire exclusive lock on file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _pulse_unlock_file(fd: int) -> None:
        """Release lock on file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _pulse_try_lock(fd: int, timeout_sec: float) -> bool:
        """Attempt to acquire lock with timeout. Returns True if acquired."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                # Lock held by another process, wait briefly
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                time.sleep(min(0.5, remaining))  # Longer sleep than gh wrapper (0.1s)
        return False

except ImportError:
    # Windows - no fcntl, skip locking
    def _pulse_lock_file(fd: int) -> None:  # type: ignore[misc]
        pass

    def _pulse_unlock_file(fd: int) -> None:  # type: ignore[misc]
        pass

    def _pulse_try_lock(fd: int, timeout_sec: float) -> bool:  # type: ignore[misc]
        return True


def _check_pulse_cache_fresh(metrics_dir: Path) -> bool:
    """Check if pulse cache (metrics/latest.json) is fresh enough.

    Contracts:
        REQUIRES: metrics_dir is a valid Path
        ENSURES: Returns True if latest.json exists and was modified within PULSE_CACHE_FRESHNESS_SEC
        ENSURES: Returns False if file missing, unreadable, or too old
    """
    cache_file = metrics_dir / "latest.json"
    try:
        if not cache_file.exists():
            return False
        mtime = cache_file.stat().st_mtime
        age_sec = time.time() - mtime
        return age_sec < PULSE_CACHE_FRESHNESS_SEC
    except OSError:
        return False


def _parse_ps_etime_minutes(etime: str) -> int:
    """Parse ps etime format ([[DD-]HH:]MM:SS) into whole minutes.

    Contracts:
        REQUIRES: etime is a string in ps elapsed time format
        ENSURES: Returns integer minutes, 0 on parse failure
    """
    try:
        days = 0
        if "-" in etime:
            days_part, etime = etime.split("-", 1)
            days = int(days_part)

        parts = etime.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            hours = 0
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
        else:
            return 0

        return days * 24 * 60 + hours * 60 + minutes
    except (ValueError, IndexError):
        return 0


def _format_stuck_process_flag(
    *,
    repo_path: Path,
    ps_stdout: str,
    threshold_minutes: int,
    max_entries: int = 5,
) -> str | None:
    """Build an actionable `.flags/stuck_process` file body from `ps` output.

    Contracts:
        REQUIRES: repo_path is a valid Path
        REQUIRES: ps_stdout is the output from `ps -eo pid,etime,args`
        REQUIRES: threshold_minutes > 0
        ENSURES: Returns None if no stuck processes found
        ENSURES: Returns formatted string with timestamp, repo, and process details
    """
    repo_str = str(repo_path)
    candidates: list[tuple[int, str, int, str]] = []

    for line in (ps_stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Header formats vary across platforms
        if stripped.upper().startswith("PID "):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, etime, args = parts[0], parts[1], parts[2]

        minutes = _parse_ps_etime_minutes(etime)
        if minutes < threshold_minutes:
            continue

        args_lower = args.lower()
        if not any(k in args_lower for k in _STUCK_PROCESS_WATCH_KEYWORDS):
            continue
        if any(s in args_lower for s in _STUCK_PROCESS_EXCLUDE_SUBSTRINGS):
            continue

        # Repo attribution: only show processes whose args include this repo root
        idx = args.find(repo_str)
        if idx < 0:
            continue
        end_idx = idx + len(repo_str)
        if end_idx < len(args) and args[end_idx] not in (" ", "/"):
            continue

        try:
            pid = int(pid_str)
        except ValueError:
            continue

        candidates.append((minutes, etime, pid, args))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    lines = [datetime.now(UTC).isoformat(), f"repo={repo_str}"]
    for minutes, etime, pid, args in candidates[:max_entries]:
        lines.append(f"  pid={pid} etime={etime} mins={minutes} cmd={args}")
    if len(candidates) > max_entries:
        lines.append(f"  (+{len(candidates) - max_entries} more)")
    return "\n".join(lines) + "\n"


def _maybe_rewrite_stuck_process_flag(*, repo_path: Path, flags_dir: Path) -> None:
    """Rewrite .flags/stuck_process with actionable process details.

    Reads threshold from pulse.toml if available.

    Contracts:
        REQUIRES: repo_path and flags_dir are valid Paths
        ENSURES: Rewrites stuck_process flag if it exists with process details
        ENSURES: Never raises - catches all exceptions
    """
    threshold_minutes = _DEFAULT_STUCK_PROCESS_THRESHOLD_MINUTES
    for rel in ("pulse.toml", "ai_template_scripts/pulse.toml", ".pulse.toml"):
        if tomllib is None:
            break
        config_path = repo_path / rel
        if not config_path.exists():
            continue
        try:
            config = tomllib.loads(config_path.read_text())
        except Exception:
            continue
        thresholds = config.get("thresholds", {})
        candidate = thresholds.get("long_running_process_minutes")
        if isinstance(candidate, int) and candidate > 0:
            threshold_minutes = candidate
            break

    stuck = flags_dir / "stuck_process"
    if not stuck.exists():
        return

    try:
        ps = subprocess.run(
            # Force wide output so long command lines aren't truncated
            ["ps", "-eo", "pid,etime,args", "-ww"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if ps.returncode != 0:
            return

        rewritten = _format_stuck_process_flag(
            repo_path=repo_path,
            ps_stdout=ps.stdout,
            threshold_minutes=threshold_minutes,
        )
        if rewritten is not None:
            stuck.write_text(rewritten)
    except Exception:
        debug_swallow("rewrite_stuck_process_flag")


def _read_file_tail_text(path: Path, *, max_bytes: int) -> str:
    """Read the last max_bytes of a file as text.

    Contracts:
        REQUIRES: path is a valid Path
        REQUIRES: max_bytes > 0
        ENSURES: Returns UTF-8 decoded text, empty string on error
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ""
            start = max(0, size - max_bytes)
            f.seek(start, os.SEEK_SET)
            data = f.read()
    except OSError:
        return ""

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return ""

    # If we started mid-line, drop the partial first line
    if start > 0:
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
    return text


def _summarize_worker_log_for_untraceable_flag(
    *, repo_path: Path, log_path_text: str
) -> dict[str, str]:
    """Best-effort log summary for rewriting `.flags/untraceable_failures`.

    Contracts:
        REQUIRES: repo_path is a valid Path
        REQUIRES: log_path_text is a path string (absolute or relative)
        ENSURES: Returns dict with keys like 'terminated', 'silence_kind', 'last', 'tool', 'hint'
    """
    log_path = Path(log_path_text)
    if not log_path.is_absolute():
        log_path = repo_path / log_path

    out: dict[str, str] = {}
    if not log_path.exists():
        out["log_missing"] = "1"
        return out

    tail = _read_file_tail_text(log_path, max_bytes=256_000)
    if not tail:
        out["log_empty"] = "1"
        return out

    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        out["log_empty"] = "1"
        return out

    # Termination reason + silence kind (used for exit=125 infra cases)
    for ln in reversed(lines):
        if '"event":"terminated"' not in ln and '"event": "terminated"' not in ln:
            continue
        try:
            parsed = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("type") != "system" or parsed.get("event") != "terminated":
            continue
        reason = parsed.get("reason")
        silence_kind = parsed.get("silence_kind")
        if isinstance(reason, str) and reason:
            out["terminated"] = reason
        if isinstance(silence_kind, str) and silence_kind:
            out["silence_kind"] = silence_kind
        break

    # Last event kind
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        last = None
    if isinstance(last, dict):
        last_type = last.get("type")
        last_event = last.get("event")
        if last_event is None:
            last_event = last.get("subtype")
        if isinstance(last_type, str) and isinstance(last_event, str):
            out["last"] = f"{last_type}.{last_event}"
        elif isinstance(last_type, str):
            out["last"] = last_type
        elif isinstance(last_event, str):
            out["last"] = last_event

    # Last tool name (if present)
    for ln in reversed(lines):
        if '"tool_use"' not in ln:
            continue
        try:
            parsed = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("type") != "assistant":
            continue
        msg = parsed.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            tool_name = item.get("name")
            if isinstance(tool_name, str) and tool_name:
                out["tool"] = tool_name
                break
        if "tool" in out:
            break

    if "[cargo-lock] Waiting for" in tail or "[cargo] Waiting for" in tail:
        out["hint"] = "cargo_build_lock_wait"
        out["cargo_lock"] = f"~/.ait_cargo_lock/{repo_path.name}/lock.json"

    if "[git commit lock]" in tail:
        out["hint"] = "git_commit_lock_wait"
        out["git_lock"] = f"{repo_path}/.git/ait_commit_lock/"

    try:
        if (
            log_path.stat().st_size <= 256_000
            and len(lines) == 1
            and out.get("last") == "system.init"
        ):
            out["hint"] = "stuck_at_init"
    except OSError:
        pass

    return out


def _format_untraceable_failures_flag(*, repo_path: Path, flag_text: str) -> str | None:
    """Rewrite `.flags/untraceable_failures` with actionable context for exit=125.

    Contracts:
        REQUIRES: repo_path is a valid Path
        REQUIRES: flag_text is the current flag file content
        ENSURES: Returns None if flag_text is empty
        ENSURES: Returns rewritten flag with parsed log details appended
    """
    lines_in = [ln.rstrip("\n") for ln in (flag_text or "").splitlines()]
    if not lines_in:
        return None

    count_line = None
    remaining: list[str] = []
    for ln in lines_in[1:]:
        if ln.startswith("Count:"):
            count_line = ln
            continue
        remaining.append(ln)

    out_lines = [datetime.now(UTC).isoformat(), f"repo={repo_path}"]
    if count_line is not None:
        out_lines.append(count_line)

    for ln in remaining:
        stripped = ln.strip()
        if not stripped:
            continue

        if stripped.startswith("(+") and stripped.endswith("more)"):
            out_lines.append(f"  {stripped}")
            continue

        if "session=" not in ln or "iter=" not in ln or "exit=" not in ln:
            out_lines.append(ln)
            continue

        tokens = stripped.split()
        base_tokens: list[str] = [
            tok
            for tok in tokens
            if tok.startswith(
                ("session=", "iter=", "exit=", "worker=", "silence=", "log=")
            )
        ]

        m_exit = None
        try:
            m_exit = int(
                next(
                    tok.split("=", 1)[1]
                    for tok in base_tokens
                    if tok.startswith("exit=")
                )
            )
        except Exception:
            m_exit = None

        log_path_text = None
        for tok in base_tokens:
            if tok.startswith("log="):
                log_path_text = tok.split("=", 1)[1]
                break

        if m_exit != 125 or not isinstance(log_path_text, str) or not log_path_text:
            out_lines.append(ln)
            continue

        summary = _summarize_worker_log_for_untraceable_flag(
            repo_path=repo_path, log_path_text=log_path_text
        )
        parts = base_tokens
        if "terminated" in summary:
            parts.append(f"terminated={summary['terminated']}")
        if "silence_kind" in summary:
            parts.append(f"silence_kind={summary['silence_kind']}")
        if "last" in summary:
            parts.append(f"last={summary['last']}")
        if "tool" in summary:
            parts.append(f"tool={summary['tool']}")
        if summary.get("hint"):
            parts.append(f"hint={summary['hint']}")
            if summary.get("cargo_lock"):
                parts.append(f"cargo_lock={summary['cargo_lock']}")
            if summary.get("git_lock"):
                parts.append(f"git_lock={summary['git_lock']}")
        out_lines.append("  " + " ".join(parts))

    return "\n".join(out_lines) + "\n"


def _maybe_rewrite_untraceable_failures_flag(*, repo_path: Path, flags_dir: Path) -> None:
    """Rewrite .flags/untraceable_failures with actionable context.

    Contracts:
        REQUIRES: repo_path and flags_dir are valid Paths
        ENSURES: Rewrites untraceable_failures flag if it exists with log details
        ENSURES: Never raises - catches all exceptions
    """
    path = flags_dir / "untraceable_failures"
    if not path.exists():
        return
    try:
        rewritten = _format_untraceable_failures_flag(
            repo_path=repo_path, flag_text=path.read_text()
        )
        if rewritten is not None:
            path.write_text(rewritten)
    except Exception:
        debug_swallow("rewrite_untraceable_failures_flag")


class StatusManager:
    """Handle status files, log rotation, pulse checks, and crash logging.

    Contracts:
        REQUIRES: repo_path is a valid Path
        REQUIRES: mode is a non-empty string (e.g., "worker", "manager")
        REQUIRES: status_file and crash_log are Paths (relative to repo_path OK)
        REQUIRES: config is StatusManagerConfig (TypedDict with pulse_interval_minutes, scrub_logs)
        REQUIRES: get_ait_version is a callable returning (version, synced) or None
        ENSURES: All paths resolved relative to repo_path
        ENSURES: Thread-unsafe - single process usage only
    """

    def __init__(
        self,
        repo_path: Path,
        mode: str,
        status_file: Path,
        crash_log: Path,
        config: StatusManagerConfig,
        get_ait_version: Callable[[], tuple[str, str] | None],
        log_dir: Path = LOG_DIR,
    ) -> None:
        """Initialize StatusManager.

        Contracts:
            REQUIRES: All parameters as documented in class docstring
            ENSURES: self.repo_path is absolute
            ENSURES: status_file, crash_log, log_dir anchored to repo_path
            ENSURES: _last_pulse_time initialized to 0.0
            ENSURES: _started_at initialized to None
        """
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
        """Anchor relative paths to the repo root.

        Contracts:
            REQUIRES: path is a Path object
            ENSURES: Returns absolute path unchanged
            ENSURES: Returns relative path joined with self.repo_path
        """
        if path.is_absolute():
            return path
        return self.repo_path / path

    def set_started_at(self, started_at: str) -> None:
        """Store session start time for status updates.

        Contracts:
            REQUIRES: started_at is a string (ISO format preferred)
            ENSURES: self._started_at is set to provided value
        """
        self._started_at = started_at

    def run_pulse(self) -> bool:
        """Run pulse.py to update health metrics and flags.

        Multi-session coordination (#2444): When multiple sessions start
        simultaneously, only one runs pulse.py. Others wait for the lock,
        then use cached results if fresh (< 60s old).

        Contracts:
            REQUIRES: pulse_interval_minutes in config (default 30)
            ENSURES: Returns False if pulse interval not elapsed
            ENSURES: Returns False if pulse script missing
            ENSURES: Returns True on successful pulse execution or cache hit
            ENSURES: Updates self._last_pulse_time on completion
            ENSURES: Prints flag status or error message
            ENSURES: Never raises - catches TimeoutExpired and all exceptions
        """
        interval_minutes = self.config.get("pulse_interval_minutes", 30)
        now = time.time()

        # Check if enough time has passed (per-session interval)
        elapsed_minutes = (now - self._last_pulse_time) / 60
        if self._last_pulse_time > 0 and elapsed_minutes < interval_minutes:
            return False

        # Run pulse.py as subprocess (quiet mode - just writes files)
        pulse_script = self._anchor_path(Path("ai_template_scripts/pulse.py"))
        if not pulse_script.exists():
            return False

        metrics_dir = self._anchor_path(Path("metrics"))

        # Phase 1: Pre-lock cache check (#2444)
        # If another session just ran pulse, use cached results
        if _check_pulse_cache_fresh(metrics_dir):
            self._last_pulse_time = now
            return self._report_pulse_flags(cached=True)

        # Phase 2: Acquire lock and run pulse
        return self._run_pulse_with_lock(pulse_script, metrics_dir, now)

    def _run_pulse_with_lock(
        self, pulse_script: Path, metrics_dir: Path, now: float
    ) -> bool:
        """Run pulse.py with coordination lock (#2444, #2685).

        Emits progress every 10s during lock wait to prevent silent gaps.

        Contracts:
            REQUIRES: pulse_script exists
            REQUIRES: metrics_dir is a valid Path
            ENSURES: Only one session runs pulse at a time
            ENSURES: Sessions waiting for lock get cached results when available
            ENSURES: No silent gap >10s during lock wait (#2685)
            ENSURES: Never raises - catches all exceptions
        """
        lock_path = metrics_dir / PULSE_LOCK_FILENAME
        lock_fd = None

        try:
            # Ensure metrics directory exists for lock file
            metrics_dir.mkdir(exist_ok=True)
            lock_fd = open(lock_path, "w")

            # Try to acquire lock in 10s intervals with progress (#2685)
            # Total timeout matches PULSE_LOCK_TIMEOUT_SEC (180s)
            lock_attempt_sec = 10
            attempts = 0
            max_attempts = int(PULSE_LOCK_TIMEOUT_SEC / lock_attempt_sec)
            got_lock = False

            while attempts < max_attempts:
                got_lock = _pulse_try_lock(lock_fd.fileno(), lock_attempt_sec)
                if got_lock:
                    break
                attempts += 1
                # Check if cache became fresh while waiting
                if _check_pulse_cache_fresh(metrics_dir):
                    self._last_pulse_time = now
                    return self._report_pulse_flags(cached=True)
                waited = attempts * lock_attempt_sec
                log_info(
                    f"  pulse: waiting for lock ({waited}s / "
                    f"{PULSE_LOCK_TIMEOUT_SEC}s)..."
                )

            if not got_lock:
                # Timeout waiting for lock - check if cache is now fresh
                if _check_pulse_cache_fresh(metrics_dir):
                    self._last_pulse_time = now
                    return self._report_pulse_flags(cached=True)
                log_warning("⚠️  Pulse lock timeout, no fresh cache")
                return False

            # Phase 3: Double-check cache after lock acquisition (#2444)
            # Another session may have just finished while we waited
            if _check_pulse_cache_fresh(metrics_dir):
                self._last_pulse_time = now
                return self._report_pulse_flags(cached=True)

            # Phase 4: We're the one - run pulse
            return self._execute_pulse(pulse_script, now)

        except OSError as e:
            # Lock file operations failed - proceed without coordination
            debug_swallow("pulse_lock", e)
            return self._execute_pulse(pulse_script, now)

        finally:
            # Release lock
            if lock_fd is not None:
                try:
                    _pulse_unlock_file(lock_fd.fileno())
                    lock_fd.close()
                except OSError:
                    pass

    def _execute_pulse(self, pulse_script: Path, now: float) -> bool:
        """Execute pulse.py subprocess, streaming stderr progress in real-time.

        Contracts:
            REQUIRES: pulse_script exists
            ENSURES: Returns True on success, False on failure
            ENSURES: Updates self._last_pulse_time on completion
            ENSURES: Pulse stderr progress lines are emitted via log_info as they arrive
        """
        log_info("Pre-iteration: running pulse health check...")
        timeout_sec = 180
        proc: subprocess.Popen[str] | None = None
        try:
            # Pulse makes multiple sequential GitHub API calls (each up to 60s)
            # Need enough time for: issue counts, blocked, stale, long-blocked,
            # velocity, reopened - plus local scans. 180s covers typical case.
            #
            # Stream stderr so pulse _progress() messages appear in real-time
            # instead of being swallowed by capture_output=True (#2619).
            proc = subprocess.Popen(
                ["python3", str(pulse_script)],
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Stream stderr in a daemon thread so progress lines appear in
            # real-time while proc.wait(timeout=) enforces the deadline.
            # Iterating proc.stderr in the main thread would block until a
            # line arrives, making timeout enforcement unreliable.
            stderr_lines: list[str] = []

            def _drain_stderr() -> None:
                for raw in proc.stderr or ():
                    line = raw.rstrip("\n")
                    if line:
                        stderr_lines.append(line)
                        log_info(f"  pulse: {line.lstrip()}")

            reader = threading.Thread(target=_drain_stderr, daemon=True)
            reader.start()
            proc.wait(timeout=timeout_sec)
            reader.join(timeout=2)  # Give reader a moment to finish
            self._last_pulse_time = now

            if proc.returncode == 0:
                return self._report_pulse_flags(cached=False)
            stderr_text = "\n".join(stderr_lines)
            log_warning(
                f"⚠️  Pulse error: {stderr_text[:100] if stderr_text else 'unknown'}"
            )
            return False
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
                proc.wait()
            log_warning("⚠️  Pulse timeout")
            return False
        except Exception as e:
            log_warning(f"⚠️  Pulse failed: {e}")
            return False

    def _report_pulse_flags(self, *, cached: bool) -> bool:
        """Report pulse status based on flags.

        Contracts:
            REQUIRES: cached is a boolean
            ENSURES: Logs pulse status with (cached) indicator if from cache
            ENSURES: Rewrites diagnostic flags with actionable details
            ENSURES: Returns True
        """
        flags_dir = self._anchor_path(FLAGS_DIR)
        flags = list(flags_dir.glob("*")) if flags_dir.exists() else []

        # Rewrite diagnostic flags with actionable details (#1638)
        if flags_dir.exists():
            try:
                _maybe_rewrite_stuck_process_flag(
                    repo_path=self.repo_path,
                    flags_dir=flags_dir,
                )
                _maybe_rewrite_untraceable_failures_flag(
                    repo_path=self.repo_path,
                    flags_dir=flags_dir,
                )
            except Exception:
                debug_swallow("pulse_rewrite_diagnostic_flags")

        cache_indicator = " (cached)" if cached else ""
        if flags:
            flag_names = ", ".join(f.name for f in flags[:5])
            log_info(f"⚡ Pulse: {flag_names}{cache_indicator}")
        else:
            log_info(f"⚡ Pulse: OK (no flags){cache_indicator}")
        return True

    def rotate_logs(self) -> None:
        """Remove old log files to prevent unbounded growth.

        Preserves recent logs within LOG_RETENTION_HOURS to ensure failed
        iterations remain diagnosable. Part of #1373.

        Contracts:
            REQUIRES: self.log_dir is a valid Path
            REQUIRES: crash_log uses UTF-8 encoding when present
            ENSURES: On success, keeps all logs from last LOG_RETENTION_HOURS
            ENSURES: On success, keeps at most MAX_LOG_FILES older .jsonl files
            ENSURES: On success, removes oldest files first (by mtime)
            ENSURES: On success, truncates crash_log to MAX_CRASH_LOG_LINES
            ENSURES: Prints message when files rotated
            ENSURES: Never raises - catches OSError on file operations
        """
        try:
            log_files = sorted(
                self.log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime
            )
            # Partition into recent (protected) and older (eligible for rotation)
            retention_cutoff = time.time() - (LOG_RETENTION_HOURS * 3600)
            recent_files: list[Path] = []
            older_files: list[Path] = []
            for f in log_files:
                try:
                    if f.stat().st_mtime > retention_cutoff:
                        recent_files.append(f)
                    else:
                        older_files.append(f)
                except OSError as e:
                    debug_swallow("rotate_logs_stat", e)
                    older_files.append(f)  # Can't stat, treat as old

            # Only rotate older files, keeping MAX_LOG_FILES of them
            if len(older_files) > MAX_LOG_FILES:
                for f in older_files[:-MAX_LOG_FILES]:
                    f.unlink()
                log_info(
                    f"Rotated logs: removed {len(older_files) - MAX_LOG_FILES} old files"
                    f" (kept {len(recent_files)} recent)"
                )

            # Rotate crash log
            if self.crash_log.exists():
                lines = self.crash_log.read_text().splitlines()
                if len(lines) > MAX_CRASH_LOG_LINES:
                    self.crash_log.write_text(
                        "\n".join(lines[-MAX_CRASH_LOG_LINES:]) + "\n"
                    )
        except OSError as e:
            debug_swallow("rotate_logs", e)

    def scrub_log_file(self, log_file: Path) -> bool:
        """Scrub secrets from log file if scrub_logs config is enabled.

        Contracts:
            REQUIRES: log_file is a valid Path to an existing file
            ENSURES: Returns False if scrub_logs config is False (default)
            ENSURES: Returns False if scrubber script missing
            ENSURES: Returns True on successful scrub
            ENSURES: Prints status message on completion or error
            ENSURES: Never raises - catches TimeoutExpired and all exceptions
        """
        if not self.config.get("scrub_logs", False):
            return False

        scrubber = self._anchor_path(Path("ai_template_scripts/log_scrubber.py"))
        if not scrubber.exists():
            log_warning("⚠️  Log scrubber not found, skipping scrub")
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
                log_info(f"=== Log scrubbed: {log_file.name} ===")
                return True
            log_warning(f"⚠️  Log scrub error: {result.stderr[:100]}")
            return False
        except subprocess.TimeoutExpired:
            log_warning("⚠️  Log scrub timeout")
            return False
        except Exception as e:
            # Catch-all: scrubber execution failed, log and skip scrub
            log_warning(f"⚠️  Log scrub failed: {e}")
            return False

    def check_headless_violation(self, log_file: Path, role: str) -> bool:
        """Check log file for headless role violations.

        Headless roles (WORKER, PROVER, RESEARCHER, MANAGER) must not ask
        users for direction. This checks the log and warns if violations found.

        Contracts:
            REQUIRES: log_file is a valid Path to an existing file
            REQUIRES: role is a string like "worker", "prover", etc.
            ENSURES: Returns False for USER role (no check needed)
            ENSURES: Returns True if violations found, False otherwise
            ENSURES: Logs warning on violation
            ENSURES: Never raises - catches all exceptions
        """
        # Only check headless roles
        if role.lower() == "user":
            return False

        checker = self._anchor_path(
            Path("ai_template_scripts/hooks/check_headless_violation.py")
        )
        if not checker.exists():
            return False

        try:
            result = subprocess.run(
                ["python3", str(checker), str(log_file)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 1:
                # Violation detected
                log_warning(
                    f"⚠️  HEADLESS VIOLATION: AI asked for user direction "
                    f"in {log_file.name}"
                )
                if result.stdout:
                    for line in result.stdout.strip().split("\n")[:3]:
                        log_warning(f"   {line}")
                # Emit flag file for quick attention
                self._emit_headless_violation_flag(log_file, result.stdout)
                return True
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            debug_swallow("check_headless_violation")
            return False

    def _emit_headless_violation_flag(self, log_file: Path, details: str) -> None:
        """Emit a flag file for headless violation detection.

        Creates a file in .flags/ directory with violation details for
        quick attention by manager or monitoring tools.

        Contracts:
            REQUIRES: log_file is a valid Path
            ENSURES: Creates .flags/headless_violation_<timestamp> if possible
            ENSURES: Never raises - catches all exceptions
        """
        try:
            flags_dir = self._anchor_path(Path(".flags"))
            flags_dir.mkdir(exist_ok=True)
            # Use single timestamp for filename and content consistency
            now = datetime.now(UTC)
            timestamp = now.strftime("%Y%m%dT%H%M%S")
            # Include role in flag name for role-specific consumption (#2404)
            flag_file = flags_dir / f"headless_violation_{self.mode}_{timestamp}"
            # Write violation details to flag file
            content = f"HEADLESS VIOLATION DETECTED\n"
            content += f"Log file: {log_file.name}\n"
            content += f"Time: {now.isoformat()}\n"
            if details:
                content += f"Details:\n{details[:500]}\n"
            flag_file.write_text(content)
        except Exception:
            debug_swallow("emit_headless_violation_flag")  # Best effort

    def write_status(
        self,
        iteration: int,
        status: str,
        log_file: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write worker status to a JSON file for manager visibility.

        Contracts:
            REQUIRES: iteration is a non-negative integer
            REQUIRES: status is a non-empty string
            REQUIRES: extra is JSON-serializable when provided
            ENSURES: On success, JSON file includes pid, mode, project, iteration, status
            ENSURES: On success, includes ait_version if available
            ENSURES: On success, atomic write via temp file + rename
            ENSURES: On success, extra dict merged into status data
            ENSURES: Never raises - catches OSError on file write
        """
        now = datetime.now(UTC).isoformat()
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

        # Atomic write with unique temp path to avoid cross-thread/process collisions
        # when multiple writers overlap (#3164). Keep temp file in same directory so
        # os.replace remains atomic on POSIX filesystems.
        tmp = self.status_file.with_name(
            f"{self.status_file.name}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        )
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.status_file)
        except OSError as e:
            debug_swallow("write_status", e)
            # Clean up orphaned temp file if write/replace failed.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass  # Best effort cleanup

    def clear_status(self) -> None:
        """Remove status file on clean exit.

        Contracts:
            ENSURES: status_file is deleted if it exists
            ENSURES: No-op if status_file does not exist
            ENSURES: Never raises - catches OSError on unlink
        """
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except OSError as e:
            debug_swallow("clear_status", e)

    def log_exit(
        self,
        iteration: int,
        exit_code: int,
        ai_tool: str,
        session_committed: bool = False,
        graceful_stop: bool = False,
    ) -> None:
        """Log iteration exit details.

        Handles all non-zero exits: errors, timeouts, signals, early aborts.

        Contracts:
            REQUIRES: iteration is a positive integer
            REQUIRES: exit_code is an integer
            REQUIRES: ai_tool is a non-empty string
            ENSURES: Appends to error log only for real errors (not early aborts)
            ENSURES: Skips logging if session_committed and not a real error
            ENSURES: SIGTERM (15) with graceful_stop=True shows calm message, not error
            ENSURES: Prints error info to stdout on real errors
            ENSURES: Exit codes > 128 indicate signal (signal_num = code - 128)
            ENSURES: Exit code 124 = timeout, 125 = silence, 126 = early abort
            ENSURES: Never raises - catches OSError on file write
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check for SIGTERM (15) - may be graceful stop from STOP file
        signal_num = None
        if exit_code < 0:
            signal_num = -exit_code
        elif exit_code > 128:
            signal_num = exit_code - 128

        # SIGTERM (15) with graceful_stop is not a crash (#1988)
        if signal_num == 15 and graceful_stop:
            log_info("")
            log_info(f"*** {self.mode.upper()} stopped by STOP file (signal 15) ***")
            log_info("")
            return  # Not a crash, don't log to failures.log

        if exit_code < 0:
            error_msg = f"{ai_tool} killed by signal {signal_num}"
            is_error = True
        elif exit_code > 128:
            error_msg = f"{ai_tool} killed by signal {signal_num}"
            is_error = True
        elif exit_code == 124:
            error_msg = f"{ai_tool} timed out"
            is_error = not session_committed  # Timeout after commit = not an error
        elif exit_code == 125:
            error_msg = f"{ai_tool} killed due to silence (stale connection)"
            is_error = False  # Expected after sleep/resume, will restart cleanly
        elif exit_code == 127:
            error_msg = "repo not initialized (no VISION.md + no issues)"
            is_error = False  # Init failure, not an error - user needs to init
        else:
            error_msg = f"{ai_tool} exited with code {exit_code}"
            # Exit code 1 after successful commit = likely EPIPE or graceful exit
            is_error = not session_committed

        if not is_error:
            # Not a real error - don't log to errors.log (#1909)
            if session_committed:
                log_info("")
                log_info(
                    f"Note: {ai_tool} exited with code {exit_code} but session "
                    "committed successfully"
                )
                log_info("")
            return  # Don't log to errors.log

        # Get memory state for enhanced error logging (#1470)
        memory_info = ""
        mem_state = get_memory_state()
        if mem_state:
            memory_info = (
                f" | Mem: {mem_state.used_gb}/{mem_state.total_gb}GB "
                f"({mem_state.used_percent}%, {mem_state.pressure_level})"
            )

        try:
            with open(self.crash_log, "a") as f:
                oom_indicator = " [OOM]" if is_oom_signal(exit_code) else ""
                f.write(
                    f"[{timestamp}] Iteration {iteration}: "
                    f"{error_msg}{oom_indicator}{memory_info}\n"
                )
        except OSError as e:
            debug_swallow("log_exit", e)

        log_info("")
        log_error(f"*** {self.mode.upper()} ERROR: {error_msg}")
        if is_oom_signal(exit_code):
            log_error("    WARNING: SIGKILL detected - likely OOM kill")
        if mem_state:
            log_error(
                f"    Memory: {mem_state.used_gb}/{mem_state.total_gb}GB "
                f"({mem_state.used_percent}% used, {mem_state.pressure_level})"
            )
        log_error(f"    Log: {self.crash_log}")
        log_error("    See: docs/troubleshooting.md#session-issues")
        log_info("")

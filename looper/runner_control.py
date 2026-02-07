# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_control.py - Signal handling and graceful shutdown.

Provides RunnerControlMixin for LoopRunner:
- Signal handlers for SIGINT/SIGTERM
- STOP file detection for graceful shutdown
- Instance-specific STOP_W1, role STOP_WORKER, and global STOP file support
- STOP file expiry (stale files are ignored with warning)
- Stop reason extraction and session audit logging
- Subprocess cleanup on termination

Constants:
    SHUTDOWN_TIMEOUT_SEC: Grace period for subprocess termination (default 5s)
    DEFAULT_STOP_EXPIRY_SEC: Default STOP file expiry time (default 3600s = 1h)

See docs/looper.md "Runner Architecture" for integration details.
See designs/2026-02-04-session-coordination.md for the full design.
"""

from __future__ import annotations

__all__ = ["RunnerControlMixin", "SHUTDOWN_TIMEOUT_SEC", "DEFAULT_STOP_EXPIRY_SEC"]

import os
import subprocess
import time
import types
from datetime import datetime, timezone
from pathlib import Path

from looper.log import debug_swallow, log_info, log_warning

SHUTDOWN_TIMEOUT_SEC = 5
DEFAULT_STOP_EXPIRY_SEC = 3600  # 1 hour

# Heartbeat-based sleep detection for STOP file expiry
# Heartbeat files are per-session: .looper_heartbeat_{mode} or .looper_heartbeat_{mode}_{worker_id}
HEARTBEAT_FILE_PREFIX = ".looper_heartbeat"
SLEEP_THRESHOLD_SEC = 300  # 5 min gap between heartbeats = was asleep
WAKE_GRACE_SEC = 600  # 10 min grace period after wake - respect all STOP files


class RunnerControlMixin:
    """Signal handling and shutdown control.

    Required attributes:
        running: bool
        iteration: int
        worker_id: int | None
        mode: str
        iteration_runner: IterationRunner
        status_manager: StatusManager
        pid_file: Path
        _stop_dir: Path
        _role_stop_file: str
        _instance_stop_file: str | None (set when worker_id is provided)
        _session_log: Path (.coord/session.log)

    Internal state:
        _warned_expired_stops: set[str] - Tracks which expired STOP files we've warned about
        _wake_time: float | None - Timestamp when wake from sleep was detected
    """

    def _check_stop_file(self) -> str | None:
        """Check for graceful shutdown request files.

        Check order (most specific first):
        1. Instance-specific: STOP_W1, STOP_W2, etc. (if worker_id set)
        2. Role-specific: STOP_WORKER, STOP_MANAGER, etc.
        3. Global: STOP

        Expired STOP files (older than AIT_STOP_EXPIRY_SEC, default 1h) are
        ignored with a warning (logged once per file to avoid spam).

        Returns:
            Name of the stop file if found and not expired, None otherwise.
        """
        # Check order: instance -> role -> global (most specific first)
        candidates = []
        if hasattr(self, "_instance_stop_file") and self._instance_stop_file:
            candidates.append(self._instance_stop_file)
        candidates.append(self._role_stop_file)
        candidates.append("STOP")

        # Track warned files to avoid spam during _wait_interruptible
        if not hasattr(self, "_warned_expired_stops"):
            self._warned_expired_stops: set[str] = set()

        for stop_file in candidates:
            stop_path = self._stop_dir / stop_file
            if stop_path.exists():
                # Check if expired
                if self._is_stop_file_expired(stop_path):
                    # Only warn once per expired file
                    if stop_file not in self._warned_expired_stops:
                        log_warning(
                            f"Ignoring expired {stop_file} "
                            f"(age > {self._get_stop_expiry_sec()}s)"
                        )
                        self._warned_expired_stops.add(stop_file)
                    continue
                return stop_file
        return None

    def _get_heartbeat_filename(self) -> str:
        """Get per-session heartbeat filename.

        Returns .looper_heartbeat_{mode} or .looper_heartbeat_{mode}_{worker_id}
        to ensure each session tracks its own heartbeat independently.
        """
        if hasattr(self, "worker_id") and self.worker_id:
            return f"{HEARTBEAT_FILE_PREFIX}_{self.mode}_{self.worker_id}"
        return f"{HEARTBEAT_FILE_PREFIX}_{self.mode}"

    def _update_heartbeat(self) -> None:
        """Update heartbeat file and detect wake from sleep.

        Called at the start of each iteration. If the gap since last heartbeat
        exceeds SLEEP_THRESHOLD_SEC (5 min), the computer was asleep and we
        record the wake time to extend STOP file validity.

        Uses per-session heartbeat files to avoid interference between sessions.
        Skips sleep detection on the first call since the heartbeat file may be
        stale from a previous session (#2559).

        Side effects:
            - Touches per-session heartbeat file
            - Sets self._wake_time if wake from sleep detected
        """
        heartbeat_path = self._stop_dir / self._get_heartbeat_filename()
        now = time.time()

        # Initialize wake time tracking if not present
        if not hasattr(self, "_wake_time"):
            self._wake_time: float | None = None

        # Track whether this is the first heartbeat call in this session.
        # On the first call, the heartbeat file is stale from a previous
        # session, so any gap is a session restart, not machine sleep (#2559).
        first_call = not hasattr(self, "_heartbeat_initialized")
        self._heartbeat_initialized = True

        try:
            if heartbeat_path.exists() and not first_call:
                last_beat = heartbeat_path.stat().st_mtime
                gap = now - last_beat

                if gap > SLEEP_THRESHOLD_SEC:
                    # Computer was asleep - record wake time
                    self._wake_time = now
                    log_info(
                        f"Wake from sleep detected (gap: {gap:.0f}s) - "
                        f"respecting STOP files for {WAKE_GRACE_SEC}s"
                    )

            # Update heartbeat
            heartbeat_path.touch()
        except OSError:
            # Best effort - continue if heartbeat fails
            pass

    def _in_wake_grace_period(self) -> bool:
        """Check if we're within the grace period after waking from sleep.

        Returns:
            True if wake was detected and we're within WAKE_GRACE_SEC of it.
        """
        if not hasattr(self, "_wake_time") or self._wake_time is None:
            return False
        return (time.time() - self._wake_time) < WAKE_GRACE_SEC

    def _touch_heartbeat(self) -> None:
        """Touch heartbeat file without wake detection.

        Used during long waits to keep heartbeat fresh and avoid false
        wake detection on next iteration.
        """
        try:
            heartbeat_path = self._stop_dir / self._get_heartbeat_filename()
            heartbeat_path.touch()
        except OSError:
            pass  # Best effort

    def _is_stop_file_expired(self, stop_path: Path) -> bool:
        """Check if a STOP file is expired (older than expiry threshold).

        If we recently woke from sleep (within WAKE_GRACE_SEC), all STOP files
        are considered valid regardless of age - sleep time shouldn't count
        toward expiry.

        Args:
            stop_path: Path to the STOP file.

        Returns:
            True if file is older than AIT_STOP_EXPIRY_SEC (default 1h),
            unless we're in the wake grace period.
        """
        # If we just woke from sleep, respect all STOP files
        if self._in_wake_grace_period():
            return False

        try:
            mtime = stop_path.stat().st_mtime
            now = time.time()
            age_sec = now - mtime
            return age_sec > self._get_stop_expiry_sec()
        except OSError:
            # If we can't stat the file, treat as not expired
            return False

    def _get_stop_expiry_sec(self) -> int:
        """Get STOP file expiry in seconds from environment or default."""
        try:
            return int(os.environ.get("AIT_STOP_EXPIRY_SEC", DEFAULT_STOP_EXPIRY_SEC))
        except ValueError:
            return DEFAULT_STOP_EXPIRY_SEC

    def _get_stop_file_reason(self, stop_path: Path) -> str:
        """Extract reason from STOP file content.

        Content format (all optional):
        - Empty file: no reason
        - Single line: treated as reason
        - Multi-line with "reason:" prefix: extracts that line

        Args:
            stop_path: Path to the STOP file.

        Returns:
            Reason string, or empty string if none found.
        """
        try:
            content = stop_path.read_text().strip()
            if not content:
                return ""
            # Check for "reason: ..." format
            for line in content.splitlines():
                if line.lower().startswith("reason:"):
                    return line.split(":", 1)[1].strip()
            # Fall back to first line as reason
            return content.splitlines()[0][:100]  # Truncate long reasons
        except OSError:
            return ""

    def _log_session_event(
        self, event: str, *, reason: str = "", clean: bool = True
    ) -> None:
        """Append event to session audit log (.coord/session.log).

        Format: ISO_TIMESTAMP ROLE_ID EVENT [key=value ...]

        Example:
            2026-02-04T14:30:00 W1 START pid=12345
            2026-02-04T15:00:00 W1 STOP  reason="switching tasks" clean=true

        Args:
            event: Event type (START, STOP).
            reason: Optional reason string.
            clean: Whether this is a clean shutdown (vs crash/signal).
        """
        if not hasattr(self, "_session_log"):
            return

        try:
            # Build role identifier (e.g., "W1" or "WORKER")
            role_letter = self.mode[0].upper()
            if hasattr(self, "worker_id") and self.worker_id is not None:
                role_id = f"{role_letter}{self.worker_id}"
            else:
                role_id = self.mode.upper()

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            pid = os.getpid()

            # Build event line
            parts = [timestamp, role_id, event]
            if event == "START":
                parts.append(f"pid={pid}")
            elif event == "STOP":
                # Quote reason if present
                if reason:
                    # Escape quotes in reason
                    safe_reason = reason.replace('"', '\\"')
                    parts.append(f'reason="{safe_reason}"')
                else:
                    parts.append('reason=""')
                parts.append(f"clean={str(clean).lower()}")

            line = " ".join(parts)

            # Ensure parent directory exists
            self._session_log.parent.mkdir(parents=True, exist_ok=True)

            # Append to log
            with self._session_log.open("a") as f:
                f.write(line + "\n")
        except OSError as e:
            debug_swallow("log_session_event", e)

    def _consume_stop_file(self, stop_file: str) -> None:
        """Consume a STOP file after stopping.

        Consumption rules (per designs/2026-02-04-session-coordination.md):
        - Instance-specific (STOP_W1): Consumed after stop (single target)
        - Role-specific (STOP_WORKER): NOT consumed (shared, expires)
        - Global (STOP): NOT consumed (shared, expires)

        Args:
            stop_file: Name of the stop file to potentially consume.
        """
        # Only consume instance-specific stop files
        is_instance_file = (
            hasattr(self, "_instance_stop_file")
            and self._instance_stop_file
            and stop_file == self._instance_stop_file
        )
        if is_instance_file:
            try:
                (self._stop_dir / stop_file).unlink()
                log_info(f"*** {stop_file} consumed ***")
            except OSError as e:
                debug_swallow("consume_stop_file", e)

    def handle_signal(self, signum: int, frame: types.FrameType | None) -> None:
        """Handle shutdown signals gracefully."""
        log_info("")
        log_info(f"Received signal {signum}, shutting down...")
        # Log STOP event with signal info (not a clean shutdown)
        signal_name = {2: "SIGINT", 15: "SIGTERM"}.get(signum, f"signal {signum}")
        self._log_session_event("STOP", reason=signal_name, clean=False)
        self.running = False
        if self.iteration_runner.current_process is not None:
            proc = self.iteration_runner.current_process
            terminated = self._terminate_subprocess(proc, reason="signal handler")
            if terminated:
                self.iteration_runner.current_process = None
            else:
                log_warning("Warning: subprocess still running after signal handler")

    def _terminate_subprocess(self, proc: subprocess.Popen[bytes], reason: str) -> bool:
        """Terminate a subprocess, escalating to SIGKILL."""
        try:
            if proc.poll() is not None:
                return True
        except Exception as exc:
            # Best-effort: poll can fail during shutdown; log and continue
            log_warning(
                f"Warning: could not check subprocess status during {reason}: {exc}"
            )

        try:
            proc.terminate()
        except ProcessLookupError:
            return True
        except Exception as exc:
            # Best-effort: terminate failure shouldn't block wait/kill path
            log_warning(
                f"Warning: failed to terminate subprocess during {reason}: {exc}"
            )

        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT_SEC)
            return True
        except subprocess.TimeoutExpired:
            log_info("Grace period expired, sending SIGKILL")
        except Exception as exc:
            # Best-effort: wait failure shouldn't block SIGKILL path
            log_warning(f"Warning: error waiting for subprocess during {reason}: {exc}")

        try:
            proc.kill()
        except ProcessLookupError:
            return True
        except Exception as exc:
            # Best-effort: SIGKILL failure shouldn't block status checks
            log_warning(f"Warning: failed to SIGKILL subprocess during {reason}: {exc}")

        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT_SEC)
            return True
        except subprocess.TimeoutExpired:
            log_warning(
                f"Warning: subprocess still running after SIGKILL during {reason}"
            )
            return False
        except Exception as exc:
            # Best-effort: post-SIGKILL wait failure falls back to poll
            log_warning(f"Warning: error waiting after SIGKILL during {reason}: {exc}")
            try:
                return proc.poll() is not None
            except Exception as e:
                debug_swallow("terminate_subprocess_poll", e)
                return False

    def cleanup(self) -> None:
        """Clean up on exit."""
        if self.iteration_runner.current_process is not None:
            proc = self.iteration_runner.current_process
            terminated = self._terminate_subprocess(proc, reason="cleanup")
            if terminated:
                self.iteration_runner.current_process = None
            else:
                log_warning("Warning: subprocess still running after cleanup")
        # Stop memory watchdog daemon (#1468)
        self._stop_memory_watchdog()
        self.status_manager.clear_status()
        if self.pid_file.exists():
            self.pid_file.unlink()
        log_info(f"Completed {self.iteration - 1} iterations")

    def _wait_interruptible(self, delay: int) -> None:
        """Wait with interruptible sleep that responds to STOP file."""
        if delay <= 0:
            return

        self.status_manager.write_status(
            self.iteration, "waiting", extra={"next_iteration_in": delay}
        )
        if delay > 60:
            log_info(f"Next iteration in {delay // 60} minutes...")
        else:
            log_info(f"Next iteration in {delay} seconds...")

        for i in range(delay):
            if not self.running:
                break
            stop_file = self._check_stop_file()
            if stop_file:
                log_info(f"\n*** {stop_file} detected - stopping ***")
                # Extract reason and log before consuming
                stop_path = self._stop_dir / stop_file
                reason = self._get_stop_file_reason(stop_path)
                if reason:
                    log_info(f"    Reason: {reason}")
                self._log_session_event("STOP", reason=reason, clean=True)
                self._consume_stop_file(stop_file)
                self.running = False
                break
            # Touch heartbeat every 60s during wait to avoid false wake detection
            if i > 0 and i % 60 == 0:
                self._touch_heartbeat()
            time.sleep(1)

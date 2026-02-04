# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_control.py - Signal handling and graceful shutdown.

Provides RunnerControlMixin for LoopRunner:
- Signal handlers for SIGINT/SIGTERM
- STOP file detection for graceful shutdown
- Role-specific STOP_<ROLE> file support
- Subprocess cleanup on termination

Constants:
    SHUTDOWN_TIMEOUT_SEC: Grace period for subprocess termination (default 5s)

See docs/looper.md "Runner Architecture" for integration details.
"""

from __future__ import annotations

__all__ = ["RunnerControlMixin", "SHUTDOWN_TIMEOUT_SEC"]

import subprocess
import time
import types

from looper.log import debug_swallow, log_info, log_warning

SHUTDOWN_TIMEOUT_SEC = 5


class RunnerControlMixin:
    """Signal handling and shutdown control.

    Required attributes:
        running: bool
        iteration: int
        iteration_runner: IterationRunner
        status_manager: StatusManager
        pid_file: Path
        _stop_dir: Path
        _role_stop_file: str
    """

    def _check_stop_file(self) -> str | None:
        """Check for graceful shutdown request files.

        Returns:
            Name of the stop file if found, None otherwise.
            Checks STOP (all roles) first, then STOP_<ROLE> (per-role).
        """
        if (self._stop_dir / "STOP").exists():
            return "STOP"
        if (self._stop_dir / self._role_stop_file).exists():
            return self._role_stop_file
        return None

    def _consume_stop_file(self, stop_file: str) -> None:
        """Consume a per-role STOP file after stopping.

        STOP (global) is NOT consumed - user removes it manually so all roles see it.
        STOP_<ROLE> IS consumed - it's targeted at this role only.
        """
        if stop_file == self._role_stop_file:
            try:
                (self._stop_dir / stop_file).unlink()
                log_info(f"*** {stop_file} consumed ***")
            except OSError as e:
                debug_swallow("consume_stop_file", e)

    def handle_signal(self, signum: int, frame: types.FrameType | None) -> None:
        """Handle shutdown signals gracefully."""
        log_info("")
        log_info(f"Received signal {signum}, shutting down...")
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

        for _ in range(delay):
            if not self.running:
                break
            stop_file = self._check_stop_file()
            if stop_file:
                log_info(f"\n*** {stop_file} detected - stopping ***")
                self._consume_stop_file(stop_file)
                self.running = False
                break
            time.sleep(1)

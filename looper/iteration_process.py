# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""AI subprocess execution and lifecycle management for looper.

Extracted from iteration.py per designs/2026-02-01-iteration-split.md Phase 3.

Responsibilities:
- Run AI subprocesses (claude, codex, dasher) with proper I/O handling
- Manage timeouts (iteration and silence)
- Handle graceful/forceful termination
- Stream output to log files and display

Part of #1804.
"""

import json
import select
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Protocol

from looper.log import debug_swallow, log_error, log_info, log_warning

__all__ = ["ProcessManager", "ToolEventProcessor"]


class ToolEventProcessor(Protocol):
    """Protocol for processing tool call events from AI output."""

    def process_event(self, msg: dict[str, Any]) -> None:
        """Process a tool call event message.

        REQUIRES: msg is a parsed JSON message dict
        ENSURES: tool event is recorded/handled by the implementation
        """
        ...


class ProcessManager:
    """Manage AI subprocess execution and lifecycle.

    Handles:
    - Process spawning with proper I/O piping
    - Timeout management (iteration and silence)
    - Sleep detection to avoid false silence timeouts
    - Graceful/forceful termination
    - Output streaming to log files and display
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize process manager with configuration.

        REQUIRES: config includes iteration_timeout (seconds)
        ENSURES: self.config is set for subsequent runs
        ENSURES: self.current_process is None

        Args:
            config: Configuration dict with iteration_timeout, silence_timeout keys
        """
        self.config = config
        self.current_process: subprocess.Popen[bytes] | None = None

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration for next iteration.

        REQUIRES: config includes iteration_timeout (seconds)
        ENSURES: self.config is updated for subsequent runs
        """
        self.config = config

    def run_ai_process(
        self,
        cmd: list[str],
        log_file: Path,
        ai_tool: str,
        start_time: float,
        tool_event_processor: ToolEventProcessor | None = None,
    ) -> int:
        """Run the AI subprocess and stream output, returning exit code.

        REQUIRES: cmd is a non-empty list of strings
        REQUIRES: ai_tool in ("claude", "codex", "dasher")
        REQUIRES: start_time is a Unix timestamp (seconds)
        ENSURES: returns exit code int (0=success, 1=exception, 124=timeout,
                 125=silence, or subprocess exit code)
        ENSURES: self.current_process is None on return

        Args:
            cmd: Command to execute
            log_file: Path to write JSON log output
            ai_tool: Tool name for logging ("claude", "codex", "dasher")
            start_time: Unix timestamp when iteration started
            tool_event_processor: Optional processor for tool call events

        Returns:
            Exit code: 0=success, 124=timeout, 125=silence, other=error
        """
        timeout_sec = self.config["iteration_timeout"]
        silence_timeout_sec = self.config.get("silence_timeout", 600)
        progress_interval_sec = 60
        last_output_time = start_time
        last_output_mono = time.monotonic()  # For sleep detection (#1357)
        last_progress_time = start_time
        last_command: str | None = None
        last_command_start: float | None = None
        exit_code = 0
        timed_out = False
        silence_killed = False
        text_proc_alive = True
        ai_proc: subprocess.Popen[bytes] | None = None
        text_proc: subprocess.Popen[bytes] | None = None

        def write_to_text_proc(data: bytes) -> None:
            nonlocal text_proc_alive
            if not text_proc_alive or text_proc is None or text_proc.stdin is None:
                return
            try:
                text_proc.stdin.write(data)
                text_proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                debug_swallow("text_proc_write", e)
                text_proc_alive = False

        try:
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, "w") as log_f:
                ai_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self.current_process = ai_proc
                assert ai_proc.stdout is not None

                try:
                    text_proc = subprocess.Popen(
                        ["./ai_template_scripts/json_to_text.py"],
                        stdin=subprocess.PIPE,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                    )
                except OSError as e:
                    # If text_proc fails to start, still need to cleanup ai_proc
                    log_warning(f"Warning: json_to_text.py failed to start: {e}")
                    text_proc = None

                try:
                    while ai_proc.poll() is None:
                        now = time.time()

                        # Check iteration timeout
                        if self._check_iteration_timeout(
                            ai_proc, start_time, timeout_sec
                        ):
                            timed_out = True
                            break

                        # Print progress for long-running commands
                        if (
                            last_command
                            and now - last_progress_time > progress_interval_sec
                        ):
                            self._print_progress(
                                last_command, last_command_start, start_time
                            )
                            last_progress_time = now

                        # Check silence timeout (with sleep detection #1357)
                        silence_timed_out = self._check_silence_timeout(
                            ai_proc,
                            last_output_time,
                            last_output_mono,
                            last_command,
                            last_command_start,
                            silence_timeout_sec,
                            start_time,
                        )
                        if silence_timed_out:
                            # Actually timed out (not sleep)
                            silence_killed = True
                            break
                        # Check if sleep was detected (wall >> mono by >60s)
                        # Reset timers so we don't keep detecting the same sleep
                        wall_elapsed = time.time() - last_output_time
                        mono_elapsed = time.monotonic() - last_output_mono
                        if wall_elapsed > mono_elapsed + 60:
                            # Sleep was detected - reset timers to now
                            last_output_time = time.time()
                            last_output_mono = time.monotonic()

                        # Read and process output
                        ready, _, _ = select.select([ai_proc.stdout], [], [], 1.0)
                        if ready:
                            line = ai_proc.stdout.readline()
                            if line:
                                last_output_time = time.time()
                                last_output_mono = time.monotonic()
                                last_progress_time = time.time()
                                log_f.write(line.decode())
                                log_f.flush()
                                write_to_text_proc(line)

                                try:
                                    msg = json.loads(line.decode())
                                    cmd_str, is_completed = (
                                        self._extract_command_from_message(msg)
                                    )
                                    if is_completed:
                                        last_command = None
                                        last_command_start = None
                                    elif cmd_str:
                                        last_command = cmd_str
                                        last_command_start = time.time()
                                    # Record tool call events for checkpointing (#1625)
                                    if tool_event_processor is not None:
                                        tool_event_processor.process_event(msg)
                                except (
                                    json.JSONDecodeError,
                                    KeyError,
                                    TypeError,
                                ) as e:
                                    debug_swallow("process_tool_call_event", e)

                    # Drain remaining output
                    self._drain_remaining_output(ai_proc, log_f, write_to_text_proc)

                finally:
                    # Close stdin if text_proc is still alive, then wait
                    if text_proc_alive and text_proc is not None:
                        if text_proc.stdin is not None:
                            try:
                                text_proc.stdin.close()
                            except (BrokenPipeError, OSError) as e:
                                debug_swallow("text_proc_stdin_close", e)
                    self._cleanup_text_proc(text_proc)
                    if ai_proc and ai_proc.stdout:
                        try:
                            ai_proc.stdout.close()
                        except OSError as e:
                            debug_swallow("ai_proc_stdout_close", e)

                if timed_out:
                    exit_code = 124
                elif silence_killed:
                    exit_code = 125
                else:
                    exit_code = ai_proc.returncode or 0
                self.current_process = None

        except Exception as e:
            log_error(f"Error running {ai_tool}: {e}")
            exit_code = 1
            self._cleanup_text_proc(
                text_proc, force_close_stdin=True, terminate_first=True
            )
            self._cleanup_ai_proc(ai_proc)
            self.current_process = None

        return exit_code

    def terminate_process(self, proc: subprocess.Popen[bytes]) -> None:
        """Gracefully terminate a process, falling back to SIGKILL if needed.

        REQUIRES: proc was started and is a valid Popen instance
        ENSURES: proc is terminated (or killed) and reaped

        Args:
            proc: Process to terminate
        """
        proc.terminate()
        try:
            proc.wait(timeout=10)
            log_info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            log_info("Grace period expired, sending SIGKILL")
            proc.kill()
            proc.wait()  # Reap zombie after SIGKILL

    def _check_iteration_timeout(
        self,
        proc: subprocess.Popen[bytes],
        start_time: float,
        timeout_sec: int,
    ) -> bool:
        """Check if iteration timeout exceeded, terminate if so.

        REQUIRES: timeout_sec > 0
        REQUIRES: start_time is a Unix timestamp (seconds)
        ENSURES: returns True iff timeout exceeded and termination attempted

        Returns:
            True if timeout exceeded and process terminated
        """
        if time.time() - start_time > timeout_sec:
            log_info(f"\nTimeout after {timeout_sec // 60} minutes")
            self.terminate_process(proc)
            return True
        return False

    def _check_silence_timeout(
        self,
        proc: subprocess.Popen[bytes],
        last_output_time: float,
        last_output_mono: float,
        last_command: str | None,
        last_command_start: float | None,
        silence_timeout_sec: int,
        start_time: float,
    ) -> bool:
        """Check if silence timeout exceeded, terminate if so.

        Sleep detection (#1357):
        - Compare wall time vs monotonic time elapsed since last output
        - If wall time >> monotonic time, machine was asleep
        - Don't count sleep time against silence timeout

        REQUIRES: silence_timeout_sec > 0
        REQUIRES: last_output_time/last_output_mono are timestamps of last output
        ENSURES: returns True iff silence timeout exceeded and termination attempted
        ENSURES: returns False if no termination (including sleep detected)

        Returns:
            True if silence timeout exceeded and process terminated
        """
        now_wall = time.time()
        now_mono = time.monotonic()

        # Detect laptop sleep: wall time advances during sleep, monotonic doesn't
        # If discrepancy > 60s, machine was asleep - reset timeout (#1357)
        wall_elapsed = now_wall - last_output_time
        mono_elapsed = now_mono - last_output_mono
        sleep_detected = wall_elapsed > mono_elapsed + 60

        if sleep_detected:
            sleep_duration = int(wall_elapsed - mono_elapsed)
            log_info(
                f"\n[sleep detected: {sleep_duration}s gap, not counting as silence]",
            )
            # Return False to indicate we should NOT kill the process
            # The caller will reset the timers
            return False

        # Long-running commands (builds, tests) get extended grace period
        long_running_patterns = (
            "cargo test",
            "cargo build",
            "cargo check",
            "cargo clippy",
            "pytest",
            "npm test",
            "npm run",
            "make",
            "go test",
        )
        is_long_command = last_command and any(
            p in last_command for p in long_running_patterns
        )
        max_silence_sec = 3600  # 1 hour max even for long commands
        timeout_config = self.config.get("timeouts")
        if isinstance(timeout_config, dict):
            raw_max = timeout_config.get("max_silence")
            if isinstance(raw_max, bool):
                raw_max = None
            if isinstance(raw_max, (int, float)) and raw_max > 0:
                max_silence_sec = int(raw_max)

        # Use monotonic elapsed for actual silence check (immune to sleep)
        silence_exceeded = mono_elapsed > silence_timeout_sec
        max_exceeded = mono_elapsed > max_silence_sec

        # Only timeout if: (1) silence exceeded AND (2) not a long command OR max exceeded
        if silence_exceeded and (not is_long_command or max_exceeded):
            elapsed_min = int(mono_elapsed // 60)
            log_info(f"\nSilence timeout: no output for {elapsed_min} minutes")
            if last_command:
                cmd_elapsed = int(now_wall - (last_command_start or start_time))
                log_info(
                    f"Last activity: {last_command[:80]} "
                    f"(started {cmd_elapsed // 60}m ago)"
                )
            log_info("Connection may be stale (e.g., after sleep/resume)")
            self.terminate_process(proc)
            return True
        return False

    def _print_progress(
        self,
        last_command: str | None,
        last_command_start: float | None,
        start_time: float,
    ) -> None:
        """Print progress message for long-running commands.

        REQUIRES: last_command is set when logging is needed
        ENSURES: logs a progress message when last_command is provided
        """
        if not last_command:
            return
        now = time.time()
        elapsed = int(now - (last_command_start or start_time))
        elapsed_min = elapsed // 60
        elapsed_sec = elapsed % 60
        if elapsed_min > 0:
            log_info(
                f"  ... Running: {last_command[:60]} ({elapsed_min}m {elapsed_sec}s)"
            )
        else:
            log_info(f"  ... Running: {last_command[:60]} ({elapsed_sec}s)")

    @staticmethod
    def _extract_command_from_message(msg: dict[str, Any]) -> tuple[str | None, bool]:
        """Extract command info from a JSON message.

        REQUIRES: msg is a parsed JSON message dict
        ENSURES: command string is truncated to 100 chars when present
        ENSURES: is_completed True only for completion messages

        Returns:
            (command_str, is_completed) tuple.
            - command_str: The command being executed, or None if not a command message
            - is_completed: True if this message indicates command completion
        """
        msg_type = msg.get("type")

        if msg_type == "assistant":
            content = msg.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    cmd_line = block.get("input", {}).get("command", "")
                    if cmd_line:
                        return cmd_line[:100], False
        elif msg_type == "item.started":
            item = msg.get("item", {})
            if item.get("type") == "command_execution":
                cmd_line = item.get("command", "")
                if cmd_line:
                    return cmd_line[:100], False
        elif msg_type == "item.completed":
            item = msg.get("item", {})
            if item.get("type") == "command_execution":
                return None, True
        elif msg.get("role") == "user":
            content = msg.get("content", [])
            for block in content:
                if block.get("type") == "tool_result":
                    return None, True

        return None, False

    def _drain_remaining_output(
        self,
        ai_proc: subprocess.Popen[bytes],
        log_f: IO[str],
        write_func: Callable[[bytes], None],
    ) -> None:
        """Drain any remaining output from the AI process after it ends.

        REQUIRES: ai_proc.stdout is not None and log_f is open
        ENSURES: drains remaining output until timeout or EOF
        """
        drain_timeout = 30
        drain_start = time.time()
        assert ai_proc.stdout is not None
        while time.time() - drain_start < drain_timeout:
            ready, _, _ = select.select([ai_proc.stdout], [], [], 1.0)
            if ready:
                line = ai_proc.stdout.readline()
                if not line:
                    break
                log_f.write(line.decode())
                log_f.flush()
                write_func(line)
            elif ai_proc.poll() is not None:
                break

    def _cleanup_text_proc(
        self,
        text_proc: subprocess.Popen[bytes] | None,
        force_close_stdin: bool = False,
        terminate_first: bool = False,
    ) -> None:
        """Clean up the text processor subprocess.

        REQUIRES: text_proc is a subprocess or None
        ENSURES: attempts to terminate and reap the subprocess
        ENSURES: stdin is closed when force_close_stdin is True

        Args:
            text_proc: The text processor to clean up, or None
            force_close_stdin: If True, always try to close stdin (for exception paths)
            terminate_first: If True, call terminate() before wait() (for exception paths)
        """
        if text_proc is None:
            return

        if force_close_stdin and text_proc.stdin is not None:
            try:
                text_proc.stdin.close()
            except (BrokenPipeError, OSError) as e:
                debug_swallow("text_proc_stdin_close_force", e)

        if terminate_first:
            try:
                text_proc.terminate()
            except OSError as e:
                debug_swallow("text_proc_terminate", e)

        try:
            text_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                text_proc.kill()
                text_proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError) as e:
                debug_swallow("text_proc_kill", e)

    def _cleanup_ai_proc(self, ai_proc: subprocess.Popen[bytes] | None) -> None:
        """Clean up the AI subprocess on exception.

        REQUIRES: ai_proc is a subprocess or None
        ENSURES: attempts to terminate and reap the subprocess
        ENSURES: ai_proc.stdout is closed when available
        ENSURES: self.current_process is None
        """
        if ai_proc is None:
            return

        try:
            if ai_proc.poll() is None:
                ai_proc.terminate()
                ai_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                ai_proc.kill()
                ai_proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError) as e:
                debug_swallow("ai_proc_kill", e)
        except OSError as e:
            debug_swallow("ai_proc_cleanup", e)
        finally:
            if ai_proc.stdout is not None:
                try:
                    ai_proc.stdout.close()
                except OSError as e:
                    debug_swallow("ai_proc_stdout_close_cleanup", e)
            self.current_process = None

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
bg_task.py - Background task management for long-running operations

PURPOSE: Run tasks that outlive worker iteration timeouts (builds, tests, etc).
CALLED BY: WORKER AI (via CLI), human (debugging)
REFERENCED: Not yet integrated into main workflow

Manages background tasks:
- Start tasks with metadata (issue, timeout, description)
- Monitor running tasks
- Tail output logs
- Wait for completion
- Clean up old tasks

Directory structure:
    .background_tasks/
    ├── manifest.json              # Summary of all tasks
    └── <task_id>/                 # Task directory
        ├── meta.json              # Who, what, when, status
        ├── output.log             # Streaming output
        ├── pid                    # Process ID file
        └── result.json            # Final result (exit code, duration)

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Directory for background tasks (relative to git root)
TASKS_DIR = ".background_tasks"

# Default timeout: 2 hours
DEFAULT_TIMEOUT = 7200


@dataclass
class TaskMeta:
    """Metadata for a background task."""

    task_id: str
    command: str
    description: str
    issue: int | None
    timeout: int
    status: str  # "running", "completed", "failed", "timeout", "killed"
    started_at: str
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    machine: str = field(default_factory=lambda: os.uname().nodename)
    worker_iteration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMeta:
        return cls(**data)


def get_git_root() -> Path:
    """Get the git repository root directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_tasks_dir() -> Path:
    """Get the background tasks directory."""
    tasks_dir = get_git_root() / TASKS_DIR
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir


def get_task_dir(task_id: str) -> Path:
    """Get directory for a specific task."""
    return get_tasks_dir() / task_id


def load_manifest() -> dict[str, Any]:
    """Load the manifest file."""
    manifest_path = get_tasks_dir() / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"tasks": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Save the manifest file."""
    manifest_path = get_tasks_dir() / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_task_meta(task_id: str) -> TaskMeta | None:
    """Load metadata for a specific task."""
    meta_path = get_task_dir(task_id) / "meta.json"
    if meta_path.exists():
        return TaskMeta.from_dict(json.loads(meta_path.read_text()))
    return None


def save_task_meta(meta: TaskMeta) -> None:
    """Save metadata for a task."""
    task_dir = get_task_dir(meta.task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    meta_path = task_dir / "meta.json"
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")


def is_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def update_task_status(task_id: str) -> TaskMeta | None:
    """Update task status based on process state."""
    meta = load_task_meta(task_id)
    if meta is None:
        return None

    if meta.status == "running" and meta.pid:
        if not is_process_alive(meta.pid):
            # Process finished - check for result
            result_path = get_task_dir(task_id) / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                meta.exit_code = result.get("exit_code")
                meta.finished_at = result.get("finished_at")
                meta.status = "completed" if meta.exit_code == 0 else "failed"
            else:
                # Process died without writing result
                meta.status = "failed"
                meta.finished_at = datetime.now(timezone.utc).isoformat()
            save_task_meta(meta)

            # Update manifest
            manifest = load_manifest()
            if task_id in manifest["tasks"]:
                manifest["tasks"][task_id]["status"] = meta.status
                save_manifest(manifest)

    return meta


def start_task(
    command: list[str],
    task_id: str,
    issue: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    description: str = "",
) -> TaskMeta:
    """Start a background task."""
    task_dir = get_task_dir(task_id)

    # Check if task already exists and is running
    existing = load_task_meta(task_id)
    if existing and existing.status == "running":
        if existing.pid and is_process_alive(existing.pid):
            raise ValueError(f"Task {task_id} is already running (PID {existing.pid})")

    # Create task directory
    task_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output log
    output_path = task_dir / "output.log"
    result_path = task_dir / "result.json"
    pid_path = task_dir / "pid"

    # Get iteration from environment (AI_ITERATION set by looper.py)
    worker_iteration = None
    if os.environ.get("AI_ITERATION"):
        worker_iteration = int(os.environ["AI_ITERATION"])

    # Create metadata
    meta = TaskMeta(
        task_id=task_id,
        command=" ".join(command),
        description=description or f"Background task: {task_id}",
        issue=issue,
        timeout=timeout,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        worker_iteration=worker_iteration,
    )

    # Build wrapper script that handles timeout and result writing
    wrapper_script = f"""
import subprocess
import sys
import json
import signal
from datetime import datetime, timezone

def handler(signum, frame):
    sys.exit(124)  # Timeout exit code

signal.signal(signal.SIGALRM, handler)
signal.alarm({timeout})

try:
    with open({str(output_path)!r}, 'w') as output_file:
        result = subprocess.run(
            {command!r},
            stdout=output_file,
            stderr=subprocess.STDOUT,
        )
    exit_code = result.returncode
except Exception as e:
    with open({str(output_path)!r}, 'a') as f:
        f.write(f"\\nError: {{e}}\\n")
    exit_code = 1

# Write result
with open({str(result_path)!r}, 'w') as f:
    json.dump({{
        "exit_code": exit_code,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }}, f, indent=2)

sys.exit(exit_code)
"""

    # Start the background process
    process = subprocess.Popen(
        [sys.executable, "-c", wrapper_script],
        start_new_session=True,  # Detach from parent
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    meta.pid = process.pid
    pid_path.write_text(str(process.pid))

    # Save metadata
    save_task_meta(meta)

    # Update manifest
    manifest = load_manifest()
    manifest["tasks"][task_id] = {
        "status": "running",
        "started_at": meta.started_at,
        "issue": issue,
        "pid": meta.pid,
    }
    save_manifest(manifest)

    return meta


def list_tasks(show_all: bool = False) -> list[TaskMeta]:
    """List all tasks, updating their status."""
    tasks_dir = get_tasks_dir()
    tasks = []

    for task_dir in tasks_dir.iterdir():
        if task_dir.is_dir() and (task_dir / "meta.json").exists():
            meta = update_task_status(task_dir.name)
            if meta:
                if show_all or meta.status == "running":
                    tasks.append(meta)

    return sorted(tasks, key=lambda t: t.started_at, reverse=True)


def get_status(task_id: str) -> TaskMeta | None:
    """Get status of a specific task."""
    return update_task_status(task_id)


def tail_output(task_id: str, lines: int = 50) -> str:
    """Get the last N lines of task output."""
    output_path = get_task_dir(task_id) / "output.log"
    if not output_path.exists():
        return "(no output yet)"

    content = output_path.read_text()
    all_lines = content.splitlines()
    return "\n".join(all_lines[-lines:])


def wait_for_task(task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskMeta:
    """Wait for a task to complete."""
    start_time = time.time()

    while True:
        meta = update_task_status(task_id)
        if meta is None:
            raise ValueError(f"Task {task_id} not found")

        if meta.status != "running":
            return meta

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise TimeoutError(f"Timeout waiting for task {task_id} after {timeout}s")

        time.sleep(poll_interval)


def kill_task(task_id: str) -> TaskMeta | None:
    """Kill a running task."""
    meta = load_task_meta(task_id)
    if meta is None:
        return None

    if meta.status != "running" or meta.pid is None:
        return meta

    if is_process_alive(meta.pid):
        try:
            os.kill(meta.pid, signal.SIGTERM)
            time.sleep(1)
            if is_process_alive(meta.pid):
                os.kill(meta.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    meta.status = "killed"
    meta.finished_at = datetime.now(timezone.utc).isoformat()
    save_task_meta(meta)

    # Update manifest
    manifest = load_manifest()
    if task_id in manifest["tasks"]:
        manifest["tasks"][task_id]["status"] = "killed"
        save_manifest(manifest)

    return meta


def cleanup_tasks(days: int = 7, force: bool = False) -> list[str]:
    """Remove old completed tasks."""
    tasks_dir = get_tasks_dir()
    removed = []
    cutoff = time.time() - (days * 24 * 60 * 60)

    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue

        meta_path = task_dir / "meta.json"
        if not meta_path.exists():
            continue

        meta = load_task_meta(task_dir.name)
        if meta is None:
            continue

        # Don't remove running tasks
        if meta.status == "running" and not force:
            continue

        # Check age
        try:
            started = datetime.fromisoformat(meta.started_at.replace("Z", "+00:00"))
            if started.timestamp() < cutoff or force:
                shutil.rmtree(task_dir)
                removed.append(task_dir.name)
        except (ValueError, OSError):
            pass

    # Update manifest
    if removed:
        manifest = load_manifest()
        for task_id in removed:
            manifest["tasks"].pop(task_id, None)
        save_manifest(manifest)

    return removed


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def print_task_table(tasks: list[TaskMeta]) -> None:
    """Print tasks in a table format."""
    if not tasks:
        print("No tasks found.")
        return

    # Header
    print(
        f"{'ID':<25} {'STATUS':<10} {'ISSUE':<7} {'PID':<8} {'STARTED':<20} {'DESCRIPTION'}"
    )
    print("-" * 100)

    for task in tasks:
        started = task.started_at[:19].replace("T", " ")
        issue = f"#{task.issue}" if task.issue else "-"
        pid = str(task.pid) if task.pid else "-"
        desc = (
            task.description[:40] + "..."
            if len(task.description) > 40
            else task.description
        )

        print(
            f"{task.task_id:<25} {task.status:<10} {issue:<7} {pid:<8} {started:<20} {desc}"
        )


@dataclass
class CLIArgs:
    """Parsed CLI arguments for command handlers."""

    command: str | None
    # start command
    id: str | None = None
    issue: int | None = None
    timeout: int = DEFAULT_TIMEOUT
    description: str = ""
    command_args: list[str] = field(default_factory=list)
    # list command
    all: bool = False
    # status/tail/wait/kill commands
    task_id: str | None = None
    # tail command
    lines: int = 50
    follow: bool = False
    # cleanup command
    days: int = 7
    force: bool = False


def handle_start(args: CLIArgs) -> int:
    """Handle the 'start' command."""
    assert args.id is not None  # Required by argparse
    cmd = args.command_args
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Error: No command specified", file=sys.stderr)
        return 1

    try:
        meta = start_task(
            command=cmd,
            task_id=args.id,
            issue=args.issue,
            timeout=args.timeout,
            description=args.description,
        )
        print(f"Started task {meta.task_id} (PID {meta.pid})")
        print(f"  Output: {get_task_dir(meta.task_id) / 'output.log'}")
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_list(args: CLIArgs) -> int:
    """Handle the 'list' command."""
    tasks = list_tasks(show_all=args.all)
    print_task_table(tasks)
    return 0


def handle_status(args: CLIArgs) -> int:
    """Handle the 'status' command."""
    assert args.task_id is not None  # Required by argparse
    meta = get_status(args.task_id)
    if meta is None:
        print(f"Task {args.task_id} not found", file=sys.stderr)
        return 1

    print(json.dumps(meta.to_dict(), indent=2))
    return 0


def handle_tail(args: CLIArgs) -> int:
    """Handle the 'tail' command."""
    assert args.task_id is not None  # Required by argparse
    if args.follow:
        tail_follow(args.task_id, args.lines)
    else:
        output = tail_output(args.task_id, args.lines)
        print(output)
    return 0


def tail_follow(task_id: str, lines: int) -> None:
    """Follow task output continuously."""
    try:
        while True:
            output = tail_output(task_id, lines)
            os.system("clear")
            print(f"=== Task: {task_id} ===\n")
            print(output)

            meta = get_status(task_id)
            if meta and meta.status != "running":
                print(f"\n=== Task {meta.status} ===")
                break

            time.sleep(2)
    except KeyboardInterrupt:
        pass


def handle_wait(args: CLIArgs) -> int:
    """Handle the 'wait' command."""
    assert args.task_id is not None  # Required by argparse
    try:
        meta = wait_for_task(args.task_id, timeout=args.timeout)
        print(f"Task {meta.task_id} {meta.status}")
        if meta.exit_code is not None:
            print(f"Exit code: {meta.exit_code}")
        return 0 if meta.status == "completed" else 1
    except (ValueError, TimeoutError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_kill(args: CLIArgs) -> int:
    """Handle the 'kill' command."""
    assert args.task_id is not None  # Required by argparse
    meta = kill_task(args.task_id)
    if meta is None:
        print(f"Task {args.task_id} not found", file=sys.stderr)
        return 1
    print(f"Task {meta.task_id} killed")
    return 0


def handle_cleanup(args: CLIArgs) -> int:
    """Handle the 'cleanup' command."""
    removed = cleanup_tasks(days=args.days, force=args.force)
    if removed:
        print(f"Removed {len(removed)} tasks: {', '.join(removed)}")
    else:
        print("No tasks to clean up")
    return 0


# Command handler dispatch table
COMMAND_HANDLERS: dict[str, Callable[..., int]] = {
    "start": handle_start,
    "list": handle_list,
    "status": handle_status,
    "tail": handle_tail,
    "wait": handle_wait,
    "kill": handle_kill,
    "cleanup": handle_cleanup,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        description="Background task management for long-running operations",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # start
    start_parser = subparsers.add_parser("start", help="Start a background task")
    start_parser.add_argument("--id", required=True, help="Task ID")
    start_parser.add_argument("--issue", type=int, help="Related issue number")
    start_parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds"
    )
    start_parser.add_argument(
        "--description", "-d", default="", help="Task description"
    )
    start_parser.add_argument(
        "command_args", nargs=argparse.REMAINDER, help="Command to run"
    )

    # list
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument(
        "--all", "-a", action="store_true", help="Show all tasks, not just running"
    )

    # status
    status_parser = subparsers.add_parser("status", help="Get task status")
    status_parser.add_argument("task_id", help="Task ID")

    # tail
    tail_parser = subparsers.add_parser("tail", help="Tail task output")
    tail_parser.add_argument("task_id", help="Task ID")
    tail_parser.add_argument(
        "-n", "--lines", type=int, default=50, help="Number of lines"
    )
    tail_parser.add_argument(
        "-f", "--follow", action="store_true", help="Follow output"
    )

    # wait
    wait_parser = subparsers.add_parser("wait", help="Wait for task completion")
    wait_parser.add_argument("task_id", help="Task ID")
    wait_parser.add_argument(
        "--timeout", type=int, default=300, help="Wait timeout in seconds"
    )

    # kill
    kill_parser = subparsers.add_parser("kill", help="Kill a running task")
    kill_parser.add_argument("task_id", help="Task ID")

    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old tasks")
    cleanup_parser.add_argument(
        "--days", type=int, default=7, help="Remove tasks older than N days"
    )
    cleanup_parser.add_argument(
        "--force", action="store_true", help="Force removal of all tasks"
    )

    return parser


def namespace_to_cli_args(ns: argparse.Namespace) -> CLIArgs:
    """Convert argparse Namespace to CLIArgs dataclass."""
    return CLIArgs(
        command=ns.command,
        id=getattr(ns, "id", None),
        issue=getattr(ns, "issue", None),
        timeout=getattr(ns, "timeout", DEFAULT_TIMEOUT),
        description=getattr(ns, "description", ""),
        command_args=getattr(ns, "command_args", []),
        all=getattr(ns, "all", False),
        task_id=getattr(ns, "task_id", None),
        lines=getattr(ns, "lines", 50),
        follow=getattr(ns, "follow", False),
        days=getattr(ns, "days", 7),
        force=getattr(ns, "force", False),
    )


def main() -> int:
    """Main entry point for bg_task CLI."""
    parser = build_parser()
    ns = parser.parse_args()
    args = namespace_to_cli_args(ns)

    if args.command is None:
        parser.print_help()
        return 1

    handler = COMMAND_HANDLERS.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

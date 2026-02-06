# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""CLI handlers and argument parsing for bg_task package.

Contains:
- CLIArgs dataclass
- Command handlers (handle_start, handle_list, etc.)
- Argument parser builder
- Main entry point
- Formatting utilities (format_duration, print_task_table)
"""

from __future__ import annotations

__all__ = [
    "CLIArgs",
    "main",
    "format_duration",
    "print_task_table",
]

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Add repo root to path for version import
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.bg_task.api import (
    get_status,
    list_tasks,
    tail_output,
    wait_for_task,
)
from ai_template_scripts.bg_task.process import (
    cleanup_tasks,
    kill_task,
    start_task,
)
from ai_template_scripts.bg_task.storage import get_task_dir
from ai_template_scripts.bg_task.types import DEFAULT_TIMEOUT, TaskMeta
from ai_template_scripts.subprocess_utils import format_duration_compact
from ai_template_scripts.version import get_version  # noqa: E402


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


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form.

    DEPRECATED: Use format_duration_compact from subprocess_utils (#2535).
    """
    return format_duration_compact(seconds)


def print_task_table(tasks: list[TaskMeta]) -> None:
    """Print tasks in a table format.

    REQUIRES: tasks is list of TaskMeta (may be empty)
    ENSURES: output printed to stdout
    ENSURES: descriptions truncated to 40 chars with "..."
    """
    if not tasks:
        print("No tasks found.")
        return

    # Header
    hdr = f"{'ID':<25} {'STATUS':<10} {'ISSUE':<7} {'PID':<8} {'STARTED':<20} DESC"
    print(hdr)
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

        row = f"{task.task_id:<25} {task.status:<10} {issue:<7} {pid:<8}"
        print(f"{row} {started:<20} {desc}")


def tail_follow(task_id: str, lines: int) -> None:
    """Follow task output continuously.

    REQUIRES: task_id is non-empty string
    REQUIRES: lines > 0
    ENSURES: loops until task completes or KeyboardInterrupt
    ENSURES: clears screen and reprints output every 2 seconds
    """
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


def handle_start(args: CLIArgs) -> int:
    """Handle the 'start' command.

    REQUIRES: args.id is not None (enforced by argparse)
    ENSURES: return 0 on success, 1 on error
    ENSURES: task started in background on success
    """
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
    """Handle the 'list' command.

    ENSURES: return 0 (always succeeds)
    ENSURES: task table printed to stdout
    """
    tasks = list_tasks(show_all=args.all)
    print_task_table(tasks)
    return 0


def handle_status(args: CLIArgs) -> int:
    """Handle the 'status' command.

    REQUIRES: args.task_id is not None (enforced by argparse)
    ENSURES: return 0 if task found, 1 otherwise
    ENSURES: JSON metadata printed to stdout on success
    """
    assert args.task_id is not None  # Required by argparse
    meta = get_status(args.task_id)
    if meta is None:
        print(f"Task {args.task_id} not found", file=sys.stderr)
        return 1

    print(json.dumps(meta.to_dict(), indent=2))
    return 0


def handle_tail(args: CLIArgs) -> int:
    """Handle the 'tail' command.

    REQUIRES: args.task_id is not None (enforced by argparse)
    ENSURES: return 0 (always succeeds)
    ENSURES: task output printed to stdout
    """
    assert args.task_id is not None  # Required by argparse
    if args.follow:
        tail_follow(args.task_id, args.lines)
    else:
        output = tail_output(args.task_id, args.lines)
        print(output)
    return 0


def handle_wait(args: CLIArgs) -> int:
    """Handle the 'wait' command.

    REQUIRES: args.task_id is not None (enforced by argparse)
    ENSURES: return 0 if task completed successfully
    ENSURES: return 1 if task failed/killed/timeout or error
    """
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
    """Handle the 'kill' command.

    REQUIRES: args.task_id is not None (enforced by argparse)
    ENSURES: return 0 if task found, 1 otherwise
    ENSURES: task terminated on success
    """
    assert args.task_id is not None  # Required by argparse
    meta = kill_task(args.task_id)
    if meta is None:
        print(f"Task {args.task_id} not found", file=sys.stderr)
        return 1
    print(f"Task {meta.task_id} killed")
    return 0


def handle_cleanup(args: CLIArgs) -> int:
    """Handle the 'cleanup' command.

    ENSURES: return 0 (always succeeds)
    ENSURES: old tasks removed based on args.days threshold
    """
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
    """Build the argument parser for CLI commands.

    ENSURES: return has subparsers for all commands (start, list, status, tail, wait, kill, cleanup)
    """
    parser = argparse.ArgumentParser(
        description="Background task management for long-running operations",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("bg_task.py"),
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
    """Convert argparse Namespace to CLIArgs dataclass.

    REQUIRES: ns is valid argparse.Namespace
    ENSURES: return is CLIArgs with all fields populated from ns
    """
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
    """Main entry point for bg_task CLI.

    ENSURES: return 0 on success, 1 on error
    ENSURES: dispatches to appropriate handler based on command
    """
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

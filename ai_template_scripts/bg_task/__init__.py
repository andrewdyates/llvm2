# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Background task management for long-running operations.

PURPOSE: Run tasks that outlive worker iteration timeouts (builds, tests, etc).
CALLED BY: WORKER AI (via CLI), human (debugging)

Public API:
    from ai_template_scripts.bg_task import (
        TaskMeta,         # Task metadata dataclass
        start_task,       # Start a background task
        list_tasks,       # List all tasks
        get_status,       # Get task status
        tail_output,      # Get task output
        wait_for_task,    # Wait for completion
        kill_task,        # Kill running task
        cleanup_tasks,    # Remove old tasks
        DEFAULT_TIMEOUT,  # Default task timeout (2 hours)
        TASKS_DIR,        # Background tasks directory
        main,             # CLI entry point
    )

CLI usage:
    python -m ai_template_scripts.bg_task start --id <id> -- <command>
    python -m ai_template_scripts.bg_task list [--all]
    python -m ai_template_scripts.bg_task status <task_id>
    python -m ai_template_scripts.bg_task tail <task_id> [-f]
    python -m ai_template_scripts.bg_task wait <task_id>
    python -m ai_template_scripts.bg_task kill <task_id>
    python -m ai_template_scripts.bg_task cleanup [--days N] [--force]

MODULE CONTRACTS:
    INV: manifest.json reflects state of task subdirectories
    INV: Running tasks have valid PID files with live processes
    INV: Task status transitions: running -> {completed, failed, timeout, killed}
"""

from ai_template_scripts.bg_task.api import (
    get_status,
    list_tasks,
    tail_output,
    wait_for_task,
)
from ai_template_scripts.bg_task.cli import (
    CLIArgs,
    build_parser,
    handle_cleanup,
    handle_kill,
    handle_list,
    handle_start,
    handle_status,
    handle_tail,
    handle_wait,
    main,
    namespace_to_cli_args,
    print_task_table,
    tail_follow,
)
from ai_template_scripts.bg_task.process import (
    cleanup_tasks,
    is_process_alive,
    kill_task,
    start_task,
    update_task_status,
)
from ai_template_scripts.bg_task.storage import (
    get_git_root,
    get_task_dir,
    get_tasks_dir,
    load_manifest,
    load_task_meta,
    save_manifest,
    save_task_meta,
)
from ai_template_scripts.bg_task.types import DEFAULT_TIMEOUT, TASKS_DIR, TaskMeta

__all__ = [
    # Public API
    "TaskMeta",
    "start_task",
    "list_tasks",
    "get_status",
    "tail_output",
    "wait_for_task",
    "kill_task",
    "cleanup_tasks",
    "DEFAULT_TIMEOUT",
    "TASKS_DIR",
    "main",
    # Internal - re-exported for test compatibility
    "CLIArgs",
    "build_parser",
    "namespace_to_cli_args",
    "print_task_table",
    "tail_follow",
    "handle_start",
    "handle_list",
    "handle_status",
    "handle_tail",
    "handle_wait",
    "handle_kill",
    "handle_cleanup",
    "get_git_root",
    "get_task_dir",
    "get_tasks_dir",
    "load_manifest",
    "load_task_meta",
    "save_manifest",
    "save_task_meta",
    "is_process_alive",
    "update_task_status",
]

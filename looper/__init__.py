"""
looper - Autonomous continuous loop for AI workers and managers

Modular package structure:
- config: Configuration loading, frontmatter parsing, role config
- context: Session context gathering (git log, issues, directives, feedback)
- rotation: Phase rotation state management
- hooks: Git hook installation and management

The main LoopRunner class is in looper/runner.py.

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0
"""

from looper.config import (
    ITERATION_FILE_TEMPLATE,
    LOG_DIR,
    MAX_CRASH_LOG_LINES,
    MAX_LOG_FILES,
    PID_FILE_TEMPLATE,
    ROLES_DIR,
    STATUS_FILE_TEMPLATE,
    build_codex_context,
    get_project_name,
    inject_content,
    load_project_config,
    load_role_config,
    parse_frontmatter,
    parse_phase_blocks,
)
from looper.context import (
    run_session_start_commands,
)
from looper.hooks import (
    HOOK_CONFIGS,
    install_hooks,
)
from looper.rotation import (
    ROTATION_STATE_FILE,
    get_rotation_focus,
    load_rotation_state,
    save_rotation_state,
    select_phase_by_priority,
    update_rotation_state,
)
from looper.runner import (
    LoopRunner,
    build_audit_prompt,
    main,
    show_prompt,
)

__all__ = [
    # config
    "LOG_DIR",
    "ROLES_DIR",
    "ITERATION_FILE_TEMPLATE",
    "PID_FILE_TEMPLATE",
    "STATUS_FILE_TEMPLATE",
    "MAX_LOG_FILES",
    "MAX_CRASH_LOG_LINES",
    "parse_frontmatter",
    "parse_phase_blocks",
    "load_project_config",
    "load_role_config",
    "get_project_name",
    "inject_content",
    "build_codex_context",
    # context
    "run_session_start_commands",
    # rotation
    "ROTATION_STATE_FILE",
    "load_rotation_state",
    "save_rotation_state",
    "update_rotation_state",
    "select_phase_by_priority",
    "get_rotation_focus",
    # hooks
    "HOOK_CONFIGS",
    "install_hooks",
    # runner
    "build_audit_prompt",
    "LoopRunner",
    "show_prompt",
    "main",
]

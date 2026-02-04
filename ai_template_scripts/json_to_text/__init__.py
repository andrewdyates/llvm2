#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
json_to_text - Format AI CLI streaming output for human readability

PURPOSE: Converts Claude/Codex JSON streaming output to readable terminal text.
CALLED BY: looper.py (run_iteration pipes AI output through this)
REFERENCED: looper.py (setup checks existence), tests/test_json_to_text.py
SCHEMA: docs/logging.md (OpenTelemetry-aligned log schema)

Supports:
- Claude CLI: --output-format stream-json
- Codex CLI: --json (JSONL events)

Public API (library usage):
    from ai_template_scripts.json_to_text import (
        MessageFormatter,       # Claude message formatter class
        CodexFormatter,         # Codex event formatter class
        CompactFormatter,       # Compact formatter for manager audits
        clean_output,           # Remove system noise from output
        format_tool_output,     # Format tool output intelligently
        process_message,        # Process a Claude message
        process_codex_event,    # Process a Codex JSONL event
        is_codex_event,         # Detect Codex vs Claude event
        COMPACT_MODE,           # Global flag for compact output mode
    )

CLI usage:
    Pipe JSON streaming output: <ai-cli> | ./json_to_text.py
    ./json_to_text.py --version  # Show version information
    ./json_to_text.py --compact  # Compact mode for manager audits

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import json
import sys
from pathlib import Path

# Re-export submodules for direct access (enables jtt.formatters._session_complete_printed)
from . import codex, formatters, utils

# Re-export public API from submodules
from .codex import (
    CODEX_EVENT_HANDLERS,
    CODEX_ITEM_HANDLERS,
    CodexFormatter,
    _handle_error,
    _handle_item_event,
    _handle_thread_started,
    _handle_turn_completed,
    _handle_turn_failed,
    _handle_turn_started,
    _print_usage_stats,
    codex_formatter,
    is_codex_event,
    process_codex_event,
)
from .formatters import (
    CLAUDE_MSG_HANDLERS,
    COMPACT_MODE,
    MAX_PENDING_TOOLS,
    TOOL_DESC_FORMATTERS,
    CompactFormatter,
    MessageFormatter,
    _build_tool_description,
    _desc_bash,
    _desc_edit,
    _desc_glob,
    _desc_grep,
    _desc_lsp,
    _desc_read,
    _desc_task,
    _desc_todo_write,
    _desc_web_fetch,
    _desc_web_search,
    _desc_write,
    _extract_result_text,
    _extract_role_and_content,
    _handle_init,
    _handle_result,
    _handle_text_block,
    _handle_thinking_block,
    _handle_tool_result_block,
    _print_claude_stats,
    _process_content_block,
    _session_complete_printed,
    _store_tool_use,
    clear_pending_tools,
    compact_formatter,
    formatter,
    pending_tool_uses,
    process_message,
)
from .utils import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    ERROR_PATTERNS,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    USE_COLORS,
    YELLOW,
    _coerce_output_text,
    _decode_bytes,
    _is_error_result,
    _truncate,
    clean_output,
    format_tool_output,
    timestamp,
)

__all__ = [
    # Submodules (re-exported for direct access like jtt.formatters)
    "codex",
    "formatters",
    "utils",
    # Main public API
    "MessageFormatter",
    "CodexFormatter",
    "CompactFormatter",
    "clean_output",
    "format_tool_output",
    "process_message",
    "process_codex_event",
    "is_codex_event",
    "COMPACT_MODE",
    "main",
    # Formatter instances
    "formatter",
    "compact_formatter",
    "codex_formatter",
    # Color constants
    "USE_COLORS",
    "BLUE",
    "GREEN",
    "YELLOW",
    "RED",
    "CYAN",
    "MAGENTA",
    "BOLD",
    "DIM",
    "RESET",
    # Utility functions
    "timestamp",
    "_decode_bytes",
    "_coerce_output_text",
    "_is_error_result",
    "_truncate",
    # Error patterns
    "ERROR_PATTERNS",
    # Tool description formatters
    "TOOL_DESC_FORMATTERS",
    "_build_tool_description",
    "_desc_read",
    "_desc_write",
    "_desc_edit",
    "_desc_bash",
    "_desc_grep",
    "_desc_glob",
    "_desc_todo_write",
    "_desc_task",
    "_desc_web_fetch",
    "_desc_web_search",
    "_desc_lsp",
    # Claude message handlers
    "CLAUDE_MSG_HANDLERS",
    "_extract_role_and_content",
    "_handle_init",
    "_handle_result",
    "_handle_text_block",
    "_handle_thinking_block",
    "_store_tool_use",
    "_extract_result_text",
    "_handle_tool_result_block",
    "_process_content_block",
    "_print_claude_stats",
    # Codex handlers
    "CODEX_EVENT_HANDLERS",
    "CODEX_ITEM_HANDLERS",
    "_handle_thread_started",
    "_handle_turn_started",
    "_handle_turn_completed",
    "_handle_turn_failed",
    "_handle_error",
    "_handle_item_event",
    "_print_usage_stats",
    # Tool state
    "pending_tool_uses",
    "MAX_PENDING_TOOLS",
    "clear_pending_tools",
    "_session_complete_printed",
]


def _get_version() -> str:
    """Get version information."""
    # Support running as script (not just as module)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from ai_template_scripts.version import get_version

        return get_version("json_to_text.py")
    except ImportError:
        return "json_to_text.py (version unavailable)"


def main() -> None:
    """Main entry point"""
    # Import formatters module to set COMPACT_MODE

    if "--version" in sys.argv:
        print(_get_version())
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --version     Show version information")
        print("  --compact     Compact mode for manager audits (one line per action)")
        print("  -h, --help    Show this help message")
        return

    # Enable compact mode
    if "--compact" in sys.argv:
        formatters.COMPACT_MODE = True

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                # Dispatch to appropriate handler based on format
                if is_codex_event(msg):
                    process_codex_event(
                        msg,
                        compact_mode=formatters.COMPACT_MODE,
                        compact_formatter=compact_formatter,
                        clear_pending_tools_func=clear_pending_tools,
                    )
                else:
                    process_message(msg)
                sys.stdout.flush()
            except json.JSONDecodeError:
                # Not valid JSON, might be regular output
                print(line)
                continue

    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️  Interrupted{RESET}")
        sys.exit(0)
    except BrokenPipeError:
        # Handle pipe closing gracefully
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Codex JSONL event formatting.

This module handles Codex CLI --json JSONL event formatting.
"""

from typing import Any

from .utils import (
    BLUE,
    BOLD,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    _coerce_output_text,
    clean_output,
    timestamp,
)

__all__ = [
    "CodexFormatter",
    "codex_formatter",
    "process_codex_event",
    "is_codex_event",
    "CODEX_EVENT_HANDLERS",
    "CODEX_ITEM_HANDLERS",
    # Internal handlers (exported for testing)
    "_handle_thread_started",
    "_handle_turn_started",
    "_handle_turn_completed",
    "_handle_turn_failed",
    "_handle_error",
    "_handle_item_event",
    "_print_usage_stats",
]


class CodexFormatter:
    """Format Codex CLI --json JSONL events"""

    def __init__(self) -> None:
        self.last_was_text = False
        self.in_session = False

    def format_agent_message(
        self,
        item: dict[str, Any],
        compact_mode: bool = False,
        compact_formatter: Any = None,
    ) -> None:
        """Format agent text messages"""
        # Codex uses 'text' field in newer versions, 'content' in older
        content = item.get("text", item.get("content", ""))
        if not content or not content.strip():
            return

        text = clean_output(content)
        if not text.strip():
            return

        # Compact mode: delegate to compact formatter
        if compact_mode and compact_formatter is not None:
            compact_formatter.format_text(text)
            return

        paragraphs = text.split("\n\n")
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            if i == 0:
                print(f"\n{DIM}[{timestamp()}]{RESET} {BOLD}{BLUE}💬{RESET} {para}")
            else:
                print(f"   {para}")

        self.last_was_text = True

    def format_command_execution(
        self,
        item: dict[str, Any],
        status: str = "completed",
        compact_mode: bool = False,
        compact_formatter: Any = None,
    ) -> None:
        """Format shell command executions"""
        command = item.get("command", "")
        exit_code = item.get("exit_code")
        output = item.get("aggregated_output", "")
        if output is None:
            output = ""
        elif not isinstance(output, str):
            output = _coerce_output_text(output)

        is_error = exit_code is not None and exit_code != 0

        # Compact mode: use compact formatter
        if compact_mode and compact_formatter is not None:
            compact_formatter.format_tool(
                "Bash",
                {"command": command},
                output if is_error else None,
            )
            return

        # Show full command (no truncation)
        cmd_display = command

        if is_error:
            print(f"\n  {RED}✗{RESET} bash: {cmd_display} (exit {exit_code})")
            if output:
                lines = output.strip().split("\n")[:10]
                for line in lines:
                    print(f"    {RED}{line}{RESET}")
        else:
            print(f"  {DIM}•{RESET} bash: {cmd_display}")
            if output and status == "completed":
                lines = output.strip().split("\n")
                if len(lines) <= 3:
                    for line in lines:
                        print(f"    {DIM}→{RESET} {line}")
                else:
                    print(f"    {DIM}→{RESET} {lines[0]}")
                    print(f"    {DIM}... ({len(lines) - 2} more lines){RESET}")
                    print(f"    {DIM}→{RESET} {lines[-1]}")

        self.last_was_text = False

    def format_file_change(
        self,
        item: dict[str, Any],
        compact_mode: bool = False,
        compact_formatter: Any = None,
    ) -> None:
        """Format file operations"""
        filepath = item.get("file_path", item.get("path", ""))
        change_type = item.get("change_type", "modify")

        # Compact mode: use compact formatter
        if compact_mode and compact_formatter is not None:
            tool_name = "Write" if change_type == "create" else "Edit"
            compact_formatter.format_tool(tool_name, {"file_path": filepath}, None)
            return

        if change_type == "create":
            print(f"  {DIM}•{RESET} write: {filepath}")
        elif change_type == "delete":
            print(f"  {DIM}•{RESET} delete: {filepath}")
        else:
            print(f"  {DIM}•{RESET} edit: {filepath}")

        self.last_was_text = False

    def format_reasoning(
        self,
        item: dict[str, Any],
        compact_mode: bool = False,
        compact_formatter: Any = None,  # Accept but ignore - reasoning has no compact output
    ) -> None:
        """Format reasoning/thinking blocks with preview"""
        # Compact mode: suppress reasoning blocks entirely (no compact alternative)
        if compact_mode:
            return

        # Codex uses 'text' field in newer versions, 'content' or 'summary' in older
        content = item.get("text", item.get("content", item.get("summary", "")))
        if content and len(content) > 50:
            # Show first ~100 chars as preview
            preview = content[:100].replace("\n", " ").strip()
            if len(content) > 100:
                preview += "..."
            print(f"  {DIM}💭 {preview}{RESET}")
            self.last_was_text = False

    def format_mcp_tool_call(self, item: dict[str, Any]) -> None:
        """Format MCP tool calls"""
        tool_name = item.get("tool_name", item.get("name", "mcp_tool"))
        print(f"  {DIM}•{RESET} mcp: {tool_name}")
        self.last_was_text = False

    def format_web_search(self, item: dict[str, Any]) -> None:
        """Format web search operations"""
        query = item.get("query", "")
        print(f"  {DIM}•{RESET} search: {query[:60]}{'...' if len(query) > 60 else ''}")
        self.last_was_text = False

    def format_todo_list(self, item: dict[str, Any]) -> None:
        """Format todo list updates"""
        todos = item.get("todos", [])
        print(f"  {DIM}•{RESET} todo: update ({len(todos)} items)")
        self.last_was_text = False

    def format_error(self, msg: dict[str, Any]) -> None:
        """Format error events"""
        error = msg.get("error", {})
        message = (
            error.get("message", str(error)) if isinstance(error, dict) else str(error)
        )
        print(f"\n  {RED}✗ Error: {message}{RESET}")
        self.last_was_text = False


# Global codex formatter instance
codex_formatter = CodexFormatter()


def _print_usage_stats(usage: dict) -> None:
    """Print token usage statistics."""
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("cached_input_tokens", 0)
    print(f"{DIM}  Input: {input_tokens:,} tokens", end="")
    if cached_tokens:
        print(f" (cached: {cached_tokens:,})", end="")
    print(f" | Output: {output_tokens:,} tokens{RESET}")


def _handle_thread_started(
    msg: dict,
    clear_pending_tools_func: Any = None,
) -> None:
    """Handle thread.started event."""
    thread_id = msg.get("thread_id", "")
    if clear_pending_tools_func is not None:
        clear_pending_tools_func()
    print(f"\n{BOLD}{MAGENTA}{'═' * 80}{RESET}")
    print(f"{BOLD}{MAGENTA}  🚀  Codex Session Started  {RESET}")
    if thread_id:
        print(f"{DIM}  Thread: {thread_id}{RESET}")
    print(f"{BOLD}{MAGENTA}{'═' * 80}{RESET}")
    codex_formatter.in_session = True


def _handle_turn_started(_msg: dict) -> None:
    """Handle turn.started event (no-op, just note silently)."""


def _handle_turn_completed(msg: dict) -> None:
    """Handle turn.completed event."""
    codex_formatter.in_session = False
    print(f"\n{DIM}{'─' * 80}{RESET}")
    print(f"{BOLD}{GREEN}  ✓  Turn Complete{RESET}")
    usage = msg.get("usage", {})
    if usage:
        _print_usage_stats(usage)
    print(f"{DIM}{'─' * 80}{RESET}")


def _handle_turn_failed(msg: dict) -> None:
    """Handle turn.failed event."""
    error = msg.get("error", {})
    message = (
        error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
    )
    print(f"\n{RED}{'─' * 80}{RESET}")
    print(f"{BOLD}{RED}  ✗  Turn Failed: {message}{RESET}")
    print(f"{RED}{'─' * 80}{RESET}")


def _handle_error(msg: dict) -> None:
    """Handle error event."""
    codex_formatter.format_error(msg)


# Dispatch table for Codex item types
CODEX_ITEM_HANDLERS: dict[str, Any] = {
    "agent_message": lambda item, status, **kw: codex_formatter.format_agent_message(
        item, **kw
    ),
    "command_execution": lambda item,
    status,
    **kw: codex_formatter.format_command_execution(item, status, **kw),
    "file_change": lambda item, status, **kw: codex_formatter.format_file_change(
        item, **kw
    ),
    "reasoning": lambda item, status, **kw: codex_formatter.format_reasoning(
        item, **kw
    ),
    "mcp_tool_call": lambda item, status, **kw: codex_formatter.format_mcp_tool_call(
        item
    ),
    "web_search": lambda item, status, **kw: codex_formatter.format_web_search(item),
    "todo_list": lambda item, status, **kw: codex_formatter.format_todo_list(item),
}


def _handle_item_event(
    msg: dict,
    event_type: str,
    compact_mode: bool = False,
    compact_formatter: Any = None,
) -> None:
    """Handle item.completed, item.started, or item.updated events."""
    item = msg.get("item", {})
    item_type = item.get("type", "")
    status = item.get("status", "in_progress")

    # Only show completed items (or started for streaming agent_message)
    should_show = event_type == "item.completed" or (
        event_type == "item.started" and item_type == "agent_message"
    )
    if not should_show:
        return

    handler = CODEX_ITEM_HANDLERS.get(item_type)
    if handler:
        handler(
            item,
            status,
            compact_mode=compact_mode,
            compact_formatter=compact_formatter,
        )


# Dispatch table for Codex event types
CODEX_EVENT_HANDLERS: dict[str, Any] = {
    "thread.started": _handle_thread_started,
    "turn.started": _handle_turn_started,
    "turn.completed": _handle_turn_completed,
    "turn.failed": _handle_turn_failed,
    "error": _handle_error,
}


def process_codex_event(
    msg: dict,
    compact_mode: bool = False,
    compact_formatter: Any = None,
    clear_pending_tools_func: Any = None,
) -> None:
    """Process a Codex JSONL event."""
    event_type = msg.get("type", "")

    # Check direct event handlers first
    handler = CODEX_EVENT_HANDLERS.get(event_type)
    if handler:
        if event_type == "thread.started":
            handler(msg, clear_pending_tools_func=clear_pending_tools_func)
        else:
            handler(msg)
        return

    # Handle item events
    if event_type in ("item.completed", "item.started", "item.updated"):
        _handle_item_event(
            msg,
            event_type,
            compact_mode=compact_mode,
            compact_formatter=compact_formatter,
        )


def is_codex_event(msg: dict) -> bool:
    """Detect if this is a Codex event vs Claude message"""
    event_type = msg.get("type", "")
    # Codex uses dot-notation event types
    if "." in event_type:
        return True
    # Codex-specific top-level types
    if event_type in ("error",) and "item" not in msg and "message" not in msg:
        return True
    return False

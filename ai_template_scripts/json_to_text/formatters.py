#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Claude message formatting and tool result handling.

This module handles:
- Claude CLI stream-json message formatting
- Tool description formatting
- Compact mode formatting for manager audits
"""

import sys
from pathlib import Path
from typing import Any

from .utils import (
    BLUE,
    BOLD,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    _is_error_result,
    _truncate,
    format_tool_output,
    timestamp,
)

__all__ = [
    # Compact mode flag
    "COMPACT_MODE",
    # Formatters
    "CompactFormatter",
    "MessageFormatter",
    "compact_formatter",
    "formatter",
    # Tool description
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
    # Message processing
    "process_message",
    "clear_pending_tools",
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
    # Internal state
    "pending_tool_uses",
    "MAX_PENDING_TOOLS",
    "_session_complete_printed",
]

# Compact mode for manager audits (set via --compact flag or by main)
COMPACT_MODE = False


def _desc_read(input_data: dict[str, Any]) -> str:
    """Format description for Read tool call."""
    return f"read: {input_data.get('file_path', '')}"


def _desc_write(input_data: dict[str, Any]) -> str:
    """Format description for Write tool call."""
    path = input_data.get("file_path", "")
    size = len(input_data.get("content", ""))
    return f"write: {path} ({size} chars)"


def _desc_edit(input_data: dict[str, Any]) -> str:
    """Format description for Edit tool call."""
    return f"edit: {input_data.get('file_path', '')}"


def _desc_bash(input_data: dict[str, Any]) -> str:
    """Format description for Bash tool call."""
    cmd = input_data.get("command", "")
    return f"bash: {cmd}"


def _desc_grep(input_data: dict[str, Any]) -> str:
    """Format description for Grep tool call."""
    pattern = input_data.get("pattern", "")
    path = input_data.get("path", ".")
    return f"grep: '{pattern}' in {path}"


def _desc_glob(input_data: dict[str, Any]) -> str:
    """Format description for Glob tool call."""
    return f"glob: {input_data.get('pattern', '')}"


def _desc_todo_write(input_data: dict[str, Any]) -> str:
    """Format description for TodoWrite tool call."""
    todos = input_data.get("todos", [])
    return f"todo: update ({len(todos)} items)"


def _desc_task(input_data: dict[str, Any]) -> str:
    """Format description for Task tool call."""
    subagent = input_data.get("subagent_type", "agent")
    task_desc = input_data.get("description", "")
    if task_desc:
        return f"task: {subagent} → {task_desc}"
    return f"task: spawn {subagent}"


def _desc_web_fetch(input_data: dict[str, Any]) -> str:
    """Format description for WebFetch tool call."""
    url = _truncate(input_data.get("url", ""), 60)
    return f"fetch: {url}"


def _desc_web_search(input_data: dict[str, Any]) -> str:
    """Format description for WebSearch tool call."""
    query = input_data.get("query", "")
    return f"search: {_truncate(query, 50)}"


def _desc_lsp(input_data: dict[str, Any]) -> str:
    """Format description for LSP tool call."""
    operation = input_data.get("operation", "")
    filepath = input_data.get("filePath", "")
    return f"lsp: {operation} in {filepath}"


# Tool name -> description formatter mapping
TOOL_DESC_FORMATTERS: dict[str, Any] = {
    "Read": _desc_read,
    "Write": _desc_write,
    "Edit": _desc_edit,
    "Bash": _desc_bash,
    "Grep": _desc_grep,
    "Glob": _desc_glob,
    "TodoWrite": _desc_todo_write,
    "Task": _desc_task,
    "WebFetch": _desc_web_fetch,
    "WebSearch": _desc_web_search,
    "LSP": _desc_lsp,
}


def _build_tool_description(tool_name: str, input_data: dict[str, Any]) -> str:
    """Build human-readable description for a tool call."""
    formatter = TOOL_DESC_FORMATTERS.get(tool_name)
    if formatter:
        return formatter(input_data)
    return tool_name.lower()


def _print_tool_result(
    desc: str,
    tool_name: str,
    tool_result: Any,
    is_error: bool,
) -> None:
    """Print formatted tool result to stdout."""
    if is_error:
        print(f"\n  {RED}✗{RESET} {desc}")
        if tool_result:
            output_lines = format_tool_output(tool_result, tool_name, is_error=True)
            if output_lines:
                for line in output_lines:
                    print(f"    {RED}{line}{RESET}")
    else:
        print(f"  {DIM}•{RESET} {desc}")
        if tool_result:
            output_lines = format_tool_output(tool_result, tool_name, is_error=False)
            if output_lines:
                for line in output_lines:
                    print(f"    {DIM}→{RESET} {line}")


# =============================================================================
# Compact Mode for Manager Audits
# =============================================================================


def _classify_action(tool_name: str, input_data: dict[str, Any]) -> str:
    """Classify tool action into audit-friendly categories."""
    if tool_name in ("Read", "Grep", "Glob"):
        return "read"
    if tool_name in ("Write", "Edit"):
        return "write"
    if tool_name == "Bash":
        cmd = input_data.get("command", "").lower()
        if any(t in cmd for t in ("test", "pytest", "cargo test", "npm test")):
            return "test"
        if any(t in cmd for t in ("git", "gh ")):
            return "git"
        return "bash"
    if tool_name == "Task":
        return "task"
    if tool_name == "TodoWrite":
        return "todo"
    return "other"


def _compact_tool_line(
    tool_name: str, input_data: dict[str, Any], tool_result: Any, is_error: bool
) -> str:
    """Generate single-line compact output for a tool call."""
    ts = timestamp()
    action = _classify_action(tool_name, input_data)
    status = "ERR" if is_error else "ok"

    # Build a brief description
    if tool_name == "Bash":
        cmd = input_data.get("command", "")[:50]
        desc = cmd.replace("\n", " ")
    elif tool_name == "Read":
        desc = Path(input_data.get("file_path", "")).name
    elif tool_name in ("Write", "Edit"):
        desc = Path(input_data.get("file_path", "")).name
    elif tool_name == "Grep":
        desc = input_data.get("pattern", "")[:30]
    elif tool_name == "Glob":
        desc = input_data.get("pattern", "")[:30]
    elif tool_name == "Task":
        desc = input_data.get("description", "")[:30]
    else:
        desc = tool_name

    # Truncate description
    if len(desc) > 40:
        desc = desc[:37] + "..."

    color = RED if is_error else DIM
    return f"{DIM}{ts}{RESET} [{action:5}] {color}{status}{RESET} {desc}"


class CompactFormatter:
    """Compact output formatter for manager audits.

    Outputs one line per action with: timestamp, action type, status, description.
    Suppresses verbose thinking/reasoning blocks and collapses repeated actions.
    """

    def __init__(self) -> None:
        self.last_action = ""
        self.repeat_count = 0
        self.error_count = 0
        self.action_counts: dict[str, int] = {}

    def format_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        tool_result: Any,
    ) -> None:
        """Format a tool call in compact mode."""
        is_error = _is_error_result(tool_result)
        if is_error:
            self.error_count += 1

        line = _compact_tool_line(tool_name, input_data, tool_result, is_error)

        # Track action type for summary
        action = _classify_action(tool_name, input_data)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        # Collapse repeated identical actions
        if line == self.last_action:
            self.repeat_count += 1
            return

        # Flush any accumulated repeats
        if self.repeat_count > 0:
            print(f"  {DIM}... repeated {self.repeat_count}x{RESET}")
            self.repeat_count = 0

        print(line)
        self.last_action = line

    def format_text(self, text: str) -> None:
        """Format text messages - show only first line in compact mode."""
        text = text.strip()
        if not text:
            return
        first_line = text.split("\n")[0][:80]
        if len(text) > len(first_line):
            first_line += "..."
        print(f"{DIM}{timestamp()}{RESET} [msg  ] {first_line}")

    def format_session_start(self) -> None:
        """Format session start marker."""
        print(f"\n{BOLD}━━━ Session Start ━━━{RESET}")

    def format_session_end(self, usage: dict[str, Any] | None = None) -> None:
        """Format session end with summary."""
        # Flush any remaining repeats
        if self.repeat_count > 0:
            print(f"  {DIM}... repeated {self.repeat_count}x{RESET}")

        print(f"\n{BOLD}━━━ Session End ━━━{RESET}")

        # Print action summary
        if self.action_counts:
            summary_parts = [f"{k}:{v}" for k, v in sorted(self.action_counts.items())]
            print(f"{DIM}Actions: {', '.join(summary_parts)}{RESET}")

        if self.error_count > 0:
            print(f"{RED}Errors: {self.error_count}{RESET}")

        if usage:
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            print(f"{DIM}Tokens: in={inp:,} out={out:,}{RESET}")


# Global compact formatter instance
compact_formatter = CompactFormatter()


class MessageFormatter:
    """Format Claude CLI stream-json messages and tool results."""

    def __init__(self) -> None:
        self.last_was_text = False

    def format_text_message(self, text: str) -> None:
        """Format Claude's main messages"""
        if not text.strip():
            return

        # Compact mode: delegate to compact formatter
        if COMPACT_MODE:
            compact_formatter.format_text(text)
            return

        # Add spacing if last output was also text
        if self.last_was_text:
            print()

        # Clean up the text
        text = text.strip()

        # Split into paragraphs
        paragraphs = text.split("\n\n")

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # First paragraph gets timestamp and icon
            if i == 0:
                print(f"\n{DIM}[{timestamp()}]{RESET} {BOLD}{BLUE}💬{RESET} {para}")
            else:
                # Subsequent paragraphs are indented slightly
                print(f"   {para}")

        self.last_was_text = True

    def format_tool_use(
        self, tool_name: str, input_data: dict[str, Any] | None, tool_result: Any = None
    ) -> None:
        """Format a tool call with its result."""
        if input_data is None:
            input_data = {}

        # Compact mode: delegate to compact formatter
        if COMPACT_MODE:
            compact_formatter.format_tool(tool_name, input_data, tool_result)
            return

        desc = _build_tool_description(tool_name, input_data)
        is_error = _is_error_result(tool_result)
        _print_tool_result(desc, tool_name, tool_result, is_error)
        self.last_was_text = False

    def format_thinking(self, thinking_text: str) -> None:
        """Format thinking blocks with preview"""
        # Compact mode: suppress thinking blocks entirely
        if COMPACT_MODE:
            return

        if thinking_text and len(thinking_text) > 50:
            # Show first ~100 chars as preview
            preview = thinking_text[:100].replace("\n", " ").strip()
            if len(thinking_text) > 100:
                preview += "..."
            print(f"  {DIM}💭 {preview}{RESET}")
            self.last_was_text = False


formatter = MessageFormatter()

# Track tool uses and their results
# Limited to prevent memory leaks from interrupted sessions
MAX_PENDING_TOOLS = 100
pending_tool_uses: dict[str, dict] = {}

# Track session complete to prevent duplicates (#168)
_session_complete_printed = False


def clear_pending_tools() -> None:
    """Clear pending tools at session boundaries"""
    pending_tool_uses.clear()


def _extract_role_and_content(msg: dict) -> tuple[str | None, list]:
    """Extract role and content from a message, handling nested structures."""
    if "message" in msg:
        inner_msg = msg["message"]
        if isinstance(inner_msg, str):
            return "assistant", [{"type": "text", "text": inner_msg}]
        return inner_msg.get("role"), inner_msg.get("content", [])
    return msg.get("role"), msg.get("content", [])


def _handle_init(_msg: dict) -> None:
    """Handle init message type."""
    global _session_complete_printed
    clear_pending_tools()
    _session_complete_printed = False  # Reset for new session

    if COMPACT_MODE:
        compact_formatter.format_session_start()
        return

    print(f"\n{BOLD}{MAGENTA}{'═' * 80}{RESET}")
    print(f"{BOLD}{MAGENTA}  🚀  Claude Session Started  {RESET}")
    print(f"{BOLD}{MAGENTA}{'═' * 80}{RESET}")


def _print_claude_stats(stats: dict) -> None:
    """Print Claude session statistics."""
    input_tokens = stats.get("input_tokens", 0)
    output_tokens = stats.get("output_tokens", 0)
    cache_read = stats.get("cache_read_input_tokens", 0)
    print(f"{DIM}  Input: {input_tokens:,} tokens", end="")
    if cache_read:
        print(f" (cached: {cache_read:,})", end="")
    print(f" | Output: {output_tokens:,} tokens{RESET}")


def _handle_result(msg: dict) -> None:
    """Handle result message type. Only prints once per session (#168)."""
    global _session_complete_printed
    if _session_complete_printed:
        return  # Ignore duplicate result messages
    _session_complete_printed = True
    stats = msg.get("usage") or msg.get("stats", {})

    if COMPACT_MODE:
        compact_formatter.format_session_end(stats)
        return

    print(f"\n{DIM}{'─' * 80}{RESET}")
    print(f"{BOLD}{GREEN}  ✓  Session Complete{RESET}")
    if stats:
        _print_claude_stats(stats)
    print(f"{DIM}{'─' * 80}{RESET}\n")


def _handle_text_block(block: dict, role: str | None) -> None:
    """Handle text content block."""
    text = block.get("text", "")
    if role == "assistant" and text.strip():
        formatter.format_text_message(text)


def _handle_thinking_block(block: dict) -> None:
    """Handle thinking content block."""
    thinking = block.get("thinking", "")
    formatter.format_thinking(thinking)


def _store_tool_use(block: dict) -> None:
    """Store tool_use block for later pairing with result."""
    tool_id = block.get("id", "")
    tool_name = block.get("name", "")
    input_data = block.get("input", {})
    # Limit size to prevent memory leaks
    if len(pending_tool_uses) >= MAX_PENDING_TOOLS:
        oldest_key = next(iter(pending_tool_uses))
        del pending_tool_uses[oldest_key]
    pending_tool_uses[tool_id] = {
        "name": tool_name,
        "input": input_data,
    }


def _extract_result_text(content_data: Any) -> Any:
    """Extract text from tool result content."""
    if isinstance(content_data, list):
        text_parts = [
            item.get("text", "")
            for item in content_data
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_parts:
            return "\n".join(text_parts)
        return content_data
    return content_data


def _handle_tool_result_block(block: dict) -> None:
    """Handle tool_result content block."""
    tool_id = block.get("tool_use_id", "")
    content_data = block.get("content", "")
    result_text = _extract_result_text(content_data)

    if tool_id in pending_tool_uses:
        tool_info = pending_tool_uses[tool_id]
        formatter.format_tool_use(
            tool_info["name"],
            tool_info["input"],
            result_text,
        )
        del pending_tool_uses[tool_id]
    else:
        print(
            f"  {DIM}• tool_result (orphan, id="
            f"{tool_id[:8] if tool_id else '?'}...){RESET}",
            file=sys.stderr,
        )


def _process_content_block(block: dict, role: str | None) -> None:
    """Process a single content block."""
    block_type = block.get("type")

    if block_type == "text":
        _handle_text_block(block, role)
    elif block_type == "thinking":
        _handle_thinking_block(block)
    elif block_type == "tool_use":
        _store_tool_use(block)
    elif block_type == "tool_result":
        _handle_tool_result_block(block)


# Dispatch table for Claude message types
CLAUDE_MSG_HANDLERS: dict[str, Any] = {
    "init": _handle_init,
    "result": _handle_result,
}


def process_message(msg: dict) -> None:
    """Process a single message from the stream."""
    msg_type = msg.get("type")

    # Check for special message types
    if msg_type is not None:
        handler = CLAUDE_MSG_HANDLERS.get(msg_type)
        if handler:
            handler(msg)
            return

    # Extract role and content
    role, content = _extract_role_and_content(msg)

    # Normalize content to list
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    # Process each content block
    for block in content:
        _process_content_block(block, role)

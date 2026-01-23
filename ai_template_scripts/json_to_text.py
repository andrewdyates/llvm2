#!/usr/bin/env python3
"""
json_to_text.py - Format AI CLI streaming output for human readability

PURPOSE: Converts Claude/Codex JSON streaming output to readable terminal text.
CALLED BY: looper.py (run_iteration pipes AI output through this)
REFERENCED: looper.py (setup checks existence), tests/test_json_to_text.py

Supports:
- Claude CLI: --output-format stream-json
- Codex CLI: --json (JSONL events)

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

# Detect if we should use colors (TTY or forced via env)
USE_COLORS = sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "") == "1"

# ANSI color codes (empty strings if no TTY)
if USE_COLORS:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
else:
    BLUE = GREEN = YELLOW = RED = CYAN = MAGENTA = BOLD = DIM = RESET = ""


def timestamp():
    """Get current timestamp for display"""
    return datetime.now().strftime("%H:%M:%S")


def clean_output(text):
    """Remove system noise from output"""
    if text is None:
        return ""
    lines = text.strip().split("\n")
    filtered = []
    skip_mode = False

    for line in lines:
        # Skip Co-Authored-By
        if "Co-Authored-By:" in line or "🤖 Generated with" in line:
            continue
        # Skip system reminders
        if "<system-reminder>" in line:
            skip_mode = True
            continue
        if "</system-reminder>" in line:
            skip_mode = False
            continue
        if skip_mode:
            continue
        # Skip malware check reminders
        if "you should consider whether it would be considered malware" in line.lower():
            continue
        filtered.append(line)

    return "\n".join(filtered)


def format_tool_output(content, tool_name, is_error=False):
    """Format tool output intelligently"""
    text = clean_output(content)
    if not text.strip():
        return None

    lines = text.split("\n")

    # For errors, show more context
    if is_error:
        preview_lines = lines[:15]
        if len(lines) > 15:
            preview_lines.append(f"{DIM}... ({len(lines) - 15} more lines){RESET}")
        return preview_lines

    # For successful tools, show smart preview
    if tool_name == "Bash":
        # Show first few lines and last line
        if len(lines) <= 3:
            return lines
        return [
            lines[0],
            lines[1],
            f"{DIM}... ({len(lines) - 3} more lines){RESET}",
            lines[-1],
        ]

    if tool_name == "Read":
        # Just show how many lines read
        return [f"{DIM}({len(lines)} lines read){RESET}"]

    if tool_name in ["Write", "Edit"]:
        # Just confirm success
        return None

    if tool_name in ["Grep", "Glob"]:
        # Show first few matches
        if len(lines) <= 5:
            return lines
        result = lines[:5]
        result.append(f"{DIM}... ({len(lines) - 5} more matches){RESET}")
        return result

    # Generic: show first line
    if len(lines) <= 2:
        return lines
    return [lines[0], f"{DIM}... ({len(lines) - 1} more lines){RESET}"]


# Error patterns for tool result detection (lowercased for comparison)
ERROR_PATTERNS = (
    "error:",  # Error: message
    "error -",  # Error - message
    "failed:",  # Failed: message
    "command failed",  # Command failed
    "exit code",  # Non-zero exit code
    "permission denied",
    "no such file",
    "not found",
    "traceback",  # Python traceback
    "exception:",  # Exception: message
    '"is_error": true',  # JSON error flag
    '"is_error":true',
)


def _is_error_result(tool_result: str | None) -> bool:
    """Check if tool result indicates an error."""
    if not tool_result:
        return False
    result_lower = tool_result.lower()
    return any(pattern in result_lower for pattern in ERROR_PATTERNS)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _desc_read(input_data: dict[str, Any]) -> str:
    return f"read: {input_data.get('file_path', '')}"


def _desc_write(input_data: dict[str, Any]) -> str:
    path = input_data.get("file_path", "")
    size = len(input_data.get("content", ""))
    return f"write: {path} ({size} chars)"


def _desc_edit(input_data: dict[str, Any]) -> str:
    return f"edit: {input_data.get('file_path', '')}"


def _desc_bash(input_data: dict[str, Any]) -> str:
    cmd = input_data.get("command", "")
    return f"bash: {cmd}"


def _desc_grep(input_data: dict[str, Any]) -> str:
    pattern = input_data.get("pattern", "")
    path = input_data.get("path", ".")
    return f"grep: '{pattern}' in {path}"


def _desc_glob(input_data: dict[str, Any]) -> str:
    return f"glob: {input_data.get('pattern', '')}"


def _desc_todo_write(input_data: dict[str, Any]) -> str:
    todos = input_data.get("todos", [])
    return f"todo: update ({len(todos)} items)"


def _desc_task(input_data: dict[str, Any]) -> str:
    subagent = input_data.get("subagent_type", "agent")
    task_desc = input_data.get("description", "")
    if task_desc:
        return f"task: {subagent} → {task_desc}"
    return f"task: spawn {subagent}"


def _desc_web_fetch(input_data: dict[str, Any]) -> str:
    url = _truncate(input_data.get("url", ""), 60)
    return f"fetch: {url}"


def _desc_web_search(input_data: dict[str, Any]) -> str:
    query = input_data.get("query", "")
    return f"search: {_truncate(query, 50)}"


def _desc_lsp(input_data: dict[str, Any]) -> str:
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
    tool_result: str | None,
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


class MessageFormatter:
    def __init__(self):
        self.last_was_text = False

    def format_text_message(self, text):
        """Format Claude's main messages"""
        if not text.strip():
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

    def format_tool_use(self, tool_name, input_data, tool_result=None):
        """Format a tool call with its result."""
        if input_data is None:
            input_data = {}

        desc = _build_tool_description(tool_name, input_data)
        is_error = _is_error_result(tool_result)
        _print_tool_result(desc, tool_name, tool_result, is_error)
        self.last_was_text = False

    def format_thinking(self, thinking_text):
        """Format thinking blocks with preview"""
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


def clear_pending_tools():
    """Clear pending tools at session boundaries"""
    pending_tool_uses.clear()


# =============================================================================
# Codex JSONL Format Handler
# =============================================================================


class CodexFormatter:
    """Format Codex CLI --json JSONL events"""

    def __init__(self):
        self.last_was_text = False
        self.in_session = False

    def format_agent_message(self, item):
        """Format agent text messages"""
        # Codex uses 'text' field in newer versions, 'content' in older
        content = item.get("text", item.get("content", ""))
        if not content or not content.strip():
            return

        text = clean_output(content)
        if not text.strip():
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

    def format_command_execution(self, item, status="completed"):
        """Format shell command executions"""
        command = item.get("command", "")
        exit_code = item.get("exit_code")
        output = item.get("aggregated_output", "")

        # Show full command (no truncation)
        cmd_display = command

        is_error = exit_code is not None and exit_code != 0

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

    def format_file_change(self, item):
        """Format file operations"""
        filepath = item.get("file_path", item.get("path", ""))
        change_type = item.get("change_type", "modify")

        if change_type == "create":
            print(f"  {DIM}•{RESET} write: {filepath}")
        elif change_type == "delete":
            print(f"  {DIM}•{RESET} delete: {filepath}")
        else:
            print(f"  {DIM}•{RESET} edit: {filepath}")

        self.last_was_text = False

    def format_reasoning(self, item):
        """Format reasoning/thinking blocks with preview"""
        # Codex uses 'text' field in newer versions, 'content' or 'summary' in older
        content = item.get("text", item.get("content", item.get("summary", "")))
        if content and len(content) > 50:
            # Show first ~100 chars as preview
            preview = content[:100].replace("\n", " ").strip()
            if len(content) > 100:
                preview += "..."
            print(f"  {DIM}💭 {preview}{RESET}")
            self.last_was_text = False

    def format_mcp_tool_call(self, item):
        """Format MCP tool calls"""
        tool_name = item.get("tool_name", item.get("name", "mcp_tool"))
        print(f"  {DIM}•{RESET} mcp: {tool_name}")
        self.last_was_text = False

    def format_web_search(self, item):
        """Format web search operations"""
        query = item.get("query", "")
        print(f"  {DIM}•{RESET} search: {query[:60]}{'...' if len(query) > 60 else ''}")
        self.last_was_text = False

    def format_todo_list(self, item):
        """Format todo list updates"""
        todos = item.get("todos", [])
        print(f"  {DIM}•{RESET} todo: update ({len(todos)} items)")
        self.last_was_text = False

    def format_error(self, msg):
        """Format error events"""
        error = msg.get("error", {})
        message = (
            error.get("message", str(error)) if isinstance(error, dict) else str(error)
        )
        print(f"\n  {RED}✗ Error: {message}{RESET}")
        self.last_was_text = False


codex_formatter = CodexFormatter()


def _handle_thread_started(msg: dict) -> None:
    """Handle thread.started event."""
    thread_id = msg.get("thread_id", "")
    clear_pending_tools()
    print(f"\n{BOLD}{MAGENTA}{'═' * 80}{RESET}")
    print(f"{BOLD}{MAGENTA}  🚀  Codex Session Started  {RESET}")
    if thread_id:
        print(f"{DIM}  Thread: {thread_id}{RESET}")
    print(f"{BOLD}{MAGENTA}{'═' * 80}{RESET}")
    codex_formatter.in_session = True


def _handle_turn_started(_msg: dict) -> None:
    """Handle turn.started event (no-op, just note silently)."""


def _print_usage_stats(usage: dict) -> None:
    """Print token usage statistics."""
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("cached_input_tokens", 0)
    print(f"{DIM}  Input: {input_tokens:,} tokens", end="")
    if cached_tokens:
        print(f" (cached: {cached_tokens:,})", end="")
    print(f" | Output: {output_tokens:,} tokens{RESET}")


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
    "agent_message": lambda item, status: codex_formatter.format_agent_message(item),
    "command_execution": lambda item, status: codex_formatter.format_command_execution(
        item, status
    ),
    "file_change": lambda item, status: codex_formatter.format_file_change(item),
    "reasoning": lambda item, status: codex_formatter.format_reasoning(item),
    "mcp_tool_call": lambda item, status: codex_formatter.format_mcp_tool_call(item),
    "web_search": lambda item, status: codex_formatter.format_web_search(item),
    "todo_list": lambda item, status: codex_formatter.format_todo_list(item),
}


def _handle_item_event(msg: dict, event_type: str) -> None:
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
        handler(item, status)


# Dispatch table for Codex event types
CODEX_EVENT_HANDLERS: dict[str, Any] = {
    "thread.started": _handle_thread_started,
    "turn.started": _handle_turn_started,
    "turn.completed": _handle_turn_completed,
    "turn.failed": _handle_turn_failed,
    "error": _handle_error,
}


def process_codex_event(msg: dict) -> None:
    """Process a Codex JSONL event."""
    event_type = msg.get("type", "")

    # Check direct event handlers first
    handler = CODEX_EVENT_HANDLERS.get(event_type)
    if handler:
        handler(msg)
        return

    # Handle item events
    if event_type in ("item.completed", "item.started", "item.updated"):
        _handle_item_event(msg, event_type)


def is_codex_event(msg):
    """Detect if this is a Codex event vs Claude message"""
    event_type = msg.get("type", "")
    # Codex uses dot-notation event types
    if "." in event_type:
        return True
    # Codex-specific top-level types
    if event_type in ("error",) and "item" not in msg and "message" not in msg:
        return True
    return False


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
    stats = msg.get("stats", {})
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


def _extract_result_text(content_data: Any) -> str:
    """Extract text from tool result content."""
    if isinstance(content_data, list):
        text_parts = [
            item.get("text", "")
            for item in content_data
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(text_parts)
    return str(content_data)


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
            f"  {DIM}• tool_result (orphan, id={tool_id[:8] if tool_id else '?'}...){RESET}",
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


def main():
    """Main entry point"""
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                # Dispatch to appropriate handler based on format
                if is_codex_event(msg):
                    process_codex_event(msg)
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

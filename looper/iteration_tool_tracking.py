# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Tool call tracking for the looper iteration runner.

Extracted from iteration.py per designs/2026-02-01-iteration-split.md Phase 2.

Responsibilities:
- Process tool call events from AI subprocess JSON output
- Map Claude and Codex event formats to unified tool call tracking
- Provide tool call statistics for telemetry

Part of #1804.
"""

from pathlib import Path
from typing import Any, Literal

from looper.checkpoint import ToolCallLog, get_tool_call_log_filename

__all__ = ["ToolCallTracker"]


class ToolCallTracker:
    """Track tool calls from AI subprocess output for checkpointing and metrics.

    Handles both Claude (tool_use in assistant message, tool_result in user message)
    and Codex (item.started/item.completed) event formats.

    See: designs/2026-02-01-tool-call-checkpointing.md
    See: designs/2026-02-01-tool-call-metrics-format.md
    """

    def __init__(self, mode: str, worker_id: int | None = None) -> None:
        """Initialize tool call tracker.

        REQUIRES: mode is a role string (worker/prover/researcher/manager)
        REQUIRES: worker_id is None or a non-negative integer
        ENSURES: _tool_call_log is None until initialize() is called

        Args:
            mode: Role name (worker, prover, researcher, manager)
            worker_id: Optional worker instance ID for multi-worker mode
        """
        self.mode = mode
        self.worker_id = worker_id
        self._tool_call_log: ToolCallLog | None = None

    def initialize(self, source: Literal["claude", "codex", "dasher"]) -> None:
        """Initialize the tool call log for a new iteration.

        REQUIRES: source in ("claude", "codex", "dasher")
        ENSURES: _tool_call_log is initialized for this iteration

        Args:
            source: AI tool being used (determines event format)
        """
        tool_call_log_file = Path(get_tool_call_log_filename(self.mode, self.worker_id))
        self._tool_call_log = ToolCallLog(tool_call_log_file, source)

    def clear(self) -> None:
        """Clear tool call log for a new main iteration.

        Should only be called on main iteration (audit_round == 0) to accumulate
        stats across audit rounds for telemetry (#1630).

        ENSURES: clears the current log if initialized, otherwise no-op
        """
        if self._tool_call_log is not None:
            self._tool_call_log.clear()

    def get_log_info(self) -> tuple[str | None, int, int]:
        """Get tool call log info for checkpoint updates.

        ENSURES: Returns (log_path, finished_count, last_seq)
        ENSURES: All values are valid even if no log exists

        Returns:
            Tuple of (log_path, finished_count, last_seq).
            All values are valid even if no log exists.
        """
        if self._tool_call_log is None:
            return None, 0, 0
        return (
            str(self._tool_call_log.log_file),
            self._tool_call_log.get_finished_count(),
            self._tool_call_log.get_last_seq(),
        )

    def get_stats(self) -> tuple[int, dict[str, int], dict[str, int]]:
        """Get aggregated tool call statistics for telemetry.

        Returns:
            Tuple of (total_count, type_counts, duration_ms_per_type).
            - total_count: Total number of tool calls
            - type_counts: Count per tool type (e.g., {"Bash": 10, "Read": 15})
            - duration_ms_per_type: Duration in milliseconds per tool type

        Returns (0, {}, {}) if no tool call log exists.

        ENSURES: Returns aggregates from the current log
        ENSURES: Returns (0, {}, {}) if no tool call log exists
        """
        if self._tool_call_log is None:
            return 0, {}, {}
        return self._tool_call_log.get_aggregate_stats()

    def process_event(self, msg: dict[str, Any]) -> None:
        """Process tool call events from JSON messages for checkpointing.

        Handles both Claude (tool_use in assistant message, tool_result in user message)
        and Codex (item.started/item.completed) event formats.

        REQUIRES: msg is a parsed JSON message dict
        ENSURES: records tool events when log is initialized
        """
        if self._tool_call_log is None:
            return

        msg_type = msg.get("type")

        # Claude: tool_use in assistant message
        if msg_type == "assistant":
            self._process_claude_assistant(msg)
        # Claude: tool_result in user message
        elif msg_type == "user":
            self._process_claude_user(msg)
        # Codex: item.started
        elif msg_type == "item.started":
            self._process_codex_started(msg)
        # Codex: item.completed
        elif msg_type == "item.completed":
            self._process_codex_completed(msg)

    def _process_claude_assistant(self, msg: dict[str, Any]) -> None:
        """Process Claude assistant message with tool_use blocks.

        REQUIRES: _tool_call_log is initialized
        ENSURES: starts a tool call for each tool_use block
        """
        content = msg.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_id = block.get("id")
                tool_input = block.get("input", {})
                assert self._tool_call_log is not None
                self._tool_call_log.start(tool_name, tool_input, tool_id)

    def _process_claude_user(self, msg: dict[str, Any]) -> None:
        """Process Claude user message with tool_result blocks.

        REQUIRES: _tool_call_log is initialized
        ENSURES: completes or fails tool calls for each tool_result block
        """
        content = msg.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                result = block.get("content", "")
                if isinstance(result, list):
                    # Handle structured content
                    result = str(result)
                is_error = block.get("is_error", False)
                assert self._tool_call_log is not None
                if is_error:
                    self._tool_call_log.fail(tool_id=tool_id, error_message=result)
                else:
                    self._tool_call_log.complete(tool_id=tool_id, result=result)

    def _process_codex_started(self, msg: dict[str, Any]) -> None:
        """Process Codex item.started event.

        REQUIRES: _tool_call_log is initialized
        ENSURES: starts tool calls for supported item types
        """
        item = msg.get("item", {})
        item_type = item.get("type", "")
        tool_id = item.get("id")
        assert self._tool_call_log is not None

        if item_type == "command_execution":
            tool_name = "Bash"
            tool_input = item.get("command", "")
            self._tool_call_log.start(tool_name, tool_input, tool_id)
        elif item_type == "file_read":
            tool_name = "Read"
            tool_input = item.get("path", "")
            self._tool_call_log.start(tool_name, tool_input, tool_id)
        elif item_type == "file_write":
            tool_name = "Write"
            tool_input = item.get("path", "")
            self._tool_call_log.start(tool_name, tool_input, tool_id)
        elif item_type == "file_change":
            # file_change can be create/modify - map to Write/Edit
            change_type = item.get("change_type", "modify")
            tool_name = "Write" if change_type == "create" else "Edit"
            tool_input = item.get("file_path", item.get("path", ""))
            self._tool_call_log.start(tool_name, tool_input, tool_id)
        elif item_type == "mcp_tool_call":
            tool_name = f"MCP:{item.get('tool_name', item.get('name', 'unknown'))}"
            tool_input = item.get("arguments", {})
            self._tool_call_log.start(tool_name, tool_input, tool_id)
        elif item_type == "web_search":
            tool_name = "WebSearch"
            tool_input = item.get("query", "")
            self._tool_call_log.start(tool_name, tool_input, tool_id)
        # Note: todo_list is not a tool call, it's internal state update

    def _process_codex_completed(self, msg: dict[str, Any]) -> None:
        """Process Codex item.completed event.

        REQUIRES: _tool_call_log is initialized
        ENSURES: completes or fails tool calls for supported item types
        """
        item = msg.get("item", {})
        item_type = item.get("type", "")
        tool_id = item.get("id")

        # Handle all item types that we started tracking
        if item_type in (
            "command_execution",
            "file_read",
            "file_write",
            "file_change",
            "mcp_tool_call",
            "web_search",
        ):
            output = item.get("output", "")
            status = item.get("status", "completed")
            assert self._tool_call_log is not None
            if status == "failed":
                self._tool_call_log.fail(tool_id=tool_id, error_message=output)
            else:
                self._tool_call_log.complete(tool_id=tool_id, result=output)

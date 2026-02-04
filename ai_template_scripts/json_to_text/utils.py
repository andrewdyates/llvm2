#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Utility functions and constants for json_to_text formatting.

This module contains:
- Color constants for terminal output
- Timestamp formatting
- Text cleaning and coercion utilities
- Error detection patterns
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

__all__ = [
    # Color detection
    "USE_COLORS",
    # Color constants
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
    "clean_output",
    "format_tool_output",
    # Internal but exposed for testing
    "_decode_bytes",
    "_coerce_output_text",
    "_is_error_result",
    "_truncate",
    # Error patterns
    "ERROR_PATTERNS",
]

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


def timestamp() -> str:
    """Get current timestamp for display"""
    return datetime.now().strftime("%H:%M:%S")


def _decode_bytes(value: bytes | bytearray | memoryview) -> str:
    """Decode byte-like values into display-safe text."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    elif isinstance(value, bytearray):
        value = bytes(value)
    return value.decode("utf-8", errors="replace")


def _coerce_output_text(value: Any) -> str:
    """Coerce non-string tool output into display text."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _decode_bytes(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(
                value,
                sort_keys=True,
                default=lambda obj: _decode_bytes(obj)
                if isinstance(obj, (bytes, bytearray, memoryview))
                else str(obj),
            )
        except (TypeError, ValueError, RecursionError):
            return str(value)
    return str(value)


def clean_output(text: Any) -> str:
    """Remove system noise from output."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = _coerce_output_text(text)
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


def format_tool_output(
    content: Any, tool_name: str, is_error: bool = False
) -> list[str] | None:
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


def _is_error_result(tool_result: Any) -> bool:
    """Check if tool result indicates an error.

    Contracts:
        REQUIRES: tool_result is any type (None, str, bytes, dict, list, etc.)
        ENSURES: Returns bool indicating error detection
        ENSURES: None returns False
        ENSURES: Empty string returns False
        ENSURES: Non-string types coerced via _coerce_output_text
        ENSURES: Circular references handled (returns False, no crash)
    """
    if tool_result is None:
        return False
    if not isinstance(tool_result, str):
        tool_result = _coerce_output_text(tool_result)
    if not tool_result:
        return False
    result_lower = tool_result.lower()
    return any(pattern in result_lower for pattern in ERROR_PATTERNS)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if max_len <= 0:
        return ""
    if len(text) > max_len:
        if max_len <= 3:
            return "..."[:max_len]
        return text[: max_len - 3] + "..."
    return text

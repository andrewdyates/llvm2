#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Guardrail to detect headless role violations.

Headless roles (WORKER, PROVER, RESEARCHER, MANAGER) must not ask users for
direction. This script analyzes AI output logs to detect such violations.

Exit codes:
    0: No violation detected
    1: Violation detected (AI asked for user direction)
    2: Error reading log file

Usage:
    python3 check_headless_violation.py <log_file>

Or as a library:
    from check_headless_violation import check_log_for_violations
    violations = check_log_for_violations(Path("worker_logs/..."))
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Patterns that indicate the AI is asking for user direction
# These are phrases that autonomous roles should never use
VIOLATION_PATTERNS = [
    # Direct questions asking what to do
    r"what would you like me to do",
    r"what should I (?:do|focus on|work on|prioritize)",
    r"how would you like me to (?:proceed|continue|handle)",
    r"how should I (?:proceed|continue|handle)",
    r"where should I (?:start|begin|focus)",
    r"which (?:one|option|approach) (?:would you|should I|do you)",
    # Offering choices instead of deciding
    r"would you like me to",
    r"should I (?:start|begin|continue|proceed|focus)",
    r"do you want me to",
    r"shall I (?:start|begin|continue|proceed)",
    r"let me know (?:what|how|which|if)",
    r"please (?:let me know|tell me|specify)",
    # Requesting user input
    r"if you (?:want|prefer|would like)",
    r"(?:just )?point me (?:at|to) (?:the |a )?task",
    r"waiting for (?:your |)(?:input|direction|instruction)",
    r"awaiting (?:your |)(?:input|direction|instruction)",
    # Meta questions about the session
    r"what.{1,30}in ~?/",  # "What would you like me to do in ~/repo?"
]

# Compiled regex patterns for efficiency
_COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in VIOLATION_PATTERNS
]


def extract_text_from_jsonl(log_file: Path) -> list[str]:
    """Extract text content from JSONL log file.

    Args:
        log_file: Path to JSONL log file

    Returns:
        List of text content strings from assistant messages
    """
    texts = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Handle different message formats
                    if isinstance(data, dict):
                        # Claude streaming format: {"type": "content_block_delta", "delta": {"text": "..."}}
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            text = delta.get("text")
                            if isinstance(text, str):
                                texts.append(text)
                        # Message format: {"type": "message", "content": [...]}
                        elif data.get("type") == "message":
                            content = data.get("content", [])
                            if isinstance(content, str):
                                texts.append(content)
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text = block.get("text")
                                        if isinstance(text, str):
                                            texts.append(text)
                        # Claude nested format: {"type": "assistant", "message": {"content": [...]}}
                        elif data.get("type") == "assistant":
                            message = data.get("message", {})
                            content = message.get("content", [])
                            if isinstance(content, str):
                                texts.append(content)
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text = block.get("text")
                                        if isinstance(text, str):
                                            texts.append(text)
                        # Codex item.completed format: {"type": "item.completed", "item": {"text": "..."}}
                        # Skip non-user-facing item types (#2314, #2327):
                        # - reasoning: Internal AI reasoning
                        # - file_change: File diffs (may contain violation patterns in comments)
                        # - command_execution: Shell output (may echo violation phrases)
                        # Only agent_message items are user-facing and should be checked.
                        elif data.get("type") == "item.completed":
                            item = data.get("item", {})
                            item_type = item.get("type")
                            if item_type in ("reasoning", "file_change", "command_execution"):
                                continue
                            text = item.get("text")
                            if isinstance(text, str):
                                texts.append(text)
                        # Codex format: {"content": "...", "role": "assistant"}
                        elif data.get("role") == "assistant":
                            content = data.get("content", "")
                            if isinstance(content, str):
                                texts.append(content)
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text = block.get("text")
                                        if isinstance(text, str):
                                            texts.append(text)
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        pass
    return texts


def check_text_for_violations(text: str) -> list[tuple[str, str]]:
    """Check text for violation patterns.

    Args:
        text: Text to check

    Returns:
        List of (pattern, matched_text) tuples for violations found
    """
    violations = []
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append((pattern.pattern, match.group()))
    return violations


def check_log_for_violations(log_file: Path) -> list[tuple[str, str]]:
    """Check a log file for headless violations.

    Args:
        log_file: Path to JSONL log file

    Returns:
        List of (pattern, matched_text) tuples for violations found
    """
    texts = extract_text_from_jsonl(log_file)
    all_violations = []
    for text in texts:
        violations = check_text_for_violations(text)
        all_violations.extend(violations)
    return all_violations


def main() -> int:
    """Main entry point.

    Returns:
        0 if no violations, 1 if violations found, 2 on error
    """
    if len(sys.argv) < 2:
        print("Usage: check_headless_violation.py <log_file>", file=sys.stderr)
        return 2

    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        return 2

    violations = check_log_for_violations(log_file)
    if violations:
        print(f"HEADLESS VIOLATION DETECTED in {log_file.name}:")
        for pattern, matched in violations[:3]:  # Limit output
            print(f"  - Pattern: {pattern}")
            truncated = matched[:80] + "..." if len(matched) > 80 else matched
            print(f"    Matched: {truncated}")
        if len(violations) > 3:
            print(f"  ... and {len(violations) - 3} more violations")
        return 1

    return 0


__all__ = [
    "VIOLATION_PATTERNS",
    "extract_text_from_jsonl",
    "check_text_for_violations",
    "check_log_for_violations",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())

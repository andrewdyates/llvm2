#!/usr/bin/env bash
# record_file_read.sh - PostToolUse hook for Read
# Records timestamp when files are read
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>

set -euo pipefail

STATE_FILE="${CLAUDE_PROJECT_DIR:-.}/.claude_read_timestamps"

# Read JSON input from stdin
INPUT=$(cat)

# Extract file_path from tool_input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -n "$FILE_PATH" ]; then
    # Record timestamp (Unix epoch)
    TIMESTAMP=$(date +%s)

    # Remove old entry for this file, add new one
    if [ -f "$STATE_FILE" ]; then
        grep -v "^${FILE_PATH}:" "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null || true
        mv "${STATE_FILE}.tmp" "$STATE_FILE"
    fi

    echo "${FILE_PATH}:${TIMESTAMP}" >> "$STATE_FILE"
fi

exit 0

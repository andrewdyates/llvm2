#!/usr/bin/env bash
# check_file_freshness.sh - PreToolUse hook for Edit
# Blocks edits to files modified since last read
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>

set -euo pipefail

STATE_FILE="${CLAUDE_PROJECT_DIR:-.}/.claude_read_timestamps"

# Read JSON input from stdin
INPUT=$(cat)

# Extract file_path from tool_input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0  # No file path, allow
fi

if [ ! -f "$FILE_PATH" ]; then
    exit 0  # File doesn't exist yet (new file), allow
fi

if [ ! -f "$STATE_FILE" ]; then
    # No read timestamps recorded - file was never read
    echo "BLOCK: You haven't read $FILE_PATH in this session. Read it first." >&2
    exit 2
fi

# Get last read timestamp for this file
LAST_READ=$(grep "^${FILE_PATH}:" "$STATE_FILE" 2>/dev/null | tail -1 | cut -d: -f2)

if [ -z "$LAST_READ" ]; then
    echo "BLOCK: You haven't read $FILE_PATH in this session. Read it first." >&2
    exit 2
fi

# Get file modification time (Unix epoch)
if [[ "$OSTYPE" == "darwin"* ]]; then
    FILE_MTIME=$(stat -f %m "$FILE_PATH")
else
    FILE_MTIME=$(stat -c %Y "$FILE_PATH")
fi

# Compare timestamps
if [ "$FILE_MTIME" -gt "$LAST_READ" ]; then
    LAST_READ_HUMAN=$(date -r "$LAST_READ" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "@$LAST_READ" "+%Y-%m-%d %H:%M:%S")
    FILE_MTIME_HUMAN=$(date -r "$FILE_MTIME" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "@$FILE_MTIME" "+%Y-%m-%d %H:%M:%S")

    echo "BLOCK: $FILE_PATH was modified at $FILE_MTIME_HUMAN but you last read it at $LAST_READ_HUMAN." >&2
    echo "Re-read the file before editing." >&2
    exit 2
fi

exit 0

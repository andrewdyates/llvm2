#!/bin/bash
# spawn_session.sh - Spawn worker or manager in new iTerm2 tab
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# Note: Intentionally simple. DashTerm2 will replace this with a proper
# terminal manager supporting multiple backends, session tracking, etc.
#
# Usage:
#   ./ai_template_scripts/spawn_session.sh worker        # Spawn worker in current project
#   ./ai_template_scripts/spawn_session.sh manager       # Spawn manager in current project
#   ./ai_template_scripts/spawn_session.sh worker ~/z4   # Spawn worker in specific project
#   ./ai_template_scripts/spawn_session.sh --help        # Show help

set -e

usage() {
    cat <<'EOF'
Usage: spawn_session.sh [OPTIONS] [MODE] [PROJECT_PATH]

Spawn a worker, prover, researcher, or manager session in a new iTerm2 tab.

Arguments:
  MODE          worker (default), prover, researcher, or manager
  PROJECT_PATH  Path to project directory (default: current directory)

Options:
  -h, --help    Show this help message
  --dry-run     Show what would be executed without doing it

Examples:
  spawn_session.sh                     # Worker in current directory
  spawn_session.sh worker              # Worker in current directory
  spawn_session.sh prover              # Prover in current directory
  spawn_session.sh researcher          # Researcher in current directory
  spawn_session.sh manager             # Manager in current directory
  spawn_session.sh worker ~/z4         # Worker in ~/z4
  spawn_session.sh ~/z4                # Worker in ~/z4 (auto-detected)
  spawn_session.sh ~/z4 worker         # Worker in ~/z4 (swapped args ok)

See also:
  spawn_all.sh                           # Spawn all 4 loops at once
EOF
}

err() {
    echo "Error: $1" >&2
    [[ -n "$2" ]] && echo "       $2" >&2
    exit 1
}

# Parse options
DRY_RUN=false
while [[ "$1" == -* ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) err "Unknown option: $1" "Use --help for usage" ;;
    esac
done

# Parse positional arguments with smart detection
ARG1="${1:-}"
ARG2="${2:-}"

# Check for extra arguments
if [[ -n "${3:-}" ]]; then
    err "Too many arguments" "Use --help for usage"
fi

# Detect if arguments are swapped (path given before mode)
VALID_MODES="worker|prover|researcher|manager"
if [[ -d "$ARG1" && ("$ARG2" =~ ^($VALID_MODES)$ || -z "$ARG2") ]]; then
    # First arg is a path
    PROJECT="$ARG1"
    MODE="${ARG2:-worker}"
elif [[ "$ARG1" =~ ^($VALID_MODES)$ ]]; then
    # Normal order: mode then path
    MODE="$ARG1"
    PROJECT="${ARG2:-$(pwd)}"
elif [[ -z "$ARG1" ]]; then
    # No arguments
    MODE="worker"
    PROJECT="$(pwd)"
else
    # First arg is neither mode nor existing directory
    err "Invalid argument: $ARG1" "Must be 'worker', 'prover', 'researcher', 'manager', or a valid path"
fi

# Validate mode
if [[ ! "$MODE" =~ ^($VALID_MODES)$ ]]; then
    err "Invalid mode: $MODE" "Must be 'worker', 'prover', 'researcher', or 'manager'"
fi

# Resolve to absolute path (save original for error message)
PROJECT_ORIG="$PROJECT"
PROJECT=$(cd "$PROJECT" 2>/dev/null && pwd) || {
    err "Project path does not exist: $PROJECT_ORIG"
}

# Check looper.py exists
if [[ ! -f "$PROJECT/looper.py" ]]; then
    err "$PROJECT/looper.py not found" "Make sure you're pointing to a project with looper.py"
fi

# Check looper.py is executable
if [[ ! -x "$PROJECT/looper.py" ]]; then
    err "$PROJECT/looper.py is not executable" "Run: chmod +x $PROJECT/looper.py"
fi

# Get project name and role for tab title
PROJECT_NAME=$(basename "$PROJECT")
# Sanitize project name for osascript (remove quotes and backslashes)
PROJECT_NAME_SAFE="${PROJECT_NAME//[\"\'\\\$]/}"
case "$MODE" in
    worker)     ROLE="W" ;;
    prover)     ROLE="P" ;;
    researcher) ROLE="R" ;;
    manager)    ROLE="M" ;;
esac
TAB_TITLE="[$ROLE]$PROJECT_NAME_SAFE"

# Build the command to run (escape inner quotes for AppleScript)
CMD="cd '$PROJECT' && ./looper.py $MODE"

if $DRY_RUN; then
    echo "Would create iTerm2 tab: $TAB_TITLE"
    echo "Would run: $CMD"
    exit 0
fi

# Check iTerm2 is running (only when actually creating a tab)
# Note: pgrep -x doesn't work for GUI apps on macOS, use -f with app bundle path
if ! pgrep -f "iTerm.app" > /dev/null 2>&1; then
    err "iTerm2 is not running" "Start iTerm2 first, then run this script"
fi

# Create new tab and run command
# Use "front window" to target the user's active window, not iTerm2's internal "current"
if ! osascript -e "
tell application \"iTerm2\"
    tell front window
        create tab with default profile
        tell current session
            set name to \"$TAB_TITLE\"
            write text \"$CMD\"
        end tell
    end tell
end tell
" 2>&1; then
    err "Failed to create iTerm2 tab" "Make sure iTerm2 has an open window"
fi

echo "Spawned $MODE session: $TAB_TITLE"

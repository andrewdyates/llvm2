#!/bin/bash
# spawn_all.sh - Spawn all 4 AI loops (worker, prover, researcher, manager)
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# Usage:
#   ./ai_template_scripts/spawn_all.sh           # All 4 roles in current project
#   ./ai_template_scripts/spawn_all.sh ~/z4      # All 4 roles in specific project
#   ./ai_template_scripts/spawn_all.sh --help    # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    cat <<'EOF'
Usage: spawn_all.sh [OPTIONS] [PROJECT_PATH]

Spawn all 4 AI loops (worker, prover, researcher, manager) in new iTerm2 tabs.

Arguments:
  PROJECT_PATH  Path to project directory (default: current directory)

Options:
  -h, --help    Show this help message
  --dry-run     Show what would be executed without doing it
  --roles       Comma-separated roles to spawn (default: worker,prover,researcher,manager)

Examples:
  spawn_all.sh                              # All 4 roles in current directory
  spawn_all.sh ~/z4                         # All 4 roles in ~/z4
  spawn_all.sh --roles worker,manager       # Just worker and manager
  spawn_all.sh --dry-run                    # Preview what would run
EOF
}

# Defaults
DRY_RUN=false
ROLES="worker,prover,researcher,manager"
PROJECT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --roles)
            ROLES="$2"
            shift 2
            ;;
        --roles=*)
            ROLES="${1#*=}"
            shift
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage" >&2
            exit 1
            ;;
        *)
            if [[ -z "$PROJECT" ]]; then
                PROJECT="$1"
            else
                echo "Error: Too many arguments" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Default to current directory
PROJECT="${PROJECT:-$(pwd)}"

# Check for existing STOP files (STOP and STOP_*)
STOP_FILES=()
if [[ -f "$PROJECT/STOP" ]]; then
    STOP_FILES+=("STOP")
fi
for stop_file in "$PROJECT"/STOP_*; do
    [[ -f "$stop_file" ]] && STOP_FILES+=("$(basename "$stop_file")")
done

if [[ ${#STOP_FILES[@]} -gt 0 ]]; then
    echo "Warning: Removing stale STOP files from previous session: ${STOP_FILES[*]}"
    if $DRY_RUN; then
        echo "[dry-run] Would remove: ${STOP_FILES[*]}"
    else
        for f in "${STOP_FILES[@]}"; do
            rm "$PROJECT/$f"
        done
    fi
fi

# Upgrade AI tools before spawning
echo "Checking for AI tool updates..."
if command -v claude &>/dev/null; then
    echo "  Upgrading claude (stable channel)..."
    claude install --channel stable 2>/dev/null || echo "  claude update failed or up to date"
fi
if command -v brew &>/dev/null && brew list codex &>/dev/null 2>&1; then
    echo "  Upgrading codex..."
    brew upgrade codex 2>/dev/null || echo "  codex already up to date"
fi
echo

# Convert roles to array
IFS=',' read -ra ROLE_ARRAY <<< "$ROLES"

# Validate roles
for role in "${ROLE_ARRAY[@]}"; do
    case "$role" in
        worker|prover|researcher|manager) ;;
        *)
            echo "Error: Invalid role: $role" >&2
            echo "Valid roles: worker, prover, researcher, manager" >&2
            exit 1
            ;;
    esac
done

echo "Spawning ${#ROLE_ARRAY[@]} loops in: $PROJECT"
echo "Roles: ${ROLE_ARRAY[*]}"
echo

# Spawn each role
for role in "${ROLE_ARRAY[@]}"; do
    if $DRY_RUN; then
        echo "[dry-run] $SCRIPT_DIR/spawn_session.sh $role $PROJECT"
    else
        "$SCRIPT_DIR/spawn_session.sh" "$role" "$PROJECT"
        sleep 0.5  # Small delay between spawns to avoid race conditions
    fi
done

echo
echo "All loops spawned. Use 'touch STOP' (all) or 'touch STOP_WORKER' (per-role) to stop."

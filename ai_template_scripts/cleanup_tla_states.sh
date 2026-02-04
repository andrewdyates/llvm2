#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# cleanup_tla_states.sh - Remove old TLA+ model checking state files
#
# TLA+ model checking creates large state files (*.fp up to 1GB each) in states/
# directories. These are ephemeral and not needed after model checking completes.
# This script removes states/ directories older than MAX_AGE_DAYS to prevent
# disk bloat recurrence (#1551, #668).
#
# Usage:
#   cleanup_tla_states.sh              # Dry run - show what would be deleted
#   cleanup_tla_states.sh --delete     # Actually delete old states
#   cleanup_tla_states.sh --help       # Show help
#   cleanup_tla_states.sh --version    # Show version

set -euo pipefail

# Configuration
MAX_AGE_DAYS="${TLA_STATES_MAX_AGE_DAYS:-7}"  # Days before states are considered stale

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'  # No Color

# Version function
version() {
    local git_hash
    git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "cleanup_tla_states.sh ${git_hash} ($(date +%Y-%m-%d))"
    exit 0
}

# Usage function
usage() {
    local exit_code="${1:-0}"
    local output_fd=1
    if [[ "$exit_code" -ne 0 ]]; then
        output_fd=2
    fi
    cat >&"$output_fd" <<'EOF'
Usage: cleanup_tla_states.sh [OPTIONS] [PATH]

Remove old TLA+ model checking state files to prevent disk bloat.

Arguments:
  PATH          Directory to scan (default: current directory)

Options:
  --delete      Actually delete old states (default: dry run)
  --max-age N   Days before states are considered stale (default: 7)
  --version     Show version information
  -h, --help    Show this help message

Environment:
  TLA_STATES_MAX_AGE_DAYS    Override default max age (days)

Examples:
  cleanup_tla_states.sh                     # Dry run in current dir
  cleanup_tla_states.sh --delete            # Delete old states
  cleanup_tla_states.sh --max-age 3 ~/proj  # Delete states > 3 days in ~/proj
EOF
    exit "$exit_code"
}

# Parse arguments
DELETE=false
SCAN_PATH="."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete)
            DELETE=true
            shift
            ;;
        --max-age)
            if [[ -z "${2:-}" ]] || [[ "$2" =~ ^- ]]; then
                echo -e "${RED}Error: --max-age requires a numeric argument${NC}" >&2
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: --max-age must be a positive integer (got: $2)${NC}" >&2
                exit 1
            fi
            MAX_AGE_DAYS="$2"
            shift 2
            ;;
        --version)
            version
            ;;
        -h|--help)
            usage 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            usage 1
            ;;
        *)
            SCAN_PATH="$1"
            shift
            ;;
    esac
done

# Validate scan path
if [[ ! -d "$SCAN_PATH" ]]; then
    echo -e "${RED}Error: Directory not found: $SCAN_PATH${NC}" >&2
    exit 1
fi

# Find states/ directories
echo "Scanning $SCAN_PATH for states/ directories..." >&2

# Use find to locate states directories with old content
# -mtime +N means modified more than N days ago
TOTAL_SIZE=0
TOTAL_COUNT=0
DELETED_SIZE=0
DELETED_COUNT=0

while IFS= read -r -d '' states_dir; do
    # Skip if empty
    [[ -z "$states_dir" ]] && continue

    # Get size of directory
    DIR_SIZE_KB=$(du -sk "$states_dir" 2>/dev/null | cut -f1 || echo "0")
    DIR_SIZE_MB=$((DIR_SIZE_KB / 1024))

    # Check modification time of the directory itself
    # Try macOS stat first, then Linux stat
    DIR_MTIME=$(stat -f %m "$states_dir" 2>/dev/null || stat -c %Y "$states_dir" 2>/dev/null || echo "")
    if [[ -z "$DIR_MTIME" ]]; then
        echo -e "  ${YELLOW}Skipping:${NC} $states_dir (could not determine age)"
        continue
    fi
    DIR_AGE_DAYS=$(( ($(date +%s) - DIR_MTIME) / 86400 ))

    TOTAL_SIZE=$((TOTAL_SIZE + DIR_SIZE_MB))
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    if [[ $DIR_AGE_DAYS -gt $MAX_AGE_DAYS ]]; then
        if [[ "$DELETE" == "true" ]]; then
            echo -e "${GREEN}Deleting:${NC} $states_dir (${DIR_SIZE_MB}MB, ${DIR_AGE_DAYS} days old)"
            rm -rf "$states_dir"
            DELETED_SIZE=$((DELETED_SIZE + DIR_SIZE_MB))
            DELETED_COUNT=$((DELETED_COUNT + 1))
        else
            echo -e "${YELLOW}Would delete:${NC} $states_dir (${DIR_SIZE_MB}MB, ${DIR_AGE_DAYS} days old)"
            DELETED_SIZE=$((DELETED_SIZE + DIR_SIZE_MB))
            DELETED_COUNT=$((DELETED_COUNT + 1))
        fi
    else
        echo -e "  Keeping: $states_dir (${DIR_SIZE_MB}MB, ${DIR_AGE_DAYS} days old)"
    fi
done < <(find "$SCAN_PATH" -type d -name "states" -print0 2>/dev/null)

# Also check for *.fp files outside states/ directories (edge cases)
FP_FILES=$(find "$SCAN_PATH" -name "*.fp" -type f -mtime "+${MAX_AGE_DAYS}" 2>/dev/null | wc -l)
FP_FILES="${FP_FILES//[[:space:]]/}"
if [[ $FP_FILES -gt 0 ]]; then
    echo "" >&2
    echo -e "${YELLOW}Warning:${NC} Found $FP_FILES orphaned .fp files older than ${MAX_AGE_DAYS} days" >&2
fi

# Summary
echo "" >&2
echo "Summary:" >&2
echo "  Total states/ directories: $TOTAL_COUNT (${TOTAL_SIZE}MB)" >&2
if [[ "$DELETE" == "true" ]]; then
    echo -e "  ${GREEN}Deleted:${NC} $DELETED_COUNT directories (${DELETED_SIZE}MB)" >&2
else
    echo -e "  ${YELLOW}Would delete:${NC} $DELETED_COUNT directories (${DELETED_SIZE}MB)" >&2
    if [[ $DELETED_COUNT -gt 0 ]]; then
        echo "" >&2
        echo "Run with --delete to remove old states." >&2
    fi
fi

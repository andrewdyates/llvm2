#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# sync_all.sh - Batch sync ai_template to all sibling repos
#
# CANONICAL SOURCE: ayates_dbx/ai_template
# DO NOT EDIT in other repos - file issues to ai_template for changes.
#
# Usage:
#   ./ai_template_scripts/sync_all.sh              # Sync all sibling repos
#   ./ai_template_scripts/sync_all.sh --dry-run    # Show what would be synced (with per-repo details)
#   ./ai_template_scripts/sync_all.sh --summary    # Quick status summary only (no per-repo details)
#   ./ai_template_scripts/sync_all.sh --parallel 4 # Parallel sync (default: serial)
#   ./ai_template_scripts/sync_all.sh --behind     # Only sync repos behind current
#   ./ai_template_scripts/sync_all.sh --timeout 60 # Per-repo timeout in seconds
#   ./ai_template_scripts/sync_all.sh --max-repos 10  # Limit repos processed
#   ./ai_template_scripts/sync_all.sh repo1 repo2  # Sync specific repos

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Version function
version() {
    local git_hash
    git_hash=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "sync_all.sh ${git_hash} (${date})"
    exit 0
}

# Cache for archived repos (avoid repeated API calls)
ARCHIVED_CACHE_DIR="${HOME}/.cache/ai_template"
ARCHIVED_CACHE_FILE="${ARCHIVED_CACHE_DIR}/archived_repos.txt"
ARCHIVED_CACHE_AGE=3600  # 1 hour

# Get file mtime portably (Linux uses -c %Y, macOS uses -f %m)
get_file_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

# Check if repo is archived (with REST fallback and caching)
is_repo_archived() {
    local repo_name="$1"
    if [[ -f "$ARCHIVED_CACHE_FILE" ]]; then
        local cache_age=$(($(date +%s) - $(get_file_mtime "$ARCHIVED_CACHE_FILE")))
        if [[ $cache_age -lt $ARCHIVED_CACHE_AGE ]]; then
            if grep -qx "$repo_name" "$ARCHIVED_CACHE_FILE" 2>/dev/null; then
                return 0
            fi
        fi
    fi
    local result stderr_file
    stderr_file=$(mktemp)
    # NOTE: Do NOT use gh -q or --jq - has caching bugs in v2.83.2+ (#1047)
    if result=$(gh repo view "ayates_dbx/$repo_name" --json isArchived 2>"$stderr_file" | jq -r '.isArchived'); then
        rm -f "$stderr_file"
        if [[ "$result" == "true" ]]; then
            mkdir -p "$ARCHIVED_CACHE_DIR"
            # Avoid duplicates in cache file
            grep -qx "$repo_name" "$ARCHIVED_CACHE_FILE" 2>/dev/null || echo "$repo_name" >> "$ARCHIVED_CACHE_FILE"
            return 0
        fi
        return 1
    fi
    if grep -qiE 'rate.?limit' "$stderr_file" 2>/dev/null; then
        rm -f "$stderr_file"
        if result=$(gh api "repos/ayates_dbx/$repo_name" 2>/dev/null | jq -r '.archived'); then
            if [[ "$result" == "true" ]]; then
                mkdir -p "$ARCHIVED_CACHE_DIR"
                # Avoid duplicates in cache file
                grep -qx "$repo_name" "$ARCHIVED_CACHE_FILE" 2>/dev/null || echo "$repo_name" >> "$ARCHIVED_CACHE_FILE"
                return 0
            fi
            return 1
        fi
    fi
    rm -f "$stderr_file"
    return 1
}

# Must run from ai_template root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$AI_TEMPLATE_ROOT"

[[ -f "CLAUDE.md" ]] || { log_error "Run from ai_template root"; exit 1; }

# Parse args
DRY_RUN=false
SUMMARY_ONLY=false
PARALLEL=1
BEHIND_ONLY=false
SPECIFIC_REPOS=()
NO_PUSH=false
TIMEOUT=0  # Per-repo timeout in seconds (0 = no timeout)
MAX_REPOS=0  # Maximum repos to process (0 = no limit)
TIMEOUT_CMD=""
TIMEOUT_ACTIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --summary)
            SUMMARY_ONLY=true
            DRY_RUN=true  # Summary implies dry-run
            shift
            ;;
        --parallel)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                log_error "--parallel requires a number argument"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                log_error "--parallel must be a positive integer, got: $2"
                exit 1
            fi
            PARALLEL="$2"
            shift 2
            ;;
        --behind)
            BEHIND_ONLY=true
            shift
            ;;
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --timeout)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                log_error "--timeout requires a number argument (seconds)"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                log_error "--timeout must be a positive integer, got: $2"
                exit 1
            fi
            TIMEOUT="$2"
            shift 2
            ;;
        --max-repos)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                log_error "--max-repos requires a number argument"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                log_error "--max-repos must be a positive integer, got: $2"
                exit 1
            fi
            MAX_REPOS="$2"
            shift 2
            ;;
        --version)
            version
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [repo1 repo2 ...]"
            echo ""
            echo "Batch sync ai_template to multiple repos."
            echo ""
            echo "Options:"
            echo "  --dry-run      Show what would be synced (with per-repo details)"
            echo "  --summary      Quick status summary only (faster than --dry-run)"
            echo "  --parallel N   Sync N repos in parallel (default: serial)"
            echo "  --behind       Only sync repos that are behind (skip untracked)"
            echo "  --no-push      Sync and commit but don't push to remote"
            echo "  --timeout N    Per-repo timeout in seconds (default: no limit)"
            echo "  --max-repos N  Limit number of repos to process (default: all)"
            echo "  --version      Show version information"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                    # Preview all sibling repos"
            echo "  $0 --behind --parallel 4        # Sync behind repos in parallel"
            echo "  $0 --timeout 60 --max-repos 10  # Process max 10 repos, 60s each"
            echo "  $0 ~/proj1 ~/proj2              # Sync specific repos"
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            SPECIFIC_REPOS+=("$1")
            shift
            ;;
    esac
done

# Resolve timeout command (if requested)
if [[ "$TIMEOUT" -gt 0 ]]; then
    if command -v timeout &>/dev/null; then
        TIMEOUT_CMD="timeout"
        TIMEOUT_ACTIVE=true
    elif command -v gtimeout &>/dev/null; then
        TIMEOUT_CMD="gtimeout"
        TIMEOUT_ACTIVE=true
    else
        log_warn "--timeout set but no timeout command found; running without time limits"
    fi
fi

# Get ai_template current version
CURRENT_VERSION=$(git rev-parse HEAD)
CURRENT_SHORT=$(git rev-parse --short HEAD)

# Collect repos to sync
REPOS=()

if [[ ${#SPECIFIC_REPOS[@]} -gt 0 ]]; then
    # User specified repos
    for repo in "${SPECIFIC_REPOS[@]}"; do
        if [[ -d "$repo/.git" ]]; then
            REPOS+=("$(cd "$repo" && pwd)")
        else
            log_warn "Skipping non-repo: $repo"
        fi
    done
else
    # Default: sibling directories that are ayates_dbx repos
    PARENT_DIR="$(dirname "$AI_TEMPLATE_ROOT")"
    for dir in "$PARENT_DIR"/*/; do
        [[ -d "$dir/.git" ]] || continue
        [[ "$(basename "$dir")" == "ai_template" ]] && continue

        # Only sync ayates_dbx repos (check git remote)
        repo_name=$(basename "$dir")
        remote_url=$(git -C "$dir" remote get-url origin 2>/dev/null || echo "")
        if [[ ! "$remote_url" =~ ayates_dbx ]]; then
            continue  # Skip non-ayates_dbx repos
        fi

        # Skip archived repos
        if is_repo_archived "$repo_name"; then
            continue
        fi

        if [[ "$BEHIND_ONLY" == "true" ]]; then
            version_file="$dir/.ai_template_version"
            if [[ ! -f "$version_file" ]]; then
                continue  # Skip untracked repos when --behind is used
            fi
            repo_version=$(head -1 "$version_file" | tr -d '[:space:]')
            if [[ "$repo_version" == "$CURRENT_VERSION" ]]; then
                continue  # Skip up-to-date repos
            fi
        fi

        REPOS+=("$dir")
    done
fi

if [[ ${#REPOS[@]} -eq 0 ]]; then
    log_ok "No repos to sync"
    exit 0
fi

# Apply --max-repos limit
TOTAL_REPOS=${#REPOS[@]}
if [[ "$MAX_REPOS" -gt 0 && ${#REPOS[@]} -gt "$MAX_REPOS" ]]; then
    REPOS=("${REPOS[@]:0:$MAX_REPOS}")
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ai_template batch sync - $CURRENT_SHORT"
echo "════════════════════════════════════════════════════════════════"
echo ""
if [[ "$MAX_REPOS" -gt 0 && "$TOTAL_REPOS" -gt "$MAX_REPOS" ]]; then
    echo "Repos to sync: ${#REPOS[@]} (limited from $TOTAL_REPOS)"
else
    echo "Repos to sync: ${#REPOS[@]}"
fi
[[ "$SUMMARY_ONLY" == "true" ]] && echo "Mode: SUMMARY ONLY"
[[ "$DRY_RUN" == "true" && "$SUMMARY_ONLY" == "false" ]] && echo "Mode: DRY RUN"
[[ "$PARALLEL" -gt 1 ]] && echo "Parallelism: $PARALLEL"
if [[ "$TIMEOUT" -gt 0 ]]; then
    if [[ "$TIMEOUT_ACTIVE" == "true" ]]; then
        echo "Per-repo timeout: ${TIMEOUT}s"
    else
        echo "Per-repo timeout: unavailable (missing timeout command)"
    fi
fi
echo ""

# Show summary of what will change
for repo in "${REPOS[@]}"; do
    repo_name=$(basename "$repo")
    version_file="$repo/.ai_template_version"

    if [[ -f "$version_file" ]]; then
        repo_version=$(head -1 "$version_file" | tr -d '[:space:]')
        if [[ "$repo_version" == "$CURRENT_VERSION" ]]; then
            printf "  ${GREEN}✓${NC} %-30s (current)\n" "$repo_name"
        else
            commits_behind=$(git rev-list --count "$repo_version..$CURRENT_VERSION" 2>/dev/null || echo "?")
            printf "  ${YELLOW}↻${NC} %-30s ($commits_behind commits behind)\n" "$repo_name"
        fi
    else
        printf "  ${BLUE}+${NC} %-30s (untracked)\n" "$repo_name"
    fi
done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Run without --dry-run to sync these repos."
    echo ""

    # Skip per-repo details in summary mode
    if [[ "$SUMMARY_ONLY" == "true" ]]; then
        exit 0
    fi

    # Show per-repo diff summary
    log_info "Preview of changes per repo:"
    echo ""

    for repo in "${REPOS[@]}"; do
        repo_name=$(basename "$repo")
        echo "─── $repo_name ───"
        preview_exit=0
        if [[ "$TIMEOUT_ACTIVE" == "true" ]]; then
            preview_output=$("$TIMEOUT_CMD" "$TIMEOUT" "$SCRIPT_DIR/sync_repo.sh" "$repo" --dry-run 2>&1) || preview_exit=$?
        else
            preview_output=$("$SCRIPT_DIR/sync_repo.sh" "$repo" --dry-run 2>&1) || preview_exit=$?
        fi
        if [[ "$TIMEOUT_ACTIVE" == "true" && $preview_exit -eq 124 ]]; then
            echo "  (timed out after ${TIMEOUT}s)"
        elif [[ $preview_exit -ne 0 ]]; then
            echo "  (preview failed, exit $preview_exit)"
            echo "$preview_output" | tail -5 | sed 's/^/  /'
        else
            if ! echo "$preview_output" | grep -E "^\s+\["; then
                echo "  (no changes)"
            fi
        fi
        echo ""
    done
    exit 0
fi

# Track per-repo status (needed for parallel exit codes)
STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ai_template_sync_all.XXXXXX")
cleanup_status_dir() {
    rm -rf "$STATUS_DIR"
}
trap cleanup_status_dir EXIT

# Sync function for single repo
sync_single_repo() {
    local repo="$1"
    local repo_name
    repo_name=$(basename "$repo")
    local sync_args=()

    [[ "$NO_PUSH" == "true" ]] && sync_args+=("--no-push")

    local output
    local exit_code=0
    local timed_out=false
    local timeout_used=false

    if [[ "$TIMEOUT_ACTIVE" == "true" ]]; then
        timeout_used=true
        output=$("$TIMEOUT_CMD" "$TIMEOUT" "$SCRIPT_DIR/sync_repo.sh" "$repo" ${sync_args[@]+"${sync_args[@]}"} 2>&1) || exit_code=$?
    else
        output=$("$SCRIPT_DIR/sync_repo.sh" "$repo" ${sync_args[@]+"${sync_args[@]}"} 2>&1) || exit_code=$?
    fi

    if [[ "$timeout_used" == "true" && $exit_code -eq 124 ]]; then
        timed_out=true
        echo -e "${YELLOW}⏱${NC} $repo_name (timeout after ${TIMEOUT}s)"
    elif [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓${NC} $repo_name"
    else
        echo -e "${RED}✗${NC} $repo_name"
        echo "$output" | tail -5 | sed 's/^/    /'
    fi

    if [[ -n "${STATUS_DIR:-}" ]]; then
        local status_file="$STATUS_DIR/${repo_name}.status"
        printf '%s %s\n' "$exit_code" "$timed_out" > "$status_file"
    fi

    return $exit_code
}

# Execute syncs
FAILED=()
TIMED_OUT=()
SUCCESS=0

if [[ "$PARALLEL" -gt 1 ]]; then
    log_info "Syncing in parallel (max $PARALLEL at a time)..."
    echo ""

    # Export function and variables for parallel execution
    export -f sync_single_repo
    export SCRIPT_DIR NO_PUSH GREEN RED YELLOW NC TIMEOUT TIMEOUT_CMD TIMEOUT_ACTIVE STATUS_DIR

    # Use xargs for parallel execution (null-delimited to handle spaces in paths)
    printf '%s\0' "${REPOS[@]}" | xargs -0 -P "$PARALLEL" -I {} bash -c 'sync_single_repo "$@"' _ {} || true

else
    log_info "Syncing serially..."
    echo ""

    for repo in "${REPOS[@]}"; do
        sync_single_repo "$repo"
    done
fi

# Aggregate status results
FAILED=()
TIMED_OUT=()
SUCCESS=0
for repo in "${REPOS[@]}"; do
    repo_name=$(basename "$repo")
    status_file="$STATUS_DIR/${repo_name}.status"
    if [[ ! -f "$status_file" ]]; then
        FAILED+=("$repo_name")
        continue
    fi
    read -r exit_code timed_out < "$status_file"
    if [[ "$timed_out" == "true" ]]; then
        TIMED_OUT+=("$repo_name")
    elif [[ "$exit_code" -eq 0 ]]; then
        ((SUCCESS++))
    else
        FAILED+=("$repo_name")
    fi
done

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Sync complete"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Success:   $SUCCESS"
[[ ${#TIMED_OUT[@]} -gt 0 ]] && echo "  Timed out: ${#TIMED_OUT[@]}"
echo "  Failed:    ${#FAILED[@]}"

if [[ ${#TIMED_OUT[@]} -gt 0 ]]; then
    echo ""
    echo "Timed out repos:"
    for t in "${TIMED_OUT[@]}"; do
        echo "  - $t"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "Failed repos:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi

if [[ ${#FAILED[@]} -gt 0 || ${#TIMED_OUT[@]} -gt 0 ]]; then
    exit 1
fi

log_ok "All repos synced successfully!"

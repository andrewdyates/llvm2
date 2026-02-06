#!/usr/bin/env bash
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "spawn_all.sh ${git_hash} (${date})"
    exit 0
}

usage() {
    cat <<'EOF'
Usage: spawn_all.sh [OPTIONS] [PROJECT_PATH]

Spawn all 4 AI loops (worker, prover, researcher, manager) in new iTerm2 tabs.

Arguments:
  PROJECT_PATH  Path to project directory (default: current directory)

Options:
  -h, --help           Show this help message
  --version            Show version information
  --dry-run            Show what would be executed without doing it
  --allow-dirty        Skip dirty worktree check (spawn without committing)
  --push               Push after auto-committing dirty worktree
  --isolated           Use isolated checkouts (separate git clones per role)
  --shared             Use shared checkout (default, all roles in same worktree)
  --roles              Comma-separated roles to spawn (default: worker,prover,researcher,manager)
  --workers=N          Number of workers to spawn (default: 2, max: 5)
  --provers=N          Number of provers to spawn (default: 1, max: 5)
  --researchers=N      Number of researchers to spawn (default: 1, max: 5)
  --managers=N         Number of managers to spawn (default: 1, max: 5)

Examples:
  spawn_all.sh                              # All 4 roles in current directory
  spawn_all.sh ~/z4                         # All 4 roles in ~/z4
  spawn_all.sh --roles worker,manager       # Just worker and manager
  spawn_all.sh --workers=3                  # 3 workers + prover, researcher, manager
  spawn_all.sh --provers=2                  # 2 provers (for heavy testing backlogs)
  spawn_all.sh --managers=2                 # 2 managers (for large audit backlogs)
  spawn_all.sh --shared                     # All roles in shared checkout (default)
  spawn_all.sh --isolated                   # All roles in isolated checkouts
  spawn_all.sh --dry-run                    # Preview what would run
EOF
}

# Defaults
DRY_RUN=false
ALLOW_DIRTY=false
PUSH_AFTER_COMMIT=false
ISOLATED=false # Default to shared checkout; use --isolated for separate clones
ROLES="worker,prover,researcher,manager"
PROJECT=""
NUM_WORKERS=2
NUM_PROVERS=1
NUM_RESEARCHERS=1
NUM_MANAGERS=1
SPAWN_COUNT=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --version)
        version
        ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    --allow-dirty)
        ALLOW_DIRTY=true
        shift
        ;;
    --push)
        PUSH_AFTER_COMMIT=true
        shift
        ;;
    --isolated)
        ISOLATED=true
        shift
        ;;
    --shared)
        ISOLATED=false
        shift
        ;;
    --roles)
        if [[ -z "${2-}" ]]; then
            echo "Error: --roles requires a comma-separated list" >&2
            exit 1
        fi
        ROLES="$2"
        shift 2
        ;;
    --roles=*)
        ROLES="${1#*=}"
        shift
        ;;
    --workers=*)
        NUM_WORKERS="${1#*=}"
        if ! [[ "$NUM_WORKERS" =~ ^[1-5]$ ]]; then
            echo "Error: --workers must be between 1 and 5" >&2
            exit 1
        fi
        shift
        ;;
    --provers=*)
        NUM_PROVERS="${1#*=}"
        if ! [[ "$NUM_PROVERS" =~ ^[1-5]$ ]]; then
            echo "Error: --provers must be between 1 and 5" >&2
            exit 1
        fi
        shift
        ;;
    --researchers=*)
        NUM_RESEARCHERS="${1#*=}"
        if ! [[ "$NUM_RESEARCHERS" =~ ^[1-5]$ ]]; then
            echo "Error: --researchers must be between 1 and 5" >&2
            exit 1
        fi
        shift
        ;;
    --managers=*)
        NUM_MANAGERS="${1#*=}"
        if ! [[ "$NUM_MANAGERS" =~ ^[1-5]$ ]]; then
            echo "Error: --managers must be between 1 and 5" >&2
            exit 1
        fi
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

# Default to current directory, resolved to absolute path
PROJECT="${PROJECT:-$(pwd)}"
PROJECT_INPUT="$PROJECT"
PROJECT=$(cd "$PROJECT" 2>/dev/null && pwd) || {
    echo "Error: Project path does not exist: $PROJECT_INPUT" >&2
    exit 1
}

# Get project name for isolated mode paths
PROJECT_NAME=$(basename "$PROJECT")

# Coordination directory for PID files when using --isolated mode
# This centralizes PID management so spawn_all.sh can detect running loopers
# regardless of which directory they run in. See #1385.
if $ISOLATED; then
    COORD_DIR="$HOME/repos/$PROJECT_NAME/_coord"
else
    COORD_DIR="$PROJECT"
fi

# Convert roles to array
IFS=',' read -ra ROLE_ARRAY <<<"$ROLES"

# Validate roles
for role in "${ROLE_ARRAY[@]}"; do
    case "$role" in
    worker | prover | researcher | manager) ;;
    *)
        echo "Error: Invalid role: $role" >&2
        echo "Valid roles: worker, prover, researcher, manager" >&2
        exit 1
        ;;
    esac
done

# Check for existing STOP files (STOP and STOP_*)
STOP_FILES=()
collect_stop_files() {
    local target_dir="$1"
    [[ -d "$target_dir" ]] || return 0
    local stop_file
    if [[ -f "$target_dir/STOP" ]]; then
        STOP_FILES+=("$target_dir/STOP")
    fi
    for stop_file in "$target_dir"/STOP_*; do
        [[ -f "$stop_file" ]] && STOP_FILES+=("$stop_file")
    done
    return 0
}
collect_stop_files "$PROJECT"
if $ISOLATED; then
    for role in "${ROLE_ARRAY[@]}"; do
        case "$role" in
        worker) role_count=$NUM_WORKERS ;;
        prover) role_count=$NUM_PROVERS ;;
        researcher) role_count=$NUM_RESEARCHERS ;;
        manager) role_count=$NUM_MANAGERS ;;
        esac
        if [[ "$role_count" -gt 1 ]]; then
            for ((i = 1; i <= role_count; i++)); do
                collect_stop_files "$HOME/repos/$PROJECT_NAME/${role}-${i}"
            done
        else
            collect_stop_files "$HOME/repos/$PROJECT_NAME/$role"
        fi
    done
fi

if [[ ${#STOP_FILES[@]} -gt 0 ]]; then
    echo "Warning: Removing stale STOP files from previous session: ${STOP_FILES[*]}"
    if $DRY_RUN; then
        echo "[dry-run] Would remove: ${STOP_FILES[*]}"
    else
        for f in "${STOP_FILES[@]}"; do
            rm "$f"
        done
    fi
fi

# Check for existing PID files and validate running status
# Use COORD_DIR for PID files (centralized location for isolated mode)
# Create COORD_DIR if it doesn't exist (first run of isolated mode)
if [[ ! -d "$COORD_DIR" ]] && ! $DRY_RUN; then
    mkdir -p "$COORD_DIR"
fi
RUNNING_PIDS=()
STALE_PIDS=()
for pid_file in "$COORD_DIR"/.pid_*; do
    [[ -f "$pid_file" ]] || continue
    role_name=$(basename "$pid_file" | sed 's/^.pid_//')
    pid=$(cat "$pid_file" 2>/dev/null)
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
        RUNNING_PIDS+=("$role_name (PID $pid)")
    else
        STALE_PIDS+=("$(basename "$pid_file")")
    fi
done

# Remove stale PID files
if [[ ${#STALE_PIDS[@]} -gt 0 ]]; then
    echo "Warning: Removing stale PID files (processes not running): ${STALE_PIDS[*]}"
    if $DRY_RUN; then
        echo "[dry-run] Would remove: ${STALE_PIDS[*]}"
    else
        for f in "${STALE_PIDS[@]}"; do
            rm "$COORD_DIR/$f"
        done
    fi
fi

# Abort if any roles are already running
if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
    echo "Error: The following roles are already running:" >&2
    for info in "${RUNNING_PIDS[@]}"; do
        echo "  - $info" >&2
    done
    echo "" >&2
    echo "Stop them first with 'touch STOP' (all) or 'touch STOP_<ROLE>' (per-role)" >&2
    echo "Or manually remove stale PID files if processes are stuck." >&2
    exit 1
fi

# Upgrade AI tools before spawning
if ! $DRY_RUN; then
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
fi

# Check for dirty worktree and offer to commit
check_dirty_worktree() {
    # Skip if project doesn't exist or isn't a git repo
    [[ -d "$PROJECT" ]] || return 0
    [[ -d "$PROJECT/.git" ]] || return 0

    cd "$PROJECT" || return 0

    # Get list of changed files (staged + unstaged + untracked)
    local changed_files
    changed_files=$(git status --porcelain 2>/dev/null | head -20)

    if [[ -z "$changed_files" ]]; then
        return 0 # Clean worktree
    fi

    local file_count
    file_count=$(echo "$changed_files" | wc -l | tr -d ' ')

    echo "⚠️  Dirty worktree detected ($file_count file(s) with uncommitted changes)"
    echo ""
    echo "$changed_files" | head -10
    [[ "$file_count" -gt 10 ]] && echo "  ... and $((file_count - 10)) more"
    echo ""

    # Generate helpful commit message from changed files
    local file_summary
    file_summary=$(echo "$changed_files" | awk '{print $2}' | head -5 | xargs -I{} basename {} | tr '\n' ', ' | sed 's/,$//')
    local commit_msg="[U]: WIP changes before team spawn ($file_summary)"

    echo "Suggested commit message:"
    echo "  $commit_msg"
    echo ""
    echo "Press ENTER to commit with this message, or type a new message."
    echo "Auto-committing in 5 seconds..."
    echo ""

    # Read with 5 second timeout
    local user_input
    if read -r -t 5 user_input; then
        if [[ -n "$user_input" ]]; then
            commit_msg="[U]: $user_input"
        fi
    else
        echo "" # Newline after timeout
    fi

    echo "Committing: $commit_msg"
    git add -A
    if git commit -m "$commit_msg"; then
        echo "✓ Changes committed successfully"
        # Push if requested
        if $PUSH_AFTER_COMMIT; then
            echo "Pushing to remote..."
            if git push; then
                echo "✓ Pushed successfully"
            else
                echo "Warning: Push failed (continuing anyway)" >&2
            fi
        fi
        echo ""
    else
        echo "Error: Failed to commit changes" >&2
        echo "Please resolve manually and retry." >&2
        exit 1
    fi
}

if ! $ALLOW_DIRTY && ! $DRY_RUN; then
    check_dirty_worktree
fi

# Calculate total number of sessions to spawn
TOTAL_SESSIONS=0
for role in "${ROLE_ARRAY[@]}"; do
    case "$role" in
    worker) TOTAL_SESSIONS=$((TOTAL_SESSIONS + NUM_WORKERS)) ;;
    prover) TOTAL_SESSIONS=$((TOTAL_SESSIONS + NUM_PROVERS)) ;;
    researcher) TOTAL_SESSIONS=$((TOTAL_SESSIONS + NUM_RESEARCHERS)) ;;
    manager) TOTAL_SESSIONS=$((TOTAL_SESSIONS + NUM_MANAGERS)) ;;
    esac
done

echo "Spawning $TOTAL_SESSIONS loops in: $PROJECT"
# Show multi-instance info
MULTI_INFO=""
[[ "$NUM_WORKERS" -gt 1 ]] && MULTI_INFO="${MULTI_INFO}Workers: $NUM_WORKERS  "
[[ "$NUM_PROVERS" -gt 1 ]] && MULTI_INFO="${MULTI_INFO}Provers: $NUM_PROVERS  "
[[ "$NUM_RESEARCHERS" -gt 1 ]] && MULTI_INFO="${MULTI_INFO}Researchers: $NUM_RESEARCHERS  "
[[ "$NUM_MANAGERS" -gt 1 ]] && MULTI_INFO="${MULTI_INFO}Managers: $NUM_MANAGERS  "
[[ -n "$MULTI_INFO" ]] && echo "$MULTI_INFO"
echo "Roles: ${ROLE_ARRAY[*]}"
echo

# Helper to spawn N instances of a role
spawn_role() {
    local role=$1
    local count=$2
    if [[ "$count" -gt 1 ]]; then
        # Multi-instance mode: spawn N instances with --id=1, --id=2, etc.
        for ((i = 1; i <= count; i++)); do
            local -a cmd=("$SCRIPT_DIR/spawn_session.sh")
            # After the first spawn, sessions may dirty the worktree (logs, pid files).
            local allow_dirty=false
            if $ALLOW_DIRTY; then
                allow_dirty=true
            elif ! $ISOLATED && [[ $SPAWN_COUNT -gt 0 ]]; then
                allow_dirty=true
            fi
            $allow_dirty && cmd+=(--allow-dirty)
            $ISOLATED && cmd+=(--isolated "--coord-dir=$COORD_DIR")
            cmd+=(--id="$i" "$role" "$PROJECT")
            if $DRY_RUN; then
                echo "[dry-run] ${cmd[*]}"
            else
                "${cmd[@]}"
                sleep 0.5 # Small delay between spawns to avoid race conditions
            fi
            SPAWN_COUNT=$((SPAWN_COUNT + 1))
        done
    else
        local -a cmd=("$SCRIPT_DIR/spawn_session.sh")
        # After the first spawn, sessions may dirty the worktree (logs, pid files).
        local allow_dirty=false
        if $ALLOW_DIRTY; then
            allow_dirty=true
        elif ! $ISOLATED && [[ $SPAWN_COUNT -gt 0 ]]; then
            allow_dirty=true
        fi
        $allow_dirty && cmd+=(--allow-dirty)
        $ISOLATED && cmd+=(--isolated "--coord-dir=$COORD_DIR")
        cmd+=("$role" "$PROJECT")
        if $DRY_RUN; then
            echo "[dry-run] ${cmd[*]}"
        else
            "${cmd[@]}"
            sleep 0.5 # Small delay between spawns to avoid race conditions
        fi
        SPAWN_COUNT=$((SPAWN_COUNT + 1))
    fi
}

# Spawn each role
for role in "${ROLE_ARRAY[@]}"; do
    case "$role" in
    worker) spawn_role "$role" "$NUM_WORKERS" ;;
    prover) spawn_role "$role" "$NUM_PROVERS" ;;
    researcher) spawn_role "$role" "$NUM_RESEARCHERS" ;;
    manager) spawn_role "$role" "$NUM_MANAGERS" ;;
    esac
done

echo
echo "All loops spawned. Use 'touch STOP' (all) or 'touch STOP_WORKER' (per-role) to stop."

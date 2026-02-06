#!/usr/bin/env bash
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# spawn_session.sh - Spawn worker, prover, researcher, or manager in new iTerm2 tab
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# Usage:
#   ./ai_template_scripts/spawn_session.sh worker              # Spawn worker in current project
#   ./ai_template_scripts/spawn_session.sh prover              # Spawn prover in current project
#   ./ai_template_scripts/spawn_session.sh researcher ~/z4     # Spawn researcher in specific project
#   ./ai_template_scripts/spawn_session.sh manager ~/z4        # Spawn manager in specific project
#   ./ai_template_scripts/spawn_session.sh --help              # Show help

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: spawn_session.sh [OPTIONS] [MODE] [PROJECT_PATH]

Spawn a worker, prover, researcher, or manager session in a new iTerm2 tab.

Arguments:
  MODE          worker (default), prover, researcher, or manager
  PROJECT_PATH  Path to project directory (default: current directory)

Options:
  -h, --help        Show this help message
  --version         Show version information
  --dry-run         Show what would be executed without doing it
  --allow-dirty     Allow spawning with uncommitted changes
  --id=N            Worker identity for multi-worker mode (e.g., --id=1)
  --isolated        Use isolated checkouts (per-role clones)
  --shared          Use shared checkout (default)
  --coord-dir=PATH  Coordination directory for PID files (set AIT_COORD_DIR)
                    (override auto-set from --isolated if needed)

Examples:
  spawn_session.sh                       # Worker in current directory
  spawn_session.sh worker                # Worker in current directory
  spawn_session.sh prover                # Prover in current directory
  spawn_session.sh researcher            # Researcher in current directory
  spawn_session.sh manager               # Manager in current directory
  spawn_session.sh worker ~/z4           # Worker in ~/z4
  spawn_session.sh ~/z4                  # Worker in ~/z4 (auto-detected)
  spawn_session.sh ~/z4 worker           # Worker in ~/z4 (swapped args ok)
  spawn_session.sh --id=1 worker         # Worker 1 (multi-worker mode)
  spawn_session.sh --id=2 worker         # Worker 2 (multi-worker mode)
  spawn_session.sh --isolated worker     # Worker in isolated checkout (per-role clone)
  spawn_session.sh --shared worker       # Worker in shared checkout (default)
  spawn_session.sh --coord-dir=/tmp/coord worker  # Worker with custom PID coordination dir

See also:
  spawn_all.sh                           # Spawn all 4 loops at once
EOF
}

version() {
    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    local git_hash
    git_hash=$(git -C "$script_dir/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "spawn_session.sh ${git_hash} (${date})"
    exit 0
}

err() {
    echo "Error: $1" >&2
    if [[ -n "${2-}" ]]; then
        echo "       $2" >&2
    fi
    exit 1
}

# Sanitize git URL to avoid credential leakage in error messages (#2240)
sanitize_url() {
    local url="$1"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 "$script_dir/url_sanitizer.py" "$url" 2>/dev/null || echo "[REDACTED]"
}

# Parse options
DRY_RUN=false
ALLOW_DIRTY=false
WORKER_ID=""
ISOLATED=false # Default to shared checkout; use --isolated for separate clones
COORD_DIR=""
while [[ "${1-}" == -* ]]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --version) version ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    --allow-dirty)
        ALLOW_DIRTY=true
        shift
        ;;
    --id=*)
        WORKER_ID="${1#--id=}"
        # Must be integer 1-5 to match pulse.py GraphQL label counting
        # See #1080: prevents in-progress-W43 label sprawl
        if ! [[ "$WORKER_ID" =~ ^[1-5]$ ]]; then
            err "Invalid --id value: $WORKER_ID" "Must be 1-5 (e.g., --id=1)"
        fi
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
    --coord-dir=*)
        COORD_DIR="${1#--coord-dir=}"
        if [[ -z "$COORD_DIR" ]]; then
            err "--coord-dir requires a non-empty path" "Example: --coord-dir=/tmp/coord"
        fi
        shift
        ;;
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

# Verify path is a git repository (required for both shared and isolated modes)
if [[ ! -d "$PROJECT/.git" ]]; then
    err "$PROJECT is not a git repository" "spawn_session requires a git repository"
fi

# Check looper.py exists
if [[ ! -f "$PROJECT/looper.py" ]]; then
    err "$PROJECT/looper.py not found" "Make sure you're pointing to a project with looper.py"
fi

# Check looper.py is executable
if [[ ! -x "$PROJECT/looper.py" ]]; then
    err "$PROJECT/looper.py is not executable" "Run: chmod +x $PROJECT/looper.py"
fi

# Get project name for isolation and tab title
PROJECT_NAME=$(basename "$PROJECT")
PROJECT_NAME_ORIG="$PROJECT_NAME"

# Handle isolated mode: use per-role clone instead of shared checkout
# Layout: ~/repos/<project>/<role>-<id>/ (or <role>/ if no id)
if $ISOLATED; then
    # Verify source is a git repo before proceeding
    if [[ ! -d "$PROJECT/.git" ]]; then
        err "$PROJECT is not a git repository" "Isolated mode requires a git repository with remote 'origin'"
    fi

    # Get remote URL from source project
    REMOTE_URL=$(cd "$PROJECT" && git remote get-url origin 2>/dev/null) || {
        err "Cannot determine remote URL for $PROJECT" "Isolated mode requires git remote 'origin'"
    }

    # Validate REMOTE_URL is actually a remote URL, not a local path
    # This prevents issues where a chain of local clones causes origin to point
    # to a non-bare local repo (which breaks push). See #1783.
    # Fixes: #1792 (file:// origins), #1793 (relative path resolution),
    #        #1794 (bare repo detection)
    resolve_remote_url() {
        local url="$1"
        local base_dir="$2" # Base directory for resolving relative paths
        local max_depth=5
        local depth=0

        # Helper: check if path is a local path (not a remote URL)
        is_local_path() {
            local u="$1"
            # file:// prefix (#1792)
            [[ "$u" =~ ^file:// ]] && return 0
            # Absolute path
            [[ "$u" =~ ^/ ]] && return 0
            # Relative path (., .., or any path without ://)
            [[ "$u" =~ ^\.\.?(/|$) ]] && return 0
            # Tilde expansion
            [[ "$u" =~ ^~ ]] && return 0
            # Path without protocol (no ://) that exists as directory
            [[ ! "$u" =~ :// ]] && [[ -d "$base_dir/$u" || -d "$u" ]] && return 0
            return 1
        }

        while is_local_path "$url"; do
            # Strip file:// prefix if present (#1792)
            if [[ "$url" =~ ^file:// ]]; then
                url="${url#file://}"
            fi

            # Expand tilde
            if [[ "$url" =~ ^~ ]]; then
                url="${url/#\~/$HOME}"
            fi

            # Resolve relative paths from base_dir, not cwd (#1793)
            local abs_path
            if [[ "$url" =~ ^/ ]]; then
                # Absolute path
                abs_path="$url"
            else
                # Relative path - resolve from base_dir
                abs_path=$(cd "$base_dir" && cd "$url" 2>/dev/null && pwd) || {
                    err "Origin '$url' is a local path that doesn't exist" \
                        "(resolved relative to $base_dir)"
                }
            fi

            # Verify it's a git repository - handle both normal and bare repos (#1794)
            if [[ -d "$abs_path/.git" ]]; then
                # Normal repository with .git directory
                :
            elif git -C "$abs_path" rev-parse --is-bare-repository >/dev/null 2>&1; then
                # Bare repository (no .git directory, but is a valid git repo)
                # Bare repos are valid remotes - return as-is since they have no origin
                echo "$abs_path"
                return 0
            else
                err "Origin '$abs_path' is not a git repository"
            fi

            url=$(git -C "$abs_path" remote get-url origin 2>/dev/null) || {
                err "Cannot get origin from local repo '$abs_path'" \
                    "The origin chain leads to a repo without a remote origin configured"
            }

            # Update base_dir for next iteration (chain resolution)
            base_dir="$abs_path"

            ((depth++))
            if [[ $depth -ge $max_depth ]]; then
                err "Origin resolution exceeded $max_depth levels" \
                    "Local clone chain is too deep; please fix the origin remote"
            fi
        done
        echo "$url"
    }

    # Resolve through any local path chain to find the actual remote URL
    # Pass PROJECT as base_dir for resolving relative paths (#1793)
    REMOTE_URL=$(resolve_remote_url "$REMOTE_URL" "$PROJECT") || exit 1

    # Final validation: must be a remote protocol or a bare repo path (#1794)
    # Bare repos are returned as absolute paths by resolve_remote_url
    if [[ ! "$REMOTE_URL" =~ ^(https://|git@|ssh://|git://|/) ]]; then
        err "Origin URL '$(sanitize_url "$REMOTE_URL")' is not a recognized remote protocol" \
            "Expected https://, git@, ssh://, git://, or absolute path to bare repo"
    fi

    # Build isolated path
    if [[ -n "$WORKER_ID" ]]; then
        ISOLATED_PATH="$HOME/repos/$PROJECT_NAME/$MODE-$WORKER_ID"
    else
        ISOLATED_PATH="$HOME/repos/$PROJECT_NAME/$MODE"
    fi

    if ! $DRY_RUN; then
        # Create or update clone
        if [[ ! -d "$ISOLATED_PATH/.git" ]]; then
            echo "Creating isolated clone at $ISOLATED_PATH..."
            mkdir -p "$(dirname "$ISOLATED_PATH")"
            git clone "$REMOTE_URL" "$ISOLATED_PATH" || {
                err "Failed to clone $(sanitize_url "$REMOTE_URL") to $ISOLATED_PATH"
            }
        else
            echo "Updating isolated clone at $ISOLATED_PATH..."
            (cd "$ISOLATED_PATH" && git fetch origin 2>/dev/null) || {
                echo "Warning: Failed to fetch updates" >&2
            }
        fi

        # Ensure on main branch
        CURRENT_BRANCH=$(cd "$ISOLATED_PATH" && git rev-parse --abbrev-ref HEAD 2>/dev/null)
        if [[ "$CURRENT_BRANCH" != "main" ]]; then
            echo "Switching to main branch (was: $CURRENT_BRANCH)..."
            (cd "$ISOLATED_PATH" && git checkout main 2>/dev/null) || {
                err "Failed to switch to main branch in $ISOLATED_PATH"
            }
        fi

        # Reset to origin/main to ensure clean state
        (cd "$ISOLATED_PATH" && git reset --hard origin/main 2>/dev/null) || {
            echo "Warning: Failed to reset to origin/main" >&2
        }
    fi

    # Use the isolated path as the project
    PROJECT="$ISOLATED_PATH"

    if ! $DRY_RUN; then
        # Verify looper.py exists in the clone
        if [[ ! -f "$PROJECT/looper.py" ]]; then
            err "looper.py not found in clone at $PROJECT" "Clone may need sync from ai_template"
        fi
        if [[ ! -x "$PROJECT/looper.py" ]]; then
            chmod +x "$PROJECT/looper.py" 2>/dev/null || {
                err "Failed to make looper.py executable in $PROJECT"
            }
        fi
    fi

    # Auto-set COORD_DIR for isolated mode if not explicitly provided
    # This matches spawn_all.sh behavior and ensures PID files are centralized
    # even when spawn_session.sh is called directly. See #1385.
    if [[ -z "$COORD_DIR" ]]; then
        COORD_DIR="$HOME/repos/$PROJECT_NAME/_coord"
    fi
fi

# Guardrail: if not isolated and using known baseline path, check it's clean and on main
# Known baseline paths: ~/<project>, ~/Developer/<project>
# This prevents cross-role interference when multiple roles share a checkout
if ! $ISOLATED && ! $ALLOW_DIRTY; then
    HOME_PATTERN="^$HOME/[^/]+$"
    DEV_PATTERN="^$HOME/Developer/[^/]+$"
    if [[ "$PROJECT" =~ $HOME_PATTERN || "$PROJECT" =~ $DEV_PATTERN ]]; then
        # Check if on main branch
        CURRENT_BRANCH=$(cd "$PROJECT" && git rev-parse --abbrev-ref HEAD 2>/dev/null)
        if [[ "$CURRENT_BRANCH" != "main" ]]; then
            err "Baseline path $PROJECT is on branch '$CURRENT_BRANCH', not 'main'" \
                "Use --isolated for per-role clones, --allow-dirty to bypass, or checkout main"
        fi

        # Check for uncommitted changes
        if [[ -n $(cd "$PROJECT" && git status --porcelain 2>/dev/null) ]]; then
            err "Baseline path $PROJECT has uncommitted changes" \
                "Use --isolated for per-role clones, --allow-dirty to bypass, or commit changes (WIP is fine)"
        fi
    fi
fi

# Get project name and role for tab title
if $ISOLATED; then
    PROJECT_NAME="$PROJECT_NAME_ORIG"
else
    PROJECT_NAME=$(basename "$PROJECT")
fi
# Sanitize project name for osascript (remove quotes and backslashes)
PROJECT_NAME_SAFE="${PROJECT_NAME//[\"\'\\\$]/}"
case "$MODE" in
worker) ROLE="W" ;;
prover) ROLE="P" ;;
researcher) ROLE="R" ;;
manager) ROLE="M" ;;
esac

# Build role display: W or W1 for multi-worker
if [[ -n "$WORKER_ID" ]]; then
    ROLE_DISPLAY="${ROLE}${WORKER_ID}"
else
    ROLE_DISPLAY="$ROLE"
fi
TAB_TITLE="[$ROLE_DISPLAY]$PROJECT_NAME_SAFE"

# Build command
DIRTY_FLAG=""
$ALLOW_DIRTY && DIRTY_FLAG=" --allow-dirty"

# Prepend ai_template_scripts/bin to PATH for wrapper precedence (#1690)
# This ensures gh/cargo wrappers are used even in fresh shell environments
# Also include $HOME/.local/bin for claude CLI and /opt/homebrew/bin for macOS tools
# Mixed quoting: single quotes keep $PATH literal, double quotes expand $PROJECT and $HOME
PATH_PREFIX='export PATH='"$PROJECT"'/ai_template_scripts/bin:'"$HOME"'/.local/bin:/opt/homebrew/bin:$PATH && '

# Set AIT_COORD_DIR for centralized PID file management (see #1385)
COORD_PREFIX=""
if [[ -n "$COORD_DIR" ]]; then
    COORD_PREFIX="export AIT_COORD_DIR='$COORD_DIR' && "
fi

# Enable GitHub Apps for autonomous sessions (see #2294)
# Autonomous sessions use per-app rate limits (5000/hr each) instead of shared quota
# User sessions (interactive claude) continue using default gh auth
GH_APPS_PREFIX="export AIT_USE_GITHUB_APPS=1 && "

if [[ -n "$WORKER_ID" ]]; then
    CMD="${PATH_PREFIX}${COORD_PREFIX}${GH_APPS_PREFIX}cd '$PROJECT' && ./looper.py $MODE --id=$WORKER_ID$DIRTY_FLAG"
else
    CMD="${PATH_PREFIX}${COORD_PREFIX}${GH_APPS_PREFIX}cd '$PROJECT' && ./looper.py $MODE$DIRTY_FLAG"
fi

if $DRY_RUN; then
    echo "Would create iTerm2 tab: $TAB_TITLE"
    echo "Would run: $CMD"
    exit 0
fi

# Clear stale STOP files from previous sessions (#2368, #2369)
# Spawning = intent to start, so clear any stops that would block this session
# STOP files older than STOP_EXPIRY_SEC are marked as expired in output
MODE_UPPER=$(echo "$MODE" | tr '[:lower:]' '[:upper:]')
CLEARED_STOPS=()
STOP_EXPIRY_SEC="${AIT_STOP_EXPIRY_SEC:-3600}" # Default 1 hour

# Helper to clear a stop file and show its reason/age
clear_stop_file() {
    local stop_file="$1"
    local display_name="$2"
    if [[ -f "$stop_file" ]]; then
        local reason="" age_note=""
        # Get reason from file content
        if [[ -s "$stop_file" ]]; then
            reason=$(head -1 "$stop_file" 2>/dev/null | tr -d '\n')
        fi
        # Check age using stat (macOS format)
        local mtime now age_sec
        mtime=$(stat -f %m "$stop_file" 2>/dev/null) || mtime=0
        now=$(date +%s)
        age_sec=$((now - mtime))
        if [[ $age_sec -gt $STOP_EXPIRY_SEC ]]; then
            local age_min=$((age_sec / 60))
            age_note=" [expired: ${age_min}m old]"
        fi
        rm -f "$stop_file"
        local msg="$display_name"
        [[ -n "$reason" ]] && msg="$msg (reason: $reason)"
        [[ -n "$age_note" ]] && msg="$msg$age_note"
        CLEARED_STOPS+=("$msg")
    fi
}

# Check from most specific to least specific
# Instance-specific: STOP_W1, STOP_W2, etc.
if [[ -n "$WORKER_ID" ]]; then
    clear_stop_file "$PROJECT/STOP_${MODE_UPPER}${WORKER_ID}" "STOP_${MODE_UPPER}${WORKER_ID}"
fi
# Role-specific: STOP_WORKER, STOP_MANAGER, etc.
clear_stop_file "$PROJECT/STOP_${MODE_UPPER}" "STOP_${MODE_UPPER}"
# Global: STOP
clear_stop_file "$PROJECT/STOP" "STOP"

# Report what was cleared
if [[ ${#CLEARED_STOPS[@]} -gt 0 ]]; then
    echo "Cleared stale STOP files: ${CLEARED_STOPS[*]}"
fi

# Check iTerm2 is running (only when actually creating a tab)
# Note: pgrep -x doesn't work for GUI apps on macOS, use -f with app bundle path
if ! pgrep -f "iTerm.app" >/dev/null 2>&1; then
    err "iTerm2 is not running" "Start iTerm2 first, then run this script"
fi

# Create new tab and run command
# IMPORTANT: Capture the new tab reference explicitly - "current session" after
# create tab doesn't reliably point to the new tab, causing the caller's tab
# to be reused (#342)
if ! osascript -e "
tell application \"iTerm2\"
    tell front window
        set newTab to (create tab with default profile)
        tell current session of newTab
            set name to \"$TAB_TITLE\"
            write text \"$CMD\"
        end tell
    end tell
end tell
" 2>&1; then
    err "Failed to create iTerm2 tab" "Make sure iTerm2 has an open window"
fi

echo "Spawned $MODE session: $TAB_TITLE"

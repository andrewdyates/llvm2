#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# sync_repo.sh - Sync template files to a target repo
#
# CANONICAL SOURCE: dropbox-ai-prototypes/ai_template
# DO NOT EDIT in other repos - file issues to ai_template for changes.
#
# Copies template files from ai_template to a target repository, commits,
# and pushes. Handles git pull/push automatically.
#
# Usage:
#   ./ai_template_scripts/sync_repo.sh /path/to/target_repo
#   ./ai_template_scripts/sync_repo.sh /path/to/target_repo --dry-run
#   ./ai_template_scripts/sync_repo.sh /path/to/target_repo --no-push
#
# Copyright 2026 Dropbox, Inc.
# Licensed under the Apache License, Version 2.0

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
    echo "sync_repo.sh ${git_hash} (${date})"
    exit 0
}

# Cache for archived repos (avoid repeated API calls)
ARCHIVED_CACHE_DIR="${HOME}/.cache/ai_template"
ARCHIVED_CACHE_FILE="${ARCHIVED_CACHE_DIR}/archived_repos.txt"
ARCHIVED_CACHE_AGE=3600 # 1 hour

# Check if repo is archived (with REST fallback and caching)
is_repo_archived() {
    local repo_name="$1"
    if [[ -f "$ARCHIVED_CACHE_FILE" ]]; then
        local cache_age=$(($(date +%s) - $(stat -f %m "$ARCHIVED_CACHE_FILE" 2>/dev/null || echo 0)))
        if [[ $cache_age -lt $ARCHIVED_CACHE_AGE ]]; then
            if grep -qx "$repo_name" "$ARCHIVED_CACHE_FILE" 2>/dev/null; then
                return 0
            fi
        fi
    fi
    local result stderr_file
    stderr_file=$(mktemp)
    # NOTE: Do NOT use gh -q or --jq - has caching bugs in v2.83.2+ (#1047)
    if result=$(gh repo view "dropbox-ai-prototypes/$repo_name" --json isArchived 2>"$stderr_file" | jq -r '.isArchived'); then
        rm -f "$stderr_file"
        if [[ "$result" == "true" ]]; then
            mkdir -p "$ARCHIVED_CACHE_DIR"
            echo "$repo_name" >>"$ARCHIVED_CACHE_FILE"
            return 0
        fi
        return 1
    fi
    if grep -qiE 'rate.?limit' "$stderr_file" 2>/dev/null; then
        rm -f "$stderr_file"
        if result=$(gh api "repos/dropbox-ai-prototypes/$repo_name" 2>/dev/null | jq -r '.archived'); then
            if [[ "$result" == "true" ]]; then
                mkdir -p "$ARCHIVED_CACHE_DIR"
                echo "$repo_name" >>"$ARCHIVED_CACHE_FILE"
                return 0
            fi
            return 1
        fi
    fi
    rm -f "$stderr_file"
    return 1
}

# Must run from ai_template root (not just any repo with CLAUDE.md)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$AI_TEMPLATE_ROOT"

# Verify this is actually ai_template by checking git remote
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [[ ! "$REMOTE_URL" =~ ai_template(\.git)?$ ]]; then
    log_error "Must run from ai_template repo (got: $REMOTE_URL)"
    exit 1
fi

# Parse args
TARGET_REPO=""
DRY_RUN=false
NO_PUSH=false
CLEAN=false
SHOW_DIFF=false
SHOW_HELP=false
declare -a ONLY_PATTERNS=()

# Parse arguments (need manual loop for --only with value)
while [[ $# -gt 0 ]]; do
    case "$1" in
    --version) version ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    --no-push)
        NO_PUSH=true
        shift
        ;;
    --clean)
        CLEAN=true
        shift
        ;;
    --diff)
        SHOW_DIFF=true
        DRY_RUN=true
        shift
        ;;
    -h | --help)
        SHOW_HELP=true
        shift
        ;;
    --only)
        if [[ -z "${2:-}" ]]; then
            log_error "--only requires a glob pattern argument"
            exit 1
        fi
        ONLY_PATTERNS+=("$2")
        shift 2
        ;;
    --only=*)
        ONLY_PATTERNS+=("${1#--only=}")
        shift
        ;;
    -*)
        log_error "Unknown option: $1"
        exit 1
        ;;
    *)
        TARGET_REPO="$1"
        shift
        ;;
    esac
done

if [[ "$SHOW_HELP" == "true" ]] || [[ -z "$TARGET_REPO" ]]; then
    echo "Usage: $0 /path/to/target_repo [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run        Show what would change without modifying files"
    echo "  --no-push        Commit changes but don't push to remote"
    echo "  --clean          Run alignment audit with --clean flag"
    echo "  --diff           Show detailed diff of changes (implies --dry-run)"
    echo "  --only <glob>    Only sync entries matching pattern (can repeat, OR logic)"
    echo "  --version        Show version information"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Selective sync examples:"
    echo "  --only 'looper/*'     Sync only looper directory"
    echo "  --only '*.md'         Sync only markdown files"
    echo "  --only 'looper/*' --only '.claude/*'  Sync both (OR logic)"
    echo ""
    echo "Target-side skip patterns:"
    echo "  Create .ai_template_skip in target repo to skip specific files:"
    echo "    ruff.toml             # Keep project-specific ruff config"
    echo "    looper.py             # Keep customized looper"
    echo "    .claude/roles/*.md    # Keep custom roles"
    echo ""
    echo "Post-sync hooks:"
    echo "  If .ai_template_hooks/post-sync.sh exists and is executable in the target"
    echo "  repo, it will run after sync. Environment vars: AI_TEMPLATE_VERSION,"
    echo "  SYNCED_FILES_STR (space-separated list of synced files)."
    # Exit 0 for explicit --help, 1 for missing required arg
    [[ "$SHOW_HELP" == "true" ]] && exit 0 || exit 1
fi

# Resolve to absolute path, with ~ fallback
if [[ -d "$TARGET_REPO" ]]; then
    TARGET_REPO="$(cd "$TARGET_REPO" && pwd)"
elif [[ -d "$HOME/$TARGET_REPO" ]]; then
    log_warn "Path '$TARGET_REPO' not found, using ~/$TARGET_REPO"
    TARGET_REPO="$(cd "$HOME/$TARGET_REPO" && pwd)"
else
    log_error "Target repo not found: $TARGET_REPO"
    log_error "Checked: $TARGET_REPO and ~/$TARGET_REPO"
    exit 1
fi

[[ -d "$TARGET_REPO/.git" ]] || {
    log_error "Target is not a git repo: $TARGET_REPO"
    exit 1
}

# SAFETY: Prevent syncing TO ai_template itself (#2203)
# Two checks for defense-in-depth:
# 1. Remote URL pattern (requires path separator before ai_template to avoid matching "my_ai_template")
# 2. Directory path basename (catches repos without origin remote set)
TARGET_REMOTE=$(cd "$TARGET_REPO" && git remote get-url origin 2>/dev/null || echo "")
TARGET_BASENAME=$(basename "$TARGET_REPO")
if [[ "$TARGET_REMOTE" =~ [:/]ai_template(\.git)?$ ]] || [[ "$TARGET_BASENAME" == "ai_template" ]]; then
    log_error "Cannot sync TO ai_template - sync is only FROM ai_template to other repos"
    exit 1
fi

# Check source repo (ai_template) for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo ""
    log_warn "━━━ ai_template has UNCOMMITTED changes ━━━"
    log_warn "Working tree will be synced, but .ai_template_version"
    log_warn "will record HEAD commit - version marker will be stale!"
    log_warn "Uncommitted files:"
    git diff --name-only | sed 's/^/  /'
    git diff --cached --name-only | sed 's/^/  /'
    log_warn "Recommend: commit first, then sync"
    log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# Get ai_template version
AI_TEMPLATE_VERSION=$(git rev-parse --short HEAD)
AI_TEMPLATE_VERSION_FULL=$(git rev-parse HEAD)

log_info "Syncing ai_template ($AI_TEMPLATE_VERSION) to $(basename "$TARGET_REPO")"
[[ "$DRY_RUN" == "true" ]] && log_warn "DRY RUN - no files will be modified"

# Check if repo is archived
TARGET_NAME=$(basename "$TARGET_REPO")
if is_repo_archived "$TARGET_NAME"; then
    log_error "Repository is archived on GitHub. Skipping."
    exit 1
fi

# Check target repo state and pull if needed
if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    log_info "Checking target repo git state..."
    pushd "$TARGET_REPO" >/dev/null

    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_error "Target repo has uncommitted changes. Commit first (never stash)."
        popd >/dev/null
        exit 1
    fi

    # Fetch and check if we need to pull
    git fetch origin 2>/dev/null || log_warn "Could not fetch from origin"

    LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "none")
    REMOTE=$(git rev-parse '@{u}' 2>/dev/null || echo "none")
    BASE=$(git merge-base HEAD '@{u}' 2>/dev/null || echo "none")

    if [[ "$LOCAL" != "$REMOTE" && "$REMOTE" != "none" ]]; then
        if [[ "$LOCAL" == "$BASE" ]]; then
            log_info "Pulling latest changes from remote..."
            git pull --ff-only || {
                log_error "Pull failed (not fast-forward). Resolve divergence manually."
                popd >/dev/null
                exit 1
            }
        elif [[ "$REMOTE" == "$BASE" ]]; then
            log_info "Local is ahead of remote (will push after sync)"
        else
            log_error "Local and remote have diverged. Resolve manually."
            popd >/dev/null
            exit 1
        fi
    fi

    popd >/dev/null
fi

# Files and directories to sync
# These are template infrastructure files that should be identical across repos

# Track synced files for post-sync hooks
declare -a SYNCED_FILES=()

sync_file() {
    local src="$1"
    local dst="$TARGET_REPO/$1"

    if [[ ! -e "$src" ]]; then
        log_warn "Source missing: $src"
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ -e "$dst" ]]; then
            if diff -q "$src" "$dst" >/dev/null 2>&1; then
                echo "  [unchanged] $src"
            else
                echo "  [update] $src"
                if [[ "$SHOW_DIFF" == "true" ]]; then
                    echo "    ────────────────────────────────────────"
                    diff -u "$dst" "$src" 2>/dev/null | head -50 | sed 's/^/    /' || true
                    echo "    ────────────────────────────────────────"
                fi
            fi
        else
            echo "  [create] $src"
            if [[ "$SHOW_DIFF" == "true" ]]; then
                echo "    ────────────────────────────────────────"
                head -20 "$src" | sed 's/^/    + /'
                line_count=$(wc -l <"$src" | tr -d ' ')
                [[ $line_count -gt 20 ]] && echo "    ... ($line_count lines total)"
                echo "    ────────────────────────────────────────"
            fi
        fi
    else
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        SYNCED_FILES+=("$src")
        echo "  $src"
    fi
}

sync_dir() {
    local src="$1"
    local dst="$TARGET_REPO/$1"

    if [[ ! -d "$src" ]]; then
        log_warn "Source directory missing: $src"
        return
    fi

    # Build directory-specific rsync excludes from target skip patterns (#2245)
    # Converts patterns like ".claude/roles/*.md" to rsync --exclude "*.md"
    # when syncing the ".claude/roles/" directory
    local -a dir_excludes=()
    local dir_exclude_count=0
    for pattern in ${TARGET_SKIPS[@]+"${TARGET_SKIPS[@]}"}; do
        # Check if pattern starts with this directory
        if [[ "$pattern" == "$src/"* ]]; then
            # Extract the suffix after the directory path
            local suffix="${pattern#$src/}"
            dir_excludes+=("--exclude" "$suffix")
            ((dir_exclude_count++))
        elif [[ "$pattern" == "**/"* ]]; then
            # Recursive glob pattern - always include for directory syncs
            local glob_suffix="${pattern#\*\*/}"
            dir_excludes+=("--exclude" "$glob_suffix")
            ((dir_exclude_count++))
        fi
    done

    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ $dir_exclude_count -gt 0 ]]; then
            echo "  [sync dir] $src/ (with $dir_exclude_count skip pattern(s))"
        else
            echo "  [sync dir] $src/"
        fi
    else
        mkdir -p "$dst"
        # Copy contents, preserving structure
        # Note: dir_excludes must come BEFORE RSYNC_EXCLUDE_ARGS to have higher priority
        rsync -a --delete ${dir_excludes[@]+"${dir_excludes[@]}"} "${RSYNC_EXCLUDE_ARGS[@]}" "$src/" "$dst/"
        SYNCED_FILES+=("$src/")
        if [[ $dir_exclude_count -gt 0 ]]; then
            echo "  $src/ (with $dir_exclude_count skip pattern(s))"
        else
            echo "  $src/"
        fi
    fi
}

# Special handler for .gitignore - preserves project-specific entries
GITIGNORE_MARKER="# === Project-specific (preserved during sync) ==="

sync_gitignore() {
    local src=".gitignore"
    local dst="$TARGET_REPO/.gitignore"

    if [[ ! -e "$src" ]]; then
        log_warn "Source missing: $src"
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ -e "$dst" ]]; then
            if diff -q "$src" "$dst" >/dev/null 2>&1; then
                echo "  [unchanged] .gitignore"
            else
                echo "  [update] .gitignore (preserving project-specific entries)"
            fi
        else
            echo "  [create] .gitignore"
        fi
        return
    fi

    # If target doesn't exist, just copy source
    if [[ ! -e "$dst" ]]; then
        cp "$src" "$dst"
        SYNCED_FILES+=(".gitignore")
        echo "  .gitignore (created)"
        return
    fi

    # Extract project-specific entries from target (everything after marker + 2 description lines)
    # Deduplicate to prevent accumulation across repeated syncs
    local project_entries=""
    if grep -q "^$GITIGNORE_MARKER" "$dst" 2>/dev/null; then
        # Get everything after the marker and its 2 description lines (skip 3 lines total)
        # Use awk to dedupe while preserving order
        project_entries=$(sed -n "/^$GITIGNORE_MARKER/,\$p" "$dst" | tail -n +4 | awk '!seen[$0]++')
    fi

    # Copy template gitignore
    cp "$src" "$dst"
    SYNCED_FILES+=(".gitignore")

    # Append project-specific entries if any exist
    if [[ -n "$project_entries" ]]; then
        echo "$project_entries" >>"$dst"
        echo "  .gitignore (preserved project-specific entries)"
    else
        echo "  .gitignore"
    fi
}

# Read manifest and sync files
MANIFEST="$AI_TEMPLATE_ROOT/.sync_manifest"
if [[ ! -f "$MANIFEST" ]]; then
    log_error "Missing .sync_manifest"
    exit 1
fi

echo ""
log_info "Reading .sync_manifest..."

# Collect exclusions first
declare -a EXCLUSIONS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"                      # Remove comments
    line="${line%"${line##*[![:space:]]}"}" # Trim trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}" # Trim leading whitespace
    [[ -z "$line" ]] && continue
    if [[ "$line" == !* ]]; then
        EXCLUSIONS+=("${line#!}")
    fi
done <"$MANIFEST"

is_excluded() {
    local file="$1"
    for pattern in ${EXCLUSIONS[@]+"${EXCLUSIONS[@]}"}; do
        # Handle **/*.ext patterns - bash [[ == ]] doesn't support ** recursive glob
        # Convert to basename match: **/*.pyc matches foo.pyc, a/foo.pyc, a/b/foo.pyc
        if [[ "$pattern" == "**/"* ]]; then
            local suffix="${pattern#\*\*/}" # Extract suffix after **/
            local basename="${file##*/}"
            # shellcheck disable=SC2053
            if [[ "$basename" == $suffix ]]; then
                return 0
            fi
            continue
        fi
        # shellcheck disable=SC2053
        if [[ "$file" == $pattern ]]; then
            return 0
        fi
    done
    return 1
}

RSYNC_EXCLUDE_ARGS=()
build_rsync_excludes() {
    local pattern trimmed base
    RSYNC_EXCLUDE_ARGS=()
    for pattern in ${EXCLUSIONS[@]+"${EXCLUSIONS[@]}"}; do
        trimmed="${pattern%/}"
        base="${trimmed##*/}"
        if [[ "$pattern" == */* && -n "$base" ]]; then
            if [[ "$pattern" == */ ]]; then
                RSYNC_EXCLUDE_ARGS+=("--exclude" "${base}/")
            else
                RSYNC_EXCLUDE_ARGS+=("--exclude" "$base")
            fi
        fi
        RSYNC_EXCLUDE_ARGS+=("--exclude" "$pattern")
    done
}
build_rsync_excludes

# Read target-side skip patterns from .ai_template_skip (#787)
# These patterns let target repos opt-out of specific files
declare -a TARGET_SKIPS=()
TARGET_SKIP_FILE="$TARGET_REPO/.ai_template_skip"
if [[ -f "$TARGET_SKIP_FILE" ]]; then
    log_info "Reading target skip patterns from .ai_template_skip..."
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"                      # Remove comments
        line="${line%"${line##*[![:space:]]}"}" # Trim trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}" # Trim leading whitespace
        [[ -z "$line" ]] && continue
        TARGET_SKIPS+=("$line")
    done <"$TARGET_SKIP_FILE"
    echo "  ${#TARGET_SKIPS[@]} skip pattern(s) loaded"
fi

is_target_skipped() {
    local file="$1"
    for pattern in ${TARGET_SKIPS[@]+"${TARGET_SKIPS[@]}"}; do
        # Handle **/*.ext patterns
        if [[ "$pattern" == "**/"* ]]; then
            local suffix="${pattern#\*\*/}"
            local basename="${file##*/}"
            # shellcheck disable=SC2053
            if [[ "$basename" == $suffix ]]; then
                return 0
            fi
            continue
        fi
        # shellcheck disable=SC2053
        if [[ "$file" == $pattern ]]; then
            return 0
        fi
    done
    return 1
}

# Check if entry matches any --only pattern (#785)
# Returns 0 (true) if no patterns specified or if entry matches at least one pattern
matches_only_pattern() {
    local entry="$1"
    # If no --only patterns, match everything
    [[ ${#ONLY_PATTERNS[@]} -eq 0 ]] && return 0

    local entry_base="${entry%/}" # Remove trailing slash for directory entries

    for pattern in "${ONLY_PATTERNS[@]}"; do
        # Handle directory pattern matching: looper/ should match looper/ and looper/foo.py
        # but NOT looper.py (sibling file with similar name)
        if [[ "$pattern" == */ ]]; then
            local dir_name="${pattern%/}"
            # Exact match (entry is the directory itself)
            [[ "$entry_base" == "$dir_name" ]] && return 0
            # Entry is under the directory (must have slash after dir name)
            [[ "$entry_base" == "$dir_name/"* ]] && return 0
        elif [[ "$pattern" == */* ]]; then
            # Pattern has path - check path prefix or glob match
            # looper/* should match looper/foo.py
            local pattern_dir="${pattern%/*}"
            local pattern_file="${pattern##*/}"

            # shellcheck disable=SC2053
            if [[ "$entry_base" == "$pattern_dir/"* ]]; then
                # Entry is under pattern's directory
                local entry_file="${entry_base#$pattern_dir/}"
                # shellcheck disable=SC2053
                if [[ "$entry_file" == $pattern_file ]] || [[ "$pattern_file" == "*" ]]; then
                    return 0
                fi
            fi
            # Also try direct glob match
            # shellcheck disable=SC2053
            [[ "$entry_base" == $pattern ]] && return 0
        else
            # Simple pattern (no path separator) - match basename or exact
            local entry_base_name="${entry_base##*/}"
            # shellcheck disable=SC2053
            if [[ "$entry_base_name" == $pattern ]] || [[ "$entry_base" == $pattern ]]; then
                return 0
            fi
        fi
    done
    return 1
}

# Process manifest entries
echo ""
if [[ ${#ONLY_PATTERNS[@]} -gt 0 ]]; then
    log_info "Selective sync mode: filtering to ${#ONLY_PATTERNS[@]} pattern(s)"
    for pattern in "${ONLY_PATTERNS[@]}"; do
        echo "  --only '$pattern'"
    done
fi
log_info "Syncing from manifest..."
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"                      # Remove comments
    line="${line%"${line##*[![:space:]]}"}" # Trim trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}" # Trim leading whitespace
    [[ -z "$line" ]] && continue
    [[ "$line" == !* ]] && continue # Skip exclusions (already processed)

    # Filter by --only patterns if specified (#785)
    if ! matches_only_pattern "$line"; then
        continue
    fi

    # Directory sync (ends with /)
    if [[ "$line" == */ ]]; then
        dir="${line%/}"
        if ! is_excluded "$dir"; then
            if is_target_skipped "$dir" || is_target_skipped "$line"; then
                echo "  [skip] $dir/ (target .ai_template_skip)"
            else
                sync_dir "$dir"
            fi
        fi
        continue
    fi

    # Special handling for .gitignore
    if [[ "$line" == ".gitignore" ]]; then
        if is_target_skipped ".gitignore"; then
            echo "  [skip] .gitignore (target .ai_template_skip)"
        else
            sync_gitignore
        fi
        continue
    fi

    # Glob pattern (contains * or ?)
    if [[ "$line" == *\** || "$line" == *\?* ]]; then
        # Expand glob and sync each file
        glob_matched=false
        for file in $line; do
            [[ -e "$file" ]] || continue
            glob_matched=true
            is_excluded "$file" && continue
            if is_target_skipped "$file"; then
                echo "  [skip] $file (target .ai_template_skip)"
                continue
            fi
            sync_file "$file"
        done
        if [[ "$glob_matched" == "false" ]]; then
            log_warn "Glob pattern '$line' matched no files"
        fi
        continue
    fi

    # Exact file path
    if ! is_excluded "$line"; then
        if is_target_skipped "$line"; then
            echo "  [skip] $line (target .ai_template_skip)"
        else
            sync_file "$line"
        fi
    fi
done <"$MANIFEST"

# Check for obsolete template files that should be deleted from target
echo ""
log_info "Checking for obsolete template files..."
OLD_TEMPLATE_FILES=(
    "ai_template_scripts/gh_discussion.sh"      # Replaced by gh_discussion.py
    "ai_template_scripts/gh_post.sh"            # Replaced by gh_post.py
    "ai_template_scripts/init.sh"               # Renamed to install_dev_tools.sh
    "ai_template_scripts/create_github_apps.py" # Orphaned, never part of template
    "ai_template_scripts/stop_all.sh"           # Removed - just use: touch STOP
    "ai_template_scripts/verify_closure.py"     # Removed - Manager searches intelligently
    "ai_template_scripts/frontpage.sh"          # DashNews-specific, not template
    "run_loop.py"                               # Renamed to looper.py
    "run_loop_context.md"                       # Renamed to looper_context.md
    "tests/test_run_loop.py"                    # Renamed to tests/test_looper.py
    ".claude/rules/postmortems.md"              # Merged into ai_template.md
    "looper/context.py"                         # Replaced by looper/context/ subpackage (#748)
    # Legacy root-level scripts (replaced by ai_template_scripts/ versions)
    "init.sh"                   # Use ai_template_scripts/install_dev_tools.sh
    "setup_labels.sh"           # Use ai_template_scripts/init_labels.sh
    "json_to_text.py"           # Use ai_template_scripts/json_to_text.py
    "code_stats.py"             # Use ai_template_scripts/code_stats.py
    "markdown_to_issues.py"     # Use ai_template_scripts/markdown_to_issues.py
    "frontpage.sh"              # DashNews-specific, not template
    "mail.sh"                   # Legacy mail system removed
    "roadmap_to_issues.sh"      # Legacy roadmap system removed
    "scripts/validate_claim.py" # Moved to ai_template_scripts/validate_claim.py
)

# Legacy directories to remove
OLD_TEMPLATE_DIRS=(
    "mail" # Legacy mail system removed
)

for old_file in "${OLD_TEMPLATE_FILES[@]}"; do
    if [[ -f "$TARGET_REPO/$old_file" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [delete] $old_file (obsolete)"
        else
            rm "$TARGET_REPO/$old_file"
            echo "  [deleted] $old_file (obsolete)"
        fi
    fi
done

for old_dir in "${OLD_TEMPLATE_DIRS[@]}"; do
    if [[ -d "$TARGET_REPO/$old_dir" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [delete] $old_dir/ (obsolete directory)"
        else
            rm -rf "${TARGET_REPO:?}/$old_dir"
            echo "  [deleted] $old_dir/ (obsolete directory)"
        fi
    fi
done

# Optional features sync (#1076)
# Repos opt-in via .ai_template_features file
echo ""
log_info "Processing optional features..."

OPTIONAL_DIR="$AI_TEMPLATE_ROOT/ai_template_scripts/optional"
FEATURES_FILE="$TARGET_REPO/.ai_template_features"

# Discover available features by listing directories under optional/
declare -a AVAILABLE_FEATURES=()
if [[ -d "$OPTIONAL_DIR" ]]; then
    while IFS= read -r -d '' feature_dir; do
        feature_name=$(basename "$feature_dir")
        # Skip README.md and non-directories
        [[ -d "$feature_dir" ]] && AVAILABLE_FEATURES+=("$feature_name")
    done < <(find "$OPTIONAL_DIR" -mindepth 1 -maxdepth 1 -print0 2>/dev/null)
fi

# Parse enabled features from target repo
declare -a ENABLED_FEATURES=()
if [[ -f "$FEATURES_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Strip inline comments and whitespace
        line="${line%%#*}"
        line="${line%"${line##*[![:space:]]}"}"
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" ]] && continue

        # Validate feature name (no path separators or traversal)
        if [[ "$line" == */* || "$line" == *\\* || "$line" == *..* ]]; then
            log_warn "Invalid feature name (contains path separator): $line"
            continue
        fi

        ENABLED_FEATURES+=("$line")
    done <"$FEATURES_FILE"
fi

# Sync enabled features
# Note: ${arr[@]+"${arr[@]}"} syntax required for bash 3.2 (macOS) with set -u
for feature in ${ENABLED_FEATURES[@]+"${ENABLED_FEATURES[@]}"}; do
    src="$OPTIONAL_DIR/$feature"
    dst="$TARGET_REPO/ai_template_scripts/optional/$feature"

    if [[ -d "$src" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [sync optional] $feature/"
        else
            mkdir -p "$(dirname "$dst")"
            rsync -a --delete "$src/" "$dst/"
            echo "  [synced] ai_template_scripts/optional/$feature/"
        fi
    else
        log_warn "Unknown optional feature requested: $feature"
    fi
done

# Delete disabled optional features that exist in target
for feature in ${AVAILABLE_FEATURES[@]+"${AVAILABLE_FEATURES[@]}"}; do
    target_feature="$TARGET_REPO/ai_template_scripts/optional/$feature"
    # Check if feature is NOT in enabled list
    feature_enabled=false
    for enabled in ${ENABLED_FEATURES[@]+"${ENABLED_FEATURES[@]}"}; do
        if [[ "$enabled" == "$feature" ]]; then
            feature_enabled=true
            break
        fi
    done

    if [[ "$feature_enabled" == "false" && -d "$target_feature" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [delete optional] $feature/ (not enabled)"
        else
            rm -rf "${target_feature:?}"
            echo "  [deleted] ai_template_scripts/optional/$feature/ (not enabled)"
        fi
    fi
done

# Clean up empty optional directory
if [[ -d "$TARGET_REPO/ai_template_scripts/optional" ]]; then
    if [[ -z "$(ls -A "$TARGET_REPO/ai_template_scripts/optional" 2>/dev/null)" ]]; then
        if [[ "$DRY_RUN" != "true" ]]; then
            rmdir "$TARGET_REPO/ai_template_scripts/optional" 2>/dev/null || true
        fi
    fi
fi

if [[ ${#ENABLED_FEATURES[@]} -eq 0 ]]; then
    echo "  (no features enabled - create .ai_template_features to opt in)"
fi

# Write version file (commit hash + sync timestamp)
SYNC_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Would write .ai_template_version: $AI_TEMPLATE_VERSION @ $SYNC_TIMESTAMP"
else
    cat >"$TARGET_REPO/.ai_template_version" <<EOF
$AI_TEMPLATE_VERSION_FULL
$SYNC_TIMESTAMP
EOF
    log_ok "Wrote .ai_template_version: $AI_TEMPLATE_VERSION @ $SYNC_TIMESTAMP"
fi

# Ensure required labels exist
echo ""
log_info "Ensuring required labels exist..."

if [[ "$DRY_RUN" == "true" ]]; then
    pushd "$TARGET_REPO" >/dev/null
    "$SCRIPT_DIR/init_labels.sh" --dry-run
    popd >/dev/null
else
    pushd "$TARGET_REPO" >/dev/null
    if ! "$SCRIPT_DIR/init_labels.sh"; then
        log_warn "Label creation encountered errors (see warnings above)"
    else
        log_ok "Labels synced"
    fi
    popd >/dev/null
fi

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry run complete. Run without --dry-run to apply changes."
else
    # Commit and push
    log_info "Committing changes..."
    pushd "$TARGET_REPO" >/dev/null

    git add -A
    if git diff --cached --quiet; then
        log_ok "No changes to commit (already up to date)"
    else
        git commit -m "Sync ai_template $AI_TEMPLATE_VERSION"
        log_ok "Committed sync changes"

        if [[ "$NO_PUSH" == "true" ]]; then
            log_info "Skipping push (--no-push specified)"
        else
            log_info "Pushing to remote..."
            git push || {
                log_error "Push failed. Push manually."
                popd >/dev/null
                exit 1
            }
            log_ok "Pushed to remote"
        fi
    fi

    # Install git hooks (respect core.hooksPath and worktrees)
    HOOKS_DIR=$(git rev-parse --git-path hooks 2>/dev/null || true)
    if [[ -z "$HOOKS_DIR" ]]; then
        log_error "Failed to resolve hooks directory"
        popd >/dev/null
        exit 1
    fi
    if [[ "$HOOKS_DIR" != /* ]]; then
        HOOKS_DIR="$TARGET_REPO/$HOOKS_DIR"
    fi
    mkdir -p "$HOOKS_DIR"

    if [[ -f "$TARGET_REPO/ai_template_scripts/commit-msg-hook.sh" ]]; then
        log_info "Installing commit-msg hook..."
        cp "$TARGET_REPO/ai_template_scripts/commit-msg-hook.sh" "$HOOKS_DIR/commit-msg"
        chmod +x "$HOOKS_DIR/commit-msg"
        log_ok "Installed commit-msg hook"
    fi

    if [[ -f "$TARGET_REPO/ai_template_scripts/pre-commit-hook.sh" ]]; then
        log_info "Installing pre-commit hook..."
        cp "$TARGET_REPO/ai_template_scripts/pre-commit-hook.sh" "$HOOKS_DIR/pre-commit"
        chmod +x "$HOOKS_DIR/pre-commit"
        log_ok "Installed pre-commit hook"
    fi

    if [[ -f "$TARGET_REPO/ai_template_scripts/post-commit-hook.sh" ]]; then
        log_info "Installing post-commit hook..."
        cp "$TARGET_REPO/ai_template_scripts/post-commit-hook.sh" "$HOOKS_DIR/post-commit"
        chmod +x "$HOOKS_DIR/post-commit"
        log_ok "Installed post-commit hook"
    fi

    # Validate author in manifests (warning only)
    for manifest in Cargo.toml pyproject.toml; do
        if [[ -f "$TARGET_REPO/$manifest" ]]; then
            if ! grep -qiE 'andrew.*yates|andrewdyates' "$TARGET_REPO/$manifest" 2>/dev/null; then
                log_warn "$manifest missing author (Andrew Yates / andrewdyates)"
            fi
        fi
    done

    # Validate synced scripts work
    echo ""
    log_info "Validating synced scripts..."
    VALIDATION_FAILED=false

    # Check looper.py can at least import (catches broken stubs)
    if [[ -f "$TARGET_REPO/looper.py" ]]; then
        validation_output=$(python3 -c "import sys; sys.path.insert(0, '$TARGET_REPO'); exec(open('$TARGET_REPO/looper.py').read().split('if __name__')[0])" 2>&1) || {
            log_error "looper.py is broken - check if looper/ package was synced"
            log_error "Validation output:"
            # shellcheck disable=SC2001
            echo "$validation_output" | sed 's/^/    /'
            VALIDATION_FAILED=true
        }
        if [[ "$VALIDATION_FAILED" != "true" ]]; then
            echo "  ✓ looper.py syntax OK"
        fi
    fi

    # Check cargo wrapper is executable
    if [[ -f "$TARGET_REPO/ai_template_scripts/bin/cargo" ]]; then
        if [[ -x "$TARGET_REPO/ai_template_scripts/bin/cargo" ]]; then
            echo "  ✓ cargo wrapper executable"
        else
            log_warn "cargo wrapper not executable - fixing"
            chmod +x "$TARGET_REPO/ai_template_scripts/bin/cargo"
        fi
    fi

    # Check gh wrapper is executable
    if [[ -f "$TARGET_REPO/ai_template_scripts/bin/gh" ]]; then
        if [[ -x "$TARGET_REPO/ai_template_scripts/bin/gh" ]]; then
            echo "  ✓ gh wrapper executable"
        else
            log_warn "gh wrapper not executable - fixing"
            chmod +x "$TARGET_REPO/ai_template_scripts/bin/gh"
        fi
    fi

    if [[ "$VALIDATION_FAILED" == "true" ]]; then
        log_error "Script validation failed - sync may be incomplete"
        exit 1
    fi
    log_ok "Script validation passed"

    # Run alignment audit
    echo ""
    log_info "Running alignment audit..."
    if [[ -f "$TARGET_REPO/ai_template_scripts/audit_alignment.sh" ]]; then
        AUDIT_ARGS=""
        [[ "$CLEAN" == "true" ]] && AUDIT_ARGS="--clean"
        if ! "$TARGET_REPO/ai_template_scripts/audit_alignment.sh" $AUDIT_ARGS; then
            log_warn "Alignment audit found issues - review output above"
        fi
    fi

    # Run post-sync hook if exists (#2214)
    POST_SYNC_HOOK="$TARGET_REPO/.ai_template_hooks/post-sync.sh"
    if [[ -f "$POST_SYNC_HOOK" ]]; then
        echo ""
        log_info "Running post-sync hook..."
        if [[ -x "$POST_SYNC_HOOK" ]]; then
            # Export environment variables for the hook
            export AI_TEMPLATE_VERSION="$AI_TEMPLATE_VERSION_FULL"
            # Note: Arrays can't be exported in bash; use SYNCED_FILES_STR
            SYNCED_FILES_STR="${SYNCED_FILES[*]}"
            export SYNCED_FILES_STR

            if "$POST_SYNC_HOOK"; then
                log_ok "Post-sync hook completed"
            else
                log_warn "Post-sync hook exited with non-zero status"
            fi
        else
            log_warn "Post-sync hook exists but is not executable: $POST_SYNC_HOOK"
            log_warn "Run: chmod +x $POST_SYNC_HOOK"
        fi
    fi

    popd >/dev/null
    log_ok "Sync complete!"
fi

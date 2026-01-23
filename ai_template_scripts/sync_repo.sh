#!/usr/bin/env bash
# sync_repo.sh - Sync template files to a target repo
#
# CANONICAL SOURCE: ayates_dbx/ai_template
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

# Must run from ai_template root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$AI_TEMPLATE_ROOT"

[[ -f "CLAUDE.md" ]] || { log_error "Run from ai_template root"; exit 1; }

# Parse args
TARGET_REPO=""
DRY_RUN=false
NO_PUSH=false
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --no-push) NO_PUSH=true ;;
        --clean) CLEAN=true ;;
        -*) log_error "Unknown option: $arg"; exit 1 ;;
        *) TARGET_REPO="$arg" ;;
    esac
done

if [[ -z "$TARGET_REPO" ]]; then
    echo "Usage: $0 /path/to/target_repo [--dry-run] [--no-push] [--clean]"
    exit 1
fi

# Resolve to absolute path
TARGET_REPO="$(cd "$TARGET_REPO" && pwd)"

[[ -d "$TARGET_REPO/.git" ]] || { log_error "Target is not a git repo: $TARGET_REPO"; exit 1; }

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

# Check target repo state and pull if needed
if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    log_info "Checking target repo git state..."
    pushd "$TARGET_REPO" > /dev/null

    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_error "Target repo has uncommitted changes. Commit first (never stash)."
        popd > /dev/null
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
            git pull --rebase || { log_error "Pull failed. Resolve manually."; popd > /dev/null; exit 1; }
        elif [[ "$REMOTE" == "$BASE" ]]; then
            log_info "Local is ahead of remote (will push after sync)"
        else
            log_error "Local and remote have diverged. Resolve manually."
            popd > /dev/null
            exit 1
        fi
    fi

    popd > /dev/null
fi

# Files and directories to sync
# These are template infrastructure files that should be identical across repos

sync_file() {
    local src="$1"
    local dst="$TARGET_REPO/$1"

    if [[ ! -e "$src" ]]; then
        log_warn "Source missing: $src"
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ -e "$dst" ]]; then
            if diff -q "$src" "$dst" > /dev/null 2>&1; then
                echo "  [unchanged] $src"
            else
                echo "  [update] $src"
            fi
        else
            echo "  [create] $src"
        fi
    else
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
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

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [sync dir] $src/"
    else
        mkdir -p "$dst"
        # Copy contents, preserving structure
        rsync -a --delete "$src/" "$dst/"
        echo "  $src/"
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
            if diff -q "$src" "$dst" > /dev/null 2>&1; then
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
        echo "  .gitignore (created)"
        return
    fi

    # Extract project-specific entries from target (everything after marker + 2 description lines)
    local project_entries=""
    if grep -q "^$GITIGNORE_MARKER" "$dst" 2>/dev/null; then
        # Get everything after the marker and its 2 description lines (skip 3 lines total)
        project_entries=$(sed -n "/^$GITIGNORE_MARKER/,\$p" "$dst" | tail -n +4)
    fi

    # Copy template gitignore
    cp "$src" "$dst"

    # Append project-specific entries if any exist
    if [[ -n "$project_entries" ]]; then
        echo "$project_entries" >> "$dst"
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
declare -a EXCLUSIONS
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"  # Remove comments
    line="${line%"${line##*[![:space:]]}"}"  # Trim trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"  # Trim leading whitespace
    [[ -z "$line" ]] && continue
    if [[ "$line" == !* ]]; then
        EXCLUSIONS+=("${line#!}")
    fi
done < "$MANIFEST"

is_excluded() {
    local file="$1"
    for pattern in "${EXCLUSIONS[@]}"; do
        # shellcheck disable=SC2053
        if [[ "$file" == $pattern ]]; then
            return 0
        fi
    done
    return 1
}

# Process manifest entries
echo ""
log_info "Syncing from manifest..."
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"  # Remove comments
    line="${line%"${line##*[![:space:]]}"}"  # Trim trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"  # Trim leading whitespace
    [[ -z "$line" ]] && continue
    [[ "$line" == !* ]] && continue  # Skip exclusions (already processed)

    # Directory sync (ends with /)
    if [[ "$line" == */ ]]; then
        dir="${line%/}"
        if ! is_excluded "$dir"; then
            sync_dir "$dir"
        fi
        continue
    fi

    # Special handling for .gitignore
    if [[ "$line" == ".gitignore" ]]; then
        sync_gitignore
        continue
    fi

    # Glob pattern (contains * or ?)
    if [[ "$line" == *\** || "$line" == *\?* ]]; then
        # Expand glob and sync each file
        for file in $line; do
            [[ -e "$file" ]] || continue
            is_excluded "$file" && continue
            sync_file "$file"
        done
        continue
    fi

    # Exact file path
    if ! is_excluded "$line"; then
        sync_file "$line"
    fi
done < "$MANIFEST"

# Check for obsolete template files that should be deleted from target
echo ""
log_info "Checking for obsolete template files..."
OLD_TEMPLATE_FILES=(
    "ai_template_scripts/gh_discussion.sh"  # Replaced by gh_discussion.py
    "ai_template_scripts/gh_post.sh"        # Replaced by gh_post.py
    "ai_template_scripts/init.sh"           # Renamed to install_dev_tools.sh
    "ai_template_scripts/create_github_apps.py"  # Orphaned, never part of template
    "ai_template_scripts/stop_all.sh"       # Removed - just use: touch STOP
    "ai_template_scripts/verify_closure.py" # Removed - Manager searches intelligently
    "run_loop.py"                           # Renamed to looper.py
    "run_loop_context.md"                   # Renamed to looper_context.md
    "tests/test_run_loop.py"                # Renamed to tests/test_looper.py
    ".claude/rules/postmortems.md"          # Merged into ai_template.md
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

# Write version file (commit hash + sync timestamp)
SYNC_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Would write .ai_template_version: $AI_TEMPLATE_VERSION @ $SYNC_TIMESTAMP"
else
    cat > "$TARGET_REPO/.ai_template_version" <<EOF
$AI_TEMPLATE_VERSION_FULL
$SYNC_TIMESTAMP
EOF
    log_ok "Wrote .ai_template_version: $AI_TEMPLATE_VERSION @ $SYNC_TIMESTAMP"
fi

# Ensure required labels exist
echo ""
log_info "Ensuring required labels exist..."
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [would create] needs-review, do-audit, in-progress, blocked, mail, P0-P3"
else
    pushd "$TARGET_REPO" > /dev/null
    gh label create needs-review --color c5def5 --description "passed self-audit, needs manager review" 2>/dev/null || true
    gh label create do-audit --color fbca04 --description "self-audit required before review" 2>/dev/null || true
    gh label create in-progress --color 5319e7 --description "in progress" 2>/dev/null || true
    gh label create blocked --color 000000 --description "blocked" 2>/dev/null || true
    gh label create mail --color 1d76db --description "Inter-project mail message" 2>/dev/null || true
    gh label create P0 --color b60205 --description "critical" 2>/dev/null || true
    gh label create P1 --color d93f0b --description "high" 2>/dev/null || true
    gh label create P2 --color fbca04 --description "medium" 2>/dev/null || true
    gh label create P3 --color 0e8a16 --description "low" 2>/dev/null || true
    popd > /dev/null
    log_ok "Labels synced"
fi

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry run complete. Run without --dry-run to apply changes."
else
    # Commit and push
    log_info "Committing changes..."
    pushd "$TARGET_REPO" > /dev/null

    git add -A
    if git diff --cached --quiet; then
        log_ok "No changes to commit (already up to date)"
    else
        git commit -m "Sync ai_template $AI_TEMPLATE_VERSION

Co-Authored-By: Claude <noreply@anthropic.com>"
        log_ok "Committed sync changes"

        if [[ "$NO_PUSH" == "true" ]]; then
            log_info "Skipping push (--no-push specified)"
        else
            log_info "Pushing to remote..."
            git push || { log_error "Push failed. Push manually."; popd > /dev/null; exit 1; }
            log_ok "Pushed to remote"
        fi
    fi

    # Install git hooks
    mkdir -p "$TARGET_REPO/.git/hooks"

    if [[ -f "$TARGET_REPO/ai_template_scripts/commit-msg-hook.sh" ]]; then
        log_info "Installing commit-msg hook..."
        cp "$TARGET_REPO/ai_template_scripts/commit-msg-hook.sh" "$TARGET_REPO/.git/hooks/commit-msg"
        chmod +x "$TARGET_REPO/.git/hooks/commit-msg"
        log_ok "Installed commit-msg hook"
    fi

    if [[ -f "$TARGET_REPO/ai_template_scripts/pre-commit-hook.sh" ]]; then
        log_info "Installing pre-commit hook..."
        cp "$TARGET_REPO/ai_template_scripts/pre-commit-hook.sh" "$TARGET_REPO/.git/hooks/pre-commit"
        chmod +x "$TARGET_REPO/.git/hooks/pre-commit"
        log_ok "Installed pre-commit hook"
    fi

    if [[ -f "$TARGET_REPO/ai_template_scripts/post-commit-hook.sh" ]]; then
        log_info "Installing post-commit hook..."
        cp "$TARGET_REPO/ai_template_scripts/post-commit-hook.sh" "$TARGET_REPO/.git/hooks/post-commit"
        chmod +x "$TARGET_REPO/.git/hooks/post-commit"
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
        if python3 -c "import sys; sys.path.insert(0, '$TARGET_REPO'); exec(open('$TARGET_REPO/looper.py').read().split('if __name__')[0])" 2>/dev/null; then
            echo "  ✓ looper.py syntax OK"
        else
            log_error "looper.py is broken - check if looper/ package was synced"
            VALIDATION_FAILED=true
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

    popd > /dev/null
    log_ok "Sync complete!"
fi

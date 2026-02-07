#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# self_sync.sh - Sync this repo from remote ai_template without a local sibling
#
# CANONICAL SOURCE: dropbox-ai-prototypes/ai_template
# DO NOT EDIT in other repos - file issues to ai_template for changes.
#
# Shallow-clones ai_template from GitHub into a temp directory, then runs
# sync_repo.sh against the current working directory. This allows any repo
# to sync itself even when ~/ai_template/ doesn't exist locally.
#
# Usage:
#   ./ai_template_scripts/self_sync.sh
#   ./ai_template_scripts/self_sync.sh --dry-run
#   ./ai_template_scripts/self_sync.sh --no-push
#   ./ai_template_scripts/self_sync.sh --only looper/
#
# All arguments are passed through to sync_repo.sh.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[self_sync]${NC} $1"; }
log_ok() { echo -e "${GREEN}[self_sync]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[self_sync]${NC} $1" >&2; }
log_error() { echo -e "${RED}[self_sync]${NC} $1" >&2; }

# --- Safety: refuse to run inside ai_template itself ---

# Check by directory basename
repo_basename="$(basename "$PWD")"
if [[ "$repo_basename" == "ai_template" ]]; then
    log_error "Cannot self-sync ai_template itself (detected by basename)."
    log_error "ai_template IS the source. Use sync_repo.sh to sync TO other repos."
    exit 1
fi

# Check by git remote
if git remote get-url origin 2>/dev/null | grep -qE '[:/]ai_template(\.git)?$'; then
    log_error "Cannot self-sync ai_template itself (detected by git remote)."
    log_error "ai_template IS the source. Use sync_repo.sh to sync TO other repos."
    exit 1
fi

# --- Determine GitHub org ---

GITHUB_ORG="dropbox-ai-prototypes"

# Read from ait_identity.toml if it exists
if [[ -f "ait_identity.toml" ]]; then
    # Extract github_org value (simple grep, no toml parser needed)
    org_line=$(grep -E '^\s*github_org\s*=' ait_identity.toml 2>/dev/null || true)
    if [[ -n "$org_line" ]]; then
        # Strip key, quotes, and whitespace
        extracted=$(echo "$org_line" | sed 's/.*=\s*//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//; s/\s*$//')
        if [[ -n "$extracted" ]]; then
            GITHUB_ORG="$extracted"
        fi
    fi
fi

REPO_URL="https://github.com/${GITHUB_ORG}/ai_template.git"
log_info "Source: ${REPO_URL}"
log_info "Target: ${PWD}"

# --- Shallow clone to temp directory ---

TMPDIR_PATH=""
cleanup() {
    if [[ -n "$TMPDIR_PATH" && -d "$TMPDIR_PATH" ]]; then
        rm -rf "$TMPDIR_PATH"
    fi
}
trap cleanup EXIT

TMPDIR_PATH="$(mktemp -d)"
log_info "Cloning ai_template (shallow)..."

if ! git clone --depth 1 "$REPO_URL" "$TMPDIR_PATH/ai_template" 2>&1; then
    log_error "Failed to clone ${REPO_URL}"
    log_error "Check network connectivity and GitHub authentication."
    exit 1
fi

# --- Run sync_repo.sh from the cloned copy ---

SYNC_SCRIPT="$TMPDIR_PATH/ai_template/ai_template_scripts/sync_repo.sh"
if [[ ! -x "$SYNC_SCRIPT" ]]; then
    log_error "sync_repo.sh not found or not executable in cloned ai_template"
    exit 1
fi

log_info "Running sync_repo.sh..."
"$SYNC_SCRIPT" "$PWD" "$@"
log_ok "Self-sync complete."

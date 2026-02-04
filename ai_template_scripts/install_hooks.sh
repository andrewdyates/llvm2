#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# install_hooks.sh - Install pre-commit framework and hooks
#
# This script installs the pre-commit framework and sets up git hooks.
# Run from repository root after cloning.
#
# Usage:
#   ./ai_template_scripts/install_hooks.sh

set -euo pipefail

# Version function
version() {
    echo "install_hooks.sh $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(date +%Y-%m-%d))"
    exit 0
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install pre-commit framework and git hooks."
    echo ""
    echo "Installs pre-commit if not present and configures git hooks"
    echo "using .pre-commit-config.yaml. Run from repository root."
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0           # Install pre-commit and hooks"
    exit 0
}

# Parse args first
case "${1:-}" in
    --version) version ;;
    -h|--help) usage ;;
    "") ;; # No option is valid
    -*) echo "ERROR: Unknown option: $1"; exit 1 ;;
    *) echo "ERROR: Unexpected argument: $1"; exit 1 ;;
esac

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

# Find repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    log_error "Not in a git repository"
    exit 1
}
cd "$REPO_ROOT"

# Check for .pre-commit-config.yaml
if [[ ! -f ".pre-commit-config.yaml" ]]; then
    log_error "No .pre-commit-config.yaml found in $REPO_ROOT"
    log_info "Run sync_repo.sh from ai_template to sync config files"
    exit 1
fi

# Check if pre-commit is installed
if ! command -v pre-commit &>/dev/null; then
    log_warn "pre-commit not found, attempting to install..."

    # Try pip first
    if command -v pip3 &>/dev/null; then
        log_info "Installing pre-commit via pip3..."
        pip3 install pre-commit
    elif command -v pip &>/dev/null; then
        log_info "Installing pre-commit via pip..."
        pip install pre-commit
    elif command -v brew &>/dev/null; then
        log_info "Installing pre-commit via Homebrew..."
        brew install pre-commit
    else
        log_error "Could not install pre-commit. Please install manually:"
        log_error "  pip install pre-commit"
        log_error "  OR"
        log_error "  brew install pre-commit"
        exit 1
    fi

    # Verify installation
    if ! command -v pre-commit &>/dev/null; then
        log_error "pre-commit installation failed"
        exit 1
    fi
fi

log_ok "pre-commit $(pre-commit --version) found"

# Install git hooks
log_info "Installing git hooks..."
pre-commit install --install-hooks

# Also install commit-msg hook
log_info "Installing commit-msg hook..."
pre-commit install --hook-type commit-msg

# Install pre-push hook for claim validation
log_info "Installing pre-push hook..."
pre-commit install --hook-type pre-push

log_ok "Git hooks installed successfully"

# Install post-commit hook (standalone, not managed by pre-commit)
# This hook handles Claims/Unclaims/Reopens/Fixes keywords for issue label management
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POST_COMMIT_SCRIPT="$SCRIPT_DIR/post-commit-hook.sh"
# Use git rev-parse to handle worktrees correctly (same as audit_alignment.sh)
HOOKS_DIR="$(git rev-parse --git-path hooks 2>/dev/null || echo ".git/hooks")"
POST_COMMIT_TARGET="$HOOKS_DIR/post-commit"

if [[ -f "$POST_COMMIT_SCRIPT" ]]; then
    log_info "Installing post-commit hook..."
    if cp "$POST_COMMIT_SCRIPT" "$POST_COMMIT_TARGET" && chmod +x "$POST_COMMIT_TARGET"; then
        log_ok "post-commit hook installed"
    else
        log_error "Failed to install post-commit hook"
        exit 1
    fi
else
    log_error "post-commit-hook.sh not found at $POST_COMMIT_SCRIPT"
    log_info "Run sync_repo.sh to sync ai_template_scripts/"
    # Don't exit - other hooks are still useful
fi

# Install post-merge hook (standalone, auto-bumps dependencies after git pull)
# Configured via [auto_bump] section in cargo_wrapper.toml
POST_MERGE_SCRIPT="$SCRIPT_DIR/post-merge-hook.sh"
POST_MERGE_TARGET="$HOOKS_DIR/post-merge"

if [[ -f "$POST_MERGE_SCRIPT" ]]; then
    log_info "Installing post-merge hook..."
    if cp "$POST_MERGE_SCRIPT" "$POST_MERGE_TARGET" && chmod +x "$POST_MERGE_TARGET"; then
        log_ok "post-merge hook installed"
    else
        log_warn "Failed to install post-merge hook (non-critical)"
        # Don't exit - this hook is optional
    fi
else
    log_info "post-merge-hook.sh not found (optional, skipping)"
fi

# Verify hooks are in place (HOOKS_DIR already set above for worktree support)
# Core hooks are required; optional hooks are informational
HOOKS_MISSING=()
for hook in pre-commit commit-msg pre-push post-commit; do
    if [[ ! -f "$HOOKS_DIR/$hook" ]]; then
        HOOKS_MISSING+=("$hook")
    fi
done

if [[ ${#HOOKS_MISSING[@]} -eq 0 ]]; then
    log_ok "All core hooks verified:"
    log_info "  $HOOKS_DIR/pre-commit"
    log_info "  $HOOKS_DIR/commit-msg"
    log_info "  $HOOKS_DIR/pre-push (claim validation)"
    log_info "  $HOOKS_DIR/post-commit (issue keywords)"
    if [[ -f "$HOOKS_DIR/post-merge" ]]; then
        log_info "  $HOOKS_DIR/post-merge (dependency auto-bump)"
    fi
else
    log_warn "Missing hooks: ${HOOKS_MISSING[*]}"
    log_info "Run this script again after ensuring ai_template_scripts/ is synced"
fi

echo ""
log_info "To run hooks manually: pre-commit run --all-files"
log_info "To update hook versions: pre-commit autoupdate"

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

log_ok "Git hooks installed successfully"

# Verify hooks are in place
if [[ -f ".git/hooks/pre-commit" ]] && [[ -f ".git/hooks/commit-msg" ]]; then
    log_ok "Hooks verified:"
    log_info "  .git/hooks/pre-commit"
    log_info "  .git/hooks/commit-msg"
else
    log_warn "Some hooks may not be properly installed"
fi

echo ""
log_info "To run hooks manually: pre-commit run --all-files"
log_info "To update hook versions: pre-commit autoupdate"

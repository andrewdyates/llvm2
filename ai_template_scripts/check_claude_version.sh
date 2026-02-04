#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
#
# CANONICAL SOURCE: dropbox-ai-prototypes/ai_template
# DO NOT EDIT in other repos - file issues to ai_template for changes.
#
# check_claude_version.sh - Verify installed Claude Code CLI matches pinned version
#
# Usage:
#   ./ai_template_scripts/check_claude_version.sh [--strict]
#
# Options:
#   --strict    Exit with error if version mismatch (default: warning only)
#
# Reads .claude-version from repo root to determine required version.
# Returns:
#   0 - Version matches or no .claude-version file
#   1 - Version mismatch (--strict mode)
#   2 - Claude CLI not installed

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Find repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../.claude-version" ]]; then
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
elif [[ -f "./.claude-version" ]]; then
    REPO_ROOT="."
else
    # No version pinning configured
    exit 0
fi

VERSION_FILE="$REPO_ROOT/.claude-version"
STRICT=false

for arg in "$@"; do
    case "$arg" in
    --strict) STRICT=true ;;
    -h | --help)
        echo "Usage: $0 [--strict]"
        echo ""
        echo "Verify installed Claude Code CLI matches pinned version in .claude-version"
        echo ""
        echo "Options:"
        echo "  --strict    Exit with error if version mismatch"
        exit 0
        ;;
    esac
done

# Read pinned version
if [[ ! -f "$VERSION_FILE" ]]; then
    exit 0
fi

PINNED_VERSION=$(head -1 "$VERSION_FILE" | tr -d '[:space:]')
if [[ -z "$PINNED_VERSION" ]]; then
    log_warn ".claude-version is empty"
    exit 0
fi

# Get installed version
if ! command -v claude &>/dev/null; then
    log_error "Claude Code CLI not installed"
    echo "Install with: npm install -g @anthropic-ai/claude-code@$PINNED_VERSION"
    exit 2
fi

INSTALLED_VERSION=$(claude --version 2>/dev/null | awk '{print $1}' || echo "unknown")

# Compare versions
if [[ "$INSTALLED_VERSION" == "$PINNED_VERSION" ]]; then
    log_ok "Claude Code CLI version: $INSTALLED_VERSION (matches pinned)"
    exit 0
fi

# Version mismatch
if [[ "$STRICT" == "true" ]]; then
    log_error "Claude Code CLI version mismatch!"
    echo "  Pinned:    $PINNED_VERSION"
    echo "  Installed: $INSTALLED_VERSION"
    echo ""
    echo "Fix with: npm install -g @anthropic-ai/claude-code@$PINNED_VERSION"
    exit 1
else
    log_warn "Claude Code CLI version mismatch"
    echo "  Pinned:    $PINNED_VERSION"
    echo "  Installed: $INSTALLED_VERSION"
    echo ""
    echo "Fix with: npm install -g @anthropic-ai/claude-code@$PINNED_VERSION"
    exit 0
fi

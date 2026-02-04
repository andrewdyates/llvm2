#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# check_path_order.sh - Verify ai_template wrappers have PATH precedence
#
# Checks that ai_template_scripts/bin comes BEFORE system paths like Homebrew.
# This is critical for gh/cargo wrappers to work in Codex and other AI tools
# that spawn their own shells.
#
# Usage:
#   ./ai_template_scripts/check_path_order.sh
#   ./ai_template_scripts/check_path_order.sh --quiet  # Exit code only
#
# Exit codes:
#   0 - PATH ordering is correct
#   1 - PATH ordering issue detected
#
# Part of #1860

set -euo pipefail

usage() {
    cat <<'EOF'
check_path_order.sh - Verify ai_template wrappers have PATH precedence

Usage:
  ./ai_template_scripts/check_path_order.sh          # Show results
  ./ai_template_scripts/check_path_order.sh --quiet  # Exit code only
  ./ai_template_scripts/check_path_order.sh --help   # Show this help

Exit codes:
  0 - PATH ordering is correct
  1 - PATH ordering issue detected

See README.md "Shell Configuration" section for setup instructions.
EOF
}

QUIET=false
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    -q|--quiet) QUIET=true ;;
esac

# Find the project root (script is in ai_template_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AIT_BIN="$PROJECT_ROOT/ai_template_scripts/bin"

# Check if ai_template_scripts/bin is on PATH
if [[ ":$PATH:" != *":$AIT_BIN:"* ]]; then
    $QUIET || echo "ERROR: ai_template_scripts/bin not on PATH"
    $QUIET || echo "  Expected: $AIT_BIN"
    $QUIET || echo "  Current PATH: $PATH"
    exit 1
fi

# Check if gh wrapper takes precedence over system gh
GH_PATH=$(command -v gh 2>/dev/null || true)
if [[ -z "$GH_PATH" ]]; then
    $QUIET || echo "ERROR: gh not found on PATH"
    exit 1
fi

if [[ "$GH_PATH" == "$AIT_BIN/gh" ]]; then
    $QUIET || echo "OK: ai_template gh wrapper has precedence ($GH_PATH)"
else
    $QUIET || echo "ERROR: System gh takes precedence over ai_template wrapper"
    $QUIET || echo "  Active: $GH_PATH"
    $QUIET || echo "  Expected: $AIT_BIN/gh"
    $QUIET || echo ""
    $QUIET || echo "  Fix: Add this line AFTER Homebrew setup in ~/.zprofile:"
    # Use full path from PROJECT_ROOT, not just basename (may be nested: ~/repos/project/)
    $QUIET || echo "    export PATH=\"$AIT_BIN:\$PATH\""
    exit 1
fi

# Check cargo wrapper if it exists
CARGO_PATH=$(command -v cargo 2>/dev/null || true)
if [[ -n "$CARGO_PATH" && -f "$AIT_BIN/cargo" ]]; then
    if [[ "$CARGO_PATH" == "$AIT_BIN/cargo" ]]; then
        $QUIET || echo "OK: ai_template cargo wrapper has precedence ($CARGO_PATH)"
    else
        $QUIET || echo "WARNING: System cargo takes precedence over ai_template wrapper"
        $QUIET || echo "  Active: $CARGO_PATH"
        $QUIET || echo "  Expected: $AIT_BIN/cargo"
        # Don't exit 1 for cargo - it's less critical than gh
    fi
fi

exit 0

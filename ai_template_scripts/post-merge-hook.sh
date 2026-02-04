#!/bin/bash
# post-merge-hook.sh - Git post-merge hook for dependency auto-bump
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# License: Apache-2.0
#
# PURPOSE: Automatically bump git dependencies after git pull/merge.
# CALLED BY: Git (installed to .git/hooks/post-merge by install_hooks.sh)
#
# Configuration (cargo_wrapper.toml):
#   [auto_bump]
#   repos = ["https://github.com/dropbox-ai-prototypes/z4", "https://github.com/dropbox-ai-prototypes/tMIR"]
#
# Skip behavior:
#   - Skips if "Blocked by" found in any Cargo.toml
#   - Skips if no [auto_bump] section configured
#   - Skips if bump_git_dep_rev.sh not found
#
# Issue: #1994
# Verified: ba9f1775 | 2026-02-02 | [M]506

set -euo pipefail

# Find repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$REPO_ROOT"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "[post-merge] $1" >&2; }
log_ok() { echo -e "${GREEN}[post-merge]${NC} $1" >&2; }
log_skip() { echo -e "${YELLOW}[post-merge]${NC} $1" >&2; }

# Check for blocker in Cargo.toml files
if grep -rq "Blocked by\|BLOCKED:" Cargo.toml */Cargo.toml 2>/dev/null; then
    log_skip "Dependency bump BLOCKED (found 'Blocked by' in Cargo.toml)"
    exit 0
fi

# Find config file (cargo_wrapper.toml or .cargo_wrapper.toml)
CONFIG_FILE=""
for candidate in "cargo_wrapper.toml" ".cargo_wrapper.toml"; do
    if [[ -f "$REPO_ROOT/$candidate" ]]; then
        CONFIG_FILE="$REPO_ROOT/$candidate"
        break
    fi
done

if [[ -z "$CONFIG_FILE" ]]; then
    # No config file - skip silently (not all repos need auto-bump)
    exit 0
fi

# Check for [auto_bump] section
if ! grep -q '^\[auto_bump\]' "$CONFIG_FILE" 2>/dev/null; then
    # No auto_bump config - skip silently
    exit 0
fi

# Find bump script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUMP_SCRIPT="$SCRIPT_DIR/bump_git_dep_rev.sh"

if [[ ! -x "$BUMP_SCRIPT" ]]; then
    log_skip "bump_git_dep_rev.sh not found at $BUMP_SCRIPT"
    exit 0
fi

# Extract repos from config using Python (handles TOML properly)
if ! command -v python3 &>/dev/null; then
    log_skip "python3 not available for config parsing"
    exit 0
fi

REPOS=$(python3 -c "
import sys
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit(0)  # Can't parse TOML, skip silently

try:
    with open('$CONFIG_FILE', 'rb') as f:
        config = tomllib.load(f)
    repos = config.get('auto_bump', {}).get('repos', [])
    for repo in repos:
        print(repo)
except Exception:
    pass
" 2>/dev/null)

if [[ -z "$REPOS" ]]; then
    exit 0
fi

# Bump each configured dependency
log_info "Auto-bumping configured dependencies..."
BUMPED=0

while IFS= read -r repo; do
    if [[ -n "$repo" ]]; then
        log_info "Checking $repo..."
        if "$BUMP_SCRIPT" "$repo" 2>&1 | grep -q "Updated"; then
            ((BUMPED++)) || true
        fi
    fi
done <<<"$REPOS"

if [[ $BUMPED -gt 0 ]]; then
    log_ok "Bumped $BUMPED dependency(ies). Run 'cargo check' to verify."
else
    log_info "Dependencies already up to date."
fi

exit 0

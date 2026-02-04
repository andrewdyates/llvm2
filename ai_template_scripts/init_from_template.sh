#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# init_from_template.sh - Enforce required project setup
#
# Creates GitHub labels and cleans up ai_template-specific files.
# Run once after creating a new project from ai_template.
#
# Usage:
#   ./ai_template_scripts/init_from_template.sh             # Full init
#   ./ai_template_scripts/init_from_template.sh --verify    # Check placeholders
#   ./ai_template_scripts/init_from_template.sh --check-deps # Check dependencies only
#
# Copyright 2026 Dropbox, Inc.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Version function
version() {
    echo "init_from_template.sh $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(date +%Y-%m-%d))"
    exit 0
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Initialize a project from ai_template."
    echo ""
    echo "Removes ai_template-specific files, sets up GitHub labels, installs git hooks,"
    echo "and creates ideas/ directory. Run once after creating a new project from ai_template."
    echo ""
    echo "Options:"
    echo "  --verify       Check for unreplaced placeholders in CLAUDE.md and VISION.md"
    echo "  --check-deps   Check project dependencies only"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0             # Full initialization"
    echo "  $0 --verify    # Check placeholders and dependencies"
    echo "  $0 --check-deps # Check dependencies only"
    exit 0
}

# Parse args first
case "${1:-}" in
    --version) version ;;
    -h|--help) usage ;;
    --check-deps|--verify|"") ;; # Valid options or no option
    -*) echo "ERROR: Unknown option: $1"; exit 1 ;;
    *) echo "ERROR: Unexpected argument: $1"; exit 1 ;;
esac

[[ -f "CLAUDE.md" ]] || { echo "ERROR: Run from project root"; exit 1; }

# SAFETY: Refuse to run in ai_template itself (#2202)
# Note: .ai_template_self directory IS tracked in git and is copied to fresh clones.
# This script is supposed to REMOVE it from clones, so we can't use it as a guard.
# Use repo name check only - fresh clones have different names.
# Fall back to directory name if no remote is configured.
# Note: With pipefail, git remote failure would exit script, so use || true to suppress.
REPO_NAME=$(git remote get-url origin 2>/dev/null | sed 's|.*/||;s|\.git||' || true)
[[ -z "$REPO_NAME" ]] && REPO_NAME=$(basename "$(pwd)")
if [[ "$REPO_NAME" == "ai_template" ]]; then
    echo "ERROR: This script is for projects created FROM ai_template, not ai_template itself."
    echo "       Running in ai_template would delete critical test files."
    exit 1
fi

# --check-deps mode: just check dependencies
if [[ "${1:-}" == "--check-deps" ]]; then
    exec "$SCRIPT_DIR/check_deps.py"
fi

# Test mode: skip slow external calls (GitHub API, pre-commit install)
# Used by test_init_from_template.py to avoid timeouts
TEST_MODE="${INIT_TEST_MODE:-0}"

# --verify mode: check placeholders
if [[ "${1:-}" == "--verify" ]]; then
    VERIFY_FAILED=0

    # Check CLAUDE.md
    if grep -q '<[^>]*>' CLAUDE.md; then
        echo "ERROR: Unreplaced placeholders in CLAUDE.md:"
        grep -n '<[^>]*>' CLAUDE.md
        echo ""
        VERIFY_FAILED=1
    fi

    # Check VISION.md
    if [[ -f "VISION.md" ]] && grep -q '<[^>]*>' VISION.md; then
        echo "ERROR: Unreplaced placeholders in VISION.md:"
        grep -n '<[^>]*>' VISION.md
        echo ""
        VERIFY_FAILED=1
    fi

    if [[ "$VERIFY_FAILED" == "1" ]]; then
        exit 1
    fi
    echo "OK: All placeholders replaced in CLAUDE.md and VISION.md"

    # Also check dependencies
    echo ""
    echo "Checking dependencies..."
    "$SCRIPT_DIR/check_deps.py" || echo "WARNING: Some dependencies missing"
    exit 0
fi

echo "Initializing project from ai_template..."

# Delete ai_template-specific directories
rm -rf .ai_template_self
rm -rf worker_logs

# Delete ai_template-specific files
rm -f docs/plans/PLAN_bot_identity.md

# Delete ai_template-specific post-mortems (keep TEMPLATE.md)
find postmortems -name '*.md' ! -name 'TEMPLATE.md' -delete 2>/dev/null || true

# Delete ai_template-specific tests (keep test_example.py and conftest.py)
find tests -name '*.py' ! -name 'test_example.py' ! -name 'conftest.py' ! -name '__init__.py' -delete 2>/dev/null || true

# Create ideas directory with README (replaces obsolete IDEAS.md)
rm -f IDEAS.md  # Remove obsolete file if present
mkdir -p ideas
cat > ideas/README.md << 'EOF'
# Ideas

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

Future considerations and backlog items. These are NOT actionable tasks.

**For AI:** Do not convert these to issues or work on them unless explicitly asked.
**For humans:** Add ideas freely. Promote to issues when ready to implement.

## How to use

- Add files like `YYYY-MM-DD-idea-name.md`
- Keep them simple - just enough to capture the idea
- When ready to implement, file a GitHub issue and reference the idea

---
EOF

# Create VISION.md template
cat > VISION.md << 'EOF'
# Vision: <project>

> One-sentence mission statement. What does success look like?

## Problem Statement

Why does this project need to exist? What gap does it fill?

## Success Metrics

External benchmarks, not internal vanity metrics.

| Metric | Target | Current | Baseline |
|--------|--------|---------|----------|
| Example | 100% | 50% | 0% |

## Readiness

**Level:** BUILDING

| Level | Meaning |
|-------|---------|
| PLANNED | Not yet started |
| BUILDING | In development, not ready for others to use |
| USABLE | Works, others can depend on it, not feature-complete |
| V1 | Production ready, stable API |
| HOLD | Paused/deprioritized, not archived |

**Lifecycle:** PLANNED → BUILDING → USABLE → V1 (or → HOLD → Archived)

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core | In Progress | - |

## Phases

### Phase 1: Foundation (IN PROGRESS)

**Goal:** Core functionality working.

**Done when:** Explicit, testable criteria.

---

> Updated by Researcher, approved by User.
EOF

# Flush README.md to template
cat > README.md << 'EOF'
# Project Name

One-line description.

## Motivation

Why this project exists. What problem it solves.

## Goal

What success looks like. The end state we're building toward.

## Setup

```bash
# Setup instructions
```

## Usage

```bash
# Usage examples
```
EOF

# Create required GitHub labels for AI workflow
if [[ "$TEST_MODE" == "1" ]]; then
    # In test mode, use --dry-run to skip actual GitHub API calls
    "$SCRIPT_DIR/init_labels.sh" --dry-run || true
else
    if ! "$SCRIPT_DIR/init_labels.sh"; then
        echo "WARNING: Label creation encountered errors"
    fi
fi

# Install git hooks
echo ""
echo "Installing git hooks..."
if [[ "$TEST_MODE" == "1" ]]; then
    # In test mode, skip pre-commit install (downloads hook repos)
    echo "[TEST MODE] Skipping actual hook installation"
elif [[ -x "$SCRIPT_DIR/install_hooks.sh" ]]; then
    "$SCRIPT_DIR/install_hooks.sh" || echo "WARNING: Hook installation failed"
else
    echo "WARNING: install_hooks.sh not found or not executable"
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
if [[ "$TEST_MODE" == "1" ]]; then
    echo "[TEST MODE] Skipping dependency check"
else
    "$SCRIPT_DIR/check_deps.py" || echo "WARNING: Some dependencies missing (install with check_deps.py --fix)"
fi

# Ensure cargo wrapper is executable and lock dir exists
chmod +x "$SCRIPT_DIR/bin/cargo" 2>/dev/null || true
mkdir -p ~/.ait_cargo_lock

# Update user-level Claude settings with recommended values
echo ""
echo "Checking Claude user settings..."
if [[ "$TEST_MODE" == "1" ]]; then
    echo "[TEST MODE] Skipping Claude user settings update"
elif python3 "$SCRIPT_DIR/update_claude_user_settings.py"; then
    : # Success message printed by script
else
    echo "WARNING: Failed to update Claude user settings"
fi

# Check shell PATH configuration for codex/dasher compatibility
echo ""
if [[ "$TEST_MODE" == "1" ]]; then
    echo "[TEST MODE] Skipping shell configuration check"
else
    echo "Checking shell configuration..."
    PROJ_NAME=$(basename "$(pwd)")
    PATH_FILES=("$HOME/.zprofile" "$HOME/.zshenv" "$HOME/.zshrc")
    FOUND_PATH_FILE=""
    for path_file in "${PATH_FILES[@]}"; do
        if [[ -f "$path_file" ]] && grep -q "ai_template_scripts/bin" "$path_file"; then
            FOUND_PATH_FILE="$path_file"
            break
        fi
    done

    if [[ -n "$FOUND_PATH_FILE" ]]; then
        # Check PATH ordering: ai_template_scripts/bin must come BEFORE Homebrew (#1860)
        # The export should be AFTER homebrew eval in the file so it prepends on top
        AIT_LINE=$(grep -n "ai_template_scripts/bin" "$FOUND_PATH_FILE" | head -1 | cut -d: -f1) || AIT_LINE=""
        BREW_LINE=$(grep -n "homebrew" "$FOUND_PATH_FILE" | head -1 | cut -d: -f1) || BREW_LINE=""
        if [[ -n "$BREW_LINE" && -n "$AIT_LINE" && "$AIT_LINE" -lt "$BREW_LINE" ]]; then
            echo "WARNING: Shell PATH order issue in $FOUND_PATH_FILE"
            echo ""
            echo "  ai_template_scripts/bin export (line $AIT_LINE) is BEFORE Homebrew (line $BREW_LINE)."
            echo "  This means Homebrew's /opt/homebrew/bin will shadow our wrappers."
            echo ""
            echo "  Fix: Move the ai_template PATH export to AFTER the Homebrew line:"
            echo "    # After: eval \"\$(/opt/homebrew/bin/brew shellenv)\""
            echo "    export PATH=\"\$HOME/$PROJ_NAME/ai_template_scripts/bin:\$PATH\""
            echo ""
            echo "  Or add to the END of $FOUND_PATH_FILE to ensure wrapper precedence."
        else
            echo "OK: Shell PATH includes ai_template_scripts/bin ($FOUND_PATH_FILE)"
        fi
    else
        ANY_FILE=""
        for path_file in "${PATH_FILES[@]}"; do
            if [[ -f "$path_file" ]]; then
                ANY_FILE="1"
                break
            fi
        done
        if [[ -n "$ANY_FILE" ]]; then
            echo "WARNING: Shell configuration needed for codex/dasher compatibility"
            echo ""
            echo "  AI tools that spawn login shells (codex, dasher) bypass PATH modifications"
            echo "  made at runtime. Add this line AFTER Homebrew setup in ~/.zprofile:"
            echo ""
            echo "    export PATH=\"\$HOME/$PROJ_NAME/ai_template_scripts/bin:\$PATH\""
            echo ""
            echo "  IMPORTANT: Add this AFTER 'eval \"\$(/opt/homebrew/bin/brew shellenv)\"'"
            echo "  so that ai_template wrappers take precedence over Homebrew binaries."
            echo ""
            echo "  Without this, cargo commands may run without serialization (causing OOM),"
            echo "  and gh commands may bypass rate limiting."
            echo "  See README.md 'Shell Configuration' section for details."
        else
            echo "WARNING: ~/.zprofile, ~/.zshenv, and ~/.zshrc not found - shell PATH configuration may be needed"
        fi
    fi
fi

echo ""
echo "Done. Next steps:"
echo "  1. Edit CLAUDE.md (replace placeholders)"
echo "  2. Run: ./ai_template_scripts/init_from_template.sh --verify"
echo "  3. Edit VISION.md and README.md"
echo "  4. Create issues: gh issue create --title \"Task\" --label \"P1\""

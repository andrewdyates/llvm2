#!/usr/bin/env bash
# pre-commit-hook.sh - Validates copyright headers and author metadata
#
# WARNING mode: Warns but allows commit to proceed.
# Run: git commit --no-verify to skip entirely.
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# Colors
YELLOW='\033[1;33m'
NC='\033[0m'

warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

WARNINGS=0

# Get staged files (new or modified) - use newline as delimiter to handle spaces in filenames
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)
OLD_IFS="$IFS"
IFS=$'\n'

# Directory containing header templates (relative to repo root)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
HEADERS_DIR="$REPO_ROOT/ai_template_scripts/headers"

# Map file extension to header template
get_header_template() {
    local ext="$1"
    case "$ext" in
        py)         echo "$HEADERS_DIR/python.txt" ;;
        sh|bash)    echo "$HEADERS_DIR/bash.txt" ;;
        rs)         echo "$HEADERS_DIR/rust.txt" ;;
        js|ts)      echo "$HEADERS_DIR/javascript.txt" ;;
        swift)      echo "$HEADERS_DIR/swift.txt" ;;
        c|cpp|h)    echo "$HEADERS_DIR/rust.txt" ;;  # Same // style as rust
        go)         echo "$HEADERS_DIR/rust.txt" ;;  # Same // style as rust
        java)       echo "$HEADERS_DIR/rust.txt" ;;  # Same // style as rust
        *)          echo "" ;;
    esac
}

# Check and auto-fix copyright headers in source files
check_header() {
    local file="$1"
    local ext="${file##*.}"

    # Only check known source file types
    case "$ext" in
        py|sh|bash|rs|js|ts|swift|c|cpp|h|go|java) ;;
        *) return 0 ;;  # Skip unknown extensions
    esac

    # Check if file contains copyright notice
    if ! head -10 "$file" 2>/dev/null | grep -qi "copyright.*dropbox\|copyright.*andrew.*yates"; then
        local template
        template=$(get_header_template "$ext")

        if [[ -n "$template" && -f "$template" ]]; then
            # Auto-add header, preserving shebang and permissions
            local tmpfile
            tmpfile=$(mktemp)
            local first_line
            first_line=$(head -1 "$file")

            if [[ "$first_line" == "#!"* ]]; then
                # File has shebang - put it first, then header
                {
                    echo "$first_line"
                    cat "$template"
                    echo ""
                    tail -n +2 "$file"
                } > "$tmpfile"
            else
                # No shebang - header goes first
                {
                    cat "$template"
                    echo ""
                    cat "$file"
                } > "$tmpfile"
            fi

            # Preserve original file permissions
            chmod --reference="$file" "$tmpfile" 2>/dev/null || chmod "$(stat -f %Lp "$file")" "$tmpfile" 2>/dev/null || true
            mv "$tmpfile" "$file"

            # Re-stage the file with the header
            git add "$file"

            echo -e "${YELLOW}[AUTO-FIX]${NC} Added copyright header to: $file"
        else
            warn "Missing copyright header (no template for .$ext): $file"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
}

# Check author in package manifests
check_manifest_author() {
    local file="$1"

    if [[ ! -f "$file" ]]; then
        return 0
    fi

    # Valid author patterns: Andrew Yates, andrewdyates
    if ! grep -qiE 'andrew.*yates|andrewdyates' "$file" 2>/dev/null; then
        warn "Missing or incorrect author in $file (should be Andrew Yates or andrewdyates)"
        WARNINGS=$((WARNINGS + 1))
    fi
}

# Check staged source files for headers
for file in $STAGED_FILES; do
    if [[ -f "$file" ]]; then
        check_header "$file"
    fi
done

# Check manifests if staged
if echo "$STAGED_FILES" | grep -q "Cargo.toml"; then
    check_manifest_author "Cargo.toml"
fi

if echo "$STAGED_FILES" | grep -q "pyproject.toml"; then
    check_manifest_author "pyproject.toml"
fi

# Check AppleScript syntax in staged shell files (macOS only)
if [[ "$(uname)" == "Darwin" ]]; then
    LINTER="$REPO_ROOT/ai_template_scripts/lint_applescript.sh"
    if [[ -x "$LINTER" ]]; then
        # Check only staged shell files that contain osascript
        APPLESCRIPT_FILES=()
        for file in $STAGED_FILES; do
            if [[ "$file" == *.sh && -f "$file" ]] && grep -q 'osascript' "$file" 2>/dev/null; then
                APPLESCRIPT_FILES+=("$file")
            fi
        done
        if [[ ${#APPLESCRIPT_FILES[@]} -gt 0 ]]; then
            if ! "$LINTER" "${APPLESCRIPT_FILES[@]}" 2>/dev/null; then
                warn "AppleScript syntax errors found in staged files"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    fi
fi

# Run Python linter (ruff) on staged Python files
if command -v ruff &>/dev/null; then
    PYTHON_FILES=()
    for file in $STAGED_FILES; do
        if [[ "$file" == *.py && -f "$file" ]]; then
            PYTHON_FILES+=("$file")
        fi
    done
    if [[ ${#PYTHON_FILES[@]} -gt 0 ]]; then
        # Run ruff check and capture output
        RUFF_OUTPUT=$(ruff check "${PYTHON_FILES[@]}" 2>&1) || true
        if [[ -n "$RUFF_OUTPUT" && "$RUFF_OUTPUT" != "All checks passed!" ]]; then
            warn "Python linter (ruff) found issues:"
            echo "$RUFF_OUTPUT" | head -20
            if [[ $(echo "$RUFF_OUTPUT" | wc -l) -gt 20 ]]; then
                echo "  ... (truncated, run 'ruff check' for full output)"
            fi
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
fi

# Run shell script linter (shellcheck) on staged bash files
if command -v shellcheck &>/dev/null; then
    SHELL_FILES=()
    for file in $STAGED_FILES; do
        if [[ "$file" == *.sh && -f "$file" ]]; then
            SHELL_FILES+=("$file")
        fi
    done
    if [[ ${#SHELL_FILES[@]} -gt 0 ]]; then
        # Run shellcheck and capture output
        SHELLCHECK_OUTPUT=$(shellcheck "${SHELL_FILES[@]}" 2>&1) || true
        if [[ -n "$SHELLCHECK_OUTPUT" ]]; then
            warn "Shell linter (shellcheck) found issues:"
            echo "$SHELLCHECK_OUTPUT" | head -20
            if [[ $(echo "$SHELLCHECK_OUTPUT" | wc -l) -gt 20 ]]; then
                echo "  ... (truncated, run 'shellcheck' for full output)"
            fi
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
fi

# Check for new test ignores without issue references (#267)
# Patterns: #[ignore, @skip, xfail, @pytest.mark.skip
check_new_ignores() {
    local file="$1"
    local ext="${file##*.}"

    # Only check source files that can have test ignores
    case "$ext" in
        rs|py|js|ts|tsx) ;;
        *) return 0 ;;
    esac

    # Get ignore patterns added in this commit (new lines only, indicated by +)
    local new_ignores
    new_ignores=$(git diff --cached -U0 "$file" 2>/dev/null | grep -E '^\+.*#\[ignore|^\+.*@skip|^\+.*xfail|^\+.*@pytest\.mark\.skip' | grep -v '^+++' || true)

    if [[ -n "$new_ignores" ]]; then
        # Check if each new ignore has an issue reference (#N)
        while IFS= read -r line; do
            if ! echo "$line" | grep -qE '#[0-9]+'; then
                warn "New test ignore without issue reference in $file:"
                echo "  $line"
                echo "  Ignores must reference an issue: #[ignore = \"Needs X #123\"]"
                WARNINGS=$((WARNINGS + 1))
            fi
        done <<< "$new_ignores"
    fi
}

# Check staged files for new ignores without issue refs
# Only relevant for PROVER (test correctness) and MANAGER (audit) roles
if [[ "${AI_ROLE:-}" == "PROVER" ]] || [[ "${AI_ROLE:-}" == "MANAGER" ]]; then
    for file in $STAGED_FILES; do
        if [[ -f "$file" ]]; then
            check_new_ignores "$file"
        fi
    done
fi

# WORKER should not create new #[ignore] annotations - tests are Prover's domain
# Moving/refactoring existing ignores is OK (net change <= 0)
if [[ "${AI_ROLE:-}" == "WORKER" ]]; then
    for file in $STAGED_FILES; do
        if [[ -f "$file" ]]; then
            ext="${file##*.}"
            case "$ext" in
                rs|py|js|ts|tsx)
                    # Count added vs removed ignores to detect net new (not just moves)
                    diff_output=$(git diff --cached -U0 "$file" 2>/dev/null || true)
                    added=$(echo "$diff_output" | { grep -E '^\+.*#\[ignore|^\+.*@skip|^\+.*xfail|^\+.*@pytest\.mark\.skip' || true; } | wc -l | tr -d ' ')
                    removed=$(echo "$diff_output" | { grep -E '^-.*#\[ignore|^-.*@skip|^-.*xfail|^-.*@pytest\.mark\.skip' || true; } | wc -l | tr -d ' ')
                    net_new=$((added - removed))
                    if [[ "$net_new" -gt 0 ]]; then
                        warn "WORKER adding $net_new new test ignore(s) in $file (tests are Prover's domain)"
                        echo "  If disabling a test, coordinate with Prover or file an issue"
                        WARNINGS=$((WARNINGS + 1))
                    fi
                    ;;
            esac
        fi
    done
fi

# Report warnings but don't block
if [[ $WARNINGS -gt 0 ]]; then
    echo ""
    warn "$WARNINGS issue(s) found (missing headers, authors, or ignore refs)"
    echo "  Add headers from: ai_template_scripts/headers/"
    echo "  Ignores must reference issues: #[ignore = \"Reason #123\"]"
    echo "  To skip this check: git commit --no-verify"
    echo ""
fi

# Self-audit reminder for AI sessions
# Only shows for AI commits (looper sets AI_ROLE)
if [[ -n "${AI_ROLE:-}" ]]; then
    CYAN='\033[1;36m'
    BOLD='\033[1m'
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} ${BOLD}SELF-AUDIT CHECKLIST${NC}                                          ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Before committing, did you:                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Look for MORE bugs beyond the obvious ones?               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Check edge cases and error handling?                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Verify ALL tests pass?                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Consider what you might have missed?                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC} The follow-up audit often catches 2-4 more issues!            ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# Restore IFS
IFS="$OLD_IFS"

# Always exit 0 (warning mode, not blocking)
exit 0

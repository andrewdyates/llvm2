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
RED='\033[1;31m'
NC='\033[0m'

warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

WARNINGS=0
ERRORS=0  # Blocking errors (ignores without issue refs)

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

# NOTE: ruff and shellcheck are handled by pre-commit framework
# See .pre-commit-config.yaml for configuration

# FORBID test ignores entirely (#341)
# Tests must PASS, FAIL, or be DELETED. No ignore. No hiding.
#
# Exclusions can be configured via:
#   1. .ignore-check-exclude file (one path prefix per line)
#   2. IGNORE_CHECK_EXCLUDE env var (colon-separated paths)
#
# Example .ignore-check-exclude:
#   vendor/
#   third_party/
#   generated/

# Load exclusion paths
IGNORE_EXCLUSIONS=()
if [[ -f ".ignore-check-exclude" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        IGNORE_EXCLUSIONS+=("$line")
    done < ".ignore-check-exclude"
fi
if [[ -n "${IGNORE_CHECK_EXCLUDE:-}" ]]; then
    IFS=':' read -ra env_excludes <<< "$IGNORE_CHECK_EXCLUDE"
    IGNORE_EXCLUSIONS+=("${env_excludes[@]}")
fi

is_excluded() {
    local file="$1"
    # Handle empty array safely with set -u
    if [[ ${#IGNORE_EXCLUSIONS[@]} -gt 0 ]]; then
        for excl in "${IGNORE_EXCLUSIONS[@]}"; do
            if [[ "$file" == "$excl"* ]]; then
                return 0
            fi
        done
    fi
    return 1
}

check_new_ignores() {
    local file="$1"
    local ext="${file##*.}"

    # Check exclusions
    if is_excluded "$file"; then
        return 0
    fi

    # Language-specific patterns (precise to avoid false positives)
    # Note: Use [[:space:]] not \s for POSIX ERE compatibility
    local pattern=""
    case "$ext" in
        rs)
            # Rust: #[ignore] or #[ignore = "..."]
            pattern='#\[ignore[[:space:]]*(\]|=)'
            ;;
        py)
            # Python: @skip(), @pytest.mark.skip, @pytest.mark.xfail, @unittest.skip*
            pattern='@skip[[:space:]]*\(|@pytest\.mark\.skip|@pytest\.mark\.xfail|@unittest\.skip'
            ;;
        js|ts|tsx)
            # JS/TS: .skip(, it.skip(, test.skip(, describe.skip(, xit(, xdescribe(, xtest(
            pattern='\.skip\(|^[[:space:]]*(xit|xdescribe|xtest)[[:space:]]*\('
            ;;
        *)
            return 0
            ;;
    esac

    # Get ignore patterns added in this commit (new lines only, indicated by +)
    local new_ignores
    new_ignores=$(git diff --cached -U0 "$file" 2>/dev/null | grep -E "^\+.*($pattern)" | grep -v '^+++' || true)

    if [[ -n "$new_ignores" ]]; then
        while IFS= read -r line; do
            error "FORBIDDEN: Test ignore in $file"
            echo "  $line"
            echo ""
            echo "  Tests must PASS, FAIL, or be DELETED. No ignore."
            echo "  - Slow? Add timeout. If timeout exceeded, it FAILS."
            echo "  - Blocked? Let it FAIL. Failure is visibility."
            echo "  - Flaky? Fix flakiness or DELETE."
            echo "  - Obsolete? DELETE."
            echo ""
            ERRORS=$((ERRORS + 1))
        done <<< "$new_ignores"
    fi
}

# Check staged files for new ignores (ALL roles, BLOCKING)
for file in $STAGED_FILES; do
    if [[ -f "$file" ]]; then
        check_new_ignores "$file"
    fi
done

# Report warnings but don't block
if [[ $WARNINGS -gt 0 ]]; then
    echo ""
    warn "$WARNINGS warning(s) found (missing headers, authors, etc.)"
    echo "  Add headers from: ai_template_scripts/headers/"
    echo "  To skip warnings: git commit --no-verify"
    echo ""
fi

# Report blocking errors
if [[ $ERRORS -gt 0 ]]; then
    echo ""
    error "$ERRORS BLOCKING error(s) - commit rejected"
    echo "  Test ignores are FORBIDDEN. Tests must PASS, FAIL, or be DELETED."
    echo "  See ai_template.md 'Test ignores FORBIDDEN' rule."
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

# === Project-specific extensions ===
# Hooks in .pre-commit-local.d/ are NOT synced from ai_template.
# Projects can add custom pre-commit checks there.
# See ai_template_scripts/README.md for documentation.
PROJECT_HOOKS_DIR="$REPO_ROOT/.pre-commit-local.d"
if [[ -d "$PROJECT_HOOKS_DIR" ]]; then
    for hook in "$PROJECT_HOOKS_DIR"/*.sh; do
        [[ -f "$hook" && -x "$hook" ]] || continue
        echo "Running project hook: $(basename "$hook")"
        if ! "$hook"; then
            error "Project hook failed: $(basename "$hook")"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

# Exit non-zero if blocking errors found
if [[ $ERRORS -gt 0 ]]; then
    exit 1
fi
exit 0

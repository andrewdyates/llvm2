#!/usr/bin/env bash
# lint_applescript.sh - Validate AppleScript syntax in shell scripts
#
# Copyright 2026 Dropbox, Inc.
# SPDX-License-Identifier: Apache-2.0
# Author: Andrew Yates
#
# Usage:
#   ./ai_template_scripts/lint_applescript.sh                    # Check all scripts
#   ./ai_template_scripts/lint_applescript.sh script.sh          # Check specific file
#   ./ai_template_scripts/lint_applescript.sh --fix script.sh    # Show what to fix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "lint_applescript.sh ${git_hash} (${date})"
    exit 0
}

usage() {
    cat <<'EOF'
Usage: lint_applescript.sh [OPTIONS] [FILE...]

Validate AppleScript syntax embedded in shell scripts.

Arguments:
  FILE...       Files to check (default: all *.sh in ai_template_scripts/)

Options:
  -h, --help    Show this help message
  --version     Show version information
  --verbose     Show each script being checked
  --fix         Show detailed error context

Checks osascript -e blocks for valid AppleScript syntax using osacompile.
EOF
}

VERBOSE=false
FIX=false
FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --version) version ;;
        --verbose) VERBOSE=true; shift ;;
        --fix) FIX=true; shift ;;
        -*) echo "Error: Unknown option: $1" >&2; exit 1 ;;
        *) FILES+=("$1"); shift ;;
    esac
done

# Default to all shell scripts
if [[ ${#FILES[@]} -eq 0 ]]; then
    while IFS= read -r -d '' file; do
        FILES+=("$file")
    done < <(find "$SCRIPT_DIR" -maxdepth 1 -name "*.sh" -print0)
fi

ERRORS=0
CHECKED=0

# Extract and validate AppleScript from a file
# shellcheck disable=SC2094 # False positive: $file is only read, not written
check_file() {
    local file="$1"
    local filename
    filename=$(basename "$file")

    $VERBOSE && echo "Checking: $filename"

    # Look for osascript -e patterns with multiline strings
    # Pattern: osascript -e "..." or osascript -e '...'
    local in_osascript=false
    local script_content=""
    local line_num=0
    local start_line=0
    local quote_char=""

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))

        if ! $in_osascript; then
            # Look for start of osascript block
            # Must be at start of line, after ; && || or !, not inside [[ ]] or quotes
            if [[ "$line" =~ ^[[:space:]]*(osascript|.*[\;\&\|!][[:space:]]*osascript)[[:space:]]+-e[[:space:]]+\" ]] && \
               [[ ! "$line" =~ \[\[.*osascript ]] && [[ ! "$line" =~ =~.*osascript ]]; then
                in_osascript=true
                quote_char='"'
                start_line=$line_num
                # Extract content after osascript -e "
                script_content="${line#*osascript -e \"}"
                # Check if it ends on the same line
                if [[ "$script_content" =~ \"[[:space:]]*(\;|\&\&|\|\||$|\|) ]]; then
                    # Single line osascript - extract up to closing quote
                    script_content="${script_content%%\"*}"
                    check_applescript "$file" "$start_line" "$script_content"
                    in_osascript=false
                    script_content=""
                fi
            elif [[ "$line" =~ ^[[:space:]]*(osascript|.*[\;\&\|!][[:space:]]*osascript)[[:space:]]+-e[[:space:]]+\' ]] && \
                 [[ ! "$line" =~ \[\[.*osascript ]] && [[ ! "$line" =~ =~.*osascript ]]; then
                in_osascript=true
                quote_char="'"
                start_line=$line_num
                script_content="${line#*osascript -e \'}"
                if [[ "$script_content" =~ \'[[:space:]]*(\;|\&\&|\|\||$|\|) ]]; then
                    script_content="${script_content%%\'*}"
                    check_applescript "$file" "$start_line" "$script_content"
                    in_osascript=false
                    script_content=""
                fi
            fi
        else
            # Inside osascript block - look for end quote
            if [[ "$quote_char" == '"' && "$line" =~ \"[[:space:]]*(\;|\||2\>\&1) ]]; then
                # Found closing quote
                script_content+=$'\n'"${line%%\"*}"
                check_applescript "$file" "$start_line" "$script_content"
                in_osascript=false
                script_content=""
            elif [[ "$quote_char" == "'" && "$line" =~ \'[[:space:]]*(\;|\||2\>\&1) ]]; then
                script_content+=$'\n'"${line%%\'*}"
                check_applescript "$file" "$start_line" "$script_content"
                in_osascript=false
                script_content=""
            else
                script_content+=$'\n'"$line"
            fi
        fi
    done < "$file"
}

# Validate AppleScript using osacompile
check_applescript() {
    local file="$1"
    local line="$2"
    local script="$3"

    ((CHECKED++))

    # Unescape bash escapes for AppleScript
    # \" -> "
    script="${script//\\\"/\"}"

    # Create temp file
    local tmpfile
    tmpfile=$(mktemp)
    echo "$script" > "$tmpfile"

    # Try to compile (syntax check only)
    local output
    if ! output=$(osacompile -o /dev/null "$tmpfile" 2>&1); then
        ((ERRORS++))
        echo "ERROR: $file:$line - AppleScript syntax error"
        if $FIX; then
            echo "--- Script content ---"
            echo "$script"
            echo "--- Compiler output ---"
            echo "$output"
            echo "----------------------"
        else
            # Extract just the error message
            echo "  ${output##*:}"
        fi
        echo
    fi

    rm -f "$tmpfile"
}

# Check each file
for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
        check_file "$file"
    else
        echo "Warning: File not found: $file" >&2
    fi
done

# Summary
if [[ $CHECKED -eq 0 ]]; then
    echo "No AppleScript blocks found"
    exit 0
fi

if [[ $ERRORS -gt 0 ]]; then
    echo "Found $ERRORS error(s) in $CHECKED AppleScript block(s)"
    exit 1
else
    $VERBOSE && echo "All $CHECKED AppleScript block(s) valid"
    exit 0
fi

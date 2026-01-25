#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# sync_check.sh - Check template drift across repos
#
# Reads .ai_template_version from repos and compares to current ai_template HEAD.
# Reports which repos are behind and by how many commits.
#
# Usage:
#   ./ai_template_scripts/sync_check.sh              # Check all sibling repos
#   ./ai_template_scripts/sync_check.sh /path/to/repos  # Check repos in directory
#   ./ai_template_scripts/sync_check.sh repo1 repo2     # Check specific repos

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Must run from ai_template root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$AI_TEMPLATE_ROOT"

[[ -f "CLAUDE.md" ]] || { echo "ERROR: Run from ai_template root"; exit 1; }

# Get ai_template current version
CURRENT_VERSION=$(git rev-parse HEAD)
CURRENT_SHORT=$(git rev-parse --short HEAD)

echo "ai_template current version: $CURRENT_SHORT"
echo ""

# Collect repos to check
REPOS=()
SKIPPED=()
ARGS=()
CHECK_FILES_ARG=false

for arg in "$@"; do
    case "$arg" in
        --files) CHECK_FILES_ARG=true ;;
        -*) echo "ERROR: Unknown option: $arg"; exit 1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
    # Default: check sibling directories (same parent as ai_template)
    PARENT_DIR="$(dirname "$AI_TEMPLATE_ROOT")"
    for dir in "$PARENT_DIR"/*/; do
        [[ -d "$dir/.git" ]] && [[ "$(basename "$dir")" != "ai_template" ]] && REPOS+=("$dir")
    done
elif [[ ${#ARGS[@]} -eq 1 && -d "${ARGS[0]}" && ! -d "${ARGS[0]}/.git" ]]; then
    # Single directory argument that's not a repo - scan it
    for dir in "${ARGS[0]}"/*/; do
        [[ -d "$dir/.git" ]] && REPOS+=("$dir")
    done
else
    # Explicit repo list
    for arg in "${ARGS[@]}"; do
        if [[ -d "$arg/.git" ]]; then
            REPOS+=("$arg")
        else
            SKIPPED+=("$arg")
        fi
    done
fi

if [[ ${#REPOS[@]} -eq 0 ]]; then
    echo "No repos found to check."
    if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo "Skipped non-repo paths:"
        for skipped in "${SKIPPED[@]}"; do
            echo "  - $skipped"
        done
    fi
    exit 0
fi

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "Warning: skipped non-repo paths:"
    for skipped in "${SKIPPED[@]}"; do
        echo "  - $skipped"
    done
    echo ""
fi

# Check each repo
UP_TO_DATE=0
BEHIND=0
UNTRACKED=0
TOTAL=${#REPOS[@]}

printf "%-30s %-10s %-10s %s\n" "REPO" "VERSION" "STATUS" "COMMITS BEHIND"
printf "%-30s %-10s %-10s %s\n" "----" "-------" "------" "--------------"

for repo in "${REPOS[@]}"; do
    repo_name=$(basename "$repo")
    version_file="$repo/.ai_template_version"

    if [[ ! -f "$version_file" ]]; then
        printf "%-30s %-10s ${YELLOW}%-10s${NC} %s\n" "$repo_name" "-" "untracked" "no .ai_template_version"
        ((UNTRACKED++))
        continue
    fi

    repo_version=$(head -1 "$version_file" | tr -d '[:space:]')

    if [[ "$repo_version" == "$CURRENT_VERSION" ]]; then
        printf "%-30s %-10s ${GREEN}%-10s${NC}\n" "$repo_name" "${repo_version:0:7}" "current"
        ((UP_TO_DATE++))
    else
        # Count commits behind
        commits_behind=$(git rev-list --count "$repo_version..$CURRENT_VERSION" 2>/dev/null || echo "?")

        if [[ "$commits_behind" == "?" ]]; then
            printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "unknown" "version not found in history"
        elif [[ "$commits_behind" -gt 10 ]]; then
            printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "STALE" "$commits_behind commits behind"
        else
            printf "%-30s %-10s ${YELLOW}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "behind" "$commits_behind commits behind"
        fi
        ((BEHIND++))
    fi
done

echo ""
echo "Summary: $UP_TO_DATE current, $BEHIND behind, $UNTRACKED untracked (of $TOTAL repos)"

if [[ $BEHIND -gt 0 || $UNTRACKED -gt 0 ]]; then
    echo ""
    echo "To sync a repo:"
    echo "  ./ai_template_scripts/sync_repo.sh /path/to/repo"
fi

# File-level drift detection (optional, slower)
if [[ "${CHECK_FILES:-}" == "1" ]] || [[ "$CHECK_FILES_ARG" == "true" ]]; then
    echo ""
    echo "=== File-level drift check ==="

    MANIFEST="$AI_TEMPLATE_ROOT/.sync_manifest"
    if [[ ! -f "$MANIFEST" ]]; then
        echo "Warning: .sync_manifest not found, skipping file check"
        exit 0
    fi

    # Collect files from manifest (skip directories, exclusions, comments)
    declare -a SYNC_FILES
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line%"${line##*[![:space:]]}"}"
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" ]] && continue
        [[ "$line" == !* ]] && continue
        [[ "$line" == */ ]] && continue  # Skip directories
        [[ "$line" == *\** ]] && continue  # Skip globs (too complex)
        SYNC_FILES+=("$line")
    done < "$MANIFEST"

    DRIFT_REPOS=()
    for repo in "${REPOS[@]}"; do
        repo_name=$(basename "$repo")
        drifted=()

        for file in "${SYNC_FILES[@]}"; do
            src="$AI_TEMPLATE_ROOT/$file"
            dst="$repo/$file"

            [[ ! -f "$src" ]] && continue
            [[ ! -f "$dst" ]] && { drifted+=("$file (missing)"); continue; }

            if ! diff -q "$src" "$dst" > /dev/null 2>&1; then
                drifted+=("$file")
            fi
        done

        if [[ ${#drifted[@]} -gt 0 ]]; then
            echo ""
            echo -e "${YELLOW}$repo_name${NC} has ${#drifted[@]} drifted files:"
            for f in "${drifted[@]}"; do
                echo "  - $f"
            done
            DRIFT_REPOS+=("$repo_name")
        fi
    done

    if [[ ${#DRIFT_REPOS[@]} -eq 0 ]]; then
        echo "No file-level drift detected."
    else
        echo ""
        echo "Found drift in ${#DRIFT_REPOS[@]} repos. Run sync_repo.sh to fix."
    fi
fi

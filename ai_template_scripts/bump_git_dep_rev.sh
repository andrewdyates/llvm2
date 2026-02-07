#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
#
# Bump git dependency revision in Cargo.toml files.
# Updates all occurrences of a git dep to the latest commit.
#
# Usage:
#   bump_git_dep_rev.sh REPO_URL           # Bump to HEAD
#   bump_git_dep_rev.sh REPO_URL REV       # Bump to specific rev
#   bump_git_dep_rev.sh --dry-run REPO_URL # Show what would change
#   bump_git_dep_rev.sh --version          # Show script version
#
# Examples:
#   bump_git_dep_rev.sh https://github.com/dropbox-ai-prototypes/z4
#   bump_git_dep_rev.sh https://github.com/dropbox-ai-prototypes/z4 cdfa08fb
#   bump_git_dep_rev.sh --dry-run https://github.com/dropbox-ai-prototypes/z4

set -euo pipefail

# Load identity configuration from ait_identity.toml
_BUMP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=identity.sh
source "$_BUMP_SCRIPT_DIR/identity.sh" 2>/dev/null || true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

DRY_RUN=false

version() {
    echo "bump_git_dep_rev.sh $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(date +%Y-%m-%d))"
    exit 0
}

usage() {
    echo "Usage: bump_git_dep_rev.sh [--dry-run] REPO_URL [REV]"
    echo ""
    echo "Bump git dependency revision in Cargo.toml files."
    echo ""
    echo "Arguments:"
    echo "  REPO_URL   Git repository URL to bump (e.g., https://github.com/$AIT_GITHUB_ORG/z4)"
    echo "  REV        Target revision (default: HEAD of default branch)"
    echo ""
    echo "Options:"
    echo "  --dry-run  Show what would change without modifying files"
    echo "  --version  Show script version"
    echo "  --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  bump_git_dep_rev.sh https://github.com/$AIT_GITHUB_ORG/z4"
    echo "  bump_git_dep_rev.sh https://github.com/$AIT_GITHUB_ORG/z4 cdfa08fb"
    echo "  bump_git_dep_rev.sh --dry-run https://github.com/$AIT_GITHUB_ORG/z4"
    echo "  bump_git_dep_rev.sh --version"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    --version)
        version
        ;;
    --help | -h)
        usage
        exit 0
        ;;
    -*)
        echo -e "${RED}Error: Unknown option $1${NC}" >&2
        usage >&2
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error: REPO_URL required${NC}" >&2
    usage >&2
    exit 1
fi

REPO_URL="$1"
TARGET_REV="${2:-}"

# Normalize URL (remove trailing .git if present)
REPO_URL="${REPO_URL%.git}"
REPO_URL_REGEX=$(printf '%s' "$REPO_URL" | sed -e 's/[][\\.^$*+?{}|()]/\\&/g')
REPO_URL_REGEX="${REPO_URL_REGEX}(\\.git)?"

# Get target revision
if [[ -z "$TARGET_REV" ]]; then
    echo "Fetching HEAD from $REPO_URL..." >&2

    # Extract owner/repo from URL for gh api
    if [[ "$REPO_URL" =~ github\.com[:/]([^/]+)/([^/]+) ]]; then
        OWNER="${BASH_REMATCH[1]}"
        REPO="${BASH_REMATCH[2]}"
        REPO="${REPO%.git}" # Remove .git suffix if present

        # Use gh api to get default branch HEAD
        TARGET_REV=$(gh api "repos/$OWNER/$REPO/commits/HEAD" --jq '.sha' 2>/dev/null || true)
    fi

    # Fallback to git ls-remote if gh failed
    if [[ -z "$TARGET_REV" ]]; then
        TARGET_REV=$(git ls-remote "$REPO_URL" HEAD 2>/dev/null | cut -f1 || true)
    fi

    if [[ -z "$TARGET_REV" ]]; then
        echo -e "${RED}Error: Could not fetch HEAD from $REPO_URL${NC}" >&2
        echo "Make sure the repository exists and you have access." >&2
        exit 1
    fi
fi

# Shorten rev for display (keep first 12 chars)
SHORT_REV="${TARGET_REV:0:12}"

echo "Target revision: $SHORT_REV" >&2

# Find Cargo.toml files with this git dependency
# Pattern matches: { git = "REPO_URL", rev = "..." }
CARGO_FILES=$(find . -name "Cargo.toml" -type f 2>/dev/null || true)

if [[ -z "$CARGO_FILES" ]]; then
    echo -e "${YELLOW}No Cargo.toml files found in current directory${NC}" >&2
    exit 0
fi

FOUND_FILES=()
UPDATED_COUNT=0

for cargo_file in $CARGO_FILES; do
    # Check if this file references the repo
    if grep -Eq "git[[:space:]]*=[[:space:]]*\"${REPO_URL_REGEX}\"" "$cargo_file" 2>/dev/null; then
        FOUND_FILES+=("$cargo_file")

        # Count matches before update
        # shellcheck disable=SC1087 # The [^"] is a regex character class, not array syntax
        # Use case-insensitive match for hex (git commits are lowercase but users may paste uppercase)
        OLD_REVS=$(grep -oiE "git[[:space:]]*=[[:space:]]*\"${REPO_URL_REGEX}\"[^\\n]*rev[[:space:]]*=[[:space:]]*\"[a-f0-9]+\"" "$cargo_file" 2>/dev/null | grep -oiE "rev[[:space:]]*=[[:space:]]*\"[a-f0-9]+\"" | sort -u || true)

        if [[ -n "$OLD_REVS" ]]; then
            echo "" >&2
            echo -e "${GREEN}Found:${NC} $cargo_file" >&2
            echo "  Old revisions:" >&2
            echo "$OLD_REVS" | while read -r line; do echo "    $line"; done >&2
            echo "  New revision: rev = \"$TARGET_REV\"" >&2

            if [[ "$DRY_RUN" == "false" ]]; then
                # Update all rev = "..." that follow git = "REPO_URL..."
                # This is tricky with sed; we use perl for multi-line safety
                # Use case-insensitive match for hex (/i flag)
                # Export variables for perl to avoid shell escaping issues
                export BUMP_REPO_REGEX="$REPO_URL_REGEX"
                export BUMP_TARGET_REV="$TARGET_REV"
                perl -i -pe '
                    my $repo_regex = $ENV{BUMP_REPO_REGEX};
                    my $target_rev = $ENV{BUMP_TARGET_REV};
                    if (/git\s*=\s*"$repo_regex"/) {
                        s/rev\s*=\s*"[a-f0-9]+"/rev = "$target_rev"/gi;
                    }
                ' "$cargo_file"

                ((UPDATED_COUNT++)) || true
                echo -e "  ${GREEN}✓ Updated${NC}" >&2
            else
                echo -e "  ${YELLOW}(dry-run, no changes made)${NC}" >&2
            fi
        fi
    fi
done

echo "" >&2

if [[ ${#FOUND_FILES[@]} -eq 0 ]]; then
    echo -e "${YELLOW}No Cargo.toml files reference $REPO_URL${NC}" >&2
    exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}Dry run complete. ${#FOUND_FILES[@]} file(s) would be updated.${NC}" >&2
else
    echo -e "${GREEN}Updated $UPDATED_COUNT file(s).${NC}" >&2
    echo "" >&2
    echo "Next steps:" >&2
    echo "  1. Run 'cargo check' to verify the update" >&2
    echo "  2. Run 'cargo update' to refresh Cargo.lock" >&2
    echo "  3. Commit with: git commit -am \"Bump ${REPO_URL##*/} to $SHORT_REV\"" >&2
fi

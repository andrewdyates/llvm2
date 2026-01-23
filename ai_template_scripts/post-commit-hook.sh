#!/bin/bash
# post-commit-hook.sh - Git post-commit hook for issue label management
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# License: Apache-2.0
#
# PURPOSE: Automatically manage issue labels based on commit keywords.
# CALLED BY: Git (installed to .git/hooks/post-commit by looper.py)
# REFERENCED: .claude/rules/ai_template.md (issue keywords)
#
# Keywords handled:
#   Claims #N   -> adds 'in-progress' label
#   Unclaims #N -> removes 'in-progress' label
#   Reopens #N  -> reopens the issue

set -euo pipefail

# Get the commit message
MSG=$(git log -1 --pretty=%B)

# Check if gh is available
if ! command -v gh &> /dev/null; then
    exit 0
fi

# Extract and process Claims
echo "$MSG" | grep -oE 'Claims #[0-9]+' | grep -oE '[0-9]+' | while read -r ISSUE; do
    if [[ -n "$ISSUE" ]]; then
        echo "Adding 'in-progress' label to #$ISSUE"
        gh issue edit "$ISSUE" --add-label "in-progress" 2>/dev/null || true
    fi
done

# Extract and process Unclaims
echo "$MSG" | grep -oE 'Unclaims #[0-9]+' | grep -oE '[0-9]+' | while read -r ISSUE; do
    if [[ -n "$ISSUE" ]]; then
        echo "Removing 'in-progress' label from #$ISSUE"
        gh issue edit "$ISSUE" --remove-label "in-progress" 2>/dev/null || true
    fi
done

# Extract and process Reopens
echo "$MSG" | grep -oE 'Reopens #[0-9]+' | grep -oE '[0-9]+' | while read -r ISSUE; do
    if [[ -n "$ISSUE" ]]; then
        echo "Reopening #$ISSUE"
        gh issue reopen "$ISSUE" 2>/dev/null || true
    fi
done

exit 0

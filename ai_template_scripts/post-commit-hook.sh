#!/bin/bash
# post-commit-hook.sh - Git post-commit hook for issue label management
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# License: Apache-2.0
#
# PURPOSE: Automatically manage issue labels based on commit keywords.
# CALLED BY: Git (installed to .git/hooks/post-commit by install_hooks.sh or looper.py)
# REFERENCED: .claude/rules/ai_template.md (issue keywords)
#
# Keywords handled:
#   Claims #N   -> adds in-progress label (worker-specific when AI_WORKER_ID set)
#   Unclaims #N -> removes in-progress label (worker-specific when AI_WORKER_ID set)
#   Reopens #N  -> reopens the issue
#   Fixes/Closes/Resolves #N -> removes workflow labels + invalidates historical cache (#1277)
#
# Multi-worker support:
#   When AI_WORKER_ID env var is set (e.g., "1"), uses in-progress-W1 label
#   Otherwise uses in-progress label
#
# Note: The claim comment (e.g., "Claiming this issue") is added by gh_post.py
# when it detects --add-label in-progress, not by this script. See #1017.

set -euo pipefail

# Get the commit message
MSG=$(git log -1 --pretty=%B)

# --- Helper: Check if issue ID is a local issue (L-prefixed) ---
is_local_issue() {
    local issue_id="$1"
    [[ "$issue_id" =~ ^L[0-9]+$ ]]
}

# --- Helper: Edit local issue labels ---
# Usage: local_issue_edit <issue_id> add|remove <label>
local_issue_edit() {
    local issue_id="$1"
    local action="$2"
    local label="$3"
    local issue_file=".issues/${issue_id}.md"

    if [[ ! -f "$issue_file" ]]; then
        echo "Warning: Local issue $issue_id not found" >&2
        return 1
    fi

    # Use Python to update the YAML frontmatter labels array
    python3 -c "
import json
import re
import sys

issue_file = '$issue_file'
action = '$action'
label = '$label'

try:
    with open(issue_file, 'r') as f:
        content = f.read()

    # Split frontmatter from body
    parts = content.split('---', 2)
    if len(parts) < 3:
        sys.exit(1)

    frontmatter = parts[1].strip()
    body = parts[2]

    # Parse labels from frontmatter
    labels_match = re.search(r'^labels:\s*(\[.*?\])', frontmatter, re.MULTILINE)
    if labels_match:
        labels = json.loads(labels_match.group(1))
    else:
        labels = []

    # Update labels
    if action == 'add':
        if label not in labels:
            labels.append(label)
    elif action == 'remove':
        labels = [l for l in labels if l != label]

    # Update frontmatter
    new_labels_str = json.dumps(labels)
    if labels_match:
        new_frontmatter = re.sub(r'^labels:\s*\[.*?\]', f'labels: {new_labels_str}', frontmatter, flags=re.MULTILINE)
    else:
        new_frontmatter = frontmatter + f'\nlabels: {new_labels_str}'

    # Write back
    with open(issue_file, 'w') as f:
        f.write('---\n' + new_frontmatter + '\n---' + body)

    print(f'Updated local issue {issue_file}: {action} {label}', file=sys.stderr)
except Exception as e:
    print(f'Error updating local issue: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || return 1
    return 0
}

# --- Helper: Reopen local issue ---
local_issue_reopen() {
    local issue_id="$1"
    local issue_file=".issues/${issue_id}.md"

    if [[ ! -f "$issue_file" ]]; then
        echo "Warning: Local issue $issue_id not found" >&2
        return 1
    fi

    # Update state in YAML frontmatter
    sed -i.bak "s/^state:.*/state: open/" "$issue_file" && rm -f "${issue_file}.bak"
    echo "Reopened local issue #$issue_id" >&2
    return 0
}

# --- Helper: Close local issue ---
local_issue_close() {
    local issue_id="$1"
    local issue_file=".issues/${issue_id}.md"

    if [[ ! -f "$issue_file" ]]; then
        echo "Warning: Local issue $issue_id not found" >&2
        return 1
    fi

    # Update state in YAML frontmatter
    sed -i.bak "s/^state:.*/state: closed/" "$issue_file" && rm -f "${issue_file}.bak"
    echo "Closed local issue #$issue_id" >&2
    return 0
}

# Check if gh is available (not needed for local-only mode)
# In full local mode (AIT_LOCAL_MODE=full), force GH_AVAILABLE to false
# to prevent any GitHub API calls
GH_AVAILABLE=false
if [[ "${AIT_LOCAL_MODE:-}" != "full" ]] && command -v gh &> /dev/null; then
    GH_AVAILABLE=true
fi

# Get repo slug once for REST API calls (avoids GraphQL, see #1074)
# Extract from git remote URL using sanitizer to avoid credential leakage (#2240)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SLUG=$(python3 "$SCRIPT_DIR/url_sanitizer.py" --repo-slug "$(git remote get-url origin 2>/dev/null)" 2>/dev/null || true)

# Extract and process Claims (title line only to avoid body text matches)
# Use || true to prevent pipefail from exiting when grep finds no matches
# Fix for #1453: Use orthogonal labels (in-progress + WN) instead of combined (in-progress-WN)
# Supports both GitHub issues (#N) and local issues (#LN)
TITLE_LINE=$(echo "$MSG" | head -1)
echo "$TITLE_LINE" | grep -oE 'Claims #L?[0-9]+' | grep -oE 'L?[0-9]+' | while read -r ISSUE; do
    if [[ -n "$ISSUE" ]]; then
        if is_local_issue "$ISSUE"; then
            echo "Claiming local #$ISSUE with in-progress label"
            local_issue_edit "$ISSUE" add "in-progress" || true
            if [[ -n "${AI_WORKER_ID:-}" ]]; then
                OWNERSHIP_LABEL="W${AI_WORKER_ID}"
                echo "Adding ownership label '$OWNERSHIP_LABEL' to local #$ISSUE"
                local_issue_edit "$ISSUE" add "$OWNERSHIP_LABEL" || true
            fi
        elif [[ "$GH_AVAILABLE" == "true" ]]; then
            echo "Claiming #$ISSUE with in-progress label"
            gh issue edit "$ISSUE" --add-label "in-progress" 2>/dev/null || true
            if [[ -n "${AI_WORKER_ID:-}" ]]; then
                OWNERSHIP_LABEL="W${AI_WORKER_ID}"
                echo "Adding ownership label '$OWNERSHIP_LABEL' to #$ISSUE"
                gh issue edit "$ISSUE" --add-label "$OWNERSHIP_LABEL" 2>/dev/null || true
            fi
        fi
    fi
done || true

# Extract and process Unclaims (title line only)
# Fix for #1453: Remove both in-progress and ownership labels
# Supports both GitHub issues (#N) and local issues (#LN)
echo "$TITLE_LINE" | grep -oE 'Unclaims #L?[0-9]+' | grep -oE 'L?[0-9]+' | while read -r ISSUE; do
    if [[ -n "$ISSUE" ]]; then
        if is_local_issue "$ISSUE"; then
            echo "Removing 'in-progress' label from local #$ISSUE"
            local_issue_edit "$ISSUE" remove "in-progress" || true
            if [[ -n "${AI_WORKER_ID:-}" ]]; then
                OWNERSHIP_LABEL="W${AI_WORKER_ID}"
                echo "Removing ownership label '$OWNERSHIP_LABEL' from local #$ISSUE"
                local_issue_edit "$ISSUE" remove "$OWNERSHIP_LABEL" || true
            fi
        elif [[ "$GH_AVAILABLE" == "true" ]]; then
            echo "Removing 'in-progress' label from #$ISSUE"
            gh issue edit "$ISSUE" --remove-label "in-progress" 2>/dev/null || true
            if [[ -n "${AI_WORKER_ID:-}" ]]; then
                OWNERSHIP_LABEL="W${AI_WORKER_ID}"
                echo "Removing ownership label '$OWNERSHIP_LABEL' from #$ISSUE"
                gh issue edit "$ISSUE" --remove-label "$OWNERSHIP_LABEL" 2>/dev/null || true
            fi
        fi
    fi
done || true

# Extract and process Reopens (title line only)
# Supports both GitHub issues (#N) and local issues (#LN)
echo "$TITLE_LINE" | grep -oE 'Reopens #L?[0-9]+' | grep -oE 'L?[0-9]+' | while read -r ISSUE; do
    if [[ -n "$ISSUE" ]]; then
        if is_local_issue "$ISSUE"; then
            echo "Reopening local #$ISSUE"
            local_issue_reopen "$ISSUE" || true
        elif [[ "$GH_AVAILABLE" == "true" ]]; then
            echo "Reopening #$ISSUE"
            gh issue reopen "$ISSUE" 2>/dev/null || true
        fi
    fi
done || true

# Extract and process auto-close keywords - clean workflow labels only after issue is closed (#1005)
# Case-insensitive match: Fix/Close/Resolve variants (any tense).
# Keywords can appear in body; match full message to mirror GitHub auto-close behavior.
# Repo-qualified refs (owner/repo#N) are only applied when they match the current repo.
# Supports both GitHub issues (#N) and local issues (#LN)
# - Title format: "[M]N: Fixes #<number>, Fixes #<number>"
# - commit-msg-hook normalizes "Fixes #<number>, #<number>" to "Fixes #<number>, Fixes #<number>"
# - Label cleanup is guarded: only runs if issue is actually closed (prevents label loss on malformed messages)
AUTO_CLOSE_KEYWORDS="Fixes|Fix|Fixed|Closes|Close|Closed|Resolves|Resolve|Resolved"
REPO_REF_PATTERN='[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+'
# Updated pattern to also match local issue IDs (L-prefixed)
AUTO_CLOSE_REF_PATTERN="(${REPO_REF_PATTERN})?#L?[0-9]+"
AUTO_CLOSE_PATTERN="(^|[^[:alnum:]_])(${AUTO_CLOSE_KEYWORDS}):?[[:space:]]*${AUTO_CLOSE_REF_PATTERN}"
echo "$MSG" | grep -oiE "$AUTO_CLOSE_PATTERN" | grep -oE "$AUTO_CLOSE_REF_PATTERN" | while read -r ISSUE_REF; do
    ISSUE_NUM="${ISSUE_REF##*#}"
    ISSUE_REPO="${ISSUE_REF%#*}"
    if [[ "$ISSUE_REF" == *"/"* ]]; then
        if [[ -z "$REPO_SLUG" || "$ISSUE_REPO" != "$REPO_SLUG" ]]; then
            continue
        fi
    fi
    ISSUE="$ISSUE_NUM"
    if [[ -n "$ISSUE" ]]; then
        # Handle local issues differently - close and clean labels locally
        if is_local_issue "$ISSUE"; then
            echo "Closing local issue #$ISSUE"
            local_issue_close "$ISSUE" || true
            # Clean workflow labels from local issue
            local_issue_edit "$ISSUE" remove "needs-review" || true
            local_issue_edit "$ISSUE" remove "do-audit" || true
            local_issue_edit "$ISSUE" remove "in-progress" || true
            continue
        fi

        # GitHub issue handling (requires gh)
        if [[ "$GH_AVAILABLE" != "true" ]]; then
            continue
        fi

        # Guard: only clean most labels if issue is actually closed (#1005)
        # This prevents label loss if a malformed message slips through the commit-msg hook
        # Exception: needs-review is cleaned preemptively (#1878) - see below
        ISSUE_STATE=""
        if [[ -n "$REPO_SLUG" ]]; then
            ISSUE_STATE=$(gh api "/repos/${REPO_SLUG}/issues/${ISSUE}" -q '.state' 2>/dev/null || true)
        fi

        # Preemptively remove needs-review on Fixes commits (#1878)
        # The issue will auto-close on push, but this hook runs at commit time (before push).
        # Without this, needs-review persists on closed issues until manually removed.
        # Safe to run regardless of state - if issue ends up not closing, cleanup is harmless.
        echo "Preemptively removing needs-review from #$ISSUE (Fixes commit)"
        gh issue edit "$ISSUE" --remove-label "needs-review" 2>/dev/null || true

        if [[ "$ISSUE_STATE" != "closed" ]]; then
            echo "Skipping remaining label cleanup for #$ISSUE (state: ${ISSUE_STATE:-unknown}, not closed yet)"
            continue
        fi
        echo "Cleaning remaining workflow labels from #$ISSUE (confirmed closed)"
        gh issue edit "$ISSUE" --remove-label "do-audit" 2>/dev/null || true
        gh issue edit "$ISSUE" --remove-label "in-progress" 2>/dev/null || true
        # Use REST API (not GraphQL via gh issue view) to get labels (#1074)
        if [[ -n "$REPO_SLUG" ]]; then
            ISSUE_LABELS=$(gh api "/repos/${REPO_SLUG}/issues/${ISSUE}" -q '.labels[].name' 2>/dev/null || true)
        else
            ISSUE_LABELS=""
        fi
        while read -r label; do
            # Remove legacy combined labels (in-progress-W1, etc.)
            if [[ -n "$label" && "$label" == in-progress-W* ]]; then
                gh issue edit "$ISSUE" --remove-label "$label" 2>/dev/null || true
            fi
            # Remove orthogonal ownership labels (W1-W5, prov1-prov3, R1-R3, M1-M3)
            if [[ -n "$label" && "$label" =~ ^(W[1-5]|prov[1-3]|R[1-3]|M[1-3])$ ]]; then
                gh issue edit "$ISSUE" --remove-label "$label" 2>/dev/null || true
            fi
        done <<< "$ISSUE_LABELS"
        # Invalidate historical cache since issue is now closed (#1277)
        # Path format: ~/.ait_gh_cache/historical/<owner>/<repo>/issue-<num>.json
        if [[ -n "$REPO_SLUG" ]]; then
            CACHE_FILE="$HOME/.ait_gh_cache/historical/${REPO_SLUG}/issue-${ISSUE}.json"
            if [[ -f "$CACHE_FILE" ]]; then
                rm -f "$CACHE_FILE" 2>/dev/null || true
                echo "Invalidated historical cache for #$ISSUE"
            fi
        fi
    fi
done || true

# === Worker File Tracking: Clear Committed Files ===
# In multi-worker mode, remove committed files from the tracker
if [[ -n "${AI_WORKER_ID:-}" ]]; then
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    TRACKER_FILE="$REPO_ROOT/.worker_${AI_WORKER_ID}_files.json"

    if [[ -f "$TRACKER_FILE" ]] && command -v python3 &>/dev/null; then
        # Get list of files in this commit
        COMMITTED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null || true)

        if [[ -n "$COMMITTED_FILES" ]]; then
            # Update tracker to remove committed files
            python3 -c "
import json
import sys

tracker_file = '$TRACKER_FILE'
committed = '''$COMMITTED_FILES'''.strip().split('\n')
committed_set = set(f.strip() for f in committed if f.strip())

try:
    with open(tracker_file) as f:
        data = json.load(f)

    original_files = set(data.get('files', []))
    remaining_files = sorted(original_files - committed_set)

    if len(remaining_files) != len(original_files):
        data['files'] = remaining_files
        with open(tracker_file, 'w') as f:
            json.dump(data, f, indent=2)
        removed = len(original_files) - len(remaining_files)
        print(f'[file_tracker] Cleared {removed} committed file(s) from tracker', file=sys.stderr)
except Exception as e:
    # Non-fatal - tracker will be refreshed next iteration
    pass
" 2>/dev/null || true
        fi
    fi
fi

exit 0

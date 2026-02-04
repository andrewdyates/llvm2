#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# pre-push-hook.sh - Validates benchmark claims in commits before push
#
# This hook validates benchmark claims (e.g., "39/55 CHC") in commit messages
# against results files in evals/results/. By default, it warns on INVALID
# claims but allows the push to proceed.
#
# Usage:
#   Install: git config core.hooksPath includes pre-push, or copy to .git/hooks/
#   Strict mode: Set CLAIM_VALIDATION_STRICT=1 to block push on INVALID claims
#
# Environment variables:
#   CLAIM_VALIDATION_STRICT - Set to "1" to block push on INVALID claims
#   CLAIM_VALIDATION_SKIP   - Set to "1" to skip validation entirely
#   CLAIM_VALIDATION_ROOT   - Override results root (default: evals/results)

set -euo pipefail

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }

# Skip if explicitly disabled
if [[ "${CLAIM_VALIDATION_SKIP:-}" == "1" ]]; then
    exit 0
fi

# Find repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    error "Not in a git repository"
    exit 1
}

# Locate validate_claim.py script
VALIDATE_SCRIPT="$REPO_ROOT/ai_template_scripts/validate_claim.py"
if [[ ! -f "$VALIDATE_SCRIPT" ]]; then
    # Fallback to legacy location for repos not yet synced
    VALIDATE_SCRIPT="$REPO_ROOT/scripts/validate_claim.py"
    if [[ ! -f "$VALIDATE_SCRIPT" ]]; then
        # Script not present in this repo - skip validation
        exit 0
    fi
fi

# Results root can be overridden
RESULTS_ROOT="${CLAIM_VALIDATION_ROOT:-$REPO_ROOT/evals/results}"

# Read stdin for refs being pushed
# Format: <local ref> <local sha> <remote ref> <remote sha>
INVALID_FOUND=0
UNKNOWN_FOUND=0
ERROR_FOUND=0
COMMITS_CHECKED=0

indent_stream() {
    while IFS= read -r line; do
        printf '  %s\n' "$line"
    done
}

while read -r _local_ref local_sha _remote_ref remote_sha; do
    # Skip delete pushes
    if [[ "$local_sha" == "0000000000000000000000000000000000000000" ]]; then
        continue
    fi

    # Determine commit range to check
    if [[ "$remote_sha" == "0000000000000000000000000000000000000000" ]]; then
        # New branch - check all commits not in any remote branch
        range="$local_sha"
        # Only check commits not already on remote
        commits=$(git rev-list "$range" --not --remotes 2>/dev/null || echo "$local_sha")
    else
        # Existing branch - check new commits
        range="$remote_sha..$local_sha"
        commits=$(git rev-list "$range" 2>/dev/null || true)
    fi

    # Validate claims in each commit
    for commit in $commits; do
        COMMITS_CHECKED=$((COMMITS_CHECKED + 1))

        # Run validate_claim.py, capture output and exit status
        set +e
        output=$(python3 "$VALIDATE_SCRIPT" \
            --commit "$commit" \
            --results-root "$RESULTS_ROOT" 2>&1)
        status=$?
        set -e

        if [[ $status -ne 0 ]] && ! echo "$output" | grep -qE "^(INVALID|UNKNOWN)"; then
            ERROR_FOUND=$((ERROR_FOUND + 1))
            echo ""
            error "Claim validation failed for commit ${commit:0:7}:"
            if [[ -n "$output" ]]; then
                printf '%s\n' "$output" | indent_stream
            else
                echo "  (no output)"
            fi
            continue
        fi

        # Check for INVALID or UNKNOWN in output
        if echo "$output" | grep -q "^INVALID"; then
            INVALID_FOUND=$((INVALID_FOUND + 1))
            echo ""
            warn "Claim validation issue in commit ${commit:0:7}:"
            printf '%s\n' "$output" | grep -E "^(INVALID|UNKNOWN|Summary)" | indent_stream
        elif echo "$output" | grep -q "^UNKNOWN"; then
            UNKNOWN_FOUND=$((UNKNOWN_FOUND + 1))
            if [[ "${CLAIM_VALIDATION_STRICT:-}" == "1" ]]; then
                echo ""
                warn "Unverified claim in commit ${commit:0:7}:"
                printf '%s\n' "$output" | grep -E "^(UNKNOWN|Summary)" | indent_stream
            fi
        fi
    done
done

# Report results
if [[ $COMMITS_CHECKED -eq 0 ]]; then
    exit 0
fi

echo ""
if [[ $INVALID_FOUND -gt 0 ]]; then
    if [[ "${CLAIM_VALIDATION_STRICT:-}" == "1" ]]; then
        error "$INVALID_FOUND commit(s) have INVALID benchmark claims"
        error "Push blocked. Fix claims or set CLAIM_VALIDATION_SKIP=1 to bypass."
        exit 1
    else
        warn "$INVALID_FOUND commit(s) have INVALID benchmark claims"
        warn "Consider fixing before claims propagate. Set CLAIM_VALIDATION_STRICT=1 to block."
    fi
fi
if [[ $ERROR_FOUND -gt 0 ]]; then
    if [[ "${CLAIM_VALIDATION_STRICT:-}" == "1" ]]; then
        error "$ERROR_FOUND commit(s) could not be validated"
        error "Push blocked. Fix validation errors or set CLAIM_VALIDATION_SKIP=1 to bypass."
        exit 1
    else
        warn "$ERROR_FOUND commit(s) could not be validated"
        warn "Fix validation errors or set CLAIM_VALIDATION_STRICT=1 to block."
    fi
fi
if [[ $INVALID_FOUND -eq 0 && $ERROR_FOUND -eq 0 ]]; then
    if [[ $UNKNOWN_FOUND -gt 0 && "${CLAIM_VALIDATION_STRICT:-}" == "1" ]]; then
        warn "$UNKNOWN_FOUND commit(s) have UNKNOWN (unverified) benchmark claims"
        warn "Add results files or set CLAIM_VALIDATION_SKIP=1 to bypass."
        # UNKNOWN in strict mode is a warning, not a block (per design)
    fi
fi

if [[ $INVALID_FOUND -eq 0 && $UNKNOWN_FOUND -eq 0 && $ERROR_FOUND -eq 0 ]]; then
    ok "All benchmark claims validated ($COMMITS_CHECKED commits checked)"
fi

exit 0

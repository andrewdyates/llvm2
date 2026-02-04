#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# init_labels.sh - Create required GitHub labels for AI workflow.
#
# Usage:
#   ./ai_template_scripts/init_labels.sh
#   ./ai_template_scripts/init_labels.sh --dry-run
#   ./ai_template_scripts/init_labels.sh --version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "init_labels.sh ${git_hash} (${date})"
    exit 0
}

case "${1:-}" in
    --version) version ;;
esac

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: init_labels.sh [--dry-run]"
    echo ""
    echo "Create required GitHub labels for AI workflow."
    echo ""
    echo "Options:"
    echo "  --dry-run     Show what would be done without making changes"
    echo "  --version     Show version information"
    echo "  -h, --help    Show this help message"
    exit 0
fi

if [[ "$#" -gt 1 ]]; then
    echo "ERROR: Too many arguments" >&2
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ -n "${1:-}" ]]; then
    echo "ERROR: Unknown option: ${1}" >&2
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi

[[ -f "CLAUDE.md" ]] || { echo "ERROR: Run from project root" >&2; exit 1; }

# Get repo from origin remote (not gh repo view, which can pick wrong repo with multiple remotes)
ORIGIN_URL=$(git remote get-url origin 2>/dev/null) || {
    echo "ERROR: No 'origin' remote found. Are you in a git repo?" >&2
    exit 1
}
# Parse owner/repo safely - use url_sanitizer to avoid credential leakage (#2240)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=$(python3 "$SCRIPT_DIR/url_sanitizer.py" --repo-slug "$ORIGIN_URL" 2>/dev/null)
if [[ -z "$REPO" || "$REPO" == "[REDACTED]" ]]; then
    # Sanitize URL before error message to avoid leaking credentials
    SAFE_URL=$(python3 "$SCRIPT_DIR/url_sanitizer.py" "$ORIGIN_URL" 2>/dev/null || echo "[REDACTED]")
    echo "ERROR: Could not parse repo from origin URL: $SAFE_URL" >&2
    exit 1
fi

LABELS=(
    # Priority labels
    "P0|B60205|System compromised"
    "P1|D93F0B|High priority"
    "P2|FBCA04|Medium priority"
    "P3|0E8A16|Low priority"
    # Workflow state labels (mutually exclusive)
    "urgent|D93F0B|Work on NOW"
    "in-progress|1D76DB|Currently claimed"
    "do-audit|FBCA04|Ready for self-audit (Worker only)"
    # Ownership labels (orthogonal to state - kept through workflow)
    # Worker ownership (W1-W5)
    "W1|C5DEF5|Owned by Worker 1"
    "W2|C5DEF5|Owned by Worker 2"
    "W3|C5DEF5|Owned by Worker 3"
    "W4|C5DEF5|Owned by Worker 4"
    "W5|C5DEF5|Owned by Worker 5"
    # Prover ownership uses prov1-prov3 (to avoid conflict with P1-P3 priority labels)
    "prov1|C5DEF5|Owned by Prover 1"
    "prov2|C5DEF5|Owned by Prover 2"
    "prov3|C5DEF5|Owned by Prover 3"
    # Researcher ownership (R1-R3)
    "R1|C5DEF5|Owned by Researcher 1"
    "R2|C5DEF5|Owned by Researcher 2"
    "R3|C5DEF5|Owned by Researcher 3"
    # Manager ownership (M1-M3)
    "M1|C5DEF5|Owned by Manager 1"
    "M2|C5DEF5|Owned by Manager 2"
    "M3|C5DEF5|Owned by Manager 3"
    # Legacy combined labels (kept for backward compatibility)
    "in-progress-W1|1D76DB|Claimed by Worker 1"
    "in-progress-W2|1D76DB|Claimed by Worker 2"
    "in-progress-W3|1D76DB|Claimed by Worker 3"
    "in-progress-W4|1D76DB|Claimed by Worker 4"
    "in-progress-W5|1D76DB|Claimed by Worker 5"
    "in-progress-P1|1D76DB|Claimed by Prover 1"
    "in-progress-P2|1D76DB|Claimed by Prover 2"
    "in-progress-P3|1D76DB|Claimed by Prover 3"
    "in-progress-R1|1D76DB|Claimed by Researcher 1"
    "in-progress-R2|1D76DB|Claimed by Researcher 2"
    "in-progress-R3|1D76DB|Claimed by Researcher 3"
    "in-progress-M1|1D76DB|Claimed by Manager 1"
    "in-progress-M2|1D76DB|Claimed by Manager 2"
    "in-progress-M3|1D76DB|Claimed by Manager 3"
    "tracking|D4C5F9|Monitor, don't schedule"
    "deferred|C5DEF5|Considered but not now - query when seeking new work"
    "needs-review|FBCA04|Ready for Manager review"
    "urgent-handoff|D93F0B|Urgently needs target role (skip delay)"
    "blocked|D4C5F9|Waiting on dependency"
    "environmental|D4C5F9|Environment setup issue (no code fix)"
    "stale|D4C5F9|Auto-generated child issue obsoleted by parent closure"
    "duplicate|D4C5F9|Duplicate of another issue"
    "blocker-cycle|B60205|Circular dependency requiring USER decision"
    "local-maximum|B60205|Stuck at local maximum - needs USER architecture decision"
    "mail|1D76DB|Cross-repo message (auto-added by gh wrapper)"
    "notification|C5DEF5|Auto-generated closure notification"
    # Issue type labels (informational, not for scheduling)
    "bug|D93F0B|Bug fix"
    "feature|0E8A16|New feature"
    "documentation|0075CA|Documentation update"
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Would create labels:"
    for entry in "${LABELS[@]}"; do
        IFS='|' read -r name color description <<< "$entry"
        echo "  $name ($color) - $description"
    done
    exit 0
fi

create_label() {
    local name="$1"
    local color="$2"
    local description="$3"
    local output

    if output=$(gh label create "$name" --repo "$REPO" --color "$color" --description "$description" 2>&1); then
        return 0
    fi

    if [[ "$output" == *"already exists"* ]]; then
        return 0
    fi

    echo "WARNING: Failed to create label '$name': $output" >&2
    return 1
}

label_errors=0
for entry in "${LABELS[@]}"; do
    IFS='|' read -r name color description <<< "$entry"
    create_label "$name" "$color" "$description" || ((label_errors++))
done

if [[ "$label_errors" -gt 0 ]]; then
    echo "WARNING: $label_errors label(s) failed to create" >&2
    exit 1
fi

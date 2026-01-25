#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# init_labels.sh - Create required GitHub labels for AI workflow.
#
# Usage:
#   ./ai_template_scripts/init_labels.sh
#   ./ai_template_scripts/init_labels.sh --dry-run

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [--dry-run]"
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

LABELS=(
    "P0|B60205|System compromised"
    "P1|D93F0B|High priority"
    "P2|FBCA04|Medium priority"
    "P3|0E8A16|Low priority"
    "urgent|D93F0B|Work on NOW"
    "in-progress|1D76DB|Currently claimed"
    "do-audit|FBCA04|Ready for self-audit (Worker only)"
    "tracking|D4C5F9|Monitor, don't schedule"
    "escalate|B60205|USER decision needed"
    "needs-review|FBCA04|Ready for Manager review"
    "blocked|D4C5F9|Waiting on dependency"
    "blocker-cycle|B60205|Circular dependency requiring USER decision"
    "local-maximum|B60205|Stuck at local maximum - needs USER architecture decision"
    "mail|1D76DB|Cross-repo message (auto-added by gh wrapper)"
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

    if output=$(gh label create "$name" --color "$color" --description "$description" 2>&1); then
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

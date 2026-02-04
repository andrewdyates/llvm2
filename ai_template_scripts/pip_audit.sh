#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# pip_audit.sh - Scan requirements.txt for known vulnerabilities.
#
# Usage:
#   ./ai_template_scripts/pip_audit.sh
#   ./ai_template_scripts/pip_audit.sh --fix  # Auto-fix if possible
#   ./ai_template_scripts/pip_audit.sh --version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "pip_audit.sh ${git_hash} (${date})"
    exit 0
}

case "${1:-}" in
    --version) version ;;
esac

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: pip_audit.sh [--fix]"
    echo ""
    echo "Scan requirements.txt for known vulnerabilities."
    echo ""
    echo "Options:"
    echo "  --fix         Attempt to auto-fix vulnerabilities by updating requirements.txt"
    echo "  --version     Show version information"
    echo "  -h, --help    Show this help message"
    exit 0
fi

if [[ "$#" -gt 1 ]]; then
    echo "ERROR: Too many arguments" >&2
    echo "Usage: $0 [--fix]" >&2
    exit 1
fi

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
elif [[ -n "${1:-}" ]]; then
    echo "ERROR: Unknown option: ${1}" >&2
    echo "Usage: $0 [--fix]" >&2
    exit 1
fi

[[ -f "CLAUDE.md" ]] || { echo "ERROR: Run from project root" >&2; exit 1; }

if ! command -v pip-audit &> /dev/null; then
    echo "ERROR: pip-audit not installed. Install with: pip install pip-audit" >&2
    exit 1
fi

REQUIREMENTS_FILE="requirements.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "WARNING: No $REQUIREMENTS_FILE found, nothing to audit."
    exit 0
fi

echo "Scanning $REQUIREMENTS_FILE for known vulnerabilities..."
echo ""

if [[ "$FIX_MODE" == "true" ]]; then
    pip-audit -r "$REQUIREMENTS_FILE" --fix
else
    pip-audit -r "$REQUIREMENTS_FILE"
fi

echo ""
echo "pip-audit scan complete."

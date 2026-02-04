#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# find_ignores.sh - Find forbidden test ignores in a codebase
#
# CANONICAL SOURCE: dropbox-ai-prototypes/ai_template
# DO NOT EDIT in other repos - file issues to ai_template for changes.
#
# Usage: ./find_ignores.sh [directory]
#        ./find_ignores.sh --version
#
# Tests must PASS, FAIL, or be DELETED. No ignore. No hiding.
# This script helps migrate existing codebases by finding all ignores.
#
# Default exclusions: reference/, vendor/, third_party/, node_modules/, target/, .git/,
#                     .venv*/, venv/, env/, site-packages/, .tox/, __pycache__/, .mypy_cache/, .pytest_cache/
# Additional exclusions via .ignore-check-exclude file or IGNORE_CHECK_EXCLUDE env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "find_ignores.sh ${git_hash} (${date})"
    exit 0
}

case "${1:-}" in
--version) version ;;
-h | --help)
    echo "Usage: find_ignores.sh [directory]"
    echo ""
    echo "Find forbidden test ignores in a codebase."
    echo ""
    echo "Options:"
    echo "  --version     Show version information"
    echo "  -h, --help    Show this help message"
    exit 0
    ;;
esac

DIR="${1:-.}"
FOUND=0

# Build exclusion args for grep
EXCLUDE_ARGS=()

# Default exclusions for non-production directories
# These are commonly used for vendored code, reference implementations,
# build artifacts, and generated content that we don't test.
DEFAULT_EXCLUDES=(
    "reference"     # Reference implementations, not our code
    "vendor"        # Vendored third-party code
    "third_party"   # Another common name for vendored code
    "node_modules"  # JS dependencies
    "target"        # Rust build output
    ".git"          # Git internals
    ".venv*"        # Python virtualenv (#1488, #1822 - glob covers .venv, .venv_310, etc.)
    "venv"          # Alternative virtualenv name
    "env"           # Another common virtualenv name (#1822)
    "site-packages" # Python packages (inside virtualenvs)
    ".tox"          # Tox testing environments (#1822)
    "__pycache__"   # Python bytecode cache
    ".mypy_cache"   # Mypy type checking cache
    ".pytest_cache" # Pytest cache
)

for excl in "${DEFAULT_EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=("--exclude-dir=$excl")
done

# Load from .ignore-check-exclude file
if [[ -f "$DIR/.ignore-check-exclude" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        EXCLUDE_ARGS+=("--exclude-dir=${line%/}")
    done <"$DIR/.ignore-check-exclude"
fi

# Load from environment variable
if [[ -n "${IGNORE_CHECK_EXCLUDE:-}" ]]; then
    IFS=':' read -ra env_excludes <<<"$IGNORE_CHECK_EXCLUDE"
    for excl in "${env_excludes[@]}"; do
        [[ -n "$excl" ]] && EXCLUDE_ARGS+=("--exclude-dir=${excl%/}")
    done
fi

echo "Scanning for forbidden test ignores in: $DIR"
echo "=========================================="
echo ""

# Rust: #[ignore] or #[ignore = "..."]
echo "=== Rust (#[ignore]) ==="
if grep -rn -E '#\[ignore[[:space:]]*(\]|=)' --include="*.rs" ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} "$DIR" 2>/dev/null; then
    FOUND=1
else
    echo "(none found)"
fi
echo ""

# Python: @skip(), @pytest.mark.skip, @pytest.mark.xfail, @unittest.skip*
echo "=== Python (@skip, @pytest.mark.skip/xfail, @unittest.skip) ==="
if grep -rn -E '@skip[[:space:]]*\(|@pytest\.mark\.skip|@pytest\.mark\.xfail|@unittest\.skip' --include="*.py" ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} "$DIR" 2>/dev/null; then
    FOUND=1
else
    echo "(none found)"
fi
echo ""

# JS/TS: .skip(, xit(, xdescribe(, xtest(
echo "=== JavaScript/TypeScript (.skip, xit, xdescribe, xtest) ==="
if grep -rn -E '\.skip\(|^[[:space:]]*(xit|xdescribe|xtest)[[:space:]]*\(' --include="*.js" --include="*.ts" --include="*.tsx" ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} "$DIR" 2>/dev/null; then
    FOUND=1
else
    echo "(none found)"
fi
echo ""

echo "=========================================="
if [[ $FOUND -eq 1 ]]; then
    echo "VIOLATIONS FOUND - Remove ignores or delete tests"
    echo ""
    echo "For each ignore:"
    echo "  - Slow? Add timeout instead. Let it fail if too slow."
    echo "  - Blocked? Let it fail. Failure is visibility."
    echo "  - Flaky? Fix the flakiness or delete the test."
    echo "  - Obsolete? Delete the test."
    exit 1
else
    echo "No forbidden ignores found."
    exit 0
fi

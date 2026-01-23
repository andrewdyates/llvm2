#!/usr/bin/env bash
# init_from_template.sh - Enforce required project setup
#
# Creates GitHub labels and cleans up ai_template-specific files.
# Run once after creating a new project from ai_template.
#
# Usage:
#   ./ai_template_scripts/init_from_template.sh             # Full init
#   ./ai_template_scripts/init_from_template.sh --verify    # Check placeholders
#   ./ai_template_scripts/init_from_template.sh --check-deps # Check dependencies only
#
# Copyright 2026 Dropbox, Inc.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -f "CLAUDE.md" ]] || { echo "ERROR: Run from project root"; exit 1; }

# --check-deps mode: just check dependencies
if [[ "${1:-}" == "--check-deps" ]]; then
    exec "$SCRIPT_DIR/check_deps.py"
fi

# --verify mode: check placeholders
if [[ "${1:-}" == "--verify" ]]; then
    if grep -q '<[^>]*>' CLAUDE.md; then
        echo "ERROR: Unreplaced placeholders in CLAUDE.md:"
        grep -n '<[^>]*>' CLAUDE.md
        exit 1
    fi
    echo "OK: All placeholders replaced"

    # Also check dependencies
    echo ""
    echo "Checking dependencies..."
    "$SCRIPT_DIR/check_deps.py" || echo "WARNING: Some dependencies missing"
    exit 0
fi

echo "Initializing project from ai_template..."

# Delete ai_template-specific directories
rm -rf .ai_template_self
rm -rf worker_logs

# Delete ai_template-specific files
rm -f docs/plans/PLAN_bot_identity.md

# Delete ai_template-specific post-mortems (keep TEMPLATE.md)
find postmortems -name '*.md' ! -name 'TEMPLATE.md' -delete 2>/dev/null || true

# Delete ai_template-specific tests (keep test_example.py and conftest.py)
find tests -name '*.py' ! -name 'test_example.py' ! -name 'conftest.py' ! -name '__init__.py' -delete 2>/dev/null || true

# Flush IDEAS.md to template
cat > IDEAS.md << 'EOF'
# Ideas

Future considerations and backlog items. These are NOT actionable tasks.

**For AI:** Do not convert these to issues or work on them unless explicitly asked.
**For humans:** Add ideas freely. Promote to issues when ready to implement.

---

EOF

# Create VISION.md template
cat > VISION.md << 'EOF'
# Vision: <project>

> Living strategic document. Updated by Researcher, approved by User.
> Last updated: <date>

## Current Phase

What we're building toward right now. Not tasks - direction.

## Architecture

How the system works. Key components, data flow, boundaries.

## Key Decisions

Active architectural choices and their rationale.

## Open Questions

Unresolved strategic decisions that affect multiple issues.
EOF

# Flush README.md to template
cat > README.md << 'EOF'
# Project Name

One-line description.

## Motivation

Why this project exists. What problem it solves.

## Goal

What success looks like. The end state we're building toward.

## Setup

```bash
# Setup instructions
```

## Usage

```bash
# Usage examples
```
EOF

# Create required GitHub labels for AI workflow
gh label create P0 --color b60205 --description "critical" 2>/dev/null || true
gh label create P1 --color d93f0b --description "high" 2>/dev/null || true
gh label create P2 --color fbca04 --description "medium" 2>/dev/null || true
gh label create P3 --color 0e8a16 --description "low" 2>/dev/null || true
gh label create in-progress --color 5319e7 --description "in progress" 2>/dev/null || true
gh label create blocked --color 000000 --description "blocked" 2>/dev/null || true
gh label create needs-review --color c5def5 --description "worker flagged as done, needs manager review" 2>/dev/null || true
gh label create mail --color 1d76db --description "Inter-project mail message" 2>/dev/null || true

# Check dependencies
echo ""
echo "Checking dependencies..."
"$SCRIPT_DIR/check_deps.py" || echo "WARNING: Some dependencies missing (install with check_deps.py --fix)"

# Ensure cargo wrapper is executable and lock dir exists
chmod +x "$SCRIPT_DIR/bin/cargo" 2>/dev/null || true
mkdir -p ~/.ait_cargo_lock

echo ""
echo "Done. Next: edit CLAUDE.md, then run --verify"

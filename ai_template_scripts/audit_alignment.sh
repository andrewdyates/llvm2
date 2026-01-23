#!/bin/bash
# audit_alignment.sh - Check template alignment without breaking things
#
# Guides AIs through what should/shouldn't exist in a child repo.
# - Detects missing required template files
# - Flags obsolete files that should be deleted
# - Catches forbidden files (e.g., CI workflows)
# - Warns about runtime files that look deletable but aren't

set -e

# Parse args
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "=== Template Alignment Audit ==="
[[ "$CLEAN" == "true" ]] && echo "(--clean mode: will delete obsolete/forbidden files)"
echo

# Files that SHOULD exist (from template)
# Keep in sync with sync_repo.sh
REQUIRED=(
    # Core project files
    "CLAUDE.md"
    "looper.py"
    # Rules
    ".claude/rules/ai_template.md"
    ".claude/rules/org_chart.md"
    # Role prompts (used by looper.py)
    ".claude/roles/shared.md"
    ".claude/roles/worker.md"
    ".claude/roles/prover.md"
    ".claude/roles/researcher.md"
    ".claude/roles/manager.md"
    # Key scripts
    "ai_template_scripts/bin/gh"
    "ai_template_scripts/commit-msg-hook.sh"
    "ai_template_scripts/spawn_session.sh"
    "ai_template_scripts/pulse.py"
    # Config
    "ruff.toml"
    ".gitignore"
)

# Files that should NOT exist (obsolete)
# Keep in sync with sync_repo.sh OLD_TEMPLATE_FILES
OBSOLETE=(
    "IDEAS.md"                                  # Migrated to ideas/
    "run_worker.sh"                             # Replaced by looper.py
    "run_loop.py"                               # Renamed to looper.py
    "run_loop_context.md"                       # Renamed to looper_context.md
    "tests/test_run_loop.py"                    # Renamed to tests/test_looper.py
    "ai_template_scripts/gh_discussion.sh"      # Replaced by gh_discussion.py
    "ai_template_scripts/gh_post.sh"            # Replaced by gh_post.py
    "ai_template_scripts/init.sh"               # Renamed to install_dev_tools.sh
    "ai_template_scripts/create_github_apps.py" # Orphaned, never part of template
)

# Patterns that should NOT exist (forbidden)
FORBIDDEN_PATTERNS=(
    ".github/workflows/*.yml"   # No CI/CD - GitHub Actions not available
    ".github/workflows/*.yaml"
)

# Check if we're in ai_template itself (has .ai_template_self)
IS_TEMPLATE=false
[[ -d ".ai_template_self" ]] && IS_TEMPLATE=true

# .ai_template_self should only exist in ai_template, not child repos
if ! $IS_TEMPLATE; then
    OBSOLETE+=(".ai_template_self")
fi

# Files that look deletable but AREN'T
KEEP=(
    "worker_logs"           # Used by looper.py - contains iteration logs
    "AGENTS.md"             # Used by Codex
    "GEMINI.md"             # Used by Gemini
    ".mcp.json"             # MCP config
)

echo -e "${GREEN}Required files:${NC}"
missing=0
for f in "${REQUIRED[@]}"; do
    if [[ -e "$f" ]]; then
        echo "  ✓ $f"
    else
        echo -e "  ${RED}✗ $f (MISSING)${NC}"
        ((missing++)) || true
    fi
done
echo

echo -e "${YELLOW}Obsolete files (should delete):${NC}"
found_obsolete=0
for f in "${OBSOLETE[@]}"; do
    if [[ -e "$f" ]]; then
        if [[ "$CLEAN" == "true" ]]; then
            rm -rf "$f"
            echo -e "  ${GREEN}✓ $f (DELETED)${NC}"
        else
            echo -e "  ${RED}! $f (DELETE THIS)${NC}"
        fi
        ((found_obsolete++)) || true
    fi
done
[[ $found_obsolete -eq 0 ]] && echo "  (none found)"
echo

echo -e "${RED}Forbidden files (violates template rules):${NC}"
found_forbidden=0
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    # Use nullglob to handle no matches gracefully
    shopt -s nullglob
    for f in $pattern; do
        if [[ "$CLEAN" == "true" ]]; then
            rm -f "$f"
            echo -e "  ${GREEN}✓ $f (DELETED)${NC}"
        else
            echo -e "  ${RED}! $f (CI/CD not supported - DELETE)${NC}"
        fi
        ((found_forbidden++)) || true
    done
    shopt -u nullglob
done
[[ $found_forbidden -eq 0 ]] && echo "  (none found)"
echo

echo -e "${GREEN}DO NOT DELETE (runtime files):${NC}"
for f in "${KEEP[@]}"; do
    if [[ -e "$f" ]]; then
        echo "  ✓ $f (exists, keep it)"
    else
        echo "  - $f (not present)"
    fi
done
echo

# Check CLAUDE.md for content that belongs elsewhere
echo -e "${YELLOW}CLAUDE.md content audit:${NC}"
claude_issues=0
if [[ -f "CLAUDE.md" ]]; then
    # Patterns that suggest content belongs elsewhere
    if grep -qi "## Roadmap\|## Current Tasks\|## Status\|## Progress" CLAUDE.md 2>/dev/null; then
        echo -e "  ${RED}! Contains status/progress content - use GitHub Issues for tasks${NC}"
        ((claude_issues++)) || true
    fi
    if grep -qi "## Anti-Patterns\|## Commit Structure\|## Session Start" CLAUDE.md 2>/dev/null; then
        echo -e "  ${RED}! Contains rules covered by ai_template.md - remove duplication${NC}"
        ((claude_issues++)) || true
    fi
    if grep -qi "\[x\]\|\[ \]\|TODO:\|DONE:\|RESOLVED:" CLAUDE.md 2>/dev/null; then
        echo -e "  ${RED}! Contains task tracking - move to GitHub Issues${NC}"
        ((claude_issues++)) || true
    fi
    if grep -qi "## WORKER\|## MANAGER\|## RESEARCHER\|## PROVER" CLAUDE.md 2>/dev/null; then
        echo -e "  ${RED}! Contains role definitions - covered by ai_template.md${NC}"
        ((claude_issues++)) || true
    fi
    [[ $claude_issues -eq 0 ]] && echo "  ✓ CLAUDE.md looks clean"
else
    echo "  - CLAUDE.md not found"
fi
echo

# Check for VISION.md
echo -e "${YELLOW}Strategic docs:${NC}"
vision_missing=0
if [[ -f "VISION.md" ]]; then
    echo "  ✓ VISION.md exists"
else
    echo -e "  ${YELLOW}! VISION.md missing - should document strategic direction${NC}"
    vision_missing=1
fi
echo

# Summary
echo "=== Summary ==="
exit_code=0
if [[ $missing -gt 0 ]]; then
    echo -e "${RED}Missing $missing required file(s) - run sync_repo.sh${NC}"
    exit_code=1
fi
if [[ $found_obsolete -gt 0 ]]; then
    if [[ "$CLEAN" == "true" ]]; then
        echo -e "${GREEN}Deleted $found_obsolete obsolete file(s)${NC}"
    else
        echo -e "${YELLOW}Found $found_obsolete obsolete file(s) - run with --clean to delete${NC}"
    fi
fi
if [[ $found_forbidden -gt 0 ]]; then
    if [[ "$CLEAN" == "true" ]]; then
        echo -e "${GREEN}Deleted $found_forbidden forbidden file(s)${NC}"
    else
        echo -e "${RED}Found $found_forbidden forbidden file(s) - DELETE IMMEDIATELY${NC}"
        exit_code=1
    fi
fi
if [[ $claude_issues -gt 0 ]]; then
    echo -e "${YELLOW}Found $claude_issues CLAUDE.md content issue(s) - clean up recommended${NC}"
    # CLAUDE.md issues are warnings, not errors
fi
if [[ $vision_missing -eq 1 ]]; then
    echo -e "${YELLOW}VISION.md missing - create to document strategic direction${NC}"
fi
if [[ $missing -eq 0 && $found_obsolete -eq 0 && $found_forbidden -eq 0 && $claude_issues -eq 0 && $vision_missing -eq 0 ]]; then
    echo -e "${GREEN}Template alignment looks good${NC}"
fi

# Point to migration checklist if any issues found
total_issues=$((found_obsolete + claude_issues + vision_missing))
if [[ $total_issues -gt 0 ]]; then
    echo
    echo -e "${YELLOW}See docs/MIGRATION_CHECKLIST.md for step-by-step alignment guide${NC}"
fi

exit $exit_code

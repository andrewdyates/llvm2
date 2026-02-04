#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# audit_alignment.sh - Check template alignment without breaking things
#
# Guides AIs through what should/shouldn't exist in a child repo.
# - Detects missing required template files
# - Flags obsolete files that should be deleted
# - Catches forbidden files (e.g., CI workflows)
# - Warns about runtime files that look deletable but aren't

set -euo pipefail

# Script directory for calling other scripts
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Version function
version() {
    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    local git_hash
    git_hash=$(git -C "$script_dir/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "audit_alignment.sh ${git_hash} (${date})"
    exit 0
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Check template alignment - detect missing, obsolete, and forbidden files."
    echo ""
    echo "Options:"
    echo "  --clean      Delete obsolete and forbidden files automatically"
    echo "  --version    Show version information"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Check alignment (no changes)"
    echo "  $0 --clean        # Check and delete obsolete/forbidden files"
    exit 0
}

# Parse args
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --version) version ;;
        -h|--help) usage ;;
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

# Verify we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Not in a git repository${NC}"
    exit 1
fi

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

# Patterns for obsolete root files (use glob expansion)
OBSOLETE_PATTERNS=(
    "MANAGER_DIRECTIVE*"                        # Legacy directive files
    "WORKER_DIRECTIVE*"                         # Legacy directive files
    "PLAN-*"                                    # Legacy plan files
)

# Patterns that should NOT exist (forbidden)
FORBIDDEN_PATTERNS=(
    ".github/workflows/*"       # No CI/CD - GitHub Actions not available
)

# Check if we're in ai_template itself (by git remote, not directory)
IS_TEMPLATE=false
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
[[ "$REMOTE_URL" =~ ai_template(\.git)?$ ]] && IS_TEMPLATE=true

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

# Check obsolete patterns (glob expansion)
shopt -s nullglob
for pattern in "${OBSOLETE_PATTERNS[@]}"; do
    for f in $pattern; do
        if [[ "$CLEAN" == "true" ]]; then
            rm -rf "$f"
            echo -e "  ${GREEN}✓ $f (DELETED)${NC}"
        else
            echo -e "  ${RED}! $f (DELETE THIS)${NC}"
        fi
        ((found_obsolete++)) || true
    done
done
shopt -u nullglob

[[ $found_obsolete -eq 0 ]] && echo "  (none found)"
echo

echo -e "${RED}Forbidden files (violates template rules):${NC}"
found_forbidden=0
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    # Use nullglob to handle no matches gracefully
    shopt -s nullglob
    for f in $pattern; do
        # Allowlist: DO_NOT_USE.md is a valid placeholder warning file
        [[ "${f##*/}" == "DO_NOT_USE.md" ]] && continue
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

    # Check for Director designation
    if ! grep -q '\*\*Director:\*\*' CLAUDE.md 2>/dev/null; then
        echo -e "  ${RED}! Missing **Director:** designation${NC}"
        ((claude_issues++)) || true
    else
        echo "  ✓ Director designation present"
    fi
else
    echo "  - CLAUDE.md not found"
fi
echo

# Check git hooks installed
echo -e "${YELLOW}Git hooks:${NC}"
hooks_missing=0
HOOKS_DIR=$(git rev-parse --git-path hooks 2>/dev/null || echo ".git/hooks")
[[ "$HOOKS_DIR" != /* ]] && HOOKS_DIR="$(pwd)/$HOOKS_DIR"

for hook in commit-msg pre-commit post-commit; do
    hook_file="$HOOKS_DIR/$hook"
    if [[ -f "$hook_file" && -x "$hook_file" && ! "$hook_file" =~ \.sample$ ]]; then
        echo "  ✓ $hook installed"
    elif [[ -f "$hook_file.sample" ]]; then
        echo -e "  ${RED}! $hook NOT installed (only .sample exists)${NC}"
        ((hooks_missing++)) || true
    else
        echo -e "  ${RED}! $hook NOT installed${NC}"
        ((hooks_missing++)) || true
    fi
done
echo

# Check required directories
echo -e "${YELLOW}Required directories:${NC}"
dirs_missing=0
if [[ -d "ideas" ]]; then
    echo "  ✓ ideas/ exists"
else
    echo -e "  ${YELLOW}! ideas/ missing - create for future considerations${NC}"
    ((dirs_missing++)) || true
fi
echo

# Check .looper_config.json
echo -e "${YELLOW}Looper configuration:${NC}"
looper_missing=0
if [[ -f ".looper_config.json" ]]; then
    echo "  ✓ .looper_config.json exists"
else
    echo -e "  ${YELLOW}! .looper_config.json missing - looper will use defaults${NC}"
    ((looper_missing++)) || true
fi
echo

# Check for forbidden test ignores (#341)
# Use find_ignores.sh which correctly reads .ignore-check-exclude
# Fix for #1912: audit_alignment.sh was ignoring project-specific exclusions
echo -e "${RED}Forbidden test ignores (FORBIDDEN per #341):${NC}"
ignore_count=0
if ! $IS_TEMPLATE; then
    # Call find_ignores.sh and capture output + exit code
    # find_ignores.sh correctly handles .ignore-check-exclude and env exclusions
    # Use subshell to capture exit code without set -e causing early exit
    find_ignores_exit=0
    find_ignores_output=$("$SCRIPT_DIR/find_ignores.sh" . 2>&1) || find_ignores_exit=$?

    if [[ $find_ignores_exit -eq 1 ]]; then
        # find_ignores.sh returns 1 when violations are found
        echo -e "  ${RED}! Found test ignores - FORBIDDEN per rule #341${NC}"
        # Extract counts from find_ignores.sh output
        # Count lines that start with actual file paths (not headers or empty lines)
        # Note: find_ignores.sh shows matched lines with "file:linenum:content" format
        rust_lines=$(echo "$find_ignores_output" | grep -c '\.rs:[0-9]*:' 2>/dev/null || true)
        python_lines=$(echo "$find_ignores_output" | grep -c '\.py:[0-9]*:' 2>/dev/null || true)
        js_lines=$(echo "$find_ignores_output" | grep -cE '\.(js|ts|tsx):[0-9]*:' 2>/dev/null || true)
        [[ $rust_lines -gt 0 ]] && echo -e "    ${RED}Rust (#[ignore]): $rust_lines match(es)${NC}"
        [[ $python_lines -gt 0 ]] && echo -e "    ${RED}Python (@skip): $python_lines match(es)${NC}"
        [[ $js_lines -gt 0 ]] && echo -e "    ${RED}JS/TS (.skip/xit): $js_lines match(es)${NC}"
        echo "    Tests must PASS, FAIL, or be DELETED. No ignores."
        echo "    Run: $SCRIPT_DIR/find_ignores.sh for details."
        ignore_count=1
    else
        echo "  ✓ No forbidden test ignores found"
    fi
else
    echo "  (skipped for ai_template itself)"
fi
echo

# Check for forbidden process-kill commands in worker logs (#932)
echo -e "${RED}Forbidden commands in logs:${NC}"
forbidden_cmd_count=0
if [[ -d "worker_logs" ]]; then
    # Patterns that are FORBIDDEN per ai_template.md
    # - pkill -f claude, pkill iTerm, pkill Terminal
    # - osascript -e 'tell application "..." to quit' / 'quit app'
    # - kill -9 (process termination)
    #
    # NOTE: Match commands that START with the forbidden pattern.
    # This avoids false positives from commit messages or docs that mention
    # these patterns. Real shell commands start with the executable name.
    FORBIDDEN_CMD_PATTERNS=(
        '"command":"pkill.*claude'
        '"command":"pkill.*iTerm'
        '"command":"pkill.*Terminal'
        '"command":"osascript.*quit'
        '"command":"kill -9'
    )

    # Scan recent log files (last 24 hours)
    for logfile in worker_logs/*.jsonl; do
        [[ ! -f "$logfile" ]] && continue
        # Skip old logs (check file modification time)
        if [[ $(uname) == "Darwin" ]]; then
            log_age=$(($(date +%s) - $(stat -f %m "$logfile" 2>/dev/null || echo 0)))
        else
            log_age=$(($(date +%s) - $(stat -c %Y "$logfile" 2>/dev/null || echo 0)))
        fi
        # Only check logs from last 24 hours
        [[ $log_age -gt 86400 ]] && continue

        for pattern in "${FORBIDDEN_CMD_PATTERNS[@]}"; do
            matches=$(grep -cE "$pattern" "$logfile" 2>/dev/null | tr -d '\n' || echo 0)
            matches=${matches:-0}
            if [[ "$matches" =~ ^[0-9]+$ ]] && [[ $matches -gt 0 ]]; then
                # Extract descriptive name from pattern - simplified extraction
                case "$pattern" in
                    *pkill*claude*) pattern_name="pkill claude" ;;
                    *pkill*iTerm*) pattern_name="pkill iTerm" ;;
                    *pkill*Terminal*) pattern_name="pkill Terminal" ;;
                    *osascript*quit*) pattern_name="osascript quit" ;;
                    *kill\ -9*) pattern_name="kill -9" ;;
                    *) pattern_name="$pattern" ;;
                esac
                echo -e "  ${RED}! $logfile: $matches command(s) with '$pattern_name'${NC}"
                ((forbidden_cmd_count+=matches)) || true
            fi
        done
    done

    if [[ $forbidden_cmd_count -eq 0 ]]; then
        echo "  ✓ No forbidden commands in recent logs"
    else
        echo "  Per ai_template.md: pkill, osascript quit, kill -9 are FORBIDDEN"
    fi
else
    echo "  (no worker_logs directory)"
fi
echo

# Check for oversized files (>5000 lines)
echo -e "${YELLOW}Oversized files (>5000 lines):${NC}"
oversized_count=0
if ! $IS_TEMPLATE; then
    # Find tracked source files over 5000 lines
    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        [[ ! -f "$file" ]] && continue
        # Skip binary files and known large generated files
        [[ "$file" =~ \.(lock|sum|svg|png|jpg|jpeg|gif|pdf|woff|woff2|ttf|eot)$ ]] && continue
        [[ "$file" =~ (package-lock\.json|yarn\.lock|Cargo\.lock)$ ]] && continue

        lines=$(wc -l < "$file" 2>/dev/null | tr -d ' ')
        if [[ $lines -gt 5000 ]]; then
            echo -e "  ${YELLOW}! $file: $lines lines${NC}"
            ((oversized_count++)) || true
        fi
    done < <(git ls-files 2>/dev/null | grep -E '\.(rs|py|js|ts|jsx|tsx|go|java|c|cpp|h|hpp|swift|kt)$')

    if [[ $oversized_count -eq 0 ]]; then
        echo "  ✓ No oversized source files"
    else
        echo "  Consider splitting files >5000 lines for maintainability"
    fi
else
    echo "  (skipped for ai_template itself)"
fi
echo

# Check for author headers in source files
echo -e "${YELLOW}Author headers:${NC}"
missing_author=0
if ! $IS_TEMPLATE; then
    # Sample up to 20 source files and check for author header
    # We check for "Andrew Yates" or "andrewdyates" in the first 10 lines
    sampled=0
    missing_list=()
    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        [[ ! -f "$file" ]] && continue
        [[ $sampled -ge 20 ]] && break
        ((sampled++)) || true

        # Check first 10 lines for author attribution (word boundaries to avoid jayates, etc.)
        if ! head -10 "$file" 2>/dev/null | grep -qiE '\bAndrew Yates\b|\bandrewdyates\b|\bayates\b'; then
            missing_list+=("$file")
            ((missing_author++)) || true
        fi
    done < <(git ls-files 2>/dev/null | grep -E '\.(rs|py)$' | head -30)

    if [[ $missing_author -gt 0 ]]; then
        echo -e "  ${YELLOW}! $missing_author of $sampled sampled files missing author header${NC}"
        for f in "${missing_list[@]:0:5}"; do
            echo "    - $f"
        done
        [[ ${#missing_list[@]} -gt 5 ]] && echo "    ... and $((${#missing_list[@]} - 5)) more"
    else
        echo "  ✓ Sampled files have author headers"
    fi
else
    echo "  (skipped for ai_template itself)"
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

# Check optional features (#1076)
echo -e "${YELLOW}Optional features:${NC}"
optional_issues=0
if [[ -f ".ai_template_features" ]]; then
    # Parse enabled features
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line%"${line##*[![:space:]]}"}"
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" ]] && continue

        # Validate feature name
        if [[ "$line" == */* || "$line" == *\\* || "$line" == *..* ]]; then
            echo -e "  ${RED}! Invalid feature name: $line${NC}"
            ((optional_issues++)) || true
            continue
        fi

        # Check if feature directory exists
        feature_dir="ai_template_scripts/optional/$line"
        if [[ -d "$feature_dir" ]]; then
            echo "  ✓ $line (enabled and present)"
        else
            echo -e "  ${YELLOW}! $line (enabled but missing - run sync_repo.sh)${NC}"
            ((optional_issues++)) || true
        fi
    done < ".ai_template_features"
else
    echo "  (no .ai_template_features - optional features not enabled)"
fi
echo

# List all rules files for AI awareness (#1910)
echo -e "${GREEN}Rules files (read these for repo-specific context):${NC}"
rules_files=()
if [[ -d ".claude/rules" ]]; then
    while IFS= read -r -d '' f; do
        rules_files+=("$f")
    done < <(find .claude/rules -name "*.md" -type f -print0 2>/dev/null | sort -z)
fi
if [[ ${#rules_files[@]} -gt 0 ]]; then
    for f in "${rules_files[@]}"; do
        echo "  • $f"
    done
else
    echo "  (no rules files found)"
fi
echo

# System health checks (#1910)
# These distinguish "template compliant" from "system healthy"
echo -e "${RED}System health (.flags/):${NC}"
health_failures=0
if [[ -d ".flags" ]]; then
    # Check for crash flag
    if [[ -f ".flags/crashes" ]]; then
        crash_count=$(cat ".flags/crashes" 2>/dev/null | grep -cE '^[0-9]+' || echo "0")
        if [[ "$crash_count" =~ ^[0-9]+$ ]] && [[ "$crash_count" -gt 0 ]]; then
            echo -e "  ${RED}! .flags/crashes: $crash_count crash(es) detected${NC}"
            ((health_failures++)) || true
        fi
    fi

    # Check for stuck process flag
    if [[ -f ".flags/stuck_process" ]]; then
        echo -e "  ${RED}! .flags/stuck_process exists - process may be stuck${NC}"
        ((health_failures++)) || true
    fi

    # Check for blocked issues flag
    if [[ -f ".flags/blocked_issues" ]]; then
        blocked_count=$(cat ".flags/blocked_issues" 2>/dev/null | head -1)
        if [[ "$blocked_count" =~ ^[0-9]+$ ]] && [[ "$blocked_count" -gt 0 ]]; then
            echo -e "  ${YELLOW}! .flags/blocked_issues: $blocked_count blocked issue(s)${NC}"
        fi
    fi

    if [[ $health_failures -eq 0 ]]; then
        echo "  ✓ No critical health flags"
    fi
else
    echo "  (no .flags directory - pulse.py may not have run)"
fi
echo

# Summary
echo "=== Summary ==="
exit_code=0
failures=0
warnings=0

# Critical failures (exit non-zero)
if [[ $missing -gt 0 ]]; then
    echo -e "${RED}[FAIL] Missing $missing required file(s) - run sync_repo.sh${NC}"
    ((failures+=missing)) || true
    exit_code=1
fi
if [[ $found_forbidden -gt 0 ]]; then
    if [[ "$CLEAN" == "true" ]]; then
        echo -e "${GREEN}[OK] Deleted $found_forbidden forbidden file(s)${NC}"
    else
        echo -e "${RED}[FAIL] Found $found_forbidden forbidden file(s) - DELETE IMMEDIATELY${NC}"
        ((failures+=found_forbidden)) || true
        exit_code=1
    fi
fi
if [[ $hooks_missing -gt 0 ]]; then
    echo -e "${RED}[FAIL] $hooks_missing git hook(s) not installed - run ./ai_template_scripts/install_hooks.sh${NC}"
    ((failures+=hooks_missing)) || true
    exit_code=1
fi
if [[ $ignore_count -gt 0 ]]; then
    echo -e "${RED}[FAIL] $ignore_count test ignore(s) found - FORBIDDEN per #341${NC}"
    ((failures+=ignore_count)) || true
    exit_code=1
fi
if [[ $forbidden_cmd_count -gt 0 ]]; then
    echo -e "${RED}[FAIL] $forbidden_cmd_count forbidden command(s) in logs - SAFETY VIOLATION${NC}"
    ((failures+=forbidden_cmd_count)) || true
    exit_code=1
fi
if [[ $health_failures -gt 0 ]]; then
    echo -e "${RED}[FAIL] $health_failures system health issue(s) - check .flags/ directory${NC}"
    ((failures+=health_failures)) || true
    exit_code=1
fi

# Warnings (don't affect exit code)
if [[ $found_obsolete -gt 0 ]]; then
    if [[ "$CLEAN" == "true" ]]; then
        echo -e "${GREEN}[OK] Deleted $found_obsolete obsolete file(s)${NC}"
    else
        echo -e "${YELLOW}[WARN] Found $found_obsolete obsolete file(s) - run with --clean to delete${NC}"
        ((warnings+=found_obsolete)) || true
    fi
fi
if [[ $claude_issues -gt 0 ]]; then
    echo -e "${YELLOW}[WARN] Found $claude_issues CLAUDE.md content issue(s) - clean up recommended${NC}"
    ((warnings+=claude_issues)) || true
fi
if [[ $vision_missing -eq 1 ]]; then
    echo -e "${YELLOW}[WARN] VISION.md missing - create to document strategic direction${NC}"
    ((warnings++)) || true
fi
if [[ $dirs_missing -gt 0 ]]; then
    echo -e "${YELLOW}[WARN] Missing $dirs_missing required directory(s) (ideas/)${NC}"
    ((warnings+=dirs_missing)) || true
fi
if [[ $looper_missing -gt 0 ]]; then
    echo -e "${YELLOW}[WARN] .looper_config.json missing - looper will use defaults${NC}"
    ((warnings++)) || true
fi
if [[ $oversized_count -gt 0 ]]; then
    echo -e "${YELLOW}[WARN] $oversized_count oversized file(s) >5000 lines${NC}"
    ((warnings+=oversized_count)) || true
fi
if [[ $missing_author -gt 0 ]]; then
    echo -e "${YELLOW}[WARN] $missing_author file(s) missing author header${NC}"
    ((warnings+=missing_author)) || true
fi
if [[ $optional_issues -gt 0 ]]; then
    echo -e "${YELLOW}[WARN] $optional_issues optional feature issue(s) - run sync_repo.sh${NC}"
    ((warnings+=optional_issues)) || true
fi

# Final summary
echo ""
if [[ $failures -eq 0 && $warnings -eq 0 && $health_failures -eq 0 ]]; then
    echo -e "${GREEN}✓ Template compliant AND system healthy${NC}"
elif [[ $failures -eq 0 && $health_failures -eq 0 ]]; then
    echo -e "${YELLOW}Template compliant (with $warnings warning(s)), system healthy${NC}"
elif [[ $health_failures -gt 0 ]]; then
    echo -e "${RED}Template compliance: $failures failure(s), $warnings warning(s)${NC}"
    echo -e "${RED}⚠️  SYSTEM UNHEALTHY: $health_failures health issue(s) - fix before starting work${NC}"
else
    echo -e "${RED}Template compliance: $failures failure(s), $warnings warning(s)${NC}"
fi

# Point to migration checklist if any issues found
if [[ $failures -gt 0 || $warnings -gt 0 ]]; then
    echo
    if [[ -f "docs/MIGRATION_CHECKLIST.md" ]]; then
        echo -e "${YELLOW}See docs/MIGRATION_CHECKLIST.md for step-by-step alignment guide${NC}"
    else
        echo -e "${YELLOW}See docs/MIGRATION_CHECKLIST.md in ai_template (https://github.com/ayates_dbx/ai_template/blob/main/docs/MIGRATION_CHECKLIST.md)${NC}"
        echo -e "${YELLOW}Or run ./ai_template_scripts/sync_repo.sh from the ai_template repo${NC}"
    fi
fi

exit $exit_code

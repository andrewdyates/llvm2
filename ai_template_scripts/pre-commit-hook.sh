#!/usr/bin/env bash
# pre-commit-hook.sh - Validates copyright headers and author metadata
#
# WARNING mode: Warns but allows commit to proceed.
# Run: git commit --no-verify to skip entirely.
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# Colors
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# === Critical: Check for staged new files + unstaged changes (#1437) ===
# pre-commit's stash mechanism can DELETE newly staged files when there are
# also unstaged changes present. This is a known pre-commit bug.
#
# Detect: staged new files (A/C status) AND any unstaged/untracked changes
check_new_file_stash_danger() {
    local status_line
    local staged_new=()
    local has_unstaged=0

    # Use porcelain output to detect staged adds and unstaged/untracked changes.
    while IFS= read -r status_line; do
        [[ -z "$status_line" ]] && continue
        local index_status="${status_line:0:1}"
        local worktree_status="${status_line:1:1}"
        local path="${status_line:3}"

        if [[ "$index_status" == "A" || "$index_status" == "C" ]]; then
            staged_new+=("$path")
        fi

        if [[ "$index_status" == "?" && "$worktree_status" == "?" ]]; then
            has_unstaged=1
        elif [[ "$worktree_status" != " " ]]; then
            has_unstaged=1
        fi
    done <<< "$(git status --porcelain 2>/dev/null || true)"

    # pre-commit stashes unstaged changes before running hooks. Detect that stash
    # via its patch file to avoid false negatives.
    if [[ $has_unstaged -eq 0 && -n "${PRE_COMMIT:-}" ]]; then
        local cache_root store_dir now patch mtime
        cache_root="${PRE_COMMIT_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}}"
        store_dir="${cache_root}/pre-commit"
        now=$(date +%s)
        for patch in "$store_dir"/patch*-"$PPID"; do
            [[ -e "$patch" ]] || continue
            mtime=$(stat -f %m "$patch" 2>/dev/null || stat -c %Y "$patch" 2>/dev/null || echo 0)
            if [[ "$mtime" -gt 0 && $((now - mtime)) -le 300 ]]; then
                has_unstaged=1
                break
            fi
        done
    fi

    [[ ${#staged_new[@]} -eq 0 ]] && return 0
    [[ $has_unstaged -eq 0 ]] && return 0

    # DANGER: Both conditions met
    warn "DANGER: Staged new files detected alongside unstaged changes!"
    warn ""
    warn "  pre-commit's stash/restore may DELETE your new files."
    warn "  See: https://github.com/pre-commit/pre-commit/issues/1498"
    warn ""
    warn "  Staged new files:"
    for f in "${staged_new[@]}"; do
        [[ -n "$f" ]] && warn "    + $f"
    done
    warn ""
    warn "  Options to proceed safely:"
    warn "    1. Stage ALL files: git add -A && git commit"
    warn "    2. Stash unstaged first: git stash -k -u && git commit && git stash pop"
    warn "    3. Commit with --no-verify (skip pre-commit)"
    warn ""
    WARNINGS=$((WARNINGS + 1))

    # Block if explicitly requested OR in AI mode (AIs don't notice file loss)
    # To override in AI mode: BLOCK_NEW_FILE_STASH_DANGER=0
    if [[ "${BLOCK_NEW_FILE_STASH_DANGER:-}" == "1" ]] || \
       { [[ -n "${AI_ROLE:-}" ]] && [[ "${BLOCK_NEW_FILE_STASH_DANGER:-}" != "0" ]]; }; then
        error "Blocking commit to prevent file deletion by pre-commit stash"
        echo "  pre-commit's stash/restore would delete your newly staged files."
        echo ""
        echo "  To proceed safely, choose one:"
        echo "    1. Stage all changes:    git add -A && git commit ..."
        echo "    2. Stash unstaged first: git stash -k -u && git commit ... && git stash pop"
        echo "    3. Skip pre-commit:      git commit --no-verify ..."
        echo ""
        echo "  To disable this check: BLOCK_NEW_FILE_STASH_DANGER=0 git commit ..."
        exit 1
    fi
}

WARNINGS=0
ERRORS=0  # Blocking errors (ignores without issue refs)
check_new_file_stash_danger

# Get staged files (new, modified, or renamed) - use newline as delimiter to handle spaces in filenames
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)
INITIAL_STAGED_FILES="$STAGED_FILES"
OLD_IFS="$IFS"
IFS=$'\n'

# Directory containing header templates (relative to repo root)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
HEADERS_DIR="$REPO_ROOT/ai_template_scripts/headers"

# Map file extension to header template
get_header_template() {
    local ext="$1"
    case "$ext" in
        py)         echo "$HEADERS_DIR/python.txt" ;;
        sh|bash)    echo "$HEADERS_DIR/bash.txt" ;;
        rs)         echo "$HEADERS_DIR/rust.txt" ;;
        js|ts)      echo "$HEADERS_DIR/javascript.txt" ;;
        swift)      echo "$HEADERS_DIR/swift.txt" ;;
        c|cpp|h)    echo "$HEADERS_DIR/rust.txt" ;;  # Same // style as rust
        go)         echo "$HEADERS_DIR/rust.txt" ;;  # Same // style as rust
        java)       echo "$HEADERS_DIR/rust.txt" ;;  # Same // style as rust
        *)          echo "" ;;
    esac
}

get_staged_files() {
    git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true
}

normalize_file_list() {
    printf '%s\n' "$1" | sed '/^$/d' | sort
}

NEW_STAGED_FILES=()
while IFS=$'\t' read -r status path; do
    [[ -z "$status" || -z "$path" ]] && continue
    NEW_STAGED_FILES+=("$path")
done <<< "$(git diff --cached --name-status --diff-filter=A 2>/dev/null || true)"

is_new_staged_file() {
    local target="$1"
    local new_file
    # Guard against empty array with set -u
    [[ ${#NEW_STAGED_FILES[@]} -eq 0 ]] && return 1
    for new_file in "${NEW_STAGED_FILES[@]}"; do
        [[ "$new_file" == "$target" ]] && return 0
    done
    return 1
}

# Check and auto-fix copyright headers in source files
check_header() {
    local file="$1"
    local ext="${file##*.}"

    # Only check known source file types
    case "$ext" in
        py|sh|bash|rs|js|ts|swift|c|cpp|h|go|java) ;;
        *) return 0 ;;  # Skip unknown extensions
    esac

    # Check if file contains copyright notice
    if ! head -10 "$file" 2>/dev/null | grep -qi "copyright.*dropbox\|copyright.*andrew.*yates"; then
        local template
        template=$(get_header_template "$ext")

        if [[ -n "$template" && -f "$template" ]]; then
            # Auto-add header, preserving shebang and permissions
            local tmpfile
            tmpfile=$(mktemp)
            local first_line
            first_line=$(head -1 "$file")

            if [[ "$first_line" == "#!"* ]]; then
                # File has shebang - put it first, then header
                {
                    echo "$first_line"
                    cat "$template"
                    echo ""
                    tail -n +2 "$file"
                } > "$tmpfile"
            else
                # No shebang - header goes first
                {
                    cat "$template"
                    echo ""
                    cat "$file"
                } > "$tmpfile"
            fi

            # Preserve original file permissions
            chmod --reference="$file" "$tmpfile" 2>/dev/null || chmod "$(stat -f %Lp "$file")" "$tmpfile" 2>/dev/null || true
            mv "$tmpfile" "$file"

            # Re-stage the file with the header
            git add "$file"

            echo -e "${YELLOW}[AUTO-FIX]${NC} Added copyright header to: $file"
        else
            warn "Missing copyright header (no template for .$ext): $file"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
}

# Check author in package manifests
check_manifest_author() {
    local file="$1"

    if [[ ! -f "$file" ]]; then
        return 0
    fi

    # Valid author patterns: Andrew Yates, andrewdyates
    if ! grep -qiE 'andrew.*yates|andrewdyates' "$file" 2>/dev/null; then
        warn "Missing or incorrect author in $file (should be Andrew Yates or andrewdyates)"
        WARNINGS=$((WARNINGS + 1))
    fi
}

check_new_staged_files() {
    local current_staged_files new_files
    current_staged_files=$(get_staged_files)

    new_files=$(comm -13 \
        <(normalize_file_list "$INITIAL_STAGED_FILES") \
        <(normalize_file_list "$current_staged_files") \
        || true)

    if [[ -n "$new_files" ]]; then
        warn "New files staged during pre-commit checks. Review before commit:"
        while IFS= read -r file; do
            [[ -z "$file" ]] && continue
            echo "  $file"
        done <<< "$new_files"
        WARNINGS=$((WARNINGS + 1))
    fi
}

# Check staged source files for headers
for file in $STAGED_FILES; do
    if [[ -f "$file" ]]; then
        check_header "$file"
    fi
done

# Check manifests if staged
if echo "$STAGED_FILES" | grep -q "Cargo.toml"; then
    check_manifest_author "Cargo.toml"
fi

if echo "$STAGED_FILES" | grep -q "pyproject.toml"; then
    check_manifest_author "pyproject.toml"
fi

# Check AppleScript syntax in staged shell files (macOS only)
if [[ "$(uname)" == "Darwin" ]]; then
    LINTER="$REPO_ROOT/ai_template_scripts/lint_applescript.sh"
    if [[ -x "$LINTER" ]]; then
        # Check only staged shell files that contain osascript
        APPLESCRIPT_FILES=()
        for file in $STAGED_FILES; do
            if [[ "$file" == *.sh && -f "$file" ]] && grep -q 'osascript' "$file" 2>/dev/null; then
                APPLESCRIPT_FILES+=("$file")
            fi
        done
        if [[ ${#APPLESCRIPT_FILES[@]} -gt 0 ]]; then
            if ! "$LINTER" "${APPLESCRIPT_FILES[@]}" 2>/dev/null; then
                warn "AppleScript syntax errors found in staged files"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    fi
fi

# NOTE: ruff and shellcheck are handled by pre-commit framework
# See .pre-commit-config.yaml for configuration

# FORBID test ignores entirely (#341)
# Tests must PASS, FAIL, or be DELETED. No ignore. No hiding.
#
# Exclusions can be configured via:
#   1. .ignore-check-exclude file (one path prefix per line)
#   2. IGNORE_CHECK_EXCLUDE env var (colon-separated paths)
#
# Example .ignore-check-exclude:
#   vendor/
#   third_party/
#   generated/

# Load exclusion paths
# Default exclusions for common third-party/generated directories
IGNORE_EXCLUSIONS=(
    "node_modules/"
    "vendor/"
    ".venv/"
    "venv/"
    "__pycache__/"
    "target/"
    "dist/"
    "build/"
    ".git/"
)

# Add project-specific exclusions from .ignore-check-exclude
if [[ -f ".ignore-check-exclude" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        IGNORE_EXCLUSIONS+=("$line")
    done < ".ignore-check-exclude"
fi
if [[ -n "${IGNORE_CHECK_EXCLUDE:-}" ]]; then
    IFS=':' read -ra env_excludes <<< "$IGNORE_CHECK_EXCLUDE"
    IGNORE_EXCLUSIONS+=("${env_excludes[@]}")
fi

is_excluded() {
    local file="$1"
    # Handle empty array safely with set -u
    if [[ ${#IGNORE_EXCLUSIONS[@]} -gt 0 ]]; then
        for excl in "${IGNORE_EXCLUSIONS[@]}"; do
            if [[ "$file" == "$excl"* ]]; then
                return 0
            fi
        done
    fi
    return 1
}

check_new_ignores() {
    local file="$1"
    local ext="${file##*.}"

    # Check exclusions
    if is_excluded "$file"; then
        return 0
    fi

    # Language-specific patterns (precise to avoid false positives)
    # Note: Use [[:space:]] not \s for POSIX ERE compatibility
    local pattern=""
    case "$ext" in
        rs)
            # Rust: #[ignore] or #[ignore = "..."]
            pattern='#\[ignore[[:space:]]*(\]|=)'
            ;;
        py)
            # Python: @skip(), @pytest.mark.skip, @pytest.mark.xfail, @unittest.skip*
            pattern='@skip[[:space:]]*\(|@pytest\.mark\.skip|@pytest\.mark\.xfail|@unittest\.skip'
            ;;
        js|ts|tsx)
            # JS/TS: .skip(, it.skip(, test.skip(, describe.skip(, xit(, xdescribe(, xtest(
            pattern='\.skip\(|^[[:space:]]*(xit|xdescribe|xtest)[[:space:]]*\('
            ;;
        *)
            return 0
            ;;
    esac

    # Get ignore patterns added in this commit (new lines only, indicated by +)
    local new_ignores
    new_ignores=$(git diff --cached -U0 "$file" 2>/dev/null | grep -E "^\+.*($pattern)" | grep -v '^+++' || true)

    if [[ -n "$new_ignores" ]]; then
        while IFS= read -r line; do
            error "FORBIDDEN: Test ignore in $file"
            echo "  $line"
            echo ""
            echo "  Tests must PASS, FAIL, or be DELETED. No ignore."
            echo "  - Slow? Add timeout. If timeout exceeded, it FAILS."
            echo "  - Blocked? Let it FAIL. Failure is visibility."
            echo "  - Flaky? Fix flakiness or DELETE."
            echo "  - Obsolete? DELETE."
            echo ""
            ERRORS=$((ERRORS + 1))
        done <<< "$new_ignores"
    fi
}

# Check staged files for new ignores (ALL roles, BLOCKING)
for file in $STAGED_FILES; do
    if [[ -f "$file" ]]; then
        check_new_ignores "$file"
    fi
done

# Regression lookback window (seconds)
REGRESSION_MIN_LOOKBACK_SEC=3600   # 1 hour
REGRESSION_MAX_LOOKBACK_SEC=86400  # 24 hours
FETCH_AGE_SEC=""

get_fetch_age_sec() {
    if [[ -f .git/FETCH_HEAD ]]; then
        local last_fetch now
        # macOS stat vs Linux stat
        last_fetch=$(stat -f %m .git/FETCH_HEAD 2>/dev/null || stat -c %Y .git/FETCH_HEAD 2>/dev/null || echo 0)
        if [[ "$last_fetch" -gt 0 ]]; then
            now=$(date +%s)
            echo $((now - last_fetch))
            return 0
        fi
    fi
    return 1
}

compute_regression_lookback() {
    local lookback=$REGRESSION_MIN_LOOKBACK_SEC
    if [[ -n "${FETCH_AGE_SEC:-}" && "$FETCH_AGE_SEC" -gt 0 ]]; then
        lookback=$((FETCH_AGE_SEC + 60))
        if [[ "$lookback" -lt "$REGRESSION_MIN_LOOKBACK_SEC" ]]; then
            lookback=$REGRESSION_MIN_LOOKBACK_SEC
        elif [[ "$lookback" -gt "$REGRESSION_MAX_LOOKBACK_SEC" ]]; then
            lookback=$REGRESSION_MAX_LOOKBACK_SEC
        fi
    fi
    echo "$lookback"
}

resolve_origin_ref() {
    local origin_head ref head_branch
    origin_head=$(git symbolic-ref -q refs/remotes/origin/HEAD 2>/dev/null || true)
    if [[ -n "$origin_head" ]]; then
        ref="${origin_head#refs/remotes/}"
        if git show-ref --verify --quiet "refs/remotes/${ref}" 2>/dev/null; then
            echo "$ref"
            return 0
        fi
    fi

    head_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
    if [[ -n "$head_branch" && "$head_branch" != "HEAD" ]]; then
        if git show-ref --verify --quiet "refs/remotes/origin/${head_branch}" 2>/dev/null; then
            echo "origin/${head_branch}"
            return 0
        fi
    fi

    if git show-ref --verify --quiet refs/remotes/origin/main 2>/dev/null; then
        echo "origin/main"
        return 0
    fi

    if git show-ref --verify --quiet refs/remotes/origin/master 2>/dev/null; then
        echo "origin/master"
        return 0
    fi

    return 1
}

# === Stale Tree Detection (Part of #1263) ===
# Check if working tree may be stale relative to origin
# This helps prevent silent reverts in multi-worker scenarios

check_stale_tree() {
    # Note: This function is only called when AI_WORKER_ID is set (multi-worker mode)
    # Check if FETCH_HEAD exists (won't exist on fresh clones with no fetch)
    local age=""
    if age=$(get_fetch_age_sec); then
        FETCH_AGE_SEC="$age"
        if [[ $age -gt 60 ]]; then
            warn "Working tree may be stale (last fetch: ${age}s ago)"
            warn "Consider: git fetch && git pull --rebase"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        FETCH_AGE_SEC=""
        # No FETCH_HEAD means never fetched - always warn in multi-worker mode
        warn "No recent fetch detected. Run: git fetch && git pull --rebase"
        WARNINGS=$((WARNINGS + 1))
    fi
}

# === Regression Detection (Part of #1263) ===
# Check if staged files would revert changes from recent commits by other workers
# Uses index-based hashing per design doc guidance

check_regression() {
    # Note: This function is only called when AI_WORKER_ID is set (multi-worker mode)
    # Skip if no staged files or not in git repo
    [[ -z "$STAGED_FILES" ]] && return 0
    [[ ! -d .git ]] && return 0

    local base_ref lookback fetch_age
    base_ref=$(resolve_origin_ref) || return 0
    if [[ -z "$base_ref" ]]; then
        return 0
    fi

    if [[ -z "${FETCH_AGE_SEC:-}" ]]; then
        if fetch_age=$(get_fetch_age_sec); then
            FETCH_AGE_SEC="$fetch_age"
        fi
    fi
    lookback=$(compute_regression_lookback)

    local file staged_hash recent_commits commit commit_msg parent_hash commit_hash
    local my_worker_pattern regression_found=0

    # Build pattern for our own commits (to skip them)
    # Note: AI_WORKER_ID is guaranteed set since this function is only called in multi-worker mode
    my_worker_pattern="\\[W${AI_WORKER_ID}\\]"

    # Note: We iterate on word-split STAGED_FILES. This matches git diff output format
    # where filenames with spaces are quoted. For our use case (AI repos), filenames
    # with spaces are not expected.
    for file in $STAGED_FILES; do
        [[ -f "$file" ]] || continue

        # Get hash of staged (index) version using git hash-object
        # Per design doc: hash the index version, not working tree
        staged_hash=$(git cat-file -p ":$file" 2>/dev/null | git hash-object --stdin 2>/dev/null) || continue

        # Find recent commits on the default origin branch that touched this file
        # Note: This uses the LOCAL origin ref, which may be stale if fetch hasn't run.
        # The stale_tree check warns about this separately. We don't fetch here to avoid
        # blocking the commit hook with network I/O.
        recent_commits=$(git log --since="${lookback} seconds ago" "$base_ref" --format=%H -- "$file" 2>/dev/null) || continue

        for commit in $recent_commits; do
            # Get the commit message to check if it's from our worker
            commit_msg=$(git log -1 --format=%s "$commit" 2>/dev/null) || continue

            # Skip our own commits
            if [[ "$commit_msg" =~ $my_worker_pattern ]]; then
                continue
            fi

            # Get hashes for efficient comparison (don't compare full file contents)
            parent_hash=$(git rev-parse "${commit}^:$file" 2>/dev/null) || continue
            commit_hash=$(git rev-parse "${commit}:$file" 2>/dev/null) || continue

            # REGRESSION: Our staged version matches the BEFORE state (parent),
            # meaning we're reverting someone else's changes
            if [[ "$staged_hash" == "$parent_hash" && "$parent_hash" != "$commit_hash" ]]; then
                warn "File $file may revert changes from commit ${commit:0:7}"
                warn "  That commit changed the file, but your version matches the pre-change state"
                warn "  Consider: git fetch && git checkout $base_ref -- $file"
                regression_found=1
                WARNINGS=$((WARNINGS + 1))
            fi
        done
    done

    if [[ $regression_found -eq 1 ]]; then
        warn "Potential regression detected. Review staged files before committing."
    fi
}

# Run stale tree and regression checks for multi-worker mode
if [[ -n "${AI_WORKER_ID:-}" ]]; then
    check_stale_tree
    check_regression
fi

# === Worker File Tracking Check ===
# BLOCK commits that include files not tracked by this worker.
# Prevents workers from stepping on each other's files in shared checkout.
# Override: AIT_ALLOW_UNTRACKED=1 (for emergencies only)
check_untracked_staged_files() {
    local tracker_file="$REPO_ROOT/.worker_${AI_WORKER_ID}_files.json"

    # Skip if no tracker file (first iteration or new worker)
    [[ -f "$tracker_file" ]] || return 0

    # Get tracked files from JSON
    local tracked_files
    tracked_files=$(python3 -c "
import json
import sys
try:
    with open('$tracker_file') as f:
        data = json.load(f)
    for f in data.get('files', []):
        print(f)
except Exception:
    sys.exit(0)
" 2>/dev/null || true)

    # Skip if no tracked files (first iteration)
    [[ -z "$tracked_files" ]] && return 0

    # Check each staged file
    local untracked_staged=()
    for file in $STAGED_FILES; do
        if ! echo "$tracked_files" | grep -qxF "$file"; then
            untracked_staged+=("$file")
        fi
    done

    if [[ ${#untracked_staged[@]} -gt 0 ]]; then
        echo "ERROR: Worker $AI_WORKER_ID staging files not in tracker:"
        for file in "${untracked_staged[@]}"; do
            echo "  $file"
        done
        echo ""
        echo "These files may belong to another worker or were auto-generated."
        echo "Workers must only commit files they touched to prevent interference."
        echo ""
        if [[ "${AIT_ALLOW_UNTRACKED:-}" == "1" ]]; then
            warn "Proceeding anyway (AIT_ALLOW_UNTRACKED=1)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo "To override (emergency only): AIT_ALLOW_UNTRACKED=1 git commit ..."
            exit 1
        fi
    fi
}

# Run worker file tracking check for multi-worker mode
if [[ -n "${AI_WORKER_ID:-}" ]]; then
    check_untracked_staged_files
fi

# === Non-Worker Staged Worker Files Check (#1922) ===
# BLOCK non-workers (Manager, Prover, Researcher) from committing files
# that are tracked by active workers. This prevents race conditions where
# a worker's commit is blocked and a non-worker commits those files.
# Override: AIT_ALLOW_WORKER_FILES=1 (for emergencies only)
check_non_worker_staging_worker_files() {
    # Only check for non-worker AI roles
    local role="${AI_ROLE:-}"
    case "$role" in
        MANAGER|PROVER|RESEARCHER) ;;
        *) return 0 ;;  # USER or unknown - skip check
    esac

    # Skip if no staged files
    [[ -z "$STAGED_FILES" ]] && return 0

    # Find all worker tracker files
    local worker_files=()
    local pid_alive worker_id tracker_file files_json

    for tracker_file in "$REPO_ROOT"/.worker_*_files.json; do
        [[ -f "$tracker_file" ]] || continue

        # Check if the worker process is alive
        worker_id=$(basename "$tracker_file" | sed -E 's/\.worker_([0-9]+)_files\.json/\1/')
        pid_alive=$(python3 -c "
import json
import os
try:
    with open('$tracker_file') as f:
        data = json.load(f)
    pid = data.get('pid', 0)
    if pid > 0:
        os.kill(pid, 0)
        print('alive')
except (ProcessLookupError, FileNotFoundError, json.JSONDecodeError, KeyError):
    pass
except PermissionError:
    print('alive')  # Process exists but we can't signal it
" 2>/dev/null || true)

        # Skip dead worker trackers
        [[ "$pid_alive" != "alive" ]] && continue

        # Get files tracked by this worker
        files_json=$(python3 -c "
import json
try:
    with open('$tracker_file') as f:
        data = json.load(f)
    for f in data.get('files', []):
        print(f)
except Exception:
    pass
" 2>/dev/null || true)

        # Check if any staged files are in this worker's tracker
        if [[ -n "$files_json" ]]; then
            for staged_file in $STAGED_FILES; do
                if echo "$files_json" | grep -qxF "$staged_file"; then
                    worker_files+=("W$worker_id:$staged_file")
                fi
            done
        fi
    done

    if [[ ${#worker_files[@]} -gt 0 ]]; then
        echo "ERROR: $role role staging files tracked by active workers:"
        for entry in "${worker_files[@]}"; do
            local wid="${entry%%:*}"
            local fname="${entry#*:}"
            echo "  $fname (owned by $wid)"
        done
        echo ""
        echo "This prevents race conditions where workers' uncommitted changes"
        echo "get committed by non-workers with incorrect attribution."
        echo ""
        echo "The worker's commit was likely blocked. Options:"
        echo "  1. Let the worker retry their commit"
        echo "  2. Unstage these files: git restore --staged <file>"
        echo "  3. Emergency override: AIT_ALLOW_WORKER_FILES=1 git commit ..."
        echo ""
        if [[ "${AIT_ALLOW_WORKER_FILES:-}" == "1" ]]; then
            warn "Proceeding anyway (AIT_ALLOW_WORKER_FILES=1)"
            WARNINGS=$((WARNINGS + 1))
        else
            exit 1
        fi
    fi
}

# Run non-worker check for AI roles (Manager, Prover, Researcher)
if [[ -n "${AI_ROLE:-}" ]] && [[ -z "${AI_WORKER_ID:-}" ]]; then
    check_non_worker_staging_worker_files
fi

check_new_staged_files

run_tla_property_tests() {
    local script="$REPO_ROOT/scripts/run_tla_property_tests.py"
    [[ -f "$script" ]] || return 0

    if ! python3 "$script" --staged; then
        error "TLA+ property tests failed"
        ERRORS=$((ERRORS + 1))
    fi
}

if [[ -n "$STAGED_FILES" ]]; then
    run_tla_property_tests
fi

# === Build Gate Verification (#1485) ===
# Run build check when AI commits code changes to catch compile errors early.
# Only runs for AI sessions to avoid slowing human commits.
# Per ai_template.md: "Worker MUST verify build passes at iteration start"
#
# Mode: BUILD_GATE=skip to skip, BUILD_GATE=warn for advisory (default: block)

run_build_gate() {
    local has_rust=0 has_js=0 has_python=0

    # Detect project type from staged files
    for file in $STAGED_FILES; do
        case "${file##*.}" in
            rs) has_rust=1 ;;
            js|ts|tsx|jsx) has_js=1 ;;
            py) has_python=1 ;;
        esac
    done

    # Rust: cargo check
    if [[ $has_rust -eq 1 && -f "Cargo.toml" ]]; then
        echo "Build gate: Running cargo check..."
        if ! cargo check --quiet 2>&1; then
            if [[ "${BUILD_GATE:-block}" == "warn" ]]; then
                warn "cargo check failed - build is broken"
                WARNINGS=$((WARNINGS + 1))
            else
                error "BUILD GATE FAILED: cargo check failed"
                echo "  Fix build errors before committing."
                echo "  To skip: BUILD_GATE=skip git commit ..."
                ERRORS=$((ERRORS + 1))
            fi
        fi
    fi

    # JavaScript/TypeScript: npm run build (if script exists)
    if [[ $has_js -eq 1 && -f "package.json" ]]; then
        # Check if build script exists
        if grep -q '"build"' package.json 2>/dev/null; then
            echo "Build gate: Running npm run build..."
            if ! npm run build --silent 2>&1; then
                if [[ "${BUILD_GATE:-block}" == "warn" ]]; then
                    warn "npm run build failed - build is broken"
                    WARNINGS=$((WARNINGS + 1))
                else
                    error "BUILD GATE FAILED: npm run build failed"
                    echo "  Fix build errors before committing."
                    echo "  To skip: BUILD_GATE=skip git commit ..."
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        fi
    fi

    # Python: python -m py_compile on staged .py files
    if [[ $has_python -eq 1 ]]; then
        local py_errors=0
        for file in $STAGED_FILES; do
            if [[ "${file##*.}" == "py" && -f "$file" ]]; then
                if ! python3 -m py_compile "$file" 2>/dev/null; then
                    py_errors=1
                    break
                fi
            fi
        done
        if [[ $py_errors -eq 1 ]]; then
            if [[ "${BUILD_GATE:-block}" == "warn" ]]; then
                warn "Python syntax errors in staged files"
                WARNINGS=$((WARNINGS + 1))
            else
                error "BUILD GATE FAILED: Python syntax errors in staged files"
                echo "  Fix syntax errors before committing."
                echo "  To skip: BUILD_GATE=skip git commit ..."
                ERRORS=$((ERRORS + 1))
            fi
        fi
    fi
}

# Run build gate for AI sessions (AI_ROLE is set by looper)
if [[ -n "${AI_ROLE:-}" && -n "$STAGED_FILES" && "${BUILD_GATE:-block}" != "skip" ]]; then
    run_build_gate
fi

# Report warnings but don't block
if [[ $WARNINGS -gt 0 ]]; then
    echo ""
    warn "$WARNINGS warning(s) found (missing headers, authors, etc.)"
    echo "  Add headers from: ai_template_scripts/headers/"
    echo "  To skip warnings: git commit --no-verify"
    echo ""
fi

# Report blocking errors
if [[ $ERRORS -gt 0 ]]; then
    echo ""
    error "$ERRORS BLOCKING error(s) - commit rejected"
    echo "  Test ignores are FORBIDDEN. Tests must PASS, FAIL, or be DELETED."
    echo "  See ai_template.md 'Test ignores FORBIDDEN' rule."
    echo ""
fi

# Zone enforcement for multi-worker mode
# Only checked when AI_WORKER_ID is set (multi-worker sessions)
# Mode: ZONE_ENFORCEMENT=strict to block, otherwise advisory (warnings only)
if [[ -n "${AI_WORKER_ID:-}" && -n "$STAGED_FILES" ]]; then
    # Build file list from staged files
    zone_files=()
    for file in $STAGED_FILES; do
        zone_files+=("$file")
    done

    if [[ ${#zone_files[@]} -gt 0 ]]; then
        # Check files using zones module
        if ! python3 -m looper.zones check "${zone_files[@]}" 2>/dev/null; then
            if [[ "${ZONE_ENFORCEMENT:-advisory}" == "strict" ]]; then
                error "Zone violation: files outside worker's zone (strict mode)"
                ERRORS=$((ERRORS + 1))
            else
                warn "Zone advisory: some files are outside worker's zone"
                warn "  Set ZONE_ENFORCEMENT=strict to block commits outside zone"
            fi
        fi
    fi
fi

# Self-audit reminder for AI sessions
# Only shows for AI commits (looper sets AI_ROLE)
if [[ -n "${AI_ROLE:-}" ]]; then
    CYAN='\033[1;36m'
    BOLD='\033[1m'
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} ${BOLD}SELF-AUDIT CHECKLIST${NC}                                          ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Before committing, did you:                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Look for MORE bugs beyond the obvious ones?               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Check edge cases and error handling?                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Verify ALL tests pass?                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   □ Consider what you might have missed?                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC} The follow-up audit often catches 2-4 more issues!            ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# Restore IFS
IFS="$OLD_IFS"

# === Project-specific extensions ===
# Hooks in .pre-commit-local.d/ are NOT synced from ai_template.
# Projects can add custom pre-commit checks there.
# See ai_template_scripts/README.md for documentation.
PROJECT_HOOKS_DIR="$REPO_ROOT/.pre-commit-local.d"
if [[ -d "$PROJECT_HOOKS_DIR" ]]; then
    for hook in "$PROJECT_HOOKS_DIR"/*.sh; do
        [[ -f "$hook" && -x "$hook" ]] || continue
        echo "Running project hook: $(basename "$hook")"
        if ! "$hook"; then
            error "Project hook failed: $(basename "$hook")"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

# Exit non-zero if blocking errors found
if [[ $ERRORS -gt 0 ]]; then
    exit 1
fi
exit 0

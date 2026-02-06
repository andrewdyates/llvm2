#!/usr/bin/env bash
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# pre-commit-hook.sh - Validates copyright headers and author metadata
#
# WARNING mode: Warns but allows commit to proceed.
# Run: git commit --no-verify to skip entirely.
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# Re-exec latest source hook when installed copy is stale (#2826).
# .git/hooks/pre-commit can lag behind ai_template_scripts during long sessions.
# When content differs, prefer the repo source hook.
if [[ -z "${AI_TEMPLATE_HOOK_REEXEC:-}" ]]; then
    REPO_ROOT_SYNC="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    SOURCE_HOOK_SYNC="$REPO_ROOT_SYNC/ai_template_scripts/pre-commit-hook.sh"
    CURRENT_HOOK_PATH="${BASH_SOURCE[0]}"
    if [[ -n "$REPO_ROOT_SYNC" && -f "$SOURCE_HOOK_SYNC" && "$CURRENT_HOOK_PATH" != "$SOURCE_HOOK_SYNC" ]]; then
        if grep -q "pre-commit-hook.sh - Validates" "$SOURCE_HOOK_SYNC" 2>/dev/null &&
            ! cmp -s "$CURRENT_HOOK_PATH" "$SOURCE_HOOK_SYNC" 2>/dev/null; then
            AI_TEMPLATE_HOOK_REEXEC=1 exec bash "$SOURCE_HOOK_SYNC" "$@"
        fi
    fi
fi

# Script directory (may be .git/hooks when installed, or ai_template_scripts when run directly)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find tracker_utils.py relative to REPO_ROOT (not SCRIPT_DIR)
# This handles both direct execution and when copied to .git/hooks/
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
TRACKER_UTILS="$REPO_ROOT/ai_template_scripts/tracker_utils.py"

# Load identity configuration from ait_identity.toml
# Note: `source` of non-existent files is fatal under set -e even with `|| true`,
# so we must guard with file-existence checks (#3025).
# shellcheck source=identity.sh
if [[ -f "$REPO_ROOT/ai_template_scripts/identity.sh" ]]; then
    source "$REPO_ROOT/ai_template_scripts/identity.sh"
elif [[ -f "$SCRIPT_DIR/identity.sh" ]]; then
    source "$SCRIPT_DIR/identity.sh"
fi
if [[ ! -f "$TRACKER_UTILS" ]]; then
    TRACKER_UTILS="$SCRIPT_DIR/tracker_utils.py"
fi

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
    done <<<"$(git status --porcelain 2>/dev/null || true)"

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

    # The stash risk only applies when running under the pre-commit framework,
    # which stashes unstaged changes before hooks and restores after.
    # When running as a direct git hook (no pre-commit framework), there is
    # no stash/restore mechanism, so no risk of file deletion (#2882).
    if [[ -z "${PRE_COMMIT:-}" ]]; then
        # Not running under pre-commit framework — no stash risk.
        # Still warn for visibility, but do NOT block.
        warn "Note: staged new files with unstaged changes (safe — not running under pre-commit framework)"
        WARNINGS=$((WARNINGS + 1))
        return 0
    fi

    # DANGER: Both conditions met (running under pre-commit framework)
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
    warn "    1. Stage specific files: git add <file1> <file2> && git commit"
    warn "    2. Commit with --no-verify (skip pre-commit, USER only)"
    warn ""
    WARNINGS=$((WARNINGS + 1))

    # Block if explicitly requested OR in AI mode (AIs don't notice file loss)
    # To override in AI mode: BLOCK_NEW_FILE_STASH_DANGER=0
    if [[ "${BLOCK_NEW_FILE_STASH_DANGER:-}" == "1" ]] ||
        { [[ -n "${AI_ROLE:-}" ]] && [[ "${BLOCK_NEW_FILE_STASH_DANGER:-}" != "0" ]]; }; then
        error "Blocking commit to prevent file deletion by pre-commit stash"
        echo "  pre-commit's stash/restore would delete your newly staged files."
        echo ""
        echo "  To proceed safely:"
        echo "    Stage specific files: git add <file1> <file2> && git commit"
        echo ""
        echo "  To disable this check: BLOCK_NEW_FILE_STASH_DANGER=0 git commit ..."
        exit 1
    fi
}

WARNINGS=0
ERRORS=0 # Blocking errors (ignores without issue refs)
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
    py) echo "$HEADERS_DIR/python.txt" ;;
    sh | bash) echo "$HEADERS_DIR/bash.txt" ;;
    rs) echo "$HEADERS_DIR/rust.txt" ;;
    js | ts) echo "$HEADERS_DIR/javascript.txt" ;;
    swift) echo "$HEADERS_DIR/swift.txt" ;;
    c | cpp | h) echo "$HEADERS_DIR/rust.txt" ;; # Same // style as rust
    go) echo "$HEADERS_DIR/rust.txt" ;;          # Same // style as rust
    java) echo "$HEADERS_DIR/rust.txt" ;;        # Same // style as rust
    *) echo "" ;;
    esac
}

# Generate a copyright header from identity config for the given file extension.
# Returns the header text on stdout. Empty if extension is unknown.
generate_header() {
    local ext="$1"
    local prefix
    case "$ext" in
    py | sh | bash) prefix="#" ;;
    rs | js | ts | swift | c | cpp | h | go | java) prefix="//" ;;
    *) return 0 ;;
    esac
    local holder="${AIT_COPYRIGHT_HOLDER:-$AIT_OWNER_NAME}"
    echo "$prefix Copyright $AIT_COPYRIGHT_YEAR ${holder}"
    echo "$prefix Author: $AIT_OWNER_NAME"
    echo "$prefix Licensed under the Apache License, Version 2.0"
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
done <<<"$(git diff --cached --name-status --diff-filter=A 2>/dev/null || true)"

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
    py | sh | bash | rs | js | ts | swift | c | cpp | h | go | java) ;;
    *) return 0 ;; # Skip unknown extensions
    esac

    # Guard: empty grep pattern matches everything, silently disabling the check (#3025)
    if [[ -z "${AIT_COPYRIGHT_GREP_PATTERN:-}" ]]; then
        if [[ -z "${_AIT_EMPTY_PATTERN_WARNED:-}" ]]; then
            warn "Copyright/author checks skipped: identity.sh not loaded (empty grep patterns)"
            WARNINGS=$((WARNINGS + 1))
            _AIT_EMPTY_PATTERN_WARNED=1
        fi
        return 0
    fi

    # Check if file contains copyright notice
    if ! head -10 "$file" 2>/dev/null | grep -qi "$AIT_COPYRIGHT_GREP_PATTERN"; then
        # Get header content: prefer generated from identity config, fall back to template file
        local template header_content
        header_content=$(generate_header "$ext")
        if [[ -z "$header_content" ]]; then
            template=$(get_header_template "$ext")
            if [[ -n "$template" && -f "$template" ]]; then
                header_content=$(cat "$template")
            fi
        fi

        if [[ -n "$header_content" ]]; then
            # Auto-add header, preserving shebang and permissions
            local tmpfile
            tmpfile=$(mktemp)
            local first_line
            first_line=$(head -1 "$file")

            if [[ "$first_line" == "#!"* ]]; then
                # File has shebang - put it first, then header
                {
                    echo "$first_line"
                    echo "$header_content"
                    echo ""
                    tail -n +2 "$file"
                } >"$tmpfile"
            else
                # No shebang - header goes first
                {
                    echo "$header_content"
                    echo ""
                    cat "$file"
                } >"$tmpfile"
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

    # Guard: empty grep pattern matches everything, silently disabling the check (#3025)
    if [[ -z "${AIT_AUTHOR_GREP_PATTERN:-}" ]]; then
        if [[ -z "${_AIT_EMPTY_PATTERN_WARNED:-}" ]]; then
            warn "Copyright/author checks skipped: identity.sh not loaded (empty grep patterns)"
            WARNINGS=$((WARNINGS + 1))
            _AIT_EMPTY_PATTERN_WARNED=1
        fi
        return 0
    fi

    # Valid author patterns from identity config
    if ! grep -qi "$AIT_AUTHOR_GREP_PATTERN" "$file" 2>/dev/null; then
        warn "Missing or incorrect author in $file (expected: $AIT_OWNER_NAME)"
        WARNINGS=$((WARNINGS + 1))
    fi
}

check_new_staged_files() {
    local current_staged_files new_files
    current_staged_files=$(get_staged_files)

    new_files=$(comm -13 \
        <(normalize_file_list "$INITIAL_STAGED_FILES") \
        <(normalize_file_list "$current_staged_files") ||
        true)

    if [[ -n "$new_files" ]]; then
        warn "New files staged during pre-commit checks. Review before commit:"
        while IFS= read -r file; do
            [[ -z "$file" ]] && continue
            echo "  $file"
        done <<<"$new_files"
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
    done <".ignore-check-exclude"
fi
if [[ -n "${IGNORE_CHECK_EXCLUDE:-}" ]]; then
    IFS=':' read -ra env_excludes <<<"$IGNORE_CHECK_EXCLUDE"
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
    js | ts | tsx)
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
        done <<<"$new_ignores"
    fi
}

# Check staged files for new ignores (ALL roles, BLOCKING)
for file in $STAGED_FILES; do
    if [[ -f "$file" ]]; then
        check_new_ignores "$file"
    fi
done

# Regression lookback window (seconds)
REGRESSION_MIN_LOOKBACK_SEC=3600  # 1 hour
REGRESSION_MAX_LOOKBACK_SEC=86400 # 24 hours
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

    local file staged_hash recent_commits commit parent_hash commit_hash
    local my_worker_pattern regression_found=0

    # Build pattern for our own commits (to skip them)
    # Note: AI_WORKER_ID is guaranteed set since this function is only called in multi-worker mode
    my_worker_pattern="\\[W${AI_WORKER_ID}\\]"

    # --- Optimization (#2848): reduce subprocess calls ---
    # 1. Per-file git log includes subject (--format="%H %s"), eliminating
    #    per-commit git log -1 calls (AC2: batched subject lookup).
    # 2. Per-file git log bounded by --max-count (AC1).
    # 3. Batch rev-parse: collect all ref queries, resolve in one git cat-file
    #    --batch-check call per file (reduces 2N rev-parse to 1 subprocess).
    local max_commits=10  # Cap per-file commit lookback (#2848 AC1)

    # Note: We iterate on word-split STAGED_FILES. This matches git diff output format
    # where filenames with spaces are quoted. For our use case (AI repos), filenames
    # with spaces are not expected.
    for file in $STAGED_FILES; do
        [[ -f "$file" ]] || continue

        # Get hash of staged (index) version
        # git rev-parse ":file" returns blob hash from the index (1 subprocess vs 2 for pipe)
        staged_hash=$(git rev-parse ":$file" 2>/dev/null) || continue

        # Find recent commits that touched this file with subject inline (#2848 AC2)
        # Format: "HASH SUBJECT" - one git call replaces N+1 calls (log + N subjects)
        local commits_with_subjects
        commits_with_subjects=$(git log --since="${lookback} seconds ago" "$base_ref" \
            --max-count="$max_commits" --format="%H %s" -- "$file" 2>/dev/null) || continue

        # Filter out our own commits first (no subprocess needed)
        local other_commits=""
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local commit_msg="${line#* }"
            if [[ "$commit_msg" =~ $my_worker_pattern ]]; then
                continue
            fi
            local commit_hash_only="${line%% *}"
            other_commits="${other_commits}${commit_hash_only}"$'\n'
        done <<< "$commits_with_subjects"

        [[ -z "$other_commits" ]] && continue

        # Batch resolve parent^ and commit hashes via git cat-file --batch-check (#2848 AC3)
        # Build input: "commit^:file\ncommit:file\n..." for all other commits
        local batch_input=""
        while IFS= read -r commit; do
            [[ -z "$commit" ]] && continue
            batch_input="${batch_input}${commit}^:${file}"$'\n'
            batch_input="${batch_input}${commit}:${file}"$'\n'
        done <<< "$other_commits"

        [[ -z "$batch_input" ]] && continue

        # One subprocess resolves all hashes for this file
        local batch_output
        batch_output=$(printf '%s' "$batch_input" | git cat-file --batch-check 2>/dev/null) || continue

        # Parse batch output into array (no per-line sed subprocess calls)
        # Each commit produces 2 lines: parent^:file then commit:file
        # Format per line: "HASH type size" or "REF missing"
        local batch_lines=()
        while IFS= read -r _bline; do
            batch_lines+=("$_bline")
        done <<< "$batch_output"

        local line_idx=0
        while IFS= read -r commit; do
            [[ -z "$commit" ]] && continue
            local parent_line="${batch_lines[$line_idx]:-}"
            local cmt_line="${batch_lines[$((line_idx + 1))]:-}"
            line_idx=$((line_idx + 2))

            # Extract hash (first field) from batch-check output
            parent_hash="${parent_line%% *}"
            commit_hash="${cmt_line%% *}"

            # Skip if either ref is missing (file didn't exist in parent or commit)
            [[ "$parent_line" == *"missing"* ]] && continue
            [[ "$cmt_line" == *"missing"* ]] && continue

            # REGRESSION: Our staged version matches the BEFORE state (parent),
            # meaning we're reverting someone else's changes
            if [[ "$staged_hash" == "$parent_hash" && "$parent_hash" != "$commit_hash" ]]; then
                warn "File $file may revert changes from commit ${commit:0:7}"
                warn "  That commit changed the file, but your version matches the pre-change state"
                warn "  Consider: git fetch && git checkout $base_ref -- $file"
                regression_found=1
                WARNINGS=$((WARNINGS + 1))
            fi
        done <<< "$other_commits"
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
# BLOCK commits that include files tracked by OTHER workers.
# This prevents workers from stepping on each other's files in shared checkout.
#
# Design change (Part of #2365): Instead of checking "is file in MY tracker"
# (which gets stale mid-iteration), check "is file in ANOTHER worker's tracker"
# (which correctly blocks stepping on other workers but allows committing
# files we modified during this session).
#
# Override: AIT_ALLOW_UNTRACKED=1 (for emergencies only)
check_untracked_staged_files() {
    # Skip if no staged files
    [[ -z "$STAGED_FILES" ]] && return 0

    # Use shared tracker_utils module to check ownership
    # shellcheck disable=SC2086  # Word splitting is intentional for staged files
    local result
    result=$(python3 "$TRACKER_UTILS" check_ownership "$REPO_ROOT" "$AI_WORKER_ID" $STAGED_FILES 2>/dev/null || echo "ERROR")

    if [[ "$result" == "OK" ]]; then
        # No conflicts - proceed
        return 0
    fi

    if [[ "$result" == "ERROR" ]]; then
        echo "ERROR: Failed to check worker file ownership"
        echo "  tracker_utils.py unavailable or failed to run"
        echo "  Ensure python3 is installed and retry."
        exit 1
    fi

    # Parse conflicts: "CONFLICT:worker_id:filename"
    # Parse invalid trackers: "INVALID:worker_id:tracker_file"
    local conflicts=()
    local invalid_trackers=()
    while IFS= read -r line; do
        if [[ "$line" == CONFLICT:* ]]; then
            local rest="${line#CONFLICT:}"
            local owner="${rest%%:*}"
            local file="${rest#*:}"
            conflicts+=("$owner:$file")
        elif [[ "$line" == INVALID:* ]]; then
            local rest="${line#INVALID:}"
            local owner="${rest%%:*}"
            local tracker_file="${rest#*:}"
            invalid_trackers+=("$owner:$tracker_file")
        fi
    done <<<"$result"

    if [[ ${#invalid_trackers[@]} -gt 0 ]]; then
        echo "ERROR: Worker session state is invalid:"
        for entry in "${invalid_trackers[@]}"; do
            local wid="${entry%%:*}"
            if [[ "$wid" =~ ^[0-9]+$ ]]; then
                echo "  W${wid}: session tracker is corrupted or mismatched"
            else
                echo "  ${wid}: session tracker is corrupted or mismatched"
            fi
        done
        echo ""
        echo "Ownership checks are unsafe with corrupted session state."
        echo "Resolution: Restart the affected worker session(s)."
        exit 1
    fi

    if [[ ${#conflicts[@]} -gt 0 ]]; then
        echo "ERROR: Worker $AI_WORKER_ID staging files owned by other workers:"
        for entry in "${conflicts[@]}"; do
            local wid="${entry%%:*}"
            local fname="${entry#*:}"
            echo "  $fname (owned by W$wid)"
        done
        echo ""
        echo "These files are tracked by another active worker."
        echo "Workers must only commit files they own to prevent interference."
        echo ""
        if [[ "${AIT_ALLOW_UNTRACKED:-}" == "1" ]]; then
            warn "Proceeding anyway (AIT_ALLOW_UNTRACKED=1)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo "Contact User if this is blocking legitimate work."
            exit 1
        fi
    fi
}

# === Worker Tracker Existence Check (#2364) ===
# BLOCK workers from committing if they don't have a tracker file.
# This catches initialization failures that lead to false blocks.
# The tracker should be created by looper at session start.
# No override - this indicates a broken session state.
check_worker_tracker_exists() {
    local tracker_file="$REPO_ROOT/.worker_${AI_WORKER_ID}_files.json"

    # Use shared tracker_utils module for PID check
    local status
    status=$(python3 "$TRACKER_UTILS" check_alive "$tracker_file" 2>/dev/null || echo "error")

    case "$status" in
    alive)
        # Tracker exists and PID is running - all good
        ;;
    missing)
        echo "ERROR: Worker $AI_WORKER_ID session not properly initialized" >&2
        echo "" >&2
        echo "  The file tracker was not created at session start." >&2
        echo "" >&2
        echo "  Resolution: Restart the worker session properly." >&2
        echo "" >&2
        exit 1
        ;;
    dead)
        echo "ERROR: Worker $AI_WORKER_ID session state is stale" >&2
        echo "" >&2
        echo "  The session tracker belongs to a previous session that has ended." >&2
        echo "" >&2
        echo "  Resolution: Restart the worker session (looper cleans up on startup)." >&2
        echo "" >&2
        exit 1
        ;;
    invalid)
        echo "ERROR: Worker $AI_WORKER_ID session tracker is corrupted" >&2
        echo "" >&2
        echo "  The session state is not trustworthy for ownership checks." >&2
        echo "" >&2
        echo "  Resolution: Restart the worker session (looper recreates it)." >&2
        echo "" >&2
        exit 1
        ;;
    error | *)
        echo "ERROR: Failed to validate worker $AI_WORKER_ID session state" >&2
        echo "" >&2
        echo "  Session validation requires python3 and tracker utilities." >&2
        echo "" >&2
        exit 1
        ;;
    esac
}

# Run worker file tracking check for multi-worker mode
# Only for WORKER role - other roles don't create tracker files (runner_base.py:177)
if [[ -n "${AI_WORKER_ID:-}" ]] && [[ "${AI_ROLE:-}" == "WORKER" ]]; then
    check_worker_tracker_exists
    check_untracked_staged_files
fi

# === Non-Worker Staged Files Block (#2812, #2794, #2729, #2405) ===
# BLOCK non-workers (Manager, Prover, Researcher) from committing ANY staged files.
# Non-Worker roles produce commit messages (text), not code. Any staged files in
# a non-Worker commit are contamination from a Worker's staged-but-uncommitted work
# in the shared worktree. This is the root cause behind 4 diagnosis commits and 3
# prior issues (#2405, #2729, #2794) with zero code fixes.
#
# This replaces the tracker-dependent check (#1922) which only caught files tracked
# by active workers and missed cases where trackers didn't exist.
#
# Override: AIT_ALLOW_WORKER_FILES=1 (for emergencies only)
check_non_worker_staged_files() {
    # Only check for non-worker AI roles
    local role="${AI_ROLE:-}"
    case "$role" in
    MANAGER | PROVER | RESEARCHER) ;;
    *) return 0 ;; # USER or unknown - skip check
    esac

    # Skip if no staged files - this is the expected case for non-Worker roles
    [[ -z "$STAGED_FILES" ]] && return 0

    # Non-Worker roles should not commit code files. Staged code files are likely
    # contamination from a Worker's staged-but-uncommitted work in the shared worktree.
    # Exception: documentation output directories that non-Workers legitimately produce (#2861).
    local exempt_prefixes=("reports/" "designs/" "ideas/" "postmortems/" "diagrams/" "docs/")
    local blocked_count=0
    local blocked_list=()
    for file in $STAGED_FILES; do
        local is_exempt=false
        for prefix in "${exempt_prefixes[@]}"; do
            if [[ "$file" == "$prefix"* ]]; then
                is_exempt=true
                break
            fi
        done
        if [[ "$is_exempt" != "true" ]]; then
            blocked_count=$((blocked_count + 1))
            blocked_list+=("$file")
        fi
    done

    if [[ $blocked_count -gt 0 ]]; then
        error "CROSS-ROLE STAGING CONTAMINATION: $role has $blocked_count blocked staged file(s)"
        echo ""
        echo "  Non-Worker roles (Manager, Prover, Researcher) must not commit code files."
        echo "  These staged files were likely left by a Worker in the shared worktree:"
        echo ""
        for fname in "${blocked_list[@]}"; do
            echo "    $fname"
        done
        echo ""
        echo "  This prevents incorrect attribution and lost Worker commits."
        echo "  See: #2812, #2794, #2729, #2405"
        echo ""
        local _old_ifs="$IFS"
        IFS=" "
        echo "  Exempt paths (allowed for non-Workers): ${exempt_prefixes[*]}"
        IFS="$_old_ifs"
        echo ""
        echo "  Resolution:"
        echo "    1. Unstage blocked files: git restore --staged <file>"
        echo "    2. Use: git commit --allow-empty -m \"your message\""
        echo "    3. Or use: git commit --allow-empty --only -m \"your message\""
        echo ""
        if [[ "${AIT_ALLOW_WORKER_FILES:-}" == "1" ]]; then
            warn "Proceeding anyway (AIT_ALLOW_WORKER_FILES=1)"
            WARNINGS=$((WARNINGS + 1))
        else
            exit 1
        fi
    fi
}

# Run non-worker staged files check for all autonomous AI roles (#2812)
# This check fires for ANY non-Worker AI role, regardless of whether worker
# trackers exist. It catches the root cause: non-Workers should never have
# staged files, period.
if [[ -n "${AI_ROLE:-}" ]]; then
    check_non_worker_staged_files
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
        js | ts | tsx | jsx) has_js=1 ;;
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
# Default: HARD BLOCK commits outside worker's zone to prevent conflicts (#2417)
# Override: ZONE_ENFORCEMENT=advisory for warnings only (emergencies)
if [[ -n "${AI_WORKER_ID:-}" && -n "$STAGED_FILES" ]]; then
    # Build file list from staged files
    zone_files=()
    for file in $STAGED_FILES; do
        zone_files+=("$file")
    done

    if [[ ${#zone_files[@]} -gt 0 ]]; then
        # Check if looper.zones module is available (skip in non-ai_template repos)
        if python3 -c "import looper.zones" 2>/dev/null; then
            # Check files using zones module - capture output for error details
            zone_output=$(python3 -m looper.zones check "${zone_files[@]}" 2>&1) || zone_exit=$?
            zone_exit=${zone_exit:-0}

            if [[ $zone_exit -ne 0 ]]; then
                if [[ "${ZONE_ENFORCEMENT:-strict}" == "advisory" ]]; then
                    warn "Zone advisory: some files are outside worker's zone"
                    echo "$zone_output" | sed 's/^/  /'
                    warn "  Override active (ZONE_ENFORCEMENT=advisory)"
                    WARNINGS=$((WARNINGS + 1))
                else
                    error "ZONE VIOLATION: Files outside worker's assigned zone"
                    echo ""
                    echo "$zone_output" | sed 's/^/  /'
                    echo ""
                    echo "  In multi-worker mode, each worker can only commit files in their zone."
                    echo "  This prevents merge conflicts and lost work between workers."
                    echo ""
                    echo "  Options:"
                    echo "    1. Stage only files in your zone"
                    echo "    2. Update zone config in .looper_config.json"
                    echo "    3. Emergency override: ZONE_ENFORCEMENT=advisory git commit ..."
                    echo ""
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        fi
        # Skip zone check silently if looper.zones not available (non-ai_template repo)
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

# === AGENTS.md Direct-Edit Warning ===
# AGENTS.md is a minimal stub. Codex instructions are injected by the looper
# at runtime via build_codex_context(). Warn if someone edits it directly.
check_agents_md_direct_edit() {
    for file in $STAGED_FILES; do
        if [[ "$file" == "AGENTS.md" ]]; then
            warn "AGENTS.md should not be edited directly!"
            warn "  AGENTS.md is a stub. Codex rules are injected by the looper at runtime."
            warn "  - Edit CLAUDE.md or .claude/rules/*.md for shared instructions"
            warn "  - Edit .claude/codex.md for Codex-only overrides"
            WARNINGS=$((WARNINGS + 1))
            break
        fi
    done
}

check_agents_md_direct_edit

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

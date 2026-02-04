#!/bin/bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# commit-msg-hook.sh - Git commit-msg hook for structured commits
#
# PURPOSE: Auto-adds iteration numbers, validates commit structure, warns on missing sections.
# CALLED BY: Git (installed to .git/hooks/commit-msg by looper.py)
# REFERENCED: .claude/rules/ai_template.md (commit workflow)
#
# The LLM writes structured commits. This hook:
# - Adds role prefix [U]N, [W]N, [P]N, [R]N, or [M]N
# - Warns if required sections (## Changes, ## Next) are missing
# - Validates issue links when present (but doesn't require them)
#
# NOTE: Do NOT use gh --jq or -q flags anywhere in this script.
# The gh CLI v2.83.2+ has caching bugs with --jq that return stale data.
# Always pipe to external jq instead (see #1047 for details).

COMMIT_MSG_FILE="$1"
if [[ -z "$COMMIT_MSG_FILE" ]]; then
    echo "" >&2
    echo "❌ ERROR: Commit message file path not provided to commit-msg hook" >&2
    echo "   Possible causes:" >&2
    echo "     - pre-commit misconfigured (pass_filenames=false)" >&2
    echo "     - Using 'pre-commit try-repo' without --commit-msg-filename" >&2
    echo "   For manual testing:" >&2
    echo "     pre-commit try-repo . --hook-stage commit-msg --commit-msg-filename .git/COMMIT_EDITMSG" >&2
    echo "" >&2
    exit 1
fi

if [[ ! -e "$COMMIT_MSG_FILE" ]]; then
    echo "" >&2
    echo "❌ ERROR: Commit message file '$COMMIT_MSG_FILE' does not exist" >&2
    echo "   This usually means pre-commit is misconfigured or hook input is missing." >&2
    echo "" >&2
    exit 1
fi

if [[ ! -f "$COMMIT_MSG_FILE" ]]; then
    echo "" >&2
    echo "❌ ERROR: Commit message file '$COMMIT_MSG_FILE' is not a regular file" >&2
    echo "   This usually means pre-commit is misconfigured or hook input is invalid." >&2
    echo "" >&2
    exit 1
fi

if [[ ! -r "$COMMIT_MSG_FILE" ]]; then
    echo "" >&2
    echo "❌ ERROR: Commit message file '$COMMIT_MSG_FILE' is not readable" >&2
    echo "   This usually means pre-commit is misconfigured or hook input is invalid." >&2
    echo "" >&2
    exit 1
fi

if ! MSG=$(cat "$COMMIT_MSG_FILE"); then
    echo "" >&2
    echo "❌ ERROR: Failed to read commit message file '$COMMIT_MSG_FILE'" >&2
    echo "" >&2
    exit 1
fi

# Skip merge commits entirely (no validation needed)
if [[ "$MSG" == Merge* ]]; then
    exit 0
fi

# ENFORCE: Reject literal $ITER or other shell variable placeholders in commit title
# These indicate template strings that weren't substituted (common in looper prompts)
if echo "$MSG" | head -1 | grep -qE '\$[A-Z_]+'; then
    PLACEHOLDER=$(echo "$MSG" | head -1 | grep -oE '\$[A-Z_]+' | head -1)
    echo "" >&2
    echo "❌ ERROR: Commit title contains unsubstituted variable placeholder: $PLACEHOLDER" >&2
    echo "" >&2
    echo "   The commit title appears to contain a template variable that wasn't expanded." >&2
    echo "   Common causes:" >&2
    echo "     - Using single quotes instead of double quotes in shell scripts" >&2
    echo "     - Variable not exported to the environment" >&2
    echo "     - Typo in variable name" >&2
    echo "" >&2
    exit 1
fi

# ENFORCE: Reject command substitution placeholders in ## Verified section (#2209)
# Per #1879, ## Verified must contain ACTUAL command output, not placeholders.
# Pattern: $(command), `command`, or ${var} in the ## Verified block
# These indicate template strings that weren't expanded or copy-paste errors.
# Note: Two-step extraction handles ## Verified at EOF (no following ## section)
VERIFIED_SECTION=$(echo "$MSG" | sed -n '/^## Verified/,$p' | sed -n '1p; 2,${ /^## /q; p; }')
if [[ -n "$VERIFIED_SECTION" ]]; then
    # Check for $(...) command substitution - indicates unexpanded template
    if echo "$VERIFIED_SECTION" | grep -qE '\$\([^)]+\)'; then
        PLACEHOLDER=$(echo "$VERIFIED_SECTION" | grep -oE '\$\([^)]+\)' | head -1)
        # Extract command inside $(...) by removing leading $( and trailing )
        INNER_CMD="${PLACEHOLDER#\$(}"
        INNER_CMD="${INNER_CMD%)}"
        echo "" >&2
        echo "❌ ERROR: ## Verified contains command substitution placeholder: $PLACEHOLDER" >&2
        echo "" >&2
        echo "   Per #1879: ## Verified must contain ACTUAL command output you ran." >&2
        echo "   The placeholder '$PLACEHOLDER' was not expanded - this is NOT real output." >&2
        echo "" >&2
        echo "   Fix: Run the actual command and paste its output:" >&2
        echo "     1. Run: $INNER_CMD" >&2
        echo "     2. Copy the actual output" >&2
        echo "     3. Paste into ## Verified section" >&2
        echo "" >&2
        echo "   Or use a heredoc to safely include output:" >&2
        echo "     git commit -m \"\$(cat <<'EOF'" >&2
        echo "     ... message with actual output ..." >&2
        echo "     EOF" >&2
        echo "     )\"" >&2
        echo "" >&2
        exit 1
    fi
    # Check for backtick command substitution
    if echo "$VERIFIED_SECTION" | grep -qE '\`[^\`]+\`'; then
        PLACEHOLDER=$(echo "$VERIFIED_SECTION" | grep -oE '\`[^\`]+\`' | head -1)
        echo "" >&2
        echo "❌ ERROR: ## Verified contains backtick command substitution: $PLACEHOLDER" >&2
        echo "" >&2
        echo "   Per #1879: ## Verified must contain ACTUAL command output you ran." >&2
        echo "   Backticks cause command substitution - this may inject unexpected content." >&2
        echo "" >&2
        echo "   Fix: Use a heredoc with single-quoted delimiter to prevent substitution:" >&2
        echo "     git commit -m \"\$(cat <<'EOF'" >&2
        echo "     ... message with actual output ..." >&2
        echo "     EOF" >&2
        echo "     )\"" >&2
        echo "" >&2
        exit 1
    fi
fi

# ENFORCE: Reject commit messages containing environment variable dumps (#2067)
# This catches accidental backticks in shell-quoted commit messages, which cause
# command substitution to inject env output into the message.
# Detection: Look for common env vars or high density of KEY=VALUE lines
# Pattern matches exact vars (PATH=, HOME=) or prefix vars (SSH_*, LC_*, XDG_* followed by any suffix then =)
ENV_VAR_PATTERNS='^(PATH=|HOME=|USER=|SHELL=|TERM=|PWD=|LANG=|DISPLAY=|EDITOR=|VISUAL=|TMPDIR=|HOSTNAME=|LOGNAME=|MAIL=|OLDPWD=|SHLVL=|_=|LC_[A-Z_]*=|SSH_[A-Z_]*=|XDG_[A-Z_]*=)'
ENV_VAR_MATCHES=$(printf '%s\n' "$MSG" | grep -cE "$ENV_VAR_PATTERNS" 2>/dev/null | tr -d '[:space:]' || echo "0")
ENV_VAR_MATCHES=${ENV_VAR_MATCHES:-0}
# Also check for high density of generic KEY=VALUE patterns (threshold: 5+)
KEY_VALUE_MATCHES=$(printf '%s\n' "$MSG" | grep -cE '^[A-Z][A-Z0-9_]*=' 2>/dev/null | tr -d '[:space:]' || echo "0")
KEY_VALUE_MATCHES=${KEY_VALUE_MATCHES:-0}

if [[ "$ENV_VAR_MATCHES" -ge 3 ]] || [[ "$KEY_VALUE_MATCHES" -ge 5 ]]; then
    echo "" >&2
    echo "❌ ERROR: Commit message appears to contain an environment variable dump" >&2
    echo "" >&2
    echo "   Found $ENV_VAR_MATCHES common env vars, $KEY_VALUE_MATCHES KEY=VALUE lines" >&2
    echo "   This usually happens when backticks are used inside a shell-quoted commit message:" >&2
    echo "" >&2
    echo "     BAD:  git commit -m \"Fix \`bug\` in code\"  # backticks cause command substitution" >&2
    echo "" >&2
    echo "   Use a heredoc instead to safely pass commit messages:" >&2
    echo "" >&2
    echo "     GOOD: git commit -F - <<'EOF'" >&2
    echo "           Fix bug in code" >&2
    echo "           ## Changes" >&2
    echo "           ..." >&2
    echo "           EOF" >&2
    echo "" >&2
    echo "   See ai_template.md: \"Avoid backticks in shell-quoted strings\"" >&2
    echo "" >&2
    exit 1
fi

# ENFORCE: Reject AI attribution patterns (#1394)
# Claude Code's built-in PR template includes forbidden attribution. Block it.
# Patterns must be specific to avoid false positives on documentation.
AI_ATTRIBUTION_PATTERNS=(
    # Match actual co-author trailer (starts at beginning of line)
    "^Co-Authored-By:.*[Cc]laude"
    # Match generated-with footer (starts at beginning of line or after emoji)
    "^🤖.*[Gg]enerated"
    "^[Gg]enerated with.*[Cc]laude"
)
for pattern in "${AI_ATTRIBUTION_PATTERNS[@]}"; do
    if echo "$MSG" | grep -qE "$pattern"; then
        echo "" >&2
        echo "❌ ERROR: Commit message contains forbidden AI attribution" >&2
        echo "" >&2
        echo "   Found pattern matching: $pattern" >&2
        echo "" >&2
        echo "   Per ai_template.md: NEVER add Claude/AI attribution." >&2
        echo "   No \"Co-Authored-By: Claude\", no AI signatures." >&2
        echo "   Andrew Yates is the author." >&2
        echo "" >&2
        echo "   Remove the AI attribution line and retry." >&2
        echo "" >&2
        exit 1
    fi
done

# --- Helper: Check if full local mode is enabled ---
# Full local mode (AIT_LOCAL_MODE=full) skips ALL gh API calls
# This allows development when GitHub API is unavailable
is_full_local_mode() {
    [[ "${AIT_LOCAL_MODE:-}" == "full" ]]
}

# Flag for easy checking throughout the script
FULL_LOCAL_MODE=false
if is_full_local_mode; then
    FULL_LOCAL_MODE=true
fi

# --- Helper: Check if issue ID is a local issue (L-prefixed) ---
is_local_issue() {
    local issue_id="$1"
    [[ "$issue_id" =~ ^L[0-9]+$ ]]
}

# --- Helper: Get local issue field from .issues/*.md files ---
# Usage: local_issue_field <issue_id> <field>
# Returns: exit 0 + stdout = success, exit 1 = not found
# Field can be: state, title, body, labels (returns newline-separated label names)
local_issue_field() {
    local issue_id="$1"
    local field="$2"
    local issue_file=".issues/${issue_id}.md"

    if [[ ! -f "$issue_file" ]]; then
        return 1
    fi

    # Parse YAML frontmatter
    local frontmatter
    frontmatter=$(sed -n '/^---$/,/^---$/p' "$issue_file" | sed '1d;$d')

    case "$field" in
        state)
            echo "$frontmatter" | grep -E '^state:' | sed 's/^state:[[:space:]]*//' | tr '[:lower:]' '[:upper:]'
            ;;
        title)
            # Handle quoted or unquoted title
            local title_line
            title_line=$(echo "$frontmatter" | grep -E '^title:')
            if echo "$title_line" | grep -q '"'; then
                echo "$title_line" | sed 's/^title:[[:space:]]*//' | sed 's/^"//;s/"$//'
            else
                echo "$title_line" | sed 's/^title:[[:space:]]*//'
            fi
            ;;
        body)
            # Body is everything after the second --- until ## Comments (if present)
            sed -n '/^---$/,/^---$/!p' "$issue_file" | sed '1,/^---$/d' | sed '/^## Comments$/,$d'
            ;;
        labels)
            # Parse JSON array format: ["P2", "feature"]
            local labels_line
            labels_line=$(echo "$frontmatter" | grep -E '^labels:' | sed 's/^labels:[[:space:]]*//')
            echo "$labels_line" | tr -d '[]"' | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
            ;;
        *)
            echo "$frontmatter" | grep -E "^${field}:" | sed "s/^${field}:[[:space:]]*//"
            ;;
    esac
    return 0
}

# --- Helper: Get issue field with REST API fallback ---
# Usage: gh_issue_field <issue_num> <field> [repo]
# Returns: exit 0 + stdout = success, exit 1 = not found, exit 2 = rate limited
# Field can be: state, title, body, labels (returns newline-separated label names)
# IMPORTANT: Callers MUST check for exit code 2 and handle rate limiting!
gh_issue_field() {
    local issue_num="$1"
    local field="$2"
    local repo="${3:-}"
    local result=""
    local stderr_output=""
    local exit_code=0

    # Build gh command args
    local gh_args=("issue" "view" "$issue_num" "--json" "$field")
    if [[ -n "$repo" ]]; then
        gh_args+=("--repo" "$repo")
    fi

    # Try GraphQL first (default gh behavior)
    stderr_output=$(mktemp)
    if result=$(gh "${gh_args[@]}" 2>"$stderr_output"); then
        rm -f "$stderr_output"
        case "$field" in
            labels)
                # Handle both object format (real gh) and string format (wrapper cache)
                echo "$result" | jq -r '.labels[] | if type == "object" then .name else . end' 2>/dev/null || echo "$result"
                ;;
            state)
                echo "$result" | jq -r '.state' 2>/dev/null || echo "$result"
                ;;
            title)
                echo "$result" | jq -r '.title' 2>/dev/null || echo "$result"
                ;;
            body)
                echo "$result" | jq -r '.body // ""' 2>/dev/null || echo "$result"
                ;;
            *)
                echo "$result" | jq -r ".${field} // \"\"" 2>/dev/null || echo "$result"
                ;;
        esac
        return 0
    fi
    exit_code=$?

    # Check if it's a rate limit error
    local err_msg
    err_msg=$(cat "$stderr_output" 2>/dev/null)
    rm -f "$stderr_output"

    if echo "$err_msg" | grep -qiE 'rate.?limit|API rate limit'; then
        # GraphQL rate limited - try REST API fallback
        local api_path="repos/{owner}/{repo}/issues/$issue_num"
        if [[ -n "$repo" ]]; then
            api_path="repos/$repo/issues/$issue_num"
        fi

        local rest_result=""
        stderr_output=$(mktemp)
        if rest_result=$(gh api "$api_path" 2>"$stderr_output"); then
            rm -f "$stderr_output"
            case "$field" in
                state)
                    echo "$rest_result" | jq -r '.state' | tr '[:lower:]' '[:upper:]'
                    ;;
                title)
                    echo "$rest_result" | jq -r '.title'
                    ;;
                body)
                    echo "$rest_result" | jq -r '.body // ""'
                    ;;
                labels)
                    echo "$rest_result" | jq -r '.labels[].name'
                    ;;
                *)
                    echo "$rest_result" | jq -r ".$field // \"\""
                    ;;
            esac
            return 0
        fi

        # Check if REST also rate limited
        err_msg=$(cat "$stderr_output" 2>/dev/null)
        rm -f "$stderr_output"

        if echo "$err_msg" | grep -qiE 'rate.?limit|API rate limit'; then
            # Return exit code 2 to signal rate limiting to caller
            return 2
        fi

        # REST failed for other reason (likely 404 = issue doesn't exist)
        return 1
    fi

    # Not a rate limit error - propagate original failure (likely 404)
    return $exit_code
}

# Helper to exit with rate limit error message
gh_rate_limit_error() {
    local issue_num="$1"
    echo "" >&2
    echo "❌ ERROR: GitHub API rate limits exceeded (both GraphQL and REST)" >&2
    echo "   Cannot verify issue #$issue_num exists." >&2
    echo "" >&2
    echo "   Options:" >&2
    echo "     1. Wait for rate limit reset (~1 hour)" >&2
    echo "     2. Check rate limits: gh api rate_limit" >&2
    echo "     3. Skip validation: SKIP_ISSUE_CONFIRM=1 git commit ..." >&2
    echo "" >&2
    exit 1
}

# Check if commit already has our prefix (amend scenario)
# We still need to validate, but will skip rewriting
# Pattern matches [W]N, [W1]N, and machine-prefixed [sat-W1]N formats
ALREADY_PREFIXED=false
if echo "$MSG" | grep -qE '^\[([^]]+-)?(U|W|P|R|M)[0-9]*\][0-9]+:'; then
    ALREADY_PREFIXED=true
fi

# --- Extract info automatically ---

# 1. Role: from env, existing prefix, or default to USER
# For already-prefixed commits, extract role from prefix to validate correctly
# Pattern handles [W], [W1], and [sat-W1] formats - extract base role letter
if [[ "$ALREADY_PREFIXED" == "true" ]]; then
    PREFIX_CHAR=$(echo "$MSG" | sed -nE 's/^\[[^]]*-?([UWPRM])[0-9]*\].*/\1/p')
    case "$PREFIX_CHAR" in
        W) ROLE="WORKER" ;;
        P) ROLE="PROVER" ;;
        R) ROLE="RESEARCHER" ;;
        M) ROLE="MANAGER" ;;
        *) ROLE="USER" ;;
    esac
else
    ROLE="${AI_ROLE:-USER}"
fi

# 2. Iteration: from git log (find max for this role's prefix)
# Support multi-worker mode: [W1], [W2] etc when AI_WORKER_ID is set
case "$ROLE" in
    WORKER)     BASE_PREFIX="W" ;;
    PROVER)     BASE_PREFIX="P" ;;
    RESEARCHER) BASE_PREFIX="R" ;;
    MANAGER)    BASE_PREFIX="M" ;;
    *)          BASE_PREFIX="U" ;;  # USER or any other role
esac

# Build PREFIX: W or W1 (for multi-worker), add machine prefix if set
if [[ -n "$AI_WORKER_ID" ]]; then
    PREFIX_BASE="${BASE_PREFIX}${AI_WORKER_ID}"
else
    PREFIX_BASE="$BASE_PREFIX"
fi

if [[ -n "${AI_MACHINE_PREFIX:-}" ]]; then
    PREFIX="${AI_MACHINE_PREFIX}-${PREFIX_BASE}"
else
    PREFIX="$PREFIX_BASE"
fi

# Search for iterations - need to match [W]N, [W1]N, and [sat-W1]N formats
# For iteration assignment, we use the BASE_PREFIX (all workers share iteration space)
# This prevents duplicate iterations like [W1]42 and [W2]42
LAST_ITER=$(git log --oneline -100 2>/dev/null | grep -oE "\[[^]]*-?${BASE_PREFIX}[0-9]*\]#?[0-9]+" | sed -E "s/\[[^]]*-?${BASE_PREFIX}[0-9]*\]#?//" | sort -rn | head -1)
NEXT_ITER=$((${LAST_ITER:-0} + 1))

# 3. Check for [maintain] tag
IS_MAINTAIN=false
if echo "$MSG" | head -1 | grep -qi '\[maintain\]'; then
    IS_MAINTAIN=true
fi

# --- Check for cross-role file contamination ---
# Warn if staged files include other roles' directories (issue #871)
if [[ "$ROLE" != "USER" ]]; then
    STAGED_FILES=$(git diff --cached --name-only 2>/dev/null || true)

    # Define forbidden directories per role
    FORBIDDEN_DIRS=""
    case "$ROLE" in
        WORKER)     FORBIDDEN_DIRS="reports/manager reports/prover reports/researcher" ;;
        PROVER)     FORBIDDEN_DIRS="reports/manager reports/worker reports/researcher worker_logs" ;;
        RESEARCHER) FORBIDDEN_DIRS="reports/manager reports/worker reports/prover worker_logs" ;;
        MANAGER)    FORBIDDEN_DIRS="reports/worker reports/prover reports/researcher" ;;
    esac

    CROSS_ROLE_FILES=""
    for dir in $FORBIDDEN_DIRS; do
        MATCHES=$(echo "$STAGED_FILES" | grep "^$dir/" 2>/dev/null || true)
        if [[ -n "$MATCHES" ]]; then
            CROSS_ROLE_FILES="$CROSS_ROLE_FILES$MATCHES"$'\n'
        fi
    done

    if [[ -n "${CROSS_ROLE_FILES// }" ]]; then
        echo "" >&2
        echo "⚠️  WARNING: Staged files from other roles' directories:" >&2
        echo "$CROSS_ROLE_FILES" | head -5 | sed 's/^/   /' >&2
        echo "   This may indicate cross-role contamination (see #871)" >&2
        echo "" >&2
    fi
fi

# --- Check for zero-diff commits (#969, #1379) ---
# Warn or error when commit message makes claims about changes but no files are actually modified.
# This prevents false claims that waste reviewer time and incorrectly close issues.
# - USER/MANAGER: Warning only (may have valid reasons for empty commits)
# - WORKER/PROVER/RESEARCHER: Error (must not claim changes that don't exist)
# Skip for merge commits (which legitimately have no diff sometimes).
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
    STAGED_DIFF=$(git diff --cached --stat 2>/dev/null || true)
    if [[ -z "${STAGED_DIFF// }" ]]; then
        # Check if commit message implies file changes
        MSG_LOWER=$(echo "$MSG" | tr '[:upper:]' '[:lower:]')
        CHANGE_KEYWORDS=$(echo "$MSG_LOWER" | grep -oE '(add|remove|fix|update|refactor|rename|delete|create|implement|change|modify|move|extend|enhance)' | head -3 | tr '\n' ',' | sed 's/,$//')
        if [[ -n "$CHANGE_KEYWORDS" ]]; then
            # Error for non-Manager/non-USER roles, warning for Manager/USER
            if [[ "$ROLE" != "MANAGER" ]] && [[ "$ROLE" != "USER" ]]; then
                echo "" >&2
                echo "❌ ERROR: Commit message claims changes but no files are staged" >&2
                echo "   Commit message mentions: $CHANGE_KEYWORDS" >&2
                echo "   But: git diff --cached --stat shows no file changes" >&2
                echo "" >&2
                echo "   This may indicate:" >&2
                echo "     - Work wasn't actually completed" >&2
                echo "     - Files weren't staged (git add)" >&2
                echo "     - A false claim about changes made" >&2
                echo "" >&2
                echo "   To bypass (if intentional): ALLOW_EMPTY_CLAIM=1 git commit ..." >&2
                echo "" >&2
                if [[ -z "${ALLOW_EMPTY_CLAIM:-}" ]]; then
                    exit 1
                fi
            else
                echo "" >&2
                echo "⚠️  WARNING: Commit message claims changes but no files are staged" >&2
                echo "   Commit message mentions: $CHANGE_KEYWORDS" >&2
                echo "   But: git diff --cached --stat shows no file changes" >&2
                echo "" >&2
                echo "   This may indicate:" >&2
                echo "     - Work wasn't actually completed" >&2
                echo "     - Files weren't staged (git add)" >&2
                echo "     - A false claim about changes made" >&2
                echo "" >&2
            fi
        fi
    fi
fi

# --- Helper: Check if Rust file changes are only in test blocks (#1585) ---
# Returns 0 if ALL changes are in #[cfg(test)] or mod tests blocks, 1 otherwise
# This allows PROVER to add tests inside source files without triggering prod code error
rust_changes_only_in_tests() {
    local file="$1"
    local diff_output
    diff_output=$(git diff --cached -U0 "$file" 2>/dev/null) || return 1

    # Get line numbers of added/modified lines (lines starting with +, excluding +++ header)
    local changed_lines
    changed_lines=$(echo "$diff_output" | grep -E '^@@' | sed -E 's/^@@ -[0-9,]+ \+([0-9]+).*/\1/')

    if [[ -z "$changed_lines" ]]; then
        return 0  # No changes = not production code
    fi

    # Get the full file content (staged version)
    local file_content
    file_content=$(git show ":$file" 2>/dev/null) || return 1

    # Find test block ranges: lines between #[cfg(test)] and matching close brace
    # Track line numbers that are inside test blocks
    local in_test_block=0
    local brace_depth=0
    local test_start_depth=0
    local test_block_opened=0  # Track if we've seen the opening brace
    local line_num=0
    local test_lines=""

    while IFS= read -r line; do
        ((line_num++))

        # Check for #[cfg(test)] or mod tests (with or without brace on same line)
        # Note: mod tests without brace will have in_test_block=1 until the brace is seen
        if echo "$line" | grep -qE '^\s*#\[cfg\(test\)\]|^\s*mod\s+tests\s*(\{|$)'; then
            in_test_block=1
            test_start_depth=$brace_depth
            test_block_opened=0
        fi

        # Track brace depth
        local open_braces close_braces
        open_braces=$(echo "$line" | tr -cd '{' | wc -c)
        close_braces=$(echo "$line" | tr -cd '}' | wc -c)

        # Mark test block as opened when we see first brace
        if [[ $in_test_block -eq 1 ]] && [[ $open_braces -gt 0 ]] && [[ $test_block_opened -eq 0 ]]; then
            test_block_opened=1
        fi

        brace_depth=$((brace_depth + open_braces - close_braces))

        # Mark line as test line if inside test block
        if [[ $in_test_block -eq 1 ]]; then
            test_lines="$test_lines $line_num"
        fi

        # Exit test block when we return to the depth before it started
        # Only check this AFTER the test block has been opened (seen first brace)
        if [[ $in_test_block -eq 1 ]] && [[ $test_block_opened -eq 1 ]] && [[ $brace_depth -le $test_start_depth ]]; then
            in_test_block=0
        fi
    done <<< "$file_content"

    # Check if all changed lines are in test blocks
    # Parse diff hunks to find actual changed line numbers
    local all_in_tests=1
    local hunk_info
    while IFS= read -r hunk_info; do
        # Extract starting line and count from @@ -X,Y +A,B @@
        local start_line count
        start_line=$(echo "$hunk_info" | sed -E 's/^@@ -[0-9,]+ \+([0-9]+)(,[0-9]+)? @@.*/\1/')
        count=$(echo "$hunk_info" | sed -E 's/^@@ -[0-9,]+ \+[0-9]+,?([0-9]*) @@.*/\1/')
        [[ -z "$count" ]] && count=1

        for ((i=0; i<count; i++)); do
            local check_line=$((start_line + i))
            if ! echo "$test_lines" | grep -qw "$check_line"; then
                all_in_tests=0
                break 2
            fi
        done
    done < <(echo "$diff_output" | grep -E '^@@')

    return $((1 - all_in_tests))
}

# --- Check for non-Worker committing production code (#1098, #933, #1466, #1585) ---
# Only Worker should write production code; other roles audit/verify/manage
# PROVER: ERROR (blocks commit) - verification role must not write production code
#         EXCEPTION: Changes to inline test blocks (#[cfg(test)], mod tests) are allowed (#1585)
# RESEARCHER/MANAGER: WARNING - allows commit but warns about boundary violation
# Skip if ALLOW_NON_WORKER_CODE is set (for emergency fixes)
if [[ "$ROLE" != "WORKER" ]] && [[ "$ROLE" != "USER" ]] && [[ -z "${ALLOW_NON_WORKER_CODE:-}" ]]; then
    # Check for production code files in staged changes
    # Include: .rs, .py, .ts, .tsx, .js, .jsx, .go, .java, .swift, .kt, .c, .cpp, .h
    # Exclude directories: tests/ (any depth), proofs/, spec/ (any depth), docs/, reports/, .claude/, postmortems/, ideas/
    # Exclude verification files: test_*.py, *_test.py, *_test.rs, *_test.go, *.test.(ts|tsx|js), *.spec.(ts|tsx|js)
    # Exclude Kani verification harnesses: *.kani.rs (#1466)
    INITIAL_PROD_FILES=$(git diff --cached --name-only -- \
        '*.rs' '*.py' '*.ts' '*.tsx' '*.js' '*.jsx' '*.go' '*.java' '*.swift' '*.kt' '*.c' '*.cpp' '*.h' \
        2>/dev/null \
        | grep -vE '^(tests/|docs/|reports/|\.claude/|postmortems/|ideas/|proofs/|spec/)' \
        | grep -vE '/tests/' \
        | grep -vE '/(proofs|spec)/' \
        | grep -vE 'test_.*\.py$|_test\.py$|_test\.rs$|_test\.go$|\.test\.(ts|tsx|js|jsx)$|\.spec\.(ts|tsx|js|jsx)$|\.kani\.rs$' || true)

    # For Rust files, filter out those where all changes are in test blocks (#1585)
    PROD_CODE_FILES=""
    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        if [[ "$file" == *.rs ]] && rust_changes_only_in_tests "$file"; then
            # Skip - all changes are in test blocks
            continue
        fi
        PROD_CODE_FILES="$PROD_CODE_FILES$file"$'\n'
    done <<< "$INITIAL_PROD_FILES"
    PROD_CODE_FILES=$(echo "$PROD_CODE_FILES" | sed '/^$/d')

    if [[ -n "$PROD_CODE_FILES" ]]; then
        # Count lines added/deleted in production code
        # shellcheck disable=SC2086  # Word splitting is intentional here
        CODE_STATS=$(git diff --cached --stat -- $PROD_CODE_FILES 2>/dev/null | tail -1 || true)
        FILE_COUNT=$(echo "$PROD_CODE_FILES" | wc -l | tr -d ' ')

        # PROVER is hard-blocked (#933) - must not commit production code
        if [[ "$ROLE" == "PROVER" ]]; then
            echo "" >&2
            echo "❌ ERROR: PROVER role cannot commit production code changes" >&2
            echo "   Role boundaries: Prover verifies correctness, does not write production code." >&2
            echo "   Files ($FILE_COUNT):" >&2
            echo "$PROD_CODE_FILES" | head -5 | sed 's/^/      /' >&2
            if [[ "$FILE_COUNT" -gt 5 ]]; then
                echo "      ... and $((FILE_COUNT - 5)) more" >&2
            fi
            echo "   Stats: $CODE_STATS" >&2
            echo "" >&2
            echo "   Options:" >&2
            echo "     1. Switch to WORKER role to commit this code" >&2
            echo "     2. Remove production code changes from this commit" >&2
            echo "     3. For emergency: set ALLOW_NON_WORKER_CODE=1" >&2
            echo "" >&2
            exit 1
        fi

        # Other roles (RESEARCHER, MANAGER) get warning only
        echo "" >&2
        echo "⚠️  WARNING: Non-Worker role committing production code changes" >&2
        echo "   Role: $ROLE" >&2
        echo "   Files ($FILE_COUNT):" >&2
        echo "$PROD_CODE_FILES" | head -5 | sed 's/^/      /' >&2
        if [[ "$FILE_COUNT" -gt 5 ]]; then
            echo "      ... and $((FILE_COUNT - 5)) more" >&2
        fi
        echo "   Stats: $CODE_STATS" >&2
        echo "" >&2
        echo "   Role boundaries: Worker writes all production code." >&2
        echo "   If intentional, set ALLOW_NON_WORKER_CODE=1" >&2
        echo "" >&2
    fi
fi

# --- Validate commit structure ---

WARNINGS=""

# Shared auto-close patterns (GitHub accepts dotted repo names)
AUTO_CLOSE_KEYWORDS='(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)'
REPO_REF_PATTERN='[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+'
AUTO_CLOSE_REF_PATTERN="(${REPO_REF_PATTERN})?#[0-9]+"
# GitHub auto-closes on Fix/Close/Resolve keywords even when followed by prose.
# Detect any keyword+issue reference inline, but only ALLOW occurrences in the
# header (before ## Changes) to avoid accidental closes in prose.
AUTO_CLOSE_INLINE_PATTERN="(^|[^[:alnum:]_])${AUTO_CLOSE_KEYWORDS}:?[[:space:]]*${AUTO_CLOSE_REF_PATTERN}"

# ENFORCE: Don't use Fix/Close/Resolve in @ROLE directives (triggers GitHub auto-close)
# "@WORKER: Fix #319" in ## Next would close #319 when pushed
# Match keyword ANYWHERE in the directive line (GitHub matches anywhere in commit)
# Examples caught: "@WORKER: Fix #42", "@WORKER: Implement and fix #42", "@ALL: Fixed #42"
# Include @ALL (broadcast tag) and all tenses: fix/fixes/fixed, close/closes/closed, resolve/resolves/resolved
# Also match colon format "Fix: #42" and cross-repo "fix owner/repo#42"
if echo "$MSG" | grep -iE '@(WORKER|PROVER|RESEARCHER|MANAGER|ALL):' | grep -qiE "$AUTO_CLOSE_INLINE_PATTERN"; then
    echo "" >&2
    echo "❌ ERROR: Directive uses GitHub auto-close keyword" >&2
    echo "" >&2
    echo "   Found: @ROLE: Fix/Close/Resolve #N" >&2
    echo "   These words trigger GitHub auto-close when pushed!" >&2
    echo "" >&2
    echo "   Change to:" >&2
    echo "     @WORKER: Address #N" >&2
    echo "     @PROVER: Verify #N" >&2
    echo "     @MANAGER: Review closure of #N" >&2
    echo "" >&2
    exit 1
fi

# ENFORCE: Auto-close keywords must only appear as explicit directives in header.
# Avoid accidental closes from prose like "closed #N" in ## Changes bullets.
# Fix for #1864: Check each keyword+issue occurrence independently, not just the line.
AUTO_CLOSE_BAD_LINES=""
LINE_NUM=0
IN_HEADER=true
IN_CODE_BLOCK=false
while IFS= read -r line; do
    LINE_NUM=$((LINE_NUM + 1))
    if echo "$line" | grep -qE '^##[[:space:]]+Changes'; then
        IN_HEADER=false
    fi
    if echo "$line" | grep -qE '^[[:space:]]*```'; then
        if [[ "$IN_CODE_BLOCK" == "true" ]]; then
            IN_CODE_BLOCK=false
        else
            IN_CODE_BLOCK=true
        fi
        continue
    fi
    if [[ "$IN_CODE_BLOCK" == "true" ]]; then
        continue
    fi
    # Extract each auto-close keyword+issue match independently (#1864)
    # Pattern: keyword followed by optional colon/space and issue reference
    # "fix #228 labels" should flag as bad even if same line has valid "close #233"
    MATCHES=$(printf '%s\n' "$line" | grep -oiE "${AUTO_CLOSE_KEYWORDS}:?[[:space:]]*${AUTO_CLOSE_REF_PATTERN}[^,]*" || true)
    if [[ -n "$MATCHES" ]]; then
        while IFS= read -r match; do
            [[ -z "$match" ]] && continue
            # Check if this specific match has trailing non-whitespace text after issue number
            # Valid: "fix #228" "fix #228." "fix #228)" "fix #228," "fix #228;" (punctuation ok)
            # Invalid: "fix #228 labels" "fix #228 - desc" (has word/text after issue number)
            ISSUE_REF=$(echo "$match" | grep -oE "${AUTO_CLOSE_REF_PATTERN}")
            AFTER_ISSUE="${match#*${ISSUE_REF}}"
            # Strip leading whitespace to check what follows
            AFTER_TRIMMED=$(echo "$AFTER_ISSUE" | sed 's/^[[:space:]]*//')
            # If there's non-whitespace text after the issue number that isn't punctuation,
            # it's an accidental pattern like "fix #228 labels"
            # Allow: empty, comma, period, semicolon, colon, parens, brackets, exclamation, question
            # Note: ] must be first in character class for correct parsing
            if [[ -n "$AFTER_TRIMMED" ]] && ! [[ "$AFTER_TRIMMED" =~ ^[],.\;\:\)\!\?] ]]; then
                # Not in header = always bad, or has trailing text = bad
                AUTO_CLOSE_BAD_LINES+="${LINE_NUM}: ${match} (issue followed by text)"$'\n'
            elif [[ "$IN_HEADER" != "true" ]]; then
                # In body (after ## Changes) - any auto-close keyword is bad
                AUTO_CLOSE_BAD_LINES+="${LINE_NUM}: ${match}"$'\n'
            fi
        done <<< "$MATCHES"
    fi
done <<< "$MSG"

if [[ -n "$AUTO_CLOSE_BAD_LINES" ]]; then
    echo "" >&2
    echo "❌ ERROR: Auto-close keyword used outside explicit closure directive" >&2
    echo "" >&2
    echo "   GitHub auto-closes on Fix/Close/Resolve keywords anywhere in commit text." >&2
    echo "   Move closures to a dedicated line before ## Changes (e.g., 'Fixes #N')," >&2
    echo "   or rephrase prose to avoid auto-close keywords." >&2
    echo "" >&2
    echo "   Offending lines:" >&2
    echo "$AUTO_CLOSE_BAD_LINES" | head -5 | sed 's/^/   /' >&2
    if [[ $(echo "$AUTO_CLOSE_BAD_LINES" | wc -l | tr -d ' ') -gt 5 ]]; then
        echo "   ... (truncated)" >&2
    fi
    echo "" >&2
    exit 1
fi

# WARN: Detect conditional text that contains auto-close keywords (#942)
# GitHub parses "Fixes #N" anywhere in the commit message, even in conditional phrasing
# Examples: "If this satisfies #NNN, close with a Fixes #NNN commit"
# These get auto-closed prematurely even though the intent was conditional
# Match patterns: "if.*fixes", "close with.*fixes", "mark.*fixes", "should.*fix"
CONDITIONAL_PATTERN='(if[[:space:]]+(this|it|the|work|changes?).*|close[[:space:]]+(with|using|via).*|mark[[:space:]]+(as|it|this).*|should[[:space:]]+)(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)'
if echo "$MSG" | grep -qiE "$CONDITIONAL_PATTERN.*#[0-9]+"; then
    # Check if this looks like conditional text by examining the full line
    CONDITIONAL_LINE=$(echo "$MSG" | grep -iE "$CONDITIONAL_PATTERN.*#[0-9]+" | head -1)
    echo "" >&2
    echo "⚠️  WARNING: Conditional text contains auto-close keyword" >&2
    echo "" >&2
    echo "   Line: $CONDITIONAL_LINE" >&2
    echo "" >&2
    echo "   GitHub will auto-close any issue mentioned with Fix/Close/Resolve" >&2
    echo "   even when used in conditional phrasing like 'if this works, close with...'." >&2
    echo "" >&2
    echo "   Suggestion: Rephrase to avoid auto-close keywords:" >&2
    echo "     - 'If satisfactory, close issue #N' (no 'Fixes')" >&2
    echo "     - 'Ready for closure of #N' (no 'Close')" >&2
    echo "     - 'Address #N when complete'" >&2
    echo "" >&2
fi

# Limit auto-close directives to the commit header (before ## Changes).
HEADER_MSG=$(echo "$MSG" | awk 'BEGIN{in_header=1} /^##[[:space:]]+Changes/{in_header=0} { if (in_header) print }')

# ENFORCE: Don't direct work to closed issues
# Extract issue numbers from @ROLE directives and check their state
# Include @ALL and handle cross-repo format (owner/repo#N extracts just the number)
# Skip in full local mode - no gh API calls allowed
if [[ "$FULL_LOCAL_MODE" != "true" ]] && command -v gh &> /dev/null; then
    DIRECTIVE_ISSUES=$(echo "$MSG" | grep -oE '@(WORKER|PROVER|RESEARCHER|MANAGER|ALL):[^#]*#[0-9]+' | grep -oE '#[0-9]+' | tr -d '#' | sort -u)
    for DIRECTIVE_ISSUE in $DIRECTIVE_ISSUES; do
        ISSUE_STATE=$(gh_issue_field "$DIRECTIVE_ISSUE" "state")
        GH_EXIT=$?
        if [[ $GH_EXIT -eq 2 ]]; then
            gh_rate_limit_error "$DIRECTIVE_ISSUE"
        fi
        if [[ "$ISSUE_STATE" == "CLOSED" ]]; then
            echo "" >&2
            echo "❌ ERROR: Directive references closed issue #$DIRECTIVE_ISSUE" >&2
            echo "   Either reopen the issue or remove the directive." >&2
            echo "" >&2
            exit 1
        fi
    done
fi

# AUTO-FIX: GitHub only closes first issue in "Fixes #<number>, #<number>" format
# Transform "Fixes #<number>, #<number>, #<number>" -> "Fixes #<number>, fixes #<number>, fixes #<number>"
# Also handles: "Fixes owner/repo#<number>, #<number>" and "Fixes owner/repo#<number>, owner/repo#<number>"
# Evidence: commit d1ee8ef had multi-issue "Fixes" but only first was closed
if echo "$MSG" | grep -qiE "\\b${AUTO_CLOSE_KEYWORDS}[[:space:]]+${AUTO_CLOSE_REF_PATTERN},[[:space:]]*${AUTO_CLOSE_REF_PATTERN}"; then
    # Extract the keyword used and normalize to present tense
    KEYWORD=$(echo "$MSG" | grep -oiE "\\b${AUTO_CLOSE_KEYWORDS}[[:space:]]+${AUTO_CLOSE_REF_PATTERN}," | head -1 | grep -oiE "${AUTO_CLOSE_KEYWORDS}" | head -1)
    case "$(echo "$KEYWORD" | tr '[:upper:]' '[:lower:]')" in
        fix|fixes|fixed) REPEAT_KW="fixes" ;;
        close|closes|closed) REPEAT_KW="closes" ;;
        resolve|resolves|resolved) REPEAT_KW="resolves" ;;
        *) REPEAT_KW="fixes" ;;
    esac
    # Replace ", [owner/repo]#N" with ", fixes [owner/repo]#N" only in lines 1-5
    # The Fixes/Closes line is typically line 3-4 (after title + blank line)
    # This safely avoids code blocks and examples which appear later
    # Note: use [[:space:]]* not \s* for macOS sed compatibility
    # Match keyword anywhere in line (not just start) to handle role prefixes like [M]N: Fixes #<number>, #<number>
    # Loop to handle 3+ issues (e.g., "Fixes #<number>, #<number>, #<number>" -> "Fixes #<number>, fixes #<number>, fixes #<number>")
    PREV_MSG=""
    while [[ "$MSG" != "$PREV_MSG" ]]; do
        PREV_MSG="$MSG"
        MSG=$(echo "$MSG" | sed -E "1,5 s/(^|[[:space:]])(Fixes|Fix|Fixed|Closes|Close|Closed|Resolves|Resolve|Resolved)[[:space:]]+([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)?#([0-9]+),[[:space:]]*([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)?#([0-9]+)/\1\2 \3#\4, $REPEAT_KW \5#\6/gI")
    done
    echo "$MSG" > "$COMMIT_MSG_FILE"
    echo "ℹ️  Auto-fixed: Added '$REPEAT_KW' before each issue number (GitHub requires keyword per issue)" >&2
fi

# ENFORCE: Only USER can close issues in OTHER repos (cross-repo format)
# Cross-repo closes are highly disruptive - AIs should not close issues they don't own
# Format: "Fixes owner/repo#N" or "Closes org/project#N"
if echo "$MSG" | grep -qiE "\\b${AUTO_CLOSE_KEYWORDS}:?[[:space:]]*${REPO_REF_PATTERN}#[0-9]+"; then
    if [[ "$ROLE" != "USER" ]]; then
        # Extract the cross-repo reference for helpful message
        CROSS_REF=$(echo "$MSG" | grep -oiE "${REPO_REF_PATTERN}#[0-9]+" | head -1)
        echo "" >&2
        echo "❌ ERROR: Only USER can close issues in other repos" >&2
        echo "   You used 'Fixes $CROSS_REF' which closes an issue in another repo." >&2
        echo "   This is disruptive - only humans should close cross-repo issues." >&2
        echo "" >&2
        echo "   Instead, comment on that repo's issue:" >&2
        echo "     gh issue comment ${CROSS_REF#*#} --repo ${CROSS_REF%#*} --body \"Addressed in <this-repo> commit <hash>\"" >&2
        echo "" >&2
        exit 1
    fi
fi

# ENFORCE: Only MANAGER or USER can close issues with auto-close keywords
# WORKER/PROVER/RESEARCHER should use "Part of #N" to contribute without closing
# Match all GitHub auto-close keywords: fix/fixes/fixed, close/closes/closed, resolve/resolves/resolved
# GitHub supports: "fixes #42", "fixes: #42", "fixes#42"
# Match auto-close keywords anywhere in the header (title or pre-## Changes lines).
# Other auto-close keyword uses are blocked earlier to prevent accidental closes.
if echo "$HEADER_MSG" | grep -qiE "$AUTO_CLOSE_INLINE_PATTERN"; then
    if [[ "$ROLE" != "MANAGER" ]] && [[ "$ROLE" != "USER" ]]; then
        echo "" >&2
        echo "❌ ERROR: Only MANAGER role can close issues" >&2
        echo "   You used 'Fixes #N' which auto-closes the issue." >&2
        echo "   Use 'Part of #N' instead to link without closing." >&2
        echo "" >&2
        exit 1
    fi

    if [[ -z "$SKIP_ISSUE_CONFIRM" ]]; then
        HAS_NEXT_SECTION=false
        if echo "$MSG" | grep -q '^## Next'; then
            HAS_NEXT_SECTION=true
        fi

        # Extract closing refs from any location (title or body)
        CLOSING_REFS=$(echo "$HEADER_MSG" | grep -oiE "$AUTO_CLOSE_INLINE_PATTERN" | grep -oE "${AUTO_CLOSE_REF_PATTERN}" | sort -u | grep -v '^$')
        for ISSUE_REF in $CLOSING_REFS; do
            ISSUE_REPO=""
            ISSUE_NUM="${ISSUE_REF#\#}"
            DISPLAY_REF="#$ISSUE_NUM"

            if echo "$ISSUE_REF" | grep -q '/'; then
                ISSUE_REPO="${ISSUE_REF%#*}"
                ISSUE_NUM="${ISSUE_REF#*#}"
                DISPLAY_REF="${ISSUE_REPO}#${ISSUE_NUM}"
            fi

            echo ""
            echo "⚠️  This commit will CLOSE issue ${DISPLAY_REF}"
            echo ""

            TITLE="unknown"
            ISSUE_BODY=""
            ISSUE_LABELS=""
            RATE_LIMITED_WARNING=""
            # Handle local issues
            if is_local_issue "$ISSUE_NUM"; then
                TITLE=$(local_issue_field "$ISSUE_NUM" "title")
                ISSUE_BODY=$(local_issue_field "$ISSUE_NUM" "body")
                ISSUE_LABELS=$(local_issue_field "$ISSUE_NUM" "labels")
                [[ -z "$TITLE" ]] && TITLE="unknown"
            # Skip gh calls in full local mode for non-local issues
            elif [[ "$FULL_LOCAL_MODE" == "true" ]]; then
                echo "   (skipping GitHub validation - full local mode)" >&2
            elif command -v gh &> /dev/null; then
                if [[ -n "$ISSUE_REPO" ]]; then
                    TITLE=$(gh_issue_field "$ISSUE_NUM" "title" "$ISSUE_REPO")
                    [[ $? -eq 2 ]] && RATE_LIMITED_WARNING="yes"
                    ISSUE_BODY=$(gh_issue_field "$ISSUE_NUM" "body" "$ISSUE_REPO")
                    ISSUE_LABELS=$(gh_issue_field "$ISSUE_NUM" "labels" "$ISSUE_REPO")
                else
                    TITLE=$(gh_issue_field "$ISSUE_NUM" "title")
                    [[ $? -eq 2 ]] && RATE_LIMITED_WARNING="yes"
                    ISSUE_BODY=$(gh_issue_field "$ISSUE_NUM" "body")
                    ISSUE_LABELS=$(gh_issue_field "$ISSUE_NUM" "labels")
                fi
                [[ -z "$TITLE" ]] && TITLE="unknown"
            fi
            if [[ -n "$RATE_LIMITED_WARNING" ]]; then
                echo "   ⚠️  GitHub API rate limited - issue details may be incomplete"
            fi

            echo "   Title: $TITLE"

            UNCHECKED=0
            if [[ -n "$ISSUE_BODY" ]]; then
                UNCHECKED=$(echo "$ISSUE_BODY" | grep -cE '[-*]\s*\[ \]' 2>/dev/null | head -1 || echo "0")
                UNCHECKED=${UNCHECKED:-0}
            fi
            if [[ "$UNCHECKED" -gt 0 ]]; then
                echo "   ❌ Issue has $UNCHECKED unchecked items!" >&2
                echo "" >&2
                echo "❌ ERROR: Auto-close blocked due to unchecked checklist items" >&2
                echo "   Resolve unchecked items or use 'Part of ${DISPLAY_REF}' instead." >&2
                echo "" >&2
                exit 1
            fi

            # Check for acceptance criteria section (P1/P2 issues should have it per ai_template.md)
            # Only warn, don't block - acceptance criteria are recommended, not required
            if [[ -n "$ISSUE_BODY" ]]; then
                if ! echo "$ISSUE_BODY" | grep -qiE '##[[:space:]]+Acceptance[[:space:]]+Criteria'; then
                    # Skip warning for P3 issues (optional per ai_template.md)
                    if ! echo "$ISSUE_LABELS" | grep -q "P3"; then
                        echo "   ⚠️  Issue has no acceptance criteria section"
                        echo "      Consider adding ## Acceptance Criteria before closing"
                    fi
                fi
            fi

            if echo "$ISSUE_LABELS" | grep -q "P0"; then
                echo "   🚨 This is a P0 issue - requires postmortem AND fix"
            fi

            if [[ "$HAS_NEXT_SECTION" == "true" ]]; then
                echo "   ⚠️  Commit has '## Next' section with pending work"
            fi

            echo ""
            # Auto-confirm in non-interactive mode (looper/Claude Code)
            if [[ ! -t 0 ]]; then
                CONFIRM="y"
                echo "Auto-confirming (non-interactive)"
            else
                read -rp "Close issue ${DISPLAY_REF}? [y/N] " CONFIRM
            fi
            if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
                echo ""
                echo "Aborting commit."
                echo "Tip: Use 'Part of ${DISPLAY_REF}' for partial work."
                exit 1
            fi
        done
    fi
fi

# Check for ## Changes section (required)
if ! echo "$MSG" | grep -q '^## Changes'; then
    WARNINGS="${WARNINGS}⚠️  Missing '## Changes' section - explain WHY changes were made\n"
fi

# Check for ## Next section (required)
if ! echo "$MSG" | grep -q '^## Next'; then
    WARNINGS="${WARNINGS}⚠️  Missing '## Next' section - directive for next session\n"
fi

# Check for issue link (recommended for workers, not required)
# Match all issue link patterns including past tense, colon format, and cross-repo
# Also match local issue IDs: #L1, #L42, etc.
ISSUE_NUM=$(echo "$MSG" | grep -oiE "$AUTO_CLOSE_INLINE_PATTERN|(part of|re:|reopens)[[:space:]]+#(L?[0-9]+)" | grep -oE '#L?[0-9]+' | tr -d '#' | head -1)
if [[ -z "$ISSUE_NUM" ]] && [[ "$ROLE" == "WORKER" ]] && [[ "$IS_MAINTAIN" == "false" ]]; then
    WARNINGS="${WARNINGS}⚠️  No issue link (Fixes #N, Part of #N, etc.) - consider linking to an issue\n"
fi

# Validate issue exists if referenced
if [[ -n "$ISSUE_NUM" ]]; then
    if is_local_issue "$ISSUE_NUM"; then
        # Local issue - use local_issue_field
        ISSUE_STATE=$(local_issue_field "$ISSUE_NUM" "state")
        if [[ -z "$ISSUE_STATE" ]]; then
            echo "" >&2
            echo "❌ ERROR: Local issue #$ISSUE_NUM does not exist!" >&2
            echo "   Check .issues/${ISSUE_NUM}.md" >&2
            echo "" >&2
            exit 1
        fi
    elif [[ "$FULL_LOCAL_MODE" == "true" ]]; then
        # Full local mode - skip GitHub validation for non-local issues
        # Assume issue exists since we can't verify
        :
    elif command -v gh &> /dev/null; then
        # GitHub issue - use gh_issue_field
        ISSUE_STATE=$(gh_issue_field "$ISSUE_NUM" "state")
        GH_EXIT=$?
        if [[ $GH_EXIT -eq 2 ]]; then
            gh_rate_limit_error "$ISSUE_NUM"
        fi
        if [[ -z "$ISSUE_STATE" ]]; then
            echo "" >&2
            echo "❌ ERROR: Issue #$ISSUE_NUM does not exist!" >&2
            echo "" >&2
            exit 1
        fi
    fi
fi

# FEATURE FREEZE: Block commits on feature-labeled issues during freeze
# Bugs, docs, refactoring allowed; new features blocked
if [[ -f "FEATURE_FREEZE" ]] && [[ -n "$ISSUE_NUM" ]]; then
    FREEZE_LABELS=""
    if is_local_issue "$ISSUE_NUM"; then
        FREEZE_LABELS=$(local_issue_field "$ISSUE_NUM" "labels")
    elif [[ "$FULL_LOCAL_MODE" == "true" ]]; then
        # Full local mode - skip GitHub validation, allow commit
        :
    elif command -v gh &> /dev/null; then
        FREEZE_LABELS=$(gh_issue_field "$ISSUE_NUM" "labels")
    fi
    if echo "$FREEZE_LABELS" | grep -qx "feature"; then
        IS_P0=$(echo "$FREEZE_LABELS" | grep -qx "P0" && echo "true" || echo "false")
        if [[ "$IS_P0" == "false" ]]; then
            echo "" >&2
            echo "❌ ERROR: Feature freeze active! Cannot work on feature issues." >&2
            echo "   Issue #$ISSUE_NUM has 'feature' label." >&2
            echo "   During freeze: bugs, docs, refactoring only." >&2
            echo "   Remove FEATURE_FREEZE file to lift the freeze." >&2
            echo "" >&2
            exit 1
        fi
    fi
fi

# ENFORCE: Workers must claim issues with in-progress label (including worker-specific) before working on them
# This prevents multiple workers from picking up the same issue (#831)
# Fix for #1071: Accept generic in-progress as fallback when API caching returns stale data
# Fix for #1075: Only accept THIS worker's label or generic, NOT other workers' labels
# Fix for #1097: Accept do-audit and needs-review (issue already claimed/completed)
# Fix for #1453: Support both legacy (in-progress-W1) and new orthogonal (in-progress + W1) labels
# Local issues: Skip claim check (single-user local development)
# Full local mode: Skip claim check entirely (no gh API calls)
if [[ "$ROLE" == "WORKER" ]] && [[ -n "$ISSUE_NUM" ]]; then
    # Skip claim enforcement for local issues (single-user local development)
    if is_local_issue "$ISSUE_NUM"; then
        ISSUE_LABELS=$(local_issue_field "$ISSUE_NUM" "labels")
    elif [[ "$FULL_LOCAL_MODE" == "true" ]]; then
        # Full local mode - skip GitHub validation, skip claim enforcement
        ISSUE_LABELS="in-progress"  # Pretend claimed to skip enforcement
    elif command -v gh &> /dev/null; then
        ISSUE_LABELS=$(gh_issue_field "$ISSUE_NUM" "labels")
        GH_EXIT=$?
        if [[ $GH_EXIT -eq 2 ]]; then
            gh_rate_limit_error "$ISSUE_NUM"
        fi
    else
        # No gh and not local - skip validation
        ISSUE_LABELS=""
    fi
    CLAIM_LABEL="in-progress"
    CLAIMED="false"

    # First check for workflow state labels (do-audit, needs-review)
    # These indicate the issue was previously claimed and work is done/in-review
    if echo "$ISSUE_LABELS" | grep -qxE '(do-audit|needs-review)'; then
        CLAIMED="true"
    elif [[ -n "${AI_WORKER_ID:-}" ]]; then
        # Multi-worker mode: check ownership
        OWNERSHIP_LABEL="W${AI_WORKER_ID}"
        LEGACY_CLAIM_LABEL="in-progress-W${AI_WORKER_ID}"

        # New orthogonal pattern: in-progress + W<N> (both present)
        if echo "$ISSUE_LABELS" | grep -qx "in-progress" && echo "$ISSUE_LABELS" | grep -qx "$OWNERSHIP_LABEL"; then
            CLAIMED="true"
        # Legacy pattern: in-progress-W<N>
        elif echo "$ISSUE_LABELS" | grep -qx "$LEGACY_CLAIM_LABEL"; then
            CLAIMED="true"
        # Fallback: generic in-progress without ownership (API caching or single-worker)
        elif echo "$ISSUE_LABELS" | grep -qx "in-progress"; then
            CLAIMED="true"
            echo "⚠️  Note: Found in-progress but no ownership label $OWNERSHIP_LABEL (may be API caching)" >&2
        fi
    else
        # Single-worker mode: accept any in-progress variant
        if echo "$ISSUE_LABELS" | grep -Eq '^in-progress(-W[0-9]+)?$'; then
            CLAIMED="true"
        fi
    fi
    if [[ "$CLAIMED" != "true" ]]; then
        echo "" >&2
        echo "❌ ERROR: Issue #$ISSUE_NUM not claimed with in-progress label" >&2
        echo "" >&2
        echo "   Workers must claim issues before starting work:" >&2
        if [[ -n "${AI_WORKER_ID:-}" ]]; then
            echo "     gh issue edit $ISSUE_NUM --add-label in-progress --add-label W${AI_WORKER_ID}" >&2
        else
            echo "     gh issue edit $ISSUE_NUM --add-label $CLAIM_LABEL" >&2
        fi
        echo "" >&2
        echo "   This prevents duplicate work when multiple workers run." >&2
        echo "   After claiming, retry your commit." >&2
        echo "" >&2
        exit 1
    fi
fi

# Warn on large deletions (>200 lines) - may indicate accidental removal
DELETIONS=$(git diff --cached --numstat -- '*.rs' '*.py' '*.ts' '*.tsx' '*.js' '*.sh' '*.md' 2>/dev/null | awk '{sum+=$2} END {print sum+0}')
if [[ "$DELETIONS" -gt 200 ]]; then
    WARNINGS="${WARNINGS}⚠️  Large deletion: $DELETIONS lines removed. Verify this is intentional.\n"
fi

# Print warnings (but don't block commit)
if [[ -n "$WARNINGS" ]]; then
    echo "" >&2
    echo "━━━ Commit Structure Warnings ━━━" >&2
    echo -e "$WARNINGS" >&2
    echo "See .claude/rules/ai_template.md for commit message format" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "" >&2
fi

# --- Rewrite commit message ---

# SAFETY NET for already-prefixed commits: Final check before exit (#1269)
# For already-prefixed commits, the primary check should have blocked Fixes.
# This is a redundant check to catch any bypass of the primary enforcement.
if [[ "$ALREADY_PREFIXED" == "true" ]]; then
    if [[ "$BASE_PREFIX" != "U" ]] && [[ "$BASE_PREFIX" != "M" ]]; then
        if echo "$HEADER_MSG" | grep -qiE "$AUTO_CLOSE_INLINE_PATTERN"; then
            echo "" >&2
            echo "❌ ERROR: Safety check failed - non-Manager role attempting to close issues" >&2
            echo "   Role prefix: [$BASE_PREFIX]" >&2
            echo "   Message contains auto-close keyword (Fixes/Closes/Resolves #N)" >&2
            echo "" >&2
            echo "   This may indicate the primary enforcement was bypassed." >&2
            echo "   Use 'Part of #N' instead to link without closing." >&2
            echo "" >&2
            exit 1
        fi
    fi
    exit 0
fi

FIRST_LINE=$(echo "$MSG" | head -1)
BODY=$(echo "$MSG" | tail -n +2)

# Add role prefix if not present (handles [W]N, [W1]N, and [sat-W1]N formats)
# Pattern requires iteration number ([U]42, not [U]:) - prevents skipping when user types [U]: manually
if ! echo "$FIRST_LINE" | grep -qE '^\[[^]]*-?(U|W|P|R|M)[0-9]*\][0-9]+'; then
    if [[ "$IS_MAINTAIN" == "true" ]]; then
        # Keep [maintain] but add iteration
        FIRST_LINE="[$PREFIX]$NEXT_ITER: $FIRST_LINE"
    else
        FIRST_LINE="[$PREFIX]$NEXT_ITER: $FIRST_LINE"
    fi
fi

# SAFETY NET: Final check before writing - catch any bypasses (#1269)
# If we're assigning a non-MANAGER/USER prefix (W/P/R) and message has auto-close keywords,
# this is a violation that should have been caught earlier. Block it now.
# This catches cases where AI_ROLE env var was wrong/missing but prefix was assigned anyway.
if [[ "$BASE_PREFIX" != "U" ]] && [[ "$BASE_PREFIX" != "M" ]]; then
    if echo "$HEADER_MSG" | grep -qiE "$AUTO_CLOSE_INLINE_PATTERN"; then
        echo "" >&2
        echo "❌ ERROR: Safety check failed - non-Manager role attempting to close issues" >&2
        echo "   Role prefix: [$PREFIX]" >&2
        echo "   Message contains auto-close keyword (Fixes/Closes/Resolves #N)" >&2
        echo "" >&2
        echo "   This may indicate:" >&2
        echo "     - AI_ROLE env var was not set correctly" >&2
        echo "     - Hook enforcement was bypassed earlier" >&2
        echo "" >&2
        echo "   Use 'Part of #N' instead to link without closing." >&2
        echo "" >&2
        exit 1
    fi
fi

# 4. Type: from keywords in title (check [maintain] first)
TITLE=$(echo "$FIRST_LINE" | tr '[:upper:]' '[:lower:]')
if [[ "$IS_MAINTAIN" == "true" ]]; then
    TYPE="maintain"
elif [[ "$TITLE" == *fix* ]] || [[ "$TITLE" == *bug* ]] || [[ "$TITLE" == *patch* ]]; then
    TYPE="fix"
elif [[ "$TITLE" == *add* ]] || [[ "$TITLE" == *implement* ]] || [[ "$TITLE" == *feat* ]] || [[ "$TITLE" == *new* ]]; then
    TYPE="feat"
elif [[ "$TITLE" == *refactor* ]] || [[ "$TITLE" == *clean* ]] || [[ "$TITLE" == *simplif* ]]; then
    TYPE="refactor"
elif [[ "$TITLE" == *doc* ]] || [[ "$TITLE" == *readme* ]] || [[ "$TITLE" == *directive* ]]; then
    TYPE="docs"
elif [[ "$TITLE" == *test* ]]; then
    TYPE="test"
elif [[ "$TITLE" == *audit* ]] || [[ "$TITLE" == *review* ]]; then
    TYPE="audit"
else
    TYPE="chore"
fi

# --- Detect LLM model and agentic coder ---

# Model: check common env vars (Anthropic, OpenAI, Google)
MODEL="${ANTHROPIC_MODEL:-${OPENAI_MODEL:-${GOOGLE_MODEL:-}}}"

# Coder: detect from env vars (set by looper.py or coder tools)
# Format: "<type> v<version>" e.g., "claude-code v1.0.0", "codex v1.12", "gemini v0.1"
if [[ -n "$CLAUDE_CODE_VERSION" ]]; then
    CODER="claude-code v$CLAUDE_CODE_VERSION"
elif [[ -n "$CODEX_CLI_VERSION" ]]; then
    CODER="codex v$CODEX_CLI_VERSION"
elif [[ -n "$GEMINI_CLI_VERSION" ]]; then
    CODER="gemini v$GEMINI_CLI_VERSION"
elif [[ -n "$AI_CODER" ]]; then
    # Fallback: AI_CODER should be full string like "claude-code v1.0.0"
    CODER="$AI_CODER"
else
    CODER=""
fi

# AWS settings
AWS_REGION_VAL="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
AWS_PROFILE_VAL="${AWS_PROFILE:-}"

# Build new message with trailers
{
    echo "$FIRST_LINE"
    echo "$BODY"
    echo ""
    echo "---"
    echo "Role: $ROLE"
    echo "Type: $TYPE"
    echo "Iteration: $NEXT_ITER"
    [[ -n "$AI_WORKER_ID" ]] && echo "Worker-Id: $AI_WORKER_ID"
    [[ -n "$ISSUE_NUM" ]] && echo "Issue: $ISSUE_NUM"
    [[ -n "$AI_SESSION" ]] && echo "Session: $AI_SESSION"
    [[ -n "$MODEL" ]] && echo "Model: $MODEL"
    [[ -n "$CODER" ]] && echo "Coder: $CODER"
    [[ -n "$AWS_REGION_VAL" ]] && echo "AWS-Region: $AWS_REGION_VAL"
    [[ -n "$AWS_PROFILE_VAL" ]] && echo "AWS-Profile: $AWS_PROFILE_VAL"
    [[ -n "$AIT_VERSION" ]] && echo "AIT-Version: $AIT_VERSION"
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$COMMIT_MSG_FILE"

exit 0

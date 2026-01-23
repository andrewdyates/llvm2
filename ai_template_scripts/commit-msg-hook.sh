#!/bin/bash
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

COMMIT_MSG_FILE="$1"
MSG=$(cat "$COMMIT_MSG_FILE")

# Skip merge commits entirely (no validation needed)
if [[ "$MSG" == Merge* ]]; then
    exit 0
fi

# Check if commit already has our prefix (amend scenario)
# We still need to validate, but will skip rewriting
ALREADY_PREFIXED=false
if echo "$MSG" | grep -qE '^\[(U|W|P|R|M)\][0-9]+:'; then
    ALREADY_PREFIXED=true
fi

# --- Extract info automatically ---

# 1. Role: from env, existing prefix, or default to USER
# For already-prefixed commits, extract role from prefix to validate correctly
if [[ "$ALREADY_PREFIXED" == "true" ]]; then
    PREFIX_CHAR=$(echo "$MSG" | grep -oE '^\[(U|W|P|R|M)\]' | tr -d '[]')
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
case "$ROLE" in
    WORKER)     PREFIX="W" ;;
    PROVER)     PREFIX="P" ;;
    RESEARCHER) PREFIX="R" ;;
    MANAGER)    PREFIX="M" ;;
    *)          PREFIX="U" ;;  # USER or any other role
esac
LAST_ITER=$(git log --oneline -100 2>/dev/null | grep -oE "\[${PREFIX}\]#?[0-9]+" | sed -E "s/\[${PREFIX}\]#?//" | sort -rn | head -1)
NEXT_ITER=$((${LAST_ITER:-0} + 1))

# 3. Check for [maintain] tag
IS_MAINTAIN=false
if echo "$MSG" | head -1 | grep -qi '\[maintain\]'; then
    IS_MAINTAIN=true
fi

# --- Validate commit structure ---

WARNINGS=""

# ENFORCE: Don't use Fix/Close/Resolve in @ROLE directives (triggers GitHub auto-close)
# "@WORKER: Fix #319" in ## Next would close #319 when pushed
# Match keyword ANYWHERE in the directive line (GitHub matches anywhere in commit)
# Examples caught: "@WORKER: Fix #42", "@WORKER: Implement and fix #42", "@ALL: Fixed #42"
# Include @ALL (broadcast tag) and all tenses: fix/fixes/fixed, close/closes/closed, resolve/resolves/resolved
# Also match colon format "Fix: #42" and cross-repo "fix owner/repo#42"
if echo "$MSG" | grep -iE '@(WORKER|PROVER|RESEARCHER|MANAGER|ALL):' | grep -qiE '\b(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved):?\s*([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)?#[0-9]+'; then
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

# ENFORCE: Don't direct work to closed issues
# Extract issue numbers from @ROLE directives and check their state
# Include @ALL and handle cross-repo format (owner/repo#N extracts just the number)
if command -v gh &> /dev/null; then
    DIRECTIVE_ISSUES=$(echo "$MSG" | grep -oE '@(WORKER|PROVER|RESEARCHER|MANAGER|ALL):[^#]*#[0-9]+' | grep -oE '#[0-9]+' | tr -d '#' | sort -u)
    for DIRECTIVE_ISSUE in $DIRECTIVE_ISSUES; do
        ISSUE_STATE=$(gh issue view "$DIRECTIVE_ISSUE" --json state -q '.state' 2>/dev/null)
        if [[ "$ISSUE_STATE" == "CLOSED" ]]; then
            echo "" >&2
            echo "❌ ERROR: Directive references closed issue #$DIRECTIVE_ISSUE" >&2
            echo "   Either reopen the issue or remove the directive." >&2
            echo "" >&2
            exit 1
        fi
    done
fi

# AUTO-FIX: GitHub only closes first issue in "Fixes #1, #2" format
# Transform "Fixes #1, #2, #3" -> "Fixes #1, fixes #2, fixes #3"
# Also handles: "Fixes owner/repo#1, #2" and "Fixes owner/repo#1, owner/repo#2"
# Evidence: commit d1ee8ef had "Fixes #260, #257" but only #260 was closed
if echo "$MSG" | grep -qiE '\b(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)[[:space:]]+([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)?#[0-9]+,[[:space:]]*([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)?#[0-9]+'; then
    # Extract the keyword used and normalize to present tense
    KEYWORD=$(echo "$MSG" | grep -oiE '\b(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)[[:space:]]+([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)?#[0-9]+,' | head -1 | grep -oiE '^[a-z]+')
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
    MSG=$(echo "$MSG" | sed -E "1,5 {/^(Fixes|Fix|Fixed|Closes|Close|Closed|Resolves|Resolve|Resolved)[[:space:]]/I s/,[[:space:]]*([a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+)?#([0-9]+)/, $REPEAT_KW \1#\2/g;}")
    echo "$MSG" > "$COMMIT_MSG_FILE"
    echo "ℹ️  Auto-fixed: Added '$REPEAT_KW' before each issue number (GitHub requires keyword per issue)" >&2
fi

# ENFORCE: Only USER can close issues in OTHER repos (cross-repo format)
# Cross-repo closes are highly disruptive - AIs should not close issues they don't own
# Format: "Fixes owner/repo#N" or "Closes org/project#N"
if echo "$MSG" | grep -qiE '\b(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved):?\s*[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+#[0-9]+'; then
    if [[ "$ROLE" != "USER" ]]; then
        # Extract the cross-repo reference for helpful message
        CROSS_REF=$(echo "$MSG" | grep -oiE '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+#[0-9]+' | head -1)
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

# ENFORCE: Only MANAGER or USER can close issues in THIS repo with "Fixes #N"
# WORKER/PROVER/RESEARCHER should use "Part of #N" to contribute without closing
# Match all GitHub auto-close keywords: fix/fixes/fixed, close/closes/closed, resolve/resolves/resolved
# GitHub supports: "fixes #42", "fixes: #42", "fixes#42"
if echo "$MSG" | grep -qiE '\b(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved):?\s*#[0-9]+'; then
    if [[ "$ROLE" != "MANAGER" ]] && [[ "$ROLE" != "USER" ]]; then
        echo "" >&2
        echo "❌ ERROR: Only MANAGER role can close issues" >&2
        echo "   You used 'Fixes #N' which auto-closes the issue." >&2
        echo "   Use 'Part of #N' instead to link without closing." >&2
        echo "" >&2
        exit 1
    fi

    # ENFORCE: Block closure if issue has unchecked checkboxes
    # - [ ] = unchecked (blocks closure)
    # - [x] = checked (OK)
    # - [~] = refused/won't-do (OK, Manager explicitly declined)
    CLOSING_ISSUE=$(echo "$MSG" | grep -oiE '\b(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved):?\s*([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)?#[0-9]+' | grep -oE '#[0-9]+' | tr -d '#' | head -1)
    if command -v gh &> /dev/null && [[ -n "$CLOSING_ISSUE" ]]; then
        ISSUE_BODY=$(gh issue view "$CLOSING_ISSUE" --json body -q '.body' 2>/dev/null || echo "")
        # Count unchecked boxes (- [ ] or * [ ]) but not refused boxes ([~])
        # Match anywhere in line to catch nested lists and inline checkboxes
        # Use head -1 to ensure single number (grep -c can output multiple lines on some inputs)
        UNCHECKED=$(echo "$ISSUE_BODY" | grep -cE '[-*]\s*\[ \]' 2>/dev/null | head -1 || echo "0")
        UNCHECKED=${UNCHECKED:-0}
        if [[ "$UNCHECKED" -gt 0 ]]; then
            echo "" >&2
            echo "❌ ERROR: Cannot close #$CLOSING_ISSUE - $UNCHECKED unchecked checkbox(es) remain" >&2
            echo "" >&2
            echo "   Options:" >&2
            echo "   1. Complete the items and check them off [x]" >&2
            echo "   2. Mark as refused [~] with explanation (Manager only)" >&2
            echo "   3. Convert unchecked items to new issues" >&2
            echo "" >&2
            exit 1
        fi
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
ISSUE_NUM=$(echo "$MSG" | grep -oiE '(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved):?\s*([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)?#[0-9]+|(part of|re:|reopens)\s*#[0-9]+' | grep -oE '#[0-9]+' | tr -d '#' | head -1)
if [[ -z "$ISSUE_NUM" ]] && [[ "$ROLE" == "WORKER" ]] && [[ "$IS_MAINTAIN" == "false" ]]; then
    WARNINGS="${WARNINGS}⚠️  No issue link (Fixes #N, Part of #N, etc.) - consider linking to an issue\n"
fi

# Validate issue exists if referenced
if [[ -n "$ISSUE_NUM" ]] && command -v gh &> /dev/null; then
    ISSUE_STATE=$(gh issue view "$ISSUE_NUM" --json state -q '.state' 2>/dev/null)
    if [[ -z "$ISSUE_STATE" ]]; then
        echo "❌ ERROR: Issue #$ISSUE_NUM does not exist!" >&2
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

# Skip rewriting for already-prefixed commits (amend scenario)
# Validation already ran above, just preserve the message
if [[ "$ALREADY_PREFIXED" == "true" ]]; then
    exit 0
fi

FIRST_LINE=$(echo "$MSG" | head -1)
BODY=$(echo "$MSG" | tail -n +2)

# Add role prefix if not present
if ! echo "$FIRST_LINE" | grep -qE '^\[(U|W|P|R|M)\]'; then
    if [[ "$IS_MAINTAIN" == "true" ]]; then
        # Keep [maintain] but add iteration
        FIRST_LINE="[$PREFIX]$NEXT_ITER: $FIRST_LINE"
    else
        FIRST_LINE="[$PREFIX]$NEXT_ITER: $FIRST_LINE"
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

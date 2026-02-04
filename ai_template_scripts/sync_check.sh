#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# sync_check.sh - Check template drift across repos
#
# Reads .ai_template_version from repos and compares to current ai_template HEAD.
# Reports which repos are behind and by how many commits.
#
# Usage:
#   ./ai_template_scripts/sync_check.sh              # Check all sibling repos
#   ./ai_template_scripts/sync_check.sh /path/to/repos  # Check repos in directory
#   ./ai_template_scripts/sync_check.sh repo1 repo2     # Check specific repos
#   ./ai_template_scripts/sync_check.sh --files         # Also check file-level drift
#   ./ai_template_scripts/sync_check.sh --exit-code     # Exit non-zero if drift found

set -euo pipefail

# Colors - disable if not a TTY or NO_COLOR is set
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi
SEVERE_THRESHOLD=100
EXIT_STATUS=0

# Cache for archived repos (avoid repeated API calls)
ARCHIVED_CACHE_DIR="${HOME}/.cache/ai_template"
ARCHIVED_CACHE_FILE="${ARCHIVED_CACHE_DIR}/archived_repos.txt"
ARCHIVED_CACHE_AGE=3600 # 1 hour

# Temp file for batch archive check results (bash 3.x compatible - no assoc arrays)
ARCHIVED_BATCH_RESULTS=""

# Version function
version() {
    local git_hash
    git_hash=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "sync_check.sh ${git_hash} (${date})"
    exit 0
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [PATH|REPOS...]"
    echo ""
    echo "Check template drift across repos by comparing .ai_template_version."
    echo ""
    echo "Arguments:"
    echo "  PATH        Directory containing repos to check"
    echo "  REPOS...    Specific repo paths to check"
    echo "              (Default: all sibling repos of ai_template)"
    echo ""
    echo "Options:"
    echo "  --files      Also check file-level drift"
    echo "  --exit-code  Exit non-zero if drift found"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Check all sibling repos"
    echo "  $0 /path/to/repos            # Check repos in directory"
    echo "  $0 repo1 repo2               # Check specific repos"
    echo "  $0 --files                   # Also check file-level drift"
    echo "  $0 --exit-code               # Exit non-zero if drift found"
    exit 0
}

# Batch check which repos are archived using GraphQL (single API call for all repos)
# Sets ARCHIVED_BATCH_RESULTS (newline-separated list of archived repo names)
batch_check_archived() {
    local repos=("$@")
    [[ ${#repos[@]} -eq 0 ]] && return 0

    ARCHIVED_BATCH_RESULTS=""

    # Check local file cache first (repos known to be archived)
    local cached_archived=""
    if [[ -f "$ARCHIVED_CACHE_FILE" ]]; then
        local cache_age=$(($(date +%s) - $(stat -f %m "$ARCHIVED_CACHE_FILE" 2>/dev/null || echo 0)))
        if [[ $cache_age -lt $ARCHIVED_CACHE_AGE ]]; then
            cached_archived=$(cat "$ARCHIVED_CACHE_FILE")
            ARCHIVED_BATCH_RESULTS="$cached_archived"
        fi
    fi

    # Filter out repos already known to be archived
    local repos_to_check=()
    for repo in "${repos[@]}"; do
        if ! echo "$cached_archived" | grep -qx "$repo" 2>/dev/null; then
            repos_to_check+=("$repo")
        fi
    done

    [[ ${#repos_to_check[@]} -eq 0 ]] && return 0

    # Build GraphQL query with aliases for each repo
    # Example: query { r0: repository(owner:"dropbox-ai-prototypes", name:"repo1") { isArchived } ... }
    # Validate repo names to prevent GraphQL injection (special chars like " or })
    local query="query {"
    local i=0
    local valid_repos=()
    for repo in "${repos_to_check[@]}"; do
        # Validate repo name: only alphanumeric, underscore, hyphen, and dot allowed
        if [[ ! "$repo" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
            echo "Warning: skipping invalid repo name: $repo" >&2
            continue
        fi
        query+=" r${i}: repository(owner: \"dropbox-ai-prototypes\", name: \"$repo\") { isArchived }"
        valid_repos+=("$repo")
        i=$((i + 1))
    done
    query+=" }"

    # Early return if no valid repos
    [[ ${#valid_repos[@]} -eq 0 ]] && return 0

    # Execute single GraphQL query
    local result stderr_file
    stderr_file=$(mktemp)
    if result=$(gh api graphql -f query="$query" 2>"$stderr_file"); then
        rm -f "$stderr_file"
        # Parse results and populate ARCHIVED_BATCH_RESULTS
        # jq: extract each r<N> result
        i=0
        for repo in "${valid_repos[@]}"; do
            local is_archived
            is_archived=$(echo "$result" | jq -r ".data.r${i}.isArchived // false")
            if [[ "$is_archived" == "true" ]]; then
                if [[ -n "$ARCHIVED_BATCH_RESULTS" ]]; then
                    ARCHIVED_BATCH_RESULTS="${ARCHIVED_BATCH_RESULTS}"$'\n'"${repo}"
                else
                    ARCHIVED_BATCH_RESULTS="$repo"
                fi
                # Add to file cache
                mkdir -p "$ARCHIVED_CACHE_DIR"
                echo "$repo" >>"$ARCHIVED_CACHE_FILE"
            fi
            i=$((i + 1))
        done
        return 0
    fi

    # GraphQL failed - check if rate limited
    if grep -qiE 'rate.?limit' "$stderr_file" 2>/dev/null; then
        rm -f "$stderr_file"
        # REST fallback: batch via gh api with paginate (slower but works)
        echo "Warning: GraphQL rate-limited, using REST fallback for archive check" >&2
        for repo in "${repos_to_check[@]}"; do
            local api_output is_archived api_message
            # Use || true to prevent pipefail exit on 404 (repo not in dropbox-ai-prototypes)
            api_output=$(gh api "repos/dropbox-ai-prototypes/$repo" 2>/dev/null || true)
            if [[ -z "$api_output" ]]; then
                # Empty response - skip silently
                continue
            fi
            # Check for API error response (404 returns JSON with message field)
            api_message=$(echo "$api_output" | jq -r '.message // empty' 2>/dev/null)
            if [[ "$api_message" == "Not Found" ]]; then
                # Repo doesn't exist in dropbox-ai-prototypes - skip silently
                continue
            fi
            is_archived=$(echo "$api_output" | jq -r '.archived // false')
            if [[ "$is_archived" == "true" ]]; then
                if [[ -n "$ARCHIVED_BATCH_RESULTS" ]]; then
                    ARCHIVED_BATCH_RESULTS="${ARCHIVED_BATCH_RESULTS}"$'\n'"${repo}"
                else
                    ARCHIVED_BATCH_RESULTS="$repo"
                fi
                mkdir -p "$ARCHIVED_CACHE_DIR"
                echo "$repo" >>"$ARCHIVED_CACHE_FILE"
            fi
        done
    fi
    rm -f "$stderr_file"
}

# Check if repo is archived (uses batch results from ARCHIVED_BATCH_RESULTS)
is_repo_archived() {
    local repo_name="$1"
    echo "$ARCHIVED_BATCH_RESULTS" | grep -qx "$repo_name" 2>/dev/null
}

# Find ai_template root - works whether run from ai_template or synced copy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if we're in ai_template or a synced copy
find_ai_template() {
    local candidate="$1"
    local remote
    remote=$(cd "$candidate" && git remote get-url origin 2>/dev/null || echo "")
    [[ "$remote" =~ ai_template(\.git)?$ ]] && echo "$candidate" && return 0
    return 1
}

# Try script location first
if ! AI_TEMPLATE_ROOT=$(find_ai_template "$AI_TEMPLATE_ROOT"); then
    # Script is in a synced repo - look for ai_template as sibling or in common locations
    PARENT_DIR="$(dirname "$AI_TEMPLATE_ROOT")"
    if [[ -d "$PARENT_DIR/ai_template" ]] && AI_TEMPLATE_ROOT=$(find_ai_template "$PARENT_DIR/ai_template"); then
        : # AI_TEMPLATE_ROOT set by find_ai_template
    elif [[ -d ~/ai_template ]] && AI_TEMPLATE_ROOT=$(find_ai_template ~/ai_template); then
        : # AI_TEMPLATE_ROOT set by find_ai_template
    else
        echo "ERROR: Cannot find ai_template repo." >&2
        echo "  Script location: $SCRIPT_DIR" >&2
        echo "  Checked:" >&2
        echo "    - $(dirname "$SCRIPT_DIR") (script parent)" >&2
        echo "    - $PARENT_DIR/ai_template (sibling)" >&2
        echo "    - ~/ai_template (home)" >&2
        echo "" >&2
        echo "To fix, either:" >&2
        echo "  1. Clone ai_template as a sibling: cd .. && git clone https://github.com/dropbox-ai-prototypes/ai_template" >&2
        echo "  2. Or run directly from ai_template: ~/ai_template/ai_template_scripts/sync_check.sh" >&2
        exit 1
    fi
fi

cd "$AI_TEMPLATE_ROOT"

# Load exclude list (repos to skip - external repos, forks, etc.)
EXCLUDE_FILE="$AI_TEMPLATE_ROOT/.sync_exclude"
EXCLUDED_REPOS=()
if [[ -f "$EXCLUDE_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        EXCLUDED_REPOS+=("$line")
    done <"$EXCLUDE_FILE"
fi

# Check if repo should be excluded
is_excluded() {
    local repo_name="$1"
    # Use ${arr[@]+"${arr[@]}"} pattern for bash 3.x compatibility with set -u
    for excluded in ${EXCLUDED_REPOS[@]+"${EXCLUDED_REPOS[@]}"}; do
        [[ "$repo_name" == "$excluded" ]] && return 0
    done
    return 1
}

# Collect repos to check
REPOS=()
SKIPPED=()
ARGS=()
CHECK_FILES_ARG=false
EXIT_CODE=false

# Parse args first (before any output or network calls)
for arg in "$@"; do
    case "$arg" in
    -h | --help) usage ;;
    --version) version ;;
    --files) CHECK_FILES_ARG=true ;;
    --exit-code) EXIT_CODE=true ;;
    -*)
        echo "ERROR: Unknown option: $arg"
        exit 1
        ;;
    *) ARGS+=("$arg") ;;
    esac
done

set_exit_status() {
    local status="$1"
    if [[ "$EXIT_CODE" == "true" ]] && ((EXIT_STATUS < status)); then
        EXIT_STATUS=$status
    fi
}

# Fetch latest from remote to ensure accurate version comparison
# Without this, a stale local clone reports wrong "current version" (#1124)
git fetch --quiet origin main 2>/dev/null || true

# Get ai_template current version from remote (not local HEAD which may be stale)
CURRENT_VERSION=$(git rev-parse origin/main)
CURRENT_SHORT=$(git rev-parse --short origin/main)

echo "ai_template current version: $CURRENT_SHORT"
echo ""

# Check if repo is an dropbox-ai-prototypes repo (by git remote)
is_dropbox-ai-prototypes_repo() {
    local dir="$1"
    local remote_url
    remote_url=$(git -C "$dir" remote get-url origin 2>/dev/null || echo "")
    [[ "$remote_url" =~ dropbox-ai-prototypes ]]
}

if [[ ${#ARGS[@]} -eq 0 ]]; then
    # Default: check sibling directories that are dropbox-ai-prototypes repos
    PARENT_DIR="$(dirname "$AI_TEMPLATE_ROOT")"
    for dir in "$PARENT_DIR"/*/; do
        repo_name="$(basename "$dir")"
        [[ -d "$dir/.git" ]] && [[ "$repo_name" != "ai_template" ]] && is_dropbox-ai-prototypes_repo "$dir" && ! is_excluded "$repo_name" && REPOS+=("$dir")
    done
elif [[ ${#ARGS[@]} -eq 1 && -d "${ARGS[0]}" && ! -d "${ARGS[0]}/.git" ]]; then
    # Single directory argument that's not a repo - scan it (only dropbox-ai-prototypes repos)
    for dir in "${ARGS[0]}"/*/; do
        repo_name="$(basename "$dir")"
        [[ -d "$dir/.git" ]] && is_dropbox-ai-prototypes_repo "$dir" && ! is_excluded "$repo_name" && REPOS+=("$dir")
    done
else
    # Explicit repo list
    for arg in "${ARGS[@]}"; do
        if [[ -d "$arg/.git" ]]; then
            REPOS+=("$arg")
        else
            SKIPPED+=("$arg")
        fi
    done
fi

if [[ ${#REPOS[@]} -eq 0 ]]; then
    echo "No repos found to check."
    if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo "Skipped non-repo paths:"
        for skipped in "${SKIPPED[@]}"; do
            echo "  - $skipped"
        done
    fi
    if [[ "$EXIT_CODE" == "true" && ${#ARGS[@]} -gt 0 ]]; then
        exit 1
    fi
    exit 0
fi

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "Warning: skipped non-repo paths:"
    for skipped in "${SKIPPED[@]}"; do
        echo "  - $skipped"
    done
    echo ""
fi

# Batch check archived status for all repos (single API call instead of N calls)
REPO_NAMES=()
for repo in "${REPOS[@]}"; do
    REPO_NAMES+=("$(basename "$repo")")
done
batch_check_archived "${REPO_NAMES[@]}"

# Check each repo
UP_TO_DATE=0
BEHIND=0
UNTRACKED=0
AHEAD=0
DIVERGED=0
UNKNOWN=0
TOTAL=${#REPOS[@]}

printf "%-30s %-10s %-10s %s\n" "REPO" "VERSION" "STATUS" "DETAILS"
printf "%-30s %-10s %-10s %s\n" "----" "-------" "------" "-------"

for repo in "${REPOS[@]}"; do
    repo_name=$(basename "$repo")
    version_file="$repo/.ai_template_version"

    # Skip archived repos
    if is_repo_archived "$repo_name"; then
        continue
    fi

    if [[ ! -f "$version_file" ]]; then
        printf "%-30s %-10s ${YELLOW}%-10s${NC} %s\n" "$repo_name" "-" "untracked" "no .ai_template_version"
        ((UNTRACKED++))
        set_exit_status 1
        continue
    fi

    # Read line 1 and extract first whitespace-delimited token
    # Handles annotated lines like "abc1234 (2026-01-30)"
    repo_version_raw=$(head -1 "$version_file")
    repo_version=$(echo "$repo_version_raw" | awk '{print $1}' | tr -d '[:space:]')
    if [[ -z "$repo_version" ]]; then
        printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "-" "unknown" "empty .ai_template_version"
        ((UNKNOWN++))
        set_exit_status 1
        continue
    fi

    if [[ "$repo_version" == "$CURRENT_VERSION" ]]; then
        printf "%-30s %-10s ${GREEN}%-10s${NC}\n" "$repo_name" "${repo_version:0:7}" "current"
        ((UP_TO_DATE++))
    else
        if ! git cat-file -e "$repo_version^{commit}" 2>/dev/null; then
            # Commit not found - try timestamp fallback from line 2
            repo_timestamp=$(sed -n '2p' "$version_file" | tr -d '[:space:]')
            if [[ -n "$repo_timestamp" ]]; then
                # Use Python for portable ISO-8601 parsing
                # Pass timestamp via stdin to avoid shell injection risks
                days_old=$(echo "$repo_timestamp" | python3 -c "
import sys
from datetime import datetime, timezone
try:
    ts_str = sys.stdin.read().strip()
    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    print(int((now - ts).days))
except (ValueError, OSError):
    print(-1)
" 2>/dev/null)
                # Handle empty days_old (python3 not found or other failure)
                if [[ -n "$days_old" && "$days_old" -ge 0 ]]; then
                    printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "unknown-origin" "last sync ${repo_timestamp%T*}, ${days_old}d ago"
                else
                    printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "unknown-origin" "(invalid timestamp)"
                fi
            else
                printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "unknown-origin" "(no timestamp)"
            fi
            set_exit_status 1
            ((UNKNOWN++))
            continue
        fi

        if git merge-base --is-ancestor "$repo_version" "$CURRENT_VERSION"; then
            commits_behind=$(git rev-list --count "$repo_version..$CURRENT_VERSION")

            if [[ "$commits_behind" -gt "$SEVERE_THRESHOLD" ]]; then
                printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "STALE" "$commits_behind commits behind"
                set_exit_status 2
            elif [[ "$commits_behind" -gt 10 ]]; then
                printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "STALE" "$commits_behind commits behind"
                set_exit_status 1
            else
                printf "%-30s %-10s ${YELLOW}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "behind" "$commits_behind commits behind"
                set_exit_status 1
            fi
            ((BEHIND++))
        elif git merge-base --is-ancestor "$CURRENT_VERSION" "$repo_version"; then
            commits_ahead=$(git rev-list --count "$CURRENT_VERSION..$repo_version")
            printf "%-30s %-10s ${YELLOW}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "ahead" "$commits_ahead commits ahead"
            set_exit_status 1
            ((AHEAD++))
        else
            printf "%-30s %-10s ${RED}%-10s${NC} %s\n" "$repo_name" "${repo_version:0:7}" "diverged" "no shared history"
            set_exit_status 1
            ((DIVERGED++))
        fi
    fi
done

echo ""
echo "Summary: $UP_TO_DATE current, $BEHIND behind, $UNTRACKED untracked, $AHEAD ahead, $DIVERGED diverged, $UNKNOWN unknown (of $TOTAL repos)"

NONCURRENT=$((BEHIND + UNTRACKED + AHEAD + DIVERGED + UNKNOWN))
if [[ $NONCURRENT -gt 0 ]]; then
    echo ""
    echo "To sync a repo:"
    echo "  ./ai_template_scripts/sync_repo.sh /path/to/repo"
fi

# File-level drift detection (optional, slower)
if [[ "${CHECK_FILES:-}" == "1" || "${CHECK_FILES:-}" == "true" ]] || [[ "$CHECK_FILES_ARG" == "true" ]]; then
    echo ""
    echo "=== File-level drift check ==="

    MANIFEST="$AI_TEMPLATE_ROOT/.sync_manifest"
    if [[ ! -f "$MANIFEST" ]]; then
        echo "Warning: .sync_manifest not found, skipping file check"
        set_exit_status 1
    else
        # Collect files from manifest (skip directories, exclusions, comments)
        declare -a SYNC_FILES
        while IFS= read -r line || [[ -n "$line" ]]; do
            line="${line%%#*}"
            line="${line%"${line##*[![:space:]]}"}"
            line="${line#"${line%%[![:space:]]*}"}"
            [[ -z "$line" ]] && continue
            [[ "$line" == !* ]] && continue
            [[ "$line" == */ ]] && continue   # Skip directories
            [[ "$line" == *\** ]] && continue # Skip globs (too complex)
            SYNC_FILES+=("$line")
        done <"$MANIFEST"

        DRIFT_REPOS=()
        for repo in "${REPOS[@]}"; do
            repo_name=$(basename "$repo")
            drifted=()

            for file in "${SYNC_FILES[@]}"; do
                src="$AI_TEMPLATE_ROOT/$file"
                dst="$repo/$file"

                [[ ! -f "$src" ]] && continue
                [[ ! -f "$dst" ]] && {
                    drifted+=("$file (missing)")
                    continue
                }

                if ! diff -q "$src" "$dst" >/dev/null 2>&1; then
                    drifted+=("$file")
                fi
            done

            if [[ ${#drifted[@]} -gt 0 ]]; then
                echo ""
                echo -e "${YELLOW}$repo_name${NC} has ${#drifted[@]} drifted files:"
                for f in "${drifted[@]}"; do
                    echo "  - $f"
                done
                DRIFT_REPOS+=("$repo_name")
            fi
        done

        if [[ ${#DRIFT_REPOS[@]} -eq 0 ]]; then
            echo "No file-level drift detected."
        else
            echo ""
            echo "Found drift in ${#DRIFT_REPOS[@]} repos. Run sync_repo.sh to fix."
            set_exit_status 1
        fi

        # Optional feature drift detection (#1076)
        echo ""
        echo "=== Optional feature drift check ==="
        OPTIONAL_DIR="$AI_TEMPLATE_ROOT/ai_template_scripts/optional"
        OPTIONAL_DRIFT_REPOS=()

        for repo in "${REPOS[@]}"; do
            repo_name=$(basename "$repo")
            features_file="$repo/.ai_template_features"
            drifted=()

            # Skip repos without .ai_template_features
            [[ ! -f "$features_file" ]] && continue

            # Parse enabled features (reset array each iteration)
            enabled_features=()
            while IFS= read -r line || [[ -n "$line" ]]; do
                line="${line%%#*}"
                line="${line%"${line##*[![:space:]]}"}"
                line="${line#"${line%%[![:space:]]*}"}"
                [[ -z "$line" ]] && continue
                [[ "$line" == */* || "$line" == *\\* || "$line" == *..* ]] && continue
                enabled_features+=("$line")
            done <"$features_file"

            # Check each enabled feature for drift
            for feature in "${enabled_features[@]}"; do
                src="$OPTIONAL_DIR/$feature"
                dst="$repo/ai_template_scripts/optional/$feature"

                if [[ ! -d "$src" ]]; then
                    drifted+=("optional/$feature (unknown feature)")
                    continue
                fi

                if [[ ! -d "$dst" ]]; then
                    drifted+=("optional/$feature (missing)")
                    continue
                fi

                # Check if directories differ
                if ! diff -rq "$src" "$dst" >/dev/null 2>&1; then
                    drifted+=("optional/$feature")
                fi
            done

            if [[ ${#drifted[@]} -gt 0 ]]; then
                echo ""
                echo -e "${YELLOW}$repo_name${NC} has ${#drifted[@]} drifted optional features:"
                for f in "${drifted[@]}"; do
                    echo "  - $f"
                done
                OPTIONAL_DRIFT_REPOS+=("$repo_name")
            fi
        done

        if [[ ${#OPTIONAL_DRIFT_REPOS[@]} -eq 0 ]]; then
            echo "No optional feature drift detected."
        else
            echo ""
            echo "Found optional feature drift in ${#OPTIONAL_DRIFT_REPOS[@]} repos."
            set_exit_status 1
        fi

        # Pre-commit hook version drift check (#1147)
        echo ""
        echo "=== Pre-commit hook version drift check ==="
        PRECOMMIT_CONFIG="$AI_TEMPLATE_ROOT/.pre-commit-config.yaml"

        # Helper function to parse pre-commit config YAML
        # Outputs: repo_url|rev (one per line), URLs normalized (no .git suffix)
        # NOTE: Use heredoc (not -c) to avoid shell hook escaping != to \!= (#1987)
        parse_precommit_config() {
            local config_path="$1"
            PRECOMMIT_CONFIG="$config_path" python3 <<'PYEOF'
import yaml
import os
import sys
try:
    config_path = os.environ.get('PRECOMMIT_CONFIG', '.pre-commit-config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for repo in config.get('repos', []):
        url = repo.get('repo', '')
        rev = repo.get('rev', '')
        if url and url != 'local' and rev:
            # Normalize URL: strip .git suffix for consistent matching
            if url.endswith('.git'):
                url = url[:-4]
            print(f'{url}|{rev}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
PYEOF
        }

        # Helper to find version for a URL in versions list (handles regex safely)
        find_version_for_url() {
            local search_url="$1"
            local versions_data="$2"
            while IFS='|' read -r url rev; do
                [[ "$url" == "$search_url" ]] && echo "$rev" && return 0
            done <<<"$versions_data"
            return 1
        }

        if [[ ! -f "$PRECOMMIT_CONFIG" ]]; then
            echo "Warning: .pre-commit-config.yaml not found in ai_template"
        else
            # Show errors for template parsing (essential for drift check)
            TEMPLATE_PARSE_STDERR=$(mktemp)
            TEMPLATE_VERSIONS=$(parse_precommit_config "$PRECOMMIT_CONFIG" 2>"$TEMPLATE_PARSE_STDERR")

            if [[ -z "$TEMPLATE_VERSIONS" ]]; then
                echo "Warning: Could not parse ai_template pre-commit config"
                if [[ -s "$TEMPLATE_PARSE_STDERR" ]]; then
                    echo "  Error: $(cat "$TEMPLATE_PARSE_STDERR")"
                else
                    echo "  (PyYAML may not be installed)"
                fi
                rm -f "$TEMPLATE_PARSE_STDERR"
            else
                rm -f "$TEMPLATE_PARSE_STDERR"
                PRECOMMIT_DRIFT_REPOS=()
                for repo in "${REPOS[@]}"; do
                    repo_name=$(basename "$repo")
                    repo_config="$repo/.pre-commit-config.yaml"
                    drifted_hooks=()

                    [[ ! -f "$repo_config" ]] && continue

                    REPO_VERSIONS=$(parse_precommit_config "$repo_config" 2>/dev/null)

                    # Compare versions using exact string matching (not grep regex)
                    while IFS='|' read -r url rev; do
                        [[ -z "$url" ]] && continue
                        template_rev=$(find_version_for_url "$url" "$TEMPLATE_VERSIONS")
                        if [[ -n "$template_rev" && "$rev" != "$template_rev" ]]; then
                            hook_name=$(basename "$url")
                            drifted_hooks+=("$hook_name: $rev -> $template_rev")
                        fi
                    done <<<"$REPO_VERSIONS"

                    # Check for template hooks missing from target repo
                    missing_hooks=()
                    while IFS='|' read -r url rev; do
                        [[ -z "$url" ]] && continue
                        repo_rev=$(find_version_for_url "$url" "$REPO_VERSIONS")
                        if [[ -z "$repo_rev" ]]; then
                            hook_name=$(basename "$url")
                            missing_hooks+=("$hook_name (missing)")
                        fi
                    done <<<"$TEMPLATE_VERSIONS"

                    if [[ ${#drifted_hooks[@]} -gt 0 || ${#missing_hooks[@]} -gt 0 ]]; then
                        total_issues=$((${#drifted_hooks[@]} + ${#missing_hooks[@]}))
                        echo ""
                        echo -e "${YELLOW}$repo_name${NC} has $total_issues pre-commit hook issues:"
                        for hook in "${drifted_hooks[@]}"; do
                            echo "  - $hook"
                        done
                        for hook in "${missing_hooks[@]}"; do
                            echo "  - $hook"
                        done
                        PRECOMMIT_DRIFT_REPOS+=("$repo_name")
                    fi
                done

                if [[ ${#PRECOMMIT_DRIFT_REPOS[@]} -eq 0 ]]; then
                    echo "No pre-commit hook version drift detected."
                else
                    echo ""
                    echo "Found pre-commit hook drift in ${#PRECOMMIT_DRIFT_REPOS[@]} repos."
                    echo "Run sync_repo.sh to update (syncs .pre-commit-config.yaml)."
                    set_exit_status 1
                fi
            fi
        fi
    fi
fi

if [[ "$EXIT_CODE" == "true" ]]; then
    exit "$EXIT_STATUS"
fi

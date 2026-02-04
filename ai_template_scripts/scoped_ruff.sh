#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# scoped_ruff.sh - Ruff wrapper that scopes fixes based on role
#
# In multi-worker mode, auto-fix tools like `ruff --fix` can accidentally
# modify files being worked on by other workers. This wrapper scopes ruff:
#
# Modes:
# 1. Worker mode (AI_WORKER_ID set): Only run on this worker's tracked files
# 2. Manager mode (AI_ROLE=MANAGER with active workers): Only run on files NOT
#    being modified by any worker (safe = committed & not in any tracker)
# 3. Single-worker mode: Pass through unchanged
#
set -euo pipefail

warn() {
    echo "[scoped_ruff] $*" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "scoped_ruff.sh ${git_hash} (${date})"
    exit 0
}

usage() {
    cat <<'EOF'
Usage: scoped_ruff.sh [RUFF_ARGS...] [FILES...]

Ruff wrapper that scopes fixes based on role to prevent workers stepping on each other.

Modes:
  Worker (AI_WORKER_ID set):
    Only runs on this worker's tracked files (.worker_N_files.json)

  Manager (AI_ROLE=MANAGER with active workers):
    Only runs on "safe" files - committed files NOT in any worker's tracker

  Single-worker (neither set):
    Pass through to ruff unchanged

Options:
  --version     Show version information
  -h, --help    Show this help message

All other options are passed through to ruff.
EOF
    exit 0
}

# Handle our options before ruff passthrough
case "${1:-}" in
    --version) version ;;
    -h|--help) usage ;;
esac

# Find the real ruff
REAL_RUFF=""
for loc in /usr/local/bin/ruff /opt/homebrew/bin/ruff; do
    if [[ -x "$loc" ]]; then
        REAL_RUFF="$loc"
        break
    fi
done

if [[ -z "$REAL_RUFF" ]]; then
    # Try to find ruff in PATH (excluding this script's directory)
    # SCRIPT_DIR already set at top of script
    IFS=':' read -ra PATH_DIRS <<< "$PATH"
    for p in "${PATH_DIRS[@]}"; do
        [[ "$p" == "$SCRIPT_DIR" ]] && continue
        [[ -x "$p/ruff" ]] && REAL_RUFF="$p/ruff" && break
    done
fi

if [[ -z "$REAL_RUFF" ]]; then
    echo "ERROR: Real ruff not found in PATH" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Check for python3 (needed for scoped modes)
if ! command -v python3 &>/dev/null; then
    warn "python3 not found; passing through to ruff"
    exec "$REAL_RUFF" "$@"
fi

# Determine mode: worker, manager, or single
MODE="single"
if [[ -n "${AI_WORKER_ID:-}" ]]; then
    MODE="worker"
elif [[ "${AI_ROLE:-}" == "MANAGER" ]]; then
    # Check if any worker trackers exist with alive PIDs
    has_active_workers=$(REPO_ROOT="$REPO_ROOT" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

repo = Path(os.environ.get("REPO_ROOT", "."))
pattern = re.compile(r"^\.worker_(\d+)_files\.json$")

for path in repo.iterdir():
    if not pattern.match(path.name):
        continue
    try:
        data = json.loads(path.read_text())
        pid = data.get("pid", 0)
        if pid > 0:
            try:
                os.kill(pid, 0)
                print("yes")  # At least one worker is alive
                break
            except ProcessLookupError:
                pass
    except Exception:
        pass
PY
)
    if [[ "$has_active_workers" == "yes" ]]; then
        MODE="manager"
    fi
fi

# Single-worker mode: pass through unchanged
if [[ "$MODE" == "single" ]]; then
    exec "$REAL_RUFF" "$@"
fi

# Manager mode: exclude files being modified by workers
if [[ "$MODE" == "manager" ]]; then
    if ! excluded_output=$(REPO_ROOT="$REPO_ROOT" python3 - <<'PY'
import json
import os
import re
import subprocess
from pathlib import Path

repo = Path(os.environ.get("REPO_ROOT", "."))
excluded = set()

# Get all files from active worker trackers
pattern = re.compile(r"^\.worker_(\d+)_files\.json$")
for path in repo.iterdir():
    if not pattern.match(path.name):
        continue
    try:
        data = json.loads(path.read_text())
        pid = data.get("pid", 0)
        if pid > 0:
            try:
                os.kill(pid, 0)
                # Worker is alive - exclude its files
                for f in data.get("files", []):
                    excluded.add(f)
            except ProcessLookupError:
                pass
    except Exception:
        pass

# Get all uncommitted files from git status
try:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Format: XY filename or XY -> newname
            parts = line[3:].split(" -> ")
            filename = parts[-1].strip()
            if filename:
                excluded.add(filename)
except Exception:
    pass

for f in sorted(excluded):
    print(f)
PY
    ); then
        warn "Unable to determine excluded files; passing through to ruff"
        exec "$REAL_RUFF" "$@"
    fi

    EXCLUDED_FILES=()
    while IFS= read -r file; do
        [[ -n "$file" ]] && EXCLUDED_FILES+=("$file")
    done <<< "$excluded_output"

    if [[ ${#EXCLUDED_FILES[@]} -eq 0 ]]; then
        # No files to exclude - run on everything
        exec "$REAL_RUFF" "$@"
    fi

    # Build --extend-exclude patterns for ruff (appends to existing excludes)
    # Note: --extend-exclude must come after the subcommand (check/format)
    # We insert them after the first argument if it's a subcommand
    EXCLUDE_ARGS=()
    for file in "${EXCLUDED_FILES[@]}"; do
        EXCLUDE_ARGS+=("--extend-exclude" "$file")
    done

    warn "Manager mode: excluding ${#EXCLUDED_FILES[@]} file(s) being modified by workers"

    # If first arg is a subcommand, insert excludes after it
    if [[ $# -gt 0 ]]; then
        FIRST_ARG="$1"
        shift
        case "$FIRST_ARG" in
            check|format)
                exec "$REAL_RUFF" "$FIRST_ARG" "${EXCLUDE_ARGS[@]}" "$@"
                ;;
            *)
                # Not a subcommand - put excludes at end
                exec "$REAL_RUFF" "$FIRST_ARG" "$@" "${EXCLUDE_ARGS[@]}"
                ;;
        esac
    else
        # No args - default check with excludes
        exec "$REAL_RUFF" check "${EXCLUDE_ARGS[@]}"
    fi
fi

# Worker mode: scope to tracked files
TRACKER_FILE="$REPO_ROOT/.worker_${AI_WORKER_ID}_files.json"

# No tracker file: avoid running unscoped in multi-worker mode
if [[ ! -f "$TRACKER_FILE" ]]; then
    warn "No tracker file found for worker $AI_WORKER_ID; skipping unscoped run"
    exit 0
fi

if ! tracked_output=$(TRACKER_FILE="$TRACKER_FILE" python3 - <<'PY'
import json
import os
import sys

path = os.environ.get("TRACKER_FILE")
if not path:
    print("missing TRACKER_FILE", file=sys.stderr)
    sys.exit(2)
try:
    with open(path) as f:
        data = json.load(f)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
for item in data.get("files", []):
    print(item)
PY
); then
    warn "Unable to read tracked files from $TRACKER_FILE"
    exit 0
fi

TRACKED_FILES=()
while IFS= read -r file; do
    [[ -n "$file" ]] && TRACKED_FILES+=("$file")
done <<< "$tracked_output"

# No tracked files: skip unscoped run
if [[ ${#TRACKED_FILES[@]} -eq 0 ]]; then
    warn "No tracked files recorded for worker $AI_WORKER_ID; skipping unscoped run"
    exit 0
fi

# Ruff subcommands (avoid treating them as file paths)
is_ruff_subcommand() {
    case "$1" in
        check|format|rule|clean|config|lsp|server|analyze)
            return 0
            ;;
    esac
    return 1
}

flag_takes_value() {
    local flag="$1"
    for candidate in \
        -c --config --select --extend-select --ignore --extend-ignore --exclude \
        --extend-exclude --per-file-ignores --fixable --unfixable --format \
        --target-version --line-length --stdin-filename --output-format; do
        if [[ "$flag" == "$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

# Separate ruff args from file args
RUFF_ARGS=()
FILE_ARGS=()
PARSING_ARGS=true
ARGS=("$@")
arg_count=${#ARGS[@]}
i=0

while [[ $i -lt $arg_count ]]; do
    arg="${ARGS[$i]}"
    if $PARSING_ARGS; then
        if [[ "$arg" == "--" ]]; then
            PARSING_ARGS=false
            i=$((i + 1))
            continue
        fi
        if [[ "$arg" == --*=* ]]; then
            RUFF_ARGS+=("$arg")
            i=$((i + 1))
            continue
        fi
        if flag_takes_value "$arg"; then
            RUFF_ARGS+=("$arg")
            if [[ $((i + 1)) -lt $arg_count ]]; then
                RUFF_ARGS+=("${ARGS[$((i + 1))]}")
                i=$((i + 2))
                continue
            fi
            i=$((i + 1))
            continue
        fi
        if [[ "$arg" == -* ]]; then
            RUFF_ARGS+=("$arg")
            i=$((i + 1))
            continue
        fi
        if is_ruff_subcommand "$arg" && [[ ! -e "$arg" ]]; then
            RUFF_ARGS+=("$arg")
            i=$((i + 1))
            continue
        fi
        # First non-flag arg is a file/path
        PARSING_ARGS=false
        FILE_ARGS+=("$arg")
        i=$((i + 1))
        continue
    fi
    FILE_ARGS+=("$arg")
    i=$((i + 1))
done

# If no file args provided, ruff uses current directory
# In that case, we scope to only tracked files
if [[ ${#FILE_ARGS[@]} -eq 0 ]]; then
    # Filter to only tracked .py files that exist
    SCOPED_FILES=()
    for file in "${TRACKED_FILES[@]}"; do
        abs_path="$REPO_ROOT/$file"
        if [[ "$file" == *.py && -f "$abs_path" ]]; then
            SCOPED_FILES+=("$abs_path")
        fi
    done

    if [[ ${#SCOPED_FILES[@]} -eq 0 ]]; then
        # No Python files to check - exit successfully
        exit 0
    fi

    exec "$REAL_RUFF" "${RUFF_ARGS[@]}" "${SCOPED_FILES[@]}"
fi

# File args provided: filter to only tracked files
FILTERED_FILES=()

add_filtered_file() {
    local file="$1"
    for existing in "${FILTERED_FILES[@]}"; do
        [[ "$existing" == "$file" ]] && return 0
    done
    FILTERED_FILES+=("$file")
}

for file in "${FILE_ARGS[@]}"; do
    if [[ "$file" == "." || "$file" == "./" ]]; then
        file="$REPO_ROOT"
    fi

    # Directory args: include tracked Python files under the directory
    if [[ -d "$file" ]]; then
        if [[ "$file" == "$REPO_ROOT" ]]; then
            rel_dir=""
        else
            rel_dir="${file#"$REPO_ROOT"/}"
            rel_dir="${rel_dir#./}"
            rel_dir="${rel_dir%/}"
        fi

        for tracked in "${TRACKED_FILES[@]}"; do
            if [[ -z "$rel_dir" || "$tracked" == "$rel_dir/"* || "$tracked" == "$rel_dir" ]]; then
                tracked_path="$REPO_ROOT/$tracked"
                if [[ "$tracked" == *.py && -f "$tracked_path" ]]; then
                    add_filtered_file "$tracked_path"
                fi
            fi
        done
        continue
    fi

    # Normalize path relative to repo root
    rel_path="${file#"$REPO_ROOT"/}"
    rel_path="${rel_path#./}"

    for tracked in "${TRACKED_FILES[@]}"; do
        if [[ "$rel_path" == "$tracked" || "$file" == "$tracked" ]]; then
            add_filtered_file "$REPO_ROOT/$tracked"
            break
        fi
    done
done

if [[ ${#FILTERED_FILES[@]} -eq 0 ]]; then
    # None of the specified files are tracked by this worker
    # Advisory: warn but don't block
    echo "[scoped_ruff] Advisory: none of the specified files are tracked by worker $AI_WORKER_ID" >&2
    exit 0
fi

exec "$REAL_RUFF" "${RUFF_ARGS[@]}" "${FILTERED_FILES[@]}"

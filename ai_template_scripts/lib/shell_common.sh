#!/usr/bin/env bash
# shell_common.sh - Shared shell library for error handling and cleanup
#
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
#
# Usage:
#   source "$(dirname "$0")/lib/shell_common.sh"
#   # or from git root:
#   source ./ai_template_scripts/lib/shell_common.sh
#
# Functions provided:
#   err MSG [DETAIL]     - Print error and exit 1
#   warn MSG             - Print warning (colored)
#   info MSG             - Print info (colored)
#   tmpfile_tracked      - Create tracked temp file (auto-cleanup on exit)
#   tmpdir_tracked       - Create tracked temp dir (auto-cleanup on exit)
#   cleanup_on_exit      - Register custom cleanup function
#
# Part of #1958: Shell scripts missing cleanup trap patterns

# Prevent double-sourcing
[[ -n "${_SHELL_COMMON_SOURCED:-}" ]] && return 0
_SHELL_COMMON_SOURCED=1

# === Colors ===
# Use ANSI colors for terminal output. Safe for piping (no-op if not terminal).
if [[ -t 2 ]]; then
    YELLOW='\033[1;33m'
    RED='\033[1;31m'
    CYAN='\033[1;36m'
    NC='\033[0m'
else
    YELLOW=''
    RED=''
    CYAN=''
    NC=''
fi

# === Error Handling ===

# err MSG [DETAIL] - Print error message and exit 1
# Example: err "File not found" "Check path: $filepath"
err() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    if [[ -n "${2:-}" ]]; then
        echo "        $2" >&2
    fi
    exit 1
}

# warn MSG - Print warning message (does not exit)
# Example: warn "File may be stale"
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

# info MSG - Print info message
# Example: info "Processing 42 files..."
info() {
    echo -e "${CYAN}[INFO]${NC} $1" >&2
}

# === Temporary File Management ===
# Tracked temp files are automatically cleaned up on exit/error.
# Uses a file-based registry to work around subshell limitations.

# Registry file for tracked temp files (unique per process)
_TMPFILE_REGISTRY="${TMPDIR:-/tmp}/.shell_common_$$_registry"

# Internal cleanup function - removes all tracked temp files
_cleanup_tmpfiles() {
    if [[ -f "$_TMPFILE_REGISTRY" ]]; then
        while IFS= read -r item; do
            if [[ -n "$item" && -e "$item" ]]; then
                rm -rf "$item" 2>/dev/null || true
            fi
        done < "$_TMPFILE_REGISTRY"
        rm -f "$_TMPFILE_REGISTRY" 2>/dev/null || true
    fi
}

# Array of custom cleanup functions
_CLEANUP_FUNCS=()

# Internal master cleanup - runs temp cleanup and custom cleanup functions
_master_cleanup() {
    local func
    # Run custom cleanup functions first (in reverse order)
    local i
    for ((i=${#_CLEANUP_FUNCS[@]}-1; i>=0; i--)); do
        func="${_CLEANUP_FUNCS[$i]}"
        "$func" 2>/dev/null || true
    done
    # Then clean up temp files
    _cleanup_tmpfiles
}

# Register the master cleanup trap
trap _master_cleanup EXIT INT TERM

# tmpfile_tracked [ARGS...] - Create tracked temp file
# Returns: path to temp file on stdout
# Example: local f=$(tmpfile_tracked)
# Note: File-based registry allows tracking across subshells
tmpfile_tracked() {
    local f
    f=$(mktemp "$@")
    echo "$f" >> "$_TMPFILE_REGISTRY"
    echo "$f"
}

# tmpdir_tracked [ARGS...] - Create tracked temp directory
# Returns: path to temp dir on stdout
# Example: local d=$(tmpdir_tracked)
# Note: File-based registry allows tracking across subshells
tmpdir_tracked() {
    local d
    d=$(mktemp -d "$@")
    echo "$d" >> "$_TMPFILE_REGISTRY"
    echo "$d"
}

# cleanup_on_exit FUNC - Register custom cleanup function
# Functions are called in reverse order (LIFO) before temp file cleanup.
# Example: cleanup_on_exit my_cleanup_func
# Note: Custom functions must be registered in the main shell (not subshell)
cleanup_on_exit() {
    local func="$1"
    _CLEANUP_FUNCS+=("$func")
}

# === Require Commands ===

# require_cmd CMD [MSG] - Check if command exists, exit if not
# Example: require_cmd jq "Install with: brew install jq"
require_cmd() {
    local cmd="$1"
    local msg="${2:-Install $cmd and retry}"
    if ! command -v "$cmd" &>/dev/null; then
        err "Required command not found: $cmd" "$msg"
    fi
}

# require_file PATH [MSG] - Check if file exists, exit if not
# Example: require_file "$config" "Create config file first"
require_file() {
    local path="$1"
    local msg="${2:-File required: $path}"
    if [[ ! -f "$path" ]]; then
        err "Required file not found: $path" "$msg"
    fi
}

# require_dir PATH [MSG] - Check if directory exists, exit if not
# Example: require_dir "$output_dir" "Create output directory first"
require_dir() {
    local path="$1"
    local msg="${2:-Directory required: $path}"
    if [[ ! -d "$path" ]]; then
        err "Required directory not found: $path" "$msg"
    fi
}

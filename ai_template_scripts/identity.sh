#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
#
# Shell identity loader for ait_identity.toml.
# Source this file to get AIT_OWNER_NAME, AIT_OWNER_EMAIL, etc.
#
# Usage:
#   source "$(dirname "$0")/identity.sh"
#   echo "$AIT_OWNER_NAME"    # "Andrew Yates"
#   echo "$AIT_GITHUB_ORG"    # "dropbox-ai-prototypes"
#
# See: #2974 (identity extraction for public release)

# Guard against double-sourcing
if [[ -n "${_AIT_IDENTITY_LOADED:-}" ]]; then
    return 0 2>/dev/null || true
fi

# --- TOML parser (simple key = "value" extraction) ---

_ait_toml_get() {
    # Extract a TOML value by key from a file.
    # Args: $1=file $2=section $3=key $4=default
    # Only handles: key = "value" (string) and key = N (integer)
    local file="$1" section="$2" key="$3" default="${4:-}"

    if [[ ! -f "$file" ]]; then
        echo "$default"
        return
    fi

    local in_section=0
    local value=""
    while IFS= read -r line; do
        # Skip comments and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Section header
        if [[ "$line" =~ ^\[([a-zA-Z_]+)\] ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$section" ]]; then
                in_section=1
            else
                # Exiting our section
                [[ $in_section -eq 1 ]] && break
                in_section=0
            fi
            continue
        fi

        # Key = value (only in our section)
        if [[ $in_section -eq 1 && "$line" =~ ^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*) ]]; then
            value="${BASH_REMATCH[1]}"
            # Strip quotes
            value="${value#\"}"
            value="${value%\"}"
            # Strip trailing whitespace
            value="${value%"${value##*[![:space:]]}"}"
            echo "$value"
            return
        fi
    done <"$file"

    echo "$default"
}

_ait_toml_get_array() {
    # Extract a TOML array value as a pipe-delimited string.
    # Args: $1=file $2=section $3=key $4=default
    # Handles: key = ["val1", "val2"]
    local file="$1" section="$2" key="$3" default="${4:-}"

    if [[ ! -f "$file" ]]; then
        echo "$default"
        return
    fi

    local in_section=0
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        if [[ "$line" =~ ^\[([a-zA-Z_]+)\] ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$section" ]]; then
                in_section=1
            else
                [[ $in_section -eq 1 ]] && break
                in_section=0
            fi
            continue
        fi

        if [[ $in_section -eq 1 && "$line" =~ ^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*) ]]; then
            local raw="${BASH_REMATCH[1]}"
            # Strip brackets
            raw="${raw#\[}"
            raw="${raw%\]}"
            # Replace ", " with pipe delimiter, strip quotes
            raw="$(echo "$raw" | sed 's/[[:space:]]*,[[:space:]]*/|/g; s/"//g')"
            echo "$raw"
            return
        fi
    done <"$file"

    echo "$default"
}

# --- Find config file ---

_ait_find_identity_toml() {
    # Walk up from $1 (or cwd) looking for ait_identity.toml
    local dir="${1:-$(pwd)}"
    dir="$(cd "$dir" 2>/dev/null && pwd)" || return 1

    while true; do
        if [[ -f "$dir/ait_identity.toml" ]]; then
            echo "$dir/ait_identity.toml"
            return 0
        fi
        if [[ -d "$dir/.git" ]]; then
            # Repo root reached, no config
            return 1
        fi
        local parent="$(dirname "$dir")"
        [[ "$parent" == "$dir" ]] && return 1
        dir="$parent"
    done
}

# --- Load identity ---

_AIT_IDENTITY_TOML="$(_ait_find_identity_toml 2>/dev/null || true)"

# Defaults match Python identity.py _PLACEHOLDER defaults
AIT_OWNER_NAME="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "owner" "name" "Andrew Yates")"
AIT_OWNER_EMAIL="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "owner" "email" "ayates@dropbox.com")"
AIT_OWNER_USERNAMES="$(_ait_toml_get_array "$_AIT_IDENTITY_TOML" "owner" "usernames" "ayates|andrewdyates|ayates_dbx")"
AIT_GITHUB_ORG="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "org" "github_org" "dropbox-ai-prototypes")"
AIT_COMPANY_NAME="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "org" "company_name" "Dropbox, Inc.")"
AIT_COMPANY_ABBREV="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "org" "abbreviation" "DBX")"
AIT_COPYRIGHT_YEAR="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "copyright" "year" "2026")"
AIT_COPYRIGHT_HOLDER="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "copyright" "holder" "Dropbox, Inc.")"
AIT_COPYRIGHT_LICENSE="$(_ait_toml_get "$_AIT_IDENTITY_TOML" "copyright" "license" "Apache-2.0")"

# Build derived values for grep patterns
# owner_usernames as a grep alternation pattern: "ayates|andrewdyates|ayates_dbx"
if [[ -n "$AIT_OWNER_USERNAMES" ]]; then
    AIT_AUTHOR_GREP_PATTERN="$(echo "$AIT_OWNER_USERNAMES" | sed 's/|/\\|/g')"
    # Also include owner name words for fuzzy matching
    _ait_first="$(echo "$AIT_OWNER_NAME" | awk '{print $1}')"
    _ait_last="$(echo "$AIT_OWNER_NAME" | awk '{print $NF}')"
    if [[ -n "$_ait_first" && -n "$_ait_last" && "$_ait_first" != "$_ait_last" ]]; then
        AIT_AUTHOR_GREP_PATTERN="${_ait_first}.*${_ait_last}\\|${AIT_AUTHOR_GREP_PATTERN}"
    fi
    unset _ait_first _ait_last
else
    # Fallback: use owner_name as grep pattern
    _ait_first="$(echo "$AIT_OWNER_NAME" | awk '{print $1}')"
    _ait_last="$(echo "$AIT_OWNER_NAME" | awk '{print $NF}')"
    AIT_AUTHOR_GREP_PATTERN="${_ait_first}.*${_ait_last}"
    unset _ait_first _ait_last
fi

# Copyright grep pattern: matches "copyright.*<holder>" or "copyright.*<owner_name>"
# Uses only owner name (not full author grep with usernames) to avoid
# ungrouped alternation false positives (#2974 self-audit)
_ait_first="$(echo "$AIT_OWNER_NAME" | awk '{print $1}')"
_ait_last="$(echo "$AIT_OWNER_NAME" | awk '{print $NF}')"
if [[ -n "$AIT_COPYRIGHT_HOLDER" ]]; then
    _ait_holder_word="$(echo "$AIT_COPYRIGHT_HOLDER" | awk -F'[, ]' '{print $1}')"
    AIT_COPYRIGHT_GREP_PATTERN="copyright.*${_ait_holder_word}\\|copyright.*${_ait_first}.*${_ait_last}"
    unset _ait_holder_word
else
    AIT_COPYRIGHT_GREP_PATTERN="copyright.*${_ait_first}.*${_ait_last}"
fi
unset _ait_first _ait_last

export AIT_OWNER_NAME AIT_OWNER_EMAIL AIT_OWNER_USERNAMES
export AIT_GITHUB_ORG AIT_COMPANY_NAME AIT_COMPANY_ABBREV
export AIT_COPYRIGHT_YEAR AIT_COPYRIGHT_HOLDER AIT_COPYRIGHT_LICENSE
export AIT_AUTHOR_GREP_PATTERN AIT_COPYRIGHT_GREP_PATTERN

_AIT_IDENTITY_LOADED=1

#!/bin/bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
#
# Setup GitHub Apps credentials from encrypted bundle in repo.
#
# Usage:
#   ./ai_template_scripts/gh_apps/setup_creds.sh
#
# Passphrase sources (checked in order):
#   1. AIT_GH_APPS_PASSPHRASE environment variable
#   2. .env file in repo root (AIT_GH_APPS_PASSPHRASE=...)
#   3. Interactive prompt
#
# The encrypted credentials file (.gh_apps_creds.enc) should be in the repo root.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CREDS_FILE="$REPO_ROOT/.gh_apps_creds.enc"
TARGET_DIR="$HOME/.ait_gh_apps"
ENV_FILE="$REPO_ROOT/.env"

# Check if encrypted file exists
if [[ ! -f "$CREDS_FILE" ]]; then
    echo "Error: No credentials file found at $CREDS_FILE" >&2
    echo "" >&2
    echo "This repo doesn't have bundled credentials yet." >&2
    echo "Ask the repo owner to run:" >&2
    echo "  python3 -m ai_template_scripts.gh_apps.share_credentials export-to-repo" >&2
    exit 1
fi

# Get passphrase from env var, .env file, or prompt
get_passphrase() {
    # 1. Check environment variable
    if [[ -n "${AIT_GH_APPS_PASSPHRASE:-}" ]]; then
        echo "$AIT_GH_APPS_PASSPHRASE"
        return 0
    fi

    # 2. Check .env file
    if [[ -f "$ENV_FILE" ]]; then
        local from_env
        from_env=$(grep -E "^AIT_GH_APPS_PASSPHRASE=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d'=' -f2- | sed 's/^"//;s/"$//;s/^'\''//;s/'\''$//')
        if [[ -n "$from_env" ]]; then
            echo "$from_env"
            return 0
        fi
    fi

    # 3. Interactive prompt (only if terminal available)
    if [[ -t 0 ]]; then
        read -s -p "Enter passphrase: " passphrase
        echo >&2  # newline after hidden input
        echo "$passphrase"
        return 0
    fi

    # No passphrase available
    echo "Error: No passphrase available." >&2
    echo "Set AIT_GH_APPS_PASSPHRASE env var or add to .env file" >&2
    return 1
}

# Check if already set up (skip prompt in non-interactive mode)
if [[ -f "$TARGET_DIR/config.yaml" ]]; then
    if [[ -t 0 ]]; then
        echo "Credentials already exist at $TARGET_DIR"
        read -p "Overwrite? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    else
        echo "Credentials already exist at $TARGET_DIR - overwriting (non-interactive mode)"
    fi
fi

# Create target directory
mkdir -p "$TARGET_DIR"
chmod 700 "$TARGET_DIR"

# Get passphrase
PASSPHRASE=$(get_passphrase) || exit 1

# Decrypt and extract
echo "Decrypting credentials..."
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

if ! echo "$PASSPHRASE" | openssl enc -d -aes-256-cbc -pbkdf2 -in "$CREDS_FILE" -pass stdin -out "$TMPDIR/creds.tar.gz" 2>/dev/null; then
    echo "Error: Decryption failed - wrong passphrase?" >&2
    exit 1
fi

tar xzf "$TMPDIR/creds.tar.gz" -C "$TMPDIR"

# Copy files (handle both nested and flat structures)
echo "Installing to $TARGET_DIR..."
cp "$TMPDIR"/*/*.yaml "$TARGET_DIR/" 2>/dev/null || cp "$TMPDIR"/*.yaml "$TARGET_DIR/" 2>/dev/null || true
cp "$TMPDIR"/*/*.pem "$TARGET_DIR/" 2>/dev/null || cp "$TMPDIR"/*.pem "$TARGET_DIR/" 2>/dev/null || true

# Set permissions
chmod 600 "$TARGET_DIR"/*.pem "$TARGET_DIR"/*.yaml 2>/dev/null || true

echo ""
echo "Done! Credentials installed to $TARGET_DIR"
echo ""
echo "Test with:"
echo "  AIT_GH_APPS_DEBUG=1 python3 -m ai_template_scripts.gh_apps.get_token --repo \$(basename \$(pwd))"

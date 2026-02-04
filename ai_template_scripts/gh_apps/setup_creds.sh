#!/bin/bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
#
# Setup GitHub Apps credentials from encrypted bundle in repo.
#
# Usage:
#   ./ai_template_scripts/gh_apps/setup_creds.sh
#
# The encrypted credentials file (.gh_apps_creds.enc) should be in the repo root.
# You'll be prompted for the passphrase (get it from the team).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CREDS_FILE="$REPO_ROOT/.gh_apps_creds.enc"
TARGET_DIR="$HOME/.ait_gh_apps"

# Check if encrypted file exists
if [[ ! -f "$CREDS_FILE" ]]; then
    echo "Error: No credentials file found at $CREDS_FILE" >&2
    echo "" >&2
    echo "This repo doesn't have bundled credentials yet." >&2
    echo "Ask the repo owner to run:" >&2
    echo "  python3 -m ai_template_scripts.gh_apps.share_credentials export-to-repo" >&2
    exit 1
fi

# Check if already set up
if [[ -f "$TARGET_DIR/config.yaml" ]]; then
    echo "Credentials already exist at $TARGET_DIR"
    read -p "Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Create target directory
mkdir -p "$TARGET_DIR"
chmod 700 "$TARGET_DIR"

# Decrypt and extract
echo "Decrypting credentials (enter passphrase)..."
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

openssl enc -d -aes-256-cbc -pbkdf2 -in "$CREDS_FILE" -out "$TMPDIR/creds.tar.gz"
tar xzf "$TMPDIR/creds.tar.gz" -C "$TMPDIR"

# Copy files
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

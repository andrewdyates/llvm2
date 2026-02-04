#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Update ~/.claude/settings.json with recommended settings.

Merges recommended settings into the user-level Claude settings file
without overwriting existing values.

Usage:
    python3 update_claude_user_settings.py           # Add recommended settings
    python3 update_claude_user_settings.py --check   # Check current settings
    python3 update_claude_user_settings.py --dry-run # Show what would change
"""

from __future__ import annotations

__all__ = [
    "RECOMMENDED_ENV",
    "get_settings_path",
    "load_settings",
    "save_settings",
    "check_settings",
    "merge_settings",
    "main",
]

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Settings to ensure exist in ~/.claude/settings.json
# These are merged into existing settings, not overwritten
RECOMMENDED_ENV: dict[str, str] = {
    # Prevent auto-updates during AI sessions (stability)
    "DISABLE_AUTOUPDATER": "1",
}


def get_settings_path() -> Path:
    """Get path to user-level Claude settings."""
    return Path.home() / ".claude" / "settings.json"


def load_settings(path: Path) -> dict[str, Any]:
    """Load existing settings or return empty dict."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Failed to read {path}: {e}", file=sys.stderr)
        return {}


def save_settings(path: Path, settings: dict[str, Any]) -> bool:
    """Save settings to file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        return True
    except OSError as e:
        print(f"ERROR: Failed to write {path}: {e}", file=sys.stderr)
        return False


def check_settings(settings: dict[str, Any]) -> dict[str, bool]:
    """Check which recommended settings are present."""
    env = settings.get("env", {})
    return {key: key in env for key in RECOMMENDED_ENV}


def merge_settings(settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Merge recommended settings into existing settings.

    Returns:
        Tuple of (updated_settings, list_of_changes)
    """
    updated = dict(settings)
    changes: list[str] = []

    if "env" not in updated:
        updated["env"] = {}

    for key, value in RECOMMENDED_ENV.items():
        if key not in updated["env"]:
            updated["env"][key] = value
            changes.append(f"Added env.{key}={value}")

    return updated, changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check current settings only"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without writing"
    )
    args = parser.parse_args()

    path = get_settings_path()
    settings = load_settings(path)

    if args.check:
        status = check_settings(settings)
        print(f"Settings file: {path}")
        all_present = True
        for key, present in status.items():
            marker = "✓" if present else "✗"
            print(f"  {marker} env.{key}")
            if not present:
                all_present = False
        return 0 if all_present else 1

    updated, changes = merge_settings(settings)

    if not changes:
        print(f"OK: All recommended settings already present in {path}")
        return 0

    if args.dry_run:
        print(f"Would update {path}:")
        for change in changes:
            print(f"  {change}")
        return 0

    if save_settings(path, updated):
        print(f"Updated {path}:")
        for change in changes:
            print(f"  {change}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

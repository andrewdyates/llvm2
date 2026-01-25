"""
looper/hooks.py - Git hook management

Installs and manages git hooks for:
- pre-commit: Fast linters, sensitive file detection
- commit-msg: Format validation, trailer injection
- post-commit: Issue claiming

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0
"""

from pathlib import Path
from typing import Any

# Hook configurations - external files or fallback inline content
HOOK_CONFIGS: dict[str, dict[str, Any]] = {
    "pre-commit": {
        "external_file": "ai_template_scripts/pre-commit-hook.sh",
        "marker": "Missing copyright header",  # Identifies our hook (from warning message)
        "fallback_content": """#!/bin/bash
# ai_template pre-commit hook - fast linters only
STAGED=$(git diff --cached --name-only --diff-filter=ACMR)
# Check for sensitive files (block if found)
SENSITIVE=$(echo "$STAGED" | grep -E '\\.(env|key|pem|p12|pfx)$|credentials\\.json|secrets\\.json|_secret')
if [ -n "$SENSITIVE" ]; then
    echo "ERROR: Refusing to commit potentially sensitive files:"
    echo "$SENSITIVE"
    echo "If these are safe, use: git commit --no-verify"
    exit 1
fi
echo "$STAGED" | grep '\\.py$' | xargs -r ruff check || exit 1
echo "$STAGED" | grep '\\.sh$' | xargs -r shellcheck 2>/dev/null || true
""",
    },
    "commit-msg": {
        "external_file": "ai_template_scripts/commit-msg-hook.sh",
        "marker": "[W]N:",  # From our hook's format hint output
        "fallback_content": None,  # No fallback - must have external file
    },
    "post-commit": {
        "external_file": "ai_template_scripts/post-commit-hook.sh",
        "marker": "Claims #",  # From our hook's issue claiming
        "fallback_content": None,  # No fallback - must have external file
    },
}


def _load_hook_content(config: dict[str, Any]) -> str | None:
    """Load hook content from external file or return fallback.

    Args:
        config: Hook configuration dict with external_file and fallback_content

    Returns:
        Hook script content, or None if not available.
    """
    external_path = config.get("external_file")
    if external_path:
        path = Path(external_path)
        if path.exists():
            try:
                return path.read_text()
            except (OSError, UnicodeDecodeError):
                pass  # Fall through to fallback
    return config.get("fallback_content")


def _hook_needs_update(
    hook_path: Path, marker: str, new_content: str
) -> tuple[bool, str]:
    """Check if a hook needs to be installed or updated.

    Args:
        hook_path: Path to the git hook file
        marker: String that identifies our hook content
        new_content: The content we want to install

    Returns:
        (needs_update, final_content) - final_content may include existing hook
    """
    if not hook_path.exists():
        return True, new_content

    try:
        existing = hook_path.read_text()
    except (OSError, UnicodeDecodeError):
        # Can't read existing hook, install fresh
        return True, new_content

    if marker in existing:
        # Our hook is already present
        return False, existing

    # Append to existing hook (don't replace user hooks)
    combined = existing.rstrip() + "\n\n" + new_content
    return True, combined


def install_hooks() -> None:
    """Install git hooks if not present or outdated.

    Hooks are loaded from external files in ai_template_scripts/ when available,
    falling back to inline content. Existing user hooks are preserved by appending
    our hooks rather than replacing.

    Hook installation is idempotent - running multiple times is safe.
    """
    hooks_dir = Path(".git/hooks")
    if not hooks_dir.exists():
        return  # Not a git repo

    for name, config in HOOK_CONFIGS.items():
        # Load content from external file or use fallback
        content = _load_hook_content(config)
        if content is None:
            continue  # No content available for this hook

        hook_path = hooks_dir / name
        marker = config["marker"]

        needs_update, final_content = _hook_needs_update(hook_path, marker, content)
        if needs_update:
            try:
                hook_path.write_text(final_content)
                hook_path.chmod(0o755)
                print(f"✓ Installed hook: {name}")
            except OSError as e:
                print(f"⚠ Failed to install hook {name}: {e}")

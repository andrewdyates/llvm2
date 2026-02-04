# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/hooks.py - Git hook management

Installs and manages git hooks for:
- pre-commit: Fast linters, sensitive file detection
- commit-msg: Format validation, trailer injection
- post-commit: Issue claiming

Module contracts:
    ENSURES: Hook installation is idempotent
    ENSURES: Existing user hooks are preserved when readable (appended, not replaced)
    ENSURES: All functions handle filesystem errors gracefully
"""

from pathlib import Path
from typing import Any

from looper.log import debug_swallow, log_info, log_warning

__all__ = [
    "install_hooks",
]

# Hook configurations - external files or fallback inline content
HOOK_CONFIGS: dict[str, dict[str, Any]] = {
    "pre-commit": {
        "external_file": "ai_template_scripts/pre-commit-hook.sh",
        "marker": "Missing copyright header",  # Identifies our hook
        "fallback_content": """#!/bin/bash
# ai_template pre-commit hook - fast linters only
STAGED=$(git diff --cached --name-only --diff-filter=ACMR)
# Check for sensitive files (block if found)
SENSITIVE=$(echo "$STAGED" | \\
    grep -E '\\.(env|key|pem|p12|pfx)$|credentials\\.json|secrets\\.json|_secret')
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
        "marker": "commit-msg-hook.sh",  # Appears in hook file header
        "fallback_content": None,  # No fallback - must have external file
    },
    "post-commit": {
        "external_file": "ai_template_scripts/post-commit-hook.sh",
        "marker": "post-commit-hook.sh",  # Appears in hook file header
        "fallback_content": None,  # No fallback - must have external file
    },
}


def _load_hook_content(config: dict[str, Any]) -> str | None:
    """Load hook content from external file or return fallback.

    Contracts:
        REQUIRES: config is a dict with optional keys "external_file", "fallback_content"
        ENSURES: Returns content string if external file exists and readable
        ENSURES: Returns fallback_content if external file unavailable
        ENSURES: Returns None if neither available
        ENSURES: Never raises - catches OSError and UnicodeDecodeError

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
            except (OSError, UnicodeDecodeError) as e:
                debug_swallow("load_hook_content", e)  # Fall through to fallback
    return config.get("fallback_content")


def _hook_needs_update(
    hook_path: Path, marker: str, new_content: str
) -> tuple[bool, str]:
    """Check if a hook needs to be installed or updated.

    Contracts:
        REQUIRES: hook_path is a Path object
        REQUIRES: marker is a non-empty string
        REQUIRES: new_content is a non-empty string
        ENSURES: Returns (True, content) if hook missing or needs update
        ENSURES: Returns (True, new_content) if content differs (update needed, #1180)
        ENSURES: Returns (False, existing) if marker appears exactly once AND content matches
        ENSURES: Returns (True, new_content) if marker appears multiple times (corrupted, #918)
        ENSURES: If marker not found, preserves existing user hook by appending
        ENSURES: On read errors, installs fresh content (may overwrite unreadable hook)
        ENSURES: Never raises - catches read errors and installs fresh

    Args:
        hook_path: Path to the git hook file
        marker: String that identifies our hook content
        new_content: The content we want to install

    Returns:
        (needs_update, final_content) - final_content may include existing hook
    """
    try:
        if not hook_path.exists():
            return True, new_content

        existing = hook_path.read_text()

        marker_count = existing.count(marker)
        if marker_count == 1:
            # Our hook is present - check if content matches (#1180)
            # This ensures template updates propagate to installed hooks
            # Note: If hook was appended to existing user content, we compare
            # just the portion containing our marker (the hook itself)
            if new_content.strip() in existing:
                return False, existing
            # Content differs - need to update (will overwrite since marker exists)
            return True, new_content
        if marker_count > 1:
            # Corrupted: multiple copies concatenated (#918) - overwrite
            return True, new_content

        # Marker not found: append to existing user hook (don't replace)
        combined = existing.rstrip() + "\n\n" + new_content
        return True, combined
    except (OSError, UnicodeDecodeError) as e:
        debug_swallow("hook_needs_update_read", e)
        # Can't check/read existing hook, install fresh
        return True, new_content


def install_hooks() -> None:
    """Install git hooks if not present or outdated.

    Hooks are loaded from external files in ai_template_scripts/ when available,
    falling back to inline content. Existing user hooks are preserved by appending
    our hooks rather than replacing.

    Hook installation is idempotent - running multiple times is safe.

    Contracts:
        ENSURES: No-op if not in git repo (.git/ directory missing)
        ENSURES: Creates .git/hooks/ directory if missing (fresh clones)
        ENSURES: Skips hooks with no content available
        ENSURES: Makes installed hooks executable (mode 0o755)
        ENSURES: Prints status for each installed hook
        ENSURES: Never raises - catches and logs OSError on write
    """
    git_dir = Path(".git")
    if not git_dir.exists():
        return  # Not a git repo

    # Create hooks directory if missing (fresh clones may not have it)
    hooks_dir = git_dir / "hooks"
    try:
        hooks_dir.mkdir(exist_ok=True)
    except OSError as e:
        log_warning(f"⚠ Cannot create hooks directory: {e}", stream="stdout")
        return

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
                log_info(f"✓ Installed hook: {name}")
            except OSError as e:
                log_warning(
                    f"⚠ Failed to install hook {name}: {e}",
                    stream="stdout",
                )

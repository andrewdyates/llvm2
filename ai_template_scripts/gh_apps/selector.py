# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/selector.py - Repository to App Mapping

Maps repositories to their corresponding GitHub App.

Tiered lookup (priority order):
1. Per-repo apps: dbx-<reponame-with-dashes>-ai (dedicated per-repo apps)
2. Director apps: dbx-d<DIRECTOR>-ai (shared across director's repos)
3. Wildcard app: dbx-ai (global fallback)
4. Returns None for user sessions (use normal gh auth)
"""

from __future__ import annotations

from ai_template_scripts.gh_apps.config import load_config
from ai_template_scripts.gh_apps.logging import debug_log

# Known director app names (dbx-d<DIRECTOR>-ai pattern)
# These are shared apps covering multiple repos per director
# Explicit list avoids false matches like dbx-dOS-ai (per-repo app for dOS)
KNOWN_DIRECTOR_APPS: set[str] = {
    "dbx-dMATH-ai",
    "dbx-dML-ai",
    "dbx-dLANG-ai",
    "dbx-dTOOL-ai",
    "dbx-dKNOW-ai",
    "dbx-dRS-ai",
    "dbx-dAPP-ai",
    "dbx-dDBX-ai",
}

# Director → repos mapping (source: org_chart.md)
# Per-repo apps override this - only repos WITHOUT dedicated apps use director fallback
DIRECTOR_REPOS: dict[str, list[str]] = {
    "MATH": [
        "z4", "tla2", "gamma-crown",  # Have dedicated apps
        "lean5", "dashprove", "zksolve", "proverif-rs", "galg",  # Use math-ai
    ],
    "ML": ["model_mlx_migration", "voice"],
    "LANG": [
        "zani", "sunder", "certus", "tMIR", "LLVM2", "tRust", "mly",  # Have dedicated apps
        "tSwift", "tC", "rustc-index-verified",  # Use lang-ai
    ],
    "TOOL": [
        "ai_template", "dasher", "dterm", "leadership", "dashnews", "dashboard",  # Have dedicated apps
        "dashterm2", "dterm-alacritty", "dashmap", "codex_dashflow",
        "gemini_cli_rs", "dashflow-integrations", "shared-infra", "dOS",  # Use tool-ai
    ],
    "KNOW": ["sg", "chunker", "video_audio_extracts", "dashextract", "pdfium_fast"],
    "RS": [
        "kafka2", "benchmarker",  # Have dedicated apps
        "claude_code_rs",  # Uses rs-ai
    ],
    "APP": ["dashpresent"],
    "DBX": ["dbx_datacenter", "dbx_unitq"],
}

# Director app names
DIRECTOR_APPS: dict[str, str] = {
    "MATH": "math-ai",
    "ML": "ml-ai",
    "LANG": "lang-ai",
    "TOOL": "tool-ai",
    "KNOW": "know-ai",
    "RS": "rs-ai",
    "APP": "app-ai",
    "DBX": "dbx-ai",
}

# Repos with dedicated per-repo apps (high activity)
DEDICATED_REPOS: set[str] = {
    "ai_template", "z4", "dasher", "zani", "tla2", "gamma-crown",
    "sunder", "certus", "dterm", "kafka2", "leadership", "dashnews",
    "dashboard", "tMIR", "LLVM2", "tRust", "mly", "benchmarker",
}


def get_director_for_repo(repo: str) -> str | None:
    """Get the director that owns a repository.

    Args:
        repo: Repository name.

    Returns:
        Director name (e.g., "MATH", "LANG") or None if not found.
    """
    for director, repos in DIRECTOR_REPOS.items():
        if repo in repos:
            return director
    return None


def get_app_for_repo(repo: str) -> str | None:
    """Get the GitHub App name that handles a given repository.

    Lookup order (priority):
    1. Per-repo apps: dbx-<reponame-with-dashes>-ai (dedicated per-repo apps)
    2. Director apps: dbx-d<DIRECTOR>-ai (shared across director's repos)
    3. Wildcard app: dbx-ai (global fallback)
    4. Returns None for user sessions (use normal gh auth)

    Args:
        repo: Repository name (e.g., "ai_template", "z4").

    Returns:
        App name (e.g., "dbx-ai-template-ai", "dbx-dMATH-ai") or None if not configured.
    """
    debug_log(f"get_app_for_repo({repo})")
    config = load_config()
    if not config:
        debug_log(f"no config available for repo '{repo}'")
        return None

    debug_log(f"searching {len(config.apps)} configured apps")

    # Collect matches by tier
    per_repo_app: str | None = None
    director_app: str | None = None
    wildcard_app: str | None = None

    for app_name, app_config in config.apps.items():
        # Check for wildcard app (Tier 3)
        if "*" in app_config.repos:
            wildcard_app = app_name
            debug_log(f"  found wildcard app: {app_name}")
            continue

        # Check if this app handles the repo
        if repo not in app_config.repos:
            continue

        # Classify by app type
        if app_name in KNOWN_DIRECTOR_APPS:
            # Tier 2: Director app (dbx-d<DIRECTOR>-ai)
            if not director_app:
                director_app = app_name
                debug_log(f"  found director app: {app_name}")
        else:
            # Tier 1: Per-repo app (dbx-<repo>-ai or other)
            if not per_repo_app:
                per_repo_app = app_name
                debug_log(f"  found per-repo app: {app_name}")

    # Return by priority: per-repo > director > wildcard
    if per_repo_app:
        debug_log(f"  selected per-repo app: repo '{repo}' -> app '{per_repo_app}'")
        return per_repo_app
    if director_app:
        debug_log(f"  selected director app: repo '{repo}' -> app '{director_app}'")
        return director_app
    if wildcard_app:
        debug_log(f"  selected wildcard app: repo '{repo}' -> app '{wildcard_app}'")
        return wildcard_app

    debug_log(f"  no app found for repo '{repo}'")
    return None


def list_apps() -> list[str]:
    """List all configured app names.

    Returns:
        List of app names, or empty list if not configured.
    """
    config = load_config()
    if not config:
        return []
    return list(config.apps.keys())


def get_repo_app_mapping() -> dict[str, str]:
    """Get mapping of all explicitly configured repos to their apps.

    Returns:
        Dict mapping repo name to app name. Does not include wildcard matches.
    """
    config = load_config()
    if not config:
        return {}

    mapping: dict[str, str] = {}
    for app_name, app_config in config.apps.items():
        for repo in app_config.repos:
            if repo != "*":
                mapping[repo] = app_name

    return mapping

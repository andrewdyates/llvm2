# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/selector.py - Repository to App Mapping

Maps repositories to their corresponding GitHub App.

Tiered lookup:
1. Per-repo apps (high activity repos) - exact match from config
2. Per-director apps (fallback) - based on director→repo mapping
3. User sessions use normal gh auth (no app injection)
"""

from __future__ import annotations

from ai_template_scripts.gh_apps.config import load_config

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

    Lookup order:
    1. Per-repo apps - exact match (single repo)
    2. Director apps - repo in explicit list
    3. Wildcard apps - repos: ["*"]
    4. Returns None for user sessions (use normal gh auth)

    Args:
        repo: Repository name (e.g., "ai_template", "z4").

    Returns:
        App name (e.g., "ai-template-ai", "dbx-dMATH-ai") or None if not configured.
    """
    config = load_config()
    if not config:
        return None

    wildcard_app: str | None = None

    # Tier 1: Per-repo apps (exact match - single repo)
    for app_name, app_config in config.apps.items():
        # Skip wildcard apps for now - they're fallback
        if "*" in app_config.repos:
            wildcard_app = app_name
            continue
        # Check for exact match
        if repo in app_config.repos:
            return app_name

    # Tier 2: Wildcard fallback
    if wildcard_app:
        return wildcard_app

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

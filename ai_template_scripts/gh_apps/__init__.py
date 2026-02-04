# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps - GitHub Apps for Rate Limit Scaling

This package provides GitHub App authentication for API rate limit isolation.
Each project can have its own GitHub App with independent 5000/hr quota.

Usage:
    # Get installation token for a repo
    from ai_template_scripts.gh_apps import get_token
    token = get_token("ai_template")

    # Or via CLI
    python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template

Architecture:
    - Apps are hosted in dropbox-ai-prototypes org
    - Apps are installed on ayates_dbx user repos
    - Each app gets independent rate limit quota

Configuration:
    ~/.ait_gh_apps/config.yaml - App definitions and mappings
    ~/.ait_gh_apps/<app>-ai.pem - Private keys (600 permissions)
"""

from ai_template_scripts.gh_apps.config import AppConfig, load_config
from ai_template_scripts.gh_apps.selector import get_app_for_repo
from ai_template_scripts.gh_apps.token import TokenManager, get_token

__all__ = [
    "AppConfig",
    "TokenManager",
    "get_app_for_repo",
    "get_token",
    "load_config",
]

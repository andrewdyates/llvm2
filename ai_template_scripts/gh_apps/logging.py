# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/logging.py - Debug logging for GitHub Apps authentication

Enable with: AIT_GH_APPS_DEBUG=1 or AIT_DEBUG=1

Usage:
    from ai_template_scripts.gh_apps.logging import debug_log
    debug_log("token cache hit for ai_template")
"""

from __future__ import annotations

from ai_template_scripts.shared_logging import debug_log as _shared_debug_log

__all__ = ["debug_log"]

# Module-specific debug env var
_MODULE_DEBUG_VAR = "AIT_GH_APPS_DEBUG"


def debug_log(msg: str) -> None:
    """Log debug message if AIT_GH_APPS_DEBUG or AIT_DEBUG is set.

    Args:
        msg: Message to log (prefixed with "gh_apps")
    """
    _shared_debug_log("gh_apps", msg, module_env_var=_MODULE_DEBUG_VAR)

# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/config.py - Configuration Management

Loads and validates GitHub App configuration from ~/.ait_gh_apps/config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_template_scripts.identity import get_identity as _get_ident

if TYPE_CHECKING:
    from typing import Any

# Optional yaml import - graceful degradation if not installed
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]

CONFIG_DIR = Path.home() / ".ait_gh_apps"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def _debug_log(msg: str) -> None:
    """Debug log - imported lazily to avoid circular imports."""
    try:
        from ai_template_scripts.gh_apps.logging import debug_log

        debug_log(msg)
    except ImportError:
        pass  # Module not yet available during early imports


@dataclass
class AppConfig:
    """Configuration for a single GitHub App."""

    name: str
    app_id: int
    installation_id: int
    private_key_path: Path
    repos: list[str] = field(default_factory=list)

    @property
    def private_key(self) -> str:
        """Read private key from file."""
        if not self.private_key_path.exists():
            raise FileNotFoundError(
                f"Private key not found: {self.private_key_path}\n"
                f"Run: python3 -m ai_template_scripts.gh_apps.setup create-app <project>"
            )
        return self.private_key_path.read_text()

    def matches_repo(self, repo: str) -> bool:
        """Check if this app handles the given repo."""
        if "*" in self.repos:
            return True
        return repo in self.repos


@dataclass
class Config:
    """Full configuration for GitHub Apps."""

    org: str
    default_app: str
    apps: dict[str, AppConfig] = field(default_factory=dict)

    def get_app(self, name: str) -> AppConfig | None:
        """Get app config by name."""
        return self.apps.get(name)


_cached_config: Config | None = None
_config_mtime: float = 0


def load_config(force_reload: bool = False) -> Config | None:
    """Load configuration from file.

    Args:
        force_reload: If True, ignore cache and reload from disk.

    Returns:
        Config object or None if config doesn't exist or yaml not installed.
    """
    global _cached_config, _config_mtime

    if not YAML_AVAILABLE:
        _debug_log("yaml not available - cannot load config")
        return None

    if not CONFIG_FILE.exists():
        _debug_log(f"config file not found: {CONFIG_FILE}")
        return None

    # Check if config changed since last load
    current_mtime = CONFIG_FILE.stat().st_mtime
    if not force_reload and _cached_config and current_mtime == _config_mtime:
        _debug_log(f"using cached config ({len(_cached_config.apps)} apps)")
        return _cached_config

    _debug_log(f"loading config from {CONFIG_FILE}")

    try:
        data: dict[str, Any] = yaml.safe_load(CONFIG_FILE.read_text())
        if not data:
            _debug_log("config file is empty")
            return None

        apps: dict[str, AppConfig] = {}
        for app_name, app_data in data.get("apps", {}).items():
            # Handle both single repo and multiple repos
            repos: list[str] = []
            if "repo" in app_data:
                repos = [app_data["repo"]]
            elif "repos" in app_data:
                repos = app_data["repos"]

            # Expand ~ in private_key path
            key_path = Path(os.path.expanduser(app_data["private_key"]))

            apps[app_name] = AppConfig(
                name=app_name,
                app_id=app_data["app_id"],
                installation_id=app_data["installation_id"],
                private_key_path=key_path,
                repos=repos,
            )
            _debug_log(f"  loaded app '{app_name}': repos={repos}, key={key_path}")

        config = Config(
            org=data.get("org", _get_ident().github_org),
            default_app=data.get("default_app", "shared-ai"),
            apps=apps,
        )

        _debug_log(f"config loaded: {len(apps)} apps configured")
        _cached_config = config
        _config_mtime = current_mtime
        return config

    except Exception as e:
        print(f"gh_apps: failed to load config: {e}")
        return None


def get_config_dir() -> Path:
    """Get the configuration directory, creating if needed."""
    CONFIG_DIR.mkdir(mode=0o700, exist_ok=True)
    return CONFIG_DIR

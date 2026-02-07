# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Identity configuration loader for ai_template.

Reads identity from ait_identity.toml at the repo root. Falls back to
placeholder defaults when the config file is absent or incomplete.

Usage:
    from ai_template_scripts.identity import get_identity
    ident = get_identity()
    print(ident.owner_name)       # "Andrew Yates"
    print(ident.github_org)       # "dropbox-ai-prototypes"
    print(ident.copyright_holder) # "Dropbox, Inc."

See: #2974 (identity extraction for public release)
"""

__all__ = ["Identity", "get_identity", "load_identity"]

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Identity:
    """Loaded identity configuration."""

    owner_name: str
    owner_email: str
    owner_usernames: list[str]
    github_org: str
    company_name: str
    company_abbreviation: str
    copyright_year: int
    copyright_holder: str
    copyright_license: str


_PLACEHOLDER = Identity(
    owner_name="Andrew Yates",
    owner_email="ayates@dropbox.com",
    owner_usernames=["ayates", "andrewdyates", "ayates_dbx"],
    github_org="dropbox-ai-prototypes",
    company_name="Dropbox, Inc.",
    company_abbreviation="DBX",
    copyright_year=2026,
    copyright_holder="Dropbox, Inc.",
    copyright_license="Apache-2.0",
)

_CONFIG_FILENAME = "ait_identity.toml"

_cached: Optional[Identity] = None


def _find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from start to find the repo root containing ait_identity.toml or .git."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / _CONFIG_FILENAME).exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return None


def load_identity(repo_root: Optional[Path] = None) -> Identity:
    """Load identity from ait_identity.toml. Returns placeholders if absent.

    Args:
        repo_root: Explicit repo root path. If None, auto-detects by walking
                   up from cwd looking for ait_identity.toml or .git.

    Returns:
        Identity dataclass with loaded or placeholder values.
    """
    if repo_root is None:
        repo_root = _find_repo_root()
    if repo_root is None:
        return _PLACEHOLDER

    config_path = repo_root / _CONFIG_FILENAME
    if not config_path.exists():
        return _PLACEHOLDER

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return _PLACEHOLDER

    owner = data.get("owner", {})
    org = data.get("org", {})
    copyright_ = data.get("copyright", {})

    return Identity(
        owner_name=str(owner.get("name", _PLACEHOLDER.owner_name)),
        owner_email=str(owner.get("email", _PLACEHOLDER.owner_email)),
        owner_usernames=_as_str_list(owner.get("usernames", _PLACEHOLDER.owner_usernames)),
        github_org=str(org.get("github_org", _PLACEHOLDER.github_org)),
        company_name=str(org.get("company_name", _PLACEHOLDER.company_name)),
        company_abbreviation=str(org.get("abbreviation", _PLACEHOLDER.company_abbreviation)),
        copyright_year=_as_int(copyright_.get("year", _PLACEHOLDER.copyright_year)),
        copyright_holder=str(copyright_.get("holder", _PLACEHOLDER.copyright_holder)),
        copyright_license=str(copyright_.get("license", _PLACEHOLDER.copyright_license)),
    )


def get_identity() -> Identity:
    """Get cached identity. Loads once on first call."""
    global _cached
    if _cached is None:
        _cached = load_identity()
    return _cached


def _as_str_list(value: object) -> list[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def _as_int(value: object) -> int:
    """Coerce a value to int with fallback."""
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _PLACEHOLDER.copyright_year

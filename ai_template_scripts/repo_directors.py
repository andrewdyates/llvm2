#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Repo to Director mapping derived from .claude/rules/org_chart.md."""

from __future__ import annotations

__all__ = ["load_repo_to_director", "get_director"]

from pathlib import Path

_REPO_TO_DIRECTOR_CACHE: dict[str, str] | None = None


def _default_org_chart_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / ".claude" / "rules" / "org_chart.md"


def load_repo_to_director(org_chart_path: Path | None = None) -> dict[str, str]:
    """Parse org_chart.md and return repo -> director mapping."""
    global _REPO_TO_DIRECTOR_CACHE

    if org_chart_path is None and _REPO_TO_DIRECTOR_CACHE is not None:
        return dict(_REPO_TO_DIRECTOR_CACHE)

    path = org_chart_path or _default_org_chart_path()
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}

    mapping: dict[str, str] = {}
    for line in lines:
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        director, _, repos_cell = cells[:3]
        if director.lower() == "director" or set(director) <= {"-"}:
            continue
        if not repos_cell or repos_cell.startswith("("):
            continue
        for repo in repos_cell.split(","):
            repo_name = repo.strip()
            if repo_name:
                mapping[repo_name] = director

    if org_chart_path is None:
        _REPO_TO_DIRECTOR_CACHE = mapping

    return dict(mapping)


def get_director(repo: str, default: str = "TOOL") -> str:
    """Return director for repo, defaulting to TOOL."""
    if not repo:
        return default
    mapping = load_repo_to_director()
    return mapping.get(repo, default)

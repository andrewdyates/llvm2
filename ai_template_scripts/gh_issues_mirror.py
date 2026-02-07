#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_issues_mirror.py - Dump GitHub issues to local markdown for offline search.

PURPOSE: Create local searchable copy of GitHub issues for offline grep/rg.
CALLED BY: looper/context (refresh_if_stale), human debugging
REFERENCED: Uses markdown_to_issues.export_issues for export format.

Creates a gitignored issue mirror for quick local grep/rg search. Uses the
existing markdown export format from markdown_to_issues.py.

Public API (library usage):
    from ai_template_scripts.gh_issues_mirror import (
        DEFAULT_OUT_DIR,       # Default output directory (.issues)
        DEFAULT_STATE,         # Default issue state filter (open)
        DEFAULT_MAX_AGE_HOURS, # Default staleness threshold (24h)
        MirrorExportError,     # Exception for export failures
        write_issue_mirror,    # Write mirror file (raises on failure)
        refresh_if_stale,      # Refresh if older than threshold (best-effort)
        main,                  # CLI entry point
    )

CLI usage:
    ./gh_issues_mirror.py                        # Refresh if stale
    ./gh_issues_mirror.py --state open           # Filter by state
    ./gh_issues_mirror.py --force                # Force refresh
    ./gh_issues_mirror.py --stdout               # Print to stdout
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

try:
    from ai_template_scripts.atomic_write import atomic_write_text
    from ai_template_scripts.markdown_to_issues import export_issues
    from ai_template_scripts.subprocess_utils import get_git_root as _get_git_root_or_none
    from ai_template_scripts.version import get_version
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai_template_scripts.atomic_write import atomic_write_text
    from ai_template_scripts.markdown_to_issues import export_issues
    from ai_template_scripts.subprocess_utils import get_git_root as _get_git_root_or_none
    from ai_template_scripts.version import get_version

__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFAULT_STATE",
    "DEFAULT_MAX_AGE_HOURS",
    "MirrorExportError",
    "write_issue_mirror",
    "refresh_if_stale",
    "main",
]

DEFAULT_OUT_DIR = ".issues"
DEFAULT_STATE = "open"  # Changed from "all" - most searches are for open issues (#1380)
DEFAULT_MAX_AGE_HOURS = 24


class MirrorExportError(Exception):
    """Raised when issue export fails (e.g., gh command failure)."""


def get_git_root() -> Path:
    """Return repo root or raise RuntimeError.

    Thin wrapper around subprocess_utils.get_git_root() (#2535).
    """
    root = _get_git_root_or_none()
    if root is None:
        raise RuntimeError("git rev-parse failed")
    return root


def slugify(value: str) -> str:
    """Normalize label text for filenames.

    REQUIRES: value is string
    ENSURES: Returns lowercase alphanumeric string with hyphens, or "label" if empty
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "label"


def normalize_label(label: str | None) -> str | None:
    """Normalize label value from CLI; empty strings become None.

    REQUIRES: label is string or None
    ENSURES: Returns stripped non-empty string or None
    """
    if label is None:
        return None
    trimmed = label.strip()
    return trimmed or None


def get_output_path(out_dir: Path, state: str, label: str | None) -> Path:
    """Build output path for issue mirror file.

    REQUIRES: out_dir is valid Path, state is valid state string
    ENSURES: Returns Path in format: out_dir/issues-{state}[-{label}].md
    """
    suffix = f"-{slugify(label)}" if label else ""
    return out_dir / f"issues-{state}{suffix}.md"


def is_stale(path: Path, max_age_hours: float) -> bool:
    """Return True if file missing or older than max_age_hours.

    REQUIRES: path is valid Path
    ENSURES: Returns True if max_age_hours <= 0 OR file missing OR age >= max_age_hours
    """
    if max_age_hours <= 0:
        return True
    if not path.exists():
        return True
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds >= max_age_hours * 3600


def write_issue_mirror(
    state: str,
    label: str | None = None,
    out_dir: str = DEFAULT_OUT_DIR,
) -> Path:
    """Write issue mirror file and return its path.

    REQUIRES: state is valid issue state, out_dir is valid path
    ENSURES: Returns Path to written file OR raises MirrorExportError

    Raises:
        MirrorExportError: If gh command fails (preserves existing cache).
    """
    repo_root = get_git_root()
    output_dir = repo_root / out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_label = normalize_label(label)
    output_path = get_output_path(output_dir, state, normalized_label)

    content = export_issues(state, normalized_label)
    if content is None:
        raise MirrorExportError("gh issue list failed; existing cache preserved")
    atomic_write_text(output_path, content.rstrip() + "\n")
    return output_path


def refresh_if_stale(
    state: str = DEFAULT_STATE,
    label: str | None = None,
    out_dir: str = DEFAULT_OUT_DIR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> Path | None:
    """Refresh mirror if stale; return updated path or None.

    REQUIRES: state is valid issue state, max_age_hours >= 0
    ENSURES: Returns Path if refreshed, None if fresh or on failure (best-effort)

    Best-effort: returns None (preserves existing cache) if gh fails.
    """
    repo_root = get_git_root()
    output_dir = repo_root / out_dir
    normalized_label = normalize_label(label)
    output_path = get_output_path(output_dir, state, normalized_label)

    if not is_stale(output_path, max_age_hours):
        return None

    try:
        return write_issue_mirror(state=state, label=normalized_label, out_dir=out_dir)
    except MirrorExportError:
        # Best-effort: preserve existing cache on failure
        return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments.

    REQUIRES: argv is list of strings
    ENSURES: Returns Namespace with parsed CLI arguments
    """
    parser = argparse.ArgumentParser(
        description="Dump GitHub issues to local markdown for offline search."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("gh_issues_mirror.py"),
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE,
        choices=["open", "closed", "all"],
        help="Issue state to export (default: open).",
    )
    parser.add_argument("--label", help="Filter issues by label.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help="Refresh only if older than this many hours (default: 24).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even if the mirror is still fresh.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print markdown to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress status output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """CLI entry point.

    REQUIRES: argv is list of CLI argument strings
    ENSURES: Returns 0 on success, 1 on failure; writes mirror or prints to stdout
    """
    args = parse_args(argv)
    normalized_label = normalize_label(args.label)

    try:
        if args.stdout:
            content = export_issues(args.state, normalized_label)
            if content is None:
                print("Error: gh issue list failed", file=sys.stderr)
                return 1
            sys.stdout.write(content.rstrip() + "\n")
            return 0

        path: Path | None
        if args.force or args.max_age_hours <= 0:
            path = write_issue_mirror(
                state=args.state, label=normalized_label, out_dir=args.out_dir
            )
            if not args.quiet:
                print(f"Wrote issue mirror: {path}")
            return 0

        path = refresh_if_stale(
            state=args.state,
            label=normalized_label,
            out_dir=args.out_dir,
            max_age_hours=args.max_age_hours,
        )
        if not args.quiet:
            if path is None:
                print("Issue mirror is up to date.")
            else:
                print(f"Wrote issue mirror: {path}")
        return 0
    except Exception as exc:
        if not args.quiet:
            print(f"Failed to write issue mirror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

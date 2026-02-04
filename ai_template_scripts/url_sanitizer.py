#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Git URL sanitizer - removes credentials and sensitive URL components.

PURPOSE: Sanitize git remote URLs before logging, persistence, or display.
CALLED BY: Shell scripts (CLI), Python modules (import)

Policy for persistence-safe URLs:
- Absolute URLs (https://...): preserve scheme, host, port, path; drop userinfo, query, fragment
- SCP-like URLs (git@github.com:org/repo): drop user@ prefix
- Unknown/local/file URLs: return error marker

CLI usage:
    python3 -m ai_template_scripts.url_sanitizer <url>
    python3 -m ai_template_scripts.url_sanitizer --repo-slug <url>

Module usage:
    from ai_template_scripts.url_sanitizer import sanitize_git_url, extract_repo_slug
"""

import re
import sys
from urllib.parse import urlparse

__all__ = [
    "sanitize_git_url",
    "extract_repo_slug",
]

# SCP-style git URL pattern: [user@]host:path
_SCP_URL_PATTERN = re.compile(r"^(?:([^@]+)@)?([^:]+):(.+)$")

# Windows drive letter pattern: C:\ or C:/
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def sanitize_git_url(url: str) -> str | None:
    """Sanitize git remote URL by removing credentials and sensitive components.

    Removes:
    - userinfo (username:password@)
    - query string (?token=xxx)
    - fragment (#ref)

    Args:
        url: Raw git remote URL (may contain credentials)

    Returns:
        Sanitized URL safe for logging/persistence, or None if format unknown

    Examples:
        >>> sanitize_git_url("https://user:pass@github.com/org/repo.git?token=xxx")
        'https://github.com/org/repo.git'
        >>> sanitize_git_url("git@github.com:org/repo.git")
        'github.com:org/repo.git'
        >>> sanitize_git_url("/local/path")
        None
    """
    if not url or not url.strip():
        return None

    url = url.strip()

    # Try parsing as standard URL (https://, ssh://, git://, etc.)
    parsed = urlparse(url)

    # Reject file:// and other local schemes
    if parsed.scheme in ("file",):
        return None

    if parsed.scheme and parsed.netloc:
        # Absolute URL - reconstruct without userinfo/query/fragment
        # Note: parsed.netloc may contain userinfo, so extract host:port only
        host = parsed.hostname or ""
        # IPv6 addresses need brackets in URLs (urlparse.hostname strips them)
        if ":" in host:
            host = f"[{host}]"
        port_suffix = f":{parsed.port}" if parsed.port else ""
        path = parsed.path or ""
        return f"{parsed.scheme}://{host}{port_suffix}{path}"

    # Reject Windows drive letter paths (C:\...) before SCP-like parsing
    # They would otherwise match the SCP pattern with host="C"
    if _WINDOWS_DRIVE_PATTERN.match(url):
        return None

    # Try SCP-like format: [user@]host:path
    scp_match = _SCP_URL_PATTERN.match(url)
    if scp_match:
        # user group is scp_match.group(1), host is group(2), path is group(3)
        host = scp_match.group(2)
        path = scp_match.group(3)
        # Strip query (?...) and fragment (#...) from path to avoid leaking tokens
        # e.g., "org/repo.git?access_token=abc#frag" -> "org/repo.git"
        if "?" in path:
            path = path.split("?", 1)[0]
        if "#" in path:
            path = path.split("#", 1)[0]
        return f"{host}:{path}"

    # Unknown format (local path, etc.) - don't persist
    return None


def extract_repo_slug(url: str) -> str | None:
    """Extract owner/repo slug from git URL.

    Args:
        url: Git remote URL (raw or sanitized)

    Returns:
        Repo slug like "owner/repo", or None if cannot extract

    Examples:
        >>> extract_repo_slug("https://github.com/ayates_dbx/ai_template.git")
        'ayates_dbx/ai_template'
        >>> extract_repo_slug("git@github.com:ayates_dbx/ai_template.git")
        'ayates_dbx/ai_template'
    """
    if not url or not url.strip():
        return None

    url = url.strip()

    # First sanitize to remove credentials
    sanitized = sanitize_git_url(url)
    if not sanitized:
        return None

    # Extract path portion
    parsed = urlparse(sanitized)
    if parsed.scheme and parsed.netloc:
        # Absolute URL
        path = parsed.path.lstrip("/")
    else:
        # SCP-like: host:path
        scp_match = _SCP_URL_PATTERN.match(sanitized)
        if scp_match:
            path = scp_match.group(3)
        else:
            return None

    # Remove .git suffix and clean up
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/")

    # Validate looks like owner/repo
    if "/" in path and not path.startswith("/"):
        return path

    return None


def main() -> int:
    """CLI entry point for shell script integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanitize git URLs for safe logging/persistence"
    )
    parser.add_argument("url", help="Git remote URL to sanitize")
    parser.add_argument(
        "--repo-slug",
        action="store_true",
        help="Extract owner/repo slug instead of full URL",
    )

    args = parser.parse_args()

    if args.repo_slug:
        result = extract_repo_slug(args.url)
    else:
        result = sanitize_git_url(args.url)

    if result is None:
        print("[REDACTED]", file=sys.stdout)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

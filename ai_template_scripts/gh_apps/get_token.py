# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/get_token.py - CLI to get installation tokens

Usage:
    python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template
    python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template --overflow

The token is printed to stdout for use in shell scripts.
With --overflow, checks rate limits and picks the best available app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_template_scripts.gh_apps.token import get_token, _token_manager, TokenManager


def _find_real_gh() -> str:
    """Find the real gh binary path."""
    import shutil

    # Try common locations
    for path in [
        shutil.which("gh"),
        "/opt/homebrew/bin/gh",  # macOS ARM
        "/usr/local/bin/gh",     # macOS Intel / Linux
        "/usr/bin/gh",           # Linux package manager
    ]:
        if path and Path(path).exists():
            return path
    return "gh"  # Fall back to PATH lookup


def _get_token_with_overflow(repo: str) -> tuple[str | None, str | None]:
    """Get token using overflow logic - pick best available app.

    Falls back to regular get_token if overflow check fails.

    Returns:
        Tuple of (token, app_name) or (None, None) if all exhausted.
    """
    try:
        from ai_template_scripts.gh_rate_limit.rate_state import RateState

        # Create RateState to check cached quotas
        cache_dir = Path.home() / ".ait_gh_cache"
        cache_dir.mkdir(exist_ok=True)

        state = RateState(
            cache_dir=cache_dir,
            get_real_gh=_find_real_gh,
            get_commit_hash=lambda: None,
        )

        # Get best available app based on graphql quota (most commonly exhausted)
        best_app = state.get_best_available_app("graphql", repo)
        if not best_app:
            return None, None

        # Get token for that app
        global _token_manager
        if _token_manager is None:
            _token_manager = TokenManager()

        token = _token_manager.get_token(best_app, repo=repo)
        return token, best_app

    except Exception as e:
        # Fall back to regular token selection on any error
        print(f"gh_apps: overflow check failed ({e}), using primary app", file=sys.stderr)
        token = get_token(repo)
        if token:
            from ai_template_scripts.gh_apps.selector import get_app_for_repo
            return token, get_app_for_repo(repo)
        return None, None


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Get GitHub App installation token for a repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get token for ai_template repo (primary app only)
  python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template

  # Get token with overflow (picks best available app)
  python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template --overflow

  # Token is printed to stdout for piping to GH_TOKEN env var
""",
    )
    parser.add_argument(
        "--repo",
        "-r",
        required=True,
        help="Repository name (e.g., ai_template)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress error messages, exit silently on failure",
    )
    parser.add_argument(
        "--overflow",
        action="store_true",
        help="Enable overflow: pick best available app based on rate limits",
    )

    args = parser.parse_args()

    if args.overflow:
        token, app = _get_token_with_overflow(args.repo)
        if token:
            print(token)
            if not args.quiet:
                print(f"gh_apps: using {app}", file=sys.stderr)
        else:
            if not args.quiet:
                print(
                    f"gh_apps: all apps exhausted for {args.repo}",
                    file=sys.stderr,
                )
            sys.exit(1)
    else:
        token = get_token(args.repo)
        if token:
            print(token)
        else:
            if not args.quiet:
                print(
                    f"gh_apps: no token available for {args.repo}",
                    file=sys.stderr,
                )
            sys.exit(1)


if __name__ == "__main__":
    main()

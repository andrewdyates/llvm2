# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/get_token.py - CLI to get installation tokens

Usage:
    python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template

The token is printed to stdout for use in shell scripts.
"""

from __future__ import annotations

import argparse
import sys

from ai_template_scripts.gh_apps.token import get_token


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Get GitHub App installation token for a repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get token for ai_template repo
  python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template

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

    args = parser.parse_args()

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

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
bot_token.py - GitHub App installation token management for AI fleet

Fetches GitHub App installation tokens for bot identity.
Part of the two-layer bot identity system.

Public API (library usage):
    from ai_template_scripts.bot_token import (
        get_config_dir,          # Get ai-fleet config directory
        get_project_name,        # Get project name from git remote
        load_credentials,        # Load app_id and private key
        get_jwt,                 # Generate JWT for GitHub App auth
        get_installation_id,     # Get installation ID for repo
        get_installation_token,  # Get installation access token
    )

CLI usage:
    ./bot_token.py                    # Print token for current repo
    ./bot_token.py --json             # Print JSON with token + expiry
    ./bot_token.py --project z4       # Explicit project
    ./bot_token.py --check            # Check if credentials exist

Environment:
    AI_FLEET_CONFIG_DIR  - Override ~/.config/ai-fleet

Credential structure (in config dir):
    apps.json         - Maps project name to app_id: {"z4": 123456}
    keys/{project}.pem - Private key for each app

Module contracts:
    ENSURES: All file paths returned are absolute
    ENSURES: JWT tokens follow GitHub App authentication format (RS256)
    ENSURES: API errors propagate as RuntimeError or requests.HTTPError
"""

from __future__ import annotations

__all__ = [
    "get_config_dir",
    "get_project_name",
    "load_credentials",
    "get_jwt",
    "get_installation_id",
    "get_installation_token",
    "main",
]

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jwt  # PyJWT - requires cryptography for RS256  # noqa: E402
import requests  # noqa: E402

from ai_template_scripts.subprocess_utils import (  # noqa: E402
    get_repo_name,
    run_cmd,
)
from ai_template_scripts.version import get_version  # noqa: E402


def get_config_dir() -> Path:
    """Get ai-fleet config directory.

    Contracts:
        ENSURES: Returns absolute Path
        ENSURES: Respects AI_FLEET_CONFIG_DIR env var if set
        ENSURES: Default is ~/.config/ai-fleet
    """
    return Path(
        os.environ.get("AI_FLEET_CONFIG_DIR", Path.home() / ".config" / "ai-fleet"),
    )


def get_project_name() -> str:
    """Get project name from git remote.

    Contracts:
        REQUIRES: Current directory is in a git repo with origin remote
        ENSURES: Returns repo name without .git suffix
        ENSURES: Handles both SSH (git@...) and HTTPS (https://...) URLs
        ENSURES: Raises RuntimeError if not in git repo

    Raises:
        RuntimeError: If not in a git repo or no origin remote.
    """
    # First verify we're in a git repo with an origin remote (strict mode)
    check = run_cmd(["git", "remote", "get-url", "origin"], timeout=10)
    if not check.ok:
        raise RuntimeError("Not in a git repo or no origin remote")

    # Use canonical implementation for consistent URL parsing
    result = get_repo_name()
    return result.stdout


def normalize_project_name(project: str) -> str:
    """Normalize project name for app lookup (underscore to hyphen).

    Contracts:
        REQUIRES: project is a non-empty string
        ENSURES: Returns project with underscores replaced by hyphens
    """
    return project.replace("_", "-")


def load_credentials(project: str) -> tuple[int, str]:
    """Load app_id and private key for project.

    Contracts:
        REQUIRES: project is a non-empty string
        REQUIRES: config_dir contains apps.json with project entry
        REQUIRES: config_dir/keys/{project}.pem exists
        ENSURES: Returns (app_id: int, private_key: str)
        ENSURES: app_id is a positive integer
        ENSURES: private_key is a PEM-encoded RSA key string

    Args:
        project: Project name (e.g., "z4", "ai_template")

    Returns:
        Tuple of (app_id, private_key_pem)

    Raises:
        FileNotFoundError: If apps.json or key file not found
        KeyError: If project not in apps.json
    """
    config_dir = get_config_dir()

    apps_file = config_dir / "apps.json"
    if not apps_file.exists():
        raise FileNotFoundError(f"No apps.json found at {apps_file}")

    apps = json.loads(apps_file.read_text())

    # Try both underscore and hyphen variants
    normalized = normalize_project_name(project)
    app_id = apps.get(project) or apps.get(normalized)
    if not app_id:
        raise KeyError(f"No app_id for project '{project}' in apps.json")

    # Try both naming conventions for key file
    key_file = config_dir / "keys" / f"{project}.pem"
    if not key_file.exists():
        key_file = config_dir / "keys" / f"{normalized}.pem"
    if not key_file.exists():
        raise FileNotFoundError(
            f"No private key found for {project} at {config_dir}/keys/"
        )

    return app_id, key_file.read_text()


def get_jwt(app_id: int, private_key: str) -> str:
    """Generate JWT for GitHub App authentication.

    Contracts:
        REQUIRES: app_id > 0
        REQUIRES: private_key is a valid PEM-encoded RSA key
        ENSURES: Returns RS256-signed JWT string
        ENSURES: iat <= now <= exp (with 60s clock skew buffer)
        ENSURES: exp - iat == 660s (11 minutes total: -60s skew + 10min validity)

    Args:
        app_id: GitHub App ID
        private_key: PEM-encoded private key

    Returns:
        JWT token string
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,  # Issued 60s ago (clock skew buffer)
        "exp": now + (10 * 60),  # Expires in 10 minutes
        "iss": str(app_id),  # PyJWT requires iss to be a string
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def get_installation_id(jwt_token: str, owner: str, repo: str) -> int:
    """Get installation ID for a repository.

    Contracts:
        REQUIRES: jwt_token is a valid GitHub App JWT
        REQUIRES: owner and repo are non-empty strings
        ENSURES: Returns positive integer installation ID
        ENSURES: Raises RuntimeError if app not installed

    Args:
        jwt_token: JWT for app authentication
        owner: Repository owner (e.g., "ayates_dbx")
        repo: Repository name (e.g., "z4")

    Returns:
        Installation ID

    Raises:
        RuntimeError: If app not installed on repo
        requests.HTTPError: On API errors
    """
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/installation",
        headers=headers,
        timeout=30,
    )

    if resp.status_code == 404:
        raise RuntimeError(f"App not installed on {owner}/{repo}")
    resp.raise_for_status()

    return resp.json()["id"]


def get_installation_token(jwt_token: str, installation_id: int) -> dict:
    """Get installation access token.

    Contracts:
        REQUIRES: jwt_token is a valid GitHub App JWT
        REQUIRES: installation_id > 0
        ENSURES: Returns dict with "token" and "expires_at" keys
        ENSURES: "token" is a non-empty string
        ENSURES: "expires_at" is an ISO 8601 timestamp

    Args:
        jwt_token: JWT for app authentication
        installation_id: Installation ID from get_installation_id

    Returns:
        Dict with "token" and "expires_at" keys
    """
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    resp = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()

    return resp.json()  # {"token": "...", "expires_at": "..."}


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Contracts:
        ENSURES: Returns 0 on success, 1 on error
        ENSURES: Never raises - catches all exceptions

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 success, 1 error)
    """
    parser = argparse.ArgumentParser(
        description="Fetch GitHub App installation token for bot identity",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("bot_token.py"),
    )
    parser.add_argument(
        "--project",
        help="Project name (default: from git remote)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON with token + expiry",
    )
    parser.add_argument(
        "--owner",
        default="ayates_dbx",
        help="GitHub owner (default: ayates_dbx)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if credentials exist (exit 0 if yes, 1 if no)",
    )

    parsed = parser.parse_args(args)

    project = parsed.project or get_project_name()

    if parsed.check:
        try:
            load_credentials(project)
            print(f"Credentials found for {project}")
            return 0
        except (FileNotFoundError, KeyError) as e:
            print(f"No credentials: {e}", file=sys.stderr)
            return 1

    try:
        app_id, private_key = load_credentials(project)
        jwt_token = get_jwt(app_id, private_key)
        installation_id = get_installation_id(jwt_token, parsed.owner, project)
        token_data = get_installation_token(jwt_token, installation_id)

        if parsed.json:
            token_data["app_id"] = app_id
            token_data["project"] = project
            token_data["installation_id"] = installation_id
            print(json.dumps(token_data))
        else:
            print(token_data["token"])

        return 0

    except FileNotFoundError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"GitHub App error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Catch-all: unexpected error in CLI entry point
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/token.py - JWT and Installation Token Management

Handles:
1. JWT generation (10 min validity) signed with app private key
2. JWT exchange for installation token (1 hr validity)
3. Token caching with automatic refresh 5 min before expiry
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jwt
import requests

from ai_template_scripts.gh_apps.config import AppConfig, load_config
from ai_template_scripts.gh_apps.logging import debug_log

if TYPE_CHECKING:
    pass

# Token cache file
CACHE_DIR = Path.home() / ".ait_gh_apps"
TOKEN_CACHE_FILE = CACHE_DIR / "token_cache.json"

# GitHub API base URL
GITHUB_API = "https://api.github.com"


@dataclass
class CachedToken:
    """Cached installation token."""

    token: str
    expires_at: float  # Unix timestamp


class TokenManager:
    """Manages GitHub App installation tokens.

    Tokens are cached to disk and refreshed 5 minutes before expiry.
    """

    def __init__(self) -> None:
        self._token_cache: dict[str, CachedToken] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load token cache from disk."""
        if not TOKEN_CACHE_FILE.exists():
            debug_log("no token cache file found")
            return
        try:
            data = json.loads(TOKEN_CACHE_FILE.read_text())
            for app_name, token_data in data.items():
                self._token_cache[app_name] = CachedToken(
                    token=token_data["token"],
                    expires_at=token_data["expires_at"],
                )
            debug_log(f"loaded {len(self._token_cache)} cached tokens from disk")
        except Exception as e:
            debug_log(f"failed to load token cache: {e}")
            pass

    def _save_cache(self) -> None:
        """Save token cache to disk."""
        CACHE_DIR.mkdir(mode=0o700, exist_ok=True)
        data = {
            app_name: {"token": ct.token, "expires_at": ct.expires_at}
            for app_name, ct in self._token_cache.items()
        }
        TOKEN_CACHE_FILE.write_text(json.dumps(data))
        TOKEN_CACHE_FILE.chmod(0o600)

    def _needs_refresh(self, cached: CachedToken) -> bool:
        """Check if token needs refresh (expires in < 5 minutes)."""
        return time.time() > cached.expires_at - 300

    def _generate_jwt(self, app_config: AppConfig) -> str:
        """Generate JWT for GitHub App authentication.

        JWTs are valid for 10 minutes and used to request installation tokens.
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60s ago (clock skew buffer)
            "exp": now + 600,  # Expires in 10 minutes
            "iss": str(app_config.app_id),
        }
        return jwt.encode(payload, app_config.private_key, algorithm="RS256")

    def _exchange_for_installation_token(
        self, app_config: AppConfig, jwt_token: str, repo: str | None = None
    ) -> CachedToken:
        """Exchange JWT for installation access token.

        Installation tokens are valid for 1 hour.

        Args:
            app_config: The app configuration.
            jwt_token: JWT for authentication.
            repo: Optional repo name to scope token to (least privilege).
        """
        url = f"{GITHUB_API}/app/installations/{app_config.installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Request repo-scoped token if repo provided (least privilege)
        body: dict | None = None
        if repo:
            body = {"repositories": [repo]}

        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()

        data = response.json()
        # Parse ISO timestamp to Unix timestamp
        expires_at_str = data["expires_at"]  # e.g., "2024-01-01T12:00:00Z"
        # Simple parsing - GitHub always returns UTC
        from datetime import datetime

        expires_at = datetime.fromisoformat(
            expires_at_str.replace("Z", "+00:00")
        ).timestamp()

        return CachedToken(token=data["token"], expires_at=expires_at)

    def get_token(self, app_name: str, repo: str | None = None) -> str | None:
        """Get valid installation token for an app.

        Args:
            app_name: Name of the GitHub App (e.g., "z4-ai").
            repo: Optional repo name to scope token to (least privilege).
                  Director apps should always pass repo to avoid cross-repo reuse.

        Returns:
            Installation token string, or None if app not configured.
        """
        debug_log(f"get_token called: app_name={app_name}, repo={repo}")

        config = load_config()
        if not config:
            debug_log("no config loaded - falling back to OAuth")
            return None

        app_config = config.get_app(app_name)
        if not app_config:
            debug_log(f"app '{app_name}' not found in config - available: {list(config.apps.keys())}")
            return None

        debug_log(f"found app config: app_id={app_config.app_id}, installation_id={app_config.installation_id}")

        # Cache key includes repo for repo-scoped tokens (no cross-repo reuse)
        cache_key = f"{app_name}|{repo}" if repo else app_name

        # Check cache first
        cached = self._token_cache.get(cache_key)
        if cached and not self._needs_refresh(cached):
            ttl = int(cached.expires_at - time.time())
            debug_log(f"cache HIT for {cache_key} (expires in {ttl}s)")
            return cached.token

        if cached:
            debug_log(f"cache STALE for {cache_key} - refreshing")
        else:
            debug_log(f"cache MISS for {cache_key} - generating new token")

        # Generate new token
        try:
            debug_log(f"generating JWT for app_id={app_config.app_id}")
            jwt_token = self._generate_jwt(app_config)
            debug_log(f"exchanging JWT for installation token (installation_id={app_config.installation_id})")
            cached_token = self._exchange_for_installation_token(
                app_config, jwt_token, repo
            )
            self._token_cache[cache_key] = cached_token
            self._save_cache()
            ttl = int(cached_token.expires_at - time.time())
            debug_log(f"token obtained successfully, expires in {ttl}s")
            return cached_token.token
        except FileNotFoundError as e:
            debug_log(f"FileNotFoundError: {e}")
            print(f"gh_apps: {e}", file=sys.stderr)
            return None
        except requests.RequestException as e:
            debug_log(f"RequestException: {e}")
            print(f"gh_apps: token exchange failed for {app_name}: {e}", file=sys.stderr)
            return None
        except jwt.PyJWTError as e:
            debug_log(f"PyJWTError: {e}")
            print(f"gh_apps: JWT generation failed for {app_name}: {e}", file=sys.stderr)
            return None


# Module-level singleton
_token_manager: TokenManager | None = None


def get_token(repo: str) -> str | None:
    """Get installation token for a repo.

    Convenience function that determines the correct app for the repo
    and returns its installation token. The token is scoped to the specified
    repo for least privilege (director apps won't get cross-repo access).

    Args:
        repo: Repository name (e.g., "ai_template").

    Returns:
        Installation token string, or None if no app configured.
    """
    debug_log(f"get_token(repo={repo}) - looking up app")

    global _token_manager
    if _token_manager is None:
        debug_log("initializing TokenManager singleton")
        _token_manager = TokenManager()

    from ai_template_scripts.gh_apps.selector import get_app_for_repo

    app_name = get_app_for_repo(repo)
    if not app_name:
        debug_log(f"no app found for repo '{repo}' - falling back to OAuth")
        return None

    debug_log(f"repo '{repo}' -> app '{app_name}'")
    # Pass repo for repo-scoped tokens (least privilege)
    return _token_manager.get_token(app_name, repo=repo)

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Identity and header helpers for gh_post."""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ai_template_scripts.subprocess_utils import get_github_repo
from looper.log import debug_swallow


def get_effective_role() -> str:
    """Get AI_ROLE with safe fallback to USER.

    Per AIT: "Assume USER if no role given at session start"
    This ensures USER sessions can add P0/urgent labels without
    needing to explicitly export AI_ROLE=USER (#1911).

    Returns:
        Role string: AI_ROLE env var if set, otherwise "USER"
    """
    role = os.environ.get("AI_ROLE")
    if role:
        return role
    # Per AIT: "Assume USER if no role given at session start"
    # This allows USER sessions to add P0/urgent labels without explicit AI_ROLE export
    return "USER"


def get_real_gh() -> str:
    """Find the real gh binary, skipping our wrapper."""
    script_dir = Path(__file__).resolve().parent
    wrapper_path = (script_dir.parent / "bin" / "gh").resolve()

    # First try common locations (fastest, avoids PATH issues)
    for loc in ["/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh"]:
        loc_path = Path(loc)
        if loc_path.is_file() and os.access(loc_path, os.X_OK):
            try:
                if loc_path.resolve() != wrapper_path:
                    return loc
            except OSError:
                return loc

    # Fall back to PATH search
    for path_dir in os.environ.get("PATH", "").split(":"):
        gh_path = Path(path_dir) / "gh"
        if gh_path.is_file() and os.access(gh_path, os.X_OK):
            try:
                if gh_path.resolve() == wrapper_path:
                    continue
            except OSError:
                pass
            return str(gh_path)

    raise RuntimeError("Real gh not found in PATH")


def _project_from_git_url(url: str) -> str:
    """Extract project name from common git remote URL formats.

    Normalizes .git suffixes and extra slashes.
    """
    path = url.strip().replace("\\", "/")
    path = path.split("?", 1)[0].split("#", 1)[0]
    if not path:
        return ""
    if "://" in path:
        path = path.split("://", 1)[1]
        if "/" in path:
            path = path.split("/", 1)[1]
        else:
            path = ""
    elif "@" in path and ":" in path.split("@", 1)[1]:
        path = path.split(":", 1)[1]
    path = path.rstrip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    parts = [segment for segment in path.split("/") if segment]
    return parts[-1] if parts else ""


def _read_iteration_file(role: str) -> str:
    """Read iteration from .iteration_{role} file as fallback.

    Looks for worker_logs/.iteration_{role}_{worker_id} first (if AI_WORKER_ID set),
    then falls back to worker_logs/.iteration_{role}.
    Returns empty string if not found.
    """
    role_lower = role.lower()
    worker_id = os.environ.get("AI_WORKER_ID", "")

    # Try worker_id-specific file first (e.g., .iteration_manager_1)
    if worker_id:
        iteration_file = Path("worker_logs") / f".iteration_{role_lower}_{worker_id}"
        if iteration_file.is_file():
            try:
                return iteration_file.read_text().strip()
            except Exception as e:
                debug_swallow("gh_post_read_iteration_file_worker", e)

    # Fall back to base file (e.g., .iteration_manager)
    iteration_file = Path("worker_logs") / f".iteration_{role_lower}"
    if iteration_file.is_file():
        try:
            return iteration_file.read_text().strip()
        except Exception as e:
            debug_swallow("gh_post_read_iteration_file", e)
            # Best-effort: iteration file read, empty string is safe fallback
    return ""


def get_identity() -> dict:
    """Get AI identity from env vars or derive from git.

    For iteration, falls back to .iteration_{role} file if env var is missing or 1.
    This handles USER sessions and looper sessions where AI_ITERATION may not be set
    correctly.
    """
    identity = {
        "project": os.environ.get("AI_PROJECT", ""),
        "role": get_effective_role(),
        "iteration": os.environ.get("AI_ITERATION", ""),
        "session": os.environ.get("AI_SESSION", ""),
    }

    # Fall back to .iteration_{role} file if env var is missing or "1" (default)
    # USER role uses an empty iteration since it doesn't have looper iterations
    if identity["role"] != "USER" and identity["iteration"] in ("", "1"):
        file_iteration = _read_iteration_file(identity["role"])
        if file_iteration and file_iteration.isdigit():
            identity["iteration"] = file_iteration

    # Derive project from git if not set
    if not identity["project"]:
        try:
            url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            # Handle various URL formats:
            # https://github.com/owner/repo.git
            # https://github.com/owner/repo/  (trailing slash)
            # git@github.com:owner/repo.git
            project = _project_from_git_url(url)
            if project:
                identity["project"] = project
            else:
                identity["project"] = Path.cwd().name
        except Exception as e:
            debug_swallow("gh_post_get_identity_project", e)
            identity["project"] = (
                Path.cwd().name
            )  # Best-effort: git remote lookup, cwd name is safe fallback

    return identity


def get_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        debug_swallow("gh_post_get_commit", e)
        return "-"  # Best-effort: git commit lookup, "-" is safe placeholder


def get_current_repo(real_gh: str) -> str:
    """Get current repo in owner/name format.

    Uses canonical get_github_repo() from subprocess_utils.
    Returns empty string on failure (best-effort).

    Args:
        real_gh: Path to the real gh binary (passed for compatibility,
                 but canonical function uses "gh" from PATH)
    """
    result = get_github_repo(gh_path=real_gh)
    return result.stdout if result.ok else ""


def build_header(identity: dict) -> str:
    """Build FROM header line."""
    header = f"**FROM:** {identity['project']}"
    if identity["iteration"]:
        header += f" [{identity['role']}]{identity['iteration']}"
    else:
        header += f" [{identity['role']}]"
    return header


def build_signature(identity: dict) -> str:
    """Build compact signature line."""
    from ai_template_scripts import gh_post as gh_post_module

    commit = gh_post_module.get_commit()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    sig = f"{identity['project']} | {identity['role']}"
    if identity["iteration"]:
        sig += f" #{identity['iteration']}"
    if identity["session"]:
        sig += f" | {identity['session'][:8]}"
    sig += f" | {commit} | {timestamp}"

    return f"---\n{sig}"

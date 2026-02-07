# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Queueing and change-log helpers for gh_post."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ai_template_scripts.gh_post.args import _extract_title_body
from looper.log import debug_swallow


# Pattern to extract issue number from gh issue create output
# Matches: https://github.com/owner/repo/issues/123
_ISSUE_URL_PATTERN = re.compile(r"/issues/(\d+)")


def _write_through_queued(subcommand: str, data: dict[str, Any]) -> None:
    """Write-through to local store when operation is queued (#1834).

    Called when GitHub API is rate-limited and operation is queued.
    Writes immediately to local store so local state is authoritative.

    Args:
        subcommand: Operation type (create, comment, edit, close).
        data: Operation data from the queue.
    """
    try:
        from ai_template_scripts.gh_post.write_through import write_through_from_queue

        write_through_from_queue(subcommand, data)
    except Exception as e:
        # Write-through is best-effort, don't fail the operation
        print(f"[write-through] Error during queue write-through: {e}", file=sys.stderr)


def _write_through_success(
    subcommand: str,
    args: list[str],
    parsed: dict,
    stdout: str,
) -> None:
    """Write-through to local store after successful GitHub operation (#1834).

    Called after successful GitHub API call to mirror state to local store.

    Args:
        subcommand: Operation type (create, comment, edit, close).
        args: Command line arguments (processed).
        parsed: Parsed arguments dict.
        stdout: stdout from successful gh command.
    """
    try:
        from ai_template_scripts.gh_post.write_through import (
            is_write_through_enabled,
            write_through_close,
            write_through_comment,
            write_through_create,
            write_through_edit,
        )

        if not is_write_through_enabled():
            return

        repo = parsed.get("repo")
        issue_number = parsed.get("issue_number")

        if subcommand == "create":
            # Extract issue number from stdout URL
            match = _ISSUE_URL_PATTERN.search(stdout)
            if match:
                issue_number = match.group(1)
            else:
                print(
                    f"[write-through] Could not parse issue number from: {stdout[:100]}",
                    file=sys.stderr,
                )
                return

            title, body = _extract_title_body(args, parsed)
            labels = parsed.get("create_labels", [])
            write_through_create(issue_number, title, body, labels, repo)

        elif subcommand == "comment":
            if not issue_number:
                return
            _, body = _extract_title_body(args, parsed)
            write_through_comment(issue_number, body, repo)

        elif subcommand == "edit":
            if not issue_number:
                return
            title, body = _extract_title_body(args, parsed)
            add_labels = parsed.get("add_labels", [])
            remove_labels = parsed.get("remove_labels", [])
            write_through_edit(
                issue_number,
                title or None,
                body or None,
                add_labels,
                remove_labels,
                repo,
            )

        elif subcommand == "close":
            if not issue_number:
                return
            write_through_close(issue_number, repo)

    except Exception as e:
        # Write-through is best-effort, don't fail the operation
        print(
            f"[write-through] Error during success write-through: {e}", file=sys.stderr
        )


def _gh_post():
    import ai_template_scripts.gh_post as gh_post_module

    return gh_post_module


def _get_current_repo_name() -> str:
    """Get current repo in owner/repo format for ChangeLog (#1750).

    Returns owner/repo from git remote URL, or just repo name as fallback.
    The owner/repo format is required for gh CLI API operations during
    change_log.json replay.
    """
    try:
        # Get git remote URL to extract owner/repo
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            url = result.stdout.strip().removesuffix(".git")
            # Handle SSH format: git@github.com:owner/repo
            if "github.com:" in url:
                parts = url.split("github.com:")[-1].split("/")
            # Handle HTTPS format: https://github.com/owner/repo
            elif "github.com/" in url:
                parts = url.split("github.com/")[-1].split("/")
            else:
                parts = url.split("/")
            if len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"
    except Exception as e:
        debug_swallow("gh_post_get_current_repo_name_remote", e)
    # Fallback: just directory name (may cause issues with gh API)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).name
    except Exception as e:
        debug_swallow("gh_post_get_current_repo_name_toplevel", e)
    return "unknown"


def _guess_repo_name_from_cwd() -> str:
    """Best-effort repo name without subprocess calls."""
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / ".git").exists():
            return candidate.name
    return cwd.name


def _invalidate_issue_cache(
    subcommand: str | None,
    issue_number: str | None = None,
    repo: str | None = None,
) -> None:
    """Best-effort cache invalidation after issue writes.

    Invalidates cached issue data after write operations to prevent stale reads.
    Logs warnings on failure but doesn't block the operation.

    Args:
        subcommand: The issue subcommand (edit, close, comment, etc.)
        issue_number: Optional issue number for historical cache invalidation (#2066)
        repo: Target repo for cross-repo operations (#2066). When editing an issue
            in repo B while in directory A, the cache is stored under B's name.
            If not provided, uses current directory's repo (may miss cross-repo cache).
    """
    gh_post_module = _gh_post()

    if not gh_post_module._HAS_RATE_LIMIT or not subcommand:
        return
    try:
        limiter = gh_post_module.get_limiter()
        # Include issue number in args for historical cache invalidation
        args = ["issue", subcommand]
        if issue_number:
            args.append(issue_number)
        # Pass repo for cross-repo cache invalidation (#2066)
        limiter.invalidate_cache(args, repo=repo)
    except Exception as e:
        # Log warning but don't fail the operation
        print(f"[gh_post] Warning: cache invalidation failed: {e}", file=sys.stderr)


def _sync_pending_changes(real_gh: str, max_count: int = 10) -> int:
    """Try to sync any pending changes from ChangeLog. Returns count synced."""
    gh_post_module = _gh_post()

    if not gh_post_module._HAS_RATE_LIMIT:
        return 0

    change_log = gh_post_module.get_change_log()
    pending = change_log.get_pending()
    if not pending:
        return 0

    limiter = gh_post_module.get_limiter()
    synced = 0

    for change in pending[:max_count]:  # Sync up to max_count per call
        # Check rate limit before each sync
        if not limiter.check_rate_limit(["issue", change.operation]):
            break

        # Build command from change data
        cmd = [real_gh, "issue", change.operation]
        data = change.data
        # Normalize repo to owner/repo format (fix for short repo names in change_log)
        repo_full = (
            gh_post_module._ensure_full_repo_name(data["repo"])
            if data.get("repo")
            else None
        )

        if change.operation == "create":
            if data.get("title"):
                cmd.extend(["--title", data["title"]])
            if data.get("body"):
                cmd.extend(["--body", data["body"]])
            if repo_full:
                cmd.extend(["--repo", repo_full])
            for label in data.get("labels", []):
                cmd.extend(["--label", label])
        elif change.operation == "comment":
            if data.get("issue"):
                cmd.append(str(data["issue"]))
            if data.get("body"):
                cmd.extend(["--body", data["body"]])
            if repo_full:
                cmd.extend(["--repo", repo_full])
        elif change.operation == "edit":
            if data.get("issue"):
                cmd.append(str(data["issue"]))
            if repo_full:
                cmd.extend(["--repo", repo_full])
            for label in data.get("add_labels", []):
                cmd.extend(["--add-label", label])
            for label in data.get("remove_labels", []):
                cmd.extend(["--remove-label", label])
            # Replay body and title for edit operations (#1041)
            if data.get("body"):
                cmd.extend(["--body", data["body"]])
            if data.get("title"):
                cmd.extend(["--title", data["title"]])
        elif change.operation == "close":
            if data.get("issue"):
                cmd.append(str(data["issue"]))
            if repo_full:
                cmd.extend(["--repo", repo_full])
            # Replay comment and reason for close operations (#1041)
            if data.get("comment"):
                cmd.extend(["--comment", data["comment"]])
            if data.get("reason"):
                cmd.extend(["--reason", data["reason"]])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            if result.returncode == 0:
                change_log.mark_synced(change.id)
                synced += 1
                gh_post_module._invalidate_issue_cache(
                    change.operation,
                    str(data.get("issue")) if data.get("issue") else None,
                    data.get("repo"),
                )
                print(
                    f"[gh_post] Synced pending {change.operation} ({change.id})",
                    file=sys.stderr,
                )
                # Add to project for replayed create operations (P0 #1190, #1200)
                if change.operation == "create" and result.stdout:
                    gh_post_module._auto_add_to_project_from_sync(
                        real_gh, result.stdout, data.get("repo")
                    )
            else:
                change_log.mark_error(change.id, result.stderr[:200])
        except Exception as e:
            change_log.mark_error(change.id, str(e)[:200])

    return synced


def _extract_arg_value(args: list[str], prefix: str) -> str:
    """Extract value from args for --flag=value format."""
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return ""


def _build_queue_data_create(args: list[str], parsed: dict, repo: str) -> dict:
    """Build queue data for create operations."""
    title = ""
    if parsed.get("title_index") is not None:
        title = args[parsed["title_index"]]
    else:
        title = _extract_arg_value(args, "--title=")

    body = ""
    if parsed.get("body_index") is not None:
        body = args[parsed["body_index"]]
    else:
        body = _extract_arg_value(args, "--body=")

    return {
        "repo": repo,
        "title": title,
        "body": body,
        "labels": parsed.get("create_labels", []),
    }


def _build_queue_data_comment(args: list[str], parsed: dict, repo: str) -> dict:
    """Build queue data for comment operations."""
    body = ""
    if parsed.get("body_index") is not None:
        body = args[parsed["body_index"]]
    else:
        body = _extract_arg_value(args, "--body=")

    return {
        "repo": repo,
        "issue": parsed.get("issue_number"),
        "body": body,
    }


def _build_queue_data_edit(args: list[str], parsed: dict, repo: str) -> dict:
    """Build queue data for edit operations."""
    data: dict = {
        "repo": repo,
        "issue": parsed.get("issue_number"),
        "add_labels": parsed.get("add_labels", []),
        "remove_labels": parsed.get("remove_labels", []),
    }

    body = ""
    if parsed.get("body_index") is not None:
        body = args[parsed["body_index"]]
    elif parsed.get("body_value") is not None:
        body = parsed["body_value"]
    if body:
        data["body"] = body

    title = ""
    if parsed.get("title_index") is not None:
        title = args[parsed["title_index"]]
    elif parsed.get("title_value") is not None:
        title = parsed["title_value"]
    if title:
        data["title"] = title

    return data


def _build_queue_data_close(args: list[str], parsed: dict, repo: str) -> dict:
    """Build queue data for close operations."""
    data: dict = {
        "repo": repo,
        "issue": parsed.get("issue_number"),
    }

    comment = ""
    if parsed.get("comment_index") is not None:
        comment = args[parsed["comment_index"]]
    elif parsed.get("comment_value") is not None:
        comment = parsed["comment_value"]
    if comment:
        data["comment"] = comment

    if parsed.get("reason_value"):
        data["reason"] = parsed["reason_value"]

    return data


def _build_queue_data(
    subcommand: str, args: list[str], parsed: dict, repo: str
) -> dict:
    """Build data dict for queueing an operation."""
    if subcommand == "create":
        return _build_queue_data_create(args, parsed, repo)
    elif subcommand == "comment":
        return _build_queue_data_comment(args, parsed, repo)
    elif subcommand == "edit":
        return _build_queue_data_edit(args, parsed, repo)
    elif subcommand == "close":
        return _build_queue_data_close(args, parsed, repo)
    else:
        return {"repo": repo}


def _handle_rest_fallback_result(
    gh_post_module: Any, rest_result: int | None, parsed: dict
) -> int | None:
    """Handle REST fallback result and invalidate cache if successful."""
    if rest_result is not None:
        if rest_result == 0:
            gh_post_module._invalidate_issue_cache(
                parsed.get("subcommand"),
                parsed.get("issue_number"),
                parsed.get("repo"),
            )
        return rest_result
    return None


def _execute_gh_with_queue(real_gh: str, args: list[str], parsed: dict) -> int:
    """Execute gh command with rate limit checking and offline queuing.

    If rate limited, queues the operation to ChangeLog for later sync.
    Returns exit code (0 = success or queued, non-zero = error).
    """
    gh_post_module = _gh_post()

    # Burst rate detection — throttle rapid sequential calls (#3220)
    try:
        from ai_template_scripts.gh_rate_limit.burst_tracker import get_burst_tracker

        get_burst_tracker().record_and_maybe_throttle()
    except Exception:
        debug_swallow("burst_tracker_gh_post")

    # Try to sync any pending changes first (opportunistic)
    gh_post_module._sync_pending_changes(real_gh)

    # Check rate limit if available
    if gh_post_module._HAS_RATE_LIMIT:
        limiter = gh_post_module.get_limiter()
        if not limiter.check_rate_limit(["issue", parsed["subcommand"]]):
            # Rate limited - queue for later
            change_log = gh_post_module.get_change_log()
            repo = gh_post_module._get_current_repo_name()
            target_repo = parsed.get("repo") or repo

            data = _build_queue_data(parsed["subcommand"], args, parsed, target_repo)
            change_id = change_log.add_change(repo, parsed["subcommand"], data)
            print(
                f"[gh_post] Rate limited - queued {parsed['subcommand']} as {change_id}",
                file=sys.stderr,
            )
            print(
                f"[gh_post] Pending changes: {change_log.pending_count()}",
                file=sys.stderr,
            )

            # Write-through to local store (#1834)
            _write_through_queued(parsed["subcommand"], data)

            return 75  # EX_TEMPFAIL - operation queued, not delivered

    # Proactive REST for label-only edits when GraphQL quota is low (#1502)
    if parsed.get("subcommand") == "edit" and gh_post_module._HAS_RATE_LIMIT:
        add_labels = parsed.get("add_labels", [])
        remove_labels = parsed.get("remove_labels", [])
        title, body = _extract_title_body(args, parsed)
        # Only use REST proactively for pure label edits (title/body need GraphQL)
        if (add_labels or remove_labels) and not title and not body:
            limiter = gh_post_module.get_limiter()
            if limiter._should_prefer_rest_for_quota(log=False):
                print(
                    "[gh_post] GraphQL quota low, using REST for label edit",
                    file=sys.stderr,
                )
                rest_result = gh_post_module._try_rest_fallback(real_gh, args, parsed)
                handled = _handle_rest_fallback_result(gh_post_module, rest_result, parsed)
                if handled is not None:
                    return handled
                # REST failed, fall through to try GraphQL anyway

    # Execute the command, capture output to detect rate limit errors
    result = subprocess.run(
        [real_gh] + args,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Check for GraphQL rate limit error (explicit failure)
    error_output = (result.stderr or "") + (result.stdout or "")
    if result.returncode != 0 and gh_post_module._is_graphql_rate_limited(error_output):
        print(
            "[gh_post] GraphQL rate limited, trying REST API fallback...",
            file=sys.stderr,
        )
        rest_result = gh_post_module._try_rest_fallback(real_gh, args, parsed)
        handled = _handle_rest_fallback_result(gh_post_module, rest_result, parsed)
        if handled is not None:
            return handled

        # REST fallback failed or not applicable - queue for later
        print("[gh_post] REST fallback failed, queueing for later", file=sys.stderr)
        return gh_post_module._queue_for_later(args, parsed)

    # Check for silent failure bug: gh CLI returns exit 0 with empty stdout
    # when GraphQL createIssue returns null (secondary rate limit, #1673, #1811)
    if gh_post_module._is_silent_create_failure(result, parsed):
        print(
            "[gh_post] Silent failure detected (exit 0, no URL) - GraphQL createIssue.issue=null",
            file=sys.stderr,
        )
        rest_result = gh_post_module._try_rest_fallback(real_gh, args, parsed)
        handled = _handle_rest_fallback_result(gh_post_module, rest_result, parsed)
        if handled is not None:
            return handled

        # REST also failed - queue for later
        print("[gh_post] REST fallback failed, queueing for later", file=sys.stderr)
        return gh_post_module._queue_for_later(args, parsed)

    # Print captured output
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        gh_post_module._invalidate_issue_cache(
            parsed.get("subcommand"),
            parsed.get("issue_number"),
            parsed.get("repo"),
        )
        # Auto-add to GitHub Project after successful issue create (P0 #1190)
        if parsed.get("subcommand") == "create":
            gh_post_module._auto_add_to_project(real_gh, result.stdout, parsed)

        # Write-through to local store (#1834)
        _write_through_success(
            parsed.get("subcommand", ""),
            args,
            parsed,
            result.stdout or "",
        )

    return result.returncode


def _queue_for_later(args: list[str], parsed: dict) -> int:
    """Queue operation for later when rate limit recovers."""
    gh_post_module = _gh_post()

    if not gh_post_module._HAS_RATE_LIMIT:
        return 1  # Can't queue without rate limit module

    change_log = gh_post_module.get_change_log()
    repo: str = parsed.get("repo") or gh_post_module._get_current_repo_name()
    subcommand: str = parsed.get("subcommand") or "unknown"
    title, body = _extract_title_body(args, parsed)

    data: dict = {"repo": repo}

    if subcommand == "create":
        data["title"] = title
        data["body"] = body
        data["labels"] = parsed.get("create_labels", [])
    elif subcommand == "comment":
        data["issue"] = parsed.get("issue_number")
        data["body"] = body
    elif subcommand == "edit":
        data["issue"] = parsed.get("issue_number")
        data["add_labels"] = parsed.get("add_labels", [])
        data["remove_labels"] = parsed.get("remove_labels", [])
    elif subcommand == "close":
        data["issue"] = parsed.get("issue_number")

    change_id = change_log.add_change(repo, subcommand, data)
    print(
        f"[gh_post] Queued {subcommand} as {change_id}",
        file=sys.stderr,
    )
    print(
        f"[gh_post] Pending changes: {change_log.pending_count()}",
        file=sys.stderr,
    )

    # Write-through to local store (#1834)
    _write_through_queued(subcommand, data)

    return 75  # EX_TEMPFAIL - operation queued, not delivered


def replay_pending_changes(max_per_call: int = 10) -> int:
    """Replay pending changes from ChangeLog at iteration start (#1846).

    Called by looper at iteration start to ensure pending issues are synced
    even when no new gh commands run during the iteration.

    This is a thin wrapper around _sync_pending_changes() that:
    1. Obtains real_gh path (since looper doesn't have it)
    2. Delegates to the shared sync implementation

    Contracts:
        REQUIRES: max_per_call >= 0
        ENSURES: Returns count of successfully synced changes (0 <= result <= max_per_call)
        ENSURES: If gh_rate_limit unavailable, returns 0 silently
        ENSURES: Rate limit checked before each sync attempt

    Args:
        max_per_call: Maximum pending changes to sync per call (default 10).

    Returns:
        Count of successfully synced operations.
    """
    try:
        from ai_template_scripts.gh_post.identity import get_real_gh

        real_gh = get_real_gh()
    except Exception as e:
        debug_swallow("gh_post_replay_pending_changes", e)
        return 0

    return _sync_pending_changes(real_gh, max_per_call)

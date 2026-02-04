# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""REST fallback helpers for gh_post."""

import json
import subprocess
import sys

from ai_template_scripts.gh_post.args import _extract_title_body


def _gh_post():
    import ai_template_scripts.gh_post as gh_post_module

    return gh_post_module


def _is_graphql_rate_limited(output: str) -> bool:
    """Check if output indicates GraphQL rate limiting.

    Note: The error message may appear in either stdout or stderr depending
    on gh CLI version and exact error type. Callers should pass both streams
    concatenated (e.g., stderr + stdout) to ensure detection (#1779).
    """
    if not output:
        return False
    output_lower = output.lower()
    return (
        "rate limit" in output_lower
        or "api rate limit exceeded" in output_lower
        or "secondary rate limit" in output_lower
    )


def _is_silent_create_failure(
    result: subprocess.CompletedProcess, parsed: dict
) -> bool:
    """Detect gh CLI silent failure bug on issue/PR create.

    gh v2.83.2+ returns exit 0 with empty/non-URL stdout when:
    - Hitting secondary rate limits
    - GraphQL createIssue mutation returns { issue: null }

    On successful create, gh outputs the issue/PR URL (e.g.,
    https://github.com/owner/repo/issues/123).
    Empty stdout or non-URL output indicates failure.

    See: #1671 (gh CLI silent failure on secondary rate limit)
    See: #1811 (GraphQL createIssue.issue=null returns exit 0)
    """
    if result.returncode != 0:
        return False  # Explicit failure, not silent

    subcommand = parsed.get("subcommand")
    if subcommand not in ("create", "comment"):
        return False  # Only create/comment have expected output

    # Successful create outputs URL, successful comment outputs nothing
    # But silent failure returns just whitespace/newline
    stdout = result.stdout.strip() if result.stdout else ""

    if subcommand == "create":
        # Create should return URL like https://github.com/owner/repo/issues/123
        return not stdout or not stdout.startswith("http")

    # For comment, we can't easily distinguish success from silent failure
    # without probing, so we don't treat empty as failure here
    return False


def _try_rest_fallback(real_gh: str, args: list[str], parsed: dict) -> int | None:
    """Try REST API fallback for issue operations.

    Returns exit code on success, None if fallback not applicable or failed.
    """
    gh_post_module = _gh_post()

    subcommand = parsed.get("subcommand")
    repo = parsed.get("repo") or gh_post_module._get_current_repo_name()

    if subcommand == "create":
        return _rest_create_issue(real_gh, args, parsed, repo)
    if subcommand == "comment":
        return _rest_add_comment(real_gh, args, parsed, repo)
    if subcommand == "edit":
        return _rest_edit_issue(real_gh, args, parsed, repo)
    # close doesn't have simple REST equivalent, queue it
    return None


def _rest_create_issue(
    real_gh: str, args: list[str], parsed: dict, repo: str
) -> int | None:
    """Create issue via REST API.

    Returns exit code on success, None if rate limited or not applicable.
    """
    gh_post_module = _gh_post()

    title, body = _extract_title_body(args, parsed)
    labels = parsed.get("create_labels", [])

    if not title:
        return None

    # Normalize repo for API calls
    repo_full = gh_post_module._ensure_full_repo_name(repo)

    # Build REST API command: gh api /repos/OWNER/REPO/issues --method POST
    cmd = [
        real_gh,
        "api",
        f"/repos/{repo_full}/issues",
        "--method",
        "POST",
        "-f",
        f"title={title}",
    ]

    if body:
        cmd.extend(["-f", f"body={body}"])

    if labels:
        # REST API expects labels as JSON array - use --input for raw JSON
        # or add each label individually
        for label in labels:
            cmd.extend(["--raw-field", f"labels[]={label}"])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode == 0:
        # Parse response to get issue URL and add to project
        try:
            data = json.loads(result.stdout)
            html_url = data.get("html_url", "")
            print(html_url or result.stdout)
            # Auto-add to project using node_id from REST response (P0 #1190)
            node_id = data.get("node_id")
            repo_name = repo.split("/")[-1] if "/" in repo else repo
            if node_id:
                success = gh_post_module._add_issue_to_project(
                    real_gh, node_id, repo_name
                )
                if success:
                    issue_num = data.get("number", "?")
                    print(
                        f"[gh_post] Added issue #{issue_num} to Project #1",
                        file=sys.stderr,
                    )
            return 0
        except Exception:
            print(result.stdout, end="")
            return 0

    # Check if REST also rate limited (#1779 - check both stderr and stdout)
    rest_error = (result.stderr or "") + (result.stdout or "")
    if _is_graphql_rate_limited(rest_error):
        return None

    print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def _rest_add_comment(
    real_gh: str, args: list[str], parsed: dict, repo: str
) -> int | None:
    """Add comment via REST API.

    Returns exit code on success, None if rate limited or not applicable.
    """
    gh_post_module = _gh_post()

    issue_number = parsed.get("issue_number")
    _, body = _extract_title_body(args, parsed)

    if not issue_number or not body:
        return None

    # Normalize repo for API calls
    repo_full = gh_post_module._ensure_full_repo_name(repo)

    cmd = [
        real_gh,
        "api",
        f"/repos/{repo_full}/issues/{issue_number}/comments",
        "--method",
        "POST",
        "-f",
        f"body={body}",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            print(data.get("html_url", result.stdout))
            return 0
        except Exception:
            print(result.stdout, end="")
            return 0

    # Check if REST also rate limited (#1779 - check both stderr and stdout)
    rest_error = (result.stderr or "") + (result.stdout or "")
    if _is_graphql_rate_limited(rest_error):
        return None

    print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def _rest_edit_issue(
    real_gh: str, args: list[str], parsed: dict, repo: str
) -> int | None:
    """Edit issue via REST API (label modifications).

    The GitHub REST API supports updating labels via:
    - GET /repos/{owner}/{repo}/issues/{issue_number}/labels (fetch current)
    - PATCH /repos/{owner}/{repo}/issues/{issue_number} (update labels array)
    - POST/DELETE for individual label add/remove

    This implementation:
    1. Fetches current labels
    2. Applies add/remove operations
    3. PATCHes the final label set

    Returns exit code on success, None if rate limited or not applicable.
    """
    gh_post_module = _gh_post()

    issue_number = parsed.get("issue_number")
    add_labels = parsed.get("add_labels", [])
    remove_labels = parsed.get("remove_labels", [])

    # Only handle label-only edits via REST
    # Title/body edits need full GraphQL support for consistency
    title, body = _extract_title_body(args, parsed)
    if title or body:
        return None  # Let it queue for GraphQL

    if not issue_number:
        return None

    if not add_labels and not remove_labels:
        return None  # Nothing to do

    # Normalize repo for API calls
    repo_full = gh_post_module._ensure_full_repo_name(repo)

    # Step 1: Get current labels
    get_cmd = [
        real_gh,
        "api",
        f"/repos/{repo_full}/issues/{issue_number}",
        "--jq",
        "[.labels[].name]",
    ]
    get_result = subprocess.run(get_cmd, check=False, capture_output=True, text=True)

    if get_result.returncode != 0:
        # Check if REST also rate limited (#1779 - check both stderr and stdout)
        get_error = (get_result.stderr or "") + (get_result.stdout or "")
        if _is_graphql_rate_limited(get_error):
            return None
        print(get_result.stderr, end="", file=sys.stderr)
        return get_result.returncode

    try:
        current_labels = json.loads(get_result.stdout)
    except json.JSONDecodeError:
        current_labels = []

    # Step 2: Compute new label set
    new_labels = set(current_labels)
    for label in remove_labels:
        new_labels.discard(label)
    for label in add_labels:
        new_labels.add(label)

    # Step 3: Update labels via PATCH
    patch_cmd = [
        real_gh,
        "api",
        f"/repos/{repo_full}/issues/{issue_number}",
        "--method",
        "PATCH",
        "--input",
        "-",
    ]

    # Send JSON body via stdin
    patch_body = json.dumps({"labels": list(new_labels)})
    patch_result = subprocess.run(
        patch_cmd,
        input=patch_body,
        check=False,
        capture_output=True,
        text=True,
    )

    if patch_result.returncode == 0:
        print(
            "[gh_post] REST fallback: labels updated successfully",
            file=sys.stderr,
        )
        # Print issue URL like gh issue edit does
        try:
            data = json.loads(patch_result.stdout)
            html_url = data.get("html_url", "")
            if html_url:
                print(html_url)
        except Exception:
            pass
        return 0

    # Check if REST also rate limited (#1779 - check both stderr and stdout)
    patch_error = (patch_result.stderr or "") + (patch_result.stdout or "")
    if _is_graphql_rate_limited(patch_error):
        return None

    print(patch_result.stderr, end="", file=sys.stderr)
    return patch_result.returncode

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Issue operations (edit/close) for gh_post."""

import json
import re
import subprocess
import sys

from ai_template_scripts.gh_post.labels import PROTECTED_LABELS
from ai_template_scripts.gh_post.validation import (
    _check_malformed_in_progress_labels,
    _check_user_only_labels,
    _get_p_labels_from_list,
    _is_p_label,
)


def _gh_post():
    import ai_template_scripts.gh_post as gh_post_module

    return gh_post_module


# Regex to parse FROM header: **FROM:** <project> [<ROLE>]<iteration>
# Groups: 1=project, 2=role (optional), 3=iteration (optional)
_FROM_HEADER_RE = re.compile(
    r"^\*\*FROM:\*\*\s+([a-zA-Z0-9_-]+)(?:\s+\[([A-Z]+)\](\d+)?)?",
    re.MULTILINE,
)


def _parse_from_header(body: str) -> str | None:
    """Parse sender project from FROM header in issue body.

    Returns the project name if found, None otherwise.
    The FROM header format is: **FROM:** <project> [<ROLE>]<iteration>
    """
    if not body:
        return None
    match = _FROM_HEADER_RE.search(body)
    if match:
        return match.group(1)
    return None


def _get_issue_info(
    real_gh: str, issue_number: str, repo: str | None = None
) -> dict | None:
    """Fetch issue info (body, title, url) from GitHub. Returns None on error."""
    cmd = [real_gh, "issue", "view", issue_number, "--json", "body,title,url"]
    if repo:
        cmd.extend(["--repo", repo])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def _notify_mail_sender_on_close(
    real_gh: str, issue_number: str, repo: str | None, commit_hash: str | None
) -> None:
    """Notify sender when a mail issue is closed.

    Checks if issue has mail label and FROM header, then files a notification
    issue to the sender's repo with closure status.

    Args:
        real_gh: Path to real gh executable
        issue_number: Issue being closed
        repo: Target repo (None for current)
        commit_hash: Fix commit hash if available, empty string if none,
            None to compute on-demand for mail issues
    """
    gh_post_module = _gh_post()

    # Check if this is a mail issue
    labels = gh_post_module.get_issue_labels(real_gh, issue_number, repo)
    if "mail" not in labels:
        return

    if commit_hash is None:
        has_fix, commit_hash = gh_post_module._has_fix_commit(issue_number)
        if not has_fix:
            commit_hash = ""

    # Get issue details
    issue_info = gh_post_module._get_issue_info(real_gh, issue_number, repo)
    if not issue_info:
        print(
            f"⚠️  Could not fetch issue #{issue_number} for notification",
            file=sys.stderr,
        )
        return

    body = issue_info.get("body", "")
    title = issue_info.get("title", "")
    url = issue_info.get("url", "")

    # Parse sender from FROM header
    sender = gh_post_module._parse_from_header(body)
    if not sender:
        print(
            f"⚠️  Mail issue #{issue_number} has no FROM header, skipping notification",
            file=sys.stderr,
        )
        return

    # Build notification
    identity = gh_post_module.get_identity()
    current_repo = repo or gh_post_module.get_current_repo(real_gh)
    if not current_repo:
        current_repo = identity.get("project", "")
    status_line = "**Status:** Closed"
    if commit_hash and commit_hash != "unknown":
        status_line += f" (fix: {commit_hash})"

    notif_body = f"""**Re:** {title}

{status_line}
**Original:** {url}

---

"""
    # Add identity header and signature
    notif_body = gh_post_module.process_body(notif_body, identity)

    notif_title = f"[{current_repo}] Closed: {title}"
    # Truncate title if too long (GitHub limit is 256)
    if len(notif_title) > 200:
        notif_title = notif_title[:197] + "..."

    # File notification to sender's repo
    try:
        cmd = [
            real_gh,
            "issue",
            "create",
            "--repo",
            f"dropbox-ai-prototypes/{sender}",
            "--title",
            notif_title,
            "--body",
            notif_body,
            "--label",
            "mail",
            "--label",
            "notification",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(
                f"📬 Notified {sender} of issue #{issue_number} closure",
                file=sys.stderr,
            )
        else:
            # Log warning but don't block closure
            print(
                f"⚠️  Could not notify {sender}: {result.stderr.strip()}",
                file=sys.stderr,
            )
    except Exception as e:
        # Log warning but don't block closure
        print(f"⚠️  Could not notify {sender}: {e}", file=sys.stderr)


def _handle_close(real_gh: str, args: list, parsed: dict) -> None:
    """Handle issue close: enforce role, require fix commit, and cleanup labels.

    Only MANAGER or USER can close issues. MANAGER must have a linked fix commit.
    Cleans up workflow labels before closing.
    Never returns - exits via execv or sys.exit.
    """
    gh_post_module = _gh_post()

    role = gh_post_module.get_effective_role()
    if role not in ("MANAGER", "USER"):
        print(file=sys.stderr)
        print("❌ ERROR: Only MANAGER role can close issues", file=sys.stderr)
        print(f"   Current role: {role}", file=sys.stderr)
        print(
            "   File an issue or use 'Part of #N' in commits instead.",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        sys.exit(1)

    commit_hash: str | None = None

    # MANAGER must have a fix commit to close (USER can override)
    # Exceptions: duplicate/environmental/stale issues can be closed without Fixes commit
    # stale = auto-generated child issues that became obsolete when parent closed (#962, #963)
    if role == "MANAGER" and parsed["issue_number"]:
        issue_labels = gh_post_module.get_issue_labels(
            real_gh, parsed["issue_number"], parsed["repo"]
        )
        is_duplicate = "duplicate" in issue_labels
        is_environmental = "environmental" in issue_labels
        is_stale = "stale" in issue_labels
        has_fix, commit_hash = gh_post_module._has_fix_commit(parsed["issue_number"])
        if not has_fix and not is_duplicate and not is_environmental and not is_stale:
            print(file=sys.stderr)
            print(
                "❌ ERROR: Cannot close issue without 'Fixes #N' commit",
                file=sys.stderr,
            )
            print(f"   Issue: #{parsed['issue_number']}", file=sys.stderr)
            print(file=sys.stderr)
            print("   No commit found with 'Fixes #N' in the message.", file=sys.stderr)
            print(
                "   Issues must be closed via verified fix commits, not manually.",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            print("   To close this issue:", file=sys.stderr)
            print(
                f"     1. Find or create the fix commit with 'Fixes #{parsed['issue_number']}'",
                file=sys.stderr,
            )
            print("     2. Push the commit", file=sys.stderr)
            print("     3. Then close the issue", file=sys.stderr)
            print(file=sys.stderr)
            print(
                "   Or, if this is a duplicate, environmental, or stale issue:",
                file=sys.stderr,
            )
            print(
                "     1. Add 'duplicate', 'environmental', or 'stale' label",
                file=sys.stderr,
            )
            print("     2. Then close the issue", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)

    if parsed["issue_number"]:
        gh_post_module.cleanup_workflow_labels(
            real_gh, parsed["issue_number"], parsed["repo"], role
        )
        # Notify mail sender on close (best-effort, doesn't block close)
        gh_post_module._notify_mail_sender_on_close(
            real_gh, parsed["issue_number"], parsed["repo"], commit_hash
        )
    sys.exit(gh_post_module._execute_gh_with_queue(real_gh, args, parsed))


def _handle_edit(real_gh: str, args: list, parsed: dict) -> None:
    """Handle issue edit: enforce label rules and add claiming comment.

    Enforces:
    - Non-USER cannot add USER-only labels (urgent, P0)
    - Only WORKER can add do-audit label
    - Auto-removes in-progress labels when do-audit is added (mutually exclusive)
    - Non-USER cannot remove USER-protected labels
    - Adds claiming comment when in-progress is added
    - Tracks USER-added protected labels

    Never returns - exits via sys.exit.
    """
    gh_post_module = _gh_post()

    role = gh_post_module.get_effective_role()

    # ENFORCE: Non-USER cannot add USER-only labels (urgent, P0)
    _check_user_only_labels(parsed["add_labels"], role, "edit")

    # ENFORCE: No malformed in-progress labels (#866)
    _check_malformed_in_progress_labels(parsed["add_labels"])

    # ENFORCE: Only WORKER can add do-audit label
    if "do-audit" in parsed["add_labels"] and role != "WORKER":
        print(file=sys.stderr)
        print("❌ ERROR: Only WORKER role can add 'do-audit' label", file=sys.stderr)
        print(f"   Current role: {role}", file=sys.stderr)
        print(
            "   Workflow: Worker → do-audit → needs-review → Manager closes",
            file=sys.stderr,
        )
        print("   Other roles: → needs-review → Manager closes", file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)

    # Check if we're adding any P labels or in-progress labels
    adding_p_labels = _get_p_labels_from_list(parsed["add_labels"])

    # Fetch current labels once if needed for mutual exclusion checks
    current_labels: list[str] | None = None
    if parsed["issue_number"] and (
        "do-audit" in parsed["add_labels"]
        or parsed["adding_in_progress"]
        or adding_p_labels
    ):
        current_labels = gh_post_module.get_issue_labels(
            real_gh, parsed["issue_number"], parsed["repo"]
        )

    # AUTO-REMOVE: in-progress labels when do-audit is added (mutually exclusive)
    if "do-audit" in parsed["add_labels"] and current_labels is not None:
        for label in current_labels:
            if (
                gh_post_module._is_in_progress_label(label)
                and label not in parsed["remove_labels"]
            ):
                args.extend(["--remove-label", label])
                parsed["remove_labels"].append(label)

    # AUTO-REMOVE: other in-progress labels when adding in-progress (single owner)
    if parsed["adding_in_progress"] and current_labels is not None:
        new_in_progress = [
            lbl
            for lbl in parsed["add_labels"]
            if gh_post_module._is_in_progress_label(lbl)
        ]
        for label in current_labels:
            if (
                gh_post_module._is_in_progress_label(label)
                and label not in new_in_progress
                and label not in parsed["remove_labels"]
            ):
                args.extend(["--remove-label", label])
                parsed["remove_labels"].append(label)

    # AUTO-REMOVE: other P labels when adding a P label (single priority)
    if adding_p_labels and current_labels is not None:
        for label in current_labels:
            if (
                _is_p_label(label)
                and label not in adding_p_labels
                and label not in parsed["remove_labels"]
            ):
                args.extend(["--remove-label", label])
                parsed["remove_labels"].append(label)

    # AUTO-REMOVE: P labels when adding deferred (deferred = no priority)
    adding_deferred = "deferred" in parsed["add_labels"]
    if adding_deferred and current_labels is not None:
        for label in current_labels:
            if _is_p_label(label) and label not in parsed["remove_labels"]:
                args.extend(["--remove-label", label])
                parsed["remove_labels"].append(label)

    # ENFORCE: Non-USER cannot remove USER-protected labels
    if role != "USER" and parsed["remove_labels"]:
        for label in parsed["remove_labels"]:
            if label in PROTECTED_LABELS:
                if gh_post_module.is_label_user_protected(
                    real_gh, parsed["issue_number"], label, parsed.get("repo")
                ):
                    print(file=sys.stderr)
                    print(
                        f"❌ ERROR: Cannot remove USER-protected label '{label}'",
                        file=sys.stderr,
                    )
                    print(f"   Current role: {role}", file=sys.stderr)
                    print(
                        f"   The '{label}' label was set by USER and is protected.",
                        file=sys.stderr,
                    )
                    print("   Only USER can remove labels they set.", file=sys.stderr)
                    print(file=sys.stderr)
                    sys.exit(1)

    # Check rate limit and possibly queue
    if gh_post_module._HAS_RATE_LIMIT:
        # Try to sync pending first
        gh_post_module._sync_pending_changes(real_gh)
        limiter = gh_post_module.get_limiter()
        if not limiter.check_rate_limit(["issue", "edit"]):
            # Rate limited - queue for later
            change_log = gh_post_module.get_change_log()
            repo = gh_post_module._get_current_repo_name()
            data = {
                "repo": parsed.get("repo") or repo,
                "issue": parsed.get("issue_number"),
                "add_labels": parsed.get("add_labels", []),
                "remove_labels": parsed.get("remove_labels", []),
            }
            change_id = change_log.add_change(repo, "edit", data)
            print(
                f"[gh_post] Rate limited - queued edit as {change_id}",
                file=sys.stderr,
            )
            sys.exit(0)

    # ATOMIC CLAIMING: Use atomic protocol when adding in-progress label (#731)
    # This prevents race conditions between workers claiming the same issue
    if parsed["adding_in_progress"] and current_labels is not None:
        already_claimed = any(
            gh_post_module._is_in_progress_label(lbl) for lbl in current_labels
        )
        if not already_claimed:
            # Get the in-progress label being added
            in_progress_label = next(
                (
                    lbl
                    for lbl in parsed["add_labels"]
                    if gh_post_module._is_in_progress_label(lbl)
                ),
                "in-progress",
            )

            # Use atomic claiming protocol
            success, message = gh_post_module._atomic_claim_issue(
                real_gh,
                parsed["issue_number"],
                in_progress_label,
                repo=parsed.get("repo"),
                get_identity_fn=gh_post_module.get_identity,
                get_commit_fn=gh_post_module.get_commit,
                process_body_fn=gh_post_module.process_body,
                invalidate_cache_fn=gh_post_module._invalidate_issue_cache,
            )

            if not success:
                if message.startswith("Yielding to claim:"):
                    print(f"[gh_post] {message}", file=sys.stderr)
                    sys.exit(0)
                print(f"[gh_post] Claim failed: {message}", file=sys.stderr)
                sys.exit(1)

            print(f"[gh_post] {message}", file=sys.stderr)

            # Execute remaining edit operations (non-in-progress labels)
            remaining_args = gh_post_module._strip_in_progress_additions(args)
            if gh_post_module._has_edit_modifiers(remaining_args):
                result = subprocess.run([real_gh] + remaining_args)
                if result.returncode != 0:
                    sys.exit(result.returncode)
                gh_post_module._invalidate_issue_cache(
                    "edit", parsed["issue_number"], parsed.get("repo")
                )

            sys.exit(0)

    title, body = gh_post_module._extract_title_body(args, parsed)
    label_only = (
        (parsed.get("add_labels") or parsed.get("remove_labels"))
        and not title
        and not body
    )
    if label_only and not parsed["adding_in_progress"]:
        repo = parsed.get("repo") or gh_post_module._guess_repo_name_from_cwd()
        rest_result = gh_post_module._rest_edit_issue(real_gh, args, parsed, repo)
        if rest_result is not None:
            if rest_result == 0:
                gh_post_module._invalidate_issue_cache(
                    "edit", parsed["issue_number"], repo
                )
                # Track: USER adding protected labels
                if role == "USER" and parsed["add_labels"]:
                    target_repo = parsed.get("repo")
                    for label in parsed["add_labels"]:
                        if label in PROTECTED_LABELS:
                            if not gh_post_module.is_label_user_protected(
                                real_gh, parsed["issue_number"], label, target_repo
                            ):
                                gh_post_module.add_user_label_marker(
                                    real_gh, parsed["issue_number"], label, target_repo
                                )
                sys.exit(0)
            sys.exit(rest_result)

    # Execute the edit command (non-claiming edits or already-claimed issues)
    result = subprocess.run([real_gh] + args)
    if result.returncode != 0:
        sys.exit(result.returncode)
    gh_post_module._invalidate_issue_cache(
        "edit", parsed["issue_number"], parsed.get("repo")
    )

    # Track: USER adding protected labels
    if role == "USER" and parsed["add_labels"]:
        target_repo = parsed.get("repo")
        for label in parsed["add_labels"]:
            if label in PROTECTED_LABELS:
                if not gh_post_module.is_label_user_protected(
                    real_gh, parsed["issue_number"], label, target_repo
                ):
                    gh_post_module.add_user_label_marker(
                        real_gh, parsed["issue_number"], label, target_repo
                    )

    sys.exit(0)


def _process_body_and_title(args: list, parsed: dict, identity: dict) -> list:
    """Process body and title for create/comment commands.

    Returns modified args list with identity added to body and title fixed.
    """
    gh_post_module = _gh_post()

    # Process body
    if parsed["body_index"] is not None:
        original_body = args[parsed["body_index"]]
        args[parsed["body_index"]] = gh_post_module.process_body(
            original_body, identity
        )
    elif parsed["body_value"] is not None:
        for i, arg in enumerate(args):
            if arg.startswith("--body="):
                args[i] = "--body=" + gh_post_module.process_body(
                    parsed["body_value"], identity
                )
                break

    # Process title (issue create only)
    if parsed["subcommand"] == "create":
        if parsed["title_index"] is not None:
            original_title = args[parsed["title_index"]]
            args[parsed["title_index"]] = gh_post_module.fix_title(
                original_title, identity
            )
        elif parsed["title_value"] is not None:
            for i, arg in enumerate(args):
                if arg.startswith("--title="):
                    args[i] = "--title=" + gh_post_module.fix_title(
                        parsed["title_value"], identity
                    )
                    break

    return args


def _add_mail_label_if_needed(args: list, parsed: dict, real_gh: str) -> list:
    """Add mail label for cross-repo issues, and P2 for mail without priority.

    Cross-repo mail issues must have minimum P2 priority per org rule.
    """
    gh_post_module = _gh_post()

    if parsed["subcommand"] == "create" and parsed["repo"]:
        current_repo = gh_post_module.get_current_repo(real_gh)
        if (
            current_repo
            and parsed["repo"] != current_repo
            and not parsed["has_mail_label"]
        ):
            args.extend(["--label", "mail"])
            # Mark mail label as present for P2 check below
            parsed["has_mail_label"] = True

    # Org rule: mail issues must have at least P2 priority
    if parsed["subcommand"] == "create" and parsed["has_mail_label"]:
        if not parsed["has_p_label"]:
            args.extend(["--label", "P2"])

    return args

# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_post - Wrapper for gh that adds AI identity to issues and comments

Handles:
- gh issue create/comment - adds identity header and signature
- gh issue edit --add-label in-progress (with optional ownership label like W1/prov1/R1/M1) - auto-comments to claim with identity
- gh issue edit --add-label urgent (USER) - tracks USER ownership of label
- gh issue edit --remove-label urgent - BLOCKED if USER set the label
- gh issue close - BLOCKED unless MANAGER or USER role (enforces issue closure policy)
- gh issue close - removes workflow labels to keep queues clean
- Smart deduplication of existing identity markers via regex
- Cross-repo issues auto-labeled with "mail"

USER Label Protection:
- When USER adds 'urgent' label, a tracking comment is added: <!-- USER_LABEL:urgent -->
- When non-USER tries to remove a USER-protected label, the operation is blocked
- Protected labels: urgent (extensible via PROTECTED_LABELS list)

NOTE: Do NOT use gh issue/repo view -q or --jq - has caching bugs in v2.83.2+ (#1047).
Always pipe to external jq or parse JSON in Python instead.

Public API:
- Constants: PROTECTED_LABELS, USER_ONLY_LABELS, WORKFLOW_LABELS, IN_PROGRESS_LABEL_PREFIX,
  IN_PROGRESS_ALL_LABELS, P_LABELS, PRIORITY_PREFIX_RE
- Functions: get_effective_role, get_real_gh, get_identity, get_commit, get_current_repo,
  get_issue_labels, build_header, build_signature, clean_body, fix_title, process_body,
  parse_args, is_label_user_protected, add_user_label_marker, cleanup_workflow_labels,
  cleanup_closed_issues_workflow_labels, main

Implementation details (underscore-prefixed) are not part of public API.

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import os
import sys
from pathlib import Path

# Add parent dir to path for imports when module is loaded from a synced copy
# in another repo (e.g., ~/z4/ai_template_scripts/gh_post/__init__.py).
# Without this, 'from ai_template_scripts.X' fails with ModuleNotFoundError.
_script_dir = Path(__file__).resolve().parent
_parent_dir = str(_script_dir.parent.parent)  # ai_template_scripts/../ = repo root
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from ai_template_scripts.gh_atomic_claim import _atomic_claim_issue
from ai_template_scripts.labels import IN_PROGRESS_ALL_LABELS, IN_PROGRESS_PREFIX

# Import rate limiting and change log for offline-first operations
try:
    from ai_template_scripts.gh_rate_limit import get_change_log, get_limiter

    _HAS_RATE_LIMIT = True
except ImportError:
    _HAS_RATE_LIMIT = False

# GitHub Project integration (P0 #1190)
from ai_template_scripts.identity import get_identity as _get_ident

PROJECT_OWNER = _get_ident().github_org
PROJECT_NUMBER = 1
PROJECT_SCHEMA_CACHE_FILE = Path.home() / ".ait_gh_cache" / "project_schema.json"
PROJECT_SCHEMA_TTL_SECONDS = 86400  # 24 hours

from ai_template_scripts.gh_post.args import (
    _extract_title_body,
    _find_command_index,
    _find_subcommand_index,
    _has_edit_modifiers,
    _is_global_flag_with_inline_value,
    _is_in_progress_label,
    _strip_in_progress_additions,
    parse_args,
)
from ai_template_scripts.gh_post.body_utils import (
    PRIORITY_PREFIX_RE,
    clean_body,
    fix_title,
    process_body,
)
from ai_template_scripts.gh_post.identity import (
    build_header,
    build_signature,
    get_commit,
    get_current_repo,
    get_effective_role,
    get_identity,
    get_real_gh,
)
from ai_template_scripts.gh_post.labels import (
    IN_PROGRESS_LABEL_PREFIX,
    P_LABELS,
    PROTECTED_LABELS,
    USER_ONLY_LABELS,
    WORKFLOW_LABELS,
    _get_issue_labels_rest,
    add_user_label_marker,
    cleanup_closed_issues_workflow_labels,
    cleanup_workflow_labels,
    get_issue_labels,
    is_label_user_protected,
)
from ai_template_scripts.gh_post.operations import (
    _add_mail_label_if_needed,
    _get_issue_info,
    _handle_close,
    _handle_edit,
    _notify_mail_sender_on_close,
    _parse_from_header,
    _process_body_and_title,
)
from ai_template_scripts.gh_post.project import (
    _add_issue_to_project,
    _add_issue_to_project_by_url,
    _auto_add_to_project,
    _auto_add_to_project_from_sync,
    _ensure_full_repo_name,
    _extract_issue_info,
    _fetch_project_schema,
    _get_issue_node_id,
    _get_project_schema,
    _load_cached_schema,
    _ProjectSchema,
    _save_schema_cache,
    _set_project_fields,
)
from ai_template_scripts.gh_post.queue import (
    _execute_gh_with_queue,
    _get_current_repo_name,
    _guess_repo_name_from_cwd,
    _invalidate_issue_cache,
    _queue_for_later,
    _sync_pending_changes,
    replay_pending_changes,
)
from ai_template_scripts.gh_post.rest_fallback import (
    _is_graphql_rate_limited,
    _is_silent_create_failure,
    _rest_add_comment,
    _rest_create_issue,
    _rest_edit_issue,
    _try_rest_fallback,
)
from ai_template_scripts.gh_post.validation import (
    _check_malformed_in_progress_labels,
    _check_single_p_label_on_create,
    _check_user_only_labels,
    _get_p_labels_from_list,
    _has_fix_commit,
    _is_malformed_in_progress_label,
    _is_p_label,
)
from ai_template_scripts.gh_post.write_through import (
    is_write_through_enabled,
    write_through_close,
    write_through_comment,
    write_through_create,
    write_through_edit,
    write_through_from_queue,
)

__all__ = [
    # Constants
    "PROTECTED_LABELS",
    "USER_ONLY_LABELS",
    "WORKFLOW_LABELS",
    "IN_PROGRESS_LABEL_PREFIX",
    "IN_PROGRESS_ALL_LABELS",
    "P_LABELS",
    "PRIORITY_PREFIX_RE",
    # Functions
    "get_effective_role",
    "get_real_gh",
    "get_identity",
    "get_commit",
    "get_current_repo",
    "get_issue_labels",
    "build_header",
    "build_signature",
    "clean_body",
    "fix_title",
    "process_body",
    "parse_args",
    "is_label_user_protected",
    "add_user_label_marker",
    "cleanup_workflow_labels",
    "cleanup_closed_issues_workflow_labels",
    "replay_pending_changes",
    # Write-through functions (#1834)
    "is_write_through_enabled",
    "write_through_create",
    "write_through_comment",
    "write_through_edit",
    "write_through_close",
    "write_through_from_queue",
    "main",
    # Note: Underscore-prefixed functions (_add_mail_label_if_needed, etc.)
    # are implementation details. Tests can import them directly if needed.
]


def main() -> None:
    """Entry point: wrapper around gh that adds AI identity to issues."""
    args = sys.argv[1:]

    if not args:
        real_gh = get_real_gh()
        os.execv(real_gh, [real_gh])

    # Special command: gh issue cleanup-closed [--dry-run]
    # Clean workflow labels from issues auto-closed via Fixes #N
    if args[:2] == ["issue", "cleanup-closed"]:
        dry_run = "--dry-run" in args
        real_gh = get_real_gh()
        cleaned = cleanup_closed_issues_workflow_labels(real_gh, dry_run=dry_run)
        if cleaned:
            print(f"Cleaned {len(cleaned)} issue(s)", file=sys.stderr)
        else:
            print("No closed issues with stale workflow labels", file=sys.stderr)
        sys.exit(0)

    real_gh = get_real_gh()
    parsed = parse_args(args)

    # Only process issue create/comment/edit/close
    if parsed["command"] != "issue" or parsed["subcommand"] not in (
        "create",
        "comment",
        "edit",
        "close",
    ):
        os.execv(real_gh, [real_gh] + args)

    # Handle close: role enforcement + label cleanup
    if parsed["subcommand"] == "close":
        _handle_close(real_gh, args, parsed)

    # Handle edit with issue number: label rules + claiming
    if parsed["subcommand"] == "edit" and parsed["issue_number"]:
        _handle_edit(real_gh, args, parsed)

    # Pass through other edit commands unchanged (with rate limit queue)
    if parsed["subcommand"] == "edit":
        sys.exit(_execute_gh_with_queue(real_gh, args, parsed))

    # Warn about DashNews issues - prefer Discussions instead (#1668)
    # DashNews uses GitHub Discussions (News category) for announcements
    # Warning only (not blocking) - Discussions API might be unavailable as fallback
    # Use exact match to avoid false positives on repos like "my-dashnews-fork"
    if parsed["subcommand"] == "create":
        repo = parsed["repo"] or get_current_repo(real_gh)
        repo_lower = repo.lower() if repo else ""
        if repo_lower.endswith("/dashnews") or repo_lower == "dashnews":
            print(file=sys.stderr)
            print(
                "⚠️  WARNING: DashNews typically uses Discussions, not Issues.",
                file=sys.stderr,
            )
            print(
                "   Consider using gh_discussion.py to create a Discussion:",
                file=sys.stderr,
            )
            print(
                "   python3 ai_template_scripts/gh_discussion.py create "
                f'--repo {PROJECT_OWNER}/dashnews --category News --title "..." --body "..."',
                file=sys.stderr,
            )
            print(
                f"   See: https://github.com/{PROJECT_OWNER}/dashnews",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            # Continue execution - don't block

    # Enforce USER-only labels for issue create
    if parsed["subcommand"] == "create" and parsed["create_labels"]:
        role = get_effective_role()
        _check_user_only_labels(parsed["create_labels"], role, "create")
        _check_malformed_in_progress_labels(parsed["create_labels"])
        _check_single_p_label_on_create(parsed["create_labels"])

    # Create/comment: process body and title
    has_body = parsed["body_index"] is not None or parsed["body_value"] is not None
    if not has_body:
        sys.exit(_execute_gh_with_queue(real_gh, args, parsed))

    identity = get_identity()
    args = _process_body_and_title(args, parsed, identity)
    args = _add_mail_label_if_needed(args, parsed, real_gh)

    sys.exit(_execute_gh_with_queue(real_gh, args, parsed))

#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
gh_post.py - Wrapper for gh that adds AI identity to issues and comments

Handles:
- gh issue create/comment - adds identity header and signature
- gh issue edit --add-label in-progress - auto-comments to claim with identity
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

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Labels that USER can protect from AI removal
# When USER adds these labels, a tracking comment is added
# When non-USER tries to remove them, the operation is blocked
PROTECTED_LABELS = {"urgent"}

# Workflow labels that should be cleared on issue close
WORKFLOW_LABELS = (
    "needs-review",
    "do-audit",
    "in-progress",
    "blocked",
    "tracking",
    "urgent",
)

FROM_HEADER_RE = re.compile(r"^\*\*FROM:\*\*")
METADATA_LINE_RE = re.compile(r"^(Project|Role|Iteration|Session|Commit|Timestamp):")
COMPACT_SIG_RE = re.compile(r"^[\w_-]+ \| ")
TITLE_PREFIX_RE = re.compile(r"^\[[^\]]*\](\[[A-Za-z]\])?(\d+)?\s*")


def get_real_gh() -> str:
    """Find the real gh binary, skipping our wrapper."""
    script_dir = Path(__file__).resolve().parent
    wrapper_path = (script_dir / "bin" / "gh").resolve()

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
    """Extract project name from common git remote URL formats, normalizing .git suffixes and extra slashes."""
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

    Looks for worker_logs/.iteration_{role} relative to project root.
    Returns empty string if not found.
    """
    role_lower = role.lower()
    # Try relative to current directory (project root)
    iteration_file = Path("worker_logs") / f".iteration_{role_lower}"
    if iteration_file.is_file():
        try:
            return iteration_file.read_text().strip()
        except Exception:
            pass
    return ""


def get_identity() -> dict:
    """Get AI identity from env vars or derive from git.

    For iteration, falls back to .iteration_{role} file if env var is missing or 1.
    This handles USER sessions and looper sessions where AI_ITERATION may not be set correctly.
    """
    identity = {
        "project": os.environ.get("AI_PROJECT", ""),
        "role": os.environ.get("AI_ROLE", "USER"),
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
        except Exception:
            identity["project"] = Path.cwd().name

    return identity


def get_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "-"


def get_current_repo(real_gh: str) -> str:
    """Get current repo name."""
    try:
        return subprocess.check_output(
            [
                real_gh,
                "repo",
                "view",
                "--json",
                "nameWithOwner",
                "-q",
                ".nameWithOwner",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def build_header(identity: dict) -> str:
    """Build FROM header line."""
    header = f"**FROM:** {identity['project']}"
    if identity["iteration"]:
        header += f" [{identity['role']}]{identity['iteration']}"
    else:
        header += f" [{identity['role']}]"
    return header


def is_label_user_protected(real_gh: str, issue_number: str, label: str) -> bool:
    """Check if a label was set by USER (has protection marker in comments)."""
    marker = f"<!-- USER_LABEL:{label} -->"
    try:
        # Get all issue comments
        comments = subprocess.check_output(
            [
                real_gh,
                "issue",
                "view",
                issue_number,
                "--json",
                "comments",
                "-q",
                ".comments[].body",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return marker in comments
    except Exception:
        return False


def add_user_label_marker(real_gh: str, issue_number: str, label: str) -> None:
    """Add a tracking comment when USER adds a protected label."""
    marker = f"<!-- USER_LABEL:{label} -->"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comment = f"{marker}\n🔒 USER protected label `{label}` at {timestamp}"
    subprocess.run(
        [real_gh, "issue", "comment", issue_number, "--body", comment],
        stderr=subprocess.DEVNULL,
    )


def cleanup_workflow_labels(
    real_gh: str, issue_number: str, repo: str | None, role: str
) -> None:
    """Remove workflow labels when closing issues."""
    if not issue_number:
        return

    labels_to_remove = []
    for label in WORKFLOW_LABELS:
        if label in PROTECTED_LABELS and role != "USER":
            if is_label_user_protected(real_gh, issue_number, label):
                continue
        labels_to_remove.append(label)

    if not labels_to_remove:
        return

    cmd = [real_gh, "issue", "edit", issue_number]
    if repo:
        cmd.extend(["--repo", repo])
    for label in labels_to_remove:
        cmd.extend(["--remove-label", label])

    subprocess.run(cmd, check=False)


def build_signature(identity: dict) -> str:
    """Build compact signature line."""
    commit = get_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sig = f"{identity['project']} | {identity['role']}"
    if identity["iteration"]:
        sig += f" #{identity['iteration']}"
    if identity["session"]:
        sig += f" | {identity['session'][:8]}"
    sig += f" | {commit} | {timestamp}"

    return f"---\n{sig}"


def clean_body(body: str) -> str:
    """Clean body - remove identity markers robustly."""
    # Normalize line endings (Windows \r\n -> \n)
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    lines = body.split("\n")

    # Remove FROM header at start (handles multiple formats)
    # **FROM:** project [ROLE]123
    # **FROM:** project [ROLE]
    while lines and FROM_HEADER_RE.match(lines[0]):
        lines = lines[1:]
        # Remove blank line after FROM
        if lines and not lines[0].strip():
            lines = lines[1:]

    # Find signature block at end
    # Look for --- followed by signature-like content
    # Old format: Project: / Role: / Iteration: / etc.
    # New format: word | word | word
    # IMPORTANT: Only match if the signature is truly at the END (no content after)
    sig_start = None
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line == "---":
            # Check if what follows looks like a signature
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                is_signature = False
                sig_end = i + 1  # End of signature block (exclusive)

                # Old format signatures (multi-line)
                if METADATA_LINE_RE.match(next_line):
                    is_signature = True
                    # Find end of old-format signature block
                    for j in range(i + 1, len(lines)):
                        if METADATA_LINE_RE.match(lines[j]):
                            sig_end = j + 1
                        elif lines[j].strip():
                            # Non-empty line that's not signature field = content after
                            break
                # New compact format: word | word | ... (single line)
                elif COMPACT_SIG_RE.match(next_line):
                    is_signature = True
                    sig_end = i + 2  # --- line + signature line

                if is_signature:
                    # Check if there's any real content after the signature
                    has_content_after = False
                    for j in range(sig_end, len(lines)):
                        if lines[j].strip():
                            has_content_after = True
                            break

                    if not has_content_after:
                        sig_start = i
                        break
            # Lone --- at end with nothing after
            elif i == len(lines) - 1:
                sig_start = i
                break

    if sig_start is not None:
        lines = lines[:sig_start]

    # Trim trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Trim leading blank lines
    while lines and not lines[0].strip():
        lines = lines[1:]

    return "\n".join(lines)


def fix_title(title: str, identity: dict) -> str:
    """Ensure title has [project] prefix, don't duplicate.

    Removes existing project prefix and optional role prefix like [proj][W]123
    but preserves other bracketed content like [URGENT] or [WIP].
    """
    # Pattern: [project] optionally followed by [X] or [X]123 where X is a letter
    # This handles:
    #   [proj] Title           -> Title
    #   [proj][W] Title        -> Title
    #   [proj][W]123 Title     -> Title
    #   [proj] [URGENT] Title  -> [URGENT] Title (preserves [URGENT])
    title = TITLE_PREFIX_RE.sub("", title)
    return f"[{identity['project']}] {title}"


def process_body(body: str, identity: dict) -> str:
    """Clean and add identity to body."""
    body = clean_body(body)
    header = build_header(identity)
    signature = build_signature(identity)
    return f"{header}\n\n{body}\n\n{signature}"


def parse_args(args: list[str]) -> dict:
    """Parse gh command arguments.

    Handles both --flag value and --flag=value formats, plus short flags.
    Short flags supported: -b (body), -t (title), -R (repo), -l (label)
    Note: -F (body-file) passes through without identity injection.
    """
    result = {
        "command": args[0] if args else "",
        "subcommand": args[1] if len(args) > 1 else "",
        "body_index": None,
        "body_value": None,  # For --body=value format
        "title_index": None,
        "title_value": None,  # For --title=value format
        "repo": None,
        "has_mail_label": False,
        "adding_in_progress": False,  # For issue edit --add-label in-progress
        "issue_number": None,  # For issue edit/comment N
        "add_labels": [],  # All labels being added
        "remove_labels": [],  # All labels being removed
    }

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle --flag=value format
        if arg.startswith("--body="):
            result["body_value"] = arg[7:]  # len("--body=") = 7
        elif arg.startswith("--title="):
            result["title_value"] = arg[8:]  # len("--title=") = 8
        elif arg.startswith("--repo="):
            result["repo"] = arg[7:]
        elif arg.startswith("--label="):
            if arg[8:] == "mail":
                result["has_mail_label"] = True
        elif arg.startswith("--add-label="):
            label = arg[12:]  # len("--add-label=") = 12
            result["add_labels"].append(label)
            if label == "in-progress":
                result["adding_in_progress"] = True
        elif arg.startswith("--remove-label="):
            label = arg[15:]  # len("--remove-label=") = 15
            result["remove_labels"].append(label)
        # Handle --flag value and short flag formats
        elif arg in ("--body", "-b") and i + 1 < len(args):
            result["body_index"] = i + 1
        elif arg in ("--title", "-t") and i + 1 < len(args):
            result["title_index"] = i + 1
        elif arg in ("--repo", "-R") and i + 1 < len(args):
            result["repo"] = args[i + 1]
        elif arg in ("--label", "-l") and i + 1 < len(args) and args[i + 1] == "mail":
            result["has_mail_label"] = True
        elif arg == "--add-label" and i + 1 < len(args):
            label = args[i + 1]
            result["add_labels"].append(label)
            if label == "in-progress":
                result["adding_in_progress"] = True
        elif arg == "--remove-label" and i + 1 < len(args):
            result["remove_labels"].append(args[i + 1])
        # Capture issue number (positional arg after subcommand, must be numeric)
        elif arg.isdigit() and result["issue_number"] is None:
            result["issue_number"] = arg
        i += 1

    return result


def main():
    args = sys.argv[1:]

    if not args:
        # No args, pass through
        real_gh = get_real_gh()
        os.execv(real_gh, [real_gh])

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

    # ENFORCE: Only MANAGER or USER can close issues
    # USER = human interactive session, MANAGER = AI tech lead
    # WORKER/PROVER/RESEARCHER must use "Part of #N" and let Manager close
    if parsed["subcommand"] == "close":
        role = os.environ.get("AI_ROLE", "USER")
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
        if parsed["issue_number"]:
            cleanup_workflow_labels(
                real_gh, parsed["issue_number"], parsed["repo"], role
            )
        # Manager/User can close - pass through
        os.execv(real_gh, [real_gh] + args)

    # Handle issue edit operations
    if parsed["subcommand"] == "edit" and parsed["issue_number"]:
        role = os.environ.get("AI_ROLE", "USER")

        # ENFORCE: Only WORKER can add do-audit label
        # do-audit is Worker's self-audit checkpoint - other roles skip directly to needs-review
        if "do-audit" in parsed["add_labels"] and role != "WORKER":
            print(file=sys.stderr)
            print(
                "❌ ERROR: Only WORKER role can add 'do-audit' label", file=sys.stderr
            )
            print(f"   Current role: {role}", file=sys.stderr)
            print(
                "   Workflow: Worker → do-audit → needs-review → Manager closes",
                file=sys.stderr,
            )
            print("   Other roles: → needs-review → Manager closes", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)

        # ENFORCE: Non-USER cannot remove USER-protected labels
        if role != "USER" and parsed["remove_labels"]:
            for label in parsed["remove_labels"]:
                if label in PROTECTED_LABELS:
                    if is_label_user_protected(real_gh, parsed["issue_number"], label):
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
                        print(
                            "   Only USER can remove labels they set.", file=sys.stderr
                        )
                        print(file=sys.stderr)
                        sys.exit(1)

        # Execute the edit command
        result = subprocess.run([real_gh] + args)
        if result.returncode != 0:
            sys.exit(result.returncode)

        # Track: USER adding protected labels
        if role == "USER" and parsed["add_labels"]:
            for label in parsed["add_labels"]:
                if label in PROTECTED_LABELS:
                    # Check if already protected (avoid duplicate markers)
                    if not is_label_user_protected(
                        real_gh, parsed["issue_number"], label
                    ):
                        add_user_label_marker(real_gh, parsed["issue_number"], label)

        # Add claiming comment for in-progress
        if parsed["adding_in_progress"]:
            identity = get_identity()
            claim_body = process_body("Claiming this issue.", identity)
            subprocess.run(
                [
                    real_gh,
                    "issue",
                    "comment",
                    parsed["issue_number"],
                    "--body",
                    claim_body,
                ]
            )

        sys.exit(0)

    # Pass through other edit commands unchanged
    if parsed["subcommand"] == "edit":
        os.execv(real_gh, [real_gh] + args)

    # Check if we have a body (either --body value or --body=value)
    has_body = parsed["body_index"] is not None or parsed["body_value"] is not None
    if not has_body:
        os.execv(real_gh, [real_gh] + args)

    identity = get_identity()

    # Process body
    if parsed["body_index"] is not None:
        original_body = args[parsed["body_index"]]
        args[parsed["body_index"]] = process_body(original_body, identity)
    elif parsed["body_value"] is not None:
        # Find and replace the --body=value argument
        for i, arg in enumerate(args):
            if arg.startswith("--body="):
                args[i] = "--body=" + process_body(parsed["body_value"], identity)
                break

    # Process title (issue create only)
    if parsed["subcommand"] == "create":
        if parsed["title_index"] is not None:
            original_title = args[parsed["title_index"]]
            args[parsed["title_index"]] = fix_title(original_title, identity)
        elif parsed["title_value"] is not None:
            # Find and replace the --title=value argument
            for i, arg in enumerate(args):
                if arg.startswith("--title="):
                    args[i] = "--title=" + fix_title(parsed["title_value"], identity)
                    break

    # Add mail label for cross-repo issues
    if parsed["subcommand"] == "create" and parsed["repo"]:
        current_repo = get_current_repo(real_gh)
        if (
            current_repo
            and parsed["repo"] != current_repo
            and not parsed["has_mail_label"]
        ):
            args.extend(["--label", "mail"])

    # Execute
    os.execv(real_gh, [real_gh] + args)


if __name__ == "__main__":
    main()

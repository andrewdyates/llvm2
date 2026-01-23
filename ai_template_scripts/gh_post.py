#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
gh_post.py - Wrapper for gh that adds AI identity to issues and comments

Handles:
- gh issue create/comment - adds identity header and signature
- gh issue edit --add-label in-progress - auto-comments to claim with identity
- gh issue close - BLOCKED unless MANAGER or USER role (enforces issue closure policy)
- Smart deduplication of existing identity markers via regex
- Cross-repo issues auto-labeled with "mail"

Copyright 2026 Dropbox, Inc.
Licensed under the Apache License, Version 2.0
"""

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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


def get_identity() -> dict:
    """Get AI identity from env vars or derive from git."""
    identity = {
        "project": os.environ.get("AI_PROJECT", ""),
        "role": os.environ.get("AI_ROLE", "USER"),
        "iteration": os.environ.get("AI_ITERATION", ""),
        "session": os.environ.get("AI_SESSION", ""),
    }

    # Derive project from git if not set
    if not identity["project"]:
        try:
            url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            # Handle various URL formats:
            # https://github.com/owner/repo.git
            # https://github.com/owner/repo/  (trailing slash)
            # git@github.com:owner/repo.git
            url = url.rstrip("/").rstrip(".git").rstrip("/")
            identity["project"] = url.split("/")[-1]
        except Exception:
            identity["project"] = Path.cwd().name

    return identity


def get_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "-"


def get_current_repo(real_gh: str) -> str:
    """Get current repo name."""
    try:
        return subprocess.check_output(
            [real_gh, "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            stderr=subprocess.DEVNULL, text=True
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
    while lines and re.match(r"^\*\*FROM:\*\*", lines[0]):
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
                if re.match(r"^(Project|Role|Iteration|Session|Commit|Timestamp):", next_line):
                    is_signature = True
                    # Find end of old-format signature block
                    for j in range(i + 1, len(lines)):
                        if re.match(r"^(Project|Role|Iteration|Session|Commit|Timestamp):", lines[j]):
                            sig_end = j + 1
                        elif lines[j].strip():
                            # Non-empty line that's not signature field = content after
                            break
                # New compact format: word | word | ... (single line)
                elif re.match(r"^[\w_-]+ \| ", next_line):
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
    title = re.sub(r"^\[[^\]]*\](\[[A-Za-z]\])?(\d+)?\s*", "", title)
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
            if arg[12:] == "in-progress":
                result["adding_in_progress"] = True
        # Handle --flag value and short flag formats
        elif arg in ("--body", "-b") and i + 1 < len(args):
            result["body_index"] = i + 1
        elif arg in ("--title", "-t") and i + 1 < len(args):
            result["title_index"] = i + 1
        elif arg in ("--repo", "-R") and i + 1 < len(args):
            result["repo"] = args[i + 1]
        elif arg in ("--label", "-l") and i + 1 < len(args) and args[i + 1] == "mail":
            result["has_mail_label"] = True
        elif arg == "--add-label" and i + 1 < len(args) and args[i + 1] == "in-progress":
            result["adding_in_progress"] = True
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
    if parsed["command"] != "issue" or parsed["subcommand"] not in ("create", "comment", "edit", "close"):
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
            print("   File an issue or use 'Part of #N' in commits instead.", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)
        # Manager/User can close - pass through
        os.execv(real_gh, [real_gh] + args)

    # Handle issue edit with --add-label in-progress (claiming)
    if parsed["subcommand"] == "edit" and parsed["adding_in_progress"] and parsed["issue_number"]:
        identity = get_identity()
        # Execute the edit command first
        result = subprocess.run([real_gh] + args)
        if result.returncode == 0:
            # Add a claiming comment
            claim_body = process_body("Claiming this issue.", identity)
            subprocess.run([real_gh, "issue", "comment", parsed["issue_number"], "--body", claim_body])
        sys.exit(result.returncode)

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
        if current_repo and parsed["repo"] != current_repo and not parsed["has_mail_label"]:
            args.extend(["--label", "mail"])

    # Execute
    os.execv(real_gh, [real_gh] + args)


if __name__ == "__main__":
    main()

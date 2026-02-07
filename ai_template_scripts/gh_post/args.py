# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Argument parsing helpers for gh_post."""

from ai_template_scripts.gh_post.labels import IN_PROGRESS_LABEL_PREFIX
from ai_template_scripts.gh_post.validation import _is_p_label

_GLOBAL_FLAGS_WITH_VALUES = frozenset(
    (
        "-R",
        "--repo",
        "-C",
        "--cwd",
        "--hostname",
        "--config",
    )
)


def _is_global_flag_with_inline_value(arg: str) -> bool:
    """Return True if arg is a global flag with an inline value."""
    if arg.startswith(("-R=", "-C=")):
        return True
    if arg.startswith("--") and "=" in arg:
        name = arg.split("=", 1)[0]
        return name in _GLOBAL_FLAGS_WITH_VALUES
    return False


def _find_command_index(args: list[str]) -> int | None:
    """Find the command index, skipping global flags and their values."""
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in _GLOBAL_FLAGS_WITH_VALUES:
            skip_next = True
            continue
        if _is_global_flag_with_inline_value(arg):
            continue
        if arg.startswith("-"):
            continue
        return idx
    return None


def _find_subcommand_index(args: list[str], command_index: int | None) -> int | None:
    """Find subcommand index after command, skipping global flags and values."""
    if command_index is None:
        return None
    skip_next = False
    for idx in range(command_index + 1, len(args)):
        arg = args[idx]
        if skip_next:
            skip_next = False
            continue
        if arg in _GLOBAL_FLAGS_WITH_VALUES:
            skip_next = True
            continue
        if _is_global_flag_with_inline_value(arg):
            continue
        if arg.startswith("-"):
            continue
        return idx
    return None


def parse_args(args: list[str]) -> dict:
    """Parse gh command arguments.

    Handles both --flag value and --flag=value formats, plus short flags.
    Short flags supported: -b (body), -t (title), -R (repo), -l (label)
    Note: -F (body-file) passes through without identity injection.

    Global flags (like -R/--repo) can appear before the command, so we scan
    for the actual command/subcommand positions to correctly identify issues.
    """
    add_labels: list[str] = []
    remove_labels: list[str] = []
    create_labels: list[str] = []  # Labels for issue create (--label, -l)

    command_idx = _find_command_index(args)
    subcommand_idx = _find_subcommand_index(args, command_idx)

    result: dict = {
        "command": args[command_idx] if command_idx is not None else "",
        "subcommand": args[subcommand_idx] if subcommand_idx is not None else "",
        "body_index": None,
        "body_value": None,  # For --body=value format
        "title_index": None,
        "title_value": None,  # For --title=value format
        "comment_index": None,  # For close --comment value
        "comment_value": None,  # For close --comment=value
        "reason_value": None,  # For close --reason=value (completed|not planned)
        "repo": None,
        "has_mail_label": False,
        "has_p_label": False,  # True if any P0/P1/P2/P3 label present
        "adding_in_progress": False,  # For issue edit --add-label in-progress (legacy in-progress-* supported)
        "issue_number": None,  # For issue edit/comment N
        "add_labels": add_labels,  # All labels being added (edit)
        "remove_labels": remove_labels,  # All labels being removed (edit)
        "create_labels": create_labels,  # Labels for issue create
    }
    track_issue_number = result["command"] == "issue" and subcommand_idx is not None

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
            label = arg[8:]  # len("--label=") = 8
            result["create_labels"].append(label)
            if label == "mail":
                result["has_mail_label"] = True
            if _is_p_label(label):
                result["has_p_label"] = True
        elif arg.startswith("--add-label="):
            label = arg[12:]  # len("--add-label=") = 12
            result["add_labels"].append(label)
            if _is_in_progress_label(label):
                result["adding_in_progress"] = True
            if _is_p_label(label):
                result["has_p_label"] = True
        elif arg.startswith("--remove-label="):
            label = arg[15:]  # len("--remove-label=") = 15
            result["remove_labels"].append(label)
        elif arg.startswith("--comment="):
            result["comment_value"] = arg[10:]  # len("--comment=") = 10
        elif arg.startswith("--reason="):
            result["reason_value"] = arg[9:]  # len("--reason=") = 9
        # Handle --flag value and short flag formats
        elif arg in ("--body", "-b") and i + 1 < len(args):
            result["body_index"] = i + 1
        elif arg in ("--title", "-t") and i + 1 < len(args):
            result["title_index"] = i + 1
        elif arg in ("--repo", "-R") and i + 1 < len(args):
            result["repo"] = args[i + 1]
        elif arg in ("--label", "-l") and i + 1 < len(args):
            label = args[i + 1]
            result["create_labels"].append(label)
            if label == "mail":
                result["has_mail_label"] = True
            if _is_p_label(label):
                result["has_p_label"] = True
        elif arg == "--add-label" and i + 1 < len(args):
            label = args[i + 1]
            result["add_labels"].append(label)
            if _is_in_progress_label(label):
                result["adding_in_progress"] = True
            if _is_p_label(label):
                result["has_p_label"] = True
        elif arg == "--remove-label" and i + 1 < len(args):
            result["remove_labels"].append(args[i + 1])
        elif arg in ("--comment", "-c") and i + 1 < len(args):
            result["comment_index"] = i + 1
        elif arg in ("--reason", "-r") and i + 1 < len(args):
            result["reason_value"] = args[i + 1]
        # Capture issue number (positional arg after subcommand, must be numeric)
        # Must be after the subcommand position to avoid capturing numeric repo values
        elif (
            arg.isdigit()
            and result["issue_number"] is None
            and track_issue_number
            and subcommand_idx is not None
            and i > subcommand_idx
        ):
            result["issue_number"] = arg
        i += 1

    return result


def _is_in_progress_label(label: str) -> bool:
    """Return True if label indicates in-progress (including worker-specific)."""
    return label == "in-progress" or label.startswith(IN_PROGRESS_LABEL_PREFIX)


def _strip_in_progress_additions(args: list[str]) -> list[str]:
    """Remove in-progress add-label flags from gh args."""
    filtered: list[str] = []
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--add-label" and idx + 1 < len(args):
            label = args[idx + 1]
            if _is_in_progress_label(label):
                skip_next = True
                continue
        if arg.startswith("--add-label="):
            label = arg.split("=", 1)[1]
            if _is_in_progress_label(label):
                continue
        filtered.append(arg)
    return filtered


def _has_edit_modifiers(args: list[str]) -> bool:
    """Return True if gh issue edit args contain non-repo modifiers."""
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("issue", "edit"):
            continue
        if idx >= 2 and arg.isdigit():
            continue
        if arg in ("--repo", "-R"):
            skip_next = True
            continue
        if arg.startswith("--repo="):
            continue
        if arg.startswith("-"):
            return True
    return False


def _extract_title_body(args: list[str], parsed: dict) -> tuple[str, str]:
    """Extract title and body values from args using parsed indices."""
    title = ""
    body = ""

    # Title: check index first, then --title= format
    if parsed.get("title_index") is not None:
        title = args[parsed["title_index"]]
    elif parsed.get("title_value"):
        title = parsed["title_value"]
    else:
        for arg in args:
            if arg.startswith("--title="):
                title = arg[8:]
                break

    # Body: check index first, then --body= format
    if parsed.get("body_index") is not None:
        body = args[parsed["body_index"]]
    elif parsed.get("body_value"):
        body = parsed["body_value"]
    else:
        for arg in args:
            if arg.startswith("--body="):
                body = arg[7:]
                break
            if arg.startswith("-b="):
                body = arg[3:]
                break

    return title, body

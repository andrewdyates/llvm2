# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Validation helpers for gh_post."""

import re
import subprocess
import sys

from ai_template_scripts.gh_post.labels import (
    IN_PROGRESS_LABEL_PREFIX,
    P_LABELS,
    USER_ONLY_LABELS,
)


def _is_malformed_in_progress_label(label: str) -> str | None:
    """Check if an in-progress label is malformed.

    Returns error message if malformed, None if valid.

    Valid: in-progress, in-progress-W1, in-progress-P2, in-progress-R1, in-progress-M1, etc.
    Invalid: in-progress-W (missing ID), in-progress- (trailing dash)

    See #866 for context.
    """
    if label == "in-progress":
        return None  # Valid generic label
    if label.startswith(IN_PROGRESS_LABEL_PREFIX):
        suffix = label[len(IN_PROGRESS_LABEL_PREFIX) :]
        # Valid suffixes: W1, W2, P1, R1, M1, etc. (role letter + number)
        if not re.match(r"^[WPRM]\d+$", suffix):
            return (
                f"Malformed in-progress label: '{label}'\n"
                f"  Expected: 'in-progress' or 'in-progress-XN' (X=W/P/R/M, N=number)\n"
                f"  Got suffix: '{suffix}'\n"
                f"\n"
                f"  Fix: Set AI_WORKER_ID before claiming, or use 'in-progress' directly.\n"
                f"  Example: gh issue edit N --add-label in-progress"
            )
    return None


def _is_p_label(label: str) -> bool:
    """Return True if label is a priority level (P0/P1/P2/P3)."""
    return label in P_LABELS


def _get_p_labels_from_list(labels: list[str]) -> list[str]:
    """Extract all P* labels from a list of labels."""
    return [label for label in labels if _is_p_label(label)]


def _check_single_p_label_on_create(labels: list[str]) -> None:
    """Block if multiple P* labels are being added on issue create.

    Issues must have exactly one priority level. Multiple P* labels
    indicate confusion about severity.

    Args:
        labels: List of labels being added

    Exits with error if multiple P* labels detected.
    """
    p_labels = _get_p_labels_from_list(labels)
    if len(p_labels) > 1:
        print(file=sys.stderr)
        print(
            f"❌ ERROR: Cannot add multiple priority labels: {', '.join(p_labels)}",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print(
            "   Issues must have exactly ONE priority level (P0/P1/P2/P3).",
            file=sys.stderr,
        )
        print("   Choose the correct severity:", file=sys.stderr)
        print(
            "   - P0: System compromised (soundness, security, data corruption)",
            file=sys.stderr,
        )
        print("   - P1: Blocks critical path", file=sys.stderr)
        print("   - P2: Normal priority", file=sys.stderr)
        print("   - P3: Low priority / nice to have", file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)


def _check_user_only_labels(labels: list[str], role: str, action: str) -> None:
    """Block non-USER from adding USER-only labels.

    Args:
        labels: List of labels being added
        role: Current AI role
        action: "create" or "edit" for error message context

    Exits with error if violation detected.
    """
    if role == "USER":
        return

    violations = [label for label in labels if label in USER_ONLY_LABELS]
    if not violations:
        return

    print(file=sys.stderr)
    for label in violations:
        print(f"❌ ERROR: Only USER can add '{label}' label", file=sys.stderr)
    print(f"   Current role: {role}", file=sys.stderr)
    print(file=sys.stderr)
    print("   These labels are USER prerogatives:", file=sys.stderr)
    print("   - 'urgent': USER controls scheduling priority", file=sys.stderr)
    print("   - 'P0': USER confirms system-compromised severity", file=sys.stderr)
    print(file=sys.stderr)
    print("   Instead:", file=sys.stderr)
    print(
        "   - File the issue without these labels (describe urgency/severity in body)",
        file=sys.stderr,
    )
    print("   - USER will triage and add labels as appropriate", file=sys.stderr)
    print(file=sys.stderr)
    sys.exit(1)


def _check_malformed_in_progress_labels(labels: list[str]) -> None:
    """Block malformed in-progress labels like 'in-progress-W' (missing ID).

    Args:
        labels: List of labels being added

    Exits with error if malformed label detected.
    See #866 for context.
    """
    for label in labels:
        error = _is_malformed_in_progress_label(label)
        if error:
            print(file=sys.stderr)
            print(f"❌ ERROR: {error}", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)


def _has_fix_commit(issue_number: str) -> tuple[bool, str]:
    """Check if a 'Fixes #N' commit exists for the given issue.

    Uses word boundary regex to prevent false positives (e.g., searching for #1
    should not match #10, #100, etc.).

    Returns:
        Tuple of (has_fix, commit_hash_or_empty)
    """
    # Validate issue_number is a positive integer string
    if not issue_number or not issue_number.strip().isdigit():
        return False, ""

    issue_number = issue_number.strip()

    try:
        # Search for commits with "Fixes #N" (case insensitive via -i flag)
        # Use word boundary \b to prevent #1 matching #10, #100, etc.
        # Pattern: Fixes #N followed by word boundary (comma, space, EOL, etc.)
        result = subprocess.check_output(
            [
                "git",
                "log",
                "--all",
                "--oneline",
                "-i",  # Case insensitive
                "-E",  # Extended regex
                f"--grep=Fixes #{issue_number}([^0-9]|$)",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if result:
            # Return first matching commit hash
            first_line = result.split("\n")[0]
            commit_hash = first_line.split()[0] if first_line else ""
            return True, commit_hash
        return False, ""
    except Exception:
        # If git fails, don't block (might be in test environment)
        return True, "unknown"

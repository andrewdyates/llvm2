# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Body and title processing helpers for gh_post."""

import re
import sys

from ai_template_scripts.gh_post.identity import build_header, build_signature
from ai_template_scripts.gh_post.labels import P_LABELS

FROM_HEADER_RE = re.compile(r"^\*\*FROM:\*\*")
METADATA_LINE_RE = re.compile(r"^(Project|Role|Iteration|Session|Commit|Timestamp):")
COMPACT_SIG_RE = re.compile(r"^[\w_-]+ \| ")
TITLE_PREFIX_RE = re.compile(r"^\[[^\]]*\](\[[A-Za-z]\])?(\d+)?\s*")

PRIORITY_PREFIX_RE = re.compile(r"^P[0-3](?=[:\\s-])", re.IGNORECASE)


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


def _split_leading_tags(title: str) -> tuple[list[str], str]:
    """Split leading [tags] and return (tags, remainder) for warning checks."""
    tags: list[str] = []
    remainder = title.lstrip()
    while remainder.startswith("["):
        end = remainder.find("]")
        if end == -1:
            break
        tag = remainder[1:end].strip()
        tags.append(tag)
        remainder = remainder[end + 1 :].lstrip()
        # [W]123 role prefix should not hide priority tags that follow.
        if len(tag) == 1 and tag.isalpha():
            i = 0
            while i < len(remainder) and remainder[i].isdigit():
                i += 1
            if i:
                remainder = remainder[i:].lstrip()
    return tags, remainder


def _is_priority_tag(tag: str) -> bool:
    normalized = tag.strip().upper().rstrip(":")
    return normalized in P_LABELS


def _title_has_priority_prefix(title: str) -> bool:
    tags, remainder = _split_leading_tags(title)
    if any(_is_priority_tag(tag) for tag in tags):
        return True
    remainder = remainder.lstrip()
    if remainder.upper() in P_LABELS:
        return True
    return PRIORITY_PREFIX_RE.match(remainder) is not None


def _warn_priority_in_title(title: str | None) -> None:
    """Warn if title contains priority prefix (P0:, P1:, etc.).

    Priority should be set via labels, not in the title text.
    See issue #1331.
    """
    if title is None:
        return
    if _title_has_priority_prefix(title):
        print(file=sys.stderr)
        print(
            "WARNING: Issue title contains priority prefix (e.g., 'P0:', 'P1:')",
            file=sys.stderr,
        )
        print(
            f"   Title: {title[:60]}{'...' if len(title) > 60 else ''}", file=sys.stderr
        )
        print(file=sys.stderr)
        print(
            "   Priority should be set via labels, not in the title.", file=sys.stderr
        )
        print("   Use: --label P0  or  --label P1  etc.", file=sys.stderr)
        print(file=sys.stderr)


def fix_title(title: str, identity: dict) -> str:
    """Ensure title has [project] prefix, don't duplicate.

    Removes existing project prefix and optional role prefix like [proj][W]123
    but preserves other bracketed content like [URGENT] or [WIP].
    Warns if title contains priority prefix (P0:, P1:, etc.) - use labels instead.
    """
    # Warn about priority prefixes (#1331) before stripping prefixes.
    _warn_priority_in_title(title)

    # Strip existing project/role prefix
    # Pattern: [project] optionally followed by [X] or [X]123 where X is a letter
    # This handles:
    #   [proj] Title           -> Title
    #   [proj][W] Title        -> Title
    #   [proj][W]123 Title     -> Title
    #   [proj] [URGENT] Title  -> [URGENT] Title (preserves [URGENT])
    stripped_title = TITLE_PREFIX_RE.sub("", title)

    return f"[{identity['project']}] {stripped_title}"


def process_body(body: str, identity: dict) -> str:
    """Clean and add identity to body."""
    body = clean_body(body)
    header = build_header(identity)
    signature = build_signature(identity)
    return f"{header}\n\n{body}\n\n{signature}"

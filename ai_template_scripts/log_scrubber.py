#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""log_scrubber.py - Sanitize secrets and PII from JSONL log files.

Public API (library usage):
    from ai_template_scripts.log_scrubber import (
        PATTERNS,             # Compiled regex patterns for secrets
        scrub_value,          # Recursively scrub secrets from a value
        scrub_jsonl_file,     # Scrub a JSONL file (file_path param)
        scrub_directory,      # Scrub all JSONL files in directory (dir_path param)
    )

CLI usage:
    # Scrub a single file (in-place)
    ./ai_template_scripts/log_scrubber.py worker_logs/worker_iter_1.jsonl

    # Scrub and output to stdout
    ./ai_template_scripts/log_scrubber.py --stdout worker_logs/worker_iter_1.jsonl

    # Scrub all logs in directory
    ./ai_template_scripts/log_scrubber.py worker_logs/

Scrubs:
    - API keys (Anthropic sk-ant-*, OpenAI sk-*, GitHub ghp_*)
    - Slack tokens (xoxb-*, xoxp-*, xoxa-*, xoxr-*, xoxs-*)
    - AWS credentials (AKIA*, secret keys)
    - Generic tokens (Bearer, Authorization headers)
    - Passwords (password=, --password, passwd patterns)
    - Email addresses (optional, off by default)
    - Home directory paths (replaces with ~)

Does NOT scrub:
    - Session IDs (needed for log correlation)
    - Timestamps (needed for debugging)
    - Tool names and commands (needed for audit)
"""

__all__ = [
    "PATTERNS",
    "scrub_value",
    "scrub_jsonl_file",
    "scrub_directory",
    "main",
]

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.path_utils import resolve_path_alias  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

# Patterns for secrets - compiled for performance
PATTERNS = [
    # Anthropic API keys
    (re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"), "[REDACTED:ANTHROPIC_KEY]"),
    # OpenAI API keys
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED:OPENAI_KEY]"),
    # GitHub tokens (36+ chars after prefix - tokens may vary in length)
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[REDACTED:GITHUB_TOKEN]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "[REDACTED:GITHUB_OAUTH]"),
    (re.compile(r"ghu_[a-zA-Z0-9]{36,}"), "[REDACTED:GITHUB_USER]"),
    (re.compile(r"ghr_[a-zA-Z0-9]{36,}"), "[REDACTED:GITHUB_REFRESH]"),
    (re.compile(r"ghs_[a-zA-Z0-9]{36,}"), "[REDACTED:GITHUB_INSTALL]"),
    # Fine-grained personal access tokens (longer format)
    (re.compile(r"github_pat_[a-zA-Z0-9_]{22,}"), "[REDACTED:GITHUB_PAT]"),
    # Slack tokens
    (re.compile(r"xox[bpars]-[a-zA-Z0-9-]{10,}"), "[REDACTED:SLACK_TOKEN]"),
    # AWS access keys
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:AWS_ACCESS_KEY]"),
    # Generic AWS secret pattern (40 char base64-ish)
    (
        re.compile(
            r'(?i)(aws_secret_access_key|secret_access_key)\s*[=:]\s*["\']?([a-zA-Z0-9+/]{40})["\']?'
        ),
        r"\1=[REDACTED:AWS_SECRET]",
    ),
    # Bearer tokens (including JWTs with dots)
    (re.compile(r"(?i)(Bearer\s+)[a-zA-Z0-9_.-]{20,}"), r"\1[REDACTED:BEARER]"),
    # Authorization headers (scrub entire header value)
    (re.compile(r"(?i)(Authorization:\s*)([^\n]+)"), r"\1[REDACTED:AUTH]"),
    # Password patterns
    (
        re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']+["\']?'),
        r"\1=[REDACTED:PASSWORD]",
    ),
    (re.compile(r"(?i)--password[=\s]+[^\s]+"), "--password=[REDACTED:PASSWORD]"),
    # Generic API key patterns
    (
        re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?[a-zA-Z0-9_-]{16,}["\']?'),
        r"\1=[REDACTED:API_KEY]",
    ),
    # Private keys (RSA, EC, DSA, or generic PKCS#8 format)
    (
        re.compile(
            r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"
            r".*?-----END (?:[A-Z]+ )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED:PRIVATE_KEY]",
    ),
]

# Email pattern - pre-compiled for performance
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Home directory pattern - dynamic based on current user
HOME_DIR = os.path.expanduser("~")
HOME_PATTERN = re.compile(re.escape(HOME_DIR))


def scrub_value(
    value: Any,
    scrub_emails: bool = False,
    scrub_home: bool = True,
    max_depth: int = 100,
) -> Any:
    """Recursively scrub secrets from a value.

    Args:
        value: The value to scrub (str, dict, list, or other).
        scrub_emails: If True, also scrub email addresses.
        scrub_home: If True, replace home directory paths with ~.
        max_depth: Maximum recursion depth to prevent stack overflow.
            Default 100 is sufficient for typical JSONL structures.

    Returns:
        The scrubbed value with secrets replaced.

    REQUIRES: max_depth >= 0 (values <=0 return input unchanged)
    ENSURES: Returns same type as input (str->str, dict->dict, list->list)
    ENSURES: All patterns in PATTERNS are applied to string values
    ENSURES: Never raises (returns original value on recursion limit)
    """
    if max_depth <= 0:
        return value
    if isinstance(value, str):
        result = value
        for pattern, replacement in PATTERNS:
            result = pattern.sub(replacement, result)
        if scrub_home:
            result = HOME_PATTERN.sub("~", result)
        if scrub_emails:
            result = EMAIL_PATTERN.sub("[REDACTED:EMAIL]", result)
        return result
    if isinstance(value, dict):
        return {
            k: scrub_value(v, scrub_emails, scrub_home, max_depth - 1)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            scrub_value(item, scrub_emails, scrub_home, max_depth - 1) for item in value
        ]
    return value


def scrub_jsonl_file(
    file_path: Path | None = None,
    output_to_stdout: bool = False,
    scrub_emails: bool = False,
    scrub_home: bool = True,
    **kwargs: Any,
) -> int:
    """Scrub a JSONL file.

    Args:
        file_path: Path to the JSONL file to scrub.
        output_to_stdout: If True, write to stdout instead of in-place.
        scrub_emails: If True, also scrub email addresses.
        scrub_home: If True, replace home directory paths with ~.
        **kwargs: Accepts deprecated 'filepath' alias for file_path.

    Returns:
        Number of lines processed.

    REQUIRES: file_path or filepath kwarg must resolve to existing file
    REQUIRES: File must be readable (raises OSError if not)
    ENSURES: Returns int >= 0 representing lines processed
    ENSURES: If not output_to_stdout, file is overwritten with scrubbed content
    ENSURES: JSON decode errors handled gracefully (scrubs as plain text)
    """
    resolved_path = resolve_path_alias(
        "file_path", "filepath", file_path, kwargs, "scrub_jsonl_file"
    )
    lines_processed = 0
    scrubbed_lines: list[str] = []

    with open(resolved_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if output_to_stdout:
                    print()
                else:
                    scrubbed_lines.append("")
                lines_processed += 1
                continue
            try:
                obj = json.loads(line)
                scrubbed = json.dumps(
                    scrub_value(obj, scrub_emails, scrub_home), ensure_ascii=False
                )
            except json.JSONDecodeError:
                # Non-JSON line - still scrub as plain text
                scrubbed = scrub_value(line, scrub_emails, scrub_home)
            if output_to_stdout:
                print(scrubbed)
            else:
                scrubbed_lines.append(scrubbed)
            lines_processed += 1

    if not output_to_stdout:
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.writelines(scrubbed_line + "\n" for scrubbed_line in scrubbed_lines)

    return lines_processed


def scrub_directory(
    dir_path: Path | None = None,
    output_to_stdout: bool = False,
    scrub_emails: bool = False,
    scrub_home: bool = True,
    **kwargs: Any,
) -> tuple[int, int]:
    """Scrub all JSONL files in directory.

    Args:
        dir_path: Directory containing JSONL files to scrub.
        output_to_stdout: If True, write to stdout instead of in-place.
        scrub_emails: If True, also scrub email addresses.
        scrub_home: If True, replace home directory paths with ~.
        **kwargs: Accepts deprecated 'dirpath' alias for dir_path.

    Returns:
        Tuple of (files_processed, lines_processed).

    REQUIRES: dir_path or dirpath kwarg must resolve to existing directory
    ENSURES: Returns tuple of (files_processed >= 0, lines_processed >= 0)
    ENSURES: Only processes *.jsonl files (ignores other extensions)
    ENSURES: Progress logged to stderr if not output_to_stdout
    """
    resolved_path = resolve_path_alias(
        "dir_path", "dirpath", dir_path, kwargs, "scrub_directory"
    )
    files_processed = 0
    total_lines = 0

    for file_path in resolved_path.glob("*.jsonl"):
        lines = scrub_jsonl_file(file_path, output_to_stdout, scrub_emails, scrub_home)
        files_processed += 1
        total_lines += lines
        if not output_to_stdout:
            print(f"Scrubbed {file_path.name}: {lines} lines", file=sys.stderr)

    return files_processed, total_lines


def main() -> None:
    """Entry point: sanitize secrets and PII from log files.

    REQUIRES: sys.argv contains valid CLI arguments
    ENSURES: Exits with 0 on success
    ENSURES: Exits with 1 if path doesn't exist or is invalid
    ENSURES: Scrubs all *.jsonl files if path is directory
    ENSURES: Scrubs single file if path is file
    """
    parser = argparse.ArgumentParser(
        description="Sanitize secrets and PII from JSONL log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("log_scrubber.py"),
    )
    parser.add_argument("path", type=Path, help="File or directory to scrub")
    parser.add_argument(
        "--stdout", action="store_true", help="Output to stdout instead of in-place"
    )
    parser.add_argument(
        "--scrub-emails", action="store_true", help="Also scrub email addresses"
    )
    parser.add_argument(
        "--no-scrub-home", action="store_true", help="Do not scrub home directory paths"
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    scrub_home = not args.no_scrub_home

    if args.path.is_file():
        lines = scrub_jsonl_file(args.path, args.stdout, args.scrub_emails, scrub_home)
        if not args.stdout:
            print(f"Scrubbed {lines} lines from {args.path}", file=sys.stderr)
    elif args.path.is_dir():
        files, lines = scrub_directory(
            args.path, args.stdout, args.scrub_emails, scrub_home
        )
        if not args.stdout:
            print(f"Scrubbed {files} files, {lines} total lines", file=sys.stderr)
    else:
        print(f"Error: {args.path} is not a file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

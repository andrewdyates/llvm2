# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Helpers for parsing issue numbers from gh JSON fields."""


def parse_issue_number(raw_number: object) -> int | None:
    """Parse issue number from gh JSON fields, skipping pending IDs."""
    if isinstance(raw_number, int) and not isinstance(raw_number, bool):
        number = raw_number
    elif isinstance(raw_number, str):
        stripped = raw_number.strip()
        if stripped.isdigit() and stripped.isascii():
            number = int(stripped)
        else:
            return None
    else:
        return None
    return number if number > 0 else None

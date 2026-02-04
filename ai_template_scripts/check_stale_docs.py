#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""check_stale_docs.py - Find markdown docs with stale verification headers.

Scans designs/, reports/, postmortems/ for files with verification headers
older than a threshold, or missing headers entirely.

Usage:
    check_stale_docs.py [--max-age-days=7] [--missing-only] [--json]
"""

from __future__ import annotations

__all__ = [
    "HEADER_PATTERN",
    "DOC_DIRS",
    "parse_header",
    "find_docs",
    "check_docs",
    "main",
]

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.version import get_version  # noqa: E402

# Verification header pattern: <!-- Verified: <commit> | <timestamp> | [<role>]<iter> -->
# Commit hash: hex digits (case-insensitive for robustness)
# Timestamp: ISO 8601 with optional fractional seconds and Z suffix
HEADER_PATTERN = re.compile(
    r"<!--\s*Verified:\s*(?P<commit>[a-fA-F0-9]+)\s*\|\s*"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s*\|\s*"
    r"(?P<role>\[\w+\]\d+)\s*-->",
    re.IGNORECASE,
)

# Directories to scan
DOC_DIRS = ["designs", "reports", "postmortems"]


def parse_header(file_path: Path) -> dict | None:
    """Parse verification header from file, returns None if not found."""
    try:
        content = file_path.read_text(encoding="utf-8")
        # Only check first 500 chars (header should be at top)
        match = HEADER_PATTERN.search(content[:500])
        if match:
            return {
                "commit": match.group("commit"),
                "timestamp": match.group("timestamp"),
                "role": match.group("role"),
            }
    except (OSError, UnicodeDecodeError):
        pass
    return None


def parse_timestamp(ts: str) -> datetime | None:
    """Parse ISO 8601 timestamp."""
    try:
        # Normalize to include timezone
        if not ts.endswith("Z") and "+" not in ts:
            ts = ts + "+00:00"
        elif ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def find_docs(base_dir: Path) -> list[Path]:
    """Find all markdown files in doc directories."""
    docs = []
    for dir_name in DOC_DIRS:
        doc_dir = base_dir / dir_name
        if doc_dir.exists():
            for md_file in doc_dir.glob("*.md"):
                # Skip template files
                if md_file.name == "TEMPLATE.md":
                    continue
                docs.append(md_file)
    return sorted(docs)


def check_docs(
    base_dir: Path, max_age_days: int, missing_only: bool
) -> tuple[list[dict], list[dict]]:
    """Check docs for staleness.

    Returns (stale_docs, missing_header_docs).
    """
    now = datetime.now(UTC)
    stale = []
    missing = []

    for doc in find_docs(base_dir):
        header = parse_header(doc)
        rel_path = doc.relative_to(base_dir)

        if header is None:
            missing.append({"path": str(rel_path)})
            continue

        if missing_only:
            continue

        ts = parse_timestamp(header["timestamp"])
        if ts is None:
            stale.append(
                {
                    "path": str(rel_path),
                    "reason": "invalid_timestamp",
                    "timestamp": header["timestamp"],
                }
            )
            continue

        age_days = (now - ts).days
        if age_days > max_age_days:
            stale.append(
                {
                    "path": str(rel_path),
                    "reason": "stale",
                    "timestamp": header["timestamp"],
                    "age_days": age_days,
                    "commit": header["commit"],
                    "role": header["role"],
                }
            )

    return stale, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find markdown docs with stale verification headers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    check_stale_docs.py                    # Find stale (>7 days) or missing
    check_stale_docs.py --max-age-days=30  # Custom staleness threshold
    check_stale_docs.py --missing-only     # Only report missing headers
    check_stale_docs.py --json             # Machine-readable output
""",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("check_stale_docs.py"),
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Maximum age in days before file is considered stale (default: 7)",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only report files missing headers, skip staleness check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--help-short",
        "-H",
        action="store_true",
        help="Show short help",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Base directory to scan (default: current working directory)",
    )

    args = parser.parse_args()

    if args.help_short:
        print("check_stale_docs.py - Find stale markdown verification headers")
        print(
            "Usage: check_stale_docs.py [--max-age-days=N] [--missing-only] [--json] [--path=DIR]"
        )
        return 0

    base_dir = args.path if args.path else Path.cwd()
    stale, missing = check_docs(base_dir, args.max_age_days, args.missing_only)

    if args.json:
        result = {
            "stale": stale,
            "missing_header": missing,
            "max_age_days": args.max_age_days,
            "total_issues": len(stale) + len(missing),
        }
        print(json.dumps(result, indent=2))
    else:
        if missing:
            print(f"\n=== Missing verification headers ({len(missing)} files) ===")
            for doc in missing:
                print(f"  {doc['path']}")

        if stale and not args.missing_only:
            print(
                f"\n=== Stale docs (>{args.max_age_days} days old) ({len(stale)} files) ==="
            )
            for doc in stale:
                if doc["reason"] == "stale":
                    print(
                        f"  {doc['path']} ({doc['age_days']} days, verified by {doc['role']})"
                    )
                else:
                    print(f"  {doc['path']} (invalid timestamp: {doc['timestamp']})")

        total = len(stale) + len(missing)
        if total > 0:
            print(f"\nTotal issues: {total}")
        else:
            print("\nAll docs have valid, recent verification headers.")

    # Exit code: 0 if no issues, 1 if issues found
    return 1 if (stale or missing) else 0


if __name__ == "__main__":
    sys.exit(main())

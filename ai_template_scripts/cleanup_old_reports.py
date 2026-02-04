#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""cleanup_old_reports.py - Remove old ephemeral reports to prevent repo bloat.

The reports/ directory accumulates manager audit reports, iteration summaries,
and other ephemeral files. Per CLAUDE.md, these are temporary but not auto-cleaned.
This script removes files older than a configurable retention period.

Part of #1695 - reports/ cleanup.

Usage:
    cleanup_old_reports.py              # Dry run - show what would be deleted
    cleanup_old_reports.py --delete     # Actually delete old reports
    cleanup_old_reports.py --days 14    # Set retention to 14 days (default: 30)
    cleanup_old_reports.py --help       # Show help

Environment:
    REPORTS_MAX_AGE_DAYS    Override default max age (days)
"""

__all__ = ["cleanup_old_reports", "main"]

import argparse
import os
import sys
import time
from pathlib import Path


def cleanup_old_reports(
    reports_dir: Path,
    max_age_days: int = 30,
    delete: bool = False,
    verbose: bool = True,
) -> tuple[int, int, int]:
    """Remove report files older than max_age_days.

    Args:
        reports_dir: Directory containing reports (typically reports/)
        max_age_days: Days before reports are considered stale
        delete: If True, actually delete; if False, dry run
        verbose: Print progress messages

    Returns:
        Tuple of (total_count, deleted_count, deleted_bytes)
    """
    if not reports_dir.exists():
        if verbose:
            print(f"Reports directory not found: {reports_dir}", file=sys.stderr)
        return (0, 0, 0)

    now = time.time()
    max_age_secs = max_age_days * 24 * 60 * 60

    total_count = 0
    deleted_count = 0
    deleted_bytes = 0

    for path in reports_dir.iterdir():
        if not path.is_file():
            continue
        if not path.suffix == ".md":
            continue

        total_count += 1
        file_age = now - path.stat().st_mtime
        file_age_days = int(file_age / 86400)

        if file_age > max_age_secs:
            file_size = path.stat().st_size
            if delete:
                if verbose:
                    print(f"Deleting: {path.name} ({file_age_days} days old)")
                path.unlink()
                deleted_count += 1
                deleted_bytes += file_size
            else:
                if verbose:
                    print(f"Would delete: {path.name} ({file_age_days} days old)")
                deleted_count += 1
                deleted_bytes += file_size

    return (total_count, deleted_count, deleted_bytes)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Remove old ephemeral reports to prevent repo bloat.",
        epilog="Reports older than --days are removed. Default: dry run (use --delete).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete old reports (default: dry run)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=int(os.environ.get("REPORTS_MAX_AGE_DAYS", "30")),
        help="Days before reports are considered stale (default: 30)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("reports"),
        help="Reports directory path (default: reports/)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-file output, only show summary",
    )

    args = parser.parse_args()

    if args.days < 1:
        print("Error: --days must be at least 1", file=sys.stderr)
        return 1

    print(
        f"Scanning {args.dir}/ for reports older than {args.days} days...",
        file=sys.stderr,
    )

    total, deleted, deleted_bytes = cleanup_old_reports(
        reports_dir=args.dir,
        max_age_days=args.days,
        delete=args.delete,
        verbose=not args.quiet,
    )

    deleted_kb = deleted_bytes // 1024

    print(file=sys.stderr)
    print("Summary:", file=sys.stderr)
    print(f"  Total reports: {total}", file=sys.stderr)
    if args.delete:
        print(f"  Deleted: {deleted} files ({deleted_kb}KB)", file=sys.stderr)
    else:
        print(f"  Would delete: {deleted} files ({deleted_kb}KB)", file=sys.stderr)
        if deleted > 0:
            print(file=sys.stderr)
            print("Run with --delete to remove old reports.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

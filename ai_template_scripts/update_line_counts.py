#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""
update_line_counts.py - Update CLAUDE.md with current file line counts.

PURPOSE: Automates maintenance of the "Files Loaded Per Session" table.
CALLED BY: Pre-commit hook or manual invocation.
ADDRESSES: #1608 (automate line counts), #1588 (line counts outdated)

Usage:
    ./update_line_counts.py              # Update CLAUDE.md in place
    ./update_line_counts.py --check      # Check if update needed (exit 1 if yes)
    ./update_line_counts.py --dry-run    # Show changes without writing
"""

__all__ = [
    "count_lines",
    "get_line_counts",
    "format_line_count",
    "generate_table_content",
    "update_claude_md",
    "main",
]

import argparse
import re
import sys
from pathlib import Path


def count_lines(filepath: Path) -> int:
    """Count total lines in a file (matches wc -l behavior)."""
    try:
        return len(filepath.read_text().splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def get_line_counts(repo_root: Path) -> dict[str, int]:
    """Get line counts for all files loaded per session."""
    rules_dir = repo_root / ".claude" / "rules"
    roles_dir = repo_root / ".claude" / "roles"

    counts = {}

    # Rules files
    ai_template = rules_dir / "ai_template.md"
    org_chart = rules_dir / "org_chart.md"
    if ai_template.exists():
        counts["ai_template.md"] = count_lines(ai_template)
    if org_chart.exists():
        counts["org_chart.md"] = count_lines(org_chart)

    # Roles files
    shared = roles_dir / "shared.md"
    if shared.exists():
        counts["shared.md"] = count_lines(shared)

    # Role-specific files (get range)
    role_counts = [
        count_lines(role_file)
        for role_file in roles_dir.glob("*.md")
        if role_file.name != "shared.md"
    ]

    if role_counts:
        counts["role_min"] = min(role_counts)
        counts["role_max"] = max(role_counts)

    return counts


def format_line_count(count: int, approximate: bool = True) -> str:
    """Format a line count for display."""
    if approximate:
        # Round to nearest 10 for ~N format
        rounded = round(count / 10) * 10
        return f"~{rounded}"
    return str(count)


def generate_table_content(counts: dict[str, int]) -> list[str]:
    """Generate the table rows for CLAUDE.md."""
    lines = [
        "| File | Lines | Purpose |",
        "|------|-------|---------|",
    ]

    ai_template_lines = counts.get("ai_template.md", 0)
    org_chart_lines = counts.get("org_chart.md", 0)
    shared_lines = counts.get("shared.md", 0)
    role_min = counts.get("role_min", 0)
    role_max = counts.get("role_max", 0)

    lines.append(
        f"| `.claude/rules/ai_template.md` | {format_line_count(ai_template_lines)} | "
        "All rules: workflow, anti-patterns, communication, post-mortems |"
    )
    lines.append(
        f"| `.claude/rules/org_chart.md` | {format_line_count(org_chart_lines)} | "
        "Minimal routing: directors, key repos, dependencies |"
    )
    lines.append(
        f"| `.claude/roles/shared.md` | {format_line_count(shared_lines)} | "
        "Session start protocol, injected context |"
    )

    # Role range (round to nearest 10 for consistency)
    if role_min and role_max:
        min_rounded = round(role_min / 10) * 10
        max_rounded = round(role_max / 10) * 10
        role_range = f"~{min_rounded}-{max_rounded}"
    else:
        role_range = "~150-250"
    lines.append(
        f"| `.claude/roles/{{role}}.md` | {role_range} | "
        "Role-specific config and prompt (varies by role) |"
    )

    # Calculate total
    total = (
        ai_template_lines + org_chart_lines + shared_lines + (role_min + role_max) // 2
    )
    lines.append(
        f"| **TOTAL** | **{format_line_count(total)}** | Plus injected content below |"
    )

    return lines


def update_claude_md(
    repo_root: Path,
    dry_run: bool = False,
    check_only: bool = False,
) -> bool:
    """
    Update CLAUDE.md with current line counts.

    Returns True if file was updated (or would be in check mode).
    """
    claude_md = repo_root / "CLAUDE.md"
    if not claude_md.exists():
        print(f"CLAUDE.md not found at {claude_md}", file=sys.stderr)
        return False

    counts = get_line_counts(repo_root)
    if not counts:
        print("No files found to count", file=sys.stderr)
        return False

    new_table = generate_table_content(counts)

    content = claude_md.read_text()

    # Find the table section using regex
    # Match from "| File | Lines |" to end of table (blank line or non-table line)
    table_pattern = re.compile(
        r"(\| File \| Lines \| Purpose \|\n"
        r"\|[-]+\|[-]+\|[-]+\|\n"
        r"(?:\|[^\n]+\|\n)+)",
        re.MULTILINE,
    )

    match = table_pattern.search(content)
    if not match:
        print(
            "Could not find 'Files Loaded Per Session' table in CLAUDE.md",
            file=sys.stderr,
        )
        return False

    old_table = match.group(1)
    new_table_str = "\n".join(new_table) + "\n"

    if old_table == new_table_str:
        if not check_only:
            print("Line counts are already up to date")
        return False

    if check_only:
        print("Line counts need updating:")
        print(f"  Current table:\n{old_table}")
        print(f"  New table:\n{new_table_str}")
        return True

    if dry_run:
        print("Would update CLAUDE.md with:")
        print(new_table_str)
        return True

    # Update the file
    new_content = content[: match.start()] + new_table_str + content[match.end() :]
    claude_md.write_text(new_content)
    print(f"Updated line counts in {claude_md}")
    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update CLAUDE.md with current file line counts"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if update needed (exit 1 if yes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )

    args = parser.parse_args()

    # Find repo root (look for CLAUDE.md)
    repo_root = args.repo
    if not (repo_root / "CLAUDE.md").exists():
        # Try parent directories
        for parent in repo_root.parents:
            if (parent / "CLAUDE.md").exists():
                repo_root = parent
                break

    updated = update_claude_md(
        repo_root,
        dry_run=args.dry_run,
        check_only=args.check,
    )

    if args.check and updated:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

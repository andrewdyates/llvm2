# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Generate encrypted credential bundles for all repos.

Each repo gets a .gh_apps_creds.enc containing:
- config.yaml (full config)
- dbx-ai.pem (wildcard fallback)
- Director .pem (e.g., dbx-dMATH-ai.pem for math repos)
- Repo-specific .pem (if repo has dedicated app)

Usage:
    python3 -m ai_template_scripts.gh_apps.generate_repo_bundles --password SECRET

    # Generate for specific repo
    python3 -m ai_template_scripts.gh_apps.generate_repo_bundles --repo z4 --password SECRET

    # List what would be generated
    python3 -m ai_template_scripts.gh_apps.generate_repo_bundles --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Import repo mappings
from ai_template_scripts.gh_apps.selector import (
    DEDICATED_REPOS,
    DIRECTOR_APPS,
    DIRECTOR_REPOS,
)

CONFIG_DIR = Path.home() / ".ait_gh_apps"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def get_director_for_repo(repo: str) -> str | None:
    """Get director that owns a repo."""
    for director, repos in DIRECTOR_REPOS.items():
        if repo in repos:
            return director
    return None


def get_pem_name(app_name: str) -> str:
    """Convert app name to .pem filename."""
    # Director apps: "math-ai" -> "dbx-dMATH-ai.pem"
    # Repo apps vary, check what exists
    return f"{app_name}.pem"


def get_director_pem(director: str) -> Path | None:
    """Get the .pem file for a director."""
    # Director pems are named like: dbx-dMATH-ai.pem, dbx-dLANG-ai.pem
    pem_name = f"dbx-d{director}-ai.pem"
    pem_path = CONFIG_DIR / pem_name
    if pem_path.exists():
        return pem_path
    return None


def get_repo_pem(repo: str) -> Path | None:
    """Get the dedicated .pem file for a repo (if it has one)."""
    if repo not in DEDICATED_REPOS:
        return None

    # Normalize repo name for pattern matching
    repo_normalized = repo.replace("_", "-")

    # Try common naming patterns
    patterns = [
        f"dbx-{repo}-ai.pem",
        f"dbx-{repo_normalized}-ai.pem",
        f"{repo}-ai.pem",
        f"{repo_normalized}-ai.pem",
    ]

    for pattern in patterns:
        pem_path = CONFIG_DIR / pattern
        if pem_path.exists():
            return pem_path

    return None


def get_all_repos() -> list[str]:
    """Get list of all known repos."""
    repos = set()
    for repo_list in DIRECTOR_REPOS.values():
        repos.update(repo_list)
    return sorted(repos)


def generate_bundle(repo: str, output_dir: Path, password: str) -> Path | None:
    """Generate encrypted credential bundle for a repo.

    Returns path to the generated .enc file, or None if failed.
    """
    if not CONFIG_FILE.exists():
        print(f"Error: No config file at {CONFIG_FILE}", file=sys.stderr)
        return None

    # Collect files for this repo
    files_to_include: list[tuple[Path, str]] = []

    # 1. Always include config.yaml
    files_to_include.append((CONFIG_FILE, "config.yaml"))

    # 2. Always include dbx-ai.pem (wildcard fallback)
    wildcard_pem = CONFIG_DIR / "dbx-ai.pem"
    if wildcard_pem.exists():
        files_to_include.append((wildcard_pem, "dbx-ai.pem"))

    # 3. Include director .pem
    director = get_director_for_repo(repo)
    if director:
        director_pem = get_director_pem(director)
        if director_pem:
            files_to_include.append((director_pem, director_pem.name))

    # 4. Include repo-specific .pem if it has one
    repo_pem = get_repo_pem(repo)
    if repo_pem:
        files_to_include.append((repo_pem, repo_pem.name))

    # Create tar archive
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tar_path = tmpdir_path / "creds.tar.gz"

        with tarfile.open(tar_path, "w:gz") as tar:
            for src_path, arc_name in files_to_include:
                tar.add(src_path, arcname=arc_name)

        # Encrypt
        output_path = output_dir / f"{repo}.gh_apps_creds.enc"
        result = subprocess.run(
            [
                "openssl", "enc", "-aes-256-cbc", "-salt", "-pbkdf2",
                "-pass", f"pass:{password}",
                "-in", str(tar_path),
                "-out", str(output_path),
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"Error encrypting bundle for {repo}: {result.stderr.decode()}", file=sys.stderr)
            return None

        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate encrypted credential bundles for repos",
    )
    parser.add_argument(
        "--repo", "-r",
        help="Generate for specific repo only",
    )
    parser.add_argument(
        "--password", "-p",
        required=True,
        help="Encryption password",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path.home() / "gh_apps_bundles",
        help="Output directory for bundles (default: ~/gh_apps_bundles)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be generated without doing it",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install bundles directly to repo directories",
    )

    args = parser.parse_args()

    # Get repos to process
    if args.repo:
        repos = [args.repo]
    else:
        repos = get_all_repos()

    # Create output directory
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating bundles for {len(repos)} repos...")
    print()

    for repo in repos:
        director = get_director_for_repo(repo)
        repo_pem = get_repo_pem(repo)

        files = ["config.yaml", "dbx-ai.pem"]
        if director:
            director_pem = get_director_pem(director)
            if director_pem:
                files.append(director_pem.name)
        if repo_pem:
            files.append(repo_pem.name)

        if args.dry_run:
            print(f"{repo}:")
            print(f"  Director: {director or 'none'}")
            print(f"  Files: {', '.join(files)}")
            print()
        else:
            output_path = generate_bundle(repo, args.output_dir, args.password)
            if output_path:
                print(f"  {repo}: {output_path}")

                # Optionally install directly to repo
                if args.install:
                    repo_dir = Path.home() / repo
                    if repo_dir.exists():
                        dest = repo_dir / ".gh_apps_creds.enc"
                        import shutil
                        shutil.copy(output_path, dest)
                        print(f"    -> Installed to {dest}")
            else:
                print(f"  {repo}: FAILED")

    if not args.dry_run:
        print()
        print(f"Bundles saved to: {args.output_dir}")
        if args.install:
            print("Bundles also installed to repo directories.")


if __name__ == "__main__":
    main()

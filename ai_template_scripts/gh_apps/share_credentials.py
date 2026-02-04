# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/share_credentials.py - Export/Import GitHub App credentials

Export credentials from one machine to share with another:
    python3 -m ai_template_scripts.gh_apps.share_credentials export ~/gh_apps_creds.tar.gz.enc

Import credentials on another machine:
    python3 -m ai_template_scripts.gh_apps.share_credentials import ~/gh_apps_creds.tar.gz.enc

The archive is encrypted with a passphrase using openssl.

Alternative: If you trust your network, use --no-encrypt for unencrypted transfer:
    python3 -m ai_template_scripts.gh_apps.share_credentials export ~/gh_apps_creds.tar.gz --no-encrypt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

CONFIG_DIR = Path.home() / ".ait_gh_apps"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def export_credentials(output_path: Path, encrypt: bool = True) -> bool:
    """Export credentials to an archive.

    Args:
        output_path: Path for output archive
        encrypt: If True, encrypt with passphrase

    Returns:
        True if successful
    """
    if not CONFIG_DIR.exists():
        print(f"Error: No credentials found at {CONFIG_DIR}", file=sys.stderr)
        return False

    if not CONFIG_FILE.exists():
        print(f"Error: No config file at {CONFIG_FILE}", file=sys.stderr)
        return False

    # Collect files to export
    files_to_export = [CONFIG_FILE]
    pem_files = list(CONFIG_DIR.glob("*.pem"))
    files_to_export.extend(pem_files)

    print(f"Exporting {len(files_to_export)} files:")
    print(f"  - config.yaml")
    for pem in pem_files:
        print(f"  - {pem.name}")

    # Create tar archive
    if encrypt:
        tar_path = output_path.with_suffix("")  # Remove .enc for tar
        if tar_path.suffix != ".gz":
            tar_path = tar_path.with_suffix(".tar.gz")
    else:
        tar_path = output_path
        if not str(tar_path).endswith(".tar.gz"):
            tar_path = Path(str(tar_path) + ".tar.gz")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tar = Path(tmpdir) / "creds.tar.gz"

        with tarfile.open(tmp_tar, "w:gz") as tar:
            for f in files_to_export:
                # Store relative to .ait_gh_apps/
                arcname = f.name
                tar.add(f, arcname=arcname)

        if encrypt:
            # Encrypt with openssl
            print("\nEncrypting archive (enter passphrase)...")
            result = subprocess.run(
                [
                    "openssl", "enc", "-aes-256-cbc", "-salt", "-pbkdf2",
                    "-in", str(tmp_tar),
                    "-out", str(output_path),
                ],
                check=False,
            )
            if result.returncode != 0:
                print("Encryption failed", file=sys.stderr)
                return False
            print(f"\nEncrypted credentials saved to: {output_path}")
        else:
            # Just copy the tar
            import shutil
            shutil.copy(tmp_tar, tar_path)
            print(f"\nCredentials saved to: {tar_path}")
            print("WARNING: Archive is NOT encrypted - contains private keys!")

    print("\nTo import on another machine:")
    if encrypt:
        print(f"  python3 -m ai_template_scripts.gh_apps.share_credentials import {output_path.name}")
    else:
        print(f"  python3 -m ai_template_scripts.gh_apps.share_credentials import {tar_path.name} --no-encrypt")

    return True


def import_credentials(input_path: Path, encrypt: bool = True) -> bool:
    """Import credentials from an archive.

    Args:
        input_path: Path to input archive
        encrypt: If True, decrypt with passphrase

    Returns:
        True if successful
    """
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return False

    # Create config dir if needed
    CONFIG_DIR.mkdir(mode=0o700, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        if encrypt:
            # Decrypt with openssl
            print("Decrypting archive (enter passphrase)...")
            tmp_tar = Path(tmpdir) / "creds.tar.gz"
            result = subprocess.run(
                [
                    "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
                    "-in", str(input_path),
                    "-out", str(tmp_tar),
                ],
                check=False,
            )
            if result.returncode != 0:
                print("Decryption failed - wrong passphrase?", file=sys.stderr)
                return False
        else:
            tmp_tar = input_path

        # Extract tar
        with tarfile.open(tmp_tar, "r:gz") as tar:
            # Security: check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    print(f"Error: Unsafe path in archive: {member.name}", file=sys.stderr)
                    return False

            print(f"\nImporting to {CONFIG_DIR}:")
            for member in tar.getmembers():
                dest = CONFIG_DIR / member.name
                print(f"  - {member.name}")

                # Check if file exists
                if dest.exists():
                    response = input(f"    Overwrite {dest}? [y/N] ")
                    if response.lower() != "y":
                        print(f"    Skipped")
                        continue

                tar.extract(member, CONFIG_DIR)

                # Set permissions
                if member.name.endswith(".pem"):
                    dest.chmod(0o600)
                elif member.name == "config.yaml":
                    dest.chmod(0o600)

    print(f"\nCredentials imported to {CONFIG_DIR}")
    print("\nTest with:")
    print("  AIT_GH_APPS_DEBUG=1 python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template")

    return True


def list_credentials() -> None:
    """List current credentials."""
    if not CONFIG_DIR.exists():
        print(f"No credentials directory at {CONFIG_DIR}")
        return

    if not CONFIG_FILE.exists():
        print(f"No config file at {CONFIG_FILE}")
        return

    print(f"Credentials directory: {CONFIG_DIR}")
    print(f"\nConfig file: {CONFIG_FILE}")

    pem_files = sorted(CONFIG_DIR.glob("*.pem"))
    print(f"\nPrivate keys ({len(pem_files)}):")
    for pem in pem_files:
        size = pem.stat().st_size
        print(f"  - {pem.name} ({size} bytes)")

    # Try to load config
    try:
        from ai_template_scripts.gh_apps.config import load_config
        config = load_config()
        if config:
            print(f"\nConfigured apps ({len(config.apps)}):")
            for name, app in sorted(config.apps.items()):
                key_exists = app.private_key_path.exists()
                status = "OK" if key_exists else "MISSING KEY"
                print(f"  - {name}: repos={app.repos} [{status}]")
    except Exception as e:
        print(f"\nFailed to load config: {e}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export/Import GitHub App credentials for sharing across machines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export credentials (encrypted)
  python3 -m ai_template_scripts.gh_apps.share_credentials export ~/creds.enc

  # Import credentials (encrypted)
  python3 -m ai_template_scripts.gh_apps.share_credentials import ~/creds.enc

  # Export without encryption (NOT recommended)
  python3 -m ai_template_scripts.gh_apps.share_credentials export ~/creds.tar.gz --no-encrypt

  # List current credentials
  python3 -m ai_template_scripts.gh_apps.share_credentials list
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export credentials to archive")
    export_parser.add_argument("output", type=Path, help="Output file path")
    export_parser.add_argument(
        "--no-encrypt", action="store_true",
        help="Skip encryption (NOT recommended - contains private keys)"
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import credentials from archive")
    import_parser.add_argument("input", type=Path, help="Input file path")
    import_parser.add_argument(
        "--no-encrypt", action="store_true",
        help="Input is not encrypted"
    )

    # List command
    subparsers.add_parser("list", help="List current credentials")

    args = parser.parse_args()

    if args.command == "export":
        success = export_credentials(args.output, encrypt=not args.no_encrypt)
        sys.exit(0 if success else 1)
    elif args.command == "import":
        success = import_credentials(args.input, encrypt=not args.no_encrypt)
        sys.exit(0 if success else 1)
    elif args.command == "list":
        list_credentials()


if __name__ == "__main__":
    main()

# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_apps/setup.py - Automated GitHub App Creation and Installation

Uses Playwright for browser automation to create apps and install them.
Zero manual steps after initial browser login.

Usage:
    # One-time login (interactive)
    python3 -m ai_template_scripts.gh_apps.setup login

    # Create a single app
    python3 -m ai_template_scripts.gh_apps.setup create-app z4

    # Create all Priority 1 apps
    python3 -m ai_template_scripts.gh_apps.setup create-all --priority 1

    # Delete an app
    python3 -m ai_template_scripts.gh_apps.setup delete-app test
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page

# Optional imports - check at runtime
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]

try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from ai_template_scripts.gh_apps.config import CONFIG_DIR, CONFIG_FILE
from ai_template_scripts.identity import get_identity as _get_ident

# Constants
ORG = _get_ident().github_org
TARGET_USER = ORG
BROWSER_PROFILE = Path.home() / ".playwright_github"

# Priority tiers for batch creation
PRIORITY_REPOS: dict[int, list[str]] = {
    1: ["ai_template", "z4", "dasher", "sg", "zani"],
    2: ["tRust", "leadership", "gamma-crown"],
}


class AppSetup:
    """Automated GitHub App setup via Playwright."""

    async def login(self) -> None:
        """Interactive login to GitHub in persistent browser profile.

        Only needs to be run once. After this, all create-app calls
        are fully automated.
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: playwright not installed. Run: pip install playwright")
            print("       Then: playwright install chromium")
            sys.exit(1)

        BROWSER_PROFILE.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(BROWSER_PROFILE),
                headless=False,
            )
            page = browser.pages[0] if browser.pages else await browser.new_page()

            await page.goto("https://github.com/login")

            print("\n" + "=" * 60)
            print("GitHub Login Required")
            print("=" * 60)
            print()
            print("1. Log in to GitHub in the browser window")
            print("2. Complete any 2FA if required")
            print("3. Press Enter here when logged in")
            print()
            input("Press Enter after logging in... ")

            await browser.close()
            print("\nLogin saved to browser profile. Future create-app calls are automated.")

    async def create_app(self, project: str, headless: bool = False) -> dict[str, int]:
        """Create and install a GitHub App for a project.

        Args:
            project: Project name (e.g., "z4"). App will be named "{project}-ai".
            headless: Run browser without UI (default False for debugging).

        Returns:
            Dict with app_id and installation_id.
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: playwright not installed. Run: pip install playwright")
            print("       Then: playwright install chromium")
            sys.exit(1)

        if not YAML_AVAILABLE:
            print("ERROR: pyyaml not installed. Run: pip install pyyaml")
            sys.exit(1)

        # GitHub App names cannot contain underscores, so replace with hyphens
        app_name = f"{project.replace('_', '-')}-ai"
        CONFIG_DIR.mkdir(mode=0o700, exist_ok=True)

        print(f"\nCreating GitHub App: {app_name}")
        print(f"  Organization: {ORG}")
        print(f"  Target repo: {TARGET_USER}/{project}")

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(BROWSER_PROFILE),
                headless=headless,
            )
            page = browser.pages[0] if browser.pages else await browser.new_page()

            try:
                # Step 1: Navigate to app creation page
                print("\n1. Navigating to app creation page...")
                await page.goto(
                    f"https://github.com/organizations/{ORG}/settings/apps/new"
                )
                # Wait for DOM to be ready (networkidle is too strict for GitHub)
                await page.wait_for_load_state("domcontentloaded")
                # Also wait for the form to be present
                await page.wait_for_selector('input[name="integration[name]"]', timeout=15000)

                # Check if we're on login page
                if "login" in page.url:
                    print("\nERROR: Not logged in. Run: python3 -m ai_template_scripts.gh_apps.setup login")
                    await browser.close()
                    sys.exit(1)

                # Step 2: Fill app creation form
                print("2. Filling app creation form...")
                await self._fill_app_form(page, app_name, project)

                # Step 3: Submit and get app ID
                print("3. Creating app...")
                # The form submit button is an input, not a button:
                # <input type="submit" name="commit" value="Create GitHub App" class="js-integration-submit">
                await page.click('input[type="submit"].js-integration-submit')
                await page.wait_for_url(re.compile(r"/settings/apps/[^/]+$"), timeout=30000)

                # Extract app_id from page
                app_id = await self._extract_app_id(page)
                print(f"   App ID: {app_id}")

                # Step 4: Generate private key
                print("4. Generating private key...")
                key_path = await self._generate_private_key(page, app_name)
                print(f"   Key saved: {key_path}")

                # Step 5: Install on target repo
                print("5. Installing on target repo...")
                installation_id = await self._install_app(page, app_name, project)
                print(f"   Installation ID: {installation_id}")

                await browser.close()

            except Exception as e:
                await browser.close()
                raise RuntimeError(f"Failed to create app: {e}") from e

        # Step 6: Update config
        print("6. Updating config.yaml...")
        self._update_config(app_name, app_id, installation_id, project, key_path)

        print(f"\n✓ App {app_name} created and installed successfully!")
        return {"app_id": app_id, "installation_id": installation_id}

    async def _fill_app_form(self, page: Page, app_name: str, project: str) -> None:
        """Fill the app creation form using JavaScript for reliability.

        GitHub's app creation form uses hidden inputs for permissions (not selects).
        We directly set these hidden input values via JavaScript.
        """
        homepage_url = f"https://github.com/{TARGET_USER}/{project}"

        # Permissions are stored as hidden inputs with name="integration[default_permissions][X]"
        # Values: "none", "read", "write"
        permissions = {"contents": "write", "issues": "write", "pull_requests": "write", "metadata": "read"}

        result = await page.evaluate(
            """([appName, homepageUrl, permissions]) => {
            const errors = [];

            // Fill app name
            const nameInput = document.querySelector('input[name="integration[name]"]');
            if (nameInput) {
                nameInput.value = appName;
                nameInput.dispatchEvent(new Event('input', { bubbles: true }));
            } else {
                errors.push('App name input not found');
            }

            // Fill homepage URL
            const urlInput = document.querySelector('input[name="integration[url]"]');
            if (urlInput) {
                urlInput.value = homepageUrl;
                urlInput.dispatchEvent(new Event('input', { bubbles: true }));
            } else {
                errors.push('Homepage URL input not found');
            }

            // Disable webhook - find checkbox with "active" in name
            const webhookCheckbox = document.querySelector(
                'input[type="checkbox"][name*="hook_attributes"][name*="active"]'
            );
            if (webhookCheckbox && webhookCheckbox.checked) {
                webhookCheckbox.click();
            }

            // Set permissions - these are HIDDEN INPUTS, not selects
            // The form has inputs like: name="integration[default_permissions][contents]" value="none"
            for (const [perm, level] of Object.entries(permissions)) {
                const selector = 'input[name="integration[default_permissions][' + perm + ']"]';
                const input = document.querySelector(selector);
                if (input) {
                    input.value = level;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                } else {
                    errors.push('Permission input not found: ' + perm);
                }
            }

            // Set "Any account" for installation scope (public visibility)
            const radios = document.querySelectorAll('input[type="radio"]');
            let foundPublic = false;
            for (const radio of radios) {
                if (radio.value === 'public' || (radio.id && radio.id.includes('public'))) {
                    radio.click();
                    foundPublic = true;
                    break;
                }
            }
            if (!foundPublic) {
                errors.push('Public radio button not found');
            }

            return { success: errors.length === 0, errors: errors };
        }""",
            [app_name, homepage_url, permissions],
        )

        # Report any errors found
        if not result.get("success"):
            for error in result.get("errors", []):
                print(f"   Warning: {error}")

        # Give the form time to process
        await page.wait_for_timeout(500)

    async def _extract_app_id(self, page: Page) -> int:
        """Extract app ID from the settings page."""
        # The App ID is shown in format: <p><strong>App ID:</strong> 2790147</p>
        # or possibly other formats. Use JavaScript to extract reliably.

        await page.wait_for_selector("text=App ID", timeout=10000)

        app_id = await page.evaluate(
            """() => {
            // Look for "App ID:" followed by a number
            const bodyText = document.body.innerText;
            const match = bodyText.match(/App ID:\\s*(\\d+)/);
            if (match) return parseInt(match[1], 10);

            // Fallback: look in HTML for any App ID pattern
            const html = document.body.innerHTML;
            const htmlMatch = html.match(/App ID[^\\d]*(\\d+)/);
            if (htmlMatch) return parseInt(htmlMatch[1], 10);

            return null;
        }"""
        )

        if not app_id:
            raise RuntimeError("Could not find App ID on page")

        return app_id

    async def _generate_private_key(self, page: Page, app_name: str) -> Path:
        """Generate and download private key."""
        key_path = CONFIG_DIR / f"{app_name}.pem"

        # Click "Generate a private key" inside expect_download context
        # The download is triggered by the click action
        async with page.expect_download() as download_info:
            await page.click('text=Generate a private key')

        download = await download_info.value
        await download.save_as(str(key_path))
        key_path.chmod(0o600)

        return key_path

    async def _install_app(self, page: Page, app_name: str, project: str) -> int:
        """Install app on target repository."""
        # Navigate to installation page
        await page.goto(f"https://github.com/apps/{app_name}/installations/new")
        await page.wait_for_load_state("networkidle")

        # Select target account - use link role for the account list
        # The installation page shows accounts as clickable links/buttons
        account_link = page.get_by_role("link", name=TARGET_USER).first
        if not await account_link.is_visible():
            # Fallback: find in the installation targets list specifically
            account_link = page.locator(f".installation-target >> text={TARGET_USER}").first
        if not await account_link.is_visible():
            # Last resort: any element with exact text in main content area
            account_link = page.locator(f"main >> text={TARGET_USER}").first
        await account_link.click()

        # Wait for repository selection options to appear
        await page.wait_for_timeout(1000)

        # Select "Only select repositories" using the visible radio button
        # Use get_by_role to avoid matching hidden input fields
        selected_radio = page.get_by_role("radio", name="Only select repositories")
        await selected_radio.check()

        # Search and select the repo
        # GitHub uses a search input to filter repositories
        repo_search = page.get_by_placeholder("Search repositories")
        if not await repo_search.is_visible():
            # Try alternative placeholder text
            repo_search = page.locator('input[type="text"]').filter(
                has_text=""
            ).last
        await repo_search.fill(project)
        await page.wait_for_timeout(1000)  # Wait for search results

        # Click on the repo in dropdown
        # GitHub shows repos as checkboxes or clickable list items
        # Try checkbox first (common pattern)
        repo_checkbox = page.get_by_role("checkbox", name=re.compile(project, re.IGNORECASE))
        if await repo_checkbox.count() > 0:
            await repo_checkbox.first.check()
        else:
            # Try as a list item or generic clickable element
            repo_item = page.locator(f"[data-repo-name='{project}']").first
            if not await repo_item.is_visible():
                # Try label containing repo name
                repo_item = page.get_by_label(project).first
            if not await repo_item.is_visible():
                # Last resort: text match within the repo list area
                repo_item = page.locator(f".repo-list >> text={project}").first
            if not await repo_item.is_visible():
                repo_item = page.get_by_text(project, exact=True).first
            await repo_item.click()

        # Click Install
        install_button = page.get_by_role("button", name="Install")
        await install_button.click()

        # Wait for installation to complete
        await page.wait_for_url(re.compile(r"/installations/\d+"), timeout=30000)

        # Extract installation_id from URL
        match = re.search(r"/installations/(\d+)", page.url)
        if not match:
            raise RuntimeError("Could not extract installation ID from URL")

        return int(match.group(1))

    def _update_config(
        self,
        app_name: str,
        app_id: int,
        installation_id: int,
        repo: str,
        key_path: Path,
    ) -> None:
        """Update config.yaml with new app."""
        if not YAML_AVAILABLE:
            return

        if CONFIG_FILE.exists():
            config = yaml.safe_load(CONFIG_FILE.read_text()) or {}
        else:
            config = {"org": ORG, "default_app": "shared-ai", "apps": {}}

        if "apps" not in config:
            config["apps"] = {}

        config["apps"][app_name] = {
            "app_id": app_id,
            "installation_id": installation_id,
            "private_key": str(key_path),
            "repo": repo,
        }

        CONFIG_FILE.write_text(yaml.dump(config, default_flow_style=False))
        CONFIG_FILE.chmod(0o600)

    async def create_all(self, priority: int = 1, headless: bool = False) -> list[dict]:
        """Create all apps for a priority tier.

        Args:
            priority: Priority tier (1, 2, or 3).
            headless: Run browser without UI.

        Returns:
            List of results with app_id and installation_id for each app.
        """
        repos = PRIORITY_REPOS.get(priority, [])
        if not repos:
            print(f"No repos defined for priority {priority}")
            return []

        results = []
        for repo in repos:
            try:
                result = await self.create_app(repo, headless=headless)
                results.append({"repo": repo, **result})
            except Exception as e:
                print(f"ERROR: Failed to create app for {repo}: {e}")
                results.append({"repo": repo, "error": str(e)})

        return results

    async def delete_app(self, project: str) -> None:
        """Delete a GitHub App.

        Args:
            project: Project name (e.g., "test"). App named "{project}-ai" will be deleted.
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: playwright not installed. Run: pip install playwright")
            sys.exit(1)

        # GitHub App names cannot contain underscores, so replace with hyphens
        app_name = f"{project.replace('_', '-')}-ai"
        print(f"\nDeleting GitHub App: {app_name}")

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(BROWSER_PROFILE),
                headless=False,
            )
            page = browser.pages[0] if browser.pages else await browser.new_page()

            try:
                # Navigate to app settings
                await page.goto(
                    f"https://github.com/organizations/{ORG}/settings/apps/{app_name}"
                )

                # Check if we're on login page
                if "login" in page.url:
                    print("\nERROR: Not logged in. Run: python3 -m ai_template_scripts.gh_apps.setup login")
                    await browser.close()
                    sys.exit(1)

                # Scroll to bottom and find delete button
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(500)

                # Click "Delete this GitHub App"
                await page.click('text=Delete this GitHub App')

                # Confirm deletion in dialog
                await page.fill('input[name="verify"]', app_name)
                await page.click('button:has-text("I understand, delete this")')

                await page.wait_for_timeout(2000)
                await browser.close()

            except Exception as e:
                await browser.close()
                raise RuntimeError(f"Failed to delete app: {e}") from e

        # Remove from config
        if YAML_AVAILABLE and CONFIG_FILE.exists():
            config = yaml.safe_load(CONFIG_FILE.read_text()) or {}
            if "apps" in config and app_name in config["apps"]:
                del config["apps"][app_name]
                CONFIG_FILE.write_text(yaml.dump(config, default_flow_style=False))

        # Remove private key
        key_path = CONFIG_DIR / f"{app_name}.pem"
        if key_path.exists():
            key_path.unlink()

        print(f"\n✓ App {app_name} deleted successfully!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GitHub App setup automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time login (interactive, only needed once)
  %(prog)s login

  # Create app for a project
  %(prog)s create-app z4

  # Create all Priority 1 apps (ai_template, z4, dasher, sg, zani)
  %(prog)s create-all --priority 1

  # Delete a test app
  %(prog)s delete-app test
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # login command
    subparsers.add_parser("login", help="Interactive login to GitHub (one-time setup)")

    # create-app command
    create_parser = subparsers.add_parser("create-app", help="Create and install app for a project")
    create_parser.add_argument("project", help="Project name (e.g., z4)")
    create_parser.add_argument(
        "--headless", action="store_true", help="Run browser without UI"
    )

    # create-all command
    create_all_parser = subparsers.add_parser("create-all", help="Create apps for a priority tier")
    create_all_parser.add_argument(
        "--priority", type=int, default=1, choices=[1, 2], help="Priority tier (default: 1)"
    )
    create_all_parser.add_argument(
        "--headless", action="store_true", help="Run browser without UI"
    )

    # delete-app command
    delete_parser = subparsers.add_parser("delete-app", help="Delete an app")
    delete_parser.add_argument("project", help="Project name (e.g., test)")

    args = parser.parse_args()
    setup = AppSetup()

    if args.command == "login":
        asyncio.run(setup.login())
    elif args.command == "create-app":
        asyncio.run(setup.create_app(args.project, headless=args.headless))
    elif args.command == "create-all":
        asyncio.run(setup.create_all(priority=args.priority, headless=args.headless))
    elif args.command == "delete-app":
        asyncio.run(setup.delete_app(args.project))


if __name__ == "__main__":
    main()

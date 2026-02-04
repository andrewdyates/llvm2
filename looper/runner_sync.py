# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/runner_sync.py - Multi-machine branch synchronization.

Provides RunnerSyncMixin for LoopRunner:
- Zone branch creation and switching (zone/<machine> branches)
- Sync from origin/main using rebase or merge strategy
- Pre-iteration sync for multi-machine mode
- Auto-PR creation for zone branches

Used when looper runs with --machine flag for distributed AI operations.
Sync configuration is read from .looper_config.json "sync" section.

See docs/looper.md "Multi-Machine Sync" for configuration and usage.
"""

from __future__ import annotations

__all__ = ["RunnerSyncMixin"]

import json
import sys
from datetime import UTC, datetime

from looper.config import load_sync_config
from looper.log import debug_swallow, log_error, log_info, log_warning
from looper.subprocess_utils import run_gh_command, run_git_command
from looper.sync import SyncConfig, SyncStatus, sync_from_main


class RunnerSyncMixin:
    """Multi-machine sync helpers.

    Required attributes:
        machine: str | None
    """

    def _setup_branch(self, target_branch: str) -> None:
        """Ensure we're on the correct branch for multi-machine mode.

        If branch doesn't exist locally, creates it tracking origin/main.
        If on wrong branch, switches to target branch.
        Fails fast if uncommitted changes would be lost.

        Args:
            target_branch: Branch name to use (e.g., "zone/sat")

        Raises:
            SystemExit: If branch setup fails
        """
        # Check current branch
        result = run_git_command(["branch", "--show-current"], timeout=10)
        if not result.ok:
            log_error(f"Error: Failed to get current branch: {result.error}")
            sys.exit(1)
        current_branch = result.value.strip() if result.value else ""

        if current_branch == target_branch:
            log_info(f"✓ Already on branch: {target_branch}")
            return

        # Check for uncommitted changes
        result = run_git_command(["status", "--porcelain"], timeout=10)
        if result.ok and result.value and result.value.strip():
            log_error(
                f"Error: Uncommitted changes. Commit before switching to {target_branch}"
            )
            sys.exit(1)

        # Check if target branch exists locally
        result = run_git_command(["branch", "--list", target_branch], timeout=10)
        branch_exists = result.ok and result.value and target_branch in result.value

        if branch_exists:
            # Switch to existing branch
            log_info(f"Switching to existing branch: {target_branch}")
            result = run_git_command(["checkout", target_branch], timeout=30)
            if not result.ok:
                log_error(f"Error: Failed to checkout branch: {result.error}")
                sys.exit(1)
        else:
            # Create branch from origin/main (or main if origin doesn't exist)
            log_info(f"Creating new branch: {target_branch}")
            # Try origin/main first, fall back to main
            result = run_git_command(
                ["checkout", "-b", target_branch, "origin/main"], timeout=30
            )
            if not result.ok:
                # Try without origin/
                result = run_git_command(
                    ["checkout", "-b", target_branch, "main"], timeout=30
                )
                if not result.ok:
                    log_error(f"Error: Failed to create branch: {result.error}")
                    sys.exit(1)

        log_info(f"✓ Now on branch: {target_branch}")

    def _sync_from_main(self, strategy: str = "rebase") -> bool:
        """Sync current branch with origin/main (legacy wrapper).

        This is a compatibility wrapper around the new sync module.
        Prefer using _pre_iteration_sync() for new code.

        Args:
            strategy: "rebase" (default) or "merge"

        Returns:
            True if sync succeeded (or was not needed), False on failure.
        """
        config = SyncConfig(strategy=strategy, auto_stash=False)  # type: ignore[arg-type]
        result = sync_from_main(config)

        # Print status message based on result
        if result.status == SyncStatus.UP_TO_DATE:
            log_info("✓ Sync: already up to date with origin/main")
        elif result.status == SyncStatus.SYNCED:
            log_info(
                f"✓ Sync: {result.branch} updated with {result.commits_pulled} commit(s)"
            )
        elif result.status == SyncStatus.NOT_APPLICABLE:
            pass  # On main, no message needed
        elif result.status == SyncStatus.BLOCKED:
            log_warning(f"⚠ Sync: {result.reason}")
            log_warning("  Commit changes before sync can proceed")
        elif result.status == SyncStatus.CONFLICT:
            log_warning(f"⚠ Sync: {result.reason}")
            if result.conflict_files:
                log_warning(
                    f"  Conflicting files: {', '.join(result.conflict_files[:5])}"
                )
        elif result.status == SyncStatus.FETCH_FAILED:
            log_warning(f"⚠ Sync: fetch failed: {result.reason}")
        else:
            log_warning(f"⚠ Sync: {result.reason}")

        return result.ok

    def _pre_iteration_sync(self) -> None:
        """Sync from main before iteration starts (multi-machine mode).

        Called at the start of each iteration when in multi-machine mode
        with sync trigger set to "iteration_start". Uses SyncConfig from
        .looper_config.json.

        See: designs/2026-01-25-auto-sync-protocol.md
        """
        if not self.machine:
            return

        # Load sync config from .looper_config.json
        sync_config_dict = load_sync_config()
        sync_config = SyncConfig.from_dict(sync_config_dict)

        # Only sync if trigger is iteration_start
        if sync_config.trigger != "iteration_start":
            return

        result = sync_from_main(sync_config)

        # Print status message
        if result.status == SyncStatus.UP_TO_DATE:
            pass  # Silent when up to date
        elif result.status == SyncStatus.SYNCED:
            log_info(f"✓ Synced {result.commits_pulled} commits from main")
        elif result.status == SyncStatus.CONFLICT:
            log_warning(f"⚠️ Sync conflict: {', '.join(result.conflict_files[:5])}")
            # Continue with diverged branch - don't block iteration
        elif result.status == SyncStatus.DIVERGED:
            log_warning(
                f"⚠️ Branch diverged from main ({result.commits_behind} commits behind)"
            )
        elif result.status == SyncStatus.BLOCKED:
            log_warning(f"⚠️ Sync blocked: {result.reason}")
        elif result.status == SyncStatus.FETCH_FAILED:
            log_warning(f"⚠️ Sync fetch failed: {result.reason}")
        elif result.status == SyncStatus.ERROR:
            log_warning(f"⚠️ Sync error: {result.reason}")

    def _ensure_pr_exists(self) -> bool:
        """Ensure a PR exists for the current zone branch.

        In multi-machine mode, zone branches should have PRs to main for
        code review and integration. This method:
        1. Checks if we're on a zone branch with commits ahead of main
        2. Checks if a PR already exists for this branch
        3. Creates a PR if none exists

        Returns:
            True if PR exists or was created, False on failure.

        Note:
            - Only runs when on a non-main branch (zone branches)
            - Requires branch to be pushed to origin
            - Idempotent: safe to call repeatedly
        """
        # Check if we're on a zone branch (not main)
        result = run_git_command(["branch", "--show-current"], timeout=10)
        if not result.ok:
            log_warning(f"⚠ PR: failed to get current branch: {result.error}")
            return False
        current_branch = result.value.strip() if result.value else ""

        if current_branch == "main":
            # On main branch - no PR needed
            return True

        if not current_branch:
            log_warning("⚠ PR: not on any branch (detached HEAD?)")
            return False

        # Check if we have commits ahead of main
        result = run_git_command(
            ["rev-list", "--count", f"origin/main..{current_branch}"], timeout=10
        )
        if not result.ok:
            # If this fails, likely haven't fetched origin/main yet
            log_warning(f"⚠ PR: failed to check commits ahead: {result.error}")
            return False

        commits_ahead = int(result.value.strip()) if result.value else 0
        if commits_ahead == 0:
            # No commits ahead of main - no PR needed yet
            return True

        # Push to origin if not already (PR requires remote branch)
        push_result = run_git_command(
            ["push", "-u", "origin", current_branch], timeout=60
        )
        if not push_result.ok:
            log_warning(f"⚠ PR: push failed: {push_result.error}")
            return False

        # Check if PR already exists for this branch
        pr_check = run_gh_command(
            ["pr", "list", "--head", current_branch, "--json", "number,state"],
            timeout=30,
        )
        if pr_check.ok and pr_check.value:
            try:
                prs = json.loads(pr_check.value)
                # Check for open PRs
                open_prs = [p for p in prs if p.get("state") == "OPEN"]
                if open_prs:
                    log_info(
                        f"✓ PR: #{open_prs[0]['number']} exists for {current_branch}"
                    )
                    return True
            except (json.JSONDecodeError, KeyError) as e:
                debug_swallow(
                    "ensure_pr_exists_parse", e
                )  # Best-effort: JSON parse failure, try to create PR

        # Create PR
        machine_name = self.machine or current_branch.split("/")[-1]
        pr_title = f"[{machine_name}] Zone branch sync"
        created_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        pr_body = f"""## Zone Branch PR

Auto-created by looper for multi-machine mode.

**Machine:** {machine_name}
**Branch:** {current_branch}
**Commits ahead:** {commits_ahead}
**Created:** {created_at}

This PR contains work from the {machine_name} zone. Review and merge when ready.
"""

        pr_result = run_gh_command(
            [
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                current_branch,
                "--title",
                pr_title,
                "--body",
                pr_body,
            ],
            timeout=60,
        )

        if pr_result.ok and pr_result.value:
            pr_url = pr_result.value.strip()
            log_info(f"✓ PR: created {pr_url}")
            return True
        error = pr_result.error or "unknown error"
        # Check if PR already exists (race condition)
        if "already exists" in error.lower():
            log_info(f"✓ PR: already exists for {current_branch}")
            return True
        log_warning(f"⚠ PR: creation failed: {error}")
        return False

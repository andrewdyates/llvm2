# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/sync.py - Multi-machine sync module

Handles syncing zone branches with origin/main in multi-machine mode.
Supports both rebase and merge strategies with conflict detection.

Design: designs/2026-01-25-auto-sync-protocol.md
Issues: #740 (auto-sync), #752 (this module)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from looper.log import log_warning
from looper.result import Result
from looper.subprocess_utils import run_git_command

__all__ = [
    # Classes
    "SyncStatus",
    "SyncResult",
    "SyncConfig",
    # Functions - Pure getters
    "get_current_branch",
    "get_uncommitted_changes_result",
    "get_staged_files",
    "get_uncommitted_line_count",
    "get_commits_behind",
    "get_conflict_files",
    # Functions - Predicates
    "has_uncommitted_changes",
    # Functions - Warning side effects (Phase 2 API standardization)
    "warn_uncommitted_work",
    "warn_stale_staged_files",
    # Functions - Enforcement (Phase 2 API standardization)
    "enforce_no_stale_staged_files",
    # Functions - Deprecated (use warn_* or enforce_* instead)
    "check_stale_staged_files",
    "check_uncommitted_work_size",
    # Functions - Operations
    "sync_from_main",
]


class SyncStatus(Enum):
    """Status codes for sync operations."""

    UP_TO_DATE = "up_to_date"  # No sync needed, already current
    SYNCED = "synced"  # Successfully synced commits from main
    CONFLICT = "conflict"  # Sync failed due to conflicts
    DIVERGED = "diverged"  # Branch diverged, sync skipped
    BLOCKED = "blocked"  # Uncommitted changes, cannot sync
    FETCH_FAILED = "fetch_failed"  # Failed to fetch from origin
    NOT_APPLICABLE = "not_applicable"  # On main branch, sync not needed
    ERROR = "error"  # Other error


@dataclass(frozen=True)
class SyncResult:
    """Result of a sync operation.

    Attributes:
        status: The outcome of the sync operation
        commits_pulled: Number of commits integrated from main (if synced)
        commits_behind: Number of commits behind main (before sync)
        conflict_files: List of files with conflicts (if conflict)
        strategy: The sync strategy used ("rebase" or "merge")
        reason: Human-readable explanation of the result
        branch: The branch that was synced
    """

    status: SyncStatus
    commits_pulled: int = 0
    commits_behind: int = 0
    conflict_files: list[str] = field(default_factory=list)
    strategy: str = ""
    reason: str = ""
    branch: str = ""

    @property
    def ok(self) -> bool:
        """Return True if sync succeeded or was not needed."""
        return self.status in (
            SyncStatus.UP_TO_DATE,
            SyncStatus.SYNCED,
            SyncStatus.NOT_APPLICABLE,
        )


# Valid values for SyncConfig fields (used for runtime validation)
_VALID_STRATEGIES = frozenset(("rebase", "merge"))
_VALID_TRIGGERS = frozenset(("iteration_start", "manual", "hourly"))
_VALID_CONFLICT_ACTIONS = frozenset(("abort", "continue_diverged"))


@dataclass
class SyncConfig:
    """Configuration for sync operations.

    Loaded from .looper_config.json "sync" section.

    Attributes:
        strategy: "rebase" (default) or "merge"
        trigger: When to sync - "iteration_start", "manual", or "hourly"
        auto_stash: Whether to auto-stash uncommitted changes (default: True)
        conflict_action: "abort" (default) or "continue_diverged"
    """

    strategy: Literal["rebase", "merge"] = "rebase"
    trigger: Literal["iteration_start", "manual", "hourly"] = "iteration_start"
    auto_stash: bool = True
    conflict_action: Literal["abort", "continue_diverged"] = "abort"

    @classmethod
    def from_dict(cls, data: dict) -> "SyncConfig":
        """Create SyncConfig from dictionary (e.g., JSON config).

        Validates field values and warns on invalid values, falling back to
        defaults. This improves debugging by surfacing config errors at load
        time rather than usage time.

        REQUIRES: data is a dictionary (may be empty)
        ENSURES: Returns valid SyncConfig instance
        ENSURES: Invalid values emit warning to stderr and use defaults
        ENSURES: Missing keys use defaults without warning
        """
        # Validate and extract strategy
        strategy = data.get("strategy", "rebase")
        if strategy not in _VALID_STRATEGIES:
            log_warning(
                f"Warning: invalid sync.strategy '{strategy}', using 'rebase'",
                stream="stderr",
            )
            strategy = "rebase"

        # Validate and extract trigger
        trigger = data.get("trigger", "iteration_start")
        if trigger not in _VALID_TRIGGERS:
            log_warning(
                f"Warning: invalid sync.trigger '{trigger}', using 'iteration_start'",
                stream="stderr",
            )
            trigger = "iteration_start"

        # Validate and extract conflict_action
        conflict_action = data.get("conflict_action", "abort")
        if conflict_action not in _VALID_CONFLICT_ACTIONS:
            log_warning(
                f"Warning: invalid sync.conflict_action '{conflict_action}', using 'abort'",
                stream="stderr",
            )
            conflict_action = "abort"

        # Validate and extract auto_stash (must be boolean)
        auto_stash = data.get("auto_stash", True)
        if not isinstance(auto_stash, bool):
            log_warning(
                f"Warning: invalid sync.auto_stash '{auto_stash}', using True",
                stream="stderr",
            )
            auto_stash = True

        return cls(
            strategy=strategy,  # type: ignore[arg-type]
            trigger=trigger,  # type: ignore[arg-type]
            auto_stash=auto_stash,
            conflict_action=conflict_action,  # type: ignore[arg-type]
        )


def get_current_branch() -> str | None:
    """Get the current git branch name.

    REQUIRES: Called from within a git repository
    ENSURES: Returns non-empty string if on a branch
    ENSURES: Returns None if detached HEAD or git error

    Returns:
        Branch name, or None if not on a branch (detached HEAD).
    """
    result = run_git_command(["branch", "--show-current"], timeout=10)
    if not result.ok:
        return None
    return (result.value or "").strip() or None


def get_uncommitted_changes_result() -> Result[bool]:
    """Check if there are uncommitted changes in the working directory.

    Result-returning variant that preserves error information for callers
    that need to distinguish git errors from actual change detection.

    REQUIRES: Called from within a git repository
    ENSURES: Result.ok=True with Result.value=True if changes exist
    ENSURES: Result.ok=True with Result.value=False if no changes
    ENSURES: Result.ok=False with error message on git error

    Returns:
        Result[bool] with the change status or error details.

    Part of #1776: Looper API error handling consistency
    """
    result = run_git_command(["status", "--porcelain"], timeout=10)
    if not result.ok:
        return Result.failure(result.error or "git status failed")
    return Result.success(bool((result.value or "").strip()))


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes in the working directory.

    Backward-compatible wrapper that returns True on git error (conservative).
    Logs a warning when git commands fail to provide visibility.

    For callers needing error details, use get_uncommitted_changes_result().

    REQUIRES: Called from within a git repository
    ENSURES: Returns True if staged or unstaged changes exist
    ENSURES: Returns True (conservative) on git error
    ENSURES: Logs warning on git error

    Returns:
        True if there are uncommitted changes.
    """
    result = get_uncommitted_changes_result()
    if not result.ok:
        log_warning(
            f"Warning: git status failed ({result.error}), assuming uncommitted changes",
            stream="stderr",
        )
        return True  # Conservative: assume changes if check fails
    return result.value or False


def get_staged_files() -> list[str]:
    """Get list of files currently staged for commit.

    Used to detect stale staged files from prior sessions that could
    contaminate commits. Returns empty list if no staged files or on error.

    REQUIRES: Called from within a git repository
    ENSURES: Returns list (possibly empty) of file paths
    ENSURES: Each path is non-empty string
    ENSURES: Returns empty list on git error

    Returns:
        List of staged file paths.
    """
    result = run_git_command(["diff", "--cached", "--name-only"], timeout=10)
    if not result.ok or not result.value:
        return []
    return [f for f in result.value.strip().split("\n") if f]


def get_uncommitted_line_count() -> int:
    """Get total lines of uncommitted changes (staged + unstaged).

    Used for mid-session commit reminders. Returns 0 on error.

    REQUIRES: Called from within a git repository
    ENSURES: Returns non-negative integer
    ENSURES: Returns 0 on git error or no changes
    ENSURES: Result = insertions + deletions

    Returns:
        Total lines added + deleted in uncommitted changes.
    """
    # Get diff stat for all changes (staged + unstaged)
    result = run_git_command(["diff", "HEAD", "--stat"], timeout=10)
    if not result.ok or not result.value:
        return 0

    # Parse "X files changed, Y insertions(+), Z deletions(-)" line
    # Example: "3 files changed, 150 insertions(+), 30 deletions(-)"
    lines = result.value.strip().split("\n")
    if not lines:
        return 0

    # Summary is last line
    summary = lines[-1]
    total = 0

    # Extract insertions
    ins_match = re.search(r"(\d+) insertion", summary)
    if ins_match:
        total += int(ins_match.group(1))

    # Extract deletions
    del_match = re.search(r"(\d+) deletion", summary)
    if del_match:
        total += int(del_match.group(1))

    return total


def warn_uncommitted_work(threshold: int = 100) -> int | None:
    """Warn if uncommitted lines exceed threshold.

    Pure function with side effect: prints warning to stderr when
    uncommitted changes exceed threshold.

    REQUIRES: threshold > 0
    ENSURES: Returns None if line_count < threshold
    ENSURES: Returns line_count >= threshold if threshold exceeded
    ENSURES: Emits warning to stderr when threshold exceeded

    Args:
        threshold: Line count threshold for warning (default: 100 lines)

    Returns:
        Line count if over threshold, None otherwise.

    Part of API standardization (Phase 2) per designs/2026-02-01-looper-api-standardization.md
    """
    line_count = get_uncommitted_line_count()
    if line_count < threshold:
        return None

    log_warning(
        f"⚠️  Warning: {line_count} lines of uncommitted changes", stream="stderr"
    )
    log_warning("   Consider committing your work to avoid loss.", stream="stderr")
    return line_count


def check_uncommitted_work_size(threshold: int = 100) -> int | None:
    """Deprecated: Use warn_uncommitted_work() instead.

    Mid-session commit enforcement - warns when uncommitted changes
    grow too large to encourage frequent commits.

    REQUIRES: threshold > 0
    ENSURES: Returns None if line_count < threshold
    ENSURES: Returns line_count >= threshold if threshold exceeded
    ENSURES: Emits warning to stderr when threshold exceeded

    Args:
        threshold: Line count threshold for warning (default: 100 lines)

    Returns:
        Line count if over threshold, None otherwise.

    Part of #1015: Worker needs mid-session commit enforcement
    """
    return warn_uncommitted_work(threshold)


def warn_stale_staged_files(files: list[str], limit: int = 10) -> None:
    """Print warning about stale staged files.

    Pure side effect function: prints warning to stderr about staged files.

    REQUIRES: files is a list of file paths (may be empty)
    ENSURES: If files is empty, no output
    ENSURES: If files is non-empty, prints warning with up to limit files
    ENSURES: Emits warning to stderr

    Args:
        files: List of staged file paths
        limit: Maximum number of files to display (default: 10)

    Part of API standardization (Phase 2) per designs/2026-02-01-looper-api-standardization.md
    """
    if not files:
        return
    log_warning(f"Warning: Found {len(files)} stale staged file(s)", stream="stderr")
    for f in files[:limit]:
        log_warning(f"  - {f}", stream="stderr")
    if len(files) > limit:
        log_warning(f"  ... and {len(files) - limit} more", stream="stderr")


def enforce_no_stale_staged_files() -> None:
    """Raise RuntimeError if staged files exist.

    Enforcement function: raises exception on invalid state.

    REQUIRES: Called from within a git repository
    ENSURES: If no staged files exist, returns None
    ENSURES: If staged files exist, raises RuntimeError with preview

    Raises:
        RuntimeError: If staged files exist.

    Part of API standardization (Phase 2) per designs/2026-02-01-looper-api-standardization.md
    """
    staged = get_staged_files()
    if not staged:
        return
    preview = staged[:5]
    if len(staged) > 5:
        preview.append(f"... and {len(staged) - 5} more")
    raise RuntimeError(f"Stale staged files: {preview}")


def check_stale_staged_files(abort: bool = False) -> list[str]:
    """Deprecated: Use get_staged_files() + warn_stale_staged_files() or enforce_no_stale_staged_files().

    Prevents cross-session staged file contamination by detecting files
    staged by a prior session. In shared working trees, this can cause
    mixed-scope commits.

    REQUIRES: Called from within a git repository
    ENSURES: If abort=False: returns list of staged files (may be empty)
    ENSURES: If abort=True and staged files exist: raises RuntimeError
    ENSURES: If abort=True and no staged files: returns empty list
    ENSURES: Emits warning to stderr when staged files found and abort=False

    Args:
        abort: If True, raise RuntimeError when stale staged files found.
               If False (default), just warn and return the list.

    Returns:
        List of staged file paths.

    Raises:
        RuntimeError: If abort=True and staged files exist.

    Part of #995: Prevent cross-session staged-file contamination
    """
    staged = get_staged_files()
    if not staged:
        return []

    if abort:
        enforce_no_stale_staged_files()  # Will raise

    warn_stale_staged_files(staged)
    return staged


def get_commits_behind(branch: str, target: str = "origin/main") -> int | None:
    """Get number of commits branch is behind target.

    REQUIRES: branch is a valid git ref
    REQUIRES: target is a valid git ref
    ENSURES: Returns non-negative integer on success
    ENSURES: Returns None on git error or invalid refs

    Args:
        branch: The branch to check
        target: The target branch (default: origin/main)

    Returns:
        Number of commits behind, or None on error.
    """
    result = run_git_command(["rev-list", "--count", f"{branch}..{target}"], timeout=10)
    if not result.ok:
        return None
    try:
        return int((result.value or "0").strip())
    except ValueError:
        return None


def get_conflict_files() -> list[str]:
    """Get list of files with conflicts after a failed merge/rebase.

    REQUIRES: Called from within a git repository
    ENSURES: Returns list (possibly empty) of file paths
    ENSURES: Each path is non-empty string with no leading/trailing whitespace
    ENSURES: Returns empty list on git error or no conflicts

    Returns:
        List of conflicting file paths.
    """
    result = run_git_command(["diff", "--name-only", "--diff-filter=U"], timeout=10)
    if not result.ok or not result.value:
        return []
    return [f.strip() for f in result.value.strip().split("\n") if f.strip()]


def sync_from_main(config: SyncConfig | None = None) -> SyncResult:
    """Sync current branch with origin/main.

    In multi-machine mode, secondary machines work on zone branches that
    need to stay up-to-date with main. This function fetches origin/main
    and either rebases or merges to incorporate upstream changes.

    REQUIRES: Called from within a git repository with origin remote
    ENSURES: Returns SyncResult with status reflecting operation outcome
    ENSURES: If status is SYNCED: commits_pulled > 0 and branch is updated
    ENSURES: If status is CONFLICT: conflict_files is non-empty
    ENSURES: If status is BLOCKED: working directory unchanged
    ENSURES: Stashed changes are restored after operation (success or failure)

    Args:
        config: Sync configuration. If None, uses defaults.

    Returns:
        SyncResult with status, commits_pulled, conflicts, etc.

    Note:
        - Only syncs on non-main branches (zone branches)
        - Fails fast on uncommitted changes unless auto_stash is True
        - Git errors when checking uncommitted changes return ERROR status (not BLOCKED)
        - Conflicts cause sync to abort (no auto-resolution)
    """
    if config is None:
        config = SyncConfig()

    # Get current branch
    branch = get_current_branch()
    if branch is None:
        return SyncResult(
            status=SyncStatus.ERROR,
            reason="Not on any branch (detached HEAD?)",
        )

    if branch == "main":
        # On main branch - sync not needed (main pushes directly)
        return SyncResult(
            status=SyncStatus.NOT_APPLICABLE,
            branch=branch,
            reason="On main branch - direct push, no sync needed",
        )

    # Check for uncommitted changes (using Result-returning variant for error details)
    changes_result = get_uncommitted_changes_result()
    if not changes_result.ok:
        # Git status failed - return error with details instead of assuming changes
        return SyncResult(
            status=SyncStatus.ERROR,
            branch=branch,
            reason=f"Failed to check uncommitted changes: {changes_result.error}",
        )

    has_changes = changes_result.value or False
    if has_changes:
        if config.auto_stash:
            # Auto-stash uncommitted changes
            stash_result = run_git_command(
                ["stash", "push", "-m", "auto-stash before sync from main"],
                timeout=30,
            )
            if not stash_result.ok:
                return SyncResult(
                    status=SyncStatus.BLOCKED,
                    branch=branch,
                    reason=f"Failed to stash: {stash_result.error}",
                )
            stashed = True
        else:
            return SyncResult(
                status=SyncStatus.BLOCKED,
                branch=branch,
                reason="Uncommitted changes - commit or use auto_stash",
            )
    else:
        stashed = False

    # Fetch origin/main
    fetch_result = run_git_command(["fetch", "origin", "main"], timeout=60)
    if not fetch_result.ok:
        _restore_stash(stashed)
        return SyncResult(
            status=SyncStatus.FETCH_FAILED,
            branch=branch,
            reason=f"Failed to fetch: {fetch_result.error}",
        )

    # Check if we're behind
    commits_behind = get_commits_behind(branch, "origin/main")
    if commits_behind is None:
        _restore_stash(stashed)
        return SyncResult(
            status=SyncStatus.ERROR,
            branch=branch,
            reason="Failed to check commits behind origin/main",
        )

    if commits_behind == 0:
        _restore_stash(stashed)
        return SyncResult(
            status=SyncStatus.UP_TO_DATE,
            branch=branch,
            strategy=config.strategy,
            reason="Already up to date with origin/main",
        )

    # Perform rebase or merge
    if config.strategy == "rebase":
        result = run_git_command(["rebase", "origin/main"], timeout=120)
        if not result.ok:
            # Capture conflict files BEFORE abort (abort clears conflict state)
            conflict_files = get_conflict_files()
            # Abort rebase on failure
            run_git_command(["rebase", "--abort"], timeout=10)
            _restore_stash(stashed)

            if config.conflict_action == "continue_diverged":
                return SyncResult(
                    status=SyncStatus.DIVERGED,
                    branch=branch,
                    commits_behind=commits_behind,
                    conflict_files=conflict_files,
                    strategy=config.strategy,
                    reason="Rebase failed - continuing diverged",
                )
            return SyncResult(
                status=SyncStatus.CONFLICT,
                branch=branch,
                commits_behind=commits_behind,
                conflict_files=conflict_files,
                strategy=config.strategy,
                reason=f"Rebase conflict: {result.error}",
            )
    else:  # merge
        result = run_git_command(
            ["merge", "origin/main", "-m", f"Merge origin/main into {branch}"],
            timeout=120,
        )
        if not result.ok:
            # Capture conflict files BEFORE abort (abort clears conflict state)
            conflict_files = get_conflict_files()
            # Abort merge on failure
            run_git_command(["merge", "--abort"], timeout=10)
            _restore_stash(stashed)

            if config.conflict_action == "continue_diverged":
                return SyncResult(
                    status=SyncStatus.DIVERGED,
                    branch=branch,
                    commits_behind=commits_behind,
                    conflict_files=conflict_files,
                    strategy=config.strategy,
                    reason="Merge failed - continuing diverged",
                )
            return SyncResult(
                status=SyncStatus.CONFLICT,
                branch=branch,
                commits_behind=commits_behind,
                conflict_files=conflict_files,
                strategy=config.strategy,
                reason=f"Merge conflict: {result.error}",
            )

    # Restore stashed changes
    stash_conflict = _restore_stash(stashed)

    return SyncResult(
        status=SyncStatus.SYNCED,
        branch=branch,
        commits_pulled=commits_behind,
        strategy=config.strategy,
        reason=f"Synced {commits_behind} commit(s) from main"
        + (" (stash restored)" if stashed and not stash_conflict else "")
        + (" (stash conflict - kept in stash)" if stash_conflict else ""),
    )


def _restore_stash(stashed: bool) -> bool:
    """Restore stashed changes if any.

    REQUIRES: If stashed=True, at least one stash entry exists
    ENSURES: If stashed=False: returns False (no-op)
    ENSURES: If stashed=True and pop succeeds: returns False
    ENSURES: If stashed=True and pop fails: returns True (stash preserved)

    Args:
        stashed: Whether changes were stashed before sync.

    Returns:
        True if stash pop failed (conflict), False otherwise.
    """
    if not stashed:
        return False
    result = run_git_command(["stash", "pop"], timeout=30)
    return not result.ok  # Return True if pop failed

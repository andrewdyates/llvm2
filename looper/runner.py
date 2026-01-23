# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
looper/runner.py - Main loop runner class

Contains LoopRunner class and audit prompt generation.

Context Transfer Strategy (Claude vs Codex):
- Claude Code: Uses --resume <session_id> for full conversation context continuity
- Codex: No session resume support with --json output. Instead, injects last
  commit message into audit prompt for context. Limited to 1 audit round.

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0
"""

import json
import os
import random
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# fcntl is Unix-only; Windows uses different locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from looper.config import (
    ITERATION_FILE_TEMPLATE,
    LOG_DIR,
    MAX_CRASH_LOG_LINES,
    MAX_LOG_FILES,
    PID_FILE_TEMPLATE,
    ROLES_DIR,
    STATUS_FILE_TEMPLATE,
    build_codex_context,
    get_project_name,
    inject_content,
    load_role_config,
    parse_frontmatter,
)
from looper.context import (
    get_do_audit_issues,
    get_issue_by_number,
    run_session_start_commands,
    transition_audit_to_review,
)
from looper.hooks import install_hooks
from looper.rotation import get_rotation_focus, update_rotation_state

# --- Audit Prompt ---


def build_audit_prompt(
    round_num: int,
    min_issues: int,
    max_rounds: int,
    last_commit: Optional[str] = None,
) -> str:
    """Build audit prompt for a specific round.

    Args:
        round_num: Current audit round (1-indexed)
        min_issues: Minimum issues to find before stopping
        max_rounds: Maximum number of audit rounds configured
        last_commit: Optional last commit message for context (used when session
            resume isn't available, e.g., Codex)

    Returns:
        Audit prompt string to append to the base prompt.
    """
    context = ""
    if last_commit:
        # Provide context about what was just done (for tools without session resume)
        context = f"""
## Last Commit (your recent work)
```
{last_commit}
```
"""
    return f"""
{context}
## FOLLOW-UP AUDIT (Round {round_num}/{max_rounds})

Self-audit the work just completed. Find at least {min_issues} issues.

If you find issues, fix them and commit. If you find more, fix more.
If you cannot find {min_issues} issues, include `[DONE]` in your commit message and explain why.
"""


def extract_issue_numbers(commit_msg: str) -> list[int]:
    """Extract issue numbers from a commit message.

    Finds Fixes #N, Part of #N, Re: #N, Claims #N patterns.

    Args:
        commit_msg: The commit message text.

    Returns:
        List of issue numbers found.
    """
    # Case-insensitive to match auto-fixed "fixes" from commit-msg-hook
    return [
        int(match.group(1))
        for match in re.finditer(r"(?:Fixes|Part of|Re:|Claims) #(\d+)", commit_msg, re.IGNORECASE)
    ]


class LoopRunner:
    def __init__(self, mode: str):
        self.mode = mode
        self.iteration = 1
        self.running = True
        self.current_process: Optional[subprocess.Popen] = None
        self._current_phase: str | None = None  # Track current rotation phase

        # Load role config from .claude/roles/ files
        self.config, self._prompt_template = load_role_config(mode)

        # File paths
        self.iteration_file = LOG_DIR / ITERATION_FILE_TEMPLATE.format(mode=mode)
        self.pid_file = Path(PID_FILE_TEMPLATE.format(mode=mode))
        self.status_file = Path(STATUS_FILE_TEMPLATE.format(mode=mode))
        self.crash_log = LOG_DIR / "crashes.log"

        # Check for codex availability
        self.codex_available = shutil.which("codex") is not None

        # Session identity
        self._session_id = uuid.uuid4().hex[:6]
        self._started_at = datetime.now(timezone.utc).isoformat()

        # Working issues from main iteration (for audit context)
        self._working_issues: list[int] = []

        # Per-role STOP file (e.g., STOP_WORKER, STOP_MANAGER)
        self._role_stop_file = f"STOP_{mode.upper()}"

        # Pulse (health monitoring) - track last run time
        self._last_pulse_time: float = 0.0

    def _check_stop_file(self) -> str | None:
        """Check for graceful shutdown request files.

        Returns:
            Name of the stop file if found, None otherwise.
            Checks STOP (all roles) first, then STOP_<ROLE> (per-role).
        """
        if Path("STOP").exists():
            return "STOP"
        if Path(self._role_stop_file).exists():
            return self._role_stop_file
        return None

    def _consume_stop_file(self, stop_file: str) -> None:
        """Consume a per-role STOP file after stopping.

        STOP (global) is NOT consumed - user removes it manually so all roles see it.
        STOP_<ROLE> IS consumed - it's targeted at this role only.
        """
        if stop_file == self._role_stop_file:
            try:
                Path(stop_file).unlink()
                print(f"*** {stop_file} consumed ***")
            except OSError:
                pass  # Already removed or permission issue

    def _get_max_git_iteration(self) -> int:
        """Get max iteration from git log by parsing [X]N commits for current role.

        Searches ALL commits to find the highest iteration number used.

        Returns:
            Maximum iteration number found, or 0 if none found.
        """
        try:
            # Get role prefix (W, M, P, R) from config
            role_prefix = self.config["git_author_name"][0].upper()
            # Search all commits, find ALL [X]N patterns for this role
            result = subprocess.run(
                ["git", "log", "--oneline", "--all"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                pattern = rf"\[{role_prefix}\]#?(\d+)"
                max_iteration = 0
                for line in result.stdout.split("\n"):
                    match = re.search(pattern, line)
                    if match:
                        max_iteration = max(max_iteration, int(match.group(1)))
                return max_iteration
        except Exception:
            pass
        return 0

    def _read_commit_tag_file(self, commit_tag_file: Path) -> int:
        """Read current value from commit tag file, returns 0 on error."""
        if commit_tag_file.exists():
            try:
                return int(commit_tag_file.read_text().strip())
            except (ValueError, OSError):
                pass
        return 0

    def _write_commit_tag_file(self, commit_tag_file: Path, value: int) -> bool:
        """Write value to commit tag file atomically. Returns True on success."""
        try:
            # Use PID in temp file name to avoid collisions between processes
            tmp_file = commit_tag_file.with_suffix(f".tmp.{os.getpid()}")
            tmp_file.write_text(str(value))
            tmp_file.rename(commit_tag_file)
            return True
        except OSError:
            return False

    def _acquire_lock_with_timeout(
        self, lock_file, timeout_sec: float = 30.0
    ) -> bool:
        """Try to acquire exclusive lock with timeout. Returns True if acquired."""
        if not HAS_FCNTL:
            return False

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                # Non-blocking lock attempt
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                # Lock held by another process, retry after brief sleep
                time.sleep(0.1)
            except OSError:
                # Other error (e.g., NFS issues), give up on locking
                return False
        return False

    def get_git_iteration(self) -> int:
        """Get next commit tag iteration with file lock coordination.

        Uses file locking to prevent race conditions when multiple sessions
        start concurrently. Coordinates between:
        - Git history (authoritative record of used iterations)
        - Commit tag file (fast local cache, separate from loop counter)

        The lock ensures only one session can read+increment at a time,
        preventing duplicate iteration numbers like [R]267 appearing twice.

        Note: This is separate from self.iteration (loop counter) which tracks
        how many times THIS session has looped. The commit tag iteration is
        unique across ALL sessions of this role.

        Platform notes:
        - Unix: Uses fcntl.flock() with timeout for coordination
        - Windows: Falls back to file-based coordination (less robust)

        Returns:
            Next commit tag iteration (unique across concurrent sessions).
        """
        # Ensure log directory exists for lock file
        LOG_DIR.mkdir(exist_ok=True)

        # Use separate file for commit tag coordination (not loop counter)
        commit_tag_file = LOG_DIR / f".commit_tag_{self.mode}"
        lock_file_path = LOG_DIR / f".commit_tag_lock_{self.mode}"

        # Try to acquire lock (Unix only, with timeout)
        lock_acquired = False
        lock_file = None

        if HAS_FCNTL:
            try:
                lock_file = open(lock_file_path, "w")
                lock_acquired = self._acquire_lock_with_timeout(lock_file, timeout_sec=30.0)
                if not lock_acquired:
                    print("Warning: could not acquire iteration lock (timeout)")
            except OSError as e:
                print(f"Warning: could not open lock file ({e})")

        try:
            # Read current value from commit tag file
            file_iteration = self._read_commit_tag_file(commit_tag_file)

            # Get max from git history
            git_iteration = self._get_max_git_iteration()

            # Next iteration is max of both sources + 1
            next_iteration = max(file_iteration, git_iteration) + 1

            # Persist to commit tag file (atomic via rename with unique temp)
            if not self._write_commit_tag_file(commit_tag_file, next_iteration):
                print("Warning: could not write commit tag file")

            return next_iteration

        finally:
            # Release lock if acquired
            if lock_acquired and lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            if lock_file:
                try:
                    lock_file.close()
                except OSError:
                    pass

    def setup_git_identity(self):
        """Set up rich git identity for commits.

        Format: {project}-{role}-{iteration} <{session}@{machine}.{project}.ai-fleet>
        This enables forensic tracking of which AI session made which commits.
        """
        project = get_project_name()
        role = self.config["git_author_name"].lower()  # "worker" or "manager"
        iteration = self.get_git_iteration()
        machine = socket.gethostname().split(".")[0]
        session = self._session_id

        git_name = f"{project}-{role}-{iteration}"
        git_email = f"{session}@{machine}.{project}.ai-fleet"

        os.environ["GIT_AUTHOR_NAME"] = git_name
        os.environ["GIT_AUTHOR_EMAIL"] = git_email
        os.environ["GIT_COMMITTER_NAME"] = git_name
        os.environ["GIT_COMMITTER_EMAIL"] = git_email

        # Export for MCP tools and other scripts
        os.environ["AI_PROJECT"] = project
        os.environ["AI_ROLE"] = role.upper()
        os.environ["AI_ITERATION"] = str(iteration)
        os.environ["AI_SESSION"] = session
        os.environ["AI_MACHINE"] = machine

        # Export AIT version for tracking (can be used by commit hook)
        ait_version = self.get_ait_version()
        if ait_version:
            os.environ["AIT_VERSION"] = ait_version[0]
            os.environ["AIT_SYNCED"] = ait_version[1]

        # Export coder info for commit signatures
        os.environ["AI_CODER"] = "claude-code"
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split()
                if parts:
                    os.environ["CLAUDE_CODE_VERSION"] = parts[-1]
        except Exception:
            pass  # Version detection is best-effort

        # Prepend wrapper bin to PATH so gh -> gh_post.sh transparently
        wrapper_bin = Path("ai_template_scripts/bin").resolve()
        if wrapper_bin.exists():
            os.environ["PATH"] = f"{wrapper_bin}:{os.environ.get('PATH', '')}"

        print(f"✓ Git identity: {git_name}")

    def get_ait_version(self) -> tuple[str, str] | None:
        """Get ai_template version info from .ai_template_version file.

        Returns:
            (commit_hash, sync_timestamp) or None if file doesn't exist.
        """
        version_file = Path(".ai_template_version")
        if not version_file.exists():
            return None
        try:
            lines = version_file.read_text().strip().split("\n")
            if len(lines) >= 2:
                return (lines[0][:8], lines[1])  # Short hash, timestamp
            if len(lines) == 1:
                return (lines[0][:8], "unknown")
        except Exception:
            pass
        return None

    def setup(self):
        """Initialize environment and check dependencies."""
        # Check core dependencies
        if not shutil.which("claude"):
            print("ERROR: claude CLI not found in PATH")
            print("  Install: npm install -g @anthropic-ai/claude-code")
            sys.exit(1)
        print(f"✓ Found claude: {shutil.which('claude')}")

        codex_prob = self.config.get("codex_probability", 0.0)
        if codex_prob > 0:
            if self.codex_available:
                print(f"✓ Found codex: {shutil.which('codex')} (probability: {codex_prob:.0%})")
            else:
                print(f"  (codex not found, will use claude only - probability was {codex_prob:.0%})")

        json_to_text = Path("ai_template_scripts/json_to_text.py")
        if not json_to_text.exists():
            print(f"ERROR: {json_to_text} not found")
            sys.exit(1)
        print(f"✓ Found {json_to_text}")

        # Check git
        if not shutil.which("git"):
            print("ERROR: git not found in PATH")
            sys.exit(1)
        print(f"✓ Found git: {shutil.which('git')}")

        # Check GitHub CLI
        if not shutil.which("gh"):
            print("ERROR: gh (GitHub CLI) not found in PATH")
            print("  Install: brew install gh")
            sys.exit(1)
        print(f"✓ Found gh: {shutil.which('gh')}")

        # Check gh authentication
        auth_result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if auth_result.returncode != 0:
            print("ERROR: gh not authenticated")
            print("  Run: gh auth login")
            sys.exit(1)
        print("✓ gh authenticated")

        # Check network (non-blocking warning)
        try:
            net_result = subprocess.run(
                ["gh", "api", "user", "--jq", ".login"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if net_result.returncode == 0:
                print(f"✓ GitHub connected as: {net_result.stdout.strip()}")
            else:
                print("⚠ GitHub API unreachable (offline mode)")
        except subprocess.TimeoutExpired:
            print("⚠ GitHub API timeout (offline mode)")

        # Create log directory
        LOG_DIR.mkdir(exist_ok=True)

        # Install git hooks
        install_hooks()

        # Set up git identity for commits
        self.setup_git_identity()

        # Check for existing instance
        if self.pid_file.exists():
            try:
                old_pid = int(self.pid_file.read_text().strip())
                # Check if process is still running
                os.kill(old_pid, 0)  # Doesn't kill, just checks
                print(
                    f"ERROR: Another {self.mode} loop is already running (PID {old_pid})"
                )
                print(f"  Stop it first or remove {self.pid_file}")
                sys.exit(1)
            except (ProcessLookupError, ValueError):
                # Process not running, clean up stale PID file
                self.pid_file.unlink()

        # Write our PID
        self.pid_file.write_text(str(os.getpid()))

        # Restore iteration counter
        if self.iteration_file.exists():
            try:
                self.iteration = int(self.iteration_file.read_text().strip())
                print(f"Resuming from iteration {self.iteration}")
            except ValueError:
                self.iteration = 1

        # Rotate logs
        self.rotate_logs()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        # Track start time for status
        self._started_at = datetime.now(timezone.utc).isoformat()
        self.write_status("starting")

        print()
        print(f"Starting {self.mode} loop...")
        print()

    def handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print()
        print(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.current_process:
            self.current_process.terminate()

    def cleanup(self):
        """Clean up on exit."""
        self.clear_status()
        if self.pid_file.exists():
            self.pid_file.unlink()
        print(f"Completed {self.iteration - 1} iterations")

    def run_pulse(self) -> bool:
        """Run pulse.py to update health metrics and flags.

        Checks if enough time has passed since last run (pulse_interval_minutes).
        Manager reads .flags/ to act on issues - it doesn't run pulse directly.

        Returns:
            True if pulse ran, False if skipped (not yet due).
        """
        interval_minutes = self.config.get("pulse_interval_minutes", 30)
        now = time.time()

        # Check if enough time has passed
        elapsed_minutes = (now - self._last_pulse_time) / 60
        if self._last_pulse_time > 0 and elapsed_minutes < interval_minutes:
            return False

        # Run pulse.py as subprocess (quiet mode - just writes files)
        pulse_script = Path("ai_template_scripts/pulse.py")
        if not pulse_script.exists():
            return False

        try:
            result = subprocess.run(
                ["python3", str(pulse_script)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._last_pulse_time = now

            if result.returncode == 0:
                # Check for flags set
                flags_dir = Path(".flags")
                flags = list(flags_dir.glob("*")) if flags_dir.exists() else []
                if flags:
                    flag_names = ", ".join(f.name for f in flags[:5])
                    print(f"⚡ Pulse: {flag_names}")
                else:
                    print("⚡ Pulse: OK (no flags)")
                return True
            print(f"⚠️  Pulse error: {result.stderr[:100] if result.stderr else 'unknown'}")
            return False
        except subprocess.TimeoutExpired:
            print("⚠️  Pulse timeout")
            return False
        except Exception as e:
            print(f"⚠️  Pulse failed: {e}")
            return False

    def rotate_logs(self):
        """Remove old log files to prevent unbounded growth."""
        log_files = sorted(LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if len(log_files) > MAX_LOG_FILES:
            for f in log_files[:-MAX_LOG_FILES]:
                f.unlink()
            print(f"Rotated logs: removed {len(log_files) - MAX_LOG_FILES} old files")

        # Rotate crash log
        if self.crash_log.exists():
            lines = self.crash_log.read_text().splitlines()
            if len(lines) > MAX_CRASH_LOG_LINES:
                self.crash_log.write_text(
                    "\n".join(lines[-MAX_CRASH_LOG_LINES:]) + "\n"
                )

    def write_status(
        self,
        status: str,
        log_file: Optional[Path] = None,
        extra: Optional[dict[str, Any]] = None,
    ):
        """Write worker status to .worker_status.json for manager visibility."""
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "pid": os.getpid(),
            "mode": self.mode,
            "project": get_project_name(),
            "iteration": self.iteration,
            "status": status,
            "updated_at": now,
            "started_at": getattr(self, "_started_at", now),
        }
        # Include AIT version for tracking
        ait_version = self.get_ait_version()
        if ait_version:
            data["ait_version"] = ait_version[0]
            data["ait_synced"] = ait_version[1]
        if log_file:
            data["log_file"] = str(log_file)
        if extra:
            data.update(extra)

        # Atomic write
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self.status_file)

    def clear_status(self):
        """Remove status file on clean exit."""
        if self.status_file.exists():
            self.status_file.unlink()

    def check_stuck_issues(self) -> list[tuple[int, int]]:
        """Check for issues that may be stuck (many references, not closed).

        Returns list of (issue_number, reference_count) for potentially stuck issues.
        Only runs for manager mode.
        """
        if self.mode != "manager":
            return []

        try:
            # Get last 20 commits and count issue references
            result = subprocess.run(
                ["git", "log", "--oneline", "-20", "--format=%s %b"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return []

            # Count references to each issue (case-insensitive for auto-fixed "fixes")
            issue_counts: dict[int, int] = {}
            for match in re.finditer(r'(?:Fixes|Part of|Re:|Reopens) #(\d+)', result.stdout, re.IGNORECASE):
                issue_num = int(match.group(1))
                issue_counts[issue_num] = issue_counts.get(issue_num, 0) + 1

            # Filter to issues with 5+ references (potentially stuck)
            stuck = [(num, count) for num, count in issue_counts.items() if count >= 5]
            return sorted(stuck, key=lambda x: -x[1])  # Sort by count descending
        except Exception:
            return []

    def check_thrashing_issues(self) -> list[tuple[int, int]]:
        """Check for issues with multiple close/reopen cycles (thrashing).

        Returns list of (issue_number, reopen_count) for thrashing issues.
        Only runs for manager mode. Thrashing indicates the fix wasn't actually
        correct - escalate to Prover for verification and Researcher for analysis.
        """
        if self.mode != "manager":
            return []

        try:
            # Get repo info
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return []
            repo = result.stdout.strip()

            # Get open issues with in-progress label (actively worked)
            result = subprocess.run(
                ["gh", "issue", "list", "--label", "in-progress", "--json", "number"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return []

            issues = json.loads(result.stdout)
            thrashing = []

            for issue in issues[:10]:  # Limit to 10 issues to avoid API rate limits
                issue_num = issue["number"]
                # Get timeline events for this issue
                result = subprocess.run(
                    ["gh", "api", f"repos/{repo}/issues/{issue_num}/timeline",
                     "--jq", '[.[] | select(.event=="reopened")] | length'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    reopen_count = int(result.stdout.strip() or "0")
                    if reopen_count >= 2:  # 2+ reopens = thrashing
                        thrashing.append((issue_num, reopen_count))

            return sorted(thrashing, key=lambda x: -x[1])  # Sort by count descending
        except Exception:
            return []

    def report_stuck_issues(self):
        """Print warning about potentially stuck issues and thrashing."""
        stuck = self.check_stuck_issues()
        thrashing = self.check_thrashing_issues()

        if stuck:
            print()
            print("⚠️  STUCK ISSUE DETECTOR:")
            for issue_num, count in stuck[:3]:  # Top 3 most referenced
                print(f"    #{issue_num}: {count} references in last 20 commits without closing")
            print("    Consider: different approach, escalate, or close as won't-fix")
            print()

        if thrashing:
            print()
            print("🔄 THRASHING DETECTOR:")
            for issue_num, reopen_count in thrashing[:3]:  # Top 3 most thrashed
                print(f"    #{issue_num}: {reopen_count} close/reopen cycles")
            print("    Escalate: Prover (verify original fix), Researcher (systemic analysis)")
            print()

    def _get_issue_checkboxes(self, issue_num: int) -> tuple[list[str], list[str], str]:
        """Get all checkboxes from issue body and comments.

        Returns (unchecked_items, checked_items, original_body).
        Items are deduplicated and normalized.
        """
        unchecked = []
        checked = []
        original_body = ""

        try:
            # Get issue body
            result = subprocess.run(
                ["gh", "issue", "view", str(issue_num), "--json", "body", "-q", ".body"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout:
                original_body = result.stdout
                unchecked.extend(re.findall(r'- \[ \] (.+?)(?:\n|$)', original_body))
                checked.extend(re.findall(r'- \[x\] (.+?)(?:\n|$)', original_body, re.IGNORECASE))

            # Get issue comments
            result = subprocess.run(
                ["gh", "issue", "view", str(issue_num), "--json", "comments",
                 "-q", ".comments[].body"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.split('\n'):
                    unchecked.extend(re.findall(r'- \[ \] (.+?)(?:\n|$)', line))
                    checked.extend(re.findall(r'- \[x\] (.+?)(?:\n|$)', line, re.IGNORECASE))

        except Exception:
            pass

        # Normalize and deduplicate
        unchecked = list(dict.fromkeys(item.strip() for item in unchecked if item.strip()))
        checked = list(dict.fromkeys(item.strip() for item in checked if item.strip()))

        # Remove items that appear in both (checked wins)
        checked_lower = {item.lower() for item in checked}
        unchecked = [item for item in unchecked if item.lower() not in checked_lower]

        return unchecked, checked, original_body

    def _get_existing_child_issues(self, parent_num: int) -> set[str]:
        """Get titles of existing child issues (Part of #N) to avoid duplicates."""
        existing = set()
        try:
            result = subprocess.run(
                ["gh", "issue", "list", "--search", f"Part of #{parent_num} in:body",
                 "--json", "title", "-q", ".[].title"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                for title in result.stdout.strip().split('\n'):
                    if title.strip():
                        existing.add(title.strip().lower())
        except Exception:
            pass
        return existing

    def _update_issue_body_with_consolidated_checkboxes(
        self, issue_num: int, unchecked: list[str], checked: list[str], original_body: str
    ) -> bool:
        """Update issue body with consolidated and cleaned checkbox list.

        Removes scattered checkboxes from body and adds a clean consolidated section.
        Returns True if update succeeded.
        """
        try:
            # Remove existing checkbox lines and ## Checklist header from body
            clean_body = re.sub(r'- \[[x ]\] .+?\n?', '', original_body, flags=re.IGNORECASE)
            clean_body = re.sub(r'\n*## Checklist\n*', '\n', clean_body)
            clean_body = clean_body.strip()

            # Build consolidated checkbox section
            checkbox_section = ""
            if checked or unchecked:
                checkbox_section = "\n\n## Checklist\n"
                for item in checked:
                    checkbox_section += f"- [x] {item}\n"
                for item in unchecked:
                    checkbox_section += f"- [ ] {item}\n"

            new_body = clean_body + checkbox_section

            # Update issue
            result = subprocess.run(
                ["gh", "issue", "edit", str(issue_num), "--body", new_body],
                capture_output=True, text=True, timeout=15
            )
            return result.returncode == 0
        except Exception:
            return False

    def convert_unchecked_to_issues(self) -> int:
        """Convert unchecked checkboxes in worked issues to new issues.

        Automatically creates new issues for any unchecked `- [ ]` items found in
        issues referenced by the last commit. New issues include "Part of #N" to
        link back to the parent issue.

        Also consolidates checkboxes from comments into issue body and deduplicates.

        Returns number of issues created.
        """
        try:
            # Get issue numbers from last commit
            result = subprocess.run(
                ["git", "log", "-1", "--format=%s %b"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return 0

            # Find issue references (case-insensitive for auto-fixed "fixes")
            issue_nums = set()
            for match in re.finditer(r'(?:Fixes|Part of|Re:|Claims) #(\d+)', result.stdout, re.IGNORECASE):
                issue_nums.add(int(match.group(1)))

            if not issue_nums:
                return 0

            created_count = 0

            for issue_num in list(issue_nums)[:5]:  # Limit to 5 issues
                # Get all checkboxes from body and comments
                unchecked, checked, original_body = self._get_issue_checkboxes(issue_num)

                if not unchecked and not checked:
                    continue

                # Get existing child issues to avoid duplicates
                existing_children = self._get_existing_child_issues(issue_num)

                # Create new issues for unchecked items (that don't already exist)
                items_to_convert = []
                for item_text in unchecked[:10]:  # Limit to 10 items per issue
                    if item_text.lower() in existing_children:
                        print(f"    ⊘ Skipped (exists): {item_text[:40]}...")
                        continue
                    items_to_convert.append(item_text)

                for item_text in items_to_convert:
                    new_body = f"Part of #{issue_num}\n\nAuto-created from unchecked item in parent issue."
                    create_result = subprocess.run(
                        ["gh", "issue", "create",
                         "--title", item_text,
                         "--body", new_body],
                        capture_output=True, text=True, timeout=30
                    )

                    if create_result.returncode == 0:
                        created_count += 1
                        new_url = create_result.stdout.strip()
                        print(f"    ✓ Created: {item_text[:50]}{'...' if len(item_text) > 50 else ''}")
                        print(f"      {new_url}")
                        # Mark as checked now that issue exists
                        checked.append(item_text)
                        unchecked.remove(item_text)

                # Consolidate checkboxes in issue body
                if original_body:
                    if self._update_issue_body_with_consolidated_checkboxes(
                        issue_num, unchecked, checked, original_body
                    ):
                        print(f"    ✓ Consolidated checkboxes in #{issue_num}")

            return created_count
        except Exception as e:
            print(f"    ⚠️  Error converting checkboxes: {e}")
            return 0

    def process_unchecked_checkboxes(self):
        """Convert unchecked checkboxes to new issues at session end."""
        print()
        print("📋 Processing checkboxes...")
        created = self.convert_unchecked_to_issues()
        if created > 0:
            print(f"    Created {created} new issue(s) from unchecked items")
        print()

    def _display_prompt_info(
        self, replacements: dict[str, str], final_prompt: str, ai_tool: str = "claude", audit_round: int = 0
    ) -> None:
        """Display prompt assembly information to console.

        Shows:
        - Source files used
        - Injected content summaries
        - Final prompt with clear section markers
        """
        print("### PROMPT SOURCES ###")
        idx = 1
        # For Codex, show the additional sources it receives
        if ai_tool == "codex":
            if Path("CLAUDE.md").exists():
                print(f"  {idx}. CLAUDE.md (via build_codex_context)")
                idx += 1
            rules_dir = Path(".claude/rules")
            if rules_dir.exists():
                rules_files = sorted(rules_dir.glob("*.md"))
                if rules_files:
                    print(f"  {idx}. .claude/rules/*.md ({len(rules_files)} files)")
                    idx += 1
        print(f"  {idx}. {ROLES_DIR / 'shared.md'}")
        idx += 1
        print(f"  {idx}. {ROLES_DIR / f'{self.mode}.md'}")
        idx += 1
        if Path(".looper_config.json").exists():
            print(f"  {idx}. .looper_config.json (overrides)")
            idx += 1
        if audit_round > 0:
            max_rounds = self.config.get("audit_max_rounds", 5)
            print(f"  {idx}. AUDIT_PROMPT (round {audit_round}/{max_rounds})")
        print()

        print("### INJECTED CONTENT ###")
        for key, value in replacements.items():
            if value:
                lines = value.split("\n")
                line_count = len(lines)
                preview = lines[0][:50]
                if len(lines[0]) > 50:
                    preview += "..."
                print(f"  {key}: {preview} ({line_count} lines)")
            else:
                print(f"  {key}: (empty)")
        print()

        print("### FINAL PROMPT ###")
        print("-" * 70)
        print(final_prompt)
        print("-" * 70)
        print(f"Total: {len(final_prompt)} chars, {final_prompt.count(chr(10)) + 1} lines")
        print()

    def select_ai_tool(self) -> str:
        """Select which AI tool to use for this iteration.

        Uses random selection based on codex_probability config:
        - 0.0 = always Claude (default)
        - 0.5 = 50/50 random
        - 1.0 = always Codex (if available)

        Falls back to Claude if Codex is not installed.
        """
        codex_prob = self.config.get("codex_probability", 0.0)

        # Skip if Codex not available or probability is 0
        if not self.codex_available or codex_prob <= 0:
            return "claude"

        # Always Codex if probability is 1.0
        if codex_prob >= 1.0:
            return "codex"

        # Random selection
        if random.random() < codex_prob:
            return "codex"
        return "claude"

    def _extract_session_id(self, log_file: Path, ai_tool: str) -> Optional[str]:
        """Extract session/thread ID from the JSON log file.

        Claude: session_id in final result line - used for --resume in audit rounds
        Codex: thread_id in first line - extracted for logging but NOT used for resume
               (codex exec resume doesn't support --json output needed for headless mode)
        """
        try:
            if not log_file.exists():
                return None
            content = log_file.read_text()
            lines = content.strip().split("\n")

            if ai_tool == "codex":
                # Codex: thread_id in first line
                for line in lines:
                    if '"thread_id"' in line:
                        data = json.loads(line)
                        return data.get("thread_id")
            else:
                # Claude: session_id in last result line
                for line in reversed(lines):
                    if '"session_id"' in line:
                        data = json.loads(line)
                        return data.get("session_id")
        except Exception:
            pass
        return None

    def run_iteration(
        self,
        audit_round: int = 0,
        resume_session_id: Optional[str] = None,
        force_ai_tool: Optional[str] = None,
        force_codex_model: Optional[str] = None,
    ) -> tuple[int, float, str, Optional[str], Optional[str]]:
        """Run a single AI iteration.

        Args:
            audit_round: 0 for main iteration, 1-N for audit rounds.
            resume_session_id: Session ID to resume (for audit continuity).
            force_ai_tool: Force a specific AI tool (used for audit consistency).
            force_codex_model: Force a specific codex model (used for audit consistency).

        Returns:
            (exit_code, start_time, ai_tool, session_id, codex_model)
        """
        is_audit = audit_round > 0

        # For audit rounds: minimal context injection
        # - Claude: resumed session has full conversation context
        # - Codex: last commit injected via build_audit_prompt
        # Either way, only inject working issue, not all issues
        if is_audit:
            # Minimal context for audit - just git_log and working issue
            git_log = "(see resumed session)"
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", "-3"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    git_log = result.stdout.strip()
            except Exception:
                pass

            # Get only the working issue(s), not all issues
            if self._working_issues:
                issue_lines = []
                for issue_num in self._working_issues[:3]:  # Max 3 working issues
                    issue_str = get_issue_by_number(issue_num)
                    if issue_str:
                        issue_lines.append(issue_str)
                gh_issues = "\n".join(issue_lines) if issue_lines else "(no working issue)"
            else:
                gh_issues = "(no working issue tracked)"

            replacements = {
                "git_log": git_log,
                "gh_issues": gh_issues,
                "last_directive": "",  # Session resume has full context
                "other_feedback": "",
                "role_mentions": "",
                "system_status": "",
                "audit_data": "",
                "rotation_focus": "",
            }
            selected_phase = self._current_phase  # Keep same phase
        else:
            # Main iteration: full context injection
            session_results = run_session_start_commands(self.mode)

            # Calculate rotation focus if this role has rotation
            rotation_type = self.config.get("rotation_type", "")
            rotation_phases = self.config.get("rotation_phases", [])
            phase_data = self.config.get("phase_data", {})
            freeform_frequency = self.config.get("freeform_frequency", 3)
            force_phase = self.config.get("force_phase")
            starvation_hours = self.config.get("starvation_hours", 24)
            rotation_focus, selected_phase = get_rotation_focus(
                iteration=self.iteration,
                rotation_type=rotation_type,
                phases=rotation_phases,
                phase_data=phase_data,
                role=self.mode,
                freeform_frequency=freeform_frequency,
                force_phase=force_phase,
                starvation_hours=starvation_hours,
            )

            # Build replacements for injection
            replacements = {
                "git_log": session_results.get("git_log", "(unavailable)"),
                "gh_issues": session_results.get("gh_issues", "(unavailable)"),
                "last_directive": session_results.get("last_directive", ""),
                "other_feedback": session_results.get("other_feedback", ""),
                "role_mentions": session_results.get("role_mentions", ""),
                "system_status": session_results.get("system_status", ""),
                "audit_data": session_results.get("audit_data", ""),
                "rotation_focus": rotation_focus,
            }

        # Store for state update after iteration
        self._current_phase = selected_phase

        # Inject dynamic content into prompt template
        prompt = inject_content(self._prompt_template, replacements)

        # Select AI tool (forced for audit consistency, or random for main)
        ai_tool = force_ai_tool if force_ai_tool else self.select_ai_tool()

        # Append audit prompt if this is an audit iteration
        if audit_round > 0:
            min_issues = self.config.get("audit_min_issues", 5)
            max_rounds = self.config.get("audit_max_rounds", 5)
            # For Codex audits, inject last commit since session resume isn't available
            last_commit = None
            if ai_tool == "codex":
                last_commit = self._get_last_commit_message()
            prompt = prompt + build_audit_prompt(
                audit_round, min_issues, max_rounds, last_commit
            )

        # Build the full prompt (differs for Claude vs Codex)
        codex_model = None  # Will be set if ai_tool == "codex"
        if ai_tool == "claude":
            final_prompt = prompt
            cmd = [
                "claude",
                "--dangerously-skip-permissions",
                "-p",
                final_prompt,
                "--permission-mode",
                "acceptEdits",
                "--output-format",
                "stream-json",
                "--verbose",
            ]
            # Add model if configured
            claude_model = self.config.get("claude_model")
            if claude_model:
                cmd.extend(["--model", claude_model])
            # Resume previous session for audit continuity
            if resume_session_id:
                cmd.extend(["--resume", resume_session_id])
        else:
            # Codex: prepend rules files (Claude reads them automatically, Codex doesn't)
            # Plus explicit commit instruction for headless mode
            codex_context = build_codex_context()
            final_prompt = (
                codex_context
                + prompt
                + """

IMPORTANT: After completing the work, YOU MUST create the git commit immediately using the commit template. Do NOT ask for permission - just commit. This is headless autonomous mode."""
            )
            # Note: codex exec resume doesn't support --json output, so we always
            # start fresh sessions. Audit continuity comes from git context injection.
            # The resume_session_id is unused for Codex (Claude uses it for --resume).
            cmd = [
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--json",
                final_prompt,
            ]
            # Add model if configured (e.g. o4-mini, gpt-5.2, gpt-5.2-codex)
            # force_codex_model: used for audit consistency (same model as main iteration)
            # codex_models (list): randomly selects one for main iterations
            # codex_model (single): fallback
            if force_codex_model:
                codex_model = force_codex_model
            else:
                codex_models = self.config.get("codex_models", [])
                if codex_models:
                    codex_model = random.choice(codex_models)
                else:
                    codex_model = self.config.get("codex_model")
            if codex_model:
                cmd.extend(["--model", codex_model])

        # Format iteration number: 5 for main, 5.1/5.2 for audits
        if is_audit:
            iter_display = f"{self.iteration}.{audit_round}"
        else:
            iter_display = str(self.iteration)
        # Print iteration header with settings
        print()
        print("=" * 70)
        print(f"=== {self.mode.title()} {iter_display}")
        print(f"=== Started at {datetime.now()}")
        print(f"=== Tool: {ai_tool}")
        if ai_tool == "claude":
            model = self.config.get("claude_model", "default")
            print(f"=== Model: {model}")
        elif ai_tool == "codex":
            # Show selected model (may have been randomly chosen from codex_models)
            print(f"=== Model: {codex_model or 'default'}")
        ait_version = self.get_ait_version()
        if ait_version:
            print(f"=== AIT: {ait_version[0]} (synced {ait_version[1]})")
        print("=" * 70)
        print()

        # Display prompt assembly info (shows the actual prompt being sent)
        self._display_prompt_info(replacements, final_prompt, ai_tool, audit_round)

        # Report stuck issues (manager only)
        self.report_stuck_issues()

        # Prepare log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = (
            LOG_DIR / f"{self.mode}_iter_{iter_display}_{ai_tool}_{timestamp}.jsonl"
        )

        # Update status for manager visibility
        status_extra: dict[str, Any] = {"ai_tool": ai_tool, "audit_round": audit_round}
        self.write_status("working", log_file, status_extra)

        # Set coder info for commit signatures (must reset each iteration)
        if ai_tool == "codex":
            os.environ["AI_CODER"] = "codex"
            os.environ.pop("CLAUDE_CODE_VERSION", None)
            try:
                result = subprocess.run(
                    ["codex", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split()
                    if parts:
                        os.environ["CODEX_CLI_VERSION"] = parts[-1]
            except Exception:
                pass
        else:
            # Reset to Claude for this iteration
            os.environ["AI_CODER"] = "claude-code"
            os.environ.pop("CODEX_CLI_VERSION", None)

        # Run AI with output piped through json_to_text.py
        timeout_sec = self.config["iteration_timeout"]
        silence_timeout_sec = self.config.get("silence_timeout", 600)  # Default 10 min
        progress_interval_sec = 60  # Show progress every 60s during long commands
        start_time = time.time()
        last_output_time = start_time  # Track for silence detection (sleep/resume)
        last_progress_time = start_time  # Track for progress indicator
        last_command: Optional[str] = None  # Track running command for timeout message
        last_command_start: Optional[float] = None
        exit_code = 0
        timed_out = False
        silence_killed = False
        text_proc_alive = True

        try:
            with open(log_file, "w") as log_f:
                ai_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self.current_process = ai_proc

                text_proc = subprocess.Popen(
                    ["./ai_template_scripts/json_to_text.py"],
                    stdin=subprocess.PIPE,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )

                def write_to_text_proc(data: bytes) -> None:
                    """Write to text_proc, handling failures gracefully."""
                    nonlocal text_proc_alive
                    if not text_proc_alive:
                        return
                    try:
                        text_proc.stdin.write(data)
                        text_proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        # text_proc exited early - continue draining ai_proc
                        # to prevent EPIPE in Claude CLI
                        text_proc_alive = False

                # Stream output with timeout checking
                # CRITICAL: Always drain ai_proc.stdout completely to prevent EPIPE
                # Even if text_proc dies, we must keep reading from ai_proc
                try:
                    while ai_proc.poll() is None:
                        now = time.time()

                        # Check iteration timeout (total time limit)
                        if now - start_time > timeout_sec:
                            print(f"\nTimeout after {timeout_sec // 60} minutes")
                            # Graceful shutdown: SIGTERM first, then SIGKILL
                            ai_proc.terminate()  # SIGTERM
                            try:
                                ai_proc.wait(timeout=10)  # Grace period
                                print("Process terminated gracefully")
                            except subprocess.TimeoutExpired:
                                print("Grace period expired, sending SIGKILL")
                                ai_proc.kill()
                            timed_out = True
                            break

                        # Progress indicator for long-running commands
                        if last_command and now - last_progress_time > progress_interval_sec:
                            elapsed = int(now - (last_command_start or start_time))
                            elapsed_min = elapsed // 60
                            elapsed_sec = elapsed % 60
                            if elapsed_min > 0:
                                print(f"  ⏳ Running: {last_command[:60]} ({elapsed_min}m {elapsed_sec}s)")
                            else:
                                print(f"  ⏳ Running: {last_command[:60]} ({elapsed_sec}s)")
                            last_progress_time = now

                        # Check silence timeout (detects stale connection after sleep/resume)
                        if now - last_output_time > silence_timeout_sec:
                            silence_mins = int((now - last_output_time) / 60)
                            print(f"\nSilence timeout: no output for {silence_mins} minutes")
                            if last_command:
                                cmd_elapsed = int(now - (last_command_start or start_time))
                                print(f"Last activity: {last_command[:80]} (started {cmd_elapsed // 60}m ago)")
                            print("Connection may be stale (e.g., after sleep/resume)")
                            ai_proc.terminate()
                            try:
                                ai_proc.wait(timeout=10)
                                print("Process terminated gracefully")
                            except subprocess.TimeoutExpired:
                                print("Grace period expired, sending SIGKILL")
                                ai_proc.kill()
                            silence_killed = True
                            break

                        # Non-blocking read
                        ready, _, _ = select.select([ai_proc.stdout], [], [], 1.0)
                        if ready:
                            line = ai_proc.stdout.readline()
                            if line:
                                last_output_time = time.time()  # Reset silence timer
                                last_progress_time = time.time()  # Reset progress timer
                                log_f.write(line.decode())
                                log_f.flush()
                                write_to_text_proc(line)

                                # Track running commands for progress/timeout messages
                                try:
                                    msg = json.loads(line.decode())
                                    # Claude format: tool_use with Bash
                                    if msg.get("type") == "assistant":
                                        content = msg.get("message", {}).get("content", [])
                                        for block in content:
                                            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                                                cmd = block.get("input", {}).get("command", "")
                                                if cmd:
                                                    last_command = cmd[:100]
                                                    last_command_start = time.time()
                                    # Codex format: item.started with command_execution
                                    elif msg.get("type") == "item.started":
                                        item = msg.get("item", {})
                                        if item.get("type") == "command_execution":
                                            cmd = item.get("command", "")
                                            if cmd:
                                                last_command = cmd[:100]
                                                last_command_start = time.time()
                                    # Clear command on completion (Codex)
                                    elif msg.get("type") == "item.completed":
                                        item = msg.get("item", {})
                                        if item.get("type") == "command_execution":
                                            last_command = None
                                            last_command_start = None
                                    # Clear command on tool_result (Claude)
                                    elif msg.get("role") == "user":
                                        content = msg.get("content", [])
                                        for block in content:
                                            if block.get("type") == "tool_result":
                                                last_command = None
                                                last_command_start = None
                                                break
                                except (json.JSONDecodeError, KeyError, TypeError):
                                    pass  # Not JSON or unexpected format

                    # Drain remaining output (always, even after timeout)
                    # Use timeout to prevent hanging on stuck processes
                    drain_timeout = 30  # seconds
                    drain_start = time.time()
                    while time.time() - drain_start < drain_timeout:
                        ready, _, _ = select.select([ai_proc.stdout], [], [], 1.0)
                        if ready:
                            line = ai_proc.stdout.readline()
                            if not line:
                                break  # EOF
                            log_f.write(line.decode())
                            log_f.flush()
                            write_to_text_proc(line)
                        elif ai_proc.poll() is not None:
                            break  # Process exited and no more data
                finally:
                    if text_proc_alive:
                        try:
                            text_proc.stdin.close()
                        except (BrokenPipeError, OSError):
                            pass
                    text_proc.wait()

                # Exit codes: 124=timeout, 125=silence timeout, else process exit code
                if timed_out:
                    exit_code = 124
                elif silence_killed:
                    exit_code = 125
                else:
                    exit_code = ai_proc.returncode or 0
                self.current_process = None

        except Exception as e:
            print(f"Error running {ai_tool}: {e}")
            exit_code = 1

        print()
        print(f"=== {self.mode.title()} {iter_display} ({ai_tool}) completed ===")
        print(f"=== Exit code: {exit_code} ===")
        print(f"=== Log saved to: {log_file} ===")

        # Extract session_id from log for potential resume
        session_id = self._extract_session_id(log_file, ai_tool)
        if session_id:
            print(f"=== Session: {session_id[:8]}... ===")
        print()

        # Update rotation state if we ran a focused phase (not freeform)
        # Skip for audit iterations - they continue the same phase, not a new one
        if self._current_phase and not is_audit:
            update_rotation_state(self.mode, self._current_phase)

        return exit_code, start_time, ai_tool, session_id, codex_model

    def _get_last_commit_message(self) -> Optional[str]:
        """Get the most recent commit message for context injection.

        Returns the full commit message (subject + body) of the last commit,
        or None if unavailable.
        """
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%B"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def check_session_success(self, start_time: float) -> bool:
        """Check if the AI session made a commit during this iteration.

        A session that committed is considered successful even if the exit code
        is non-zero (e.g., EPIPE at the end after work completed).
        """
        try:
            # Get commits from the last iteration window
            # Use --since with a timestamp slightly before start_time
            since_time = int(start_time) - 60  # 1 minute buffer
            # Match git author format: {project}-{role}-{iteration}
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--oneline",
                    f"--since={since_time}",
                    "--author=-worker-",
                    "--author=-prover-",
                    "--author=-researcher-",
                    "--author=-manager-",
                    "-1",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # If we got a commit hash, the session was successful
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def check_incomplete_commit(self, start_time: float) -> bool:
        """Check if the last commit was marked [INCOMPLETE].

        If so, the next session should continue immediately (0 delay).
        """
        return self._check_commit_marker(start_time, "[INCOMPLETE]")

    def check_done_commit(self, start_time: float) -> bool:
        """Check if the last commit was marked [DONE].

        Used by audit loop to detect when AI signals no more issues to find.
        """
        return self._check_commit_marker(start_time, "[DONE]")

    def _check_commit_marker(self, start_time: float, marker: str) -> bool:
        """Check if the last commit contains a specific marker."""
        try:
            since_time = int(start_time) - 60
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--format=%s",
                    f"--since={since_time}",
                    "--author=-worker-",
                    "--author=-prover-",
                    "--author=-researcher-",
                    "--author=-manager-",
                    "-1",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return marker in result.stdout
            return False
        except Exception:
            return False

    def commit_uncommitted_changes(self) -> bool:
        """Check for and commit any uncommitted changes after session ends.

        Returns True if changes were committed, False otherwise.
        Prevents work from being lost between sessions.
        """
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False  # No changes or error

            # There are uncommitted changes - commit them as WIP
            print("\n⚠️  Uncommitted changes detected - auto-committing as WIP")

            # Stage all changes
            add_result = subprocess.run(["git", "add", "-A"], timeout=10, capture_output=True)
            if add_result.returncode != 0:
                print(f"Warning: git add failed: {add_result.stderr}")
                return False

            # Create WIP commit
            commit_msg = f"[{self.mode[0].upper()}]{self.iteration}: WIP - session ended with uncommitted changes\n\n## Note\nAuto-committed by looper.py to prevent work loss.\nReview and amend or continue in next session."
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print("✓ WIP commit created")

                # Get commit hash for issue
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"

                # Push the commit
                push_result = subprocess.run(["git", "push"], capture_output=True, timeout=60)
                if push_result.returncode != 0:
                    print(f"Warning: git push failed (commit is local): {push_result.stderr}")

                # Create follow-up issue to track WIP (#266)
                self._create_wip_followup_issue(commit_hash)

                return True  # Commit succeeded even if push failed
            print(f"⚠️  WIP commit failed: {result.stderr}")
            return False

        except Exception as e:
            print(f"⚠️  Could not check/commit uncommitted changes: {e}")
            return False

    def _create_wip_followup_issue(self, commit_hash: str) -> None:
        """Create a GitHub issue to track WIP commit follow-up.

        This ensures WIP commits don't get forgotten - they appear in the issue list
        for the next session to pick up.
        """
        try:
            role_char = self.mode[0].upper()
            title = f"WIP follow-up: [{role_char}]{self.iteration} ({commit_hash})"
            body = f"""Session ended with uncommitted work.

**Commit:** {commit_hash}
**Role:** {self.mode.upper()}
**Iteration:** {self.iteration}

## Action Required
- [ ] Review changes in commit `{commit_hash}`
- [ ] Continue work OR amend with proper message OR revert if broken

---
Auto-created by looper.py to prevent work loss.
"""
            result = subprocess.run(
                ["gh", "issue", "create", "--title", title, "--body", body, "--label", "wip-followup"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Extract issue URL from output
                issue_url = result.stdout.strip()
                print(f"✓ Created follow-up issue: {issue_url}")
            else:
                print(f"⚠️  Could not create WIP follow-up issue: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not create WIP follow-up issue: {e}")

    def log_crash(self, exit_code: int, ai_tool: str, session_committed: bool = False):
        """Log crash or exit details.

        Args:
            exit_code: The process exit code
            ai_tool: Which AI tool was used (claude/codex)
            session_committed: Whether the session made a git commit
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if exit_code > 128:
            signal_num = exit_code - 128
            error_msg = f"{ai_tool} killed by signal {signal_num}"
            is_real_crash = True
        elif exit_code == 124:
            error_msg = f"{ai_tool} timed out"
            is_real_crash = not session_committed  # Timeout after commit = not a crash
        elif exit_code == 125:
            error_msg = f"{ai_tool} killed due to silence (stale connection)"
            is_real_crash = False  # Expected after sleep/resume, will restart cleanly
        else:
            error_msg = f"{ai_tool} exited with code {exit_code}"
            # Exit code 1 after successful commit = likely EPIPE or graceful exit
            is_real_crash = not session_committed

        if session_committed and not is_real_crash:
            # Session completed work - this is not a crash
            print()
            print(
                f"Note: {ai_tool} exited with code {exit_code} but session committed successfully"
            )
            print()
            return  # Don't log to crashes.log

        with open(self.crash_log, "a") as f:
            f.write(f"[{timestamp}] Iteration {self.iteration}: {error_msg}\n")

        print()
        print(f"*** {self.mode.upper()} CRASH: {error_msg}")
        print(f"    Log: {self.crash_log}")
        print()

    def run(self):
        """Main loop."""
        self.setup()

        try:
            while self.running:
                # Check for graceful shutdown request (touch STOP or STOP_<ROLE>)
                # Don't delete STOP - let all sessions see it, user runs `rm STOP` when done
                stop_file = self._check_stop_file()
                if stop_file:
                    print(f"\n*** {stop_file} file detected - shutting down gracefully ***")
                    self._consume_stop_file(stop_file)
                    self.running = False
                    break

                # Reload config each iteration (allows live tuning)
                self.config, self._prompt_template = load_role_config(self.mode)

                # Run pulse health check if due (updates .flags/ and metrics/)
                self.run_pulse()

                # Run main iteration
                exit_code, start_time, ai_tool, session_id, selected_codex_model = self.run_iteration(audit_round=0)

                if not self.running:
                    break

                # Determine delay and check for crashes
                session_committed = self.check_session_success(start_time)

                # Extract working issues from last commit for audit context
                self._working_issues = []
                if session_committed:
                    last_commit = self._get_last_commit_message()
                    if last_commit:
                        self._working_issues = extract_issue_numbers(last_commit)

                # If session didn't commit, check for uncommitted changes
                if not session_committed:
                    wip_committed = self.commit_uncommitted_changes()
                    if wip_committed:
                        session_committed = True  # WIP commit counts as success

                # Run follow-up audits only when issues have 'do-audit' label
                # Label-driven: AI adds do-audit when claiming done, audit verifies
                auto_audit = self.config.get("auto_audit", True)
                audit_max_rounds = self.config.get("audit_max_rounds", 5)
                audit_min_issues = self.config.get("audit_min_issues", 5)

                # Check for issues needing audit (label-driven gate)
                do_audit_issues = get_do_audit_issues() if auto_audit else []

                should_audit = (
                    auto_audit
                    and len(do_audit_issues) > 0
                    and exit_code == 0
                    and self.running
                    and not self._check_stop_file()  # Check STOP/STOP_<ROLE> before audit
                )
                # Codex: max 1 audit round (no session resume, but context injection)
                if ai_tool == "codex":
                    audit_max_rounds = min(audit_max_rounds, 1)
                # Track audit results for later checks
                audit_committed = False
                audit_start_time = 0.0

                if should_audit:
                    # Run up to audit_max_rounds follow-up audits
                    # Continue only if the previous round found and fixed issues
                    # Track start times for [INCOMPLETE] checking
                    audit_start_times: list[float] = []

                    # Show which issues are being audited
                    audit_issue_nums = [i["number"] for i in do_audit_issues]
                    audit_issue_list = ", ".join(f"#{n}" for n in audit_issue_nums[:3])
                    if len(audit_issue_nums) > 3:
                        audit_issue_list += f" (+{len(audit_issue_nums) - 3} more)"

                    for audit_round in range(1, audit_max_rounds + 1):
                        # Re-check for do-audit labels (may have been removed by Manager)
                        current_do_audit = get_do_audit_issues()
                        if not current_do_audit:
                            print(f"\n✓ {self.iteration}.{audit_round}: do-audit labels removed (reviewed)")
                            break

                        # Concise audit header showing issues being audited
                        print()
                        print(f"--- {self.iteration}.{audit_round}: Audit {audit_issue_list} - find {audit_min_issues}+ issues or [DONE] ---")

                        # Run audit iteration with same tool, model, and context continuity
                        # - Claude: uses session resume (--resume)
                        # - Codex: uses last commit injection (no session resume support)
                        audit_exit_code, audit_start_time, audit_ai_tool, session_id, _ = self.run_iteration(
                            audit_round=audit_round,
                            resume_session_id=session_id,
                            force_ai_tool=ai_tool,
                            force_codex_model=selected_codex_model,
                        )
                        audit_start_times.append(audit_start_time)

                        if not self.running:
                            break

                        # Check STOP/STOP_<ROLE> file between audit rounds
                        stop_file = self._check_stop_file()
                        if stop_file:
                            print(f"\n*** {stop_file} file detected - stopping audits ***")
                            self._consume_stop_file(stop_file)
                            self.running = False
                            break

                        # Check if this audit round committed
                        round_committed = self.check_session_success(audit_start_time)
                        if round_committed:
                            audit_committed = True
                            # Check for [DONE] marker - AI signals no more issues
                            if self.check_done_commit(audit_start_time):
                                print(f"✓ {self.iteration}.{audit_round}: [DONE]")
                                # Transition do-audit → needs-review for audited issues
                                for issue_num in audit_issue_nums:
                                    if transition_audit_to_review(issue_num):
                                        print(f"  → #{issue_num}: do-audit → needs-review")
                                break
                            print(f"✓ {self.iteration}.{audit_round}: issues fixed")
                        else:
                            # No commit - session ended without completing
                            print(f"✓ {self.iteration}.{audit_round}: no commit")
                            break

                        # Use audit results for crash detection
                        if audit_exit_code != 0:
                            self.log_crash(audit_exit_code, audit_ai_tool, round_committed)

                    # Update audit_start_time to check ALL audit rounds for [INCOMPLETE]
                    # We need the earliest start time to catch any [INCOMPLETE] commits
                    if audit_start_times:
                        audit_start_time = min(audit_start_times)

                if exit_code != 0:
                    self.log_crash(exit_code, ai_tool, session_committed)
                    # Use shorter delay if session committed successfully
                    delay = (
                        self.config["restart_delay"]
                        if session_committed
                        else self.config["error_delay"]
                    )
                else:
                    delay = self.config["restart_delay"]

                # Check for [INCOMPLETE] marker - continue immediately
                # Check both main iteration and audit iteration (if it ran and committed)
                main_incomplete = session_committed and self.check_incomplete_commit(start_time)
                audit_incomplete = audit_committed and self.check_incomplete_commit(audit_start_time)
                if main_incomplete or audit_incomplete:
                    print()
                    print("📝 Session marked [INCOMPLETE] - continuing immediately")
                    delay = 0

                # Convert unchecked checkboxes to new issues (Worker only)
                if self.mode == "worker" and session_committed:
                    self.process_unchecked_checkboxes()

                # Increment and persist iteration
                self.iteration += 1
                self.iteration_file.write_text(str(self.iteration))

                # Wait before next iteration (skip if no delay)
                if delay > 0:
                    self.write_status("waiting", extra={"next_iteration_in": delay})
                    if delay > 60:
                        print(f"Next iteration in {delay // 60} minutes...")
                    else:
                        print(f"Next iteration in {delay} seconds...")

                    # Interruptible sleep (responds to signals and STOP/STOP_<ROLE> file)
                    for _ in range(delay):
                        if not self.running:
                            break
                        # Check for STOP/STOP_<ROLE> file during wait
                        stop_file = self._check_stop_file()
                        if stop_file:
                            print(f"\n*** {stop_file} file detected - shutting down gracefully ***")
                            self._consume_stop_file(stop_file)
                            self.running = False
                            break
                        time.sleep(1)

        finally:
            self.cleanup()


def show_prompt(mode: str) -> None:
    """Display the full system prompt with source annotations.

    Shows how the prompt is assembled from:
    - .claude/roles/shared.md (shared template)
    - .claude/roles/{mode}.md (role-specific template)
    - Injected content (git_log, gh_issues, rotation_focus, etc.)
    """
    print("=" * 70)
    print(f"SYSTEM PROMPT FOR: {mode.upper()}")
    print("=" * 70)
    print()

    # Load role config
    config, prompt_template = load_role_config(mode)

    # Show source files
    print("### SOURCE FILES ###")
    print()
    shared_path = ROLES_DIR / "shared.md"
    role_path = ROLES_DIR / f"{mode}.md"
    print(f"1. {shared_path}")
    print(f"2. {role_path}")
    if Path(".looper_config.json").exists():
        print("3. .looper_config.json (config overrides)")
    print()

    # Show config
    print("### PARSED CONFIG ###")
    print()
    for key, value in config.items():
        if key != "phase_data":  # Skip verbose phase data
            print(f"  {key}: {value}")
    print()

    # Show phase data if present
    phase_data = config.get("phase_data", {})
    if phase_data:
        print("### ROTATION PHASES ###")
        print()
        for phase_name, data in phase_data.items():
            print(f"  {phase_name}:")
            print(f"    weight: {data.get('weight', 1)}")
            print(f"    min_findings: {data.get('min_findings', 3)}")
            goals = data.get("goals", [])
            if goals:
                print(f"    goals: ({len(goals)} items)")
        print()

    # Run session start commands to get injection content
    print("### INJECTED CONTENT ###")
    print()
    session_results = run_session_start_commands(mode)

    for key, value in session_results.items():
        if value:
            lines = value.split("\n")
            preview = lines[0][:60] + "..." if len(lines[0]) > 60 else lines[0]
            if len(lines) > 1:
                preview += f" (+{len(lines)-1} more lines)"
            print(f"  {key}: {preview}")
        else:
            print(f"  {key}: (empty)")
    print()

    # Calculate rotation focus
    rotation_type = config.get("rotation_type", "")
    rotation_phases = config.get("rotation_phases", [])
    freeform_frequency = config.get("freeform_frequency", 3)
    force_phase = config.get("force_phase")
    starvation_hours = config.get("starvation_hours", 24)

    # Use iteration 1 for demo
    rotation_focus, selected_phase = get_rotation_focus(
        iteration=1,
        rotation_type=rotation_type,
        phases=rotation_phases,
        phase_data=phase_data,
        role=mode,
        freeform_frequency=freeform_frequency,
        force_phase=force_phase,
        starvation_hours=starvation_hours,
    )

    if rotation_focus:
        print("### ROTATION FOCUS (iteration 1) ###")
        print()
        print(f"  Selected phase: {selected_phase or 'freeform'}")
        print()

    # Build replacements
    replacements = {
        "git_log": session_results.get("git_log", "(unavailable)"),
        "gh_issues": session_results.get("gh_issues", "(unavailable)"),
        "last_directive": session_results.get("last_directive", ""),
        "other_feedback": session_results.get("other_feedback", ""),
        "role_mentions": session_results.get("role_mentions", ""),
        "system_status": session_results.get("system_status", ""),
        "audit_data": session_results.get("audit_data", ""),
        "rotation_focus": rotation_focus,
    }

    # Show template before injection
    print("### TEMPLATE (before injection) ###")
    print()
    print("--- shared.md ---")
    shared_content = (ROLES_DIR / "shared.md").read_text()
    _, shared_body = parse_frontmatter(shared_content)
    # Show first 20 lines
    for line in shared_body.strip().split("\n")[:20]:
        print(f"  {line}")
    if shared_body.count("\n") > 20:
        print(f"  ... ({shared_body.count(chr(10)) - 20} more lines)")
    print()

    print(f"--- {mode}.md ---")
    role_content = (ROLES_DIR / f"{mode}.md").read_text()
    _, role_body = parse_frontmatter(role_content)
    # Show first 30 lines
    for line in role_body.strip().split("\n")[:30]:
        print(f"  {line}")
    if role_body.count("\n") > 30:
        print(f"  ... ({role_body.count(chr(10)) - 30} more lines)")
    print()

    # Apply injections and show final prompt
    final_prompt = inject_content(prompt_template, replacements)

    print("### FINAL PROMPT (after injection) ###")
    print()
    print("-" * 70)
    print(final_prompt)
    print("-" * 70)
    print()

    # Summary
    print("### SUMMARY ###")
    print()
    print(f"  Total prompt length: {len(final_prompt)} chars")
    print(f"  Total prompt lines: {final_prompt.count(chr(10)) + 1}")
    print(f"  Injection markers replaced: {len([k for k, v in replacements.items() if v])}")
    print()


def main():
    # Check for --show-prompt flag
    show_prompt_mode = "--show-prompt" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--show-prompt"]

    # Default to worker mode if no argument provided
    if len(args) == 0:
        mode = "worker"
    elif len(args) == 1 and args[0] in ("worker", "prover", "researcher", "manager"):
        mode = args[0]
    else:
        print("Usage: ./looper.py [worker|prover|researcher|manager] [--show-prompt]")
        print()
        print("  worker     - Fast autonomous loop (no delay) [default]")
        print("  prover     - Proof loop (15-min intervals)")
        print("  researcher - Research loop (10-min intervals)")
        print("  manager    - Audit loop (15-min intervals)")
        print()
        print("Options:")
        print("  --show-prompt  Display the full system prompt and exit")
        print()
        print("Stop: touch STOP (all roles) or STOP_WORKER/STOP_MANAGER/etc. (per-role)")
        print()
        print("Spawn all 4 loops: ./ai_template_scripts/spawn_all.sh")
        sys.exit(1)

    # Handle --show-prompt
    if show_prompt_mode:
        show_prompt(mode)
        sys.exit(0)

    # Require interactive terminal - running in background causes orphaned processes
    if not sys.stdin.isatty():
        print("ERROR: looper.py requires an interactive terminal")
        print("       Use spawn_session.sh to create a proper iTerm2 tab")
        sys.exit(1)

    LoopRunner(mode).run()


if __name__ == "__main__":
    main()

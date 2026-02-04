# ai_template_scripts

Utility scripts for the AI template system.

## Imports

Python modules (`.py` files) in this directory are importable as a package.
Shell scripts (`.sh` files) are executable utilities only, not importable.

```python
from ai_template_scripts.gh_issues import list_dependencies
from ai_template_scripts.pulse import get_issue_counts
```

## Directory Structure

| Directory | Purpose | Syncs |
|-----------|---------|-------|
| `ai_template_scripts/` | Shared scripts synced to all repos | Yes |
| `scripts/` | ai_template-only scripts | No |

Put new scripts in `ai_template_scripts/` if they should be available to all repos.
Use `scripts/` only when a script is specific to this repository.

## API Stability

See **[API.md](API.md)** for the complete API surface definition, including:
- **Stable CLI scripts** - Command-line interfaces with compatibility guarantees
- **Internal modules** - Import-only modules that may change without notice
- **Breaking change policy** - How changes are communicated

For V1 readiness, scripts marked "Stable" in API.md maintain backward-compatible CLIs.

## Wrapper vs Package Pattern

Some modules exist as both a **wrapper script** (`foo.py`) and a **package** (`foo/`):

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `pulse.py` | CLI wrapper script | `./ai_template_scripts/pulse.py --help` |
| `pulse/` | Package with internal modules | `from ai_template_scripts.pulse import get_issue_counts` |
| `json_to_text.py` | CLI wrapper (stdin → formatted output) | `cat log.jsonl \| ./ai_template_scripts/json_to_text.py` |
| `json_to_text/` | Package with formatters | `from ai_template_scripts.json_to_text import format_entry` |

**Guidelines:**
- **CLI usage**: Run the wrapper script directly (`./script.py` or `python3 script.py`)
- **Programmatic imports**: Import from the package (`from ai_template_scripts.package import func`)
- Wrappers delegate to packages internally - they're entry points, not separate implementations

The wrapper exists for backwards compatibility and CLI convenience. The package provides
the actual implementation and public API for programmatic use.

## Scripts

Entries are listed alphabetically by script name (LC_ALL=C / ASCII byte order).

| Script | Purpose | Called By |
|--------|---------|-----------|
| `audit_alignment.sh` | Check template alignment in a repo | AI, human |
| `bg_task.py` | Background task management | WORKER AI |
| `bot_token.py` | GitHub App installation token helper | AI, human |
| `bump_git_dep_rev.sh` | Bump git dependency revision in Cargo.toml files | AI, human |
| `cargo_lock_info.py` | Inspect serialized cargo lock status | AI, human |
| `cargo_wrapper/` | Serialize cargo builds org-wide to prevent OOM | `bin/cargo` wrapper |
| `check_claude_version.sh` | Verify Claude Code CLI matches `.claude-version` | Looper, human |
| `check_deps.py` | Check project dependencies and verification tools | MANAGER, init |
| `check_doc_claims.py` | Verify documentation claims match code reality | MANAGER, pulse.py |
| `check_path_order.sh` | Verify ai_template wrappers have PATH precedence | Codex, human |
| `check_regression.py` | Regression checker for benchmark evals | PROVER, human |
| `check_resource_patterns.py` | Detect resource leaks in Python code (open/Popen) | PROVER, health_check |
| `check_stale_docs.py` | Find markdown docs with stale verification headers | MANAGER, human |
| `clean_artifacts.sh` | Safe cleanup of Rust build artifacts (target/) | AI, human |
| `cleanup_old_reports.py` | Remove old ephemeral reports to prevent repo bloat | AI, human |
| `cleanup_tla_states.sh` | Remove TLA+ state files to prevent disk bloat | AI, human |
| `code_stats.py` | Code complexity analysis | MANAGER, human |
| `commit-msg-hook.sh` | Git hook for structured commits | Git |
| `crash_analysis.py` | Crash log analysis, system health | MANAGER, human |
| `design_issue_audit.py` | Audit design docs and issue cross-references | MANAGER, human |
| `doc_health_check.py` | Check CLAUDE.md completeness for AI operations | MANAGER doc_health phase |
| `exclude_patterns.py` | Shared directory exclusion patterns for codebase analysis | Import only |
| `find_ignores.sh` | Find forbidden test ignores for audit | Human, scripts |
| `find_tla_tools.sh` | Locate Java, tla2tools.jar, and timeout binaries | TLA+ scripts |
| `file_stuck_spec.py` | File stuck Kani/TLA+ specs to zani/tla2 | AI, human |
| `fix_lfs_offline.sh` | Configure Git LFS for offline operation | Human |
| `generate_skill_index.py` | Generate .claude/SKILLS.md from skills | AI, human |
| `gh_apps/` | GitHub App authentication for rate limit scaling (package) | `bin/gh` wrapper |
| `gh_atomic_claim.py` | Atomic issue claiming with race protection | AI |
| `gh_discussion.py` | List, get, create, and comment on GitHub discussions | AI, human |
| `gh_graphql.py` | GraphQL helper with variable encoding (internal module) | Other scripts |
| `gh_issue_numbers.py` | Parse issue numbers from gh JSON fields | Import only |
| `gh_issues_mirror.py` | Dump GitHub issues to local markdown for offline search | AI, human |
| `gh_issues.py` | GitHub Issues dependency tracking helper | AI, human |
| `gh_local.py` | Local gh command handler for offline development mode | `bin/gh` wrapper |
| `gh_post.py` | CLI wrapper for gh_post package | `bin/gh` wrapper |
| `gh_post/` | Add AI identity to gh issue create/comment (package) | `gh_post.py`, `bin/gh` |
| `gh_rate_limit/` | GitHub API rate limiting (package) | `gh_wrapper.py` |
| `gh_wrapper.py` | GitHub CLI wrapper with rate limiting | `bin/gh` |
| `health_check/` | Base class for system health checks (package) | Import `HealthCheckBase` |
| `init_from_template.sh` | Set up new project from template | Human (one-time) |
| `init_labels.sh` | Create required GitHub labels | `init_from_template.sh`, `sync_repo.sh` |
| `install_dev_tools.sh` | Install dev tools (lizard, radon, etc) | Human (one-time) |
| `install_hooks.sh` | Install pre-commit framework and git hooks | Human (one-time) |
| `install_tla_tools.sh` | Install TLA+ tools (TLC) and configure env | Human, `check_deps.py --fix` |
| `integration_audit.py` | Detect orphan modules not reachable from entry points | MANAGER, human |
| `json_to_text.py` | Format Claude/Codex JSON output for terminal | `looper.py` |
| `labels.py` | Shared label constants for GitHub issue management | Import only |
| `lint_applescript.sh` | Validate AppleScript syntax in shell scripts | AI, human |
| `local_issue_store.py` | File-based issue storage for offline development mode | `gh_local.py` |
| `log_scrubber.py` | Sanitize secrets and PII from JSONL logs | Human, scripts |
| `log_test.py` | Log test invocations for MANAGER audit | AI (via test scripts) |
| `markdown_to_issues.py` | Sync markdown roadmap with GitHub Issues | AI, human |
| `memory_watchdog.py` | Memory pressure watchdog for macOS | Looper, human |
| `path_utils.py` | Path utilities with deprecation support | Import only |
| `pip_audit.sh` | Scan requirements.txt for known vulnerabilities | Human, MANAGER |
| `post-commit-hook.sh` | Git hook for post-commit actions | Git |
| `pre-commit-hook.sh` | Git hook for copyright/author validation | Git |
| `pre-push-hook.sh` | Git hook for claim validation before push | Git |
| `provenance_capture.py` | Capture SLSA-style build/test provenance | AI, human |
| `pulse.py` | Metrics collection, threshold flags | [M] or cron |
| `pulse/` | Pulse package implementation (metrics, flags, output) | `pulse.py` |
| `repo_directors.py` | Repo to Director mapping from org_chart.md | `gh_post.py` |
| `result.py` | Result type for functional error handling | Import only |
| `scoped_ruff.sh` | Ruff wrapper that scopes fixes based on role | AI, human |
| `shared_logging.py` | Shared logging utilities (debug, stderr, rotation) | Import only |
| `spawn_all.sh` | Spawn all 4 AI loops (worker, prover, researcher, manager) | Human |
| `spawn_session.sh` | Spawn AI session (worker/prover/researcher/manager) in iTerm2 | `spawn_all.sh`, Human |
| `subprocess_utils.py` | Common subprocess helpers: `CmdResult`, `run_cmd()`, canonical repo functions | Other scripts |
| `sync_all.sh` | Batch sync to all repos | Human, scripts |
| `sync_check.sh` | Check template drift across repos | Human, scripts |
| `sync_local_issues.py` | Sync local issues to GitHub | Human, AI |
| `sync_repo.sh` | Sync template files to target repo | Human, scripts |
| `test_utils.py` | Test subprocess utilities with default timeouts | Tests |
| `update_claude_user_settings.py` | Update ~/.claude/settings.json with recommended env vars | `init_from_template.sh`, Human |
| `update_line_counts.py` | Update CLAUDE.md "Files Loaded Per Session" table line counts | Human, scripts |
| `url_sanitizer.py` | Git URL sanitizer - removes credentials for safe logging | Scripts, import |
| `validate_claim.py` | Validate benchmark claims in commit messages | Pre-push hook, AI |
| `verify_incremental.sh` | Run incremental verification across tools (Kani, TLA+) | PROVER, looper |
| `version.py` | CLI version string helper | Import only |

## Details

### audit_alignment.sh
Checks if a repository is properly aligned with the ai_template. Detects missing required files, obsolete files that should be deleted, and forbidden files (like CI workflows). Returns exit code 1 if critical issues found.

Usage:
```bash
./ai_template_scripts/audit_alignment.sh    # Run in any repo
```

### find_ignores.sh
Scans for forbidden test ignores (per ai_template.md rule "Test ignores FORBIDDEN"). Finds `#[ignore]` in Rust, `@skip/@pytest.mark.skip` in Python, `.skip()` in JS/TS.

Usage:
```bash
./ai_template_scripts/find_ignores.sh          # Check current directory
./ai_template_scripts/find_ignores.sh ~/z4     # Check specific directory
```

### find_tla_tools.sh
Shared TLA+ tooling discovery helper for Java, `tla2tools.jar`, and timeout binaries.

Usage:
```bash
./ai_template_scripts/find_tla_tools.sh --java
./ai_template_scripts/find_tla_tools.sh --jar
./ai_template_scripts/find_tla_tools.sh --timeout
```

### json_to_text.py
Converts Claude/Codex streaming JSON output to readable terminal text. Critical for `looper.py` - all AI output is piped through this.

### code_stats.py
Analyzes cyclomatic/cognitive complexity across multiple languages (Python, Rust, Go, C++, etc). Uses best-in-class tools per language (radon for Python, gocyclo for Go, etc).

### crash_analysis.py
Parses `worker_logs/crashes.log` to calculate failure rates and system health status. Used by MANAGER for auditing.

### bg_task.py
Manages long-running background tasks that survive worker iteration timeouts. Stores state in `.background_tasks/`.

### bot_token.py
Fetches GitHub App installation tokens for the AI fleet using local app credentials.

Usage:
```bash
./ai_template_scripts/bot_token.py --json
```

### bump_git_dep_rev.sh
Bumps git dependency revision in Cargo.toml files to the latest commit (or a specified revision). Useful for keeping pinned git dependencies up to date across multi-repo projects.

Usage:
```bash
./ai_template_scripts/bump_git_dep_rev.sh https://github.com/ayates_dbx/z4           # Bump to HEAD
./ai_template_scripts/bump_git_dep_rev.sh https://github.com/ayates_dbx/z4 cdfa08fb  # Bump to specific rev
./ai_template_scripts/bump_git_dep_rev.sh --dry-run https://github.com/ayates_dbx/z4 # Preview changes
```

After running, follow up with:
1. `cargo check` to verify the update compiles
2. `cargo update` to refresh Cargo.lock
3. Commit the changes

### cargo_lock_info.py
Inspects the serialized cargo lock holder and reports whether the lock appears stale.

Usage:
```bash
./ai_template_scripts/cargo_lock_info.py
./ai_template_scripts/cargo_lock_info.py --kind test
```

### cargo_wrapper/
Python package that serializes all cargo builds org-wide to prevent OOM, deadlocks, and orphaned rustc processes. Called by the `bin/cargo` wrapper via `python3 -m ai_template_scripts.cargo_wrapper` for build/test/check/run/clippy/doc/bench commands. Build and test locks are separate by default; set max concurrency to 1 to share a single lock.

Timeouts can be configured per repo via `cargo_wrapper.toml` (or `.cargo_wrapper.toml`):
```toml
[timeouts]
build_timeout_sec = 3600   # Alias: build_sec
test_timeout_sec = 600     # Alias: test_sec
kani_timeout_sec = 1800    # Alias: kani_sec
```

Both forms (`*_timeout_sec` and `*_sec`) are accepted. The verbose form explicitly states units.

Limit concurrency (build/test lock sharing) via the optional `[limits]` table:
```toml
[limits]
max_concurrent_cargo = 1   # 1 = shared lock, 2 = build/test split (default)
```

### check_claude_version.sh
Verifies the installed Claude Code CLI matches the repo's `.claude-version` pin. Used by looper to warn about mismatches; `--strict` exits non-zero.

Usage:
```bash
./ai_template_scripts/check_claude_version.sh
./ai_template_scripts/check_claude_version.sh --strict
```

### check_path_order.sh
Verifies that ai_template wrappers (gh, cargo) have PATH precedence over system binaries like Homebrew. This is critical for Codex and other AI tools that spawn their own shells. (#1860)

Usage:
```bash
./ai_template_scripts/check_path_order.sh          # Show results
./ai_template_scripts/check_path_order.sh --quiet  # Exit code only (for scripts)
./ai_template_scripts/check_path_order.sh --help   # Show usage
```

Exit codes: 0 = correct PATH order, 1 = wrappers shadowed by system binaries.

### check_deps.py
Detects verification tools needed by the project and checks if they're installed.

Usage:
```bash
./ai_template_scripts/check_deps.py          # Check all deps, exit 1 if missing
./ai_template_scripts/check_deps.py --quiet  # Only show missing deps
./ai_template_scripts/check_deps.py --fix    # Attempt to install missing deps
```

### check_resource_patterns.py
Detects common resource leak anti-patterns in Python files: open() without context manager, Popen() without cleanup, file handles stored as attributes, and socket/connection objects without cleanup.

Usage:
```bash
./ai_template_scripts/check_resource_patterns.py src/         # Check directory
./ai_template_scripts/check_resource_patterns.py --json src/  # JSON output
```

### check_stale_docs.py
Checks age of markdown docs in `designs/`, `reports/`, `postmortems/` to find stale documentation.

Usage:
```bash
./ai_template_scripts/check_stale_docs.py                    # Find stale (>7 days) or missing
./ai_template_scripts/check_stale_docs.py --max-age-days=30  # Custom staleness threshold
./ai_template_scripts/check_stale_docs.py --missing-only     # Only report missing headers
./ai_template_scripts/check_stale_docs.py --json             # Machine-readable output
./ai_template_scripts/check_stale_docs.py --path=/other/repo # Scan a different directory
```

### check_doc_claims.py
Verifies machine-checkable documentation claims in YAML frontmatter. Parses claims from CLAUDE.md and VISION.md, checks if declared code paths exist and code markers are present in the codebase. Used by pulse.py to set the `doc_claim_drift` flag.

Usage:
```bash
./ai_template_scripts/check_doc_claims.py                    # Check all docs, human output
./ai_template_scripts/check_doc_claims.py --json             # Machine-readable output
./ai_template_scripts/check_doc_claims.py --path=/other/repo # Check specific repo
```

**Claim format** in YAML frontmatter at top of doc:
```yaml
---
claims:
  - type: backend
    name: z4
    code_path: src/backends/z4/
    status: active
  - type: feature
    name: BigInt
    code_marker: "// VISION: BigInt"
---
```

**Claim types:**
- `code_path` - Verifies directory/file exists
- `code_marker` - Greps for marker string in source files
- `backend` - Alias for code_path (semantic clarity)
- `feature` - Alias for code_marker (semantic clarity)

### design_issue_audit.py
Audits bidirectional links between design documents and GitHub issues. Finds orphan designs (no issue reference), unlinked designs (issue doesn't link back), and missing designs (referenced but don't exist).

Usage:
```bash
./ai_template_scripts/design_issue_audit.py           # Markdown table output
./ai_template_scripts/design_issue_audit.py --json    # JSON output
./ai_template_scripts/design_issue_audit.py --fix     # Output gh commands to fix
./ai_template_scripts/design_issue_audit.py --repo /path/to/repo  # Check other repo
```

Exit codes: 0 = all designs properly linked, 1 = issues found.

### doc_health_check.py
Checks CLAUDE.md documentation completeness for AI operations. Designed for Manager rotation phase `doc_health`. Verifies CLI entry points, build commands, environment variables, and scripts are documented.

Usage:
```bash
./ai_template_scripts/doc_health_check.py              # Human output
./ai_template_scripts/doc_health_check.py --json       # Machine-readable
./ai_template_scripts/doc_health_check.py --path=/repo # Check specific repo
```

**Checks performed:**
- `cli_documented` - Binaries in Cargo.toml or pyproject.toml scripts are in CLAUDE.md
- `build_commands` - Cargo/pip/npm commands documented for project type
- `env_vars` - Project-specific environment variables documented
- `recent_errors` - No "not found" errors in recent worker logs
- `key_scripts` - Scripts in `scripts/` directory documented

**Exit codes:**
- `0` = all checks passed
- `1` = one or more checks failed

### cleanup_old_reports.py
Removes old ephemeral reports from `reports/` to prevent repo bloat. Per CLAUDE.md, reports are ephemeral but accumulate indefinitely. This script removes files older than a configurable retention period (#1695).

Usage:
```bash
./ai_template_scripts/cleanup_old_reports.py                # Dry run - show what would be deleted
./ai_template_scripts/cleanup_old_reports.py --delete       # Actually delete old reports
./ai_template_scripts/cleanup_old_reports.py --days 14      # 14-day retention (default: 30)
```

Environment variable `REPORTS_MAX_AGE_DAYS` can override the default 30-day threshold.

**Integration:** `pulse.py` sets the `disk_bloat` flag when `reports/` exceeds 50MB (configurable via `reports_dir_size_mb` threshold). The flag explanation suggests running this script.

### cleanup_tla_states.sh
Removes old TLA+ model checking state files to prevent disk bloat. TLA+ model checking creates large `.fp` files (up to 1GB each) in `states/` directories that grow unbounded and have caused disk issues (#1551, #668).

Usage:
```bash
./ai_template_scripts/cleanup_tla_states.sh              # Dry run - show what would be deleted
./ai_template_scripts/cleanup_tla_states.sh --delete     # Actually delete old states
./ai_template_scripts/cleanup_tla_states.sh --max-age 3  # Delete states older than 3 days
```

Environment variable `TLA_STATES_MAX_AGE_DAYS` can override the default 7-day threshold.

### gh_post.py
Wrapper for gh CLI that adds AI identity markers to GitHub issues and comments. Called automatically by the `bin/gh` wrapper for `issue create`, `comment`, `edit`, and `close` commands.

**Identity Injection:**
- Adds `**FROM:** project [ROLE]N` header at start of body
- Adds signature line at end: `project | ROLE #N | session | commit | timestamp`
- Fixes title prefix to `[project]`

**Role Enforcement:**
- Only MANAGER/USER can close issues
- USER-only labels (`urgent`, `P0`) blocked for other roles
- USER-protected labels can't be removed by non-USER

**GitHub Project Integration:**
- Automatically adds new issues to GitHub Project #1
- Sets Status=Todo, Director based on repo, Type=Task

**Offline-First Operations:**
- When rate-limited, queues operations to `~/.ait_gh_cache/change_log.json`
- Operations replayed when quota available

**Special Commands:**
```bash
gh issue cleanup-closed           # Clean workflow labels from auto-closed issues
gh issue cleanup-closed --dry-run # Preview without modifying
```

**Environment Variables:**
- `AI_ROLE` - Current role (USER, WORKER, PROVER, RESEARCHER, MANAGER)
- `AI_PROJECT` - Project name (derived from git remote if unset)
- `AI_ITERATION` - Iteration number
- `AI_SESSION` - Session UUID

### gh_discussion.py
List, get, create, and comment on GitHub discussions with AI identity markers. Used for posting to Dash News (ayates_dbx/dashnews).

Usage:
```bash
# List recent discussions
./ai_template_scripts/gh_discussion.py list
./ai_template_scripts/gh_discussion.py list --limit 5
./ai_template_scripts/gh_discussion.py list --json  # For programmatic access

# Get a specific discussion by number
./ai_template_scripts/gh_discussion.py get 211
./ai_template_scripts/gh_discussion.py get 211 --json
./ai_template_scripts/gh_discussion.py get 211 --repo ayates_dbx/other

# Create a discussion
./ai_template_scripts/gh_discussion.py create --title "Title" --body "Body"
./ai_template_scripts/gh_discussion.py create --title "Title" --body "Body" --category "Q&A"

# Comment on existing discussion
./ai_template_scripts/gh_discussion.py comment --number 42 --body "Comment"
```
Notes:
- Requires `read:discussion` to list/get discussions.
- Requires `write:discussion` to create/comment.
- Add scopes with `gh auth refresh -s read:discussion -s write:discussion`.

### generate_skill_index.py
Builds `.claude/SKILLS.md` from the skills in `.claude/commands/`.

Usage:
```bash
./ai_template_scripts/generate_skill_index.py
```

### labels.py
Shared label constants for GitHub issue management. Single source of truth for in-progress, ownership, and workflow labels used across ai_template_scripts and looper.

**Exports:**
- `IN_PROGRESS_PREFIX` - Base prefix ("in-progress")
- `IN_PROGRESS_ALL_LABELS` - All in-progress variants (in-progress, in-progress-W1..W5, etc.)
- `OWNERSHIP_WORKER_LABELS` - Worker ownership (W1-W5)
- `OWNERSHIP_ALL_LABELS` - All ownership labels
- `WORKFLOW_LABELS` - Labels cleared on issue close

Usage:
```python
from ai_template_scripts.labels import (
    IN_PROGRESS_ALL_LABELS,
    OWNERSHIP_WORKER_LABELS,
    WORKFLOW_LABELS,
)

# Check if label is a workflow label
if label in WORKFLOW_LABELS:
    # Clear on close
    pass
```

### integration_audit.py
Detects orphan modules not reachable from entry points. Helps find code that's built but never executed, dead CLI flags, and modules that produce no output.

Usage:
```bash
./ai_template_scripts/integration_audit.py            # Audit current directory
./ai_template_scripts/integration_audit.py --json    # Output as JSON
./ai_template_scripts/integration_audit.py ~/project # Audit specific repo
./ai_template_scripts/integration_audit.py --ignore=tests  # Ignore tests directory
```

**What it detects:**
- Orphan Python modules (not imported from any entry point)
- Dead CLI flags (--skip-* variants that may indicate unused features)

**Entry points detected:**
- `scripts/*.py` (project scripts)
- `**/__main__.py` (module entry points)
- `**/cli.py` (CLI entry points)
- Top-level scripts with `if __name__ == "__main__"`

### gh_atomic_claim.py
Atomic issue claiming library for race condition prevention. Used internally by
`gh_post.py` and the commit-msg hook to ensure only one worker claims an issue
when multiple workers attempt simultaneously.

Protocol:
1. Acquire local file lock (same-machine coordination)
2. Post comment with UUID marker (e.g., `[claim:abc12345]`)
3. Wait for GitHub API propagation
4. Verify our claim is first (by comment timestamp)
5. Only add `in-progress` label if verification passes

**Not a CLI tool** - imported by `gh_post.py`. The claiming is triggered
automatically when commit messages contain `Claims #N`.

Design: `designs/2026-01-30-atomic-issue-claiming.md`

### gh_issues.py
Tracks issue dependencies using `Blocked: #N` syntax in issue bodies. Note: GitHub has no native dependency API, so the `add`/`remove` commands are disabled.

Usage:
```bash
./ai_template_scripts/gh_issues.py dep list 55  # List dependencies for issue #55
```

### gh_issues_mirror.py
Exports GitHub issues to local markdown for offline search. Default output is
`.issues/issues-all.md`, refreshed if older than 24 hours.

Usage:
```bash
./ai_template_scripts/gh_issues_mirror.py
./ai_template_scripts/gh_issues_mirror.py --state open
rg -n "LoopRunner" .issues/issues-all.md
```

### gh_rate_limit/
Package for GitHub API rate limiting. Used by `gh_wrapper.py`. Provides transparent caching and rate limit checking to prevent hitting GitHub API limits.

**Package structure** (from #1689 decomposition):
- `__init__.py` - Re-exports, singleton accessors
- `rate_limiter.py` - RateLimiter facade composing all components
- `repo_context.py` - RepoContext for repo resolution and gh binary location
- `rate_state.py` - RateState for rate limit tracking and quota management
- `limiter.py` - RateLimitInfo, UsageStats, threshold logic
- `cache.py` - TtlCache, TTL constants, stale warning formatting
- `historical.py` - HistoricalCache for persistent issue storage
- `rest_fallback.py` - IssueRestFallback, GraphQL→REST conversion
- `serialize.py` - SerializedFetcher, lock file management
- `changelog.py` - Change, ChangeLog for offline queueing
- `batch.py` - Batch issue fetch helpers

### gh_wrapper.py
Transparent gh wrapper that routes commands through rate limiting and caching. Called by `bin/gh` for non-issue-write commands (`issue list`, `issue view`, `repo view`, `api`, etc.).

**How it works:**
- Delegates to `gh_rate_limit.RateLimiter` for actual rate limit management
- Caches read operations with configurable TTL (3 minutes for issue list/view)
- Blocks when quota critical, waits if reset imminent

AIs don't interact with this directly - they just run `gh` normally. The `bin/gh` shim routes commands appropriately:
- Issue create/comment/edit/close → `gh_post.py`
- Everything else → `gh_wrapper.py`

**Related:**
- `gh_rate_limit/` - Rate limiting package
- `bin/gh` - Shell shim that routes to appropriate handler

### gh_graphql.py
Shared GraphQL helper for GitHub API operations. Provides uniform variable type encoding, consistent error handling, and pagination helpers. Used internally by `gh_rate_limit/` package and can be imported directly by other scripts.

Usage:
```python
from ai_template_scripts.gh_graphql import graphql, graphql_batch

# Simple query
result = graphql('query { viewer { login } }')
if result.ok:
    print(result.data['viewer']['login'])

# Batch multiple queries in one API call
result = graphql_batch([
    ('repo1', 'repository(owner:"ayates_dbx", name:"ai_template") { id }'),
    ('repo2', 'repository(owner:"ayates_dbx", name:"leadership") { id }'),
])
```

### health_check/
Package providing shared infrastructure for `system_health_check.py` across repos. Each repo's `scripts/system_health_check.py` imports this base and adds repo-specific checks. See `docs/system_health_check.md` for the full contract.

Usage:
```python
from ai_template_scripts.health_check import (
    HealthCheckBase, CheckResult, Status, create_parser, standard_main
)

class MyHealthCheck(HealthCheckBase):
    def __init__(self):
        super().__init__()
        self.register(self.check_cargo)

    def check_cargo(self) -> CheckResult:
        ...
    check_cargo.name = "cargo_check"
```

### log_scrubber.py
Sanitizes secrets and PII from JSONL log files. Detects and redacts API keys (Anthropic, OpenAI, GitHub), paths, and sensitive data.

Usage:
```bash
./ai_template_scripts/log_scrubber.py worker_logs/worker_iter_1.jsonl  # In-place
./ai_template_scripts/log_scrubber.py --stdout file.jsonl              # To stdout
./ai_template_scripts/log_scrubber.py worker_logs/                     # All in dir
```

### log_test.py
Records test invocations for MANAGER pattern detection: concurrent test runs, duration trends, frequent failures.

Usage:
```bash
./ai_template_scripts/log_test.py run "cargo test -p solver"  # Wrap and log
./ai_template_scripts/log_test.py report                       # Analyze logs
```

### markdown_to_issues.py
Syncs markdown roadmaps to GitHub Issues (and exports issues to markdown).

Usage:
```bash
./ai_template_scripts/markdown_to_issues.py --export
```

### path_utils.py
Path utilities with deprecation support for backward-compatible parameter renaming. Used internally by scripts that renamed their path parameters.

**Exports:**
- `resolve_path_alias(new_name, old_name, positional_value, kwargs, func_name)` - Resolve path from positional or keyword arg with deprecation warning

Usage:
```python
from ai_template_scripts.path_utils import resolve_path_alias

def get_info(dir_path: Path | None = None, **kwargs) -> dict:
    resolved = resolve_path_alias("dir_path", "root", dir_path, kwargs, "get_info")
    # Use resolved path...

# All these work:
get_info(Path("."))           # positional
get_info(dir_path=Path("."))  # new name (preferred)
get_info(root=Path("."))      # deprecated (emits warning)
```

### memory_watchdog.py
Monitors macOS memory pressure and kills runaway AI-spawned processes before kernel panic. Designed to run alongside looper.py or as a standalone daemon.

Usage:
```bash
# Run as daemon (monitors continuously)
./ai_template_scripts/memory_watchdog.py --daemon

# Single check (for cron or testing)
./ai_template_scripts/memory_watchdog.py --check

# Check with custom threshold
./ai_template_scripts/memory_watchdog.py --check --threshold warn

# List killable processes
./ai_template_scripts/memory_watchdog.py --list
```

**Features:**
- Uses macOS `memory_pressure` command for accurate pressure detection
- Falls back to `vm_stat` parsing if memory_pressure unavailable
- Only kills processes matching AI patterns (z4, cbmc, goto-*, rustc, kani)
- Logs all actions to `~/ait_emergency.log`
- Configurable pressure threshold (warn, critical)
- Per-repo kill patterns via `cargo_wrapper.toml`

**Config (optional):** Add to `cargo_wrapper.toml`:
```toml
[memory_watchdog]
kill_patterns = ["z4.*solver", "cbmc", "my-custom-tool"]
```

**Exit codes:**
- `0` - Normal pressure, no action needed
- `1` - Elevated pressure, processes killed
- `2` - Error

**Looper integration:** The watchdog is automatically spawned by `looper.py` on macOS (enabled by default). Only one watchdog runs per machine (uses `~/.ait_memory_watchdog.pid` for deduplication). To disable, set `memory_watchdog_enabled: false` in role config. To change threshold, set `memory_watchdog_threshold: warn` (default: `critical`).

### spawn_session.sh
Spawns a worker, prover, researcher, or manager session in a new iTerm2 tab. Smart argument detection allows flexible ordering.

All roles share the same checkout. Workers coordinate via file tracking (.worker_N_files.json).
Run `git status` to see worker file ownership.

Usage:
```bash
./ai_template_scripts/spawn_session.sh worker              # Worker in current directory
./ai_template_scripts/spawn_session.sh prover ~/z4         # Prover in specific project
./ai_template_scripts/spawn_session.sh ~/z4 researcher     # Swapped args also work
./ai_template_scripts/spawn_session.sh manager ~/z4        # Manager in specific project
./ai_template_scripts/spawn_session.sh --dry-run manager   # Preview without running
./ai_template_scripts/spawn_session.sh --id=1 worker       # Worker 1 (multi-worker mode)
```

### spawn_all.sh
Spawns all 4 AI loops (worker, prover, researcher, manager) in new iTerm2 tabs. This is the recommended way to start a full AI team.

All roles share the same checkout. Workers coordinate via file tracking (.worker_N_files.json).

Usage:
```bash
./ai_template_scripts/spawn_all.sh           # All 4 roles in current project
./ai_template_scripts/spawn_all.sh ~/z4      # All 4 roles in specific project
./ai_template_scripts/spawn_all.sh --workers=2   # 2 workers + other roles
./ai_template_scripts/spawn_all.sh --dry-run # Preview without running
```

### subprocess_utils.py
Common subprocess helpers providing consistent error handling across scripts.

**CmdResult** - Standardized result type:
- `.ok` - True if returncode == 0 AND no exception occurred
- `.returncode` - Process exit code (0 = success)
- `.stdout`, `.stderr` - Output strings
- `.error` - Exception message if any (None on success)

**run_cmd(cmd, timeout=30, cwd=None)** - Run command with timeout and consistent error handling. Returns `CmdResult`.

**run_cmd_with_retry(cmd, timeout=60, retries=2, ...)** - Retry wrapper for transient failures (3 total attempts).

**Canonical repo functions** (per #1267 consolidation):
```python
from pathlib import Path

from ai_template_scripts.subprocess_utils import get_repo_name, get_github_repo

# Get repo name only (e.g., "ai_template")
result = get_repo_name()
name = result.stdout  # Always succeeds with fallback chain

# Get owner/repo (e.g., "ayates_dbx/ai_template")
result = get_github_repo()
if result.ok:
    repo = result.stdout

# Target a specific repo or gh binary
other_repo = get_repo_name(cwd=Path("/tmp/other_repo"))
other_owner_repo = get_github_repo(
    gh_path="/usr/local/bin/gh",
    cwd=Path("/tmp/other_repo"),
)
```

**Other callers should delegate** to these functions rather than reimplementing git remote parsing. See #1267 for rationale.
Note: `get_github_repo()` requires an authenticated gh CLI and supports a custom
`gh_path` for non-standard installs.

### install_dev_tools.sh
One-time setup script that installs development tools needed by other scripts (lizard, radon, gocyclo, etc) and TLA+ tools (TLC).

### install_tla_tools.sh
Installs TLA+ tools (TLC), sets `TLA2TOOLS_JAR`, and adds a `tlc` wrapper to `~/.local/bin`.

Usage:
```bash
./ai_template_scripts/install_tla_tools.sh
```

### init_from_template.sh
Run once after copying this template to a new project. Sets up GitHub labels, installs git hooks, removes template-specific files.

### commit-msg-hook.sh
Git commit-msg hook that auto-adds iteration numbers `[W]N` and validates issue references. Installed to `.git/hooks/commit-msg`.

**Manual testing with `pre-commit try-repo`:**
The commit-msg hook requires the message filename as an argument. When testing locally:
```bash
pre-commit try-repo . --hook-stage commit-msg --commit-msg-filename .git/COMMIT_EDITMSG
```

### pre-commit-hook.sh
Git pre-commit hook that validates:
- Copyright headers in source files (warns if missing)
- Author fields in Cargo.toml/pyproject.toml (warns if missing/wrong)
- Test ignores (blocks commits with `#[ignore]`, `@skip`, `.skip()`)
- Project-specific hooks from `.pre-commit-local.d/` (see below)

Warning mode for headers/authors - doesn't block commits. Test ignores are blocking errors.
Installed to `.git/hooks/pre-commit`.

#### Project-Specific Pre-Commit Hooks

Projects can add custom pre-commit checks without modifying the synced template file.

**Setup:**
1. Create `.pre-commit-local.d/` directory in your project
2. Add executable `.sh` scripts (numbered for order: `01-name.sh`, `02-other.sh`)
3. Scripts should check staged files and exit 0 (pass) or non-zero (fail/block commit)

**Example:** `.pre-commit-local.d/01-kani-check.sh`
```bash
#!/usr/bin/env bash
# Check theory solvers have Kani proofs (z4-specific)
set -euo pipefail

for file in $(git diff --cached --name-only | grep "crates/z4-theories/.*/lib.rs"); do
    if ! grep -q "kani::proof" "$file"; then
        echo "ERROR: $file missing Kani proofs"
        exit 1
    fi
done
exit 0
```

**Key points:**
- `.pre-commit-local.d/` is NOT synced from ai_template (preserved across sync)
- Scripts receive no arguments; use `git diff --cached` to check staged files
- Scripts run after built-in checks, in alphabetical order
- Non-executable scripts are skipped

### pre-push-hook.sh
Git pre-push hook that validates benchmark claims in commit messages. Checks claims like "39/55 CHC" against results files in `evals/results/`. Default: warns on invalid claims. Strict mode (CLAIM_VALIDATION_STRICT=1) blocks push on invalid claims.

See `docs/hooks.md` for detailed configuration options.

### post-commit-hook.sh
Git post-commit hook for post-commit actions. Installed to `.git/hooks/post-commit`.

### install_hooks.sh
Installs the pre-commit framework and sets up git hooks (pre-commit, commit-msg, and pre-push). Auto-detects and installs pre-commit via pip or Homebrew if not present.

Usage:
```bash
./ai_template_scripts/install_hooks.sh    # Install hooks in current repo
```

Requires `.pre-commit-config.yaml` in the repo root (synced from ai_template via `sync_repo.sh`).

### lint_applescript.sh
Validates AppleScript syntax embedded in shell scripts. Finds `osascript -e` blocks and checks them with `osacompile`.

Usage:
```bash
./ai_template_scripts/lint_applescript.sh                    # Check all scripts
./ai_template_scripts/lint_applescript.sh script.sh          # Check specific file
./ai_template_scripts/lint_applescript.sh --fix script.sh    # Show detailed error context
```

### scoped_ruff.sh
Ruff wrapper that scopes fixes based on role to avoid cross-worker edits. Uses `AI_WORKER_ID` or `AI_ROLE=MANAGER` to select files; otherwise passes through to ruff.

Usage:
```bash
./ai_template_scripts/scoped_ruff.sh --fix
AI_WORKER_ID=1 ./ai_template_scripts/scoped_ruff.sh --fix
AI_ROLE=MANAGER ./ai_template_scripts/scoped_ruff.sh --fix
```

### shared_logging.py
Shared logging utilities for ai_template_scripts modules (#2007 consolidation).

**Exports:**
- `debug_log(module, msg, module_env_var=None)` - Print to stderr if debug mode enabled
- `is_debug_mode(module_env_var=None)` - Check if AIT_DEBUG or module-specific flag is set
- `log_stderr(msg)` - Print to stderr with immediate flush
- `now_iso(timespec="seconds")` - Current UTC time as ISO 8601
- `format_json_entry(**fields)` - Format dict as JSON log entry (auto-adds timestamp)
- `append_log(log_path, entry)` - Append entry to log file
- `rotate_log_file(log_path, max_lines)` - Truncate log to last N lines

**Debug mode:** Set `AIT_DEBUG=1` to enable debug logging for all modules, or use
module-specific env vars (e.g., `AIT_GH_DEBUG=1`). See `docs/troubleshooting.md`.

Usage:
```python
from ai_template_scripts.shared_logging import debug_log, log_stderr, now_iso

debug_log("my_module", "processing started")  # Only prints if AIT_DEBUG=1
log_stderr("Status: complete")                 # Always prints
timestamp = now_iso()                          # "2026-02-02T12:00:00+00:00"
```

### pulse.py
Collects metrics and emits runtime flags based on thresholds. Configuration is optional.

Config search order (first found wins):
- `pulse.toml`
- `ai_template_scripts/pulse.toml`
- `.pulse.toml`

Config schema (all keys optional; defaults shown):
```toml
[thresholds]
max_file_lines = 500
max_complexity = 15
max_files_over_limit = 3
stale_issue_days = 7
memory_warning_percent = 80
memory_critical_percent = 90
disk_warning_percent = 80
disk_critical_percent = 90
large_file_size_gb = 1
tests_dir_size_gb = 10
long_running_process_minutes = 120

[large_files]
exclude_patterns = ["tests/", "crates/*/tests/"]

[runtime]
skip_orphaned_tests = true
```

Optional `.pulse_ignore` (repo root) lists pathspec patterns used only for orphan test
detection. `#` starts a comment (inline comments supported), blank lines are ignored,
and `**/` globs match nested directories.

### repo_directors.py
Maps repositories to their director based on `.claude/rules/org_chart.md`.
Used by `gh_post.py` to set the Director field on GitHub Project items.

Usage:
```python
from pathlib import Path

from ai_template_scripts.repo_directors import get_director, load_repo_to_director

director = get_director("z4")  # Returns "MATH"
fallback = get_director("unknown", default="TOOL")
mapping = load_repo_to_director()  # Uses .claude/rules/org_chart.md
custom_mapping = load_repo_to_director(Path("/tmp/org_chart.md"))
```
Note: `load_repo_to_director()` returns a repo -> director dict and supports a
custom org chart path when needed. Only the default org chart path is cached;
custom paths are parsed on each call.

### pip_audit.sh
Scans requirements.txt for known vulnerabilities using pip-audit.

Usage:
```bash
./ai_template_scripts/pip_audit.sh         # Check for vulnerabilities
./ai_template_scripts/pip_audit.sh --fix   # Auto-fix if possible
```

### provenance_capture.py
Captures SLSA-style build/test provenance manifests. Records builder identity, build type, and parameters for auditability. Schema follows SLSA v1.0 BuildDefinition + RunDetails structure.

Usage:
```bash
# Capture pytest provenance (runs the command)
./ai_template_scripts/provenance_capture.py pytest -- pytest tests/ -v

# Capture cargo test provenance
./ai_template_scripts/provenance_capture.py cargo-test -- cargo test

# Record command without executing
./ai_template_scripts/provenance_capture.py build --no-run --command "cargo build --release"

# Include output file SHA256 digests
./ai_template_scripts/provenance_capture.py build --output-files target/release/myapp -- cargo build --release
```

**Provenance output:** Stored in `reports/provenance/<timestamp>_<type>.json`

**Use in commits:** Reference provenance path in `## Verified` section:
```
## Verified
- Provenance: reports/provenance/20260125_123456_pytest.json
- All tests passed (45 tests in 2.3s)
```

### result.py
Generic Result[T] type for error handling without exceptions. Provides explicit status tracking for ok/error/skipped states. Used throughout looper and scripts for operations that may fail or be intentionally skipped.

**Exports:**
- `Result[T]` - Generic result dataclass with `value`, `error`, `status` fields
- `Result.success(value)` - Create successful result
- `Result.failure(error, value=None)` - Create error result
- `Result.skip(reason)` - Create skipped result
- `format_result(result)` - Format for prompt injection

Usage:
```python
from ai_template_scripts.result import Result, format_result

def fetch_data() -> Result[str]:
    try:
        data = subprocess.run(...)
        return Result.success(data.stdout)
    except Exception as e:
        return Result.failure(str(e))

result = fetch_data()
if result.ok:
    print(result.value)
elif result.skipped:
    print(f"Skipped: {result.error}")
else:
    print(f"Error: {result.error}")
```

### validate_claim.py
Validates benchmark claims in commit messages against eval results. Parses claims like "39/55 CHC" from commit messages and checks them against results in `evals/results/`.

Usage:
```bash
./ai_template_scripts/validate_claim.py                    # Validate HEAD commit
./ai_template_scripts/validate_claim.py --commit abc123    # Validate specific commit
./ai_template_scripts/validate_claim.py --message "39/55 CHC"  # Validate inline claim
./ai_template_scripts/validate_claim.py --strict           # Fail on UNKNOWN claims
```

**Claim patterns recognized:**
- `39/55 CHC` - passed/total eval_id
- `CHC: 39/55` - eval_id: passed/total

**Results directory structure:**
```
evals/results/<eval_id>/
  <run_id>/
    metadata.json   # Contains git_commit
    results.json    # Contains passed/total metrics
```

**Exit codes:**
- `0` = all claims valid (or no claims found)
- `1` = invalid claims (score mismatch)
- `2` = unknown claims (with --strict)

**Pre-push hook integration:** Called by `pre-push-hook.sh` to validate claims before push.

### version.py
CLI version string helper for consistent version output across ai_template scripts.

**Exports:**
- `get_version(script_name)` - Returns version string in format `{script_name} {git_hash} ({date})`

Usage:
```python
from ai_template_scripts.version import get_version

# In your CLI script
if args.version:
    print(get_version("my_script.py"))
    sys.exit(0)
```

Output example: `pulse.py abc1234 (2026-01-30)`

### verify_incremental.sh
Runs incremental verification across verification tools (Kani, TLA+). Detects available tools and only verifies proofs/specs affected by changes since a given commit.

Usage:
```bash
./ai_template_scripts/verify_incremental.sh                  # Check changes since HEAD~1
./ai_template_scripts/verify_incremental.sh --since HEAD~5   # Check last 5 commits
./ai_template_scripts/verify_incremental.sh --tool kani      # Only run Kani proofs
./ai_template_scripts/verify_incremental.sh --force --tier 3 # Full verification
```

**Options:**
- `--since COMMIT` - Only check changes since COMMIT (default: HEAD~1)
- `--force` - Ignore cache, re-verify everything
- `--tool TOOL` - Only run specific tool (kani, tla, all)
- `--tier TIER` - Verification tier: 0=smoke, 1=changed, 2=module, 3=full

**Output:**
- Prints verification results to stdout
- Writes metrics JSON to `metrics/verification/verify-{timestamp}.json`

**Metrics JSON structure:**
```json
{
  "timestamp": "2026-01-30T16:00:00Z",
  "commit": "abc123",
  "summary": {"affected": 12, "cached": 926, "passed": 12, "failed": 0},
  "tools": {
    "kani": {"affected": 12, "cached": 926, "passed": 12, "failed": 0, "duration_secs": 240},
    "tla": {"affected": 1, "cached": 20, "passed": 1, "failed": 0, "duration_secs": 3600}
  }
}
```

**Integration:**
- Integrates with project-specific incremental runners (`scripts/kani_incremental.py`)
- Falls back to full verification if no incremental runner found

### sync_repo.sh
Syncs template files from ai_template to a target repository. Copies rules, scripts, plugins, and config files. Writes `.ai_template_version` with the current commit hash for drift tracking.

Usage:
```bash
./ai_template_scripts/sync_repo.sh /path/to/target_repo           # Sync files
./ai_template_scripts/sync_repo.sh /path/to/target_repo --dry-run # Preview changes
```

**Optional Features:** Repos can opt into optional scripts by creating a `.ai_template_features` file at the repo root:
```
# .ai_template_features
# One feature per line, comments start with #
kani-safe
```

Optional features are stored in `ai_template_scripts/optional/<feature>/` and only synced to repos that explicitly request them. Available features are discovered from the directory listing - see `ai_template_scripts/optional/README.md` for the current list.

### sync_check.sh
Checks template drift across multiple repos. Reads `.ai_template_version` from each repo and compares to current ai_template HEAD. Reports which repos need syncing and flags ahead/diverged versions for follow-up.

Usage:
```bash
./ai_template_scripts/sync_check.sh                    # Check sibling repos
./ai_template_scripts/sync_check.sh /path/to/repos     # Check repos in directory
./ai_template_scripts/sync_check.sh repo1 repo2        # Check specific repos
./ai_template_scripts/sync_check.sh --files            # Include file-level drift
./ai_template_scripts/sync_check.sh --exit-code        # Non-zero exit on drift
```
Exit codes with `--exit-code`:
- `0` = all repos current
- `1` = drift or inconsistent state detected (behind/untracked/ahead/diverged, file drift, missing manifest with `--files`)
- `2` = any repo >100 commits behind

### sync_all.sh
Batch syncs ai_template to all repos in the organization. Iterates through repos found by `sync_check.sh` and applies updates.

Usage:
```bash
./ai_template_scripts/sync_all.sh             # Sync all stale repos
./ai_template_scripts/sync_all.sh --dry-run   # Preview without changes
```

### fix_lfs_offline.sh
Configures Git LFS for offline-friendly operation. When cd-ing into a repo with
LFS hooks, shell prompts that run `git status` can hang if the network is
unavailable. This script configures LFS to skip automatic downloads and use
non-blocking hooks with timeouts.

Usage:
```bash
./ai_template_scripts/fix_lfs_offline.sh                # Current directory
./ai_template_scripts/fix_lfs_offline.sh /path/to/repo  # Specific repo
```

After running, manually use:
```bash
git lfs pull          # Download LFS files when online
git lfs push --all    # Upload before pushing
```

To revert to default LFS behavior: `git lfs install --force`

### update_claude_user_settings.py
Updates `~/.claude/settings.json` with recommended environment variables for AI sessions. Merges settings without overwriting existing values.

**Recommended settings:**
- `DISABLE_AUTOUPDATER=1` - Prevents unexpected Claude Code updates during AI sessions

Usage:
```bash
./ai_template_scripts/update_claude_user_settings.py           # Add recommended settings
./ai_template_scripts/update_claude_user_settings.py --check   # Check current settings
./ai_template_scripts/update_claude_user_settings.py --dry-run # Preview changes
```

**Integration:** Called automatically by `init_from_template.sh` during project setup.

## bin/ Wrapper Scripts

The `bin/` subdirectory contains wrapper scripts that intercept system commands to add AI-specific behavior. Looper adds this directory to PATH at session start.

**Login shell requirement:** AI tools that spawn login shells (codex, dasher) bypass runtime PATH. Add to `~/.zprofile` (login shells) or `~/.zshenv` (all shells):
```bash
export PATH="$HOME/<project>/ai_template_scripts/bin:$PATH"
```
See main README.md "Shell Configuration" section for details. `~/.zshrc` is not read by non-interactive login shells.

### bin/cargo

**Purpose:** Serializes all cargo builds org-wide to prevent OOM and deadlocks.

**How it works:**
- Intercepts `build`, `test`, `check`, `run`, `clippy`, `doc`, `bench`, `miri` commands
- Routes them through `ai_template_scripts.cargo_wrapper` which acquires build/test locks per repo
- Other commands (version, help, add, etc.) pass through unchanged

**Lock location:** `~/.ait_cargo_lock/<repo>/`
**Build lock:** `lock.pid` / `lock.json`
**Test lock:** `lock.test.pid` / `lock.test.json`

**Logged in:** `~/.ait_cargo_lock/<repo>/builds.log`

```bash
cargo build   # → acquires lock, waits if another build running
cargo --help  # → passes directly to real cargo
```

### bin/gh

**Purpose:** Adds AI identity markers to GitHub issues and comments.

**How it works:**
- Intercepts `issue create`, `issue comment`, `issue edit`, `issue close`
- Routes them through `gh_post.py` which adds FROM header and signature
- Auto-adds `mail` label for cross-repo issues
- Other commands pass through unchanged

```bash
gh issue create --title "Bug"  # → adds AI identity, signature
gh repo view                   # → passes directly to real gh
```

### bin/grep

**Purpose:** Blocks recursive grep operations when system memory pressure is critical.

**How it works:**
- Detects recursive grep operations (`-r`, `-R`, `--recursive`)
- Checks macOS memory pressure before allowing recursive ops
- Blocks ALL recursive grep when memory is critical (>85% used) to prevent OOM
- Warns when memory pressure is elevated (70-85% used)
- Suggests alternatives (ripgrep, Claude Code's Grep tool)

```bash
grep -r "pattern" .    # → Blocks if memory critical, warns if elevated
grep "pattern" file    # → passes directly to real grep
```

**Memory thresholds:**
- Normal: <70% memory used
- Warn: 70-85% used (proceeds with warning)
- Critical: >85% used (blocked)

### bin/git

**Purpose:** Block branch creation and serialize commits across parallel sessions.

**How it works:**
- Intercepts `branch`, `checkout -b`, `switch -c` and blocks branch creation
- Intercepts `git commit` and acquires a per-repo lock inside `.git/`
- Releases the lock after the commit finishes
- Other git commands pass through unchanged

**Lock location:** `.git/ait_commit_lock/`

**Config:**
- `AIT_GIT_LOCK_WAIT_S` - Seconds to wait for lock (default: 300)
- `AIT_GIT_LOCK_DISABLE=1` - Disable commit locking entirely

```bash
git branch feature  # → ERROR: Branch creation blocked
git commit -m "..."  # → Acquires lock, commits, releases lock
git status           # → passes directly to real git
```

## GitHub Labels

Standard labels are created by `init_labels.sh` (called by `init_from_template.sh` and `sync_repo.sh`).
Canonical definitions live in `.claude/rules/ai_template.md`.

**Workflow:**
1. Worker claims issue: `gh issue edit N --add-label in-progress --add-label W${AI_WORKER_ID}` (omit ownership label if AI_WORKER_ID is unset)
2. Worker completes: `gh issue edit N --add-label do-audit` (looper transitions to `needs-review`)
3. Manager reviews and closes (or reopens if not done)

**USER Label Protection:**

When USER adds protected labels (like `urgent`), the gh wrapper records this with a tracking comment. AI roles cannot remove USER-set protected labels.

| Protected Label | Meaning |
|-----------------|---------|
| `urgent` | Work on this NOW (USER priority override) |

If an AI tries to remove a USER-protected label:
```
❌ ERROR: Cannot remove USER-protected label 'urgent'
   The 'urgent' label was set by USER and is protected.
   Only USER can remove labels they set.
```

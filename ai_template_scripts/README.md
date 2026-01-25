# ai_template_scripts

Utility scripts for the AI template system.

## Scripts

| Script | Purpose | Called By |
|--------|---------|-----------|
| `audit_alignment.sh` | Check template alignment in a repo | AI, human |
| `json_to_text.py` | Format Claude/Codex JSON output for terminal | `looper.py` |
| `code_stats.py` | Code complexity analysis | MANAGER, human |
| `health_check.py` | Crash log analysis, system health | MANAGER, human |
| `bg_task.py` | Background task management | WORKER AI |
| `bot_token.py` | GitHub App installation token helper | AI, human |
| `cargo_lock_info.py` | Inspect serialized cargo lock status | AI, human |
| `gh_post.py` | Add AI identity to gh issue create/comment | `bin/gh` wrapper |
| `gh_discussion.py` | Create GitHub discussions with AI identity | AI, human |
| `generate_skill_index.py` | Generate .claude/SKILLS.md from skills | AI, human |
| `markdown_to_issues.py` | Sync markdown roadmap with GitHub Issues | AI, human |
| `spawn_session.sh` | Spawn AI session (worker/prover/researcher/manager) in iTerm2 | `spawn_all.sh`, Human |
| `install_dev_tools.sh` | Install dev tools (lizard, radon, etc) | Human (one-time) |
| `init_from_template.sh` | Set up new project from template | Human (one-time) |
| `init_labels.sh` | Create required GitHub labels | `init_from_template.sh`, `sync_repo.sh` |
| `commit-msg-hook.sh` | Git hook for structured commits | Git |
| `pre-commit-hook.sh` | Git hook for copyright/author validation | Git |
| `pulse.py` | Metrics collection, threshold flags | [M] or cron |
| `sync_repo.sh` | Sync template files to target repo | Human, scripts |
| `sync_check.sh` | Check template drift across repos | Human, scripts |

## Details

### audit_alignment.sh
Checks if a repository is properly aligned with the ai_template. Detects missing required files, obsolete files that should be deleted, and forbidden files (like CI workflows). Returns exit code 1 if critical issues found.

Usage:
```bash
./ai_template_scripts/audit_alignment.sh    # Run in any repo
```

### json_to_text.py
Converts Claude/Codex streaming JSON output to readable terminal text. Critical for `looper.py` - all AI output is piped through this.

### code_stats.py
Analyzes cyclomatic/cognitive complexity across multiple languages (Python, Rust, Go, C++, etc). Uses best-in-class tools per language (radon for Python, gocyclo for Go, etc).

### health_check.py
Parses `worker_logs/crashes.log` to calculate failure rates and system health status. Used by MANAGER for auditing.

### bg_task.py
Manages long-running background tasks that survive worker iteration timeouts. Stores state in `.background_tasks/`.

### bot_token.py
Fetches GitHub App installation tokens for the AI fleet using local app credentials.

Usage:
```bash
./ai_template_scripts/bot_token.py --json
```

### cargo_lock_info.py
Inspects the serialized cargo lock holder and reports whether the lock appears stale.

Usage:
```bash
./ai_template_scripts/cargo_lock_info.py
```

### gh_post.py
Adds AI identity markers (FROM header and signature) to GitHub issues and comments. Called automatically by the `bin/gh` wrapper when using `gh issue create` or `gh issue comment`. Also auto-adds the `mail` label for cross-repo issues.

### gh_discussion.py
Creates GitHub discussions with AI identity markers. Provides a consistent interface similar to `gh issue create`. Used for posting to Dash News (ayates_dbx/dashnews).

Usage:
```bash
./ai_template_scripts/gh_discussion.py create --title "Title" --body "Body"
./ai_template_scripts/gh_discussion.py create --title "Title" --body "Body" --category "Q&A"
```

### generate_skill_index.py
Builds `.claude/SKILLS.md` from the skills in `.claude/commands/`.

Usage:
```bash
./ai_template_scripts/generate_skill_index.py
```

### markdown_to_issues.py
Syncs markdown roadmaps to GitHub Issues (and exports issues to markdown).

Usage:
```bash
./ai_template_scripts/markdown_to_issues.py --export
```

### spawn_session.sh
Spawns a worker, prover, researcher, or manager session in a new iTerm2 tab. Smart argument detection allows flexible ordering.

Usage:
```bash
./ai_template_scripts/spawn_session.sh worker              # Worker in current directory
./ai_template_scripts/spawn_session.sh prover ~/z4         # Prover in specific project
./ai_template_scripts/spawn_session.sh ~/z4 researcher     # Swapped args also work
./ai_template_scripts/spawn_session.sh manager ~/z4        # Manager in specific project
./ai_template_scripts/spawn_session.sh --dry-run manager   # Preview without running
```

### install_dev_tools.sh
One-time setup script that installs development tools needed by other scripts (lizard, radon, gocyclo, etc).

### init_from_template.sh
Run once after copying this template to a new project. Sets up GitHub labels, installs git hooks, removes template-specific files.

### commit-msg-hook.sh
Git commit-msg hook that auto-adds iteration numbers `[W]N` and validates issue references. Installed to `.git/hooks/commit-msg`.

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

### sync_repo.sh
Syncs template files from ai_template to a target repository. Copies rules, scripts, plugins, and config files. Writes `.ai_template_version` with the current commit hash for drift tracking.

Usage:
```bash
./ai_template_scripts/sync_repo.sh /path/to/target_repo           # Sync files
./ai_template_scripts/sync_repo.sh /path/to/target_repo --dry-run # Preview changes
```

### sync_check.sh
Checks template drift across multiple repos. Reads `.ai_template_version` from each repo and compares to current ai_template HEAD. Reports which repos need syncing.

Usage:
```bash
./ai_template_scripts/sync_check.sh                    # Check sibling repos
./ai_template_scripts/sync_check.sh /path/to/repos     # Check repos in directory
./ai_template_scripts/sync_check.sh repo1 repo2        # Check specific repos
```

## bin/ Wrapper Scripts

The `bin/` subdirectory contains wrapper scripts that intercept system commands to add AI-specific behavior. Looper adds this directory to PATH at session start.

### bin/cargo

**Purpose:** Serializes all cargo builds org-wide to prevent OOM and deadlocks.

**How it works:**
- Intercepts `build`, `test`, `check`, `run`, `clippy`, `doc`, `bench`, `miri` commands
- Routes them through `cargo_wrapper.py` which acquires a global lock
- Other commands (version, help, add, etc.) pass through unchanged

**Lock location:** `~/.ait_cargo_lock/`

**Logged in:** `~/.ait_cargo_lock/builds.log`

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

## GitHub Labels

Standard labels are created by `init_labels.sh` (called by `init_from_template.sh` and `sync_repo.sh`).
Canonical definitions live in `.claude/rules/ai_template.md`.

**Workflow:**
1. Worker claims issue: `gh issue edit N --add-label in-progress`
2. Worker completes: `gh issue edit N --add-label needs-review`
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

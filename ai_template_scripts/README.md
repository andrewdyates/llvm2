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
| `gh_post.py` | Add AI identity to gh issue create/comment | `bin/gh` wrapper |
| `gh_discussion.py` | Create GitHub discussions with AI identity | AI, human |
| `spawn_session.sh` | Spawn worker/manager in iTerm2 tab | Human |
| `install_dev_tools.sh` | Install dev tools (lizard, radon, etc) | Human (one-time) |
| `init_from_template.sh` | Set up new project from template | Human (one-time) |
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

### gh_post.py
Adds AI identity markers (FROM header and signature) to GitHub issues and comments. Called automatically by the `bin/gh` wrapper when using `gh issue create` or `gh issue comment`. Also auto-adds the `mail` label for cross-repo issues.

### gh_discussion.py
Creates GitHub discussions with AI identity markers. Provides a consistent interface similar to `gh issue create`. Used for posting to Dash News (ayates_dbx/dashnews).

Usage:
```bash
./ai_template_scripts/gh_discussion.py create --title "Title" --body "Body"
./ai_template_scripts/gh_discussion.py create --title "Title" --body "Body" --category "Q&A"
```

### spawn_session.sh
Spawns a worker or manager session in a new iTerm2 tab. Smart argument detection allows flexible ordering.

Usage:
```bash
./ai_template_scripts/spawn_session.sh worker              # Worker in current directory
./ai_template_scripts/spawn_session.sh manager ~/z4        # Manager in specific project
./ai_template_scripts/spawn_session.sh ~/z4 worker         # Swapped args also work
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

Warning mode only - doesn't block commits. Installed to `.git/hooks/pre-commit`.

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

## GitHub Labels

Standard labels created by `init_from_template.sh` and `sync_repo.sh`:

| Label | Color | Purpose |
|-------|-------|---------|
| `P0` | red | Critical priority - drop everything |
| `P1` | orange | High priority - do soon |
| `P2` | yellow | Medium priority - normal queue |
| `P3` | green | Low priority - when time permits |
| `in-progress` | purple | Currently being worked on |
| `blocked` | black | Cannot proceed, waiting on something |
| `needs-review` | light blue | Worker flagged as done, needs Manager review |
| `mail` | blue | Cross-repo message (auto-added by gh wrapper) |

**Workflow:**
1. Worker claims issue: `gh issue edit N --add-label in-progress`
2. Worker completes: `gh issue edit N --add-label needs-review`
3. Manager reviews and closes (or reopens if not done)

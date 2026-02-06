# Local Development Mode

Guide for using ai_template's local development features when GitHub API is unavailable or saturated.

**Author:** Andrew Yates <ayates@dropbox.com>

## Overview

ai_template supports two levels of local development mode:

| Mode | Environment | Effect |
|------|-------------|--------|
| **Basic** | `AIT_LOCAL_MODE=1` | Disable most GitHub API calls, use cache fallbacks |
| **Full** | `AIT_LOCAL_MODE=full` | Complete offline operation with local issue storage |

## Basic Local Mode

Disables GitHub API calls while maintaining normal git operations.

**Enable via any method:**
```bash
# Environment variable (highest priority)
export AIT_LOCAL_MODE=1

# Touch file
touch .local_mode

# Config file (.looper_config.json)
{"local_mode": true}
```

**Effects:**
- `gh` commands return early with "(local mode - GitHub API disabled)"
- Issue fetching and mirror refresh skipped
- Auth/network checks skipped at startup
- Git operations and commit hooks still run normally

**When to use:**
- Offline development without network access
- Avoiding rate limits during intensive testing
- Testing looper logic without API calls
- CI environments without GitHub credentials

See `docs/looper.md` "Local Mode" for full configuration.

## Full Local Mode

Complete offline operation with local issue storage in `.issues/` directory.

**Enable:**
```bash
export AIT_LOCAL_MODE=full
```

**Effects:**
- All `gh issue` commands route to local storage
- Issues stored as Markdown files with YAML frontmatter
- Full issue workflow available: create, list, view, edit, close, comment
- Non-issue `gh` commands warn but still execute

### Local Issue Format

Issues are stored in `.issues/` directory:

```
.issues/
├── L1.md           # Local issue #1
├── L2.md           # Local issue #2
├── _meta.json      # Next ID counter, metadata
└── _config.json    # Optional local config
```

**Issue file format (L1.md):**
```markdown
---
id: L1
title: "Add feature X"
labels: ["P2", "feature"]
state: open
created_at: 2026-02-03T12:00:00Z
updated_at: 2026-02-03T12:00:00Z
---

Issue body content here.

## Comments

### [W]42 @ 2026-02-03T12:05:00Z
Comment text here.
```

**ID prefix:** Local issues use `L<int>` prefix (e.g., `L1`, `L42`) to avoid collision with GitHub issue numbers. This allows mixed workflows where some issues are on GitHub and some are local.

### Issue Commands

All standard `gh issue` commands work in full local mode:

```bash
# Create issue
gh issue create --title "Add feature X" --label P2

# List issues
gh issue list
gh issue list --state open --label P2

# View issue
gh issue view L1
gh issue view L1 --json title,labels,body

# Edit issue
gh issue edit L1 --add-label in-progress
gh issue edit L1 --remove-label P2 --add-label P1
gh issue edit L1 --title "Updated title"

# Close/reopen
gh issue close L1
gh issue reopen L1

# Comment
gh issue comment L1 --body "Progress update"
```

**JSON output:** The `--json` flag works for compatible fields:
```bash
gh issue list --json number,title,labels,state
gh issue view L1 --json title,body,labels
```

### Workflow Example

```bash
# Start full local mode
export AIT_LOCAL_MODE=full

# Create work issue
gh issue create --title "Implement auth" --label P1 --label feature
# Created L1

# Claim and work
gh issue edit L1 --add-label in-progress --add-label W1

# Make progress, commit
git commit -m "[W]1: Part of #L1 - Add auth module"

# Add comment
gh issue comment L1 --body "Added basic auth module, tests pending"

# Mark for audit
gh issue edit L1 --add-label do-audit --remove-label in-progress
```

## Syncing Local Issues to GitHub

When returning to online mode, sync local issues to GitHub:

### Commands

```bash
# Preview what would be synced
python3 ai_template_scripts/sync_local_issues.py sync --dry-run

# Push local issues to GitHub
python3 ai_template_scripts/sync_local_issues.py sync

# Import existing GitHub issues locally (for offline work)
python3 ai_template_scripts/sync_local_issues.py bootstrap

# Check sync status
python3 ai_template_scripts/sync_local_issues.py status
```

### Sync Behavior

**sync command:**
- Creates new GitHub issues for each local issue
- Preserves labels, body, comments
- Maps `L<n>` to new GitHub issue number
- Stores mapping in `_meta.json` for reference

**bootstrap command:**
- Downloads open GitHub issues to `.issues/`
- Preserves issue numbers (not L-prefixed)
- Useful before going offline

**status command:**
- Shows local issues not yet synced
- Shows GitHub issues not cached locally
- Reports any sync conflicts

### ID Mapping

After sync, local issue IDs are mapped to GitHub numbers:

```json
// .issues/_meta.json (after sync)
{
  "next_id": 3,
  "synced": {
    "L1": 2345,
    "L2": 2346
  }
}
```

Commits referencing `#L1` can be updated to `#2345` if needed.

## Limitations

**Full local mode limitations:**
- No GitHub Actions/CI integration
- No GitHub notifications or webhooks
- No cross-repo issue references
- No GitHub Projects integration
- `gh pr` commands may not work (PR workflow assumes online)
- Some looper features that require API may be disabled

**Local issue limitations:**
- No assignees (single-developer mode assumed)
- No milestones
- Comments stored inline (not threaded)
- No reactions/emoji

**Recommended workflow:**
1. Use full local mode for focused offline development
2. Sync issues to GitHub periodically for backup
3. Return to online mode for collaboration and PR workflows

## Configuration

Local mode can be configured in `.looper_config.json`:

```json
{
  "local_mode": false,
  "local_config": {
    "issue_dir": ".issues",
    "auto_sync": false,
    "sync_on_close": false
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `issue_dir` | `.issues` | Directory for local issue storage |
| `auto_sync` | `false` | Auto-sync issues when returning online |
| `sync_on_close` | `false` | Sync issue to GitHub when closed |

## Troubleshooting

**Issue commands not working in full local mode:**
- Verify `AIT_LOCAL_MODE=full` is set (not just `1`)
- Check `.issues/` directory exists and is writable
- Run `gh issue list` to verify local mode is active

**Sync failing:**
- Check GitHub auth: `gh auth status`
- Verify rate limits: Check `~/.ait_gh_cache/rate_state.json`
- Try `--dry-run` first to preview changes

**Mixed local/remote issues:**
- Use `L` prefix for local issues to avoid confusion
- After sync, update commit messages if needed
- Keep `_meta.json` synced mapping for reference

## References

- `ai_template_scripts/gh_local.py` - Local gh command handler
- `ai_template_scripts/local_issue_store.py` - Storage implementation
- `ai_template_scripts/sync_local_issues.py` - Sync utilities
- `docs/looper.md` - Looper configuration including basic local mode
- `docs/wrappers.md` - Environment variables including `AIT_LOCAL_MODE`

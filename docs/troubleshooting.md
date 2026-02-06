# Troubleshooting
<!-- Issue #1524 completed: 2026-01-31 -->

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

Quick reference for common operational issues.

## Quick Diagnostics

Quick commands to check system health:

- `./ai_template_scripts/pulse.py` - Overall health
- `cat ~/.ait_gh_cache/rate_state.json | jq .resources` - GitHub quotas
- `./ai_template_scripts/cargo_lock_info.py` - Build lock status
- `ls .flags/` - Active alerts

For wrapper-specific issues (git, gh, cargo, grep), see `docs/wrappers.md`.

## Debug Logging

Enable debug output from ai_template_scripts modules using environment variables.

### Unified Debug Flag

```bash
export AIT_DEBUG=1  # Enable debug for ALL modules
```

This is the recommended way to enable debug logging. Currently integrated with:
- `gh_rate_limit` - GitHub API rate limiting, cache decisions

### Module-Specific Debug Flags

For targeted debugging, use module-specific flags:

| Module | Env Var | Uses shared_logging |
|--------|---------|---------------------|
| gh_rate_limit | `AIT_GH_DEBUG=1` | Yes - `debug_log()` |
| looper | `LOOPER_DEBUG=1` | No - own debug system |

**Note:** `looper` has its own debug system (`LOOPER_DEBUG`) that predates `shared_logging`. It's not integrated with `AIT_DEBUG`.

**Hierarchy:** For modules using `shared_logging`, `AIT_DEBUG` takes precedence over module-specific vars.

### Debug Output Format

Modules using `shared_logging.debug_log()` emit prefixed messages:

```
gh_rate_limit [DEBUG]: Using cached response for repos/dropbox-ai-prototypes/ai_template/issues
gh_rate_limit [DEBUG]: REST fallback activated - GraphQL quota exhausted
```

Filter specific module: `AIT_DEBUG=1 ./script.py 2>&1 | grep "gh_rate_limit"`

See `ai_template_scripts/shared_logging.py` for the debug_log implementation.

## Bootstrap Issues

Issues that occur during initial setup with `init_from_template.sh`.

### Prerequisites Not Met

**Symptoms:** Script fails early with dependency errors
**Check:** Run init script with --verify flag:
```bash
./ai_template_scripts/init_from_template.sh --verify
```
**Solution:**
1. Install required tools: `brew install gh jq`
2. Ensure Python 3.11+: `python3 --version`
3. Verify bash 4+: `bash --version`

### GitHub CLI Not Authenticated

**Symptoms:**
- `gh: not logged in` errors
- `gh auth status` shows "not logged in to any hosts"
- Label creation fails silently
**Check:** `gh auth status`
**Solution:**
1. Authenticate: `gh auth login`
2. Choose GitHub.com
3. Use SSH or HTTPS (SSH recommended)
4. Verify: `gh auth status` shows "Logged in"

### Git User Not Configured

**Symptoms:**
- `Author identity unknown` on first commit
- git commands fail with config warnings
**Check:**
```bash
git config user.name
git config user.email
```
**Solution:**
1. Set globally:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Label Creation Failed

**Symptoms:**
- `init_labels.sh` outputs errors
- Missing priority labels (P1, P2, etc.)
**Check:** `gh label list | head -10`
**Solution:**
1. Check rate limits: `cat ~/.ait_gh_cache/rate_state.json | jq .resources.core`
2. Wait for rate limit reset if exhausted
3. Retry: `./ai_template_scripts/init_labels.sh`
4. For network issues, verify connectivity: `gh api rate_limit`

### Wrong Working Directory

**Symptoms:**
- Scripts fail with "file not found" errors
- `CLAUDE.md` or `.claude/` not detected
**Check:** `ls CLAUDE.md .claude/roles/`
**Solution:**
1. Ensure you're in repo root: `pwd`
2. Navigate to correct directory: `cd ~/your-repo-name`
3. Verify template structure exists

## Session Issues

### Stale Connection (Silence Timeout)

**Symptoms:** `claude killed due to silence (stale connection)` in failures.log
**Cause:** Network issues, API throttling, or hung tool calls
**Solution:**
1. Check network connectivity
2. Review rate limits: `cat ~/.ait_gh_cache/rate_state.json`
3. Session auto-restarts with error_delay (defaults per role: worker 5s, others 60s; see `.claude/roles/*.md` or `.looper_config.json`)
4. If persistent, check for blocked operations or long-running commands with no output

### Timeout

**Symptoms:** `claude timed out` in failures.log
**Cause:** Operation exceeded configured timeout
**Solution:**
1. Review iteration_timeout in `.claude/roles/<role>.md` or `.looper_config.json`
2. Check for long-running operations (tests, builds)
3. Split large tasks into smaller commits

## Git Push Conflicts (Multi-Worker)

When multiple workers push to the same repo, push rejections are common.

### Push Rejected

**Symptoms:**
- `[rejected] main -> main (fetch first)`
- `error: failed to push some refs`

**Recovery Flow (Safe Path):**

```bash
# 1. Commit any uncommitted changes first (even WIP)
git add <your-modified-files> && git commit -m "[W]N: WIP - push conflict recovery"

# 2. Rebase on remote (no stash needed since already committed)
git pull --rebase origin main

# 3. If merge conflicts, resolve them then continue
git add <resolved-files>
git rebase --continue

# 4. Push
git push
```

**Key Rules:**
- ❌ `git stash` - blocked by org rules
- ❌ `git rebase -i` - blocked (interactive)
- ❌ `git reset --hard` - dangerous, avoid unless USER explicitly requests
- ✅ Always commit before pull/rebase
- ✅ Use `git pull --rebase` (not plain `git pull`)

### Uncommitted Changes + Push Rejected

**Symptoms:**
- Push rejected
- `git pull --rebase` fails: "Your local changes would be overwritten"

**Solution:**
1. **Commit first** - even a WIP commit
2. **Then rebase** - `git pull --rebase origin main`
3. **Resolve conflicts** if any
4. **Push**

### Merge Conflict During Rebase

**Symptoms:** `CONFLICT (content): Merge conflict in <file>`

**Solution:**
```bash
# 1. Edit conflicting files (keep your changes or theirs)
# 2. Mark resolved
git add <file>

# 3. Continue rebase
git rebase --continue

# 4. Push
git push
```

**If stuck:** `git rebase --abort` returns to pre-rebase state.

## GitHub Issues

See also `docs/wrappers.md` for detailed gh wrapper documentation.

### Rate Limiting

**Symptoms:**
- `gh_rate_limit: REST fallback activated` messages
- `core: N remaining` or `graphql: N remaining` near zero
**Check:** `cat ~/.ait_gh_cache/rate_state.json | jq .resources`
**Solution:**
1. Wait for reset (times shown in rate_state.json)
2. Wrapper auto-switches REST <-> GraphQL
3. For emergencies: clear TTL cache (see "Cache Issues" below)

**Multi-app overflow:** For automated AI workers using GitHub Apps (`AIT_USE_GITHUB_APPS=1`),
quota exhaustion triggers automatic fallback to lower-tier apps. See `docs/gh_apps.md` for
overflow behavior, app tier configuration, and troubleshooting.

### Cache Issues

**Symptoms:** Stale data, unexpected gh output
**Check:** `ls -la ~/.ait_gh_cache/`
**Solution:**
1. Clear TTL cache: `find ~/.ait_gh_cache -maxdepth 1 -type f -name '*.json' ! -name 'rate_state.json' ! -name 'change_log.json' -delete`
2. Preserve rate_state.json unless corrupted (and change_log.json if queued writes matter)
3. Test with fresh request: `gh issue list -L 5`

## Test Failures

### Monkeypatch AttributeError After Refactor

**Symptoms:** `AttributeError` from `monkeypatch.setattr("module.path.attr", ...)` after a
module split or rename.
**Cause:** Monkeypatch target string still points at the old import path.
**Solution:**
1. Update the target string to the new import path.
2. Search for stale targets: `rg -n "monkeypatch.setattr\\(\"old_module" tests/`
3. Re-run the specific test file that failed.
Source: `tests/test_generate_skill_index.py:394`.

### Pre-Commit Integration Failures

**Symptoms:** `test_pre_commit_all_hooks_pass` fails in `tests/test_pre_commit_integration.py`.
**Cause:** One or more pre-commit hooks (ruff, shellcheck, mypy) are failing on the repo.
**Solution:**
1. Run `pre-commit run --all-files` to identify the failing hook(s).
2. Fix the underlying lint or type errors.
3. Re-run the test to confirm green.
Source: `tests/test_pre_commit_integration.py:72`.

## Build Issues

See also `docs/wrappers.md` for detailed cargo wrapper documentation.

### Cargo Lock Held

**Symptoms:**
- Build hangs with "waiting for lock" messages
- `cargo_wrapper: waiting for build lock (N seconds)`
**Check:** `./ai_template_scripts/cargo_lock_info.py`
**Solution:**
1. Check if legitimate build is running
2. For stale locks: `rm ~/.ait_cargo_lock/<repo>/lock.*`
3. If orphaned processes are suspected, ask the USER before any manual intervention
4. With explicit approval, terminate orphan rustc: `pkill -f rustc`

### Build Timeout

**Symptoms:** Build killed after 1 hour
**Check:** `~/.ait_cargo_lock/<repo>/builds.log`
**Solution:**
1. Check for infinite compilation loops
2. Review Cargo.toml for recursive deps
3. Increase timeout in cargo_wrapper.toml: `build_sec = 7200`

### Stuck Process Detection

**Symptoms:** `.flags/stuck_process` flag is set
**Cause:** Long-running build/verification process detected (cargo, rustc, kani, cbmc, z3, cvc5, lean)
**Check:** `./ai_template_scripts/pulse.py | grep -A20 "long_running_processes"`
**Solution:**
1. **Legitimate stuck build:** Check if process is making progress: `tail -f ~/.ait_cargo_lock/<repo>/builds.log`
2. **False positive from other repo:** Process may be running in a different repo. Verify repo attribution in pulse output.
3. **If truly stuck:** Ask the USER before killing processes manually.

**Note:** Python processes (looper.py, memory_watchdog, http.server, etc.) are NOT monitored for stuck detection (#2139).

## Memory Issues

### Detecting OOM

**Symptoms:**
- Process exits with code 137 or signal -9
- Session crashes during heavy operations
**Check:**
```bash
grep '"was_oom": true' worker_logs/memory.log
grep '"event": "post_crash"' worker_logs/memory.log | tail -5
```
**Solution:**
1. Review memory pressure before crash: `jq .memory.pressure_level` on crash entries
2. Identify memory-intensive commands: `jq .command` on high-pressure entries
3. Split large operations (e.g., incremental builds instead of full rebuild)

### Memory Pressure

**Symptoms:**
- `pressure_level: "critical"` in memory.log
- System slowdown during sessions
**Check:** `python3 -c "from looper.memory_logger import get_memory_state; print(get_memory_state())"`
**Solution:**
1. Close other applications
2. Check for runaway processes: `top -o mem`
3. Adjust `memory_watchdog_threshold` in .looper_config.json if too sensitive

### Memory Log Analysis

```bash
# Recent memory state (last 10 entries)
tail -10 worker_logs/memory.log | jq .

# Commands that ran during high pressure
grep '"pressure_level": "warning"' worker_logs/memory.log | jq -r .command

# Memory trend (used_percent over time)
jq -r '[.timestamp, .memory.used_percent] | @tsv' worker_logs/memory.log
```

## Config Issues

### Unknown Config Keys

**Symptoms:** `Unknown config key(s)` warnings
**Cause:** Typo or old config format
**Solution:**
1. Check `looper/config.py` for valid keys
2. Refer to `docs/looper.md` for config documentation
3. Remove or rename unknown keys

### JSON Syntax Errors (.looper_config.json)

**Symptoms:** Config not loading, defaults used
**Check:** `python3 -c "import json; json.load(open('.looper_config.json'))"`
**Solution:**
1. Fix JSON syntax (missing quotes, brackets)
2. Use jq or IDE validation

### TOML Syntax Errors (cargo_wrapper.toml, pyproject.toml)

**Symptoms:** Config not loading, defaults used
**Check:** `python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
**Solution:**
1. Fix TOML syntax (missing quotes, brackets)
2. Use tomlcheck or IDE validation

## Checkpoint Issues

### Checkpoint Not Loading

**Symptoms:** Session does not resume, starts fresh
**Check:** `cat .looper_checkpoint_*.json | jq .`
**Solution:**
1. Verify JSON syntax
2. Check session_id matches current session
3. If corrupted: delete checkpoint (session restarts)

### Stale Checkpoint State

**Symptoms:** Wrong issues being worked on
**Cause:** Checkpoint from old session
**Solution:**
1. Delete checkpoint: `rm .looper_checkpoint_*.json`
2. Session will pick up fresh from git history

## Worker Log Analysis

**Location:** `worker_logs/`

### Key Files

| File | Content | When to Check |
|------|---------|---------------|
| `failures.log` | All failure events (fallback: `crashes.log`) | After unexpected restarts |
| `worker_1.log` | Worker output | During debugging |
| `stats.json` | Iteration statistics | Performance analysis |
| `memory.log` | Memory state snapshots | OOM debugging |

### Common Patterns

**High failure rate:**
```bash
# Count failures in last 24h
grep "$(date +%Y-%m-%d)" worker_logs/failures.log | wc -l
```

**Repeated same error:**
```bash
# Find patterns
cut -d: -f4- worker_logs/failures.log | sort | uniq -c | sort -rn
```

## When to Escalate

File an issue if:
- Same problem recurs > 3 times with no resolution
- Error message not in this doc
- Workaround required custom code changes

Use `[ai_template]` prefix for infrastructure issues.

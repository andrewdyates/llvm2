# GitHub Apps Rate Limit Scaling

Author: Andrew Yates <ayates@dropbox.com>

This document describes the GitHub Apps infrastructure for API rate limit scaling
in the AI fleet.

## Overview

GitHub's standard authentication provides 5,000 requests/hour for authenticated users.
With multiple AI workers operating concurrently across 50+ repositories, this shared
quota is quickly exhausted.

GitHub Apps provide independent 5,000/hr quotas per installation. By deploying multiple
apps, each focused on specific repositories, we achieve linear rate limit scaling.

## Architecture

### App Tiers

Apps are organized into three tiers with decreasing priority:

| Tier | Pattern | Example | Scope |
|------|---------|---------|-------|
| 1. Per-repo | `dbx-<repo>-ai` | `dbx-ai-template-ai` | Single high-activity repo |
| 2. Director | `dbx-d<DIR>-ai` | `dbx-dMATH-ai` | All repos under a director |
| 3. Wildcard | `dbx-ai` | `dbx-ai` | Global fallback |

**Lookup priority:** When requesting a token for a repository, the system checks:
1. Per-repo app (highest priority, if exists)
2. Director app (shared across director's repos)
3. Wildcard app (global fallback)

### Overflow Behavior

When an app's quota is exhausted, the system automatically falls back to lower-tier
apps. This provides resilience during high-activity periods.

```
Primary app exhausted → Director app (if different) → Wildcard app
```

Overflow events are logged to `~/.ait_gh_cache/overflow_log.json` for Manager visibility.

## Configuration

### Directory Structure

```
~/.ait_gh_apps/
├── config.yaml          # App definitions and repo mappings
├── dbx-ai-template-ai.pem  # Private key for ai_template app
├── dbx-dMATH-ai.pem     # Private key for MATH director app
├── dbx-ai.pem           # Private key for wildcard app
└── ...
```

### config.yaml Format

```yaml
# ~/.ait_gh_apps/config.yaml
apps:
  dbx-ai-template-ai:
    app_id: 12345
    repos:
      - ai_template

  dbx-dMATH-ai:
    app_id: 12346
    repos:
      - z4
      - tla2
      - gamma-crown
      - lean5

  dbx-ai:
    app_id: 12347
    repos:
      - "*"  # Wildcard - handles all repos
```

### Private Keys

Private keys (`.pem` files) must have 600 permissions:

```bash
chmod 600 ~/.ait_gh_apps/*.pem
```

Keys are generated when creating the GitHub App in the organization settings.

## Environment Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `AIT_USE_GITHUB_APPS` | Enable GitHub Apps authentication | `1` to enable |
| `AIT_GH_APP_ACTIVE` | Set by gh wrapper when using app token | `1` when active |
| `AIT_CURRENT_REPO` | Current repository name | e.g., `ai_template` |
| `AIT_GH_APPS_DEBUG` | Enable debug logging | `1` to enable |

### Enabling GitHub Apps

To enable GitHub Apps authentication for automated AI workers:

```bash
export AIT_USE_GITHUB_APPS=1
```

This causes the `gh` wrapper to request installation tokens instead of using
standard `gh auth` credentials.

## Usage

### Programmatic

```python
from ai_template_scripts.gh_apps import get_token, get_app_for_repo

# Get token for a specific repo
token = get_token("ai_template")

# Find which app handles a repo
app_name = get_app_for_repo("z4")  # Returns "dbx-dMATH-ai" or "dbx-z4-ai"
```

### CLI

```bash
# Get token for a repo
python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template

# Get token with overflow fallback (picks best available app)
python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template --overflow
```

## Rate State Tracking

Per-app quota usage is tracked in `~/.ait_gh_cache/rate_state.json`:

```json
{
  "timestamp": 1707123456.789,
  "apps": {
    "dbx-ai-template-ai": {
      "core": {"limit": 5000, "remaining": 4500, "reset": 1707127056},
      "graphql": {"limit": 5000, "remaining": 4800, "reset": 1707127056}
    },
    "dbx-dMATH-ai": {
      "core": {"limit": 5000, "remaining": 2000, "reset": 1707127056}
    }
  },
  "usage_log": [...]
}
```

## Overflow Logging

When quota overflow triggers a fallback, events are logged to
`~/.ait_gh_cache/overflow_log.json`:

```json
{
  "events": [
    {
      "timestamp": 1707123456.789,
      "resource": "graphql",
      "from_app": "dbx-ai-template-ai",
      "to_app": "dbx-dTOOL-ai",
      "repo": "ai_template"
    }
  ]
}
```

This allows Manager to detect systematic quota exhaustion patterns.

## Troubleshooting

### Symptom: "Rate limit exceeded" despite apps configured

**Cause:** App token not being used, falling back to `gh auth`.

**Check:**
1. Verify `AIT_USE_GITHUB_APPS=1` is set
2. Check `gh_rate_limit` debug output: `AIT_GH_APPS_DEBUG=1 gh issue list`
3. Verify config exists: `ls ~/.ait_gh_apps/config.yaml`

### Symptom: Token generation fails

**Cause:** Private key missing or wrong permissions.

**Check:**
```bash
ls -la ~/.ait_gh_apps/*.pem
# Should show 600 permissions: -rw-------

# Fix permissions
chmod 600 ~/.ait_gh_apps/*.pem
```

### Symptom: Wrong app selected for repo

**Cause:** App tier priority or config mismatch.

**Debug:**
```bash
AIT_GH_APPS_DEBUG=1 python3 -m ai_template_scripts.gh_apps.get_token --repo <name>
```

Look for "selected per-repo app" vs "selected director app" vs "selected wildcard app"
in the debug output.

### Symptom: Frequent overflow events

**Cause:** Per-repo app quota insufficient for workload.

**Check:**
```bash
# View recent overflow events
cat ~/.ait_gh_cache/overflow_log.json | jq '.events[-5:]'

# Check quota status
cat ~/.ait_gh_cache/rate_state.json | jq '.apps'
```

**Solution:** Consider adding a dedicated per-repo app for high-activity repositories.

## Director Mappings

Current director-to-app mappings (source: `org_chart.md`):

| Director | App Name | Key Repos |
|----------|----------|-----------|
| MATH | `dbx-dMATH-ai` | z4, tla2, gamma-crown, lean5 |
| ML | `dbx-dML-ai` | model_mlx_migration, voice |
| LANG | `dbx-dLANG-ai` | zani, sunder, tRust, tMIR |
| TOOL | `dbx-dTOOL-ai` | ai_template, dasher, dterm |
| KNOW | `dbx-dKNOW-ai` | sg, chunker, dashextract |
| RS | `dbx-dRS-ai` | kafka2, claude_code_rs |
| APP | `dbx-dAPP-ai` | dashpresent |
| DBX | `dbx-dDBX-ai` | dbx_datacenter, dbx_unitq |

## See Also

- `docs/troubleshooting.md` - General troubleshooting guide
- `docs/wrappers.md` - gh wrapper documentation
- `ai_template_scripts/gh_apps/` - Source code

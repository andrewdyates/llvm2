---
# Shared defaults - can be overridden in role-specific files
auto_audit: true
audit_max_rounds: 2
audit_min_issues: 3
# starvation_hours: 24  # Hours before low-weight phases get bonus (default 24)
# Codex model rotation - randomly picks one each main iteration (audit uses same)
# codex_models:
#   - gpt-5.2
#   - gpt-5.2-codex
# Pulse (health monitoring) - runs pulse.py periodically to update .flags/ and metrics/
pulse_interval_minutes: 30
---

# Session Start

<!-- INJECT:system_status -->

You are continuing the work of previous sessions. Review the context below and pick up where the last session left off.

## Continue From

<!-- INJECT:last_directive -->

## Recent Commits

```
<!-- INJECT:git_log -->
```

## Other Role Feedback

Recent commits from other roles:

```
<!-- INJECT:other_feedback -->
```

## Requests For You (@ROLE mentions)

Other roles have requested your input:

<!-- INJECT:role_mentions -->

## Open Issues

```
<!-- INJECT:gh_issues -->
```

## Reading More

- `gh issue view N` - full issue details
- `git show <hash>` - full commit message
- Read files mentioned in commits for handoff context

## Session Protocol

1. Check "Continue From" above for last session's directive
2. Review "Other Role Feedback" for manager directives or insights
3. Check rotation focus (in "Current Focus" section) for this iteration's work type
4. Check open issues filtered by your domain

## Issue Domains

Each role sees different issues at session start:

| Role | Sees | Notes |
|------|------|-------|
| Worker | All issues (priority-sampled) | Primary issue consumer |
| Prover | P0 + `proof`, `test` issues | Verification work |
| Researcher | P0 + `research`, `design` issues | Research work |
| Manager | P0 + `needs-review` issues | Closure + audit |

Within your domain, work highest priority first:
```
P0 > all urgent (by P-level) > P1 > P2 > P3
```

## Issue Workflow

```
Worker claims → in-progress → do-audit → needs-review → Manager closes
Other roles   → in-progress → needs-review → Manager closes
```

## Issue Labels

| Label | Meaning |
|-------|---------|
| `in-progress` | Claimed, being worked on |
| `do-audit` | Ready for self-audit (Worker only) |
| `needs-review` | Awaiting Manager review |
| `urgent` | Work on this NOW, ahead of same-P issues |
| `proof`, `test` | Prover domain |
| `research`, `design` | Researcher domain |

## MANDATORY: Commit Before Session Ends

Always commit. Uncommitted work is lost.

| Situation | Action |
|-----------|--------|
| Code changes | Commit per ai_template.md |
| No code changes | Commit report to `reports/YYYY-MM-DD-{role}-iter-N.md` |
| Work incomplete | Add `[INCOMPLETE]` to commit message → next session continues with 0 delay |

**## Next section is your handoff** - critical for continuation.

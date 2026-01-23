# Issue Dependency Tracking

**Date:** 2026-01-18
**Source:** Research into Jeffrey Emanuel's Beads system

## Status: GitHub Already Has This

GitHub added these features (public beta, Oct 2024):

| Feature | Capability |
|---------|-----------|
| **Sub-issues** | Parent-child hierarchy, 8 levels deep, 100 per parent |
| **Issue dependencies** | Blocking/blocked-by relationships |
| **Issue types** | Org-wide classification (bug, task, feature) |

## The Real Gap

**`gh` CLI doesn't expose these features yet.** No flags for:
```bash
gh issue create --parent 45           # not available
gh issue create --blocks 46,47        # not available
gh issue edit 12 --add-blocked-by 8   # not available
gh issue list --blocked               # not available
```

Could wrap via GraphQL API until `gh` catches up.

## The Harder Problem

**Discovering dependencies is harder than tracking them.** You don't know issue A blocks issue B until you're working on B and hit A.

1. **Emergent during work** - discover blockers while working
2. **Cross-repo** - issue in z4 blocks issue in tla2
3. **Implicit** - "can't do X until Y ships" never written down

Possible approaches:
- AI suggests dependencies based on commit history ("touched same files")
- Worker marks blockers as discovered during work
- Manager audits for missing dependencies during rotation

## Why Not Beads

Beads uses local JSONL (`.beads/beads.jsonl`) - parallel system to Issues. Adds merge conflicts, sync problems, duplicate tracking.

GitHub already has the features. Just need CLI access and discovery workflows.

## Action Items

1. Check if GitHub GraphQL API exposes sub-issues and dependencies
2. Wrap in `gh` alias or script if available
3. Design discovery workflow for AIs to mark blockers

## Lineage

Inspired by: Jeffrey Emanuel's Beads Viewer (github.com/Dicklesworthstone/beads_viewer)

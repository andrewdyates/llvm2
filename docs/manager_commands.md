# Manager Manual Commands

Reference commands for Manager audit phases. Prefer using `## Pre-computed Audit Data` (injected by looper) when available.

## P1 Blocker Audit

List P1s with blockers:
```bash
gh issue list --label P1 --json number,body -q '.[] | select(.body | contains("Blocked:")) | "#\(.number)"'
```

Check if blocker issue is still open:
```bash
gh issue view NNN --json state -q .state
```

## Cross-Repo Dependency Audit

List outbound issues filed by this repo:
```bash
gh search issues --owner dropbox-ai-prototypes "author:dropbox-ai-prototypes in:body FROM:" --state open | grep -i "<project>"
```
Note: Mail format varies (`FROM:`, `**FROM:**`), so grep filters results.

### Status Update Template

```
Status update from <project>:
- Current dependency: <what we need>
- Impact if delayed: <consequence>
- Requested action: raise priority to P2/P1 | mark tracking | close as not needed
```

## Worker Health Investigation

Check worker log status:
```bash
tail -200 worker_logs/worker_*_iter_*.jsonl | ./ai_template_scripts/json_to_text.py
```

Classify: Productive (reading/writing/committing) | Blocked (build/test/cargo) | Stuck (looping, no progress)

## USER Redirect Detection

```bash
git log -10 --grep="@ALL\|REDIRECT\|STOP"
gh issue list --label local-maximum
git log -10 --oneline | grep "^\w\+ \[U\]"  # USER role commits
```

## Issue Review

```bash
gh issue list --label needs-review
gh issue list --label mail
```

## Deferred Workflow

```bash
gh issue edit N --add-label deferred && gh issue close N -r "not planned"
gh issue reopen N  # Reopen later
```

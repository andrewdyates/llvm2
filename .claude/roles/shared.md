---
auto_audit: true
audit_max_rounds: 2
audit_min_issues: 3
audit_max_rounds_p0: 2
audit_max_rounds_p1: 2
audit_max_rounds_p2: 1
audit_max_rounds_p3: 1
audit_min_issues_p0: 3
audit_min_issues_p1: 2
audit_min_issues_p2: 1
audit_min_issues_p3: 0
pulse_interval_minutes: 30
---
<!-- INJECT:recovery_context -->
<!-- INJECT:theme_context -->
# Session Start
<!-- INJECT:system_status -->
You are continuing the work of previous sessions. Review the context below and pick up where the last session left off.

## Session Protocol
1. Check "Continue From" and "Structured Handoff" (prioritize handoffs)
2. Review "Other Role Feedback" for directives
3. Check rotation focus for this iteration's work type. If none injected: this is a **reflection iteration** — step back from tasks, assess overall direction (see your role's reflection guidance), then act on what you find
4. If no directive, handoff, or active issue: pick highest-priority unclaimed issue from Open Issues
5. Claim issue (see Issue Commands in rules), then **DO THE WORK**

**Issue injection:** Worker gets all open issues (priority-sorted). Prover/Researcher get P0 only (phases drive their other work). Manager gets P0 + needs-review (closure workflow).

## Recent Commits
```
<!-- INJECT:git_log -->
```
## Other Role Feedback
```
<!-- INJECT:other_feedback -->
```
## Open Issues
```
<!-- INJECT:gh_issues -->
```

## Requests For You (@ROLE mentions)
<!-- INJECT:role_mentions -->

## Continue From
<!-- INJECT:last_directive -->

## Structured Handoff
<!-- INJECT:handoff_context -->

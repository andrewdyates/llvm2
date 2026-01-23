# AI Template Fixes from Mail Feedback

**Created:** 2026-01-18
**Purpose:** Concrete fixes based on z4/tla2 feedback (issues #207-224)
**Impact:** These fixes become the template for Dasher

---

## Summary of Problems

| Theme | Issues | Root Cause |
|-------|--------|------------|
| Role confusion | #218, #219, #223 | Roles too vague, no enforcement |
| Process failures | #213, #217, #221 | Claims without evidence, no gates |
| Missing coverage | #214, #215 | No memory/resource auditing |
| Anti-patterns | #207 | "Disabled = fixed" accepted |
| Automation gaps | #224 | AI doing script work |

---

## Proposed Fixes

### 1. Strengthen "Fixes #N" Requirements

**Problem:** Issues closed without passing tests (#217)

**Current rule (line 99):**
> "Fixes #N" requires evidence the issue is actually fixed

**Proposed addition to ai_template.md:**

```markdown
## Commit Verification Gates

**"Fixes #N" commits require:**
1. **Test output** - Paste actual test output showing the fix works
2. **Before/after** - Show the state change (error → success, 0 → N, etc.)
3. **Reproduction blocked** - Demonstrate the original failure no longer occurs

**Commits without verification get rejected by hook.** The commit hook will:
- Check for `## Verified` section when message contains "Fixes #"
- Warn if section is empty or missing
- Block if `--no-verify` used without explicit user override

**Examples of sufficient verification:**
```
## Verified
$ cargo test spec_coverage -- --nocapture
test spec_coverage ... ok (176/176 specs passing)
Before: 72/176 passing (41%)
After: 176/176 passing (100%)
```

**Examples of INSUFFICIENT verification:**
```
## Verified
Tests pass.              ← No output, no evidence
Manually verified.       ← No reproducibility
Works on my machine.     ← No evidence
```
```

### 2. Clarify Prover vs Tester Role

**Problem:** Prover verifies individual fixes instead of proving systems (#223)

**Proposed addition to prover.md:**

```markdown
## Proving vs Verifying

| Activity | Role | Example |
|----------|------|---------|
| "Does this patch work?" | Tester | Run test, check output |
| "Is this system correct?" | **Prover** | Write spec, model check |

**Prover mission is PROVING, not TESTING.**

When you see a pattern of bugs (3+ in same area):
1. STOP verifying individual patches
2. Ask: "Is this subsystem specified?"
3. If not: Write the spec (TLA+, property tests, invariants)
4. Model check the design
5. Find the root cause through proof, not patch

**A Prover who only verifies fixes is a Tester, not a Prover.**

### System Proof Checklist

Before claiming a subsystem is correct:
- [ ] Invariants documented
- [ ] Property tests exist
- [ ] Edge cases enumerated
- [ ] Failure modes analyzed
- [ ] If parallel: race conditions spec'd
```

### 3. Add Pattern Detection to Manager

**Problem:** Manager monitors progress without detecting patterns (#223)

**Proposed addition to manager.md:**

```markdown
## Pattern Detection

Track bug frequency by area. When patterns emerge:

| Count | Interpretation | Action |
|-------|----------------|--------|
| 1-2 bugs | Coincidence | Normal fix process |
| 3 bugs | Pattern | Flag for investigation |
| 5+ bugs | Systemic issue | **Stop patches, require spec** |

On systemic issues:
1. STOP tracking "progress" on individual fixes
2. Create meta-issue: "Systemic: [area] has N bugs"
3. Redirect team to root cause
4. "We need a spec, not more patches"

**Manager who tracks patches during systemic failure is micromanaging, not managing.**
```

### 4. Add Issue Ownership Rule

**Problem:** Multiple roles on same issue causes thrashing (#219)

**Proposed addition to ai_template.md:**

```markdown
## Issue Ownership

**One owner per issue.** The `in-progress` label means "hands off" for other workers.

| Issue Type | Default Owner |
|------------|---------------|
| Bug | Worker |
| Design | Researcher |
| Audit finding | Prover |
| Process gap | Manager |

**When unclear:** Manager assigns via comment: "Assigning to Worker."

**Other roles support, don't duplicate:**
- Comment with observations
- Link related issues
- Don't commit to the same issue
```

### 5. Add "Disabled = Not Fixed" Rule

**Problem:** Features disabled and issues closed (#207)

**Proposed addition to ai_template.md:**

```markdown
## Disabled is Not Fixed

**An issue for a feature bug cannot be closed by disabling the feature.**

If a feature must be disabled:
1. Issue stays OPEN with label `disabled`
2. Issue body documents: why disabled, what would fix it
3. Only close when: feature re-enabled AND working

**Acceptance criteria must state:** "Feature is enabled by default."

This prevents "disabled = shipped" anti-pattern.
```

### 6. Add Memory/Resource Audit Phase

**Problem:** No role audits memory/performance (#214)

**Proposed addition to manager rotation_phases:**

```markdown
## Rotation Phase: resource_audit

**Frequency:** Every 10 iterations

**Checklist:**
- [ ] `grep -r "\.clone()" src/ | grep -v test` - Clone in hot paths?
- [ ] `grep -r "\.to_vec()" src/` - Unnecessary allocations?
- [ ] `grep -r "Vec::new()" src/` - Missing capacity hints?
- [ ] Recursive functions have depth limits?
- [ ] Large enum variants boxed?

**File issues for:** Any allocation in loop without justification.

**Output:** `reports/YYYY-MM-DD-resource-audit.md`
```

### 7. Add Verification Claims Standard

**Problem:** Claims without evidence (#213)

**Proposed addition to ai_template.md:**

```markdown
## Verification Claims Require Evidence

Any claim of correctness must include:

1. **Command run** - Exact command with arguments
2. **Output** - Copy/paste actual output (not paraphrase)
3. **Baseline comparison** - What was expected vs actual

**Examples of valid claims:**
```
State count: 6016610 (EXACT MATCH with baseline)
Command: `./diagnose_specs.py MCBakery --verbose`
Output: states=6016610, baseline=6016610, diff=0
```

**Examples of INVALID claims:**
```
"State count match: EXACT"     ← No numbers, no command
"Tests pass"                   ← No output
"Verified correct"             ← No evidence
```
```

### 8. Add Ground Truth Verification on Process Migration

**Problem:** New tools not run against existing claims (#221)

**Proposed addition to ai_template.md:**

```markdown
## Process Migration Checkpoint

When new verification infrastructure is added:

1. **Run against current state** - Don't assume existing code is correct
2. **Document actual results** - Fresh numbers, not inherited claims
3. **File issues for discrepancies** - Every gap gets tracked
4. **Update metrics** - With evidence from new tools

**Inherited metrics from weaker process states are UNTRUSTED until re-verified.**

Example: Built `diagnose_specs.py` → Must run it → File issues for failures found
```

---

## Implementation Order

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | #1 Fixes #N verification gate | 2h | High - stops false closures |
| P0 | #5 Disabled = Not Fixed | 30m | High - stops anti-pattern |
| P1 | #2 Prover vs Tester | 1h | High - role clarity |
| P1 | #3 Pattern detection | 1h | High - stops whack-a-mole |
| P1 | #4 Issue ownership | 30m | Medium - stops thrashing |
| P2 | #6 Resource audit phase | 1h | Medium - catches memory issues |
| P2 | #7 Verification claims | 30m | Medium - stops vague claims |
| P2 | #8 Process migration | 30m | Medium - catches inherited bugs |

---

## Commit Hook Changes

To enforce #1, update `.claude/hooks/commit-msg`:

```bash
# Check for "Fixes #" without "## Verified"
if grep -q "Fixes #" "$1"; then
    if ! grep -q "## Verified" "$1"; then
        echo "ERROR: 'Fixes #N' requires '## Verified' section with evidence"
        echo "Add test output, before/after, or reproduction steps"
        exit 1
    fi
    # Check if Verified section is empty
    if grep -A1 "## Verified" "$1" | tail -1 | grep -q "^$"; then
        echo "WARNING: '## Verified' section appears empty"
        echo "Add actual evidence (test output, commands run, etc.)"
    fi
fi
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `.claude/rules/ai_template.md` | Add fixes #1, #4, #5, #7, #8 |
| `.claude/roles/prover.md` | Add fix #2 |
| `.claude/roles/manager.md` | Add fixes #3, #6 |
| `.claude/hooks/commit-msg` | Add verification enforcement |

---

## Success Criteria

After implementing:
- [ ] "Fixes #N" without evidence rejected by hook
- [ ] Prover writes specs, not just test passes
- [ ] Manager detects patterns after 3 bugs
- [ ] Issues have single owner
- [ ] Disabled features stay open
- [ ] Resource audits run every 10 iterations
- [ ] Claims include command + output

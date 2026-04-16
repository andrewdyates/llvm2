# tMIR Stub Migration: Phase-by-Phase Breakdown

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Research Complete
**Part of:** #283 (P0)
**Supersedes:** `designs/2026-04-16-tmir-stub-migration.md` (Wave 34 R1 design)

---

## Executive Summary

The tMIR stub migration is **already 60% complete** as of commit `4fdff72`. The real
`tmir` crate is a git dependency (rev `f9b132a`), stubs are removed from the workspace,
and a `tmir_compat.rs` compatibility layer exists. **135 compilation errors remain**,
all in `adapter.rs` and `compute_graph.rs` in the `llvm2-lower` crate.

**Critical correction to Wave 34 R1 design:** The migration design
(`designs/2026-04-16-tmir-stub-migration.md`) was based on a **hypothetical** real tMIR
that no longer exists. The actual real tMIR crate (`~/tMIR/crates/tmir/`) has a
fundamentally different API than what that design assumed:

| Aspect | Design Assumed | Actual Real tMIR |
|--------|---------------|-----------------|
| Type system | Nested enums (`Type::Primitive(Int{IntWidth})`) | Flat enum (`Ty::I32`, `Ty::Ptr`) |
| Signedness | Collapsed ops (`BinOp::Div`) | Split ops (`BinOp::SDiv`, `BinOp::UDiv`) |
| Comparisons | Unified `CmpOp` (6 variants) | Split `ICmpOp` (10) + `FCmpOp` (12) |
| Instructions | `Instruction` (22 variants) | `Inst` (30+ variants) |
| InstrNode | Embedded results | Separate `.results` vec (like stubs!) |
| Blocks | `HashMap<BlockId, BasicBlock>` | `Vec<Block>` with `id` field (like stubs!) |
| Functions | `HashMap<FuncId, Function>` | `Vec<Function>` (like stubs!) |
| Proofs | None | `ProofAnnotation` enum (21 variants) |

The actual real tMIR is **much closer** to the LLVM2 stubs than the design assumed.
This is excellent news -- the migration is far less disruptive than predicted.

---

## Current State

### What's Done (Wave 34, commit 4fdff72)

1. Cargo.toml updated: `tmir = { git = "...", rev = "f9b132a" }` replaces 4 stub deps
2. Stubs removed from `[workspace.members]` (still on disk but unused)
3. `tmir_compat.rs` created (364 LOC) with:
   - `CmpOp` unified enum bridging `ICmpOp`/`FCmpOp`
   - `Operand` / `OperandConstant` compat types
   - `CallingConv`, `Visibility`, `DataLayout`, `Linkage` (LLVM2 extensions)
   - Re-exports of all `tmir::` types
4. `adapter.rs` partially rewritten (752 lines changed)
5. `compute_graph.rs` partially updated (213 lines changed)

### What Remains (135 errors)

Errors are concentrated in `adapter.rs` and `compute_graph.rs`:

1. **Instruction variant name changes** (~40 errors): `Instr::Cmp` -> `Inst::ICmp`/`Inst::FCmp`,
   `Instr::Alloc` -> `Inst::Alloca`, `Instr::Field` -> `Inst::ExtractField`,
   `Instr::Index` -> `Inst::ExtractElement`
2. **Operand -> ValueId** (~30 errors): Real tMIR uses bare `ValueId` in instructions
   instead of `Operand` enum. Branch args, return values, store values are all `ValueId`.
3. **Call instruction changes** (~15 errors): `Call{func, args, ret_ty}` -> `Call{callee, args}`
   (no explicit ret_ty; use module's func_types table)
4. **Missing stub-only instructions** (~20 errors): `Borrow`, `BorrowMut`, `EndBorrow`,
   `Retain`, `Release`, `IsUnique`, `Nop`, `Dealloc` -- not in real tMIR
5. **Constant model** (~15 errors): `Constant::Int(i128)` (untyped) vs stub's
   `Constant::Int { value, ty }` (typed)
6. **Switch/CondBr structural changes** (~15 errors): `SwitchCase` now has
   `value: Constant` and `args: Vec<ValueId>`; `Switch` has `default_args`

---

## Phase Breakdown

### Phase 1: Fix Remaining 135 Errors (Can Be Done NOW)

**Difficulty:** Mechanical but tedious. No design decisions needed.
**Estimated effort:** 3-5 commits, 1 techlead session
**Blocked by:** Nothing -- just code work

Sub-tasks:
1. Map `Inst::Cmp` references to `Inst::ICmp`/`Inst::FCmp` using the `CmpOp` compat layer
2. Replace `Operand` field accesses with direct `ValueId` usage (branch args, return, etc.)
3. Update `Call`/`CallIndirect` to new field names (`.callee` vs `.func`, no `.ret_ty`)
4. Map `Alloc` -> `Alloca`, `Field` -> `ExtractField`, `Index` -> `ExtractElement`
5. Remove/stub ownership instructions (`Borrow`/`Retain`/`Release` etc.)
6. Update `Constant` usage (untyped `Constant::Int(i128)`)
7. Fix `SwitchCase` and `CondBr`/`Switch` structural changes
8. Fix `compute_graph.rs` for the same changes

**Acceptance criteria:** `cargo check -p llvm2-lower` produces 0 errors.

### Phase 2: Fix Downstream Crates (Can Be Done NOW, After Phase 1)

**Difficulty:** Moderate -- mostly test code updates
**Estimated effort:** 2-3 commits
**Blocked by:** Phase 1

Sub-tasks:
1. Update `llvm2-codegen` tests that construct tMIR types directly
2. Update `llvm2-verify/src/tmir_semantics.rs` for new instruction names
3. Update `llvm2-cli` for any reader/module changes
4. Ensure all 82 adapter tests pass
5. Ensure all 17 integration tests pass

**Acceptance criteria:** `cargo check` clean across entire workspace.

### Phase 3: Delete Stubs and Cleanup (Can Be Done NOW, After Phase 2)

**Difficulty:** Easy
**Estimated effort:** 1 commit
**Blocked by:** Phase 2

Sub-tasks:
1. Delete `stubs/` directory (2,791 LOC)
2. Remove any remaining references to stub crates in comments/docs
3. Remove unused compat shims that are no longer needed
4. Update CLAUDE.md architecture description

**Acceptance criteria:** `stubs/` gone, no references to old crate names.

### Phase 4: Upstream LLVM2 Extensions to Real tMIR (Needs Upstream Work)

**Difficulty:** Cross-repo coordination
**Estimated effort:** 5-10 commits in tMIR repo
**Blocked by:** tMIR repo access and agreement on API

LLVM2-only extensions that should eventually live in real tMIR:

| Extension | LLVM2 Location | Why Upstream |
|-----------|---------------|-------------|
| `CallingConv` | `tmir_compat.rs` | Every backend needs calling conventions |
| `Visibility` | `tmir_compat.rs` | Needed for correct object file emission |
| `DataLayout` | `tmir_compat.rs` | Target-specific layout is universal |
| `Linkage` | `tmir_compat.rs` | Symbol linkage is needed by all linkers |
| `GlobalDef` enrichment | `tmir_compat.rs` | Real tMIR `Global` is minimal |
| Vector/SIMD type | NOT YET NEEDED | Real tMIR uses `Ty::Array`; SIMD is backend concern |

**Note:** The real tMIR already has `ProofAnnotation` (21 variants), `StructDef`/`FieldDef`,
`FuncTy`, `InstrNode` with results+proofs, atomics, `Select`, `GEP`, and `Const`. The
Wave 34 R1 design incorrectly listed many of these as "missing from real tMIR."

### Phase 5: tRust Bridge (Needs Upstream Work)

**Difficulty:** Major cross-repo effort
**Estimated effort:** 10-20 commits across tRust and LLVM2
**Blocked by:** #266 (trust-llvm2-bridge), #284 (pub(crate) API)

This is the final step: making tRust actually produce tMIR that LLVM2 can consume.
Currently tracked by existing issues:
- #266: trust-llvm2-bridge crate
- #284: trust-mir-extract pub(crate) blocker
- #259: tRust Llvm2Backend stub coordination

---

## Revised Findings vs Wave 34 R1 Design

The Wave 34 R1 design (`designs/2026-04-16-tmir-stub-migration.md`) contains several
**factual errors** based on a different (possibly earlier) version of the tMIR repo:

1. **WRONG:** "Real tMIR has no proof annotations." **FACT:** Real tMIR has
   `ProofAnnotation` with 21 variants plus `ProofObligation`, `ProofCertificate`.
2. **WRONG:** "Real tMIR has no atomics." **FACT:** Real tMIR has `AtomicLoad`,
   `AtomicStore`, `AtomicRMW`, `CmpXchg`, `Fence` with `Ordering` and `AtomicRMWOp`.
3. **WRONG:** "Real tMIR has no Select." **FACT:** Real tMIR has `Inst::Select`.
4. **WRONG:** "Real tMIR has no GEP." **FACT:** Real tMIR has `Inst::GEP`.
5. **WRONG:** "Real tMIR uses HashMap for blocks/functions." **FACT:** Real tMIR
   uses `Vec<Block>` and `Vec<Function>`, same as stubs.
6. **WRONG:** "Real tMIR embeds result inside each instruction." **FACT:** Real tMIR
   uses `InstrNode { inst, results, proofs, span }` -- separate results vec.
7. **WRONG:** "BinOp reduced to 10 variants." **FACT:** Real tMIR `BinOp` has 18
   variants including `SDiv`/`UDiv`/`SRem`/`URem`/`LShr`/`AShr` and `FRem`.
8. **WRONG:** "CmpOp reduced to 6 variants." **FACT:** Split into `ICmpOp` (10) +
   `FCmpOp` (12) = 22 variants total.

The R1 design was likely based on an intermediate tMIR version. The current tMIR
(rev `f9b132a`) is much more feature-complete and much closer to the LLVM2 stubs.

---

## Wave 34 Quality Assessment

Wave 34 merged 8 agents. Key observations:

### Positive
- `tmir_compat.rs` is well-designed and correct
- Cargo.toml migration was done cleanly
- adapter.rs bulk rename was systematic
- Exception handling instructions (Invoke/LandingPad/Resume) added to stubs
- Binary tMIR bitcode encoder/decoder added
- z4 solver preference switch completed
- Quantified logic (ABV) support for memory proofs

### Gaps
- **135 compilation errors remain** in llvm2-lower (expected -- WIP commit)
- Wave 34 R1 design doc contains factual errors about real tMIR API (documented above)
- Issue #288 (multiblock E2E hangs at O1+) is a regression that needs investigation
- The stubs are still physically on disk despite being removed from workspace

### Regressions
- **#288 (P1):** Multiblock E2E tests hang at O1+ optimization level. This could be an
  infinite loop in the optimization pipeline. Not directly related to stub migration but
  was filed during Wave 34 and may have been introduced by the wave's changes.

---

## Recommended Issue Structure

| Issue | Phase | Priority | Summary |
|-------|-------|----------|---------|
| New | 1 | P1 | Fix 135 compilation errors in llvm2-lower adapter |
| New | 2 | P1 | Fix downstream crate compilation (codegen, verify, cli) |
| New | 3 | P2 | Delete stubs/ directory and clean up |
| #267 | 4 | P2 | Upstream LLVM2 extensions to real tMIR |
| #266 | 5 | P1 | trust-llvm2-bridge crate |
| #284 | 5 | P2 | trust-mir-extract pub(crate) blocker |
| #259 | 5 | P1 | Coordinate backend contract with tRust |

---

## Key Files

| File | Role | Status |
|------|------|--------|
| `Cargo.toml` | Workspace deps | Done -- real tmir git dep |
| `crates/llvm2-lower/src/tmir_compat.rs` | Compat layer | Done (364 LOC) |
| `crates/llvm2-lower/src/adapter.rs` | Primary consumer | 135 errors remain |
| `crates/llvm2-lower/src/compute_graph.rs` | Graph analysis | Errors remain |
| `stubs/` | Old stubs | Remove from workspace, delete in Phase 3 |
| `designs/2026-04-16-tmir-stub-migration.md` | R1 design | Contains errors; see corrections above |

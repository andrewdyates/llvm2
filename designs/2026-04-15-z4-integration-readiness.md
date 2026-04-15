# z4 Integration Readiness Assessment

**Date:** 2026-04-15
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Research Complete
**Part of:** #34 (Phase 9: z4 verification integration), #236 (z4 native API theory gaps)

---

## Executive Summary

LLVM2's verification infrastructure has two backends for proving lowering correctness:

1. **Mock evaluation** (default): Exhaustive for 8-bit, random sampling for 32/64-bit.
   Fast but NOT formal verification -- it can miss edge-case bugs.
2. **z3 CLI subprocess** (available when z3 installed): Real SMT solving via
   `verify_with_cli()`. Fully formal. Currently used in 7 batch test functions
   covering ~82 proofs (arithmetic, NZCV, comparison, branch, peephole, load/store, bitwise).
3. **z4 native Rust API** (feature-gated, NOT functional): `verify_with_z4_api()`
   exists but the z4 crate dependency is commented out. Even if enabled, it only
   supports QF_BV (bitvectors) -- no arrays, FP, or UF.

**Bottom line:** z3 CLI subprocess is the realistic path to making SMT the default
backend. The z4 native API is a performance optimization for later. The biggest
blockers are NOT solver availability -- they are (1) expanding batch verification
to all proof categories, (2) performance testing at scale, and (3) CI integration.

---

## 1. Current Backend Status

### 1.1 Mock Evaluation (default)

| Aspect | Status |
|--------|--------|
| 8-bit proofs | Exhaustive (all 256 values) |
| 16-bit proofs | Exhaustive (65536 values) |
| 32/64-bit proofs | Random sampling (10K-100K samples) |
| Theory support | BV only (no arrays, FP, UF) |
| Soundness | NOT sound -- can miss counterexamples |
| Performance | ~1ms per proof (fast) |

**Why mock exists:** It was the bootstrap mechanism. Before z3 was integrated,
mock evaluation provided a useful sanity check. Now that z3 CLI works, mock
should be downgraded from "verification" to "fast smoke test."

### 1.2 z3 CLI Subprocess

| Aspect | Status |
|--------|--------|
| Installation | z3 4.15.4 available via `brew install z3` |
| Theory support | QF_BV, QF_ABV, QF_BVFP, QF_UFBV, ALL |
| Soundness | Sound and complete for decidable theories |
| SMT-LIB2 generation | Complete: `generate_smt2_query()` with auto logic inference |
| Model parsing | Hex, binary, decimal bitvector formats |
| Timeout handling | 5000ms default, configurable |
| Counterexample extraction | Working: parses `(get-value ...)` output |
| Batch tests passing | 7/7 batch test groups pass with z3 4.15.4 |

**Performance (measured 2026-04-15 on Apple M-series):**
- Trivial proofs (identity, commutativity): <50ms each
- Arithmetic proofs (add/sub/mul i32/i64): <100ms each
- Memory proofs (store-load roundtrip, QF_ABV): <200ms each
- Full batch (82 proofs): ~5-8 seconds total

### 1.3 z4 Native Rust API

| Aspect | Status |
|--------|--------|
| Crate dependency | COMMENTED OUT in Cargo.toml (blocks builds when z4 repo unavailable) |
| Feature gate | `z4` feature exists, triggers `compile_error!` since dep is missing |
| BV translation | Complete: all BV ops mapped to z4 API |
| Array translation | NOT implemented: returns `Err("Array theory not yet supported")` |
| FP translation | NOT implemented: returns `Err("FP theory not yet supported")` |
| UF translation | NOT implemented: returns `Err("UF theory not yet supported")` |
| z4 API readiness | CONFIRMED: z4-bindings exposes all needed theories (20K+ LOC) |

---

## 2. Blockers to Making z3/z4 the Default

### Blocker 1: Incomplete Batch Coverage (P1)

`verify_all_with_z4()` only collects arithmetic + NZCV + peephole proofs (~30).
The full proof suite has 100+ obligations across all categories. Missing:

- Comparison proofs (20)
- Branch proofs (20)
- Load/store proofs (6+)
- Bitwise/shift proofs (7+)
- CFG simplification proofs
- Constant folding proofs
- Copy propagation proofs
- CSE/LICM proofs
- DCE proofs

**Filed as:** #239

**Effort:** Medium (1-2 sessions). Each proof module already exports
`all_*_proofs()` functions. The batch verifier just needs to call them.

### Blocker 2: z3 Not Required in CI (P1)

There is no CI pipeline (per CLAUDE.md rules), but the batch tests silently
skip when z3 is not installed (`if solver.is_empty() { return; }`). This means
a developer without z3 sees all tests pass but has NOT verified anything.

**Options:**
1. **Feature-gate z3 tests:** `#[cfg(feature = "z3-tests")]` -- explicit opt-in
2. **Fail loudly:** `assert!(!solver.is_empty(), "z3 required for verification")` --
   makes z3 a hard dependency for `cargo test`
3. **Add `LLVM2_REQUIRE_SOLVER` env var:** Middle ground -- skip by default but
   fail if explicitly requested

**Recommendation:** Option 3 with a note in README. z3 is trivially installable
(`brew install z3`).

### Blocker 3: Performance at Scale (P2)

82 proofs take ~5-8 seconds. The full proof suite (100+ proofs) would take
~10-15 seconds. This is acceptable for `cargo test` but may be slow for
continuous validation during development.

**Mitigations:**
- Run z3 proofs only in `--release` mode (faster SMT solving)
- Parallelize proof checking (each proof is independent)
- Use z4 native API to avoid subprocess overhead (when available)
- Cache proof results (if inputs haven't changed, skip re-verification)

### Blocker 4: z4 Crate Accessibility (P3)

The z4 crate is behind a git dep that is frequently inaccessible. Cargo resolves
ALL git deps (even optional) at build time, blocking compilation.

**Mitigation:** The CLI fallback works. Do not gate the default backend on z4
native API. Use z3 CLI for the default, z4 native as an optional speedup.

---

## 3. Recommended Path

### Phase A: Expand z3 CLI Coverage (immediate, no blockers)

1. Expand `verify_all_with_z4()` to include ALL proof categories (#239)
2. Add `LLVM2_REQUIRE_SOLVER` env var for strict mode
3. Run full batch with z3 4.15.4, report results
4. Add `test_z4_batch_verify_ALL()` that fails if any proof fails

### Phase B: Make z3 the Primary Backend (next milestone)

1. Change the default verification path in `verify.rs` and `lowering_proof.rs`:
   - If z3 available: use z3 CLI (sound, complete)
   - If z3 NOT available: fall back to mock (with warning)
2. Mark all mock-only proofs as "unverified" in proof_database.rs
3. Add proof result tracking: a `verification_status.json` that records which
   proofs have been z3-verified vs mock-only

### Phase C: z4 Native API (when z4 crate is accessible)

1. Re-enable z4 git dep in Cargo.toml
2. Implement array theory translation (#236)
3. Implement FP theory translation (#236)
4. Implement UF theory translation (#236)
5. Benchmark: z4 native vs z3 CLI per proof
6. If z4 is faster, make it the default when feature is enabled

---

## 4. Missing Features for Full SMT Verification

| Feature | Needed For | Status |
|---------|-----------|--------|
| Quantifier support (ForAll/Exists) | Some optimization proofs | SMT-LIB2 generation handles it; z4 native does not |
| Incremental solving | Batch verification speed | Not implemented; each proof creates fresh solver |
| Proof certificates (UNSAT cores) | Debugging failed proofs | z3 supports it; not wired into LLVM2 |
| Parallel solving | Scale to 500+ proofs | Not implemented; straightforward with rayon |
| Counterexample-guided widening | Convert 8-bit proofs to 32/64-bit | CEGIS module exists but not wired to z3 |

---

## 5. Test Count and Proof Category Summary

### By Crate (total: 5,285 tests)

| Crate | Tests |
|-------|-------|
| llvm2-verify | 1,853 |
| llvm2-codegen | 1,530 |
| llvm2-lower | 695 |
| llvm2-opt | 433 |
| llvm2-regalloc | 390 |
| llvm2-ir | 384 |

### Proof Modules in llvm2-verify (1,853 tests across 47 files)

Top 10 by test count:
1. lowering_proof.rs: 133
2. unified_synthesis.rs: 141 (CEGIS)
3. z4_bridge.rs: 99
4. smt.rs: 118
5. memory_proofs.rs: 91
6. neon_semantics.rs: 64
7. call_lowering_proofs.rs: 48
8. const_fold_proofs.rs: 47
9. regalloc_proofs.rs: 46
10. synthesis.rs: 44

Bottom 5 (under-tested, filed as #238):
1. dce_proofs.rs: 16
2. frame_proofs.rs: 17
3. fp_convert_proofs.rs: 18
4. verify.rs: 14
5. verification_runner.rs: 14

### z3 Batch Test Coverage

| Batch Test | Proof Count | Status |
|-----------|-------------|--------|
| Arithmetic (add/sub/mul/neg I8-I64, div) | 16+ | PASSING |
| NZCV flags (N/Z/C/V for i32 add) | 4 | PASSING |
| Comparisons (i32: 10, i64: 10) | 20 | PASSING |
| Conditional branches (i32+i64) | 20 | PASSING |
| Peephole identities | 9+ | PASSING |
| Load/store (array theory QF_ABV) | 6+ | PASSING |
| Bitwise/shift | 7+ | PASSING |
| **Total in batch tests** | **~82** | **ALL PASSING** |
| **Not in batch tests** | **~100+** | NOT TESTED via z3 |

---

## References

- `crates/llvm2-verify/src/z4_bridge.rs` -- Full bridge implementation (2467 LOC)
- `crates/llvm2-verify/Cargo.toml` -- z4 feature gate and commented-out dep
- `designs/2026-04-14-z4-integration-guide.md` -- z4 API capabilities
- `designs/2026-04-14-z4-api-audit.md` -- z4 API audit
- Issue #34: Phase 9: z4 verification integration
- Issue #236: z4 native API theory gaps
- Issue #239: Incomplete z3 batch verification

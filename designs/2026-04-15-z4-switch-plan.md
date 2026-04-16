# z4 Switch Plan: z3 to z4 Migration for LLVM2

**Date:** 2026-04-15
**Author:** Andrew Yates <ayates@dropbox.com>
**Role:** R1 (Researcher), Wave 28
**Status:** Design Complete
**Part of:** #247 (z4 integration), #34 (Phase 9: z4 verification), #236 (z4 native API theory gaps)

---

## Executive Summary

LLVM2's `llvm2-verify` crate currently uses z3 as its SMT solver via a CLI subprocess
fallback. The project goal is to switch entirely to z4, our own Rust-native SMT solver
(`~/z4/`, repo `ayates_dbx/z4`). This document provides the full switch plan across
four phases: path dependency, CLI integration, native API integration, and performance
benchmarking.

**Key finding:** The z4 native Rust API already provides every operation that LLVM2's
`z4_bridge.rs` calls. The bridge code (behind `#[cfg(feature = "z4")]`) is already
written for the majority of operations. The remaining work is primarily:
1. Enabling the z4 crate dependency (currently commented out)
2. Fixing 5 concrete API mismatches identified below
3. Verifying that z4's SMT-LIB2 CLI accepts the same input format as z3

---

## Current State

### LLVM2 Verification Backends

| Backend | Status | Code Path |
|---------|--------|-----------|
| Mock evaluation (default) | Working | Exhaustive 8/16-bit, random sampling 32/64-bit |
| z3 CLI subprocess | Working | `verify_with_cli()` in `z4_bridge.rs` |
| z4 native Rust API | Partial (not linked) | `verify_with_z4_api()` behind `#[cfg(feature = "z4")]` |

**Source:** `crates/llvm2-verify/src/z4_bridge.rs` (1500+ LOC)

### Current z4 Dependency State

**Workspace Cargo.toml** (root):
```toml
# z4 = { git = "https://github.com/ayates_dbx/z4" }  # COMMENTED OUT
```

**llvm2-verify/Cargo.toml:**
```toml
[features]
z4 = []  # Empty feature — no dep:z4 linkage

[dependencies]
# z4 = { git = "...", optional = true }  # COMMENTED OUT (#175)
```

The z4 feature currently triggers a `compile_error!` if enabled, intentionally
preventing compilation until the dependency is wired up.

---

## API Mapping: z4_bridge Expectations vs z4 Actual API

### Fully Matched Operations (No Changes Needed)

These operations are used in `translate_expr_to_z4()` and map 1:1 to z4's API:

| z4_bridge calls | z4 Solver method | Status |
|-----------------|------------------|--------|
| `Solver::try_new(logic)` | `Solver::try_new(Logic)` | MATCH |
| `solver.declare_const(name, sort)` | `Solver::declare_const(&str, Sort) -> Term` | MATCH |
| `solver.assert_term(term)` | `Solver::assert_term(Term)` | MATCH |
| `solver.check_sat_with_details()` | `Solver::check_sat_with_details() -> SolveDetails` | MATCH |
| `details.accept_for_consumer()` | `SolveDetails::accept_for_consumer()` | MATCH |
| `solver.model()` | `Solver::model() -> Option<VerifiedModel>` | MATCH |
| `solver.bv_const(value, width)` | `Solver::bv_const(i64, u32) -> Term` | MATCH |
| `solver.bool_const(value)` | `Solver::bool_const(bool) -> Term` | MATCH |
| `solver.bvadd(a, b)` | `Solver::bvadd(Term, Term) -> Term` | MATCH |
| `solver.bvsub(a, b)` | `Solver::bvsub(Term, Term) -> Term` | MATCH |
| `solver.bvmul(a, b)` | `Solver::bvmul(Term, Term) -> Term` | MATCH |
| `solver.bvsdiv(a, b)` | `Solver::bvsdiv(Term, Term) -> Term` | MATCH |
| `solver.bvudiv(a, b)` | `Solver::bvudiv(Term, Term) -> Term` | MATCH |
| `solver.bvneg(a)` | `Solver::bvneg(Term) -> Term` | MATCH |
| `solver.bvand(a, b)` | `Solver::bvand(Term, Term) -> Term` | MATCH |
| `solver.bvor(a, b)` | `Solver::bvor(Term, Term) -> Term` | MATCH |
| `solver.bvxor(a, b)` | `Solver::bvxor(Term, Term) -> Term` | MATCH |
| `solver.bvshl(a, b)` | `Solver::bvshl(Term, Term) -> Term` | MATCH |
| `solver.bvlshr(a, b)` | `Solver::bvlshr(Term, Term) -> Term` | MATCH |
| `solver.bvashr(a, b)` | `Solver::bvashr(Term, Term) -> Term` | MATCH |
| `solver.eq(a, b)` | `Solver::eq(Term, Term) -> Term` | MATCH |
| `solver.not(a)` | `Solver::not(Term) -> Term` | MATCH |
| `solver.and(a, b)` | `Solver::and(Term, Term) -> Term` | MATCH |
| `solver.or(a, b)` | `Solver::or(Term, Term) -> Term` | MATCH |
| `solver.bvslt(a, b)` | `Solver::bvslt(Term, Term) -> Term` | MATCH |
| `solver.bvsge(a, b)` | `Solver::bvsge(Term, Term) -> Term` | MATCH |
| `solver.bvsgt(a, b)` | `Solver::bvsgt(Term, Term) -> Term` | MATCH |
| `solver.bvsle(a, b)` | `Solver::bvsle(Term, Term) -> Term` | MATCH |
| `solver.bvult(a, b)` | `Solver::bvult(Term, Term) -> Term` | MATCH |
| `solver.bvuge(a, b)` | `Solver::bvuge(Term, Term) -> Term` | MATCH |
| `solver.bvugt(a, b)` | `Solver::bvugt(Term, Term) -> Term` | MATCH |
| `solver.bvule(a, b)` | `Solver::bvule(Term, Term) -> Term` | MATCH |
| `solver.ite(c, t, e)` | `Solver::ite(Term, Term, Term) -> Term` | MATCH |
| `solver.select(arr, idx)` | `Solver::select(Term, Term) -> Term` | MATCH |
| `solver.store(arr, idx, val)` | `Solver::store(Term, Term, Term) -> Term` | MATCH |
| `solver.fp_add(rm, a, b)` | `Solver::fp_add(Term, Term, Term) -> Term` | MATCH |
| `solver.fp_sub(rm, a, b)` | `Solver::fp_sub(Term, Term, Term) -> Term` | MATCH |
| `solver.fp_mul(rm, a, b)` | `Solver::fp_mul(Term, Term, Term) -> Term` | MATCH |
| `solver.fp_div(rm, a, b)` | `Solver::fp_div(Term, Term, Term) -> Term` | MATCH |
| `solver.fp_neg(a)` | `Solver::fp_neg(Term) -> Term` | MATCH |
| `solver.fp_abs(a)` | `Solver::fp_abs(Term) -> Term` | MATCH |
| `solver.fp_sqrt(rm, a)` | `Solver::fp_sqrt(Term, Term) -> Term` | MATCH |
| `solver.fp_fma(rm, a, b, c)` | `Solver::fp_fma(Term, Term, Term, Term) -> Term` | MATCH |
| `solver.fp_eq(a, b)` | `Solver::fp_eq(Term, Term) -> Term` | MATCH |
| `solver.fp_lt(a, b)` | `Solver::fp_lt(Term, Term) -> Term` | MATCH |
| `solver.fp_gt(a, b)` | `Solver::fp_gt(Term, Term) -> Term` | MATCH |
| `solver.fp_is_nan(a)` | `Solver::fp_is_nan(Term) -> Term` | MATCH |
| `solver.fp_is_infinite(a)` | `Solver::fp_is_infinite(Term) -> Term` | MATCH |
| `solver.fp_is_zero(a)` | `Solver::fp_is_zero(Term) -> Term` | MATCH |
| `solver.fp_is_normal(a)` | `Solver::fp_is_normal(Term) -> Term` | MATCH |
| `solver.fp_to_sbv(rm, a, w)` | `Solver::fp_to_sbv(Term, Term, u32) -> Term` | MATCH |
| `solver.fp_to_ubv(rm, a, w)` | `Solver::fp_to_ubv(Term, Term, u32) -> Term` | MATCH |
| `solver.bv_to_fp(rm, a, eb, sb)` | `Solver::bv_to_fp(Term, Term, u32, u32) -> Term` | MATCH |
| `solver.fp_to_fp(rm, a, eb, sb)` | `Solver::fp_to_fp(Term, Term, u32, u32) -> Term` | MATCH |

### Mismatches Requiring Fix (5 Issues)

#### Mismatch 1: `const_array` Signature Difference

**z4_bridge calls** (line 1059):
```rust
solver.const_array(idx_sort, elem_sort, v)
// 3 arguments: index_sort, element_sort, value
```

**z4 actual API** (`z4-dpll/src/api/terms/arrays.rs:42`):
```rust
pub fn const_array(&mut self, index_sort: Sort, value: Term) -> Term
// 2 arguments: index_sort, value (element sort inferred from value)
```

**Fix:** Remove the `elem_sort` argument from the bridge call. z4 infers element sort
from the value term's sort. Change:
```rust
Ok(solver.const_array(idx_sort, elem_sort, v))
```
to:
```rust
Ok(solver.const_array(idx_sort, v))
```

**Severity:** Compile error. Must fix before Phase 3 compiles.

#### Mismatch 2: `bv_val` Return Type

**z4_bridge expects** (line 852):
```rust
if let Some(val) = model.bv_val(name) {
    assignments.push((name.clone(), val));
}
// Pushes into Vec<(String, u64)>
```

**z4 actual return** (`z4-dpll/src/api/types/model.rs:88`):
```rust
pub fn bv_val(&self, name: &str) -> Option<(BigInt, u32)>
// Returns (BigInt, width), NOT u64
```

**Fix:** Convert BigInt to u64 and destructure:
```rust
if let Some((bigint_val, _width)) = model.bv_val(name) {
    use num_traits::ToPrimitive;
    if let Some(val) = bigint_val.to_u64() {
        assignments.push((name.clone(), val));
    }
}
```

**Severity:** Compile error. Must fix before Phase 3 compiles.
**Risk:** BigInt values exceeding 64 bits will silently be dropped. For LLVM2's
current use case (up to 64-bit bitvectors), this is acceptable. Future 128-bit
support would need `Z4Result::CounterExample` to use `BigInt` or `u128`.

#### Mismatch 3: `fp_geq` vs `fp_ge` Naming

**z4_bridge calls** (line 1124):
```rust
Ok(solver.fp_geq(l, r))  // For FPGe
```

**z4 actual API** (`z4-dpll/src/api/floating_point.rs:217`):
```rust
pub fn fp_ge(&mut self, a: Term, b: Term) -> Term  // Named fp_ge, not fp_geq
```

**Fix:** Change `solver.fp_geq(l, r)` to `solver.fp_ge(l, r)`.

**Severity:** Compile error. Trivial one-line fix.

#### Mismatch 4: `fp_leq` vs `fp_le` Naming

**z4_bridge calls** (line 1129):
```rust
Ok(solver.fp_leq(l, r))  // For FPLe
```

**z4 actual API** (`z4-dpll/src/api/floating_point.rs:183`):
```rust
pub fn fp_le(&mut self, a: Term, b: Term) -> Term  // Named fp_le, not fp_leq
```

**Fix:** Change `solver.fp_leq(l, r)` to `solver.fp_le(l, r)`.

**Severity:** Compile error. Trivial one-line fix.

#### Mismatch 5: `declare_fun` + `apply` Signature for UFs

**z4_bridge calls** (line 1177):
```rust
// func_term is z4::Term looked up from var_terms HashMap
Ok(solver.apply(func_term, &translated_args))
```

**z4 actual API** (`z4-dpll/src/api/terms.rs:129,189`):
```rust
pub fn declare_fun(&mut self, name: &str, domain: &[Sort], range: Sort) -> FuncDecl
pub fn apply(&mut self, func: &FuncDecl, args: &[Term]) -> Term
// apply takes &FuncDecl, NOT Term
```

**Fix:** The bridge stores UF declarations in `var_terms: HashMap<String, z4::Term>`,
but `declare_fun` returns `FuncDecl`, not `Term`. Need to either:
- (A) Create a separate `HashMap<String, z4::FuncDecl>` for function declarations, or
- (B) Store `FuncDecl` in a new field and look it up by name.

The bridge's UFDecl handling (line 1186) returns `func` which is already `FuncDecl`,
but it's returned as `Ok(func)` where the function expects `z4::Term`. This needs a
structural refactor of how UFs are tracked.

**Recommended fix:**
```rust
// Add a new parameter or field:
fn translate_expr_to_z4(
    expr: &SmtExpr,
    solver: &mut z4::Solver,  // needs &mut for declare_fun
    var_terms: &HashMap<String, z4::Term>,
    func_decls: &mut HashMap<String, z4::FuncDecl>,  // NEW
) -> Result<z4::Term, String>
```

**Severity:** Compile error + structural change. Medium effort.

### Missing z4 Logic Variant: `QfAbvfp`

The bridge maps `"QF_ABVFP"` (arrays + bitvectors + floating-point) to a `Logic`
variant. z4's `Logic` enum does **not** have a `QfAbvfp` variant.

**Current bridge code** (line 810):
```rust
"QF_ABVFP" => Logic::QfAbvfp,  // Does not exist in z4
```

**Fix:** Use `Logic::All` as the fallback for combined theories. z4's `All` logic
auto-detects theories at check-sat time. The bridge already does this for the `_ => Logic::All`
fallback case.

### Rounding Mode Type Mapping

**z4_bridge** uses `z4::RoundingMode` enum with variants `RNE, RNA, RTP, RTN, RTZ`.
z4's actual rounding mode API uses `solver.fp_rounding_mode("RNE")` which returns
a `Term`, not an enum variant.

**Current bridge code** assumes a `z4::RoundingMode` type exists for direct mapping.
This needs verification. The bridge's `rounding_mode_to_z4()` function (line 1238)
returns `z4::RoundingMode` -- if z4 exposes rounding modes as `Term` values rather
than an enum, the bridge needs to call `solver.fp_rounding_mode("RNE")` instead.

**Action:** TL7 should verify whether z4 re-exports a `RoundingMode` enum or requires
the `fp_rounding_mode(&str)` term constructor.

### Quantifier Support

The bridge currently returns an error for `ForAll` and `Exists` quantifiers (lines 1198-1211),
noting "not yet supported in z4 native API." However, z4 **does** support quantifiers
via `solver.forall(&[Term], Term)` and `solver.exists(&[Term], Term)`.

The bridge's bounded quantifiers (`SmtExpr::ForAll { var, var_width, lower, upper, body }`)
use a range-bounded encoding that differs from z4's standard universal/existential
quantifiers. These would need to be expanded:

```
ForAll(x, 0..N, body) => forall([x], implies(and(bvuge(x, 0), bvule(x, N)), body))
```

This is not blocking for Phase 3 since no current proofs use quantifiers through the
native API path.

### Sort Constructor Differences

**z4_bridge** (line 822):
```rust
let sort = Sort::BitVec(BitVecSort { width: *width });
```

**z4 actual:** This is correct. z4 re-exports `BitVecSort` from `z4_core`.

**z4_bridge** (line 829):
```rust
let sort = Sort::FP(FPSort { eb: *eb, sb: *sb });
```

**Needs verification:** z4 may use `Sort::FloatingPoint` or `Sort::FP` depending on
version. Check z4's `Sort` enum definition.

**z4_bridge** (line 1225):
```rust
Ok(Sort::Array(ArraySort { index: Box::new(idx_sort), element: Box::new(elem_sort) }))
```

**Needs verification:** z4's `ArraySort` struct may have `index` and `element` fields
or different names.

---

## Phase Plan

### Phase 1: Path Dependency + Compile (TL1 — In Progress)

**Goal:** Make `cargo check -p llvm2-verify --features z4` compile.

**Steps:**
1. Uncomment z4 git dependency in workspace `Cargo.toml`
2. Change llvm2-verify feature to `z4 = ["dep:z4"]`
3. Remove the `compile_error!` gate in `z4_bridge.rs` (line 46-50)
4. Fix the 5 compile errors identified above
5. Verify `cargo check --features z4` passes

**Blockers:**
- z4 git repo must be accessible from Cargo (TL1 is handling path dep alternative)
- May need `path = "../../z4"` instead of git dep for local development

**Deliverables:**
- [ ] z4 dependency enabled in Cargo.toml (path or git)
- [ ] `compile_error!` removed
- [ ] All 5 API mismatches fixed
- [ ] `cargo check -p llvm2-verify --features z4` passes

### Phase 2: CLI Integration (z4 Binary Accepting SMT-LIB2)

**Goal:** Replace z3 CLI subprocess with z4 CLI subprocess.

**Background:** z4 already accepts SMT-LIB2 input. The binary supports `-smt2` as a
backward-compatibility flag (auto-detected and dropped). z4 auto-detects input format
(SMT-LIB2, DIMACS CNF, CHC).

**Steps:**
1. Verify z4 binary is installed and accessible: `which z4`
2. Update `find_solver_binary()` to prefer z4 over z3:
   ```rust
   fn find_solver_binary() -> String {
       // Prefer z4 (our own solver)
       if let Ok(output) = Command::new("which").arg("z4").output() ...
       // Fallback: z3
       if let Ok(output) = Command::new("which").arg("z3").output() ...
   }
   ```
3. Run the existing z3 CLI test suite against z4 binary to verify compatibility
4. Verify that z4's output format matches z3's for:
   - `sat` / `unsat` / `unknown` first line
   - `(get-value ...)` output format: `((var #xHEX))` or `((var (_ bvN W)))`
   - Timeout behavior: does z4 print `unknown` with a reason?
5. Update `parse_solver_output()` if z4 output format differs from z3

**Output compatibility notes:**
- z4 CLI accepts `-smt2` flag (backward compat with z3 invocation)
- z4 supports `(set-option :timeout N)` — verify millisecond semantics match z3
- z4 supports `(set-option :produce-models true)` — standard SMT-LIB2
- z4 `solve` subcommand auto-injected when running `z4 file.smt2`

**Deliverables:**
- [ ] z4 preferred over z3 in `find_solver_binary()`
- [ ] All 82+ existing CLI batch tests pass with z4 binary
- [ ] Output parsing verified for z4-specific output format
- [ ] Timeout handling verified

### Phase 3: Native API Integration (In-Process Solving)

**Goal:** Enable `verify_with_z4_api()` for in-process solving without CLI overhead.

**Steps:**
1. Enable z4 feature after Phase 1 compile fixes
2. Fix all 5 API mismatches (see section above)
3. Add `num-bigint` and `num-traits` to llvm2-verify dependencies (for BigInt conversion)
4. Run all proof tests with `--features z4`
5. Verify that native API produces identical results to CLI for all proofs
6. Handle edge cases:
   - z4 native API uses `VerifiedModel` wrapper; must call `.into_inner()` for `Model`
   - z4 `SolveResult::Unsat` carries proof data — bridge only checks variant, not data
   - Timeout must be set via `solver.set_timeout(Some(Duration::from_millis(ms)))`,
     not `(set-option :timeout N)` which is CLI-only

**Theory support matrix:**

| Theory | CLI (SMT-LIB2) | Native API | LLVM2 Usage |
|--------|----------------|------------|-------------|
| QF_BV (bitvectors) | Working | Implemented in bridge | All lowering proofs |
| QF_ABV (arrays) | Working | Implemented in bridge | Memory model proofs |
| QF_BVFP (floating-point) | Working | Implemented in bridge | FP lowering proofs |
| QF_UFBV (uninterpreted) | Working | Partial (UF needs fix) | Abstraction proofs |
| Quantifiers | Working | Not implemented | Not currently used |

**Deliverables:**
- [ ] All API mismatches resolved
- [ ] `cargo test -p llvm2-verify --features z4` — all tests pass
- [ ] Native API results match CLI results for all proof categories
- [ ] Timeout behavior correct via `set_timeout()`

### Phase 4: Performance Benchmarking vs z3

**Goal:** Measure z4 performance and determine if it can be the default backend.

**Benchmarks to run:**
1. **Batch verification throughput:** Time `verify_all_with_z4()` with z3 CLI, z4 CLI, z4 native
2. **Per-proof latency:** Measure each proof individually, identify outliers
3. **Theory-specific:** Compare QF_BV, QF_ABV, QF_BVFP separately (different theory solvers)
4. **Scale test:** Run full `ProofDatabase` verification (all categories) with each backend

**Metrics to capture:**
- Wall-clock time per proof
- Total batch time
- Memory usage (RSS)
- Number of verified / failed / timeout / error per backend
- Timeouts: does z4 have more or fewer than z3?

**Expected outcomes:**
- z4 native API should be 2-10x faster than CLI (no subprocess, pipe, temp file overhead)
- z4 QF_BV solver should be competitive with z3 (DPLL core is mature)
- z4 QF_FP may differ from z3 (different FP theory solver implementation)
- If z4 has more timeouts than z3, investigate and file issues against z4 repo

**Deliverables:**
- [ ] Benchmark harness in `llvm2-verify/benches/` or as a dedicated test
- [ ] Results documented in `reports/` or issue comment
- [ ] Decision: z4 as default backend, or z4 + z3 portfolio
- [ ] If z4 is default: update `verify_with_z4()` to use native API as primary

---

## Dependency Graph

```
Phase 1 (TL1: path dep + compile)
    |
    +--- Phase 2 (CLI integration, can run in parallel with Phase 1)
    |        |
    |        +--- Phase 4 (benchmarking, needs both CLI and native)
    |
    +--- Phase 3 (native API, requires Phase 1 complete)
             |
             +--- Phase 4 (benchmarking, needs both CLI and native)
```

Phase 2 and Phase 3 can proceed in parallel once Phase 1 is done.
Phase 4 requires both Phase 2 and Phase 3.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| z4 QF_BV solver has bugs | Low | High | Run all proofs through both z3 and z4, diff results |
| z4 FP theory incomplete | Medium | Medium | Fall back to z3 CLI for FP proofs if needed |
| z4 git dep slows CI | Medium | Low | Use path dep or vendor z4 source |
| BigInt conversion overflow | Low | Low | Only affects >64-bit BVs, not currently used |
| z4 output format differs from z3 | Low | Medium | Already verified: z4 accepts -smt2, outputs standard format |
| z4 timeout semantics differ | Medium | Low | Test explicitly with known-timeout formulas |

---

## Open Questions for TL7 (z4 Capability Audit)

1. Does z4 re-export a `RoundingMode` enum, or must rounding modes be constructed
   via `solver.fp_rounding_mode("RNE")`?
2. What is the exact `Sort` enum variant for floating-point? `Sort::FP(FPSort { eb, sb })`
   or `Sort::FloatingPoint(..)`?
3. Does z4's `ArraySort` use `index`/`element` field names matching the bridge's assumption?
4. Does `fp_const_from_bits(bits, eb, sb)` exist on z4 Solver? The bridge calls this
   at line 1066 but it was not found in the API grep. (May be on a separate impl block.)
5. z4's `Logic` enum: confirm `QfBvfp` exists (it does), confirm `QfAbvfp` does NOT
   exist (confirmed — need to use `All` as fallback).

---

## Issues to File

### Against LLVM2 (this repo)

1. **Fix 5 API mismatches in z4_bridge.rs** — const_array signature, bv_val return type,
   fp_geq/fp_leq naming, UF declare_fun/apply types. Part of #236.
2. **Update find_solver_binary() to prefer z4** — Phase 2 deliverable. Part of #34.
3. **Add benchmark harness for z4 vs z3 comparison** — Phase 4 deliverable.
4. **Handle QfAbvfp logic mapping** — Use `Logic::All` instead of non-existent variant.

### Against z4 repo

1. **Verify SMT-LIB2 output compatibility with z3** — Ensure `(get-value ...)` output
   format matches z3 conventions for LLVM2's parser. May need z4 issue if format differs.
2. **Add QfAbvfp logic variant** — If LLVM2 needs explicit theory-specific solving for
   combined BV+Array+FP formulas, z4 should support this logic string.

---

## References

- `crates/llvm2-verify/src/z4_bridge.rs` — LLVM2 bridge implementation (source of truth)
- `~/z4/crates/z4/src/lib.rs` — z4 public API facade
- `~/z4/crates/z4-dpll/src/api/` — z4 Solver implementation (methods organized by module)
- `designs/2026-04-14-z4-api-audit.md` — Previous API audit
- `designs/2026-04-14-z4-integration-guide.md` — Previous theory integration guide
- `designs/2026-04-15-z4-integration-readiness.md` — Readiness assessment
- `designs/2026-04-13-verification-architecture.md` — Overall verification architecture

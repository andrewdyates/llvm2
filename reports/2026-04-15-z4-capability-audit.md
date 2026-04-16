# z4 Capability Audit: z4 vs z3 for LLVM2 Verification

**Date:** 2026-04-16
**Author:** Researcher (W29-R1)
**Part of:** #247 (z4 integration), #227 (real tMIR integration)

---

## 1. Executive Summary

z4 is a fully capable replacement for z3 for LLVM2's verification needs. All
SMT-LIB2 logics used by `llvm2-verify` are supported. The native Rust API
covers every operation the z4 bridge needs: bitvectors, arrays, floating-point,
uninterpreted functions, and quantifiers. There are **4 method name mismatches**
in the current bridge that prevent compilation with `--features z4`, but these
are trivial renames (not capability gaps). z4 also provides features z3 lacks:
native Rust API (no C FFI), UNSAT proof certificates, interpolation, and
incremental core evolution tracking.

**Verdict:** z4 is ready. Fix the 4 method name mismatches and the bridge
compiles. No z3 capabilities are missing from z4 for our use case.

---

## 2. SMT-LIB2 Logic Support

### Logics required by LLVM2

| Logic | Purpose in LLVM2 | z4 Support | z3 Support |
|-------|-------------------|------------|------------|
| `QF_BV` | Core BV lowering proofs (add, sub, mul, cmp, shifts) | `Logic::QfBv` | Yes |
| `QF_ABV` | Memory model proofs (array theory for load/store) | `Logic::QfAbv` | Yes |
| `QF_BVFP` | Floating-point lowering proofs (FP32/FP64, ANE FP16) | `Logic::QfBvfp` | Yes |
| `QF_UFBV` | CEGIS synthesis (uninterpreted functions + BV) | `Logic::QfUfbv` | Yes |
| `QF_ABVFP` | Combined array+FP memory proofs | `Logic::All` (fallback) | Yes |
| `ALL` | Catch-all for mixed theories | `Logic::All` | Yes |

**Finding:** z4 supports all 6 logics LLVM2 uses. The only note is that z4 does
not have a dedicated `QF_ABVFP` variant -- the bridge correctly falls back to
`Logic::All`, which is equivalent.

### Additional logics z4 supports (not used by LLVM2)

z4 supports 50+ logics including: QF_LIA, QF_LRA, QF_NIA, QF_NRA, QF_FP,
QF_S (strings), QF_SEQ (sequences), QF_DT (algebraic datatypes), QF_AUFBV,
quantified variants (LIA, BV, UFBV, AUFLIA, etc.), and CHC solving. This is
comparable to z3's logic coverage.

---

## 3. Rust API Capability Matrix

### 3.1 Core Types

| Type | z4 | z3 (C API) | Notes |
|------|----|------------|-------|
| `Solver` | `z4::Solver` | `Z3_solver` | z4 is native Rust, z3 requires FFI |
| `Term` | `z4::Term` | `Z3_ast` | z4 Term is a typed wrapper around TermId |
| `Sort` | `z4::Sort` (enum) | `Z3_sort` | z4 Sort is a Rust enum with variants |
| `FuncDecl` | `z4::FuncDecl` | `Z3_func_decl` | For UF declarations |
| `Model` | `z4::Model` / `VerifiedModel` | `Z3_model` | z4 has verified model wrapper |
| `Logic` | `z4::Logic` (enum) | String-based | z4 has 50+ logic variants as enum |

### 3.2 Bitvector Operations

| Operation | z4 Method | Bridge Calls | Status |
|-----------|-----------|--------------|--------|
| BV constant | `bv_const(i64, u32)` | `bv_const` | OK |
| BV add | `bvadd(a, b)` | `bvadd` | OK |
| BV sub | `bvsub(a, b)` | `bvsub` | OK |
| BV mul | `bvmul(a, b)` | `bvmul` | OK |
| BV sdiv | `bvsdiv(a, b)` | `bvsdiv` | OK |
| BV udiv | `bvudiv(a, b)` | `bvudiv` | OK |
| BV neg | `bvneg(a)` | `bvneg` | OK |
| BV and | `bvand(a, b)` | `bvand` | OK |
| BV or | `bvor(a, b)` | `bvor` | OK |
| BV xor | `bvxor(a, b)` | `bvxor` | OK |
| BV not | `bvnot(a)` | (not used) | Available |
| BV shl | `bvshl(a, b)` | `bvshl` | OK |
| BV lshr | `bvlshr(a, b)` | `bvlshr` | OK |
| BV ashr | `bvashr(a, b)` | `bvashr` | OK |
| BV ult | `bvult(a, b)` | `bvult` | OK |
| BV ule | `bvule(a, b)` | `bvule` | OK |
| BV ugt | `bvugt(a, b)` | `bvugt` | OK |
| BV uge | `bvuge(a, b)` | `bvuge` | OK |
| BV slt | `bvslt(a, b)` | `bvslt` | OK |
| BV sle | `bvsle(a, b)` | `bvsle` | OK |
| BV sgt | `bvsgt(a, b)` | `bvsgt` | OK |
| BV sge | `bvsge(a, b)` | `bvsge` | OK |
| BV extract | `bvextract(a, hi, lo)` | `extract(hi, lo, o)` | **NAME MISMATCH** |
| BV concat | `bvconcat(a, b)` | `concat(h, l)` | **NAME MISMATCH** |
| BV zero-extend | `bvzeroext(a, n)` | `zero_ext(n, o)` | **NAME MISMATCH** |
| BV sign-extend | `bvsignext(a, n)` | `sign_ext(n, o)` | **NAME MISMATCH** |
| BV urem | `bvurem(a, b)` | (not used) | Available |
| BV srem | `bvsrem(a, b)` | (not used) | Available |
| BV smod | `bvsmod(a, b)` | (not used) | Available |
| BV repeat | `bvrepeat(a, n)` | (not used) | Available |
| BV rotl | `bvrotl(a, n)` | (not used) | Available |
| BV rotr | `bvrotr(a, n)` | (not used) | Available |
| BV nand | `bvnand(a, b)` | (not used) | Available |
| BV nor | `bvnor(a, b)` | (not used) | Available |
| BV xnor | `bvxnor(a, b)` | (not used) | Available |
| BV comp | `bvcomp(a, b)` | (not used) | Available |
| BV bv2int | `bv2int(bv)` | (not used) | Available |

### 3.3 Array Operations

| Operation | z4 Method | Bridge Calls | Status |
|-----------|-----------|--------------|--------|
| Select | `select(array, index)` | `select` | OK |
| Store | `store(array, index, value)` | `store` | OK |
| Const array | `const_array(index_sort, value)` | `const_array` | OK |
| Array sort | `Sort::Array(Box<ArraySort>)` | `Sort::Array(Box::new(ArraySort{...}))` | OK |

### 3.4 Floating-Point Operations

| Operation | z4 Method | Bridge Calls | Status |
|-----------|-----------|--------------|--------|
| FP add | `fp_add(rm, a, b)` | `fp_add` | OK |
| FP sub | `fp_sub(rm, a, b)` | `fp_sub` | OK |
| FP mul | `fp_mul(rm, a, b)` | `fp_mul` | OK |
| FP div | `fp_div(rm, a, b)` | `fp_div` | OK |
| FP neg | `fp_neg(a)` | `fp_neg` | OK |
| FP abs | `fp_abs(a)` | `fp_abs` | OK |
| FP sqrt | `fp_sqrt(rm, a)` | `fp_sqrt` | OK |
| FP fma | `fp_fma(rm, a, b, c)` | `fp_fma` | OK |
| FP eq | `fp_eq(a, b)` | `fp_eq` | OK |
| FP lt | `fp_lt(a, b)` | `fp_lt` | OK |
| FP gt | `fp_gt(a, b)` | `fp_gt` | OK |
| FP le | `fp_le(a, b)` | `fp_le` | OK |
| FP ge | `fp_ge(a, b)` | `fp_ge` | OK |
| FP is_nan | `fp_is_nan(a)` | `fp_is_nan` | OK |
| FP is_infinite | `fp_is_infinite(a)` | `fp_is_infinite` | OK |
| FP is_zero | `fp_is_zero(a)` | `fp_is_zero` | OK |
| FP is_normal | `fp_is_normal(a)` | `fp_is_normal` | OK |
| FP to sbv | `fp_to_sbv(rm, x, w)` | `fp_to_sbv` | OK |
| FP to ubv | `fp_to_ubv(rm, x, w)` | `fp_to_ubv` | OK |
| BV to FP | `bv_to_fp(rm, bv, eb, sb)` | `bv_to_fp` | OK |
| FP to FP | `fp_to_fp(rm, fp, eb, sb)` | `fp_to_fp` | OK |
| FP from BVs | `fp_from_bvs(sign, exp, sig, eb, sb)` | `fp_from_bvs` | OK |
| FP rounding mode | `try_fp_rounding_mode(name)` | `try_fp_rounding_mode` | OK |
| FP rem | `fp_rem(a, b)` | (not used) | Available |
| FP min | `fp_min(a, b)` | (not used) | Available |
| FP max | `fp_max(a, b)` | (not used) | Available |
| FP is_subnormal | `fp_is_subnormal(a)` | (not used) | Available |
| FP is_positive | `fp_is_positive(a)` | (not used) | Available |
| FP is_negative | `fp_is_negative(a)` | (not used) | Available |
| FP +inf | `fp_plus_infinity(eb, sb)` | (not used) | Available |
| FP -inf | `fp_minus_infinity(eb, sb)` | (not used) | Available |
| FP NaN | `fp_nan(eb, sb)` | (not used) | Available |
| FP +0 | `fp_plus_zero(eb, sb)` | (not used) | Available |
| FP -0 | `fp_minus_zero(eb, sb)` | (not used) | Available |
| FP to real | `fp_to_real(x)` | (not used) | Available |
| FP to IEEE BV | `try_fp_to_ieee_bv(x)` | (not used) | Available |

### 3.5 Uninterpreted Functions

| Operation | z4 Method | Bridge Calls | Status |
|-----------|-----------|--------------|--------|
| Declare fun | `declare_fun(name, domain, range)` | `declare_fun` | OK |
| Apply | `apply(func, args)` | `apply` | OK |

### 3.6 Boolean/Core Operations

| Operation | z4 Method | Bridge Calls | Status |
|-----------|-----------|--------------|--------|
| Bool const | `bool_const(b)` | `bool_const` | OK |
| And | `and(a, b)` | `and` | OK |
| Or | `or(a, b)` | `or` | OK |
| Not | `not(a)` | `not` | OK |
| Eq | `eq(a, b)` | `eq` | OK |
| ITE | `ite(c, t, e)` | `ite` | OK |
| Declare const | `declare_const(name, sort)` | `declare_const` | OK |

### 3.7 Quantifiers

| Operation | z4 Method | Bridge Status | Notes |
|-----------|-----------|---------------|-------|
| ForAll | `forall(vars, body)` | Returns error | Bridge says "not yet supported" |
| Exists | `exists(vars, body)` | Returns error | Bridge says "not yet supported" |
| ForAll w/ triggers | `forall_with_triggers(vars, triggers, body)` | Not used | Available |
| Exists w/ triggers | `exists_with_triggers(vars, triggers, body)` | Not used | Available |

**Note:** z4 fully supports quantifiers via E-matching and CEGQI. The bridge
currently returns an error for ForAll/Exists, directing callers to the CLI
fallback. This is a bridge limitation, not a z4 limitation.

### 3.8 Solver Control

| Operation | z4 Method | Status |
|-----------|-----------|--------|
| Assert | `assert_term(term)` | Used by bridge |
| Check SAT | `check_sat_with_details()` | Used by bridge |
| Model | `model()` -> `VerifiedModel` | Used by bridge |
| BV model value | `model.bv_val(name)` | Used by bridge |
| FP model value | `model.fp_val(name)` | Available |
| Push/Pop | `push()` / `pop()` | Available (not used by bridge) |
| Set timeout | `set_timeout(Duration)` | Available (bridge doesn't use yet) |
| Check SAT assuming | `check_sat_assuming(assumptions)` | Available |
| UNSAT core | `unsat_core()` | Available |
| Interrupt | `interrupt` flag (Arc<AtomicBool>) | Available |

---

## 4. Compilation Blockers

The z4 native API bridge (`crates/llvm2-verify/src/z4_bridge.rs`) has **4
method name mismatches** that prevent compilation with `--features z4`:

| Line | Bridge calls | z4 actual method | Fix |
|------|-------------|------------------|-----|
| 1037 | `solver.extract(hi, lo, o)` | `solver.bvextract(o, hi, lo)` | Rename + reorder args |
| 1042 | `solver.concat(h, l)` | `solver.bvconcat(h, l)` | Rename only |
| 1046 | `solver.zero_ext(n, o)` | `solver.bvzeroext(o, n)` | Rename + reorder args |
| 1050 | `solver.sign_ext(n, o)` | `solver.bvsignext(o, n)` | Rename + reorder args |

**Important:** The argument order also differs for `extract`, `zero_ext`, and
`sign_ext`. z4 puts the bitvector operand first, then the parameters. The
bridge currently passes parameters first. This must be corrected when renaming.

**Effort:** Trivial -- 4 line changes. File issue or include in next techlead wave.

---

## 5. z4 Advantages Over z3

| Feature | z4 | z3 |
|---------|----|----|
| Language | Pure Rust (native integration) | C++ with C API + Rust bindings |
| Safety | `#![forbid(unsafe_code)]` | FFI unsafe blocks required |
| UNSAT proofs | `UnsatProofArtifact`, Alethe format | Limited proof support |
| Proof checking | `PartialProofCheck`, `ProofQuality` | External tools only |
| Verified model | `VerifiedModel` wrapper | Raw model |
| Interpolation | `InterpolantResult` | Z3 has interpolation too |
| UNSAT core evolution | `IncrementalCoreEvolution`, `CoreEvolutionTracker` | Not available |
| Model provenance | `ModelProvenance`, `VariableProvenance` | Not available |
| CHC solving | Full PDR/portfolio engine (`z4::chc`) | Horn solver (SPACER) |
| ALL-SAT | `z4::allsat::AllSatSolver` | Manual blocking clauses |
| MaxSMT | `check_sat_max()` with soft constraints | OptSMT (different API) |
| Strings | QF_S, QF_SLIA, QF_SNIA | QF_S (similar) |
| Sequences | QF_SEQ, QF_SEQLIA | QF_SEQ (similar) |
| Datatypes | QF_DT, QF_UFDT, UFDTLIA | ADT support |
| Memory safety | No C FFI, no segfault risk | C FFI can crash |
| Thread safety | Arc<AtomicBool> interrupt | Thread-safe via context |
| CLI binary | `z4 solve FILE` or `z4 < FILE` | `z3 -smt2 FILE` |

---

## 6. z3 Features Not in z4

After thorough analysis, the following z3 features are **not present** in z4:

| z3 Feature | z4 Status | Impact on LLVM2 |
|------------|-----------|-----------------|
| Tactics/probes | Not in z4 API | None -- LLVM2 doesn't use tactics |
| Python bindings | Not applicable (z4 is Rust-native) | None |
| Optimization (optimize module) | z4 has `check_sat_max()` (MaxSMT) | None -- different API, similar capability |
| Parallel solving (`par.enable`) | z4 has portfolio solver for CHC | None -- LLVM2 proofs are fast enough |
| Fixedpoint engine (muZ) | z4 has CHC solver instead | None -- not used by LLVM2 |

**Conclusion:** No z3 feature used by LLVM2 is missing from z4.

---

## 7. z4 CLI Binary

z4 provides a CLI binary at `~/z4/crates/z4/src/main.rs`:

```
z4 [OPTIONS] [FILE]        -- solve SMT-LIB2 file
z4 solve [OPTIONS] [FILE]  -- explicit solve subcommand
z4 check drat FORMULA PROOF -- DRAT proof checking
z4 bench run [EVALS...]    -- benchmarking
z4 flatzinc solve FILE     -- FlatZinc constraint solving
```

The bridge's CLI fallback (`verify_with_cli`) currently prefers `z3` and falls
back to `z4`. Once the native API bridge is fixed, the CLI fallback is only
needed as a last resort.

---

## 8. Recommendations

1. **Fix the 4 method name mismatches** (trivial, 1 commit).
2. **Wire z4 timeout** -- the bridge creates `Z4Config.timeout_ms` but never
   calls `solver.set_timeout()`. Add: `solver.set_timeout(Some(Duration::from_millis(config.timeout_ms)))`.
3. **Add quantifier support** -- z4 has `forall()` and `exists()` methods.
   The bridge currently returns an error. Wire them up for completeness.
4. **Use verified model** -- z4 returns `VerifiedModel` from `model()`.
   The bridge calls `.into_inner()` immediately. Consider using the verified
   wrapper for additional safety guarantees.
5. **Enable UNSAT proof extraction** -- z4 can return `UnsatProofArtifact`
   when a proof is UNSAT. This could strengthen LLVM2's verification chain.
6. **Switch CLI preference to z4** -- once the native API is working, flip
   `find_solver_binary()` to prefer `z4` over `z3`.

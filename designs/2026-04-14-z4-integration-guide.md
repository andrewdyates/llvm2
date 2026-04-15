# z4 Theory Integration Guide for LLVM2

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Research Complete
**Part of:** #146 (z4-bindings API audit)

---

## Implementation Status (as of 2026-04-15)

**Overall: Research complete. z4-bindings API is documented. Integration into llvm2-verify is not yet done.**

| Component | Status | Details |
|-----------|--------|---------|
| **z4-bindings API documentation** | COMPLETE | This document covers all four theories (QF_ABV, QF_FP, QF_UF, bounded quantifiers). |
| **z4 bridge** (`z4_bridge.rs`) | IMPLEMENTED (not connected) | 2.8K LOC. SMT-LIB2 serialization exists. z4 native API feature-gated but triggers `compile_error!`. |
| **Array theory translation** | NOT IMPLEMENTED | `translate_expr_to_z4()` missing Array/FP/UF theory support. See #236. |
| **Integration into llvm2-verify** | NOT DONE | See #34. |

---

## Executive Summary

This guide documents the exact z4-bindings Rust API for four SMT theories needed by
LLVM2's `llvm2-verify` crate: arrays (QF_ABV), floating-point (QF_FP), uninterpreted
functions (QF_UF), and bounded quantifiers. All four theories are **fully implemented**
in z4 and **fully exposed** through z4-bindings. The previous researcher's claim is
confirmed and strengthened: z4 is not the bottleneck -- LLVM2 integration is.

### Audit Methodology

- Read every source file in `z4-bindings/src/expr/arrays.rs`, `fp.rs`, `fp_predict.rs`,
  `sort.rs`, `sort_constructors.rs`, `program/declarations.rs`, `program/commands.rs`,
  and `expr/mod.rs` (quantifiers, UF application).
- Counted lines of implementation: arrays theory (8,011 LOC including tests),
  FP theory (5,475 LOC), EUF theory (6,655 LOC).
- Verified test coverage exists for all four areas in z4-bindings/src/expr/tests.rs
  and z4-theories/*/src/tests/.
- Cross-referenced against LLVM2's `llvm2-verify/src/smt.rs` SmtExpr AST to confirm
  mapping feasibility.

---

## Theory 1: QF_ABV (Arrays + Bitvectors)

### z4-bindings API

**Sort creation** (`z4-bindings/src/sort_constructors.rs`):

```rust
use z4_bindings::Sort;

// Byte-addressable memory: (Array (_ BitVec 64) (_ BitVec 8))
let mem_sort = Sort::array(Sort::bv64(), Sort::bv8());

// Convenience alias (equivalent to above):
let mem_sort = Sort::memory();

// Word-addressable array:
let word_array = Sort::array(Sort::bv64(), Sort::bv64());
```

**Array operations** (`z4-bindings/src/expr/arrays.rs`):

```rust
use z4_bindings::{Expr, Sort, Z4Program};

let mut prog = Z4Program::new();
prog.set_logic("QF_ABV");

// Declare symbolic memory and address
let mem = prog.declare_const("mem", Sort::memory());
let addr = prog.declare_const("addr", Sort::bv64());
let val = prog.declare_const("val", Sort::bv8());

// select: read byte at address
let loaded: Expr = mem.clone().select(addr.clone());
// Result sort: BitVec(8) -- matches array element sort

// store: write byte at address, producing new array
let mem2: Expr = mem.clone().store(addr.clone(), val.clone());
// Result sort: Array(BitVec(64), BitVec(8))

// const_array: create array with all elements equal to value
let zero_mem = Expr::const_array(Sort::bv64(), Expr::bitvec_const(0u8, 8));
// Result sort: Array(BitVec(64), BitVec(8))

// Fallible variants available: try_select, try_store, try_const_array
// Return Result<Expr, SortError> instead of panicking
```

**Auto-coercion**: `store` automatically coerces value types when possible
(e.g., BitVec to Vec/String datatype, Int to BitVec). Source:
`z4-bindings/src/expr/arrays.rs:38-99`.

### Mapping to LLVM2 SmtExpr

| LLVM2 `SmtExpr` variant | z4-bindings API |
|--------------------------|-----------------|
| `Select { array, index }` | `array.select(index)` |
| `Store { array, index, value }` | `array.store(index, value)` |
| `ConstArray { index_sort, value }` | `Expr::const_array(index_sort, value)` |
| `SmtSort::Array(idx, elem)` | `Sort::array(idx_sort, elem_sort)` |

### Quality Assessment

- **Implementation depth**: 8,011 LOC across 18 source files in `z4-theories/arrays/`.
  Includes `axiom_checkers.rs` (McCarthy axiom enforcement), `store_chain.rs` (efficient
  store chain tracking), `equality.rs`/`equality_query.rs` (extensionality support),
  `bridge.rs` (integration with EUF solver), `model.rs` (model extraction).
- **Tests**: `tests/core_solver.rs` (779 LOC), `tests/store_chain.rs` (836 LOC),
  `tests/verification.rs` (906 LOC). Comprehensive coverage.
- **Theory solver**: Full `TheorySolver` trait implementation with propagation,
  check, and model extraction (`theory_impl.rs`, `theory_check.rs`, `theory_propagate.rs`).
- **Production readiness**: HIGH. Active crate with Kani annotations on public API.

### LLVM2 Integration Work Required

1. In `z4_bridge.rs`: translate `SmtExpr::Select/Store/ConstArray` to z4 API calls
2. In `tmir_semantics.rs`: encode tMIR `Load`/`Store` as array `select`/`store`
3. In `aarch64_semantics.rs`: encode `LDR`/`STR` as array `select`/`store`
4. In `memory_model.rs`: upgrade from concrete `HashMap<u64, u8>` to symbolic arrays

---

## Theory 2: QF_FP (Floating-Point)

### z4-bindings API

**Sort creation** (`z4-bindings/src/sort_constructors.rs`):

```rust
use z4_bindings::Sort;

// Standard IEEE 754 precisions
let fp16 = Sort::fp16();  // (_ FloatingPoint 5 11)
let fp32 = Sort::fp32();  // (_ FloatingPoint 8 24)
let fp64 = Sort::fp64();  // (_ FloatingPoint 11 53)

// Custom precision
let custom = Sort::fp(8, 24); // equivalent to fp32
```

**Arithmetic** (`z4-bindings/src/expr/fp.rs`):

```rust
use z4_bindings::{Expr, Sort, RoundingMode, Z4Program};

let mut prog = Z4Program::qf_fp();
let a = prog.declare_const("a", Sort::fp64());
let b = prog.declare_const("b", Sort::fp64());
let rm = RoundingMode::RNE; // Round nearest, ties to even

// Arithmetic (all take rounding mode)
let sum  = a.clone().fp_add(rm, b.clone());     // fp.add
let diff = a.clone().fp_sub(rm, b.clone());     // fp.sub
let prod = a.clone().fp_mul(rm, b.clone());     // fp.mul
let quot = a.clone().fp_div(rm, b.clone());     // fp.div
let root = a.clone().fp_sqrt(rm);               // fp.sqrt
let fma  = a.clone().fp_fma(rm, b.clone(), c);  // fp.fma (ternary: a*b+c)

// No rounding mode
let rem  = a.clone().fp_rem(b.clone());          // fp.rem
let abs  = a.clone().fp_abs();                   // fp.abs
let neg  = a.clone().fp_neg();                   // fp.neg
let min  = a.clone().fp_min(b.clone());          // fp.min
let max  = a.clone().fp_max(b.clone());          // fp.max
let rti  = a.clone().fp_round_to_integral(rm);   // fp.roundToIntegral
```

**Comparisons** (`z4-bindings/src/expr/fp.rs`):

```rust
// All return Bool sort (IEEE 754 semantics: NaN != NaN, +0 == -0 for fp_eq)
let eq  = a.clone().fp_eq(b.clone());   // fp.eq
let lt  = a.clone().fp_lt(b.clone());   // fp.lt
let le  = a.clone().fp_le(b.clone());   // fp.leq
let gt  = a.clone().fp_gt(b.clone());   // fp.gt
let ge  = a.clone().fp_ge(b.clone());   // fp.geq
```

**Classification predicates** (`z4-bindings/src/expr/fp_predict.rs`):

```rust
// All return Bool sort
let is_nan    = a.clone().fp_is_nan();
let is_inf    = a.clone().fp_is_infinite();
let is_zero   = a.clone().fp_is_zero();
let is_normal = a.clone().fp_is_normal();
let is_subn   = a.clone().fp_is_subnormal();
let is_pos    = a.clone().fp_is_positive();
let is_neg    = a.clone().fp_is_negative();
```

**Special values** (`z4-bindings/src/expr/fp.rs`):

```rust
let fp64 = Sort::fp64();
let pos_inf  = Expr::fp_plus_infinity(&fp64);
let neg_inf  = Expr::fp_minus_infinity(&fp64);
let nan      = Expr::fp_nan(&fp64);
let pos_zero = Expr::fp_plus_zero(&fp64);
let neg_zero = Expr::fp_minus_zero(&fp64);
```

**Conversions** (`z4-bindings/src/expr/fp_predict.rs`):

```rust
// FP -> signed/unsigned bitvector
let sbv32 = a.clone().fp_to_sbv(RoundingMode::RTZ, 32);
let ubv32 = a.clone().fp_to_ubv(RoundingMode::RTZ, 32);

// FP -> Real
let real_val = a.clone().fp_to_real();

// FP -> IEEE 754 bitvector (bit-pattern reinterpretation, no rounding)
let ieee_bv = a.clone().fp_to_ieee_bv(); // 64-bit BV for fp64

// Construct FP from components
let fp_val = Expr::fp_from_bvs(sign_1bit, exponent_11bit, significand_52bit);

// BV -> FP (interpret bits as IEEE 754)
let fp_from_bv = bv64.bv_to_fp(RoundingMode::RNE, 11, 53);

// Unsigned BV -> FP
let fp_from_ubv = ubv.bv_to_fp_unsigned(RoundingMode::RNE, 8, 24);

// FP precision conversion
let f32_from_f64 = f64_val.fp_to_fp(RoundingMode::RNE, 8, 24);
```

### Mapping to LLVM2 SmtExpr

| LLVM2 `SmtExpr` variant | z4-bindings API |
|--------------------------|-----------------|
| `FPAdd { rm, lhs, rhs }` | `lhs.fp_add(rm, rhs)` |
| `FPMul { rm, lhs, rhs }` | `lhs.fp_mul(rm, rhs)` |
| `FPDiv { rm, lhs, rhs }` | `lhs.fp_div(rm, rhs)` |
| `FPNeg { operand }` | `operand.fp_neg()` |
| `FPEq { lhs, rhs }` | `lhs.fp_eq(rhs)` |
| `FPLt { lhs, rhs }` | `lhs.fp_lt(rhs)` |
| `FPConst { bits, eb, sb }` | `Expr::fp_from_bvs(sign, exp, sig)` |
| `SmtSort::FloatingPoint(eb, sb)` | `Sort::fp(eb, sb)` |
| `RoundingMode::RNE` etc. | `z4_bindings::RoundingMode::RNE` etc. |

**Gap**: LLVM2's `SmtExpr` is missing `FPSub`, `FPSqrt`, `FPFma`, `FPRem`, `FPMin`,
`FPMax`, `FPRoundToIntegral`, and all classification predicates. These exist in
z4-bindings but not yet in LLVM2's AST. Adding them to `SmtExpr` is straightforward.

### Quality Assessment

- **Implementation depth**: 5,475 LOC across 19 source files in `z4-theories/fp/`.
  Includes `arithmetic_core.rs` (add/sub/mul/div), `arithmetic_advanced.rs` (FMA, sqrt),
  `bitblast.rs` (FP-to-BV reduction), `bv_circuits.rs` (BV circuit generation),
  `rounding.rs` (all 5 IEEE 754 modes), `special_ops.rs` (NaN/infinity/zero),
  `conversion.rs` (all FP conversions), `fma.rs` (fused multiply-add).
- **Tests**: `tests_basic.rs`, `tests_fp16_ops.rs`, `tests_model_value.rs`,
  `tests_rounding_decision.rs`. Good coverage of edge cases.
- **Bit-blasting**: z4 implements FP via bit-blasting to BV (same approach as Z3).
  Performance: practical for Float16 (exhaustive), Float32 (moderate formulas),
  expensive for Float64 (large formulas timeout).
- **Production readiness**: HIGH for Float16/Float32. MODERATE for Float64 (expect
  10-1000x slower than pure BV verification).

### LLVM2 Integration Work Required

1. Extend `SmtExpr` enum with missing FP variants (sub, sqrt, fma, predicates, conversions)
2. Map LLVM2 `RoundingMode` to z4 `RoundingMode` (1:1 correspondence already)
3. In `tmir_semantics.rs`: encode tMIR FP operations
4. In `aarch64_semantics.rs`: encode FMADD, FNEG, FCVT, etc.
5. In `neon_semantics.rs`: encode FADD.2D as `fp_add` per lane

---

## Theory 3: QF_UF (Uninterpreted Functions)

### z4-bindings API

**Function declaration** (`z4-bindings/src/program/declarations.rs`):

```rust
use z4_bindings::{Sort, Z4Program};

let mut prog = Z4Program::qf_uf();

// Declare uninterpreted function: f(BV64, BV64) -> BV64
prog.declare_fun("f", vec![Sort::bv64(), Sort::bv64()], Sort::bv64());

// Declare nullary function (constant): c() -> BV32
prog.declare_fun("c", vec![], Sort::bv32());
```

**Function application** (`z4-bindings/src/expr/mod.rs:321-365`):

```rust
use z4_bindings::{Expr, Sort};

let x = Expr::var("x", Sort::bv64());
let y = Expr::var("y", Sort::bv64());

// Apply with Bool return (default, for CHC relations):
let rel = Expr::func_app("state", vec![x.clone(), y.clone()]);
// Result sort: Bool

// Apply with explicit return sort (for UF with non-Bool return):
let result = Expr::func_app_with_sort("f", vec![x, y], Sort::bv64());
// Result sort: BV64
```

**Congruence**: The z4 solver automatically enforces the congruence axiom:
`x1 = y1 AND x2 = y2 => f(x1, x2) = f(y1, y2)`. No explicit encoding needed.

### Mapping to LLVM2 SmtExpr

| LLVM2 `SmtExpr` variant | z4-bindings API |
|--------------------------|-----------------|
| `UF { name, args, ret_sort }` | `Expr::func_app_with_sort(name, args, ret_sort)` |
| `UFDecl { name, arg_sorts, ret_sort }` | `prog.declare_fun(name, arg_sorts, ret_sort)` |

### Quality Assessment

- **EUF implementation**: 6,655 LOC across 17 source files in `z4-theories/euf/`.
  Includes `egraph.rs` (E-graph data structure), `closure.rs` (congruence closure),
  `merge.rs` (union-find merge with propagation), `explain.rs` (equality proof terms),
  `model_extraction.rs` (model generation).
- **Tests**: `tests/congruence.rs`, `tests/disequality.rs`, `tests/soundness.rs`,
  `tests/infrastructure.rs`. 2,450 lines of test code.
- **Bindings**: `func_app` and `func_app_with_sort` are fully functional. `declare_fun`
  correctly registers functions in the program state.
- **Production readiness**: HIGH. EUF is z4's "backbone" theory (same as Z3) -- all
  other theories connect through EUF for shared term equality reasoning.

### LLVM2 Integration Work Required

1. In `z4_bridge.rs`: translate `SmtExpr::UF` and `SmtExpr::UFDecl` to z4 API calls
2. Future: model external function calls (malloc, syscalls) as UF in semantics encoders

---

## Theory 4: Bounded Quantifiers

### z4-bindings API

**Universal quantifier** (`z4-bindings/src/expr/mod.rs:369-435`):

```rust
use z4_bindings::{Expr, Sort};

// Simple forall: for all i:BV64, body(i)
let i = Expr::var("i", Sort::bv64());
let body = i.clone().bvugt(Expr::bitvec_const(0u64, 64));
let forall = Expr::forall(vec![("i".into(), Sort::bv64())], body);
// Result sort: Bool

// Forall with trigger patterns (for E-matching):
let arr = Expr::var("arr", Sort::array(Sort::bv64(), Sort::bv8()));
let select_i = arr.clone().select(i.clone());
let trigger = vec![select_i.clone()]; // trigger on select(arr, i)
let bounded_body = Expr::implies(
    i.clone().bvult(Expr::bitvec_const(1024u64, 64)),
    select_i.fp_eq(Expr::bitvec_const(0u8, 8)),
);
let bounded_forall = Expr::forall_with_triggers(
    vec![("i".into(), Sort::bv64())],
    bounded_body,
    vec![trigger],
);
// Emits: (forall ((i (_ BitVec 64))) (! (=> (bvult i #x400) (= (select arr i) #x00))
//           :pattern ((select arr i))))
```

**Existential quantifier**:

```rust
let exists = Expr::exists(vec![("x".into(), Sort::bv32())], body);
let exists_triggered = Expr::exists_with_triggers(vars, body, triggers);
```

**Fallible variants**: `try_forall`, `try_forall_with_triggers`, `try_exists`,
`try_exists_with_triggers` -- return `Result<Expr, SortError>`.

**Multi-trigger support**: z4 supports multiple trigger groups and multi-term triggers:

```rust
// Multiple trigger groups (any one can fire):
let q = Expr::forall_with_triggers(vars, body, vec![
    vec![f_x.clone()],  // trigger group 1
    vec![g_x.clone()],  // trigger group 2
]);
// Emits: :pattern ((f x)) :pattern ((g x))

// Multi-term trigger (all terms must appear):
let q = Expr::forall_with_triggers(vars, body, vec![
    vec![f_x.clone(), g_x.clone()],  // both must match
]);
// Emits: :pattern ((f x) (g x))
```

### Mapping to LLVM2 SmtExpr

LLVM2's `SmtExpr` does **not** currently have quantifier variants. This is the one
genuine gap. The extension needed:

```rust
// Add to SmtExpr enum:
Forall {
    vars: Vec<(String, SmtSort)>,
    body: Box<SmtExpr>,
    triggers: Vec<Vec<SmtExpr>>,
},
Exists {
    vars: Vec<(String, SmtSort)>,
    body: Box<SmtExpr>,
    triggers: Vec<Vec<SmtExpr>>,
},
```

### Quality Assessment

- **Bindings**: `Expr::forall`, `Expr::forall_with_triggers`, `Expr::exists`,
  `Expr::exists_with_triggers` -- all fully implemented with Kani annotations,
  sort checking (body must be Bool), and multi-trigger support.
- **Tests**: 7 tests in `expr/tests.rs` covering basic forall/exists, multiple
  variables, trigger patterns, multi-trigger groups, and multi-term triggers.
- **Solver side**: z4-qbf provides Quantified Boolean Formula solving (556 LOC,
  small but functional). For SMT quantifiers (BV/array/FP sorts), z4-core handles
  them via the main DPLL(T) loop with theory-specific instantiation.
- **Production readiness**: MODERATE. The bindings are solid. The solver's quantifier
  instantiation strategy (E-matching vs MBQI) is less well-characterized than Z3's.
  For bounded BV quantifiers (small range), performance should be acceptable.
  For large ranges, unrolling is the recommended approach.

### LLVM2 Integration Work Required

1. Add `Forall`/`Exists` variants to `SmtExpr` enum
2. In `z4_bridge.rs`: translate to `Expr::forall_with_triggers` / `Expr::exists_with_triggers`
3. Add bounded quantifier helper that generates the range guard:
   ```rust
   fn bounded_forall(var: &str, lo: SmtExpr, hi: SmtExpr, body: SmtExpr) -> SmtExpr {
       SmtExpr::Forall {
           vars: vec![(var.into(), SmtSort::BitVec(64))],
           body: Box::new(SmtExpr::Implies {
               lhs: Box::new(SmtExpr::And {
                   lhs: Box::new(SmtExpr::BvUge { ... lo }),
                   rhs: Box::new(SmtExpr::BvUlt { ... hi }),
               }),
               rhs: Box::new(body),
           }),
           triggers: vec![],
       }
   }
   ```

---

## Logic Selection Guide

z4-bindings provides convenience constructors for common logics:

| Constructor | Logic | Theories | LLVM2 Use Case |
|-------------|-------|----------|----------------|
| `Z4Program::qf_bv()` | QF_BV | Bitvectors | Scalar arithmetic lowering (current) |
| `Z4Program::qf_aufbv()` | QF_AUFBV | Arrays + UF + BV | Memory load/store + external calls |
| `Z4Program::qf_fp()` | QF_FP | Floating-point | FP lowering verification |
| `Z4Program::qf_uf()` | QF_UF | Uninterpreted functions | External call abstraction |
| Custom: `prog.set_logic("QF_ABV")` | QF_ABV | Arrays + BV | Memory operations |
| Custom: `prog.set_logic("QF_FPBV")` | QF_FPBV | FP + BV | NEON FP SIMD |
| Custom: `prog.set_logic("BV")` | BV | BV + quantifiers | Bounded GPU proofs |
| Custom: `prog.set_logic("AUFBV")` | AUFBV | All + quantifiers | Full theory combination |
| Custom: `prog.set_logic("ALL")` | ALL | Everything | Fallback for mixed theories |

**Recommendation**: Start with specific logics (`QF_BV`, `QF_ABV`) and only upgrade
to `ALL` when needed. More specific logics enable solver optimizations.

---

## Gap Analysis: z4 vs LLVM2 Needs

### Fully Available (no z4 work needed)

| Capability | z4 API | LLVM2 Status |
|------------|--------|-------------|
| Array select/store | `Expr::select/store` | SmtExpr has variants, z4_bridge needs translation |
| Array constant | `Expr::const_array` | SmtExpr has variant, z4_bridge needs translation |
| FP arithmetic (all ops) | `Expr::fp_add/sub/mul/div/fma/sqrt` | SmtExpr partially covered |
| FP comparisons | `Expr::fp_eq/lt/le/gt/ge` | SmtExpr partially covered |
| FP predicates | `Expr::fp_is_nan/infinite/zero/...` | SmtExpr **missing** |
| FP conversions | `Expr::fp_to_sbv/ubv/real/ieee_bv/...` | SmtExpr **missing** |
| FP construction | `Expr::fp_from_bvs/bv_to_fp/fp_to_fp` | SmtExpr **missing** |
| FP special values | `Expr::fp_plus_infinity/nan/...` | SmtExpr **missing** |
| UF declaration | `prog.declare_fun(...)` | SmtExpr::UFDecl present |
| UF application | `Expr::func_app_with_sort(...)` | SmtExpr::UF present |
| Forall/exists | `Expr::forall/exists` | SmtExpr **missing** |
| Trigger patterns | `Expr::forall_with_triggers(...)` | SmtExpr **missing** |

### Gaps in LLVM2 (not z4)

1. **SmtExpr missing variants**: FP sub/sqrt/fma/rem/min/max/round, FP predicates,
   FP conversions, FP special values, Forall/Exists.
2. **z4_bridge.rs**: Only translates QF_BV operations currently. Needs extension for
   all four theories.
3. **Semantic encoders**: `tmir_semantics.rs` and `aarch64_semantics.rs` only produce
   QF_BV formulas. Need memory model, FP, and NEON encodings.

### True Gaps in z4

1. **Float64 performance**: FP bit-blasting for 64-bit doubles generates very large
   BV formulas. Expect timeouts on complex FP formulas. **Mitigation**: Use bounded
   testing (exhaustive for fp16, sampling for fp32/fp64) for practical verification.
2. **Quantifier instantiation maturity**: z4's quantifier strategies are less mature
   than Z3's MBQI. **Mitigation**: Use manual unrolling for small bounds (N < 1024),
   which is what LLVM2 needs for NEON (2-4 lanes) and small GPU kernels.
3. **No induction support**: z4 cannot prove inductive properties over unbounded arrays.
   **Mitigation**: Not needed for current LLVM2 scope. Bounded quantifiers + unrolling
   cover all near-term use cases.

---

## Recommended Integration Approach

### Phase 1: Array Memory Model (unblocks #122)

**Effort**: 1-2 worker sessions.

1. Extend `z4_bridge.rs` to handle `SmtExpr::Select`, `SmtExpr::Store`, `SmtExpr::ConstArray`
2. Create `SmtSort::Array` -> `Sort::array(...)` mapping
3. Encode tMIR Load/Store as array select/store in semantic encoders
4. Use logic `QF_ABV` via `prog.set_logic("QF_ABV")`
5. Test: verify `tMIR::Load(I32, addr)` lowering to `LDRWui [Xn, #off]`

### Phase 2: FP Theory (unblocks #123)

**Effort**: 2 worker sessions.

1. Add missing FP variants to `SmtExpr`
2. Extend `z4_bridge.rs` for FP translation
3. Encode AArch64 FMADD, FNEG, FCVT as z4 FP expressions
4. Use bounded testing (fp16 exhaustive, fp32/fp64 sampling) as primary strategy
5. z4 FP theory as secondary strategy for small formulas

### Phase 3: Quantifiers + UF (unblocks #124)

**Effort**: 1-2 worker sessions.

1. Add Forall/Exists to `SmtExpr`
2. Add `bounded_forall` helper
3. Extend `z4_bridge.rs` for quantifier/UF translation
4. Use manual unrolling for NEON lanes (N=2,4) -- no quantifiers needed
5. Use bounded quantifiers for GPU kernel proofs (N < 1024)

### Dependency Chain

```
Phase 1 (arrays) -----> Phase 2 (FP) -----> Phase 3 (quantifiers)
    |                       |                    |
    v                       v                    v
 Memory proofs          NEON FP proofs      GPU equivalence proofs
```

---

## Performance Expectations

| Theory Combination | Formula Size | Expected Time | Strategy |
|--------------------|-------------|---------------|----------|
| QF_BV (current) | < 100 BV ops | < 1s | Full SMT proof |
| QF_ABV | < 50 array ops + BV | 1-10s | Full SMT proof |
| QF_FP (fp16) | < 20 FP ops | 1-30s | Full SMT proof |
| QF_FP (fp32) | < 10 FP ops | 10s-5min | Full or bounded test |
| QF_FP (fp64) | < 5 FP ops | 30s-timeout | Bounded testing recommended |
| QF_AUFBV | Mixed array+UF+BV | 1-30s | Full SMT proof |
| BV (quantified) | Small range (N<64) | 1-30s | Unroll or quantifier |
| BV (quantified) | Large range (N<1024) | 10s-5min | Unroll |

**Key insight**: Keep proof obligations per-rule (small formulas). Avoid monolithic
whole-program verification. LLVM2's rule-by-rule approach is the right architecture
for SMT performance.

---

## References

1. z4-bindings array API: `~/z4/crates/z4-bindings/src/expr/arrays.rs`
2. z4-bindings FP API: `~/z4/crates/z4-bindings/src/expr/fp.rs`, `fp_predict.rs`
3. z4-bindings sort constructors: `~/z4/crates/z4-bindings/src/sort_constructors.rs`
4. z4-bindings declarations: `~/z4/crates/z4-bindings/src/program/declarations.rs`
5. z4-bindings quantifiers: `~/z4/crates/z4-bindings/src/expr/mod.rs:367-461`
6. z4-bindings UF application: `~/z4/crates/z4-bindings/src/expr/mod.rs:319-365`
7. z4 array theory: `~/z4/crates/z4-theories/arrays/src/`
8. z4 FP theory: `~/z4/crates/z4-theories/fp/src/`
9. z4 EUF theory: `~/z4/crates/z4-theories/euf/src/`
10. LLVM2 SmtExpr AST: `~/LLVM2/crates/llvm2-verify/src/smt.rs`
11. LLVM2 z4 bridge: `~/LLVM2/crates/llvm2-verify/src/z4_bridge.rs`
12. LLVM2 memory model: `~/LLVM2/crates/llvm2-verify/src/memory_model.rs`
13. Previous design: `~/LLVM2/designs/2026-04-13-z4-theory-extensions.md`

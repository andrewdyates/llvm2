# z4 Theory Extensions for Unified Solver Architecture

**Author:** Andrew Yates
**Date:** 2026-04-13
**Status:** Design
**Part of:** #121 (Master design: Unified solver architecture)

---

## Overview

LLVM2's unified solver architecture requires four SMT theory extensions beyond the current QF_BV (bitvector) baseline to support multi-target verified compilation. This document researches how Z3, CVC5, and Bitwuzla implement these theories, assesses z4's current status, and recommends an integration path for LLVM2.

**Key finding:** z4 already implements all four required theories (arrays, floating-point, EUF, and quantifiers) in the `z4-theories` crate. The primary gap is not in z4's solver capabilities but in **LLVM2's use of z4-bindings** -- the existing LLVM2 verification code (`llvm2-verify`) only exercises QF_BV. Extending LLVM2 to use z4's existing array, FP, and EUF support is primarily an LLVM2-side integration task.

---

## Theory 1: QF_ABV (Arrays + Bitvectors)

### What It Requires

Array theory adds two operations to bitvectors:
- `select(arr, idx) -> elem` -- read element at index
- `store(arr, idx, val) -> arr` -- write element, producing new array

Plus three axioms (SMT-LIB ArraysEx, v2.7):
1. **Read-after-write (same index):** `select(store(a, i, v), i) = v`
2. **Read-after-write (different index):** `i != j => select(store(a, i, v), j) = select(a, j)`
3. **Extensionality:** `(forall i. select(a, i) = select(b, i)) => a = b`

Sort: `(Array (_ BitVec 64) (_ BitVec 8))` for byte-addressed memory, or `(Array (_ BitVec 64) (_ BitVec 64))` for word-addressed arrays.

**Ref:** SMT-LIB ArraysEx theory, v2.7 (2024-07-21). https://smt-lib.org/theories-ArraysEx.shtml

### How Z3 Implements It

Z3 implements arrays by **reduction to equality and uninterpreted functions (EUF)**. Rather than maintaining explicit array models, Z3:

1. Converts `select(a, i)` and `store(a, i, v)` into first-class EUF terms
2. Applies congruence closure to detect equalities between array reads
3. Generates axiom instances lazily: only the read-over-write and extensionality axioms relevant to the current conflict are instantiated
4. Uses **model-based array instantiation**: when a candidate model violates array axioms, generates targeted lemma instances

The key algorithm is **lazy axiom instantiation** -- rather than eagerly generating all axiom instances (exponential), Z3 generates them on demand guided by DPLL(T) conflict analysis.

**Ref:** de Moura & Bjorner, "Generalized, Efficient Array Decision Procedures," FMCAD 2009.

### How CVC5 and Bitwuzla Implement It

**CVC5** uses a similar lazy approach: array lemma instantiation is demand-driven during theory combination. CVC5 supports arrays combined with all other theories (bitvectors, arithmetic, UF) through Nelson-Oppen theory combination with shared terms.

**Bitwuzla** (Niemetz & Preiner, CAV 2023) handles arrays as one of its four core theories (BV, FP, arrays, UF). Bitwuzla's approach:
- Reduces array constraints to bitvector constraints via lemma-on-demand
- Uses local search for bitvectors with array lemma refinement
- Supports full QF_ABV and QF_AUFBV logics

### z4 Current Status

z4 **already has a full array theory implementation** in `z4-theories/arrays/`:
- `theory_impl.rs` -- TheorySolver implementation
- `axiom_checkers.rs` -- McCarthy axiom enforcement
- `store_chain.rs` -- Efficient store chain tracking
- `equality.rs` / `equality_query.rs` -- Array equality with extensionality
- `model.rs` -- Array model extraction
- `bridge.rs` -- Integration with EUF solver

z4-bindings already exposes array operations:
- `z4-bindings/src/expr/arrays.rs` -- `select`, `store`, array coercion
- `z4-bindings/src/sort.rs` -- `Sort::Array(idx_sort, elem_sort)`
- `z4-bindings/src/memory.rs` -- Memory model built on arrays

### Recommended Approach for LLVM2

**The gap is in LLVM2, not z4.** LLVM2 needs to:

1. **Extend `llvm2-verify` semantic encoders** to use array operations from z4-bindings:
   - `tmir_semantics.rs`: Encode tMIR memory operations (Load, Store) as array select/store
   - `aarch64_semantics.rs`: Encode AArch64 LDR/STR as array select/store
   - Memory model: `(Array (_ BitVec 64) (_ BitVec 8))` for byte-addressed AArch64 memory

2. **Encode GPU kernel semantics** as array transformations:
   ```
   encode(parallel_map(f, arr)) =
     store(store(...store(arr, 0, f(select(arr, 0))), 1, f(select(arr, 1)))..., N-1, f(select(arr, N-1)))
   ```
   For small N, unroll directly. For large N, use bounded quantifiers (see Theory 4).

3. **Encode NEON vector operations** as array operations on small fixed-size arrays:
   ```
   FADD.4S Vd, Vn, Vm =
     for lane in 0..4: result[lane] = fp.add(Vn[lane], Vm[lane])
   ```

### Priority: P1 (Critical Path)

Array theory unlocks: memory operation verification, GPU kernel equivalence proofs, NEON lane-wise encoding. This is the highest-priority z4 integration.

### Dependencies

- z4 arrays: Already implemented (no dependency)
- z4-bindings array API: Already exposed (no dependency)
- LLVM2 `llvm2-verify` updates: New work required

---

## Theory 2: QF_FP (Floating-Point)

### What It Requires

IEEE 754 floating-point theory per SMT-LIB v2.7:

**Sorts:**
- `RoundingMode` -- five rounding modes
- `(_ FloatingPoint eb sb)` -- parameterized by exponent bits and significand bits
- Standard: `Float16 = (_ FloatingPoint 5 11)`, `Float32 = (_ FloatingPoint 8 24)`, `Float64 = (_ FloatingPoint 11 53)`

**Rounding modes:** RNE (round nearest ties-to-even), RNA (ties-to-away), RTP (toward positive), RTN (toward negative), RTZ (toward zero).

**Operations (all take RoundingMode except abs, neg, rem):**
- Arithmetic: `fp.add`, `fp.sub`, `fp.mul`, `fp.div`, `fp.sqrt`, `fp.fma`, `fp.rem`
- Comparison: `fp.lt`, `fp.leq`, `fp.gt`, `fp.geq`, `fp.eq` (IEEE equality, +0 = -0)
- Classification: `fp.isNormal`, `fp.isSubnormal`, `fp.isZero`, `fp.isInfinite`, `fp.isNaN`, `fp.isNegative`, `fp.isPositive`
- Conversions: `to_fp` (from BV, Real, other FP), `fp.to_ubv`, `fp.to_sbv`, `fp.to_real`
- Special values: `+oo`, `-oo`, `NaN`, `+zero`, `-zero`
- Misc: `fp.abs`, `fp.neg`, `fp.min`, `fp.max`, `fp.roundToIntegral`

**Ref:** SMT-LIB FloatingPoint theory, v2.7 (2024-07-21). https://smt-lib.org/theories-FloatingPoint.shtml

### How Z3 Implements It

Z3 implements FP via **bit-blasting**: floating-point operations are decomposed into bitvector operations on the underlying IEEE 754 representation. This is sound and complete but expensive -- FP multiplication on 64-bit doubles generates ~50K bitvector clauses.

Key Z3 FP implementation details:
- Each FP value is represented as three bitvectors: sign (1 bit), exponent (eb bits), significand (sb-1 bits)
- Operations encode the full IEEE 754 algorithm: unpack, align exponents, compute, round, repack
- Rounding is encoded as bitvector comparisons on guard/round/sticky bits
- Special value handling (NaN, infinity, denormals) adds conditional branches in BV encoding

**Performance:** FP verification is 10-1000x slower than pure BV. Z3's FP solver can handle small formulas (< 100 FP operations) but struggles with large ones.

### How Bitwuzla Implements It

Bitwuzla uses a **word-level approach** for FP that avoids full bit-blasting:
1. Abstracting FP constraints to BV constraints at the word level
2. Using local search on BV combined with FP-specific lemma refinement
3. Bit-blasting only as a fallback

This is typically faster than Z3's eager bit-blasting for satisfiable instances.

**Ref:** Niemetz & Preiner, "Bitwuzla," CAV 2023.

### z4 Current Status

z4 **already has a comprehensive FP theory implementation** in `z4-theories/fp/`:
- `arithmetic_core.rs` -- Core FP arithmetic (add, sub, mul, div)
- `arithmetic_advanced.rs` -- FMA, sqrt, advanced operations
- `bitblast.rs` -- FP-to-BV bit-blasting
- `bv_circuits.rs` -- BV circuit generation for FP operations
- `rounding.rs` -- All five IEEE 754 rounding modes
- `special_ops.rs` -- NaN/infinity/zero handling
- `conversion.rs` -- FP <-> BV <-> Real conversions
- `model_value.rs` -- FP model extraction
- `fma.rs` -- Fused multiply-add
- `constructors.rs` -- FP value construction

z4-bindings exposes FP operations:
- `z4-bindings/src/expr/fp.rs` -- Full FP arithmetic, comparisons, special values
- `z4-bindings/src/expr/fp_predict.rs` -- FP classification predicates
- `z4-bindings/src/sort.rs` -- `Sort::FloatingPoint(eb, sb)` with Float16/32/64/128

### Recommended Approach for LLVM2

**Again, the gap is in LLVM2, not z4.**

1. **Phase 1 (Practical MVP): Bounded testing for FP**
   The unified solver design document already notes this: "prove integer/bitwise operations exactly, use bounded testing for FP." This is what STOKE does -- verify FP operations by testing on concrete inputs (exhaustive for Float16, random sampling for Float32/Float64). No z4 FP theory needed.

2. **Phase 2: Bitwise encoding via BV**
   Encode IEEE 754 FP operations as explicit bitvector operations (sign/exponent/significand manipulation). Sound and complete, uses only QF_BV. Slow for Float64 but workable for Float16/Float32 NEON lane operations.

3. **Phase 3: Native QF_FP via z4**
   Use z4's existing FP theory solver directly. This is the cleanest approach but requires confidence in z4's FP correctness.

For NEON SIMD verification specifically:
```
FADD.2D Vd, Vn, Vm =
  lane0: fp.add(RNE, Vn[63:0], Vm[63:0])
  lane1: fp.add(RNE, Vn[127:64], Vm[127:64])
```

Each lane is an independent FP operation. The vector is just 2 (or 4) parallel FP operations.

### Priority: P2 (Important, Not Blocking)

Integer/bitwise superoptimization works without FP. NEON integer SIMD works without FP. FP is needed for NEON FP instructions and GPU FP kernels. The bounded testing approach (Phase 1) provides practical FP verification without z4 FP theory.

### Dependencies

- z4 FP theory: Already implemented
- z4-bindings FP API: Already exposed
- LLVM2 `llvm2-verify` FP encoders: New work required
- LLVM2 NEON ISel rules: Must exist before verification

---

## Theory 3: QF_UF (Uninterpreted Functions)

### What It Requires

Uninterpreted functions (UF) provide **abstraction** -- model functions whose implementation is unknown but which satisfy congruence: `a = b => f(a) = f(b)`.

Core operations:
- Declare uninterpreted function: `(declare-fun f (Sort1 Sort2) SortR)`
- Apply function: `(f x y)` produces result of sort SortR
- Congruence axiom: `x1 = y1 AND x2 = y2 => f(x1, x2) = f(y1, y2)`

### How Z3 Implements It

Z3 uses **congruence closure** (Nieuwenhuis & Oliveras, 2007):
1. E-graph data structure: equivalence classes of terms with parent pointers
2. Merge operation: when `a = b` is asserted, merge their equivalence classes
3. Congruence propagation: after merge, check if any function applications become congruent
4. Explanation: produce proof terms for derived equalities

Z3's EUF solver is the "backbone" theory -- all other theories connect through EUF for shared term equality reasoning (Nelson-Oppen theory combination).

### z4 Current Status

z4 **already has a full EUF implementation** in `z4-theories/euf/`:
- `egraph.rs` -- E-graph data structure
- `closure.rs` -- Congruence closure
- `merge.rs` -- Merge operation with propagation
- `explain.rs` -- Equality explanation/proofs
- `solver.rs` / `solver_query.rs` -- Theory solver interface
- `model_extraction.rs` -- Model generation for UF
- `theory_impl.rs` -- TheorySolver trait implementation

### Recommended Approach for LLVM2

UF is needed for:
1. **External function calls**: Abstract functions the solver cannot see (syscalls, library calls)
2. **ANE/CoreML operations**: Model ANE ops as uninterpreted functions with known properties
3. **Abstract memory operations**: Model heap allocators as uninterpreted functions

In LLVM2:
```rust
// Model an external call as an uninterpreted function
let malloc = program.declare_fun("malloc", vec![Sort::bitvec(64)], Sort::bitvec(64));
// Constraint: malloc returns aligned pointers
program.assert(malloc.apply(vec![size]).bvand(Expr::bitvec_const(7, 64)).eq(Expr::bitvec_const(0, 64)));
```

### Priority: P3 (Low -- Needed Later)

UF is not on the critical path. Scalar and NEON verification don't need UF. GPU kernel verification can work with concrete semantics (array theory). UF becomes important when modeling external calls and ANE ops.

### Dependencies

- z4 EUF: Already implemented
- z4-bindings UF API: Need to verify bindings expose `declare-fun` and application
- LLVM2 integration: Future work

---

## Theory 4: Bounded Quantifiers

### What They Require

Universal quantifiers over bounded ranges:
```smt2
(forall ((i (_ BitVec 64)))
  (=> (and (bvule #x0000000000000000 i) (bvult i N))
      (= (select result_gpu i) (select result_cpu i))))
```

This expresses: "for all indices in [0, N), the GPU result matches the CPU result."

### How Z3 Implements It

Z3 uses two complementary approaches:

1. **E-matching with triggers** (Detlefs et al., 2005):
   - User provides trigger patterns: `(forall ((i BV64)) (! (=> ...) :pattern ((select a i))))`
   - When a ground term matching the trigger appears, Z3 instantiates the quantifier
   - Fast but incomplete -- only instantiates for observed terms

2. **Model-Based Quantifier Instantiation (MBQI)** (Ge & de Moura, 2009):
   - Build a candidate model ignoring quantifiers
   - Check if model satisfies quantifiers
   - If not, derive a counterexample from the model
   - Add targeted quantifier instantiation to refute the model
   - Iterate until fixed-point or timeout

For bounded quantifiers specifically, Z3 often handles them via **finite model finding**: if the index range is small enough, enumerate all instances.

**Ref:** Ge & de Moura, "Complete Instantiation for Quantified Formulas in SMT," CAV 2009.

### How Bitwuzla Handles It

Bitwuzla supports universal (`FORALL`) and existential (`EXISTS`) quantifiers. For bit-vector quantifiers, Bitwuzla can enumerate small domains. For larger domains, it uses counterexample-guided abstraction refinement (CEGAR).

### z4 Current Status

z4 has quantifier support in:
- `z4-core/src/lib.rs` -- Core quantifier handling in the SMT solver
- `z4-qbf/` -- Quantified Boolean Formula solver (QBF)
- `z4-bindings/src/program/` -- Quantifier assertion support in the bindings

The extent of quantifier instantiation strategies (MBQI, E-matching) needs investigation.

### Recommended Approach for LLVM2

For LLVM2's specific use case (bounded array quantifiers for GPU equivalence):

1. **Phase 1: Manual unrolling for small N**
   For NEON (4 lanes) and small GPU kernels (N < 64), unroll the quantifier:
   ```rust
   for i in 0..N {
       program.assert(
           select(result_gpu, bv_const(i, 64))
               .eq(select(result_cpu, bv_const(i, 64)))
       );
   }
   ```
   This is exact and fast (each assertion is a simple array/BV check).

2. **Phase 2: Symbolic bounded quantifiers**
   For larger N, use z4's quantifier support with triggers:
   ```smt2
   (assert (forall ((i (_ BitVec 64)))
     (! (=> (bvult i N) (= (select result i) (f (select input i))))
        :pattern ((select result i)))))
   ```

3. **Phase 3: Inductive proofs**
   For arbitrary N, prove by induction:
   - Base: property holds for empty array
   - Step: if property holds for array[0..k], it holds for array[0..k+1]
   This requires z4 support for induction or can be encoded manually.

### Priority: P2 (Important for GPU Scaling)

Small-array proofs (NEON 2/4 lanes) work with unrolling. GPU kernel proofs for fixed-size arrays (N < 1024) work with unrolling. Bounded quantifiers are needed for arbitrary-size arrays.

### Dependencies

- z4 quantifier support: Partially implemented (needs assessment)
- z4-bindings quantifier API: Needs verification
- LLVM2 array encoding: Depends on Theory 1 (QF_ABV)

---

## z4 Integration Architecture for LLVM2

### Current State (llvm2-verify)

```
llvm2-verify/src/
  smt.rs              -- BV expression builder
  tmir_semantics.rs   -- tMIR -> BV encoding (QF_BV only)
  aarch64_semantics.rs -- AArch64 -> BV encoding (QF_BV only)
  lowering_proof.rs   -- Per-rule BV equivalence proofs
  peephole_proof.rs   -- Per-optimization BV equivalence proofs
  nzcv.rs             -- NZCV flag model (QF_BV)
```

### Target State

```
llvm2-verify/src/
  smt.rs              -- Expression builder (BV + Array + FP + UF)
  tmir_semantics.rs   -- tMIR -> multi-sort encoding
  aarch64_semantics.rs -- AArch64 -> multi-sort encoding
  neon_semantics.rs   -- NEW: NEON SIMD semantic encoding
  gpu_semantics.rs    -- NEW: GPU kernel semantic encoding (Metal/generic)
  ane_semantics.rs    -- NEW: ANE operation semantic encoding
  memory_model.rs     -- NEW: Array-based memory model
  lowering_proof.rs   -- Extended with memory, NEON proofs
  peephole_proof.rs   -- Extended with NEON peephole proofs
  dispatch_proof.rs   -- NEW: Multi-target dispatch correctness
  nzcv.rs             -- NZCV flag model (unchanged)
```

### Required z4-bindings API Surface

| Operation | z4-bindings Module | Status |
|-----------|-------------------|--------|
| `Expr::bitvec_const`, `bvadd`, etc. | `expr/bv.rs` | Available |
| `Sort::array(idx, elem)` | `sort.rs` | Available |
| `Expr::select(arr, idx)` | `expr/arrays.rs` | Available |
| `Expr::store(arr, idx, val)` | `expr/arrays.rs` | Available |
| `Sort::floating_point(eb, sb)` | `sort.rs` | Available |
| `Expr::fp_add(rm, a, b)` | `expr/fp.rs` | Available |
| `Expr::fp_mul(rm, a, b)` | `expr/fp.rs` | Available |
| `Expr::fp_is_nan(x)` | `expr/fp_predict.rs` | Available |
| `declare_fun(name, args, ret)` | `program/` | Needs verification |
| Bounded quantifiers | `program/` | Needs verification |

### Logic Selection

| LLVM2 Verification Task | SMT Logic | z4 Theories Used |
|------------------------|-----------|-----------------|
| Scalar arithmetic lowering | QF_BV | Bitvectors |
| Memory load/store lowering | QF_ABV | Arrays + Bitvectors |
| NEON integer SIMD | QF_BV | Bitvectors (lane decomposition) |
| NEON FP SIMD | QF_FPBV | Floating-point + Bitvectors |
| GPU kernel equivalence | QF_ABV + bounded | Arrays + BV + Quantifiers |
| External call abstraction | QF_AUFBV | Arrays + UF + BV |
| ANE operation equivalence | QF_UFBV | UF + BV |

---

## Implementation Priority and Phasing

### Phase 1: QF_ABV Integration (P1, Critical Path)

**Goal:** Verify memory operations and encode GPU/NEON array semantics.

**Work items:**
1. Add `memory_model.rs` to `llvm2-verify` using z4-bindings array API
2. Encode tMIR Load/Store as array select/store in `tmir_semantics.rs`
3. Encode AArch64 LDR/STR as array select/store in `aarch64_semantics.rs`
4. Add memory load/store lowering proofs to `lowering_proof.rs`
5. Test: verify `tMIR::Load(I32, addr) -> LDRWui [Xn, #off]`

**Estimated effort:** 1-2 techlead sessions
**Blocked by:** Nothing (z4 arrays already exist)

### Phase 2: NEON Semantic Encoding (P1, Critical Path)

**Goal:** Encode NEON SIMD instructions for verification.

**Work items:**
1. Add `neon_semantics.rs` with lane-decomposition encoding
2. Integer NEON ops (ADD.4S, MUL.4S, etc.) as BV operations per lane
3. FP NEON ops (FADD.2D, FMUL.2D, etc.) using either:
   - Bounded testing (Phase 2a, practical)
   - z4 FP theory (Phase 2b, complete)
4. Horizontal operations (FADDP, SADDLV) with cross-lane semantics
5. Test: verify NEON lowering rules

**Estimated effort:** 2-3 techlead sessions
**Blocked by:** NEON ISel rules in `llvm2-lower`

### Phase 3: FP Theory Integration (P2, Important)

**Goal:** Full FP verification for NEON FP and GPU FP operations.

**Work items:**
1. Extend `smt.rs` to create FP sorts and operations
2. Encode tMIR FP operations (Fadd, Fmul, etc.) as z4 FP expressions
3. Encode AArch64 FMADD, FNEG, FCVT as z4 FP expressions
4. Add rounding mode handling (AArch64 uses RNE by default)
5. Test: verify FP lowering rules

**Estimated effort:** 2 techlead sessions
**Blocked by:** Phase 2 (NEON encoding)

### Phase 4: GPU Kernel Encoding (P2, Important)

**Goal:** Encode Metal GPU kernel semantics for cross-target verification.

**Work items:**
1. Add `gpu_semantics.rs` with parallel map/reduce encoding
2. Encode parallel_map as iterated array store (for small N: unrolled)
3. Encode parallel_reduce as iterated fold (using Associative proof)
4. Encode GPU memory model (shared memory, thread-local storage)
5. Test: verify dot_product CPU vs GPU equivalence

**Estimated effort:** 3-4 techlead sessions
**Blocked by:** Phase 1 (arrays), GPU lowering infrastructure

### Phase 5: Bounded Quantifiers and UF (P3, Future)

**Goal:** Scale GPU proofs to arbitrary array sizes and abstract external calls.

**Work items:**
1. Verify z4 quantifier support via z4-bindings
2. Add bounded quantifier helpers to `smt.rs`
3. Encode GPU kernel equivalence with symbolic N
4. Add UF encoding for external calls and ANE operations
5. Test: verify GPU kernel for symbolic array size

**Estimated effort:** 2-3 techlead sessions
**Blocked by:** Phases 1, 4

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| z4 array theory has correctness bugs | Invalid proofs accepted | Cross-check with Z3 for critical proofs; exhaustive testing for small instances |
| FP verification too slow for Float64 | Proofs timeout | Use bounded testing for FP; full proofs only for Float16/Float32 |
| Bounded quantifier instantiation incomplete | GPU proofs fail for large N | Unroll for known N; use CEGAR for symbolic N |
| z4-bindings API doesn't expose needed operations | Integration blocked | Extend z4-bindings (same org, can file issue to z4 repo) |
| Theory combination overhead | Solver slowdown when mixing BV+Array+FP | Keep proofs per-rule (small formulas); avoid monolithic whole-program encoding |

---

## Summary: z4 Theory Status vs LLVM2 Needs

| Theory | z4-theories | z4-bindings | LLVM2 llvm2-verify | Gap |
|--------|------------|-------------|-------------------|-----|
| QF_BV (bitvectors) | Full | Full | Full | None |
| QF_ABV (arrays) | Full | Full | Not used | LLVM2 integration |
| QF_FP (floating-point) | Full | Full | Not used | LLVM2 integration |
| QF_UF (uninterpreted fns) | Full | Partial? | Not used | Bindings verification + LLVM2 integration |
| Bounded quantifiers | Partial | Needs verification | Not used | z4 assessment + LLVM2 integration |

**The critical insight is that z4 is NOT the bottleneck.** The work is in LLVM2: writing semantic encoders that use z4's existing capabilities. This reframes issues #122, #123, #124 from "z4 needs new features" to "LLVM2 needs to use z4's existing features."

---

## References

1. SMT-LIB ArraysEx Theory, v2.7. https://smt-lib.org/theories-ArraysEx.shtml
2. SMT-LIB FloatingPoint Theory, v2.7. https://smt-lib.org/theories-FloatingPoint.shtml
3. de Moura & Bjorner. "Generalized, Efficient Array Decision Procedures." FMCAD 2009.
4. Ge & de Moura. "Complete Instantiation for Quantified Formulas in SMT." CAV 2009.
5. Niemetz & Preiner. "Bitwuzla." CAV 2023.
6. Brain, Tinelli, Ruemmer, Wahl. "An Automatable Formal Semantics for IEEE-754 Floating-Point Arithmetic." FMCAD 2015.
7. Bjorner, de Moura. "Programming Z3." https://theory.stanford.edu/~nikolaj/programmingz3.html
8. z4 source: `~/z4/crates/z4-theories/{arrays,fp,euf}/`
9. z4-bindings source: `~/z4/crates/z4-bindings/src/expr/{arrays,fp,bv}.rs`
10. LLVM2 verification architecture: `designs/2026-04-13-verification-architecture.md`

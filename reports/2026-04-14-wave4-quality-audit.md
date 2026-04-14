# Wave 4 Quality Audit (Commits 382-390)

**Auditor:** Claude Opus 4.6 (Agent)
**Date:** 2026-04-14
**Scope:** Commits 5f500ed, 33af566, 4f23440, ffbee68, a8db5df, 8f0fb72, 803ca42, 50c1bba, 21c771a

---

## Summary

Wave 4 added the unified CEGIS synthesis loop for multi-target search (1302 LOC),
fixed the LslRI/LsrRI/AsrRI NOP stubs with real UBFM/SBFM encodings plus 13 new
opcodes in lower.rs (591 LOC), extended NEON semantics to 128-bit vectors with
EvalResult::Bv128 (875 LOC), integrated z4 array and floating-point theories,
a unified solver integration design doc, a Wave 3 quality audit report, a type
dedup fix, and expanded regalloc tests to 215 (2434 LOC).

Build is clean (`cargo check` passes).

**Findings:** 2 P1, 3 P2, 4 P3.

---

## P1 Critical

### P1-1: BvShl/BvLshr/BvAshr evaluator lacks u128 paths -- 128-bit NEON shift operations silently truncate

**File:** `crates/llvm2-verify/src/smt.rs:1015-1047`

Wave 4 commit 8f0fb72 added `EvalResult::Bv128` and u128 evaluation paths for
`BvAnd`, `BvOr`, and `BvXor` (the bitwise operations). However, the shift
operations `BvShl`, `BvLshr`, and `BvAshr` were NOT upgraded. They still call
`lhs.try_eval(env)?.as_u64()`, which truncates u128 results to u64 via
`EvalResult::Bv128(v) => v as u64`.

This means evaluating a 128-bit shift expression (e.g., `NEON SHL.4S` on a
128-bit vector constructed via `concat`) will silently truncate the high 64 bits
before shifting, producing wrong results.

**Impact:** Any CEGIS verification or mock evaluation involving 128-bit NEON
shift-by-immediate instructions (SHL, USHR, SSHR) on full 128-bit vectors will
produce incorrect evaluation results. The neon_semantics.rs tests work around
this by using `lane_extract` (which decomposes to sub-64-bit expressions before
evaluation), but direct evaluation of the composed 128-bit expression is broken.

**Evidence:** In smt.rs, compare:
- BvAnd (line 982): `if *width > 64 { ... as_u128() ... Bv128(...) }`
- BvShl (line 1015): No width check, always `as_u64()`

BvAdd, BvSub, BvMul, BvNeg also lack u128 paths (lines 941-1052), but those
are pre-existing issues from before Wave 4. The shift operations are newly
highlighted because the same commit that added Bv128 (8f0fb72) only upgraded
the bitwise ops but not the arithmetic/shift ops.

**Fix:** Add `if *width > 64 { ... as_u128() ... Bv128(...) }` branches to
BvShl, BvLshr, and BvAshr, mirroring the pattern used for BvAnd/BvOr/BvXor.
Also add BvAdd, BvSub, BvMul, BvNeg u128 paths for completeness.

### P1-2: NEON cost model does not adjust for lane count -- scalar/NEON cost comparison is apples-to-oranges

**File:** `crates/llvm2-verify/src/unified_synthesis.rs:78-80, 175-193`

The unified synthesis loop compares scalar and NEON candidates by raw instruction
cost (lines 688-694). However, NEON costs are instruction-level costs (cost=1 for
NEON ADD), while scalar costs are also instruction-level (cost=1 for scalar ADD).
For a 4-element vector operation, a single NEON ADD replaces 4 scalar ADDs, but
the cost comparison treats them as equal.

The doc comment at line 78 acknowledges this: "For cost comparison with scalar,
the caller should consider the lane count (a single NEON ADD replaces N scalar
ADDs)." But the actual `find_best` method at line 688 does NOT factor in lane
count -- it compares raw costs directly.

**Impact:** The unified CEGIS loop will incorrectly prefer scalar over NEON for
operations that process multiple elements, since both show cost=1 but NEON
processes N elements per instruction. This defeats the purpose of multi-target
search. For single-element operations (where the source is a single scalar),
scalar preference is correct and the tie-breaking logic works. But for vectorizable
patterns, the cost model is wrong.

**Fix:** Either (a) divide NEON instruction cost by lane_count to get per-element
cost for comparison, or (b) multiply scalar cost by the number of elements being
processed. The `NeonSynthOpcode::cost()` doc comment suggests approach (a).

---

## P2 Normal

### P2-1: Csinv encoding uses bits 30:29 = 10 -- does not match ARM ARM for CSINV

**File:** `crates/llvm2-codegen/src/lower.rs:1401-1417`

The CSINV encoding sets `(0b10 << 29)` which produces bits 30:29 = 10. According
to ARM ARM (DDI 0487, C6.2.69):
- CSEL:  sf | op=0 | S=0 | 11010100 | Rm | cond | op2=00 | Rn | Rd
- CSINC: sf | op=0 | S=0 | 11010100 | Rm | cond | op2=01 | Rn | Rd
- CSINV: sf | op=1 | S=0 | 11010100 | Rm | cond | op2=00 | Rn | Rd
- CSNEG: sf | op=1 | S=0 | 11010100 | Rm | cond | op2=01 | Rn | Rd

The code uses `(0b10 << 29)` for both CSINV and CSNEG. This sets bit 30=1 and
bit 29=0, which means op=1, S=0. Per the ARM ARM, CSINV has op=1, S=0, op2=00
and CSNEG has op=1, S=0, op2=01. So the encoding bits are actually correct --
bit 30 is the 'op' field and bit 29 is the 'S' field.

However, the test assertions (line 2905) assert bit 30 = 1 but do not verify
bit 29 = 0 (S field). If someone changes the shift constant from `0b10` to
`0b11`, both CSINV and CSNEG would silently emit S=1 variants (CCMN/CCMP
space). **Add a test assertion for bit 29 = 0.**

**Impact:** Current encoding is correct. Risk is future regression without the
bit 29 assertion.

### P2-2: UnifiedSearchConfig defaults to only S2 arrangement -- misses 128-bit NEON

**File:** `crates/llvm2-verify/src/unified_synthesis.rs:212-222`

The default `UnifiedSearchConfig` only searches `VectorArrangement::S2` (2x32-bit
in 64-bit). This means the unified CEGIS loop by default will never find 128-bit
NEON lowerings (S4, H8, B16, D2), despite commit 8f0fb72 adding full 128-bit
support in neon_semantics.rs.

The comment explains this is "testable with the u64 evaluator," but with the
Bv128 addition, this limitation is no longer necessary for bitwise ops (though
shift/arith ops still need u128 paths per P1-1).

**Impact:** Default configuration underutilizes 128-bit NEON. Users must manually
configure `neon_arrangements` to include 128-bit variants.

**Fix:** Once P1-1 is resolved, add S4 and D2 to the default arrangements (or
at least document the limitation prominently).

### P2-3: ConstArray serialization only handles BV index sorts -- non-BV array sorts will serialize but not verify

**File:** `crates/llvm2-verify/src/smt.rs:1309-1317`

The `ConstArray` serialization computes the element sort from `value.sort()` and
combines it with `index_sort` into an `Array(idx, elem)`. This works correctly
for the current use case (`Array(BitVec64, BitVec8)` for memory). However, the
`const_array()` constructor accepts *any* `SmtSort` as `index_sort`, including
`FloatingPoint` or nested `Array` sorts, which would produce syntactically valid
but semantically nonsensical SMT-LIB2.

The z4 native API translate function explicitly returns an error for all array
operations (line 796-798: "Array theory not yet supported in z4 native API").
This means array proofs can only use the CLI fallback path.

**Impact:** Low severity since the current codebase only uses `BV -> BV` arrays.
The API permits misuse but no caller exercises it.

**Fix:** Add a debug_assert in `const_array()` that `index_sort` is `BitVec`.

---

## P3 Low

### P3-1: NEON synthesis search space only generates self-operations

**File:** `crates/llvm2-verify/src/unified_synthesis.rs:304-396`

The `neon_candidates()` method only generates NEON candidates where the second
operand is the same vector as the first (`vn OP vn`). It never generates
candidates with a second independent operand or vector-of-immediates. This means
the search space only finds patterns like `ADD v, v` (doubling) or `SUB v, v`
(zeroing) but cannot find more useful patterns like `ADD v, <constant>`.

The `test_unified_neon_mul2_eq_shl1` test at line 1273 works around this by
manually constructing a `MUL v, <2,2>` source expression, but the synthesis
loop's own candidate generation cannot produce such candidates.

**Impact:** Limits the practical utility of multi-target synthesis. Acceptable
for the initial implementation but should be expanded.

### P3-2: unified_synthesis.rs encode_scalar_unary panics on non-unary opcodes

**File:** `crates/llvm2-verify/src/unified_synthesis.rs:421-427`

The `encode_scalar_unary` function panics on unsupported opcodes. In a synthesis
loop that systematically explores a search space, panics are inappropriate --
the function should return a Result. Same issue in `encode_scalar_binary` (line
430), `encode_neon_unary` (line 458), `encode_neon_binary` (line 470),
`encode_neon_bitwise` (line 485), and `encode_neon_shift` (line 498).

**Impact:** Since these are only called from the search space generator which
correctly categorizes opcodes before calling, the panics are unreachable in
practice. But they violate defensive coding principles.

### P3-3: RegAlloc test expansion is substantive (215 tests) but lacks property-based tests

**File:** `crates/llvm2-regalloc/src/*.rs`

The 82 new tests (133 -> 215) across 12 regalloc modules test meaningful
invariants: interval merging, splitting, eviction cascading, spill weights,
diamond CFG liveness, loop-carried values, call clobber handling, and
interference graph correctness. This is solid coverage.

However, all tests use hand-constructed inputs. There are no property-based
tests (e.g., "for any valid function, allocation never assigns overlapping
live ranges to the same register"). The `test_no_overlapping_allocations_invariant`
in liveness.rs (line 1475) tests a specific scenario rather than a universal
property.

**Impact:** Good coverage for known patterns, but may miss edge cases in the
combinatorial space of register allocation.

### P3-4: FP rounding mode serialization correct but incomplete

**File:** `crates/llvm2-verify/src/z4_bridge.rs:245-253`

The `rounding_mode_to_smt2` function maps all five IEEE 754 rounding modes
correctly per SMT-LIB2 spec (RNE, RNA, RTP, RTN, RTZ). Tests at lines
1242-1247 verify all five mappings. This is correct and complete.

However, the z4 native API backend (feature-gated) returns an error for all
FP operations (line 800-807), meaning FP theory is CLI-only. This is consistent
and documented but worth noting.

---

## Cross-Module Consistency Check

| Pattern | unified_synthesis.rs | neon_semantics.rs | Consistent? |
|---------|---------------------|-------------------|-------------|
| NEON ADD encoding | Delegates to `neon_semantics::encode_neon_add` | `map_lanes_binary(bvadd)` | Yes |
| NEON SUB encoding | Delegates to `neon_semantics::encode_neon_sub` | `map_lanes_binary(bvsub)` | Yes |
| NEON MUL D2 check | Skips D2 (line 323) | `debug_assert(arr != D2)` | Yes |
| NEON SHL range | `shift_amt < lane_bits` (line 338) | `debug_assert(imm < lane_bits)` | Yes |
| NEON USHR range | `shift_amt >= 1` (line 342) | `debug_assert(imm >= 1 && imm <= lane_bits)` | Partial -- synthesis checks >= 1 but not <= lane_bits |
| Bitwise ops lane-free | `encode_neon_bitwise` (no arrangement) | `encode_neon_and/orr/eor` (no arrangement) | Yes |
| SSHR in synthesis | Not included in NeonSynthOpcode | Present in neon_semantics | Expected -- SSHR is signed, less common in synthesis |

The USHR range check inconsistency (synthesis allows `shift_amt == lane_bits`
while neon_semantics asserts `imm <= lane_bits`) is technically fine per ARM ARM
(USHR accepts shift in [1, lane_bits]), but worth noting.

---

## Verification

```
$ cargo check
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.31s
```

Build is clean. No warnings or errors.

---

## Summary Table

| ID | Severity | Component | Issue |
|----|----------|-----------|-------|
| P1-1 | P1 | smt.rs evaluator | BvShl/BvLshr/BvAshr lack u128 paths, 128-bit NEON shifts silently truncate |
| P1-2 | P1 | unified_synthesis.rs | NEON cost model not adjusted for lane count, scalar/NEON comparison broken |
| P2-1 | P2 | lower.rs | Csinv/Csneg encoding correct but tests lack bit 29 (S field) assertion |
| P2-2 | P2 | unified_synthesis.rs | Default config only searches S2, misses 128-bit NEON |
| P2-3 | P2 | smt.rs ConstArray | Accepts non-BV index sorts without validation |
| P3-1 | P3 | unified_synthesis.rs | NEON search space only generates self-operations |
| P3-2 | P3 | unified_synthesis.rs | Encoding helpers panic instead of returning Result |
| P3-3 | P3 | llvm2-regalloc | 215 tests are substantive but no property-based tests |
| P3-4 | P3 | z4_bridge.rs | FP theory is CLI-only (z4 API returns errors for FP ops) |

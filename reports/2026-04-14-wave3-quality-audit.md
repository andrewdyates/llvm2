# Wave 3 Quality Audit (Commits 376-381)

**Auditor:** Claude Opus 4.6 (Agent)
**Date:** 2026-04-14
**Scope:** Commits f1bac82, e856cbb, 2413cc1, c86d6d7, 874104a

---

## Summary

Wave 3 added bitfield move encodings (UBFM/SBFM/BFM), register-offset load/store
(LdrRO/StrRO), GOT/TLV loads, TBZ/TBNZ encoding fix, memory model proofs,
computation graph analysis, and two research design documents. Build is clean
(`cargo check` passes, `cargo test --no-run` compiles all test targets).

**Findings:** 2 P1, 4 P2, 5 P3.

---

## P1 Critical

### P1-1: lower.rs still emits errors for newly-implemented opcodes

**File:** `crates/llvm2-codegen/src/lower.rs:1195-1204`

The unified encoder (`encode.rs`) now has full implementations for UBFM, SBFM,
BFM, LdrRO, StrRO, LdrGot, and LdrTlvp (commit f1bac82). However, `lower.rs`
(the production lowering path used by `lower_function()`) still returns
`Err(LowerError::UnsupportedInstruction(...))` for all seven of these opcodes.

This means any function using these instructions will fail at encoding time
through the production pipeline, despite the unified encoder supporting them.

**Impact:** Code containing bitfield moves, register-offset loads/stores, or
GOT/TLV loads will fail to compile through the production lowering path.

**Fix:** Either delegate to `encode::encode_instruction()` from `lower.rs`
for these opcodes, or duplicate the encoding logic in `lower.rs::encode_inst()`.

### P1-2: lower.rs emits NOP for LslRI/LsrRI/AsrRI immediate shifts

**File:** `crates/llvm2-codegen/src/lower.rs:523-529`

```rust
AArch64Opcode::LslRI | AArch64Opcode::LsrRI | AArch64Opcode::AsrRI => {
    // Immediate shifts use UBFM/SBFM encoding.
    // For simplicity, encode as the register-variant with the
    // immediate handled by prior lowering. Emit NOP as fallback.
    // TODO: Implement proper UBFM/SBFM encoding for immediate shifts.
    Ok(0xD503201F) // NOP
}
```

The unified encoder (encode.rs) correctly implements LSL/LSR/ASR immediate as
UBFM/SBFM with proper immr/imms calculations (verified against xcrun as ground
truth in tests). But `lower.rs` still emits NOP for these, **silently producing
wrong code** instead of returning an error. A NOP where a shift was expected
will corrupt program semantics without any diagnostic.

**Impact:** Silent miscompilation. Any function using immediate shifts through
the production path will have those shifts silently dropped.

**Fix:** Copy the UBFM/SBFM encoding logic from `encode.rs` into `lower.rs`,
or route through the unified encoder.

---

## P2 Important

### P2-1: LdrRO/StrRO FP size uses `FpSize::Double | _` wildcard pattern

**File:** `crates/llvm2-codegen/src/aarch64/encode.rs:1193-1195`

```rust
let size = match fp_sz {
    FpSize::Single => 0b10,
    FpSize::Double | _ => 0b11,
};
```

The `| _` wildcard after `FpSize::Double` silently maps `FpSize::Half` (16-bit
FP) to a 64-bit load/store size. If half-precision FP register-offset loads are
ever generated, they will produce 64-bit loads instead of 16-bit loads. Same
pattern appears in StrRO at line 1240.

Additionally, the `FpSize::Double | _` pattern in the LdrRO FP path at line
1193-1195 uses a raw integer `0b10`/`0b11` which is then re-matched into
`encoding_mem::LoadStoreSize` at line 1198 -- an unnecessary double-dispatch
that could introduce further confusion.

**Fix:** Replace the wildcard with explicit variants. For Half, either emit
an error or use size=01 (halfword).

### P2-2: Non-interference proof only tests fixed non-overlapping gaps

**File:** `crates/llvm2-verify/src/memory_proofs.rs:471-497`

The non-interference proofs use hardcoded gap values (8 bytes for I32, 16 bytes
for I64). The proof asserts: "store at A, load at A+gap is unchanged." However,
these proofs do not verify the boundary case where the store region partially
overlaps the load region. For example, storing 8 bytes at address A and loading
8 bytes at address A+4 should return a mix of stored and original bytes -- this
overlapping case is not tested and could mask bugs in the SMT array
Store/Select chain.

Additionally, the `gap` parameter is not checked against `size_bytes` at the
proof-building level. Nothing prevents `proof_non_interference("test", 4, 2)`
which would create an overlapping proof obligation that claims non-interference
but should actually show interference.

**Fix:** Add a `debug_assert!(gap >= size_bytes as u64)` in
`proof_non_interference()`, and add explicit partial-overlap test cases
alongside the non-interference tests.

### P2-3: Compute graph pattern detection has high false-positive risk

**File:** `crates/llvm2-lower/src/compute_graph.rs:482-499, 505-531`

The data-parallel detection (`detect_data_parallel`) classifies a block as
data-parallel if **any** value in the block has an Array type AND **any**
instruction is an element-wise binary op. These conditions are not correlated:

```rust
let has_array = value_types.values().any(|ty| matches!(ty, Ty::Array(_, _)));
let has_elementwise = instrs.iter().any(|node| match &node.instr { ... });
has_elementwise
```

A block that receives an array parameter but only operates on scalar values
within a loop body (e.g., `array[i] + scalar`) would be misclassified as
DataParallel. The detection does not verify that the binary operations actually
operate ON the array elements.

Similarly, `detect_matrix_heavy` only checks that FMul appears before FAdd in
the instruction list -- it does not verify they are connected by data flow.
A block with independent FMul and FAdd instructions on unrelated values would
be misclassified as MatrixHeavy.

**Fix:** Check that the binary operations actually consume array-typed values,
and that the FMul result flows into the FAdd for matrix-heavy detection.

### P2-4: Two encoders (lower.rs vs encode.rs) with diverging implementations

**File:** `crates/llvm2-codegen/src/lower.rs` and `crates/llvm2-codegen/src/aarch64/encode.rs`

There are two complete instruction encoders:
1. `lower.rs::encode_inst()` (private, used by production `lower_function()`)
2. `encode.rs::encode_instruction()` (public unified encoder)

These encoders have different error handling (encode.rs defaults non-register
operands to XZR silently; lower.rs returns errors), different FP size
derivation logic, and now different opcode coverage (7 opcodes work in
encode.rs but error in lower.rs). The TBZ/TBNZ cross-check tests prove they
agree for TBZ/TBNZ, but there is no systematic cross-check for all opcodes.

This is an architectural debt issue that will cause repeated bugs as new
encodings are added to one encoder but not the other.

**Fix:** Either unify to a single encoder (preferred) or add a comprehensive
cross-check test that verifies both encoders produce identical output for
every opcode.

---

## P3 Minor

### P3-1: memory_proofs `_` suppress unused-mut pattern

**File:** `crates/llvm2-verify/src/memory_proofs.rs:355`

```rust
let _ = &mut inputs; // suppress unused warning
```

The `inputs` variable is built but the `&mut` borrow is used solely to suppress
a warning. This is a code smell -- the variable should either be used or the
`#[allow(unused)]` attribute applied.

### P3-2: Compute graph node lookup is O(n)

**File:** `crates/llvm2-lower/src/compute_graph.rs:228-230`

```rust
pub fn node(&self, id: ComputeNodeId) -> Option<&ComputeNode> {
    self.nodes.iter().find(|n| n.id == id)
}
```

Node lookup scans the entire node list. For large graphs (hundreds of nodes
from complex programs), this linear scan repeated for every edge during
`build_edges()` and `partition_cost()` could become a performance bottleneck.
Consider using a HashMap<ComputeNodeId, usize> index.

### P3-3: TransferCost::for_bytes divides by 10^9 losing precision

**File:** `crates/llvm2-lower/src/compute_graph.rs:126-127`

```rust
let transfer_cycles = bytes.saturating_mul(per_byte_nanocycles) / 1_000_000_000;
```

For the CPU-GPU path with `per_byte_nanocycles = 1_000_000_000`, this computes
`bytes * 1e9 / 1e9 = bytes` which is correct. But for the CPU-ANE path with
`per_byte_nanocycles = 2_000_000_000`, small byte counts (< 500M) can
overflow `u64` in `bytes.saturating_mul(per_byte_nanocycles)` resulting in
`u64::MAX / 1e9` -- a massive overestimate. The saturating_mul prevents
undefined behavior but produces incorrect cost estimates for moderate-to-large
transfers.

### P3-4: LdrGot/LdrTlvp encode identically

**File:** `crates/llvm2-codegen/src/aarch64/encode.rs:1277-1297`

`LdrGot` and `LdrTlvp` have identical encoding logic (both emit
`encode_load_store_ui(0b11, 0, 0b01, scaled, rn, rd)`). The differentiation
between GOT and TLV is entirely in the relocation layer, but `lower.rs`'s
relocation collection (`collect_relocations`) does not recognize either LdrGot
or LdrTlvp -- it only collects relocations for Adrp, AddPCRel, and Bl. If
these opcodes need distinct relocation types (GOT_LOAD vs TLV_LOAD), the
relocation collector needs updating.

### P3-5: Design docs reference non-existent file paths

**File:** `designs/2026-04-14-z4-integration-guide.md`

The design doc references z4-bindings source paths like
`z4-bindings/src/expr/arrays.rs` and line counts (8,011 LOC). These are
useful for understanding the API, but the paths are to a separate repo and
not verifiable from within LLVM2. The design is well-structured and the API
mapping is clear, but the research claims (e.g., "all four theories are fully
implemented") cannot be independently verified without pulling z4-bindings.

---

## Test Coverage Assessment

| Module | New Code (LOC) | Test Count | Coverage |
|--------|---------------|------------|----------|
| encode.rs (BFM/LdrRO/etc) | ~200 | 26 | Good |
| lower.rs (TBZ/TBNZ fix) | ~40 | 7 | Good |
| memory_proofs.rs | ~1260 | 31 | Excellent |
| compute_graph.rs | ~1770 | 25 | Good |
| smt.rs (Select fix) | ~8 | 0 direct | Covered by memory_proofs |
| lowering_proof.rs (multi-input) | ~60 | 0 direct | Covered by memory_proofs |
| Design docs | N/A | N/A | N/A |

**Missing edge cases:**
- TBZ/TBNZ: bit 0 edge case not tested (only bit 3, 7, 15, 32, 63)
- UBFM/SBFM: no 32-bit variant cross-check with lower.rs
- LdrRO/StrRO: no half-precision FP test
- Memory proofs: no partial-overlap test
- Compute graph: no test for false-positive detection (scalar ops in
  array-containing blocks)

---

## Build Verification

```
$ cargo check
Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.17s

$ cargo test --no-run
Finished `test` profile [unoptimized + debuginfo] target(s) in 4.56s
```

Both pass cleanly with zero warnings in the compilation output.

---

## Conclusion

Wave 3 adds significant encoding and verification capability. The encode.rs
additions (UBFM/SBFM/BFM, LdrRO/StrRO, LdrGot/LdrTlvp) are well-tested with
ground-truth cross-checks against xcrun as. The memory proofs module is
architecturally sound, building symbolic SMT expressions that will translate
directly to z4 queries. The compute graph analysis provides a reasonable
foundation for heterogeneous compute, though the pattern detection needs
refinement.

The critical issue is the **dual-encoder divergence** (P1-1, P1-2, P2-4):
encode.rs now supports opcodes that lower.rs does not, and lower.rs silently
emits NOP for immediate shifts. This creates a window where code that "works"
in tests (which may use encode.rs) will fail or silently miscompile through the
production pipeline (which uses lower.rs). Unifying or systematically syncing
these encoders should be the top priority.

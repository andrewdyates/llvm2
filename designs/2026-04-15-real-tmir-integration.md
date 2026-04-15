# Real tMIR Integration: From Stubs to Production

**Date:** 2026-04-15
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #227 (tMIR integration is stubbed -- cannot accept real tMIR input)

---

## Executive Summary

LLVM2 currently depends on four local stub crates (`stubs/tmir-types`, `tmir-instrs`,
`tmir-func`, `tmir-semantics`) that define a minimal tMIR interface. These stubs are
well-designed -- they cover ~35 instruction variants, proof annotations, atomic
operations, and serde support -- but they are NOT the real tMIR repo (ayates_dbx/tMIR).

This document designs the migration from stubs to the real tMIR crate, identifies
the required tMIR features, and defines the adapter changes needed in LLVM2.

**Key finding:** The adapter layer (`crates/llvm2-lower/src/adapter.rs`, 82 tests)
is already substantial and maps tMIR types to internal LIR. The primary work is
(1) replacing stub crate dependencies with real tMIR git deps, (2) handling API
divergences between stubs and real tMIR, and (3) building an end-to-end test
pipeline that compiles real tMIR modules to Mach-O.

---

## 1. Current State

### 1.1 Stub Crates

| Crate | LOC | Purpose | serde | Tests |
|-------|-----|---------|-------|-------|
| `tmir-types` | 190 | Ty, ValueId, BlockId, FuncId, TmirProof, StructDef | Yes | 0 |
| `tmir-instrs` | 405 | Instr (35 variants), BinOp, UnOp, CmpOp, CastOp, atomics | Yes | 0 |
| `tmir-func` | 86 | Block, Function, Module | Yes | 0 |
| `tmir-semantics` | 171 | InstrSemantics trait, ConcreteSemantics stub | Yes | 0 |

**Total stub LOC:** ~852

The stubs have serde `Serialize`/`Deserialize` on all types, which enables a
JSON/MessagePack wire format for tMIR modules. This is a good design decision
that should be preserved in the real integration.

### 1.2 Adapter Layer (llvm2-lower)

`crates/llvm2-lower/src/adapter.rs` translates tMIR to internal LIR:
- Type mapping: `Ty -> Type` (scalars, pointers, with signedness preservation)
- Instruction mapping: `Instr -> Opcode + Instruction` (BinOp, UnOp, Cmp, Cast, Load/Store, Call, branches, constants, atomics)
- Proof extraction: `TmirProof -> Proof` re-keyed to internal Value refs
- Block parameter resolution: emits COPY pseudo-instructions for SSA phi semantics
- **82 tests** covering translation correctness

### 1.3 Verification Semantics

`crates/llvm2-verify/src/tmir_semantics.rs` encodes tMIR instruction semantics
as `SmtExpr` bitvector formulas using the internal `Opcode` enum. This module
does NOT depend on the tMIR stubs -- it uses LLVM2's own opcode definitions.
This means verification can continue to work even during the tMIR migration.

### 1.4 What Does NOT Exist

- No real tMIR repo is accessible (ayates_dbx/tMIR and dropbox-ai-prototypes/tMIR
  both not accessible as of 2026-04-15)
- No serialized tMIR test fixtures (.json or .msgpack files)
- No end-to-end test that reads a tMIR file from disk and compiles it
- tmir-semantics stub has only a placeholder `ConcreteSemantics` implementation

---

## 2. Required tMIR Features

### 2.1 Minimum Viable (Phase 1: scalar arithmetic + control flow)

The following tMIR features MUST be present in the real repo for LLVM2 to accept
any real input:

| Feature | Stub Status | Notes |
|---------|-------------|-------|
| `Ty::Bool`, `Ty::Int(N)`, `Ty::UInt(N)`, `Ty::Float(N)` | Present | Core scalar types |
| `Ty::Ptr(Box<Ty>)` | Present | Required for memory operations |
| `Ty::Void` | Present | Return type for void functions |
| `ValueId`, `BlockId`, `FuncId` | Present | SSA value/block/function refs |
| `Instr::BinOp` (17 variants) | Present | All arithmetic/logic ops |
| `Instr::UnOp` (5 variants) | Present | Neg, Not, FNeg, FAbs, FSqrt |
| `Instr::Cmp` (22 variants) | Present | Integer + float comparisons |
| `Instr::Cast` (12 variants) | Present | Type conversions |
| `Instr::Load`, `Instr::Store` | Present | Memory access |
| `Instr::Const`, `Instr::FConst` | Present | Constants |
| `Instr::Br`, `Instr::CondBr`, `Instr::Return` | Present | Control flow |
| `Instr::Call` | Present | Direct function calls |
| `Block { id, params, body }` | Present | Basic blocks with SSA params |
| `Function { id, name, ty, entry, blocks }` | Present | Function definitions |
| `Module { name, functions, structs }` | Present | Translation units |
| `FuncTy { params, returns }` | Present | Function type signatures |
| serde `Serialize`/`Deserialize` on all types | Present | Wire format |

### 2.2 Extended (Phase 2: proof-guided optimization)

| Feature | Stub Status | Notes |
|---------|-------------|-------|
| `TmirProof` enum (11 variants) | Present | Proof annotations |
| `InstrNode.proofs: Vec<TmirProof>` | Present | Per-instruction proofs |
| `Function.proofs: Vec<TmirProof>` | Present | Function-level proofs |
| `InstrSemantics` trait | Present (stub) | Formal semantics for verification |

### 2.3 Advanced (Phase 3: full tMIR coverage)

| Feature | Stub Status | Notes |
|---------|-------------|-------|
| `Instr::Alloc`, `Instr::Dealloc` | Present | Stack/heap allocation |
| `Instr::Switch` | Present | Multi-way branches |
| `Instr::CallIndirect` | Present | Indirect calls |
| `Instr::Struct`, `Instr::Field`, `Instr::Index` | Present | Aggregate operations |
| Ownership instructions (Borrow, BorrowMut, etc.) | Present | tMIR-specific runtime |
| Atomic instructions (AtomicLoad, AtomicStore, etc.) | Present | Concurrency |
| `StructDef` with layout (size, align, field offsets) | Present | Struct layout |

---

## 3. Migration Plan

### Step 1: Establish Wire Format (no repo dependency)

Before the real tMIR repo exists or is accessible, LLVM2 can establish a
serialized wire format for tMIR modules:

```
tRust/tSwift/tC  -->  tMIR JSON  -->  LLVM2 reads JSON  -->  machine code
```

**Implementation:**
1. Define a JSON schema based on the current stub types (they already have serde)
2. Write a `tmir_reader` module that deserializes `Module` from JSON
3. Create hand-written JSON test fixtures for known programs (add_i32, fibonacci,
   factorial, linked_list_traverse)
4. Add end-to-end tests: JSON -> Module -> adapter -> ISel -> regalloc -> encode -> Mach-O

This approach is **independent of the tMIR repo** and provides immediate value:
LLVM2 can accept external tMIR input without any upstream dependency.

**File:** `crates/llvm2-lower/src/tmir_reader.rs`

```rust
use tmir_func::Module;
use std::path::Path;

pub fn read_tmir_json(path: &Path) -> Result<Module, TmirReadError> {
    let content = std::fs::read_to_string(path)?;
    let module: Module = serde_json::from_str(&content)?;
    Ok(module)
}

pub fn read_tmir_msgpack(data: &[u8]) -> Result<Module, TmirReadError> {
    let module: Module = rmp_serde::from_slice(data)?;
    Ok(module)
}
```

### Step 2: Build Test Fixture Library

Create a set of canonical tMIR programs as both Rust builder functions and
serialized JSON files:

```
tests/tmir_fixtures/
  add_i32.json           -- fn add(a: i32, b: i32) -> i32 { a + b }
  fibonacci.json         -- fn fib(n: i32) -> i32 (loop-based)
  factorial.json         -- fn fact(n: i32) -> i32 (recursive)
  float_ops.json         -- fn calc(a: f64, b: f64) -> f64 { a*b + a/b }
  memory_ops.json        -- fn sum(ptr: *i32, n: i32) -> i32 (array sum)
  control_flow.json      -- fn abs(x: i32) -> i32 (conditional)
  call_chain.json        -- fn foo() { bar(); baz(); }
```

Each fixture includes:
- The tMIR Module as JSON
- Expected AArch64 instruction count (approximate, for regression testing)
- Expected function signature after ABI lowering

### Step 3: Replace Stubs with Real tMIR (when available)

When the real tMIR repo becomes available:

1. **Replace Cargo.toml deps:**
   ```toml
   # Before (stubs):
   tmir-types = { path = "../stubs/tmir-types" }
   tmir-instrs = { path = "../stubs/tmir-instrs" }
   tmir-func = { path = "../stubs/tmir-func" }

   # After (real):
   tmir-types = { git = "https://github.com/dropbox-ai-prototypes/tMIR", rev = "..." }
   tmir-instrs = { git = "https://github.com/dropbox-ai-prototypes/tMIR", rev = "..." }
   tmir-func = { git = "https://github.com/dropbox-ai-prototypes/tMIR", rev = "..." }
   ```

2. **Reconcile API differences:** The real tMIR types may differ from stubs.
   Likely divergences:
   - Enum variant naming (e.g., `BinOp::Add` vs `BinaryOp::IAdd`)
   - Proof annotation structure (enum vs trait object)
   - Block representation (Vec vs indexed arena)
   - Serde format (JSON vs bincode vs custom)

3. **Add compatibility shim if needed:** If API differences are large, add a
   thin `tmir_compat.rs` module that maps real tMIR types to what the adapter
   expects. This is cheaper than rewriting the adapter.

4. **Run all existing tests** against real tMIR types. The 82 adapter tests and
   the 17 tmir_integration tests should continue to pass.

### Step 4: Implement tmir-semantics Bridge

The verification pipeline needs tMIR instruction semantics for proof obligations.
Currently `tmir_semantics.rs` in llvm2-verify encodes these independently.

When real tMIR semantics are available:
1. Import the real `tmir-semantics` crate
2. Bridge `InstrSemantics::eval` to `SmtExpr` generation
3. This ensures LLVM2's proofs use the canonical tMIR semantics (not a
   reimplementation that could diverge)

### Step 5: End-to-End Pipeline

The final integration:

```
tMIR JSON/msgpack file
  -> tmir_reader::read_tmir_json()
  -> adapter::translate_module()     -- tMIR -> internal LIR + ProofContext
  -> isel::select_function()         -- LIR -> MachFunction
  -> opt::run_pipeline()             -- optimization passes (proof-aware)
  -> regalloc::allocate()            -- register allocation
  -> codegen::encode()               -- AArch64 binary encoding
  -> macho::write_object()           -- Mach-O object file
  -> (external) ld / lld             -- linking
```

---

## 4. Adapter Changes Required

### 4.1 Current Gaps

The adapter (`crates/llvm2-lower/src/adapter.rs`) already handles most tMIR
instructions. Remaining gaps:

| Gap | Severity | Workaround |
|-----|----------|------------|
| No `tmir_reader` module (cannot read from disk) | High | Build Step 1 |
| No end-to-end test from JSON to Mach-O | High | Build Step 2 |
| `tmir-semantics` not wired into verification | Medium | Use existing tmir_semantics.rs |
| Aggregate type lowering (Struct, Field, Index) | Low | Deferred (post-MVP) |
| Ownership instruction lowering | Low | Deferred (needs runtime) |
| Switch instruction lowering | Low | Conditional branch chain suffices |
| Varargs (va_list) support | Low | va_list.rs exists but not wired to tMIR |

### 4.2 New Modules Needed

| Module | Purpose | Priority |
|--------|---------|----------|
| `tmir_reader.rs` | Deserialize tMIR from JSON/msgpack | P1 |
| `tmir_writer.rs` | Serialize internal LIR back to tMIR (for debugging) | P3 |
| `tmir_compat.rs` | Compatibility shim for real vs stub API differences | P2 (when needed) |

### 4.3 Signedness Handling

The stubs collapse signed/unsigned integers into `Ty::Int(N)` and `Ty::UInt(N)`.
The adapter currently maps both to `Type::I<N>` and relies on opcodes (SDiv vs
UDiv, Slt vs Ult) to disambiguate. This is correct and matches LLVM's approach.

The real tMIR may use a different signedness model (e.g., a separate `Signedness`
enum, or signedness baked into the opcode rather than the type). The adapter
must handle either approach.

---

## 5. Testing Strategy

### 5.1 Unit Tests (adapter translation)

Already exist: 82 tests in adapter.rs, 17 in tmir_integration test file.
These test individual instruction translations.

### 5.2 Integration Tests (JSON fixtures)

New: Read JSON tMIR module, compile through full pipeline, verify:
- Adapter produces valid internal LIR
- ISel produces valid MachIR
- Regalloc succeeds
- Encoding produces valid AArch64 bytes
- Mach-O links and (on AArch64 macOS) runs correctly

### 5.3 Round-Trip Tests

Serialize a Module to JSON, deserialize it, translate through adapter, and
verify the result matches direct translation from the builder API.

### 5.4 Proof Annotation Tests

Verify that proof annotations survive the full pipeline:
1. Create Module with NoOverflow proof on an Add
2. Compile through adapter
3. Check ProofContext contains the proof keyed to the correct VReg
4. Verify no overflow check instructions in output

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Real tMIR API diverges significantly from stubs | Medium | High | Compatibility shim (Step 3) |
| tMIR repo not created for months | High | Medium | Wire format approach (Step 1) works independently |
| Serde format incompatibility | Low | Medium | JSON is universal; msgpack needs version negotiation |
| Performance of JSON parsing for large modules | Low | Low | Switch to msgpack/bincode for production |
| Proof annotation format changes | Medium | Medium | ProofContext is already an adapter; adjust mapping |

---

## 7. Dependencies

| Dependency | Status | Required For |
|------------|--------|--------------|
| tMIR repo (ayates_dbx/tMIR) | Not accessible | Step 3 (not Step 1-2) |
| serde_json | Available | Step 1 |
| rmp-serde (msgpack) | Not in deps | Step 1 (optional, JSON suffices) |
| z4 repo (ayates_dbx/z4) | Optional | Verification with real solver |

**Key insight:** Steps 1-2 have NO external dependencies and can proceed immediately.
The wire format approach decouples LLVM2 from the tMIR repo timeline.

---

## References

- `designs/2026-04-13-tmir-integration.md` -- Original adapter layer design
- `crates/llvm2-lower/src/adapter.rs` -- Current adapter implementation
- `stubs/tmir-types/src/lib.rs` -- Type definitions
- `stubs/tmir-instrs/src/lib.rs` -- Instruction definitions
- `stubs/tmir-func/src/lib.rs` -- Function/module definitions
- `stubs/tmir-semantics/src/lib.rs` -- Semantics trait interface
- `crates/llvm2-verify/src/tmir_semantics.rs` -- Verification semantic encoder
- Issue #227: tMIR integration is stubbed

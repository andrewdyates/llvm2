# tMIR Stub Migration Plan: Type Mapping, Breaking Changes, and Migration Order

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Research Complete
**Part of:** #283 (P0), #286, #259, #266, #284

---

## Executive Summary

LLVM2's `stubs/` directory (2,791 LOC, 4 crates) must be replaced with the real tMIR
crate from `~/tMIR/crates/` (844 LOC, 4 crates). This document provides a complete
type-by-type mapping, catalogs every breaking change, estimates LOC impact per LLVM2
crate, recommends a migration order, and identifies blockers.

**Critical finding:** There are actually **three** IR representations in play, not two:

| IR | Location | LOC | Purpose |
|----|----------|-----|---------|
| LLVM2 stubs | `~/LLVM2/stubs/` | 2,791 | LLVM2's development stubs |
| Real tMIR | `~/tMIR/crates/` | 844 | Standalone tMIR crate |
| trust-types IR | `~/tRust/crates/trust-types/` | ~6K (model.rs) | tRust's verification IR |

The real tMIR crate (844 LOC) is a **minimal subset** of the LLVM2 stubs (2,791 LOC).
LLVM2's stubs are more feature-complete. The trust-types IR is a **completely different
representation** (place-based MIR, not SSA-based tMIR). A bridge is needed between
trust-types and tMIR before LLVM2 can consume tRust output.

---

## 1. Type-by-Type Mapping

### 1.1 tmir-types: Type System

| LLVM2 Stub (`tmir_types::`) | Real tMIR (`tmir_types::`) | Delta |
|------------------------------|---------------------------|-------|
| `Ty` (enum, 12 variants) | `Type` (enum, 7 variants) | **Renamed** `Ty` -> `Type`. 5 variants missing from real. |
| `Ty::Primitive(PrimitiveType)` | `Type::Primitive(PrimitiveType)` | Identical structure |
| `Ty::Array { element, len }` | `Type::Array { element, len }` | Identical (`Box<Ty>` vs `Box<Type>`) |
| `Ty::Tuple(Vec<Ty>)` | `Type::Tuple(Vec<Type>)` | Identical |
| `Ty::StructDef { name, fields }` | `Type::Struct { name, fields }` | **Renamed** `StructDef` -> `Struct` |
| `Ty::Enum { name, variants }` | `Type::Enum { name, variants }` | Identical |
| `Ty::Ref { kind, pointee }` | `Type::Ref { kind, pointee }` | Identical |
| `Ty::FnPtr { params, ret }` | `Type::FnPtr { params, ret }` | Identical |
| `Ty::Vector { element, lanes }` | **MISSING** | LLVM2 extension, not in real tMIR |
| `Ty::Struct(StructId)` | **MISSING** | LLVM2 extension (ID indirection) |
| `Ty::Func(FuncTy)` | **MISSING** | LLVM2 extension (multi-return) |
| `IntWidth::bits() -> u16` | `IntWidth::bits() -> u32` | **Return type changed** |
| `FloatWidth::bits() -> u16` | `FloatWidth::bits() -> u32` | **Return type changed** |
| `IntWidth::from_bits(u16)` | **MISSING** | LLVM2 convenience method |
| `FloatWidth::from_bits(u16)` | **MISSING** | LLVM2 convenience method |
| `PrimitiveType` | `PrimitiveType` | Identical (Int/Float/Bool/Unit/Never) |
| `Mutability` | `Mutability` | Identical (Immutable/Mutable) |
| `RefKind` | `RefKind` | Identical (Borrow/Raw/Rc) |
| `Field { name, ty: Ty }` | `Field { name, ty: Type }` | Only type name differs |
| `Variant { name, fields: Vec<Ty> }` | `Variant { name, fields: Vec<Type> }` | Only type name differs |
| `Linkage` | **MISSING** | LLVM2 extension |
| `StructId(u32)` | **MISSING** | LLVM2 extension |
| `FuncTy { params, returns }` | **MISSING** | LLVM2 extension |
| `FieldDef { name, ty, offset }` | **MISSING** | LLVM2 extension |
| `StructDef { id, name, fields, ... }` | **MISSING** | LLVM2 extension |
| `GlobalDef` | **MISSING** | LLVM2 extension |
| `DataLayout` | **MISSING** | LLVM2 extension |
| `ValueId(u32)` | **MISSING** (in tmir-instrs as `Value`) | Moved to different crate |
| `BlockId(u32)` | **MISSING** (in tmir-instrs as `BlockId`) | Moved to different crate |
| `FuncId(u32)` | **MISSING** (in tmir-instrs as `FuncId`) | Moved to different crate |
| `TmirProof` (11 variants) | **MISSING** | LLVM2 extension, not in real tMIR |

**Convenience constructors:** Real tMIR has `Type::i8()` through `Type::u64()`, `Type::bool()`,
`Type::unit()`, `Type::never()`, `Type::array()`, `Type::borrow()`, `Type::borrow_mut()`,
`Type::raw_ptr()`, `Type::raw_ptr_mut()`, `Type::rc()`. LLVM2 stubs have equivalents on `Ty`
plus additional: `int(bits)`, `uint(bits)`, `float(bits)`, `void()`, `ptr()`, `ptr_mut()`,
`vector()`.

### 1.2 tmir-instrs: Instructions

| LLVM2 Stub (`tmir_instrs::`) | Real tMIR (`tmir_instrs::`) | Delta |
|------------------------------|---------------------------|-------|
| `Instr` (enum, 35 variants) | `Instruction` (enum, 22 variants) | **Renamed** `Instr` -> `Instruction`. 13 variants missing. |
| `BinOp` (17 variants) | `BinOp` (10 variants) | 7 missing: SDiv/UDiv/SRem/URem/AShr/LShr/FAdd/FSub/FMul/FDiv all collapsed |
| `BinOp::SDiv` / `BinOp::UDiv` | `BinOp::Div` | **Merged**: single Div, signedness from operand type |
| `BinOp::SRem` / `BinOp::URem` | `BinOp::Rem` | **Merged**: single Rem |
| `BinOp::AShr` / `BinOp::LShr` | `BinOp::Shr` | **Merged**: single Shr |
| `BinOp::FAdd/FSub/FMul/FDiv` | `BinOp::Add/Sub/Mul/Div` | **Merged**: float ops use same variants as int |
| `UnOp` (5 variants) | `UnOp` (2 variants) | FNeg/FAbs/FSqrt missing from real tMIR |
| `CmpOp` (22 variants) | `CmpOp` (6 variants) | **Massively reduced**: no S/U prefix, no float comparisons |
| `CmpOp::Slt/Ult/FOlt/FUlt` etc. | `CmpOp::Lt` | Single Lt, signedness from type context |
| `CastOp` (12 variants) | `CastOp` (6 variants) | 6 missing: FPToSI/FPToUI/SIToFP/UIToFP/FPExt/FPTrunc -> IntToFloat/FloatToInt |
| `Operand` (Value/Constant) | `Operand` (Value/Constant) | **Compatible** but `Value` vs `ValueId` |
| `Constant` | `Constant` | Identical (Int/Float/Bool/Unit) |
| `InstrNode { instr, results, proofs }` | **MISSING** (result inside Instruction variants) | **Structural change** |
| `Instr::BinOp { op, ty, lhs, rhs }` | `Instruction::BinOp { result, op, lhs, rhs }` | result embedded; no `ty` field |
| `Instr::Cmp { op, ty, lhs, rhs }` | `Instruction::Cmp { result, op, lhs, rhs }` | result embedded; no `ty` field |
| `Instr::Cast { op, src_ty, dst_ty, operand }` | `Instruction::Cast { result, op, value, to_ty }` | Different field names |
| `Instr::Load { ty, ptr }` | `Instruction::Load { result, addr, ty }` | `ptr`->`addr`, result embedded |
| `Instr::Store { ty, ptr, value }` | `Instruction::Store { addr, value }` | No `ty` field |
| `Instr::Call { func, args, ret_ty }` | `Instruction::Call { results, func, args }` | No explicit ret_ty |
| `Instr::Switch { value, cases, default }` | `Instruction::Switch { value, default, cases }` | `cases: Vec<(i128, BlockId)>` not `Vec<SwitchCase>` |
| `SwitchCase { value: i64, target }` | Inline tuple `(i128, BlockId)` | **Structural change** |
| `Instr::Select { ... }` | **MISSING** | LLVM2 extension |
| `Instr::GetElementPtr { ... }` | **MISSING** | LLVM2 extension |
| `Instr::AtomicLoad/Store/Rmw/CmpXchg/Fence` | **MISSING** | LLVM2 extension |
| `MemoryOrdering` | **MISSING** | LLVM2 extension |
| `AtomicRmwOp` | **MISSING** | LLVM2 extension |
| `Instr::Borrow { ty, value }` | `Instruction::Borrow { result, value }` | No `ty`, result embedded |
| `Instr::BorrowMut { ty, value }` | `Instruction::BorrowMut { result, value }` | No `ty`, result embedded |
| `Instr::IsUnique { value }` | `Instruction::IsUnique { result, value }` | result embedded |
| `Instr::Field { ty, value, index }` | `Instruction::Field { result, value, field_idx }` | `index`->`field_idx`, no `ty` |
| `Instr::Index { ty, base, index }` | `Instruction::Index { result, value, index }` | `base`->`value`, no `ty` |
| `Instr::Const { ty, value: i64 }` | **MISSING** (use Operand::Constant) | Real tMIR has no Const instruction |
| `Instr::FConst { ty, value: f64 }` | **MISSING** (use Operand::Constant) | Real tMIR has no FConst instruction |
| `ValueId(u32)` | `Value(u32)` | **Renamed** |
| `BlockId(u32)` | `BlockId(u32)` | Identical (but defined in tmir-instrs, not tmir-types) |
| `FuncId(u32)` | `FuncId(u32)` | Identical (but defined in tmir-instrs, not tmir-types) |

### 1.3 tmir-func: Functions and Modules

| LLVM2 Stub (`tmir_func::`) | Real tMIR (`tmir_func::`) | Delta |
|----------------------------|--------------------------|-------|
| `Block { id, params, body }` | `BasicBlock { params, instructions }` | **Renamed**; no `id` in real (HashMap key instead) |
| `Block::params: Vec<(ValueId, Ty)>` | `BasicBlock::params: Vec<BlockParam>` | `BlockParam { value, ty }` struct vs tuple |
| `Block::body: Vec<InstrNode>` | `BasicBlock::instructions: Vec<Instruction>` | `InstrNode` wrapper vs bare `Instruction` |
| `Function { id, name, ty, entry, blocks, proofs }` | `Function { name, signature, blocks, entry, ... }` | Major structural differences |
| `Function::id: FuncId` | **No `id` field** | ID is HashMap key, not embedded |
| `Function::ty: FuncTy` | `Function::signature: Signature` | **Renamed** FuncTy -> Signature |
| `Function::blocks: Vec<Block>` | `Function::blocks: HashMap<BlockId, BasicBlock>` | **Vec -> HashMap** |
| `Function::proofs: Vec<TmirProof>` | **MISSING** | LLVM2 extension |
| `Module { name, functions, structs, globals, data_layout }` | `Module { name, functions, globals, ... }` | Structural differences |
| `Module::functions: Vec<Function>` | `Module::functions: HashMap<FuncId, Function>` | **Vec -> HashMap** |
| `Module::structs: Vec<StructDef>` | **MISSING** | LLVM2 extension |
| `Module::data_layout: Option<DataLayout>` | **MISSING** | LLVM2 extension |
| `Module::globals: Vec<GlobalDef>` | `Module::globals: Vec<Global>` | Different struct type |
| `builder` module (571 LOC) | **MISSING** | LLVM2 extension |
| `reader` module (240 LOC) | **MISSING** | LLVM2 extension |
| `FuncTy { params, returns }` | `Signature { params, returns }` | **Renamed** |

### 1.4 tmir-semantics

| LLVM2 Stub | Real tMIR | Delta |
|------------|-----------|-------|
| `InstrSemantics` trait (eval/result_types/ub_condition) | `Semantics` trait (preconditions/postconditions/can_trap) | **Completely different API** |
| `BitVec`, `SemValue` types | **MISSING** | LLVM2 extension |
| `ConcreteSemantics` impl | `BinOpSemantics`, `CmpOpSemantics`, etc. | Different decomposition |
| Focus: SMT evaluation | Focus: verification conditions | Different purpose |

### 1.5 trust-types (tRust's IR) vs tMIR

The trust-types IR (`~/tRust/crates/trust-types/src/model.rs`) is a **fundamentally
different representation** from tMIR. It is NOT a drop-in replacement:

| Aspect | trust-types (tRust) | tMIR | Implication |
|--------|--------------------|----|-------------|
| IR model | Place-based MIR (Assign/Rvalue) | SSA (Value/Instruction) | Requires SSA construction to bridge |
| Type system | Flat enum (`Ty::Int{width:u32}`) | Nested enum (`Type::Primitive(Int{IntWidth})`) | Adapter needed |
| Blocks | `BasicBlock{stmts, terminator}` | `BasicBlock{params, instructions}` | Different block model |
| Control flow | Separate Terminator enum | Instructions include control flow | Terminators must merge into instruction stream |
| Operands | `Copy(Place)/Move(Place)/Constant` | `Value(Value)/Constant(Constant)` | Place-to-SSA conversion |
| Functions | `VerifiableFunction` (flat body) | `Function` (SSA blocks) | Major restructuring |
| Proofs | `TrustDisposition`/`FunctionSpec` | None (LLVM2 stubs have `TmirProof`) | Proof mapping needed |

---

## 2. Breaking Changes Summary

### 2.1 Hard Breaking (compilation errors)

1. **Type renamed**: `Ty` -> `Type` throughout (affects every import)
2. **Instruction renamed**: `Instr` -> `Instruction` (affects adapter, ISel, verify)
3. **Value renamed**: `ValueId` -> `Value` (affects ~100+ references)
4. **Instruction structure**: `InstrNode` wrapper removed; `result` is inside each instruction
5. **BinOp reduced**: 17 -> 10 variants; adapter must map signedness from type context
6. **CmpOp reduced**: 22 -> 6 variants; adapter must map signedness from type context
7. **CastOp reduced**: 12 -> 6 variants; adapter must map signed/float conversions
8. **UnOp reduced**: 5 -> 2 variants; FNeg/FAbs/FSqrt need LLVM2-local definitions
9. **`bits()` return type**: `u16` -> `u32` (affects ~20 call sites)
10. **Block storage**: `Vec<Block>` -> `HashMap<BlockId, BasicBlock>` (affects iteration)
11. **Function storage**: `Vec<Function>` -> `HashMap<FuncId, Function>` (affects lookup)
12. **SwitchCase**: Struct -> tuple `(i128, BlockId)` (affects Switch handling)
13. **ID location**: `ValueId`/`BlockId`/`FuncId` move from `tmir-types` to `tmir-instrs`

### 2.2 Missing Features (must be added to real tMIR or kept as LLVM2 extensions)

1. **Vector/SIMD type** (`Ty::Vector`): Required for NEON/AVX vectorization
2. **Proof annotations** (`TmirProof`): Core to LLVM2's verification mission
3. **Atomic instructions**: AtomicLoad/Store/Rmw/CmpXchg/Fence
4. **Select instruction**: Branchless conditional (maps to CSEL on AArch64)
5. **GetElementPtr**: Typed pointer arithmetic
6. **Builder API** (571 LOC): Test infrastructure depends on it
7. **Reader/Writer API** (240 LOC): JSON wire format for CLI
8. **DataLayout**: Target-specific layout information
9. **GlobalDef**: Rich global variable definitions (linkage, initializer, alignment)
10. **StructId/StructDef/FieldDef**: Named struct resolution infrastructure
11. **FuncTy/Linkage**: Function type with multiple returns, linkage visibility
12. **InstrNode proofs**: Per-instruction proof annotations

### 2.3 Soft Breaking (semantic differences, may cause test failures)

1. **Const instruction removed**: Real tMIR uses `Operand::Constant` exclusively
2. **No `ty` on most instructions**: Real tMIR instructions don't carry explicit type fields
3. **HashMap iteration order**: Non-deterministic block/function ordering

---

## 3. LOC Impact Per Crate

| Crate | Files Affected | Est. Changed LOC | Severity | Notes |
|-------|---------------|-----------------|----------|-------|
| **llvm2-lower** | adapter.rs (5,281), isel.rs (9,015), compute_graph.rs, target_analysis.rs, types.rs, 1 test file | ~2,500 | **HIGH** | Primary consumer of all tMIR types |
| **llvm2-verify** | tmir_semantics.rs (571), lowering_proof.rs, x86_64_lowering_proofs.rs | ~400 | MEDIUM | Imports tMIR types for proof encoding |
| **llvm2-codegen** | compiler.rs, 8 test files | ~600 | MEDIUM | Mostly test files that construct tMIR |
| **llvm2-ir** | provenance.rs | ~20 | LOW | Minimal tMIR dependency |
| **llvm2-cli** | main.rs | ~30 | LOW | JSON reader import |
| **Cargo.toml** | workspace + 5 crate Cargo.tomls | ~30 | LOW | Dependency path changes |
| **stubs/** (deletion) | 4 crates, 2,791 LOC | -2,791 | N/A | Deleted entirely |
| **Total** | ~25 files | ~3,600 changed, -2,791 deleted | | Net: ~800 new LOC for LLVM2 extensions |

---

## 4. Recommended Migration Order

### Phase 0: Prerequisite — Enrich Real tMIR (in tMIR repo, not LLVM2)

Before LLVM2 can migrate, the real tMIR needs these additions from the stubs:

1. **Vector type**: Add `Type::Vector { element, lanes }` to `tmir-types`
2. **Proof annotations**: Add `TmirProof` enum (or `ProofAnnotation`) to `tmir-types`
3. **Atomic instructions**: Add 5 atomic variants to `tmir-instrs::Instruction`
4. **Select instruction**: Add `Instruction::Select` variant
5. **Builder API**: Port `tmir-func/builder.rs` (571 LOC) to real tMIR
6. **Reader/Writer**: Port `tmir-func/reader.rs` (240 LOC) to real tMIR

**Estimated effort:** ~1,000 LOC additions to real tMIR.
**Why before LLVM2:** Without these, LLVM2 cannot compile and would need conditional compilation or wrapper types, making the migration much harder.

### Phase 1: Type Compatibility Layer (in LLVM2)

**Goal:** Make LLVM2 compile against real tMIR types without changing semantics.

1. Update `Cargo.toml` workspace dependencies to point to real tMIR
2. Create `crates/llvm2-lower/src/tmir_compat.rs`:
   - Type aliases: `type Ty = tmir_types::Type;`, etc.
   - Conversion traits for renamed types
   - Extension traits to add missing methods (`from_bits`, `scalar_bytes`, etc.)
3. Fix all `Ty` -> `Type` renames across the codebase
4. Fix `ValueId` -> `Value` renames
5. Fix `Instr` -> `Instruction` renames

**Files:** ~15 files, ~500 LOC changes + 100 LOC new compat layer
**Risk:** LOW — mechanical renaming, no semantic changes

### Phase 2: Instruction Adapter Update (in LLVM2)

**Goal:** Handle the reduced BinOp/CmpOp/CastOp/UnOp enums.

1. Update `adapter.rs` to map `BinOp::Div` -> SDiv/UDiv based on operand signedness
2. Update `adapter.rs` to map `CmpOp::Lt` -> Slt/Ult based on operand signedness
3. Update `adapter.rs` to map `CastOp::IntToFloat` -> SIToFP/UIToFP based on signedness
4. Handle `InstrNode` removal: extract `result` from each instruction variant
5. Handle HashMap-based blocks: update iteration in adapter and ISel

**Files:** adapter.rs (major), isel.rs (moderate), tmir_integration.rs tests
**Risk:** MEDIUM — semantic mapping requires careful type tracking

### Phase 3: Test Migration (in LLVM2)

**Goal:** All tests pass against real tMIR.

1. Update all test helper code to use new type constructors
2. Update builder usage (if builder was ported to real tMIR)
3. Update JSON reader tests for any format changes
4. Fix HashMap ordering issues in assertion order

**Files:** ~10 test files across llvm2-lower, llvm2-codegen, llvm2-verify
**Risk:** MEDIUM — HashMap non-determinism may cause test flakiness

### Phase 4: Cleanup

1. Delete `stubs/` directory
2. Remove `stubs/` from workspace members
3. Remove compatibility shims that are no longer needed
4. Update CLAUDE.md architecture description

**Files:** Cargo.toml, CLAUDE.md, stubs/ deletion
**Risk:** LOW

---

## 5. Blockers

### 5.1 Hard Blockers (must resolve before migration)

| # | Blocker | Owner | Issue |
|---|---------|-------|-------|
| B1 | Real tMIR has no Vector type — LLVM2 vectorization is broken without it | tMIR team | #286 |
| B2 | Real tMIR has no proof annotations — LLVM2's core value prop is verification | tMIR team | #286 |
| B3 | Real tMIR has no builder API — 22+ test files use `FunctionBuilder` | tMIR team | N/A |
| B4 | Real tMIR has no reader/writer — CLI pipeline depends on JSON I/O | tMIR team | N/A |
| B5 | `InstrNode` -> embedded-result change requires simultaneous adapter refactor | LLVM2 | #283 |

### 5.2 Soft Blockers (can work around)

| # | Blocker | Workaround |
|---|---------|------------|
| S1 | `bits()` returns u32 not u16 | Change LLVM2 internal types to u32 (cleaner anyway) |
| S2 | HashMap ordering for blocks | Sort by BlockId in iteration |
| S3 | No GlobalDef/DataLayout/Linkage | Keep as LLVM2-local types |
| S4 | No atomics in real tMIR | Keep as LLVM2-extension instruction variants |

### 5.3 Cross-Repo Blockers

| # | Blocker | Depends On |
|---|---------|-----------|
| C1 | tRust does NOT produce tMIR — it produces `VerifiableFunction` | trust-llvm2-bridge crate (#266) |
| C2 | trust-mir-extract API is `pub(crate)` — bridge needs access | #284 |
| C3 | tRust's `Ty` (flat `Int{width:u32}`) != tMIR's `Type` (nested `Primitive(Int{IntWidth})`) | Bridge crate handles conversion |
| C4 | tRust's place-based IR needs SSA construction to become tMIR | Bridge crate (#266) |

---

## 6. Recommended Strategy: Converge on Enriched tMIR

Based on the analysis, the optimal strategy is:

1. **Enrich the real tMIR crate** with LLVM2's extensions (Vector, proofs, atomics,
   builder, reader) — this makes tMIR the true "universal IR" it's designed to be.

2. **Migrate LLVM2 in phases** (1-4 above) after tMIR is enriched.

3. **Build the trust-llvm2-bridge** (in tRust) in parallel with steps 1-2. This crate
   converts `VerifiableFunction` -> tMIR JSON for LLVM2 consumption.

**Alternative rejected:** Keeping LLVM2 stubs as a "local fork" of tMIR. This was
considered but rejected because it defeats the purpose of tMIR as a shared IR — changes
to tMIR would need to be manually synchronized, and the bridge crate would need to
target two different type systems.

---

## 7. Estimated Timeline (in commits)

| Phase | Est. Commits | Prerequisite |
|-------|-------------|-------------|
| Phase 0 (enrich tMIR) | 5-10 | None |
| Phase 1 (type compat) | 3-5 | Phase 0 |
| Phase 2 (instruction adapter) | 5-8 | Phase 1 |
| Phase 3 (test migration) | 3-5 | Phase 2 |
| Phase 4 (cleanup) | 1-2 | Phase 3 |
| Bridge crate | 5-10 | Phase 0 (parallel with 1-4) |
| **Total** | **22-40** | |

---

## 8. Key Design Decisions

### 8.1 Where do LLVM2 extensions live?

Three options considered:

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| A: In real tMIR | Single source of truth | tMIR grows to serve LLVM2 needs | **Preferred** for Vector, Proofs, Select |
| B: In LLVM2 wrapper types | No tMIR changes needed | Two type systems to maintain | **Use for** StructId, DataLayout, FuncTy |
| C: Feature-gated in tMIR | Clean separation | Complex conditional compilation | **Rejected** — too complex |

**Decision:** Universally useful types (Vector, proofs, atomics, Select, builder, reader)
go into real tMIR. LLVM2-specific infrastructure (StructId indirection, DataLayout,
multi-return FuncTy, cost model types) stays in LLVM2 as local types.

### 8.2 InstrNode vs embedded results

The real tMIR embeds `result: Value` inside each instruction variant. LLVM2's stub uses
an `InstrNode { instr, results, proofs }` wrapper. The wrapper is more uniform for the
adapter (one place to extract results/proofs) but doesn't match the real IR.

**Decision:** Migrate to real tMIR's embedded-result model. Add a helper trait:
```rust
trait InstructionExt {
    fn results(&self) -> Vec<Value>;
    fn proofs(&self) -> &[TmirProof]; // If proof annotations added to real tMIR
}
```

### 8.3 Signedness resolution

Real tMIR uses a single `BinOp::Div` where LLVM2 stubs have `SDiv`/`UDiv`. The adapter
must resolve signedness from the operand type context.

**Decision:** The adapter already tracks operand types through the lowering pipeline.
Add a helper `fn resolve_signed_op(op: BinOp, signed: bool) -> LirOpcode` in the adapter
that maps tMIR's collapsed ops to LLVM2's internal signed/unsigned distinction.

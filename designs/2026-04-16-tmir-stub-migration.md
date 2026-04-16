# tMIR Stub Migration Plan: Type Mapping, Breaking Changes, and Migration Order

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Research Complete (CORRECTED 2026-04-16, see Corrections section at bottom)
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

| LLVM2 Stub (`tmir_types::`) | Real tMIR (`tmir::`) | Delta |
|------------------------------|---------------------------|-------|
| `Ty` (enum, 12 variants) | `Ty` (enum, 13 variants: I8/I16/I32/I64/I128/F32/F64/Bool/Ptr/Void/Struct/Array/Func) | **CORRECTED:** Real tMIR uses `Ty` (same name), flat enum, NOT `Type` with nested enums. |
| `Ty::Primitive(PrimitiveType)` | `Ty::I8`, `Ty::I32`, `Ty::F64`, etc. | **CORRECTED:** Flat variants, no nested `PrimitiveType`. |
| `Ty::Array { element, len }` | `Ty::Array(TyId, u64)` | Uses `TyId` indirection, not `Box<Ty>` |
| `Ty::Struct(StructId)` | `Ty::Struct(StructId)` | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `Ty::Func(FuncTy)` | `Ty::Func(FuncTyId)` | **CORRECTED:** Present in real tMIR via `FuncTyId` indirection. |
| `Ty::Vector { element, lanes }` | **MISSING** | LLVM2 extension, not in real tMIR |
| `Ty::Tuple(Vec<Ty>)` | **MISSING** | Not in real tMIR |
| `Ty::Enum { name, variants }` | **MISSING** | Not in real tMIR |
| `Ty::Ref { kind, pointee }` | **MISSING** | Not in real tMIR (has `Ty::Ptr` instead) |
| `Ty::FnPtr { params, ret }` | **MISSING** | Not in real tMIR (use `Ty::Func(FuncTyId)`) |
| `StructId(u32)` | `StructId(u32)` | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `FuncTy { params, returns }` | `FuncTy { params, returns, is_vararg }` | **CORRECTED:** Present in real tMIR with `is_vararg` field. |
| `FieldDef { name, ty, offset }` | `FieldDef { name, ty, offset }` | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `StructDef { id, name, fields, ... }` | `StructDef { id, name, fields, size, align }` | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `Linkage` | **MISSING** | LLVM2 extension |
| `GlobalDef` | `Global { name, ty, mutable, initializer }` | Simpler struct in real tMIR |
| `DataLayout` | **MISSING** | LLVM2 extension |
| `ValueId(u32)` | `ValueId(u32)` | **CORRECTED:** Same name in real tMIR (was listed as renamed to `Value`). |
| `BlockId(u32)` | `BlockId(u32)` | Identical, in `tmir::value` module |
| `FuncId(u32)` | `FuncId(u32)` | Identical, in `tmir::value` module |
| `TyId(u32)` | `TyId(u32)` | Present in real tMIR (new typed ID wrapper) |
| `FuncTyId(u32)` | `FuncTyId(u32)` | Present in real tMIR (new typed ID wrapper) |
| `ProofId(u32)` | `ProofId(u32)` | Present in real tMIR (new typed ID wrapper) |
| `TmirProof` (11 variants) | `ProofAnnotation` (21 variants) | **CORRECTED:** Real tMIR has extensive proof system (was listed as MISSING). See proof.rs. |

**Note on type system:** Real tMIR uses a flat type enum (`Ty::I32`, `Ty::Ptr`, etc.) without
the nested `PrimitiveType`/`IntWidth`/`FloatWidth` hierarchy. This is actually closer to
what LLVM2's internal LIR types use, making the migration simpler than the nested-enum
design would suggest. No `IntWidth::bits()` or `FloatWidth::bits()` methods exist;
`Ty::bit_width() -> Option<u32>` is the replacement.

### 1.2 tmir-instrs: Instructions

| LLVM2 Stub (`tmir_instrs::`) | Real tMIR (`tmir::inst`) | Delta |
|------------------------------|---------------------------|-------|
| `Instr` (enum, 35 variants) | `Inst` (enum, 33 variants) | **CORRECTED:** Named `Inst` (not `Instruction`), 33 variants (not 22). Very close to stubs. |
| `BinOp` (17 variants) | `BinOp` (18 variants) | **CORRECTED:** Real tMIR has 18 variants including SDiv/UDiv/SRem/URem/LShr/AShr/FAdd/FSub/FMul/FDiv/FRem. NOT collapsed. |
| `UnOp` (5 variants) | `UnOp` (3 variants: Neg, FNeg, Not) | **CORRECTED:** 3 variants (was listed as 2). FNeg IS present. |
| `OverflowOp` (N/A in stubs) | `OverflowOp` (3 variants: AddOverflow, SubOverflow, MulOverflow) | New in real tMIR, not in stubs. |
| `CmpOp` (22 variants unified) | Split: `ICmpOp` (10) + `FCmpOp` (12) | **CORRECTED:** 22 total variants (was listed as 6). S/U prefixed. NOT collapsed. |
| `CastOp` (12 variants) | `CastOp` (12 variants: Trunc/ZExt/SExt/FPTrunc/FPExt/FPToUI/FPToSI/UIToFP/SIToFP/PtrToInt/IntToPtr/Bitcast) | **CORRECTED:** All 12 variants present (was listed as 6). Identical to stubs. |
| `Constant` | `Constant` (Int(i128)/Float(f64)/Bool/Aggregate) | Similar; Aggregate replaces Unit |
| `InstrNode { instr, results, proofs }` | `InstrNode { inst, results, proofs, span }` | **CORRECTED:** InstrNode EXISTS in real tMIR (was listed as MISSING). Separate results vec, same pattern as stubs. |
| `Inst::BinOp { op, ty, lhs, rhs }` | `Inst::BinOp { op, ty, lhs, rhs }` | **CORRECTED:** Has `ty` field. Results via InstrNode, not embedded. |
| `Inst::ICmp { op, ty, lhs, rhs }` | `Inst::ICmp { op, ty, lhs, rhs }` | **CORRECTED:** Separate ICmp/FCmp (not unified Cmp). Has `ty` field. |
| `Inst::FCmp { op, ty, lhs, rhs }` | `Inst::FCmp { op, ty, lhs, rhs }` | New split from stubs' unified Cmp. |
| `Inst::Cast { op, src_ty, dst_ty, operand }` | `Inst::Cast { op, src_ty, dst_ty, operand }` | **CORRECTED:** Same field names as stubs. |
| `Inst::Load { ty, ptr }` | `Inst::Load { ty, ptr }` | **CORRECTED:** Same fields (ptr, not addr). |
| `Inst::Store { ty, ptr, value }` | `Inst::Store { ty, ptr, value }` | **CORRECTED:** HAS `ty` field (was listed as not having one). |
| `Inst::Call { func, args, ret_ty }` | `Inst::Call { callee, args }` | `func` -> `callee` (FuncId). No explicit ret_ty (use module's func_types table). |
| `Inst::CallIndirect` | `Inst::CallIndirect { callee, sig, args }` | Present. `callee` is ValueId, `sig` is FuncTyId. |
| `Inst::Switch` | `Inst::Switch { value, default, default_args, cases }` | `cases: Vec<SwitchCase>` (not inline tuples). Has `default_args`. |
| `SwitchCase` | `SwitchCase { value: Constant, target, args }` | **CORRECTED:** Proper struct (was listed as inline tuple). Has `args`. |
| `Inst::Select { ... }` | `Inst::Select { ty, cond, then_val, else_val }` | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `Inst::GEP { ... }` | `Inst::GEP { pointee_ty, base, indices }` | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `Inst::AtomicLoad/Store/Rmw/CmpXchg/Fence` | All 5 present with `Ordering` enum (5 variants) | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `AtomicRmwOp` | `AtomicRMWOp` (10 variants: Xchg/Add/Sub/And/Or/Xor/Max/Min/UMax/UMin) | **CORRECTED:** Present in real tMIR (was listed as MISSING). |
| `Ordering` | `Ordering` (5 variants: Relaxed/Acquire/Release/AcqRel/SeqCst) | **CORRECTED:** Present (was listed as MISSING). |
| `Inst::ExtractField` | `Inst::ExtractField { ty, aggregate, field }` | Present (replaces stubs' `Field`). |
| `Inst::InsertField` | `Inst::InsertField { ty, aggregate, field, value }` | Present. |
| `Inst::ExtractElement` | `Inst::ExtractElement { ty, array, index }` | Present (replaces stubs' `Index`). |
| `Inst::InsertElement` | `Inst::InsertElement { ty, array, index, value }` | Present. |
| `Inst::Const { ty, value }` | `Inst::Const { ty, value: Constant }` | **CORRECTED:** Present in real tMIR (was listed as MISSING). Value is `Constant` enum. |
| `Inst::NullPtr` | `Inst::NullPtr` | Present. |
| `Inst::Undef { ty }` | `Inst::Undef { ty }` | Present. |
| `Inst::Assume/Assert/Unreachable` | All 3 present | Present. |
| `Inst::Copy { ty, operand }` | `Inst::Copy { ty, operand }` | Present. |
| `Inst::Overflow { op, ty, lhs, rhs }` | Present (new) | Not in stubs; `OverflowOp` for checked arithmetic. |
| Stubs: `Borrow`/`BorrowMut`/`IsUnique` | **MISSING** from real tMIR | Ownership instructions are LLVM2 stub extensions, not in real tMIR. |
| Stubs: `Retain`/`Release`/`EndBorrow`/`Dealloc`/`Nop` | **MISSING** from real tMIR | Ownership/lifecycle instructions are LLVM2 stub extensions. |
| `ValueId(u32)` | `ValueId(u32)` | **CORRECTED:** Same name (was listed as renamed to `Value`). |
| `BlockId(u32)` | `BlockId(u32)` | Identical, in `tmir::value` module. |
| `FuncId(u32)` | `FuncId(u32)` | Identical, in `tmir::value` module. |

### 1.3 tmir-func: Functions and Modules

| LLVM2 Stub (`tmir_func::`) | Real tMIR (`tmir::`) | Delta |
|----------------------------|--------------------------|-------|
| `Block { id, params, body }` | `Block { id, params, body }` | **CORRECTED:** Same structure! Named `Block` (not `BasicBlock`), HAS `id` field. |
| `Block::params: Vec<(ValueId, Ty)>` | `Block::params: Vec<(ValueId, Ty)>` | **CORRECTED:** Same tuple format (not `BlockParam` struct). |
| `Block::body: Vec<InstrNode>` | `Block::body: Vec<InstrNode>` | **CORRECTED:** Uses `InstrNode` wrapper (was listed as bare `Instruction`). |
| `Function { id, name, ty, entry, blocks, proofs }` | `Function { id, name, ty, entry, blocks, proofs }` | **CORRECTED:** Nearly identical! |
| `Function::id: FuncId` | `Function::id: FuncId` | **CORRECTED:** HAS `id` field (was listed as absent). |
| `Function::ty: FuncTy` | `Function::ty: FuncTyId` | Uses `FuncTyId` indirection to module's `func_types` table. |
| `Function::blocks: Vec<Block>` | `Function::blocks: Vec<Block>` | **CORRECTED:** Uses `Vec<Block>` (was listed as HashMap). Same as stubs! |
| `Function::proofs: Vec<TmirProof>` | `Function::proofs: Vec<ProofAnnotation>` | **CORRECTED:** Present (was listed as MISSING). Uses `ProofAnnotation` name. |
| `Module { name, functions, ... }` | `Module { name, functions, structs, globals, func_types, types, proof_obligations, proof_certificates }` | Real tMIR Module is MORE feature-rich than stubs. |
| `Module::functions: Vec<Function>` | `Module::functions: Vec<Function>` | **CORRECTED:** Uses `Vec<Function>` (was listed as HashMap). Same as stubs! |
| `Module::structs: Vec<StructDef>` | `Module::structs: Vec<StructDef>` | **CORRECTED:** Present (was listed as MISSING). |
| `Module::func_types: Vec<FuncTy>` | Present (new) | Module-level function type table (accessed via `FuncTyId`). |
| `Module::types: Vec<Ty>` | Present (new) | Module-level type table (accessed via `TyId`). |
| `Module::proof_obligations` | `Vec<ProofObligation>` | Present (new) -- module-level proof tracking. |
| `Module::proof_certificates` | `Vec<ProofCertificate>` | Present (new) -- proof evidence storage. |
| `Module::data_layout: Option<DataLayout>` | **MISSING** | LLVM2 extension |
| `Module::globals: Vec<GlobalDef>` | `Module::globals: Vec<Global>` | Different struct type (simpler: name/ty/mutable/initializer) |
| `builder` module (571 LOC) | Separate `tmir-build` crate | Builder is in a companion crate, not missing. |
| `reader` module (240 LOC) | **MISSING** | LLVM2 extension (JSON reader) |
| `FuncTy { params, returns }` | `FuncTy { params, returns, is_vararg }` | **CORRECTED:** Named `FuncTy` (not `Signature`). Has `is_vararg`. |

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
| IR model | Place-based MIR (Assign/Rvalue) | SSA (ValueId/Inst) | Requires SSA construction to bridge |
| Type system | Flat enum (`Ty::Int{width:u32}`) | Flat enum (`Ty::I32`, `Ty::Ptr`, etc.) | **CORRECTED:** Both are flat; adapter is simpler than assumed |
| Blocks | `BasicBlock{stmts, terminator}` | `Block{id, params, body}` | Different block model |
| Control flow | Separate Terminator enum | Instructions include control flow | Terminators must merge into instruction stream |
| Operands | `Copy(Place)/Move(Place)/Constant` | `ValueId` references | Place-to-SSA conversion |
| Functions | `VerifiableFunction` (flat body) | `Function` (SSA blocks) | Major restructuring |
| Proofs | `TrustDisposition`/`FunctionSpec` | `ProofAnnotation` (21 variants), `ProofObligation`, `ProofCertificate` | **CORRECTED:** Real tMIR has full proof system |

---

## 2. Breaking Changes Summary

### 2.1 Hard Breaking (compilation errors)

**CORRECTED:** Many items originally listed here were based on incorrect assumptions
about the real tMIR API. The actual breaking changes are much fewer:

1. ~~**Type renamed**: `Ty` -> `Type`~~ **WRONG:** Real tMIR uses `Ty` (same name).
2. **Instruction renamed**: `Instr` -> `Inst` (not `Instruction` as originally stated)
3. ~~**Value renamed**: `ValueId` -> `Value`~~ **WRONG:** Real tMIR uses `ValueId` (same name).
4. ~~**InstrNode wrapper removed**~~ **WRONG:** `InstrNode { inst, results, proofs, span }` exists. Same pattern.
5. ~~**BinOp reduced to 10**~~ **WRONG:** Real tMIR `BinOp` has 18 variants including all S/U and F-prefixed ops.
6. **CmpOp split**: Unified `CmpOp` (22 variants) -> split `ICmpOp` (10) + `FCmpOp` (12). This IS a real change but the direction is opposite what was stated (not "reduced to 6").
7. ~~**CastOp reduced to 6**~~ **WRONG:** Real tMIR `CastOp` has all 12 variants.
8. **UnOp slightly reduced**: 5 -> 3 (Neg, FNeg, Not). FAbs/FSqrt need LLVM2-local defs.
9. **Type system is flat**: No `IntWidth::bits()` -- use `Ty::bit_width() -> Option<u32>` instead.
10. ~~**Block storage Vec->HashMap**~~ **WRONG:** Real tMIR uses `Vec<Block>` (same as stubs).
11. ~~**Function storage Vec->HashMap**~~ **WRONG:** Real tMIR uses `Vec<Function>` (same as stubs).
12. **SwitchCase enriched**: Now has `value: Constant` (not `i64`) and `args: Vec<ValueId>`.
13. **ID location**: `ValueId`/`BlockId`/`FuncId` now in unified `tmir::value` module.
14. **Call field rename**: `func` -> `callee` (FuncId), no explicit `ret_ty`.
15. **Function::ty**: Now `FuncTyId` (indirection) instead of inline `FuncTy`.
16. **New typed ID wrappers**: `TyId`, `FuncTyId`, `StructId`, `ProofId`, `ProofTag`.
17. **Ownership instructions removed**: `Borrow`/`BorrowMut`/`IsUnique`/`Retain`/`Release`/`EndBorrow`/`Dealloc`/`Nop` not in real tMIR.

### 2.2 Missing Features (must be added to real tMIR or kept as LLVM2 extensions)

**CORRECTED:** Many items originally listed here actually exist in real tMIR.
Updated list of genuinely missing features:

1. **Vector/SIMD type** (`Ty::Vector`): Required for NEON/AVX vectorization
2. ~~**Proof annotations**~~ **WRONG:** `ProofAnnotation` (21 variants), `ProofObligation`, `ProofCertificate` all exist.
3. ~~**Atomic instructions**~~ **WRONG:** All 5 atomic instruction variants exist.
4. ~~**Select instruction**~~ **WRONG:** `Inst::Select` exists.
5. ~~**GetElementPtr**~~ **WRONG:** `Inst::GEP` exists.
6. **Builder API**: Exists as separate `tmir-build` crate (not missing, just in different location).
7. **Reader/Writer API** (240 LOC): JSON wire format for CLI -- genuinely missing.
8. **DataLayout**: Target-specific layout information -- genuinely missing.
9. ~~**GlobalDef**~~ Partially present as `Global { name, ty, mutable, initializer }`. Missing: linkage, alignment, section.
10. ~~**StructId/StructDef/FieldDef**~~ **WRONG:** All exist in real tMIR.
11. **Linkage/Visibility**: Genuinely missing from real tMIR.
12. ~~**InstrNode proofs**~~ **WRONG:** `InstrNode.proofs: Vec<ProofAnnotation>` exists, plus `.span`.

### 2.3 Soft Breaking (semantic differences, may cause test failures)

**CORRECTED:** Original claims were mostly wrong.

1. ~~**Const instruction removed**~~ **WRONG:** `Inst::Const { ty, value: Constant }` exists.
2. ~~**No `ty` on most instructions**~~ **WRONG:** Most instructions carry `ty` fields.
3. ~~**HashMap iteration order**~~ **WRONG:** Real tMIR uses `Vec` (deterministic ordering).
4. **CmpOp split** (actual soft break): Code using unified `CmpOp` must split into ICmp/FCmp paths.
5. **Constant::Int(i128)**: Untyped `i128` value (stubs may have used typed constants).

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

**CORRECTED:** Most items originally listed here already exist in real tMIR.
The real tMIR (rev `f9b132a`) already has: proof annotations (21 variants),
atomics (5 instructions), Select, GEP, builder (tmir-build crate),
StructDef/FieldDef/FuncTy, and InstrNode with proofs.

Remaining genuine prerequisites:

1. **Vector type**: Add `Ty::Vector { element, lanes }` to `tmir::ty`
2. ~~**Proof annotations**~~ Already exist (21-variant `ProofAnnotation`).
3. ~~**Atomic instructions**~~ Already exist (AtomicLoad/Store/RMW/CmpXchg/Fence).
4. ~~**Select instruction**~~ Already exists (`Inst::Select`).
5. ~~**Builder API**~~ Already exists (`tmir-build` crate).
6. **Reader/Writer**: Port JSON reader/writer or rely on serde feature gate.

**Revised estimated effort:** ~100 LOC additions (Vector type only).
**Why Phase 0 is nearly unblocked:** The real tMIR is far more complete than
this design originally assumed. Migration can proceed now.

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

**CORRECTED:** Most originally listed blockers are not actually blockers.

| # | Blocker | Owner | Issue | Status |
|---|---------|-------|-------|--------|
| B1 | Real tMIR has no Vector type — LLVM2 vectorization is broken without it | tMIR team | #286 | **Still blocked** |
| ~~B2~~ | ~~Real tMIR has no proof annotations~~ | N/A | N/A | **WRONG:** `ProofAnnotation` (21 variants) exists. NOT a blocker. |
| ~~B3~~ | ~~Real tMIR has no builder API~~ | N/A | N/A | **WRONG:** `tmir-build` crate exists. NOT a blocker. |
| B4 | Real tMIR has no reader/writer — CLI pipeline depends on JSON I/O | tMIR team | N/A | **Still blocked** (but serde feature provides JSON via standard serde) |
| ~~B5~~ | ~~`InstrNode` -> embedded-result change~~ | N/A | N/A | **WRONG:** `InstrNode` exists with same pattern. NOT a blocker. |

### 5.2 Soft Blockers (can work around)

**CORRECTED:** S2 and S4 are not blockers.

| # | Blocker | Workaround | Status |
|---|---------|------------|--------|
| S1 | `Ty::bit_width()` returns `Option<u32>` | Use `Ty::bit_width()` method instead of `IntWidth::bits()` | Valid |
| ~~S2~~ | ~~HashMap ordering for blocks~~ | N/A | **WRONG:** Real tMIR uses `Vec<Block>` (deterministic). NOT a blocker. |
| S3 | No DataLayout/Linkage/Visibility | Keep as LLVM2-local types | Valid |
| ~~S4~~ | ~~No atomics in real tMIR~~ | N/A | **WRONG:** All 5 atomic instructions exist. NOT a blocker. |

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
| A: In real tMIR | Single source of truth | tMIR grows to serve LLVM2 needs | **Preferred** for Vector |
| B: In LLVM2 wrapper types | No tMIR changes needed | Two type systems to maintain | **Use for** DataLayout, Linkage, Visibility |
| C: Feature-gated in tMIR | Clean separation | Complex conditional compilation | **Rejected** -- too complex |

**Decision (CORRECTED):** Most "universally useful" types already exist in real tMIR
(proofs, atomics, Select, GEP, StructDef/FieldDef/FuncTy, builder). Only Vector type
and Reader/Writer need to be added. LLVM2-specific infrastructure (DataLayout, Linkage,
Visibility, CallingConv, cost model types) stays in LLVM2 as local types via
`tmir_compat.rs`.

### 8.2 InstrNode vs embedded results

**CORRECTED:** The real tMIR does NOT embed results inside instruction variants.
It uses `InstrNode { inst, results, proofs, span }` -- the same wrapper pattern as
the LLVM2 stubs. No migration of result handling is needed.

**Decision:** No change required. Real tMIR already uses the InstrNode wrapper.
The `results` field is `Vec<ValueId>`, `proofs` is `Vec<ProofAnnotation>`, and
`span` is `Option<SourceSpan>` for source location tracking.

### 8.3 Signedness resolution

**CORRECTED:** Real tMIR does NOT collapse signed/unsigned ops. It has explicit
`BinOp::SDiv`/`BinOp::UDiv`, `BinOp::SRem`/`BinOp::URem`, `BinOp::LShr`/`BinOp::AShr`,
as well as separate float ops (`FAdd`/`FSub`/`FMul`/`FDiv`/`FRem`).

Similarly, comparisons are split into `ICmpOp` (10 variants: Eq/Ne/Ult/Ule/Ugt/Uge/Slt/Sle/Sgt/Sge)
and `FCmpOp` (12 variants: OEq/ONe/OLt/OLe/OGt/OGe/UEq/UNe/ULt/ULe/UGt/UGe).

**Decision:** No signedness resolution is needed. The real tMIR API already distinguishes
signed vs unsigned operations, matching what LLVM2 needs for correct lowering. The adapter
only needs to map the `CmpOp` split (ICmpOp/FCmpOp) -- the compat layer already handles
this via `tmir_compat::CmpOp`.

---

## 9. Corrections Summary (2026-04-16)

This design document was corrected based on the Wave 35 R1 audit findings
(`reports/2026-04-16-v02-tmir-migration-breakdown.md`), verified against the real
tMIR source (`~/tMIR/crates/tmir/`, rev `f9b132a`). The original design was based
on an earlier or hypothetical version of the tMIR API that no longer matches reality.

### 8 Factual Errors Corrected

| # | Original Claim | Corrected Fact | Source |
|---|---------------|----------------|--------|
| 1 | "Real tMIR has no proof annotations" (`TmirProof` listed as MISSING) | `ProofAnnotation` enum with 21 variants, plus `ProofObligation`, `ProofCertificate`, `ProofEvidence`, `ProofSummary` | `tmir/src/proof.rs` |
| 2 | "Real tMIR has no atomics" (AtomicLoad/Store/Rmw/CmpXchg/Fence listed as MISSING) | All 5 atomic instruction variants exist, with `Ordering` (5 variants) and `AtomicRMWOp` (10 variants) | `tmir/src/inst.rs:182-210` |
| 3 | "Real tMIR has no Select" (listed as MISSING/LLVM2 extension) | `Inst::Select { ty, cond, then_val, else_val }` exists | `tmir/src/inst.rs:291-297` |
| 4 | "Real tMIR has no GEP" (listed as MISSING/LLVM2 extension) | `Inst::GEP { pointee_ty, base, indices }` exists | `tmir/src/inst.rs:175-179` |
| 5 | "Real tMIR uses HashMap for blocks/functions" | Real tMIR uses `Vec<Block>` and `Vec<Function>`, same as stubs | `tmir/src/lib.rs:100,30` |
| 6 | "Real tMIR embeds result inside each instruction" (InstrNode listed as MISSING) | `InstrNode { inst, results, proofs, span }` exists with separate results vec | `tmir/src/node.rs:11-16` |
| 7 | "BinOp reduced to 10 variants" (collapsed Div/Rem/Shr/float ops) | `BinOp` has 18 variants: Add/Sub/Mul/UDiv/SDiv/URem/SRem/FAdd/FSub/FMul/FDiv/FRem/And/Or/Xor/Shl/LShr/AShr | `tmir/src/inst.rs:11-30` |
| 8 | "CmpOp reduced to 6 variants" (collapsed signedness) | Split into `ICmpOp` (10: Eq/Ne/Ult/Ule/Ugt/Uge/Slt/Sle/Sgt/Sge) + `FCmpOp` (12: OEq/ONe/OLt/OLe/OGt/OGe/UEq/UNe/ULt/ULe/UGt/UGe) = 22 total | `tmir/src/inst.rs:49-78` |

### Additional Corrections

- Type system: Named `Ty` (not `Type`), flat enum (I8/I16/I32/.../Ptr/Void), not nested enums
- Instruction enum: Named `Inst` (not `Instruction`), 33 variants (not 22)
- ValueId: Same name in real tMIR (not renamed to `Value`)
- StructDef/FieldDef/FuncTy: All exist in real tMIR (were listed as MISSING)
- CastOp: 12 variants (same as stubs), not 6
- UnOp: 3 variants (not 2); FNeg IS present
- Builder: Exists as `tmir-build` crate
- Function has `id: FuncId` field (was listed as absent)
- Block named `Block` (not `BasicBlock`), has `id` field

### Impact on Migration Plan

The real tMIR is **much closer to the LLVM2 stubs** than this design assumed.
This means:
- Phase 0 is nearly unblocked (only Vector type needed)
- Phase 1 (type compat) is much simpler than estimated
- Phase 2 (instruction adapter) requires fewer changes
- The InstrNode pattern is identical, eliminating a major refactor
- Signedness resolution is not needed (ops are already split)
- HashMap iteration concerns are moot (Vec is used)

**Superseded by:** `reports/2026-04-16-v02-tmir-migration-breakdown.md`

# LLVM2 tMIR Integration Design

**Date:** 2026-04-13
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #29 (Phase 4: Apple AArch64 ABI + instruction selection)

---

## Implementation Status (as of 2026-04-15)

**Overall: Adapter layer is implemented but consumes tMIR development stubs, not the real tMIR crate. The adapter, ISel, and ABI lowering are functional against stub types.**

| Component | Status | Details |
|-----------|--------|---------|
| **tMIR adapter** (`llvm2-lower/adapter.rs`) | IMPLEMENTED (against stubs) | 4.8K LOC. Module/function/type translation, proof extraction from tMIR stub types. |
| **tMIR stubs** (`stubs/tmir-*`) | DEVELOPMENT STUBS | `tmir-types`, `tmir-instrs`, `tmir-func`, `tmir-semantics` provide type definitions for development. Not the real tMIR crate. |
| **ISel consuming tMIR** (`llvm2-lower/isel.rs`) | IMPLEMENTED (against stubs) | 8.9K LOC. SSA tree-pattern ISel works with adapter output. |
| **ABI lowering** (`llvm2-lower/abi.rs`) | IMPLEMENTED | 4.2K LOC. Apple AArch64 calling convention. |
| **Proof annotation consumption** (`llvm2-lower/target_analysis.rs`) | IMPLEMENTED (against stubs) | Reads tMIR proof annotations (NoOverflow, InBounds, etc.) to guide optimization. Annotations come from stubs, not real proofs. |
| **Real tMIR integration** | NOT DONE | Blocked on tMIR repo providing stable Rust API. See #227 (P1). |
| **JSON wire format** | NOT DONE | No external tMIR input path. See #237. |

---

## Overview

LLVM2's instruction selector (`crates/llvm2-lower/src/isel.rs`) currently consumes
an internal LIR (`Opcode`, `Instruction`, `Value`, `Block`) defined in
`crates/llvm2-lower/src/instructions.rs`. This document designs the adapter layer
that replaces the internal LIR with real tMIR types from the `tmir-*` crates,
maps tMIR proof annotations to optimization opportunities, and defines the MVP
subset of tMIR instructions needed for end-to-end compilation.

**Goals:**
- Replace internal LIR types with tMIR types (or thin adapter)
- Consume tMIR proof annotations to enable optimizations LLVM cannot do
- Define the minimal tMIR subset for compiling `fn add(a: i32, b: i32) -> i32`
- Preserve the two-phase ISel architecture (tree match, then late combines)

**Non-goals:**
- Aggregate types (struct, array) -- deferred to post-MVP
- Ownership instructions (Borrow, BorrowMut, Retain, Release) -- runtime support needed
- Switch statements -- deferred (conditional branches suffice for MVP)
- Indirect calls -- deferred to post-MVP

---

## 1. Current State Analysis

### Internal LIR Types (to be replaced)

| Internal LIR (`instructions.rs`) | Purpose |
|-----------------------------------|---------|
| `Value(u32)` | SSA value reference |
| `Block(u32)` | Basic block reference |
| `Opcode` (30 variants) | Instruction opcodes |
| `Instruction { opcode, args, results }` | SSA instruction |
| `IntCC` (10 variants) | Integer comparison predicates |
| `FloatCC` (8 variants) | Float comparison predicates |

Source: `crates/llvm2-lower/src/instructions.rs`

### tMIR Types (source of truth)

| tMIR Type | Crate | Purpose |
|-----------|-------|---------|
| `ValueId(u32)` | `tmir-types` | SSA value reference |
| `BlockId(u32)` | `tmir-types` | Basic block reference |
| `FuncId(u32)` | `tmir-types` | Function reference |
| `Ty` (9 variants) | `tmir-types` | Type system (Bool, Int, UInt, Float, Ptr, Array, Struct, Func, Void) |
| `Instr` (27 variants) | `tmir-instrs` | Instruction set |
| `InstrNode { instr, results }` | `tmir-instrs` | Instruction with result values |
| `Block { id, params, body }` | `tmir-func` | Basic block with params and instructions |
| `Function { id, name, ty, entry, blocks }` | `tmir-func` | Function definition |
| `Module { name, functions, structs }` | `tmir-func` | Translation unit |

Source: `stubs/tmir-types/src/lib.rs`, `stubs/tmir-instrs/src/lib.rs`, `stubs/tmir-func/src/lib.rs`

### Key Structural Differences

1. **Type system:** Internal LIR uses a flat `Type` enum (I8-I128, F32, F64, B1).
   tMIR's `Ty` is richer: signed/unsigned integers, pointers, arrays, structs,
   function types, void.

2. **Instruction encoding:** Internal LIR uses `Opcode` enum with instruction
   variants. tMIR uses a richer `Instr` enum where each variant carries its
   operands directly (e.g., `BinOp { op, ty, lhs, rhs }`), eliminating the
   separate `args` vec.

3. **Block parameters:** tMIR blocks have explicit `params: Vec<(ValueId, Ty)>`,
   functioning as phi nodes. The internal LIR uses separate `Phi` instructions
   (though not yet implemented in isel.rs).

4. **Comparison predicates:** Internal LIR splits `IntCC`/`FloatCC`. tMIR unifies
   them in `CmpOp` (22 variants including ordered/unordered float comparisons).

5. **Missing from internal LIR:** Cast operations, ownership instructions
   (Borrow, BorrowMut, EndBorrow, Retain, Release, IsUnique), aggregate
   operations (Struct, Field, Index), Alloc/Dealloc, Switch.

---

## 2. Architecture Decision: Adapter Layer vs. Direct Consumption

### Option A: Thin adapter (translate tMIR -> internal LIR, then ISel)

- Pro: Minimal changes to existing ISel code
- Pro: Decouples ISel from tMIR API changes
- Con: Extra translation pass, losing tMIR proof annotations
- Con: Maintaining two IR definitions is technical debt

### Option B: Direct consumption (ISel operates on tMIR types)

- Pro: No translation overhead
- Pro: Proof annotations available during pattern matching
- Con: ISel code must change significantly
- Con: Tighter coupling to tMIR API

### Decision: **Option A (Adapter) for MVP, evolve toward Option B**

Rationale: The existing ISel is 1700+ lines of working code with comprehensive
tests. Rewriting it to consume tMIR directly is high-risk. Instead:

1. Build an adapter module (`tmir_adapter.rs`) that translates
   `tmir_func::Function` -> `llvm2_lower::function::Function` (internal LIR)
2. Carry proof annotations as metadata alongside the translation
3. The ISel consumes internal LIR as before, but optimization passes can
   query the proof metadata
4. Post-MVP: gradually inline the adapter and have ISel consume tMIR directly

This gives us end-to-end compilation quickly while preserving the option to
tighten the integration later.

---

## 3. tMIR Type Mapping

### Scalar Types: `tmir_types::Ty` -> `llvm2_lower::types::Type`

| tMIR `Ty` | LLVM2 `Type` | Notes |
|------------|-------------|-------|
| `Ty::Bool` | `Type::B1` | 1-bit boolean |
| `Ty::Int(8)` | `Type::I8` | Signed 8-bit |
| `Ty::Int(16)` | `Type::I16` | Signed 16-bit |
| `Ty::Int(32)` | `Type::I32` | Signed 32-bit |
| `Ty::Int(64)` | `Type::I64` | Signed 64-bit |
| `Ty::Int(128)` | `Type::I128` | Signed 128-bit |
| `Ty::UInt(8)` | `Type::I8` | **Signedness lost** -- ISel treats as unsigned when instructed by opcode |
| `Ty::UInt(16)` | `Type::I16` | Same |
| `Ty::UInt(32)` | `Type::I32` | Same |
| `Ty::UInt(64)` | `Type::I64` | Same |
| `Ty::Float(32)` | `Type::F32` | IEEE 754 single |
| `Ty::Float(64)` | `Type::F64` | IEEE 754 double |
| `Ty::Ptr(_)` | `Type::I64` | All pointers are 64-bit on AArch64 |
| `Ty::Void` | *(no value)* | Void instructions produce no result |

**Critical note on signedness:** tMIR distinguishes `Int` vs `UInt`. The internal
LIR does not. The adapter must preserve signedness information as metadata so
that:
- Division selects `SDIV` vs `UDIV` correctly
- Comparisons select signed vs unsigned condition codes
- Extension selects `SXTB` vs `UXTB`

The signedness is already encoded in the tMIR opcodes (`BinOp::SDiv` vs
`BinOp::UDiv`, `CmpOp::Slt` vs `CmpOp::Ult`), so no information is lost
in practice -- the opcode disambiguates.

### Aggregate Types (post-MVP)

| tMIR `Ty` | Lowering Strategy |
|------------|-------------------|
| `Ty::Struct(StructId)` | Decompose into scalar fields via `StructDef` lookup |
| `Ty::Array(elem, len)` | Stack-allocate, access via base+index*stride |
| `Ty::Func(FuncTy)` | Function pointer = `Type::I64` |

### Register Class Assignment: `Ty` -> `RegClass`

| tMIR Type | AArch64 RegClass | Physical Regs |
|-----------|-----------------|---------------|
| `Bool` | `Gpr32` | W0-W30 |
| `Int(8)`, `Int(16)`, `Int(32)` | `Gpr32` | W0-W30 |
| `UInt(8)`, `UInt(16)`, `UInt(32)` | `Gpr32` | W0-W30 |
| `Int(64)`, `UInt(64)` | `Gpr64` | X0-X30 |
| `Int(128)`, `UInt(128)` | `Gpr64` (pair) | X0-X30 (even/odd pair) |
| `Ptr(_)` | `Gpr64` | X0-X30 |
| `Float(32)` | `Fpr32` | S0-S31 |
| `Float(64)` | `Fpr64` | D0-D31 |

---

## 4. Instruction Mapping: tMIR `Instr` -> ISel Patterns

### 4.1 Binary Operations: `Instr::BinOp`

| tMIR `BinOp` | Internal `Opcode` | AArch64 (32-bit) | AArch64 (64-bit) |
|-------------|-------------------|-------------------|-------------------|
| `Add` | `Iadd` | `ADDWrr` / `ADDWri` | `ADDXrr` / `ADDXri` |
| `Sub` | `Isub` | `SUBWrr` / `SUBWri` | `SUBXrr` / `SUBXri` |
| `Mul` | `Imul` | `MULWrrr` | `MULXrrr` |
| `SDiv` | `Sdiv` | `SDIVWrr` | `SDIVXrr` |
| `UDiv` | `Udiv` | `UDIVWrr` | `UDIVXrr` |
| `SRem` | *(missing)* | `SDIVWrr` + `MSUBWrrrr` | `SDIVXrr` + `MSUBXrrrr` |
| `URem` | *(missing)* | `UDIVWrr` + `MSUBWrrrr` | `UDIVXrr` + `MSUBXrrrr` |
| `And` | `Band` | `ANDWrr` | `ANDXrr` |
| `Or` | `Bor` | `ORRWrr` | `ORRXrr` |
| `Xor` | `Bxor` | `EORWrr` | `EORXrr` |
| `Shl` | `Ishl` | `LSLVWr` / `LSLWi` | `LSLVXr` / `LSLXi` |
| `AShr` | `Sshr` | `ASRVWr` / `ASRWi` | `ASRVXr` / `ASRXi` |
| `LShr` | `Ushr` | `LSRVWr` / `LSRWi` | `LSRVXr` / `LSRXi` |
| `FAdd` | `Fadd` | `FADDSrr` | `FADDDrr` |
| `FSub` | `Fsub` | `FSUBSrr` | `FSUBDrr` |
| `FMul` | `Fmul` | `FMULSrr` | `FMULDrr` |
| `FDiv` | `Fdiv` | `FDIVSrr` | `FDIVDrr` |

**Gaps in current ISel:** `SRem`/`URem` are missing. AArch64 has no remainder
instruction; remainder requires a `SDIV` + `MSUB` sequence (`a - (a/b)*b`).
The `MSUB` opcode needs to be added to `AArch64Opcode`.

### 4.2 Unary Operations: `Instr::UnOp`

| tMIR `UnOp` | Lowering | AArch64 |
|-------------|----------|---------|
| `Neg` | `SUB Rd, XZR/WZR, Rn` | `SUBWrr`/`SUBXrr` (with zero register) |
| `Not` | `MVN Rd, Rn` (= `ORN Rd, XZR, Rn`) | `ORNWrr`/`ORNXrr` |
| `FNeg` | `FNEG Sd/Dd, Sn/Dn` | New opcode needed: `FNEGSr`/`FNEGDr` |

**Gap:** `Neg` and `Not` unary ops are missing from the internal `Opcode` enum.
`FNEG` opcode is missing from `AArch64Opcode`.

### 4.3 Comparisons: `Instr::Cmp`

| tMIR `CmpOp` | Internal `IntCC`/`FloatCC` | AArch64 CC |
|-------------|---------------------------|------------|
| `Eq` | `IntCC::Equal` | `EQ` |
| `Ne` | `IntCC::NotEqual` | `NE` |
| `Slt` | `IntCC::SignedLessThan` | `LT` |
| `Sle` | `IntCC::SignedLessThanOrEqual` | `LE` |
| `Sgt` | `IntCC::SignedGreaterThan` | `GT` |
| `Sge` | `IntCC::SignedGreaterThanOrEqual` | `GE` |
| `Ult` | `IntCC::UnsignedLessThan` | `LO` |
| `Ule` | `IntCC::UnsignedLessThanOrEqual` | `LS` |
| `Ugt` | `IntCC::UnsignedGreaterThan` | `HI` |
| `Uge` | `IntCC::UnsignedGreaterThanOrEqual` | `HS` |
| `FOeq` | `FloatCC::Equal` | `EQ` |
| `FOne` | `FloatCC::NotEqual` | `NE` (but needs `VC` guard for NaN) |
| `FOlt` | `FloatCC::LessThan` | `MI` |
| `FOle` | `FloatCC::LessThanOrEqual` | `LS` |
| `FOgt` | `FloatCC::GreaterThan` | `GT` |
| `FOge` | `FloatCC::GreaterThanOrEqual` | `GE` |
| `FUeq` | *(missing)* | `EQ` or `VS` (needs two branches) |
| `FUne` | *(missing)* | `NE` |
| `FUlt` | *(missing)* | `LT` (unordered) |
| `FUle` | *(missing)* | `LE` (unordered) |
| `FUgt` | *(missing)* | `HI` (unordered) |
| `FUge` | *(missing)* | `HS` (unordered) |

**Gap:** Unordered float comparisons (`FU*` variants) are missing from the
internal `FloatCC`. These require special handling: `FCMP` sets `V=1` for
unordered, so `FUeq` = `EQ || VS`, typically lowered to `CSEL` chains.

### 4.4 Cast Operations: `Instr::Cast`

| tMIR `CastOp` | Internal `Opcode` | AArch64 |
|---------------|-------------------|---------|
| `ZExt` | `Uextend` | `UXTBWr`/`UXTHWr`/`MOVWrr` |
| `SExt` | `Sextend` | `SXTBWr`/`SXTHWr`/`SXTWXr` |
| `Trunc` | *(missing)* | `MOVWrr` or `AND` with mask |
| `FPToSI` | `FcvtToInt` | `FCVTZSWr`/`FCVTZSXr` |
| `FPToUI` | *(missing)* | `FCVTZUWr`/`FCVTZUXr` (new opcodes) |
| `SIToFP` | `FcvtFromInt` | `SCVTFSWr`/`SCVTFDXr` |
| `UIToFP` | *(missing)* | `UCVTFSWr`/`UCVTFDXr` (new opcodes) |
| `FPExt` | *(missing)* | `FCVT Dd, Sn` (new opcode) |
| `FPTrunc` | *(missing)* | `FCVT Sd, Dn` (new opcode) |
| `PtrToInt` | *(no-op)* | Pointer is already `I64` |
| `IntToPtr` | *(no-op)* | Integer becomes pointer via move |
| `Bitcast` | *(missing)* | `FMOV` between GPR<->FPR, or `MOV` for same-class |

**Gaps:** `Trunc`, `FPToUI`, `UIToFP`, `FPExt`, `FPTrunc`, `Bitcast` are all
missing from the internal `Opcode` enum and/or `AArch64Opcode` enum.

### 4.5 Memory Operations

| tMIR `Instr` | Internal `Opcode` | AArch64 |
|-------------|-------------------|---------|
| `Load { ty, ptr }` | `Load { ty }` | `LDRWui`/`LDRXui`/`LDRSui`/`LDRDui` |
| `Store { ty, ptr, value }` | `Store` | `STRWui`/`STRXui`/`STRSui`/`STRDui` |
| `Alloc { ty, count }` | *(missing)* | `SUB SP, SP, #size` + address computation |
| `Dealloc { ptr }` | *(missing)* | No-op on stack allocs; runtime call for heap |

### 4.6 Control Flow

| tMIR `Instr` | Internal `Opcode` | AArch64 |
|-------------|-------------------|---------|
| `Br { target, args }` | `Jump { dest }` | `B` |
| `CondBr { cond, then, else }` | `Brif { cond, then, else }` | `CMP` + `B.cc` + `B` |
| `Return { values }` | `Return` | `MOV` to ABI regs + `RET` |
| `Call { func, args, ret_ty }` | *(via select_call)* | ABI lowering + `BL` |
| `CallIndirect { callee, args }` | *(missing)* | ABI lowering + `BLR` |
| `Switch { value, cases, default }` | *(missing)* | Compare chain or jump table |

**Key difference in block arguments:** tMIR `Br` and `CondBr` carry `args`
that map to the target block's `params`. The adapter must translate these into
`COPY` pseudo-instructions that move values into the block parameter positions,
or implement explicit phi resolution during the SSA->machine lowering.

### 4.7 Constants

| tMIR `Instr` | Internal `Opcode` | AArch64 |
|-------------|-------------------|---------|
| `Const { ty, value }` | `Iconst { ty, imm }` | `MOVZ`/`MOVN`/`MOVZ+MOVK` |
| `FConst { ty, value }` | `Fconst { ty, imm }` | `FMOV` imm / constant pool load |

### 4.8 Ownership Instructions (deferred)

| tMIR `Instr` | MVP Status | Runtime Requirement |
|-------------|------------|---------------------|
| `Borrow { ty, value }` | Deferred | Pointer copy |
| `BorrowMut { ty, value }` | Deferred | Pointer copy + exclusivity check |
| `EndBorrow { borrow }` | Deferred | No-op (or debug assertion) |
| `Retain { value }` | Deferred | `atomic_fetch_add(refcount, 1)` |
| `Release { value }` | Deferred | `atomic_fetch_sub(refcount, 1)` + conditional dealloc |
| `IsUnique { value }` | Deferred | `load refcount; cmp #1` |

### 4.9 Aggregate Operations (deferred)

| tMIR `Instr` | MVP Status | Lowering Strategy |
|-------------|------------|-------------------|
| `Struct { ty, fields }` | Deferred | Store fields to struct memory layout |
| `Field { ty, value, index }` | Deferred | Load from base + field offset |
| `Index { ty, base, index }` | Deferred | Base + index * element_size |
| `Phi { ty, incoming }` | **MVP** | Block parameters (already in tMIR block params) |

---

## 5. Proof Annotations and Optimization Opportunities

tMIR carries proof annotations that enable optimizations LLVM cannot perform.
These proofs are attached to instructions or values by the source-language
compiler (tRust, tSwift, tC) and verified by z4.

### 5.1 Proof Types

The following proof types are relevant to codegen optimization. These would be
carried as metadata on `InstrNode` or as a separate proof map
(`HashMap<ValueId, Vec<Proof>>`).

```rust
/// Proof annotations attached to tMIR values/instructions.
pub enum Proof {
    /// Addition/subtraction guaranteed not to overflow.
    /// Enables: skip overflow check, use wrapping arithmetic directly.
    NoOverflow { signed: bool },

    /// Array/pointer access is within bounds.
    /// Enables: skip bounds check, use direct load/store.
    InBounds { base: ValueId, index: ValueId },

    /// Pointer is guaranteed non-null.
    /// Enables: skip null check, use load without guard page test.
    NotNull { ptr: ValueId },

    /// Borrow is guaranteed valid (lifetime within scope).
    /// Enables: skip liveness check, treat as raw pointer.
    ValidBorrow { borrow: ValueId },

    /// Value is within a specific range [lo, hi].
    /// Enables: range-based optimizations (e.g., skip sign extension if
    /// value is known non-negative, use narrow operations).
    InRange { lo: i128, hi: i128 },

    /// Division divisor is guaranteed non-zero.
    /// Enables: skip divide-by-zero check.
    NonZeroDivisor { divisor: ValueId },

    /// Shift amount is within [0, bitwidth).
    /// Enables: skip shift-amount masking.
    ValidShift { amount: ValueId, bitwidth: u16 },
}
```

### 5.2 Proof-Consuming Optimizations

| Proof | Optimization | Code Impact | AArch64 Instruction Savings |
|-------|-------------|-------------|----------------------------|
| `NoOverflow` | Eliminate overflow check branch | Remove `CMP` + `B.VS` guard | 2-3 instructions |
| `InBounds` | Eliminate bounds check | Remove `CMP` + `B.HS` + trap | 3-4 instructions |
| `NotNull` | Eliminate null check | Remove `CBZ`/`CBNZ` guard | 1-2 instructions |
| `ValidBorrow` | Lower borrow as raw pointer | Skip ARC overhead | 0 (semantic only) |
| `InRange` | Narrow operations | Use 32-bit ops for known-small values | Size/power reduction |
| `NonZeroDivisor` | Skip div-by-zero check | Remove `CBZ` + trap | 2 instructions |
| `ValidShift` | Skip shift masking | Remove `AND` amount mask | 1 instruction |

### 5.3 Integration with ISel

Proof annotations are consumed at two points:

1. **During instruction selection** (adapter layer): When translating a tMIR
   instruction, check for associated proofs. If `NoOverflow` is present on an
   `Add`, the adapter emits `Iadd` directly without inserting overflow-check
   instructions. This happens in the adapter, not the ISel itself.

2. **During optimization** (llvm2-opt): Proof metadata is carried through to
   the MachIR level. Optimization passes query the proof map to enable
   transformations:
   - **DCE pass**: Dead code from eliminated checks is removed
   - **Peephole pass**: `InRange` proofs enable width-narrowing rewrites
   - **Address-mode formation**: `InBounds` proofs allow more aggressive
     addressing mode selection (larger offsets without guard checks)

### 5.4 Proof Metadata Storage

```rust
/// Proof metadata carried through the compilation pipeline.
pub struct ProofContext {
    /// Proofs attached to specific tMIR values.
    value_proofs: HashMap<ValueId, Vec<Proof>>,
    /// Proofs attached to specific instructions (by result ValueId).
    /// After ISel, these are re-keyed to VReg.
    vreg_proofs: HashMap<VReg, Vec<Proof>>,
}
```

The `ProofContext` is constructed by the adapter layer and passed alongside the
`MachFunction` to downstream passes.

---

## 6. Adapter Layer Design

### 6.1 Module Structure

```
crates/llvm2-lower/src/
  tmir_adapter.rs       -- NEW: tMIR -> internal LIR translation
  tmir_adapter/
    type_map.rs         -- Ty -> Type conversion
    instr_map.rs        -- Instr -> Opcode + Instruction conversion
    proof_map.rs        -- Extract and carry proof annotations
    block_params.rs     -- Block parameter -> phi/copy resolution
  isel.rs              -- Existing ISel (unchanged for MVP)
  abi.rs               -- Existing ABI (unchanged)
  instructions.rs      -- Existing LIR (extended with missing opcodes)
  types.rs             -- Existing types (unchanged)
  function.rs          -- Existing function representation
```

### 6.2 Top-Level API

```rust
/// Translate a tMIR module into LLVM2 MachFunctions, ready for
/// register allocation and encoding.
pub fn compile_module(
    module: &tmir_func::Module,
) -> Result<Vec<(MachFunction, ProofContext)>, CompileError> {
    let mut results = Vec::new();
    for func in &module.functions {
        let (lir_func, proofs) = translate_function(func, &module.structs)?;
        let mach_func = select_function(&lir_func)?;
        results.push((mach_func, proofs));
    }
    Ok(results)
}

/// Translate a single tMIR function to internal LIR + proof context.
fn translate_function(
    func: &tmir_func::Function,
    structs: &[tmir_types::StructDef],
) -> Result<(Function, ProofContext), CompileError> {
    let mut adapter = TmirAdapter::new(structs);
    adapter.translate(func)
}

/// Run instruction selection on an internal LIR function.
fn select_function(func: &Function) -> Result<MachFunction, CompileError> {
    let sig = func.signature.clone();
    let mut isel = InstructionSelector::new(func.name.clone(), sig.clone());
    let entry = func.entry_block;
    isel.lower_formal_arguments(&sig, entry);
    for (&block_id, block) in &func.blocks {
        isel.select_block(block_id, &block.instructions);
    }
    Ok(isel.finalize())
}
```

### 6.3 Type Translation

```rust
/// Convert tMIR Ty to LLVM2 Type.
pub fn translate_ty(ty: &Ty) -> Result<Type, CompileError> {
    match ty {
        Ty::Bool => Ok(Type::B1),
        Ty::Int(8) | Ty::UInt(8) => Ok(Type::I8),
        Ty::Int(16) | Ty::UInt(16) => Ok(Type::I16),
        Ty::Int(32) | Ty::UInt(32) => Ok(Type::I32),
        Ty::Int(64) | Ty::UInt(64) => Ok(Type::I64),
        Ty::Int(128) | Ty::UInt(128) => Ok(Type::I128),
        Ty::Float(32) => Ok(Type::F32),
        Ty::Float(64) => Ok(Type::F64),
        Ty::Ptr(_) => Ok(Type::I64),  // All pointers are 64-bit
        Ty::Void => Err(CompileError::VoidValue),
        Ty::Struct(_) | Ty::Array(_, _) | Ty::Func(_) => {
            Err(CompileError::UnsupportedType(format!("{:?}", ty)))
        }
        _ => Err(CompileError::UnsupportedType(format!("{:?}", ty))),
    }
}
```

### 6.4 Instruction Translation

The core of the adapter: translate each `tmir_instrs::Instr` into one or more
`llvm2_lower::instructions::Instruction`.

```rust
fn translate_instr(
    &mut self,
    node: &InstrNode,
    block_id: BlockId,
) -> Result<Vec<Instruction>, CompileError> {
    match &node.instr {
        Instr::BinOp { op, ty, lhs, rhs } => {
            let opcode = match op {
                BinOp::Add => Opcode::Iadd,
                BinOp::Sub => Opcode::Isub,
                BinOp::Mul => Opcode::Imul,
                BinOp::SDiv => Opcode::Sdiv,
                BinOp::UDiv => Opcode::Udiv,
                BinOp::And => Opcode::Band,
                BinOp::Or => Opcode::Bor,
                BinOp::Xor => Opcode::Bxor,
                BinOp::Shl => Opcode::Ishl,
                BinOp::AShr => Opcode::Sshr,
                BinOp::LShr => Opcode::Ushr,
                BinOp::FAdd => Opcode::Fadd,
                BinOp::FSub => Opcode::Fsub,
                BinOp::FMul => Opcode::Fmul,
                BinOp::FDiv => Opcode::Fdiv,
                BinOp::SRem | BinOp::URem => {
                    return self.translate_remainder(op, ty, lhs, rhs, &node.results);
                }
            };
            let lhs_val = self.map_value(*lhs);
            let rhs_val = self.map_value(*rhs);
            let result = self.map_result(&node.results)?;
            Ok(vec![Instruction {
                opcode,
                args: vec![lhs_val, rhs_val],
                results: vec![result],
            }])
        }
        Instr::Const { ty, value } => {
            let llvm2_ty = translate_ty(ty)?;
            let result = self.map_result(&node.results)?;
            Ok(vec![Instruction {
                opcode: Opcode::Iconst { ty: llvm2_ty, imm: *value },
                args: vec![],
                results: vec![result],
            }])
        }
        // ... (remaining instruction translations)
        _ => todo!("translate {:?}", node.instr),
    }
}
```

### 6.5 Block Parameter Resolution

tMIR uses block parameters for SSA phi semantics. When `Br { target, args }` or
`CondBr` passes arguments to a target block, the adapter emits `COPY`
pseudo-instructions at the branch site to move values into the parameter
positions.

```rust
/// Resolve block arguments: emit COPYs for block parameter passing.
fn resolve_block_args(
    &mut self,
    target: BlockId,
    args: &[ValueId],
    target_block: &tmir_func::Block,
) -> Vec<Instruction> {
    let mut copies = Vec::new();
    for (i, (arg, (param_id, _param_ty))) in
        args.iter().zip(target_block.params.iter()).enumerate()
    {
        let src = self.map_value(*arg);
        let dst = self.map_value(*param_id);
        copies.push(Instruction {
            opcode: Opcode::Copy,  // new opcode needed
            args: vec![src],
            results: vec![dst],
        });
    }
    copies
}
```

---

## 7. Calling Convention Integration

### tMIR Function Signature -> Apple ABI

The adapter translates `tmir_types::FuncTy` into `llvm2_lower::function::Signature`
by converting each parameter and return type.

```rust
fn translate_signature(func_ty: &FuncTy) -> Result<Signature, CompileError> {
    let params: Vec<Type> = func_ty.params.iter()
        .map(|ty| translate_ty(ty))
        .collect::<Result<Vec<_>, _>>()?;
    let returns: Vec<Type> = func_ty.returns.iter()
        .map(|ty| translate_ty(ty))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Signature { params, returns })
}
```

The existing `AppleAArch64ABI::classify_params` and `classify_returns` then
handle register allocation per the Apple DarwinPCS rules. No changes needed
to the ABI module for MVP.

### Struct Return Convention

For functions returning `Ty::Struct(_)` that exceed 16 bytes, the adapter
must insert an implicit `sret` pointer parameter:
1. Caller allocates stack space for the return struct
2. Address passed in X8 (implicit first argument)
3. Callee stores result via the X8 pointer

This is post-MVP but the adapter should detect large struct returns and
emit `CompileError::UnsupportedType` for now.

---

## 8. MVP Subset Definition

### Phase 1: Arithmetic (minimum for `fn add(a: i32, b: i32) -> i32`)

| tMIR Instructions | Count | Required |
|-------------------|-------|----------|
| `Const` | 1 | Yes |
| `BinOp::Add` | 1 | Yes |
| `Return` | 1 | Yes |
| **Total** | **3** | |

### Phase 2: Comparisons and Control Flow

| tMIR Instructions | Count | Required |
|-------------------|-------|----------|
| `Cmp` (integer: Eq, Ne, Slt, Sge) | 4 | Yes |
| `CondBr` | 1 | Yes |
| `Br` | 1 | Yes |
| `BinOp::Sub` | 1 | Yes |
| **Cumulative Total** | **10** | |

### Phase 3: Full Arithmetic + Memory

| tMIR Instructions | Count | Required |
|-------------------|-------|----------|
| `BinOp` (all 17 variants) | 17 | Yes |
| `UnOp` (Neg, Not, FNeg) | 3 | Yes |
| `Cmp` (all 22 variants) | 22 | Yes |
| `Cast` (ZExt, SExt, Trunc, FPToSI, SIToFP) | 5 | Yes |
| `Load` | 1 | Yes |
| `Store` | 1 | Yes |
| `Alloc` | 1 | Yes |
| `Call` | 1 | Yes |
| `FConst` | 1 | Yes |
| **Cumulative Total** | **~25** | |

### Phase 4: Full Feature (post-MVP)

- `Cast` (remaining: FPToUI, UIToFP, FPExt, FPTrunc, PtrToInt, IntToPtr, Bitcast)
- `CallIndirect`
- `Switch`
- `Struct`, `Field`, `Index`
- Ownership instructions (Borrow, BorrowMut, EndBorrow, Retain, Release, IsUnique)
- `Phi` (if not handled via block parameters)
- `Dealloc`

### Missing Opcodes to Add

To support the full tMIR instruction set, the following must be added:

**To `Opcode` (internal LIR):**
- `Neg` (integer negate)
- `Not` (bitwise NOT)
- `Copy` (register copy for block params)
- `Trunc { from_ty, to_ty }` (integer truncation)
- `FPToUI { dst_ty }` (float to unsigned int)
- `UIToFP { src_ty }` (unsigned int to float)
- `FPExt` (float widen)
- `FPTrunc` (float narrow)
- `Bitcast { from_ty, to_ty }` (same-size reinterpret)
- `SRem` (signed remainder)
- `URem` (unsigned remainder)
- `Alloca { ty, count }` (stack allocation)
- `CallIndirect` (indirect function call)

**To `AArch64Opcode` (machine opcodes):**
- `FNEGSr` / `FNEGDr` (floating-point negate)
- `FCVTZUWr` / `FCVTZUXr` (float to unsigned int)
- `UCVTFSWr` / `UCVTFDWr` / `UCVTFSXr` / `UCVTFDXr` (unsigned int to float)
- `FCVTSDr` / `FCVTDSr` (float precision conversion)
- `MSUBWrrrr` / `MSUBXrrrr` (multiply-subtract for remainder)
- `FMOVWSr` / `FMOVXDr` / `FMOVSWr` / `FMOVDXr` (GPR <-> FPR moves for bitcast)

---

## 9. Testing Strategy

### Unit Tests: Adapter Translation

Test each tMIR instruction translates to the expected internal LIR instruction(s).

```rust
#[test]
fn translate_tmir_add_i32() {
    let module = build_tmir_add_module();  // fn add(i32, i32) -> i32
    let (lir_func, _proofs) = translate_function(
        &module.functions[0], &module.structs
    ).unwrap();
    // Verify: 1 Iconst (if any), 1 Iadd, 1 Return
    assert_eq!(lir_func.blocks[&Block(0)].instructions.len(), ...);
}
```

### Integration Tests: End-to-End Compilation

Compile known tMIR programs through the full pipeline and verify:
1. Adapter produces correct LIR
2. ISel produces valid MachIR
3. Encoding produces valid AArch64 bytes
4. Mach-O links and runs correctly

```rust
#[test]
fn compile_tmir_add_end_to_end() {
    let module = tmir_test_programs::add_i32();
    let results = compile_module(&module).unwrap();
    assert_eq!(results.len(), 1);
    let (mach_func, _proofs) = &results[0];
    // Verify function has expected block structure and instructions
}
```

### Proof Annotation Tests

Verify that proof annotations are correctly propagated and consumed:

```rust
#[test]
fn no_overflow_eliminates_check() {
    let module = build_add_with_no_overflow_proof();
    let (mach_func, proofs) = compile_and_select(&module);
    // Verify: no CMP+B.VS overflow check in output
    let has_overflow_check = mach_func.blocks.values()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i.opcode, AArch64Opcode::Bcc));
    assert!(!has_overflow_check);
}
```

---

## 10. Migration Path

### Step 1: Add adapter module (this design)
- Create `tmir_adapter.rs` with type and instruction translation
- Add missing opcodes to internal LIR (`Opcode` and `AArch64Opcode`)
- Proof metadata types and `ProofContext`

### Step 2: Wire up the pipeline
- `compile_module()` entry point that ties tMIR -> adapter -> ISel -> MachFunction
- Integration with downstream crates (regalloc, codegen)

### Step 3: Test with real tMIR programs
- Build tMIR test suite (arithmetic, control flow, memory, calls)
- Validate against LLVM/Cranelift reference output

### Step 4: Proof-consuming optimizations (post-MVP)
- Add proof queries to llvm2-opt passes
- Verify eliminated checks via z4

### Step 5: Direct tMIR consumption (future)
- Refactor ISel to pattern-match on `tmir_instrs::Instr` directly
- Remove adapter layer and internal LIR
- The `tmir_types::Ty` becomes the canonical type throughout

---

## References

- LLVM AArch64 ISel: `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64ISelLowering.cpp`
- LLVM AArch64 ABI: `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64CallingConvention.td`
- LLVM2 AArch64 backend design: `designs/2026-04-12-aarch64-backend.md`
- LLVM2 verification architecture: `designs/2026-04-13-verification-architecture.md`
- tMIR instruction set: `stubs/tmir-instrs/src/lib.rs`
- tMIR type system: `stubs/tmir-types/src/lib.rs`
- tMIR function/module: `stubs/tmir-func/src/lib.rs`
- tMIR semantics: `stubs/tmir-semantics/src/lib.rs`
- Alive2: Lopes et al., "Alive2: Bounded Translation Validation for LLVM", PLDI 2021
- CompCert: Leroy, "Formal verification of a realistic compiler", CACM 2009

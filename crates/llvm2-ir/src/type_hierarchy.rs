// llvm2-ir/type_hierarchy.rs - Documentation of the type hierarchy across crates
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! # LLVM2 Type Hierarchy Documentation
//!
//! This module documents how machine IR types flow through the compilation
//! pipeline. It exists purely for documentation — no code is defined here.
//!
//! ## Canonical Types (this crate: `llvm2-ir`)
//!
//! `llvm2-ir` is the **single source of truth** for all shared machine IR
//! types. Other crates import from here rather than defining their own.
//!
//! | Type | Module | Description |
//! |------|--------|-------------|
//! | `MachFunction` | `function` | Complete machine function (arena-based storage) |
//! | `MachInst` | `inst` | Machine instruction (opcode + operand list) |
//! | `MachBlock` | `function` | Basic block (InstId indices, preds/succs, loop_depth) |
//! | `MachOperand` | `operand` | Operand (VReg, PReg, Imm, FImm, Block, StackSlot, FrameIndex, MemOp, Special, Symbol) |
//! | `AArch64Opcode` | `inst` | AArch64 instruction opcodes |
//! | `InstFlags` | `inst` | Instruction property flags (bitflags) |
//! | `VReg` | `regs` | Virtual register (id + RegClass) |
//! | `PReg` | `regs` | Physical register (hardware register number) |
//! | `RegClass` | `regs` | Register class (Gpr32, Gpr64, Fpr32, Fpr64, Simd128) |
//! | `BlockId` | `types` | Typed index into `MachFunction::blocks` |
//! | `InstId` | `types` | Typed index into `MachFunction::insts` |
//! | `StackSlotId` | `types` | Typed index into `MachFunction::stack_slots` |
//! | `Signature` | `function` | Function signature (param/return types) |
//! | `StackSlot` | `function` | Stack slot (size + alignment) |
//! | `Type` | `function` | LIR scalar/aggregate types |
//!
//! ## Derived Types (per-phase specializations)
//!
//! Some pipeline phases need structurally different types. These are
//! **derived** from the canonical types, not duplicates — they add or
//! remove fields for phase-specific requirements.
//!
//! ### ISel types (`llvm2-lower::isel`)
//!
//! Used during instruction selection (Phase 1). Converted to canonical
//! types via `ISelFunction::to_ir_func()`.
//!
//! | ISel type | Canonical equivalent | Why separate |
//! |-----------|---------------------|--------------|
//! | `ISelFunction` | `MachFunction` | HashMap blocks (construction-friendly); no signature in IR form |
//! | `ISelInst` | `MachInst` | No flags, no implicit defs/uses, no proof annotations |
//! | `ISelBlock` | `MachBlock` | Inline `Vec<ISelInst>` (not arena-indexed) |
//! | `ISelOperand` | `MachOperand` | Has `CondCode`/`Symbol`/`StackSlot(u32)` variants |
//! | `AArch64CC` | `cc::AArch64CC` | Includes `from_intcc`/`from_floatcc` tMIR converters |
//!
//! ### RegAlloc types (`llvm2-regalloc::machine_types`)
//!
//! Used during register allocation (Phase 4-5). Converted from canonical
//! types via `From`/`TryFrom` impls and `ir_to_regalloc()` in pipeline.
//!
//! | RegAlloc type | Canonical equivalent | Why separate |
//! |--------------|---------------------|--------------|
//! | `RegAllocFunction` | `MachFunction` | HashMap stack slots, `next_stack_slot` counter |
//! | `RegAllocInst` | `MachInst` | Separated defs/uses for liveness analysis; u16 opcode |
//! | `RegAllocBlock` | `MachBlock` | Structurally identical (unified via From impl, issue #73) |
//! | `RegAllocOperand` | `MachOperand` | Subset: omits MemOp/FrameIndex/Special |
//! | `RegAllocStackSlot` | `StackSlot` | No alignment assertion (From impl, issue #73) |
//!
//! ## Unified Types (shared across all crates)
//!
//! These types are used identically across all crates — no adapters needed:
//!
//! - **Primitive types**: `VReg`, `PReg`, `RegClass`, `BlockId`, `InstId`, `StackSlotId`
//!   (re-exported by `llvm2-regalloc::machine_types`)
//! - **InstFlags**: unified bitflags type (re-exported by `llvm2-regalloc`)
//! - **AArch64Opcode**: unified opcode enum (re-exported by `llvm2-lower::isel`)
//!
//! ## Pipeline Flow
//!
//! ```text
//! Phase 1: ISel
//!   tMIR -> ISelFunction (llvm2-lower)
//!
//! Phase 2: ISel -> IR (ISelFunction::to_ir_func())
//!   ISelFunction -> MachFunction (llvm2-ir)
//!
//! Phase 3: Optimization (llvm2-opt)
//!   MachFunction -> MachFunction (uses llvm2-ir types directly)
//!
//! Phase 4: IR -> RegAlloc (ir_to_regalloc(), uses From/TryFrom impls)
//!   MachFunction -> RegAllocFunction (llvm2-regalloc)
//!
//! Phase 5: Register Allocation (llvm2-regalloc)
//!   RegAllocFunction -> AllocationResult (VReg -> PReg map)
//!
//! Phase 6: Apply allocation (apply_regalloc())
//!   MachFunction + AllocationResult -> MachFunction (PRegs only)
//!
//! Phase 7-9: Frame lowering, encoding, Mach-O emission (llvm2-codegen)
//!   MachFunction -> Vec<u8> (uses llvm2-ir types directly)
//! ```
//!
//! ## Conversion Methods
//!
//! | Direction | Method | Location |
//! |-----------|--------|----------|
//! | ISel -> IR | `ISelFunction::to_ir_func()` | `llvm2-lower/isel.rs` |
//! | ISelOperand -> MachOperand | `convert_isel_operand_to_ir()` | `llvm2-lower/isel.rs` |
//! | IR Block -> RegAlloc Block | `From<&MachBlock> for RegAllocBlock` | `llvm2-regalloc/machine_types.rs` |
//! | IR Operand -> RegAlloc Operand | `TryFrom<&MachOperand> for RegAllocOperand` | `llvm2-regalloc/machine_types.rs` |
//! | IR StackSlot -> RegAlloc StackSlot | `From<&StackSlot> for RegAllocStackSlot` | `llvm2-regalloc/machine_types.rs` |
//! | IR Function -> RegAlloc Function | `ir_to_regalloc()` (orchestrator) | `llvm2-codegen/pipeline.rs` |
//! | Lower Type -> IR Type | `From<&Type> for llvm2_ir::Type` | `llvm2-lower/types.rs` |
//! | Lower Signature -> IR Signature | `From<&Signature> for llvm2_ir::Signature` | `llvm2-lower/function.rs` |

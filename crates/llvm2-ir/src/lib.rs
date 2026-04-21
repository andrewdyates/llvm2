// llvm2-ir - Shared machine IR model
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Shared machine IR types for LLVM2.
//!
//! This crate defines the machine-level intermediate representation used by
//! all LLVM2 backend crates. It owns: MachInst, MachFunction, MachBlock,
//! registers (VReg/PReg), operands, stack slots, typed indices, and
//! AArch64-specific types (condition codes, opcode enum).
//!
//! LLVM2 consumes tMIR (the upstream IR) via the `llvm2-lower` adapter; see
//! the *tMIR GEP stride contract* section below for the ABI-critical pointer
//! arithmetic convention shared with external consumers (z4, tla2).
//!
//! # Architecture
//!
//! ```text
//! llvm2-lower ──┐
//!               v
//!           llvm2-ir (this crate)
//!               │
//!       ┌───────┼───────┬─────────────┐
//!       v       v       v             v
//!  llvm2-opt  regalloc  llvm2-codegen  llvm2-verify
//! ```
//!
//! # Type Authority
//!
//! This crate is the SINGLE SOURCE OF TRUTH for all machine IR types.
//! Other crates (llvm2-lower, llvm2-regalloc, llvm2-opt, llvm2-codegen)
//! must import types from here rather than defining their own.
//!
//! | Type | Module |
//! |------|--------|
//! | PReg, RegClass, VReg | `regs` (delegates to `aarch64_regs`) |
//! | CondCode, ShiftType, ExtendType | `regs` (delegates to `aarch64_regs`) |
//! | MachInst, AArch64Opcode, InstFlags | `inst` |
//! | MachOperand | `operand` |
//! | MachBlock, MachFunction, Signature, StackSlot, Type | `function` |
//! | BlockId, InstId, VRegId, StackSlotId, FrameIdx | `types` |
//! | AArch64CC, OperandSize, FloatSize | `cc` |
//!
//! # tMIR GEP stride contract
//!
//! `tmir::Inst::GEP { pointee_ty, base, indices }` computes
//! `base + sum_i (indices[i] * stride_i)` where the stride for the leading
//! index is `sizeof(pointee_ty)` and strides for trailing indices follow the
//! pointee type's layout. Indices are signed `I64`; negative indices are
//! well-defined and produce pointer arithmetic wrap-around following C's
//! `intptr_t` semantics. A zero-length `indices` list is the identity
//! operation.
//!
//! This contract is load-bearing for external consumers (`z4`, `tla2`) that
//! emit tMIR and consume symbol pointers via `mem::transmute`. Changing the
//! stride convention is a **P0 ABI break**.
//!
//! The LLVM2 consumer of this contract is
//! `crates/llvm2-lower/src/adapter.rs` (the `Inst::GEP` match arm), which
//! computes `elem_size = Type::bytes(translate_ty(pointee_ty))` and emits
//! `dst = base + index * elem_size` (skipping the multiply when
//! `elem_size == 1`). Multi-index GEP is currently unsupported and returns
//! `AdapterError::UnsupportedInstruction`.
//!
//! Unit test: `test_gep_stride_contract_i64` in
//! `crates/llvm2-lower/tests/tmir_integration.rs` pins `GEP { pointee_ty:
//! I64, base, indices: [idx] }` to `base + idx * 8`.
//!
//! Cross-references: issue #475 (this contract), issue #431 (sibling
//! calling-convention ABI doc).

pub mod aarch64_regs;
pub mod cc;
pub mod cost_model;
pub mod function;
pub mod inst;
pub mod operand;
pub mod provenance;
pub mod regs;
pub mod riscv_ops;
pub mod riscv_regs;
pub mod target_info;
pub mod tls;
pub mod trace;
pub mod type_hierarchy;
pub mod types;
pub mod x86_64_ops;
pub mod x86_64_regs;

// Re-export the most commonly used types at crate root.
pub use cc::{AArch64CC, FloatSize, OperandSize};
pub use function::{
    EhCallSiteEntry, ExceptionHandlingMetadata, FunctionDebugMeta, LandingPadEntry, MachBlock,
    MachFunction, Signature, StackSlot, Type,
};
pub use inst::{AArch64Opcode, InstFlags, MachInst, ProofAnnotation, SourceLoc};
pub use operand::MachOperand;
pub use provenance::{
    PassId, ProvenanceEntry, ProvenanceMap, ProvenanceStats, ProvenanceStatus, TmirInstId,
    TransformKind, TransformRecord,
};
pub use regs::{CondCode, PReg, RegClass, SpecialReg, VReg};
pub use riscv_ops::RiscVOpcode;
pub use riscv_regs::{RiscVPReg, RiscVRegClass};
pub use target_info::{AArch64Target, OpcodeCategory, TargetInfo, X86_64Target};
pub use tls::TlsModel;
pub use trace::{CompilationEvent, CompilationTrace, EventKind, Justification, RuleId, TraceLevel};
pub use types::{BlockId, FrameIdx, InstId, StackSlotId, VRegId};
pub use x86_64_ops::{X86CondCode, X86Opcode};
pub use x86_64_regs::{X86PReg, X86RegClass};

// llvm2-ir - Shared machine IR model
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Shared machine IR types for LLVM2.
//!
//! This crate defines the machine-level intermediate representation used by
//! all LLVM2 backend crates. It owns: MachInst, MachFunction, MachBlock,
//! registers (VReg/PReg), operands, stack slots, typed indices, and
//! AArch64-specific types (condition codes, opcode enum).
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

pub mod aarch64_regs;
pub mod cc;
pub mod cost_model;
pub mod function;
pub mod inst;
pub mod operand;
pub mod regs;
pub mod trace;
pub mod types;
pub mod x86_64_ops;
pub mod x86_64_regs;

// Re-export the most commonly used types at crate root.
pub use cc::{AArch64CC, FloatSize, OperandSize};
pub use function::{MachBlock, MachFunction, Signature, StackSlot, Type};
pub use inst::{AArch64Opcode, InstFlags, MachInst, ProofAnnotation};
pub use operand::MachOperand;
pub use regs::{CondCode, PReg, RegClass, SpecialReg, VReg};
pub use types::{BlockId, FrameIdx, InstId, StackSlotId, VRegId};
pub use trace::{
    CompilationEvent, CompilationTrace, EventKind, Justification, PassId, RuleId, TmirInstId,
    TraceLevel,
};
pub use x86_64_ops::{X86CondCode, X86Opcode};
pub use x86_64_regs::{X86PReg, X86RegClass};

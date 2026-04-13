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

pub mod aarch64_regs;
pub mod cc;
pub mod function;
pub mod inst;
pub mod operand;
pub mod regs;
pub mod types;

// Re-export the most commonly used types at crate root.
pub use cc::{AArch64CC, FloatSize, OperandSize};
pub use function::{MachBlock, MachFunction, Signature, StackSlot, Type};
pub use inst::{AArch64Opcode, InstFlags, MachInst, ProofAnnotation};
pub use operand::MachOperand;
pub use regs::{PReg, RegClass, SpecialReg, VReg};
pub use types::{BlockId, FrameIdx, InstId, StackSlotId, VRegId};

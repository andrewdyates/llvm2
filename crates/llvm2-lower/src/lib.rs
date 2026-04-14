// llvm2-lower - tMIR to LIR lowering
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! tMIR to Low-level IR (LIR) lowering for LLVM2.
//!
//! This crate handles the first stage of compilation: lowering tMIR
//! (the universal IR from tRust/tSwift/tC) to a low-level IR suitable
//! for optimization and machine code generation.

pub mod types;
pub mod instructions;
pub mod function;
pub mod abi;
pub mod isel;
pub mod adapter;

pub use types::Type;
pub use function::Function;
pub use abi::{AppleAArch64ABI, ArgLocation, PReg, gpr};
pub use isel::{InstructionSelector, ISelError, ISelFunction, ISelInst, ISelBlock, ISelOperand};
pub use adapter::{
    translate_module, translate_function, translate_type, extract_proofs,
    AdapterError, Proof, ProofContext,
};

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
pub mod target_analysis;
pub mod compute_graph;
pub mod dispatch;

pub use types::Type;
pub use function::Function;
pub use abi::{AppleAArch64ABI, ArgLocation, PReg, gpr};
pub use isel::{InstructionSelector, ISelError, ISelFunction, ISelInst, ISelBlock, ISelOperand, convert_isel_operand_to_ir};
pub use adapter::{
    translate_module, translate_function, translate_type, extract_proofs,
    AdapterError, Proof, ProofContext,
};
pub use target_analysis::{ComputeTarget, ProofAnalyzer, SubgraphProof, TargetLegality};
pub use compute_graph::TargetRecommendation;
pub use dispatch::{
    DispatchPlan, DispatchOp, DispatchError, ProfitabilityMismatch,
    generate_dispatch_plan, generate_profitability_aware_dispatch_plan,
    validate_dispatch_plan, validate_profitability_compliance,
};

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
pub mod va_list;
pub mod isel;
pub mod adapter;
pub mod target_analysis;
pub mod compute_graph;
pub mod dispatch;
pub mod x86_64_isel;

pub use types::Type;
pub use function::{Function, StackSlotInfo};
pub use abi::{
    AppleAArch64ABI, ArgLocation, PReg, gpr,
    UnwindInfo, SavedRegister, CompactUnwindEntry, DwarfCfiOp,
    generate_compact_unwind, generate_dwarf_cfi,
};
pub use va_list::{
    VaListIntrinsic, VaArgLowering, VaArgAccess,
    lower_va_arg, va_start_offset,
};
pub use isel::{InstructionSelector, ISelError, ISelFunction, ISelInst, ISelBlock, ISelOperand, convert_isel_operand_to_ir};
pub use x86_64_isel::{
    X86InstructionSelector, X86ISelError, X86ISelFunction, X86ISelInst, X86ISelBlock,
    X86ISelOperand, x86cc_from_intcc, x86cc_from_floatcc,
};
pub use adapter::{
    translate_module, translate_function, translate_type, extract_proofs,
    AdapterError, Proof, ProofContext,
};
pub use target_analysis::{ComputeTarget, ProofAnalyzer, SubgraphProof, TargetLegality};
pub use compute_graph::TargetRecommendation;
pub use dispatch::{
    DispatchPlan, DispatchOp, DispatchError, VerificationReport, ProfitabilityMismatch,
    generate_dispatch_plan, generate_profitability_aware_dispatch_plan,
    validate_dispatch_plan, validate_profitability_compliance, verify_dispatch_plan_properties,
};

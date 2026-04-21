// llvm2-lower/function.rs - Function representation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Function and basic block representation for LLVM2 LIR.

use crate::instructions::{Block, Instruction, Value};
use crate::types::Type;
use llvm2_ir::SourceLoc;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A basic block containing a sequence of instructions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BasicBlock {
    pub params: Vec<(Value, Type)>,
    pub instructions: Vec<Instruction>,
    /// Source locations for each instruction (parallel to `instructions`).
    ///
    /// Populated from tMIR `SourceSpan` during adapter translation.
    /// Carried through ISel to produce DWARF line number program entries.
    /// When shorter than `instructions`, missing entries are treated as None.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_locs: Vec<Option<SourceLoc>>,
}

/// Stack slot metadata for the LIR.
///
/// Tracks size and alignment for each stack allocation emitted by the adapter.
/// Propagated through ISel to the canonical `MachFunction::stack_slots`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackSlotInfo {
    /// Size in bytes.
    pub size: u32,
    /// Alignment in bytes (must be power of 2).
    pub align: u32,
}

/// A function in the LIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub signature: Signature,
    pub blocks: HashMap<Block, BasicBlock>,
    pub entry_block: Block,
    /// Stack slots allocated during adapter translation.
    ///
    /// Each entry corresponds to a `StackAddr { slot: N }` instruction,
    /// where N is the index into this Vec. Populated by the adapter to
    /// ensure each Alloc/Struct gets a unique slot index.
    pub stack_slots: Vec<StackSlotInfo>,
    /// Type hints for LIR `Value`s whose producing opcode does not carry
    /// enough type information for ISel to infer on its own.
    ///
    /// In particular, `Opcode::Call` and `Opcode::CallIndirect` result
    /// values only get a type if the adapter records it here — otherwise
    /// `InstructionSelector::value_type()` falls back to `Type::I64`,
    /// which silently miscompiles non-I64 callee returns (#381).
    ///
    /// Pipelines seed `InstructionSelector::value_types` from this map
    /// before running `select_block()`.
    pub value_types: HashMap<Value, Type>,
    /// Set of direct-call callee names known to be pure (i.e. the tMIR
    /// source function carried `ProofAnnotation::Pure`).
    ///
    /// Populated by the adapter when the tMIR module contains a function
    /// with `ProofAnnotation::Pure` that is reachable via `Inst::Call`.
    /// Pipelines seed `InstructionSelector::pure_callees` from this set so
    /// that ISel can stamp `ProofAnnotation::Pure` onto the emitted `Bl`
    /// MachInst, which SROA consumes for partial-escape reasoning (#456).
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub pure_callees: HashSet<String>,
}

/// LIR function signature (input-level, uses `llvm2_lower::Type`).
///
/// Separate from `llvm2_ir::function::Signature` which uses
/// `llvm2_ir::function::Type` (includes `Ptr`, no serde). This signature
/// is used by the instruction selector and ABI classifier before
/// lowering to the canonical MachIR representation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Signature {
    pub params: Vec<Type>,
    pub returns: Vec<Type>,
}

impl Function {
    /// Create a new function with the given name and signature.
    pub fn new(name: impl Into<String>, signature: Signature) -> Self {
        Self {
            name: name.into(),
            signature,
            blocks: HashMap::new(),
            entry_block: Block(0),
            stack_slots: Vec::new(),
            value_types: HashMap::new(),
            pure_callees: HashSet::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// From<llvm2_lower::Signature> for llvm2_ir::Signature
// ---------------------------------------------------------------------------
//
// Centralizes the signature conversion that was previously done inline in
// pipeline.rs. Uses the From<Type> impl in types.rs for individual type
// conversion.

impl From<&Signature> for llvm2_ir::function::Signature {
    fn from(sig: &Signature) -> Self {
        let params: Vec<llvm2_ir::function::Type> =
            sig.params.iter().map(|t| t.into()).collect();
        let returns: Vec<llvm2_ir::function::Type> =
            sig.returns.iter().map(|t| t.into()).collect();
        llvm2_ir::function::Signature::new(params, returns)
    }
}

impl From<Signature> for llvm2_ir::function::Signature {
    fn from(sig: Signature) -> Self {
        (&sig).into()
    }
}

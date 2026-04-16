// llvm2-lower/function.rs - Function representation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Function and basic block representation for LLVM2 LIR.

use crate::instructions::{Block, Instruction, Value};
use crate::types::Type;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A basic block containing a sequence of instructions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BasicBlock {
    pub params: Vec<(Value, Type)>,
    pub instructions: Vec<Instruction>,
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

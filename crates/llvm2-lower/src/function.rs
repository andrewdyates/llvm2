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

/// A function in the LIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub signature: Signature,
    pub blocks: HashMap<Block, BasicBlock>,
    pub entry_block: Block,
}

/// Function signature.
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
        }
    }
}

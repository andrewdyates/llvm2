// tmir-func stub — minimal tMIR function representation for LLVM2 development
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// This is a development stub. The real tMIR crate (ayates_dbx/tMIR) defines
// the full function/module IR. This stub provides only what LLVM2 needs.

#![allow(dead_code)]

pub mod builder;
pub mod reader;

use serde::{Deserialize, Serialize};
use tmir_instrs::InstrNode;
use tmir_types::{BlockId, FuncId, FuncTy, StructDef, TmirProof, Ty, ValueId};

/// A basic block in a tMIR function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    /// Block parameters (SSA phi-like arguments passed by predecessors).
    pub params: Vec<(ValueId, Ty)>,
    /// Instructions in this block, in order.
    pub body: Vec<InstrNode>,
}

/// A tMIR function definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub id: FuncId,
    pub name: String,
    pub ty: FuncTy,
    /// Entry block (always the first block).
    pub entry: BlockId,
    /// Basic blocks in layout order.
    pub blocks: Vec<Block>,
    /// Function-level proof annotations (e.g., Pure — no side effects).
    /// These apply to the entire function, not individual instructions.
    #[serde(default)]
    pub proofs: Vec<TmirProof>,
}

impl Function {
    /// Get a block by ID.
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Iterate blocks in layout order.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// A tMIR module: collection of functions, struct definitions, and globals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
    pub structs: Vec<StructDef>,
}

impl Module {
    /// Create an empty module.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            structs: Vec::new(),
        }
    }

    /// Look up a function by ID.
    pub fn function(&self, id: FuncId) -> Option<&Function> {
        self.functions.iter().find(|f| f.id == id)
    }

    /// Look up a function by name.
    pub fn function_by_name(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }
}

// tcrane-frontend/builder.rs - Function builder API
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Builder API for constructing Cranelift IR programmatically.

use tcrane_ir::function::{BasicBlock, Function, Signature};
use tcrane_ir::instructions::{Block, Value};
use tcrane_ir::types::Type;

/// Builder for constructing functions.
pub struct FunctionBuilder {
    func: Function,
    current_block: Option<Block>,
    next_value: u32,
    next_block: u32,
}

impl FunctionBuilder {
    /// Create a new function builder.
    pub fn new(name: impl Into<String>, signature: Signature) -> Self {
        let mut func = Function::new(name, signature);
        let entry = Block(0);
        func.blocks.insert(entry, BasicBlock::default());
        func.entry_block = entry;

        Self {
            func,
            current_block: Some(entry),
            next_value: 0,
            next_block: 1,
        }
    }

    /// Create a new basic block.
    pub fn create_block(&mut self) -> Block {
        let block = Block(self.next_block);
        self.next_block += 1;
        self.func.blocks.insert(block, BasicBlock::default());
        block
    }

    /// Switch to a different block for insertion.
    pub fn switch_to_block(&mut self, block: Block) {
        self.current_block = Some(block);
    }

    /// Allocate a new value.
    pub fn new_value(&mut self) -> Value {
        let v = Value(self.next_value);
        self.next_value += 1;
        v
    }

    /// Add a block parameter.
    pub fn add_block_param(&mut self, block: Block, ty: Type) -> Value {
        let v = self.new_value();
        if let Some(bb) = self.func.blocks.get_mut(&block) {
            bb.params.push((v, ty));
        }
        v
    }

    /// Finalize and return the built function.
    pub fn build(self) -> Function {
        self.func
    }
}

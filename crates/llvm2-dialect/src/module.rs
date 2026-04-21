// llvm2-dialect - DialectModule / DialectFunction / DialectBlock
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Top-level module containers: [`DialectModule`], [`DialectFunction`], and
//! [`DialectBlock`]. Storage is flat `Vec<DialectOp>` + typed indices, matching
//! the arena style used by `MachFunction` in `llvm2-ir`.

use llvm2_ir::Type;

use crate::id::{BlockId, DialectOpId, OpId, ValueId};
use crate::op::{Attributes, DialectOp, SourceRange};
use crate::registry::DialectRegistry;

/// A basic block referencing ops by arena index.
///
/// Blocks may carry SSA parameters — the dialect-level analogue of MLIR basic
/// block arguments / classical SSA phi nodes. The entry block of a function
/// has an empty `params` slice because the function-level parameters serve
/// that role (see [`DialectFunction::new`]); subsequent blocks may declare
/// their own parameter list via [`DialectFunction::new_block_with_params`].
#[derive(Debug, Clone)]
pub struct DialectBlock {
    pub id: BlockId,
    pub ops: Vec<OpId>,
    /// SSA parameters attached to this block. Each entry is
    /// `(value_id, type)` and the value IDs are unique inside the owning
    /// [`DialectFunction`].
    pub params: Vec<(ValueId, Type)>,
}

impl DialectBlock {
    pub fn new(id: BlockId) -> Self {
        Self { id, ops: Vec::new(), params: Vec::new() }
    }
}

/// A single function in a [`DialectModule`].
#[derive(Debug, Clone)]
pub struct DialectFunction {
    pub name: String,
    pub params: Vec<(ValueId, Type)>,
    pub results: Vec<Type>,
    pub blocks: Vec<DialectBlock>,
    /// Flat op arena. `OpId(i)` indexes `ops[i]`.
    pub ops: Vec<DialectOp>,
    next_value: u32,
    next_block: u32,
}

impl DialectFunction {
    /// Construct an empty function and a single entry block.
    ///
    /// Parameter SSA values are allocated in order, so callers receive the
    /// `ValueId`s they need to reference in the body.
    pub fn new(name: impl Into<String>, params: Vec<Type>, results: Vec<Type>) -> Self {
        let mut f = Self {
            name: name.into(),
            params: Vec::new(),
            results,
            blocks: Vec::new(),
            ops: Vec::new(),
            next_value: 0,
            next_block: 0,
        };
        // Entry block id 0.
        f.new_block();
        // Allocate parameter values (0..N) AFTER the entry block so their ids
        // remain low and memorable.
        for ty in params {
            let v = f.alloc_value();
            f.params.push((v, ty));
        }
        f
    }

    /// Allocate a fresh SSA `ValueId`.
    pub fn alloc_value(&mut self) -> ValueId {
        let v = ValueId(self.next_value);
        self.next_value += 1;
        v
    }

    /// Append a fresh empty block and return its id.
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.push(DialectBlock::new(id));
        id
    }

    /// Append a fresh block that carries `param_types` SSA parameters.
    ///
    /// Returns the new block's id along with the freshly-allocated parameter
    /// `ValueId`s in positional order. The block is empty until ops are
    /// appended via [`DialectFunction::append_op`].
    pub fn new_block_with_params(
        &mut self,
        param_types: Vec<Type>,
    ) -> (BlockId, Vec<ValueId>) {
        let id = self.new_block();
        let mut param_values = Vec::with_capacity(param_types.len());
        for ty in param_types {
            let v = self.alloc_value();
            param_values.push(v);
            // Safe: we just pushed the block via `new_block` above.
            self.blocks
                .last_mut()
                .expect("new_block pushed a block")
                .params
                .push((v, ty));
        }
        (id, param_values)
    }

    /// Append an op to the given block and return its [`OpId`].
    pub fn append_op(
        &mut self,
        block: BlockId,
        op: DialectOpId,
        results: Vec<(ValueId, Type)>,
        operands: Vec<ValueId>,
        attrs: Attributes,
        source: Option<SourceRange>,
    ) -> OpId {
        let id = OpId(self.ops.len() as u32);
        self.ops.push(DialectOp {
            id,
            op,
            results,
            operands,
            attrs,
            source,
        });
        self.blocks[block.0 as usize].ops.push(id);
        id
    }

    /// Iterate over ops in program order (block by block).
    pub fn iter_ops(&self) -> impl Iterator<Item = &DialectOp> {
        self.blocks
            .iter()
            .flat_map(move |b| b.ops.iter().map(move |id| &self.ops[id.0 as usize]))
    }

    /// Total op count across all blocks.
    pub fn op_count(&self) -> usize {
        self.blocks.iter().map(|b| b.ops.len()).sum()
    }

    pub fn entry_block(&self) -> Option<BlockId> {
        self.blocks.first().map(|b| b.id)
    }
}

/// A container of [`DialectFunction`]s plus the [`DialectRegistry`] describing
/// the dialects they reference.
pub struct DialectModule {
    pub name: String,
    pub functions: Vec<DialectFunction>,
    pub registry: DialectRegistry,
}

impl DialectModule {
    pub fn new(name: impl Into<String>, registry: DialectRegistry) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            registry,
        }
    }

    pub fn push_function(&mut self, f: DialectFunction) -> usize {
        self.functions.push(f);
        self.functions.len() - 1
    }

    /// Resolve an op to its static op-definition via the registry.
    pub fn resolve(&self, op: DialectOpId) -> Option<&crate::dialect::OpDef> {
        self.registry.op_def(op)
    }
}

impl std::fmt::Debug for DialectModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DialectModule")
            .field("name", &self.name)
            .field("functions", &self.functions.len())
            .field("registry", &self.registry)
            .finish()
    }
}

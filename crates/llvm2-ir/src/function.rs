// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine-level function and block types.
//!
//! Storage model: arena-based Vec indexed by typed wrappers.
//! No HashMap for blocks/values — Vec + index only (cache-friendly).

use crate::inst::MachInst;
use crate::types::{BlockId, InstId, StackSlotId};

/// A basic block of machine instructions.
#[derive(Debug, Clone)]
pub struct MachBlock {
    /// Instructions in this block (indices into MachFunction::insts).
    pub insts: Vec<InstId>,
    /// Predecessor blocks.
    pub preds: Vec<BlockId>,
    /// Successor blocks.
    pub succs: Vec<BlockId>,
}

impl MachBlock {
    /// Create a new empty block.
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
        }
    }

    /// Returns true if the block has no instructions.
    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    /// Returns the number of instructions in this block.
    pub fn len(&self) -> usize {
        self.insts.len()
    }
}

impl Default for MachBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Type information for function signatures and stack slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    /// Boolean (1-bit).
    B1,
    /// Pointer-sized integer.
    Ptr,
}

impl Type {
    /// Size of this type in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Self::I8 | Self::B1 => 1,
            Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 | Self::Ptr => 8,
            Self::I128 => 16,
        }
    }

    /// Natural alignment of this type in bytes.
    pub fn align(self) -> u32 {
        self.bytes().min(8)
    }

    /// Returns true if this is an integer type.
    pub fn is_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128)
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

/// Function signature: parameter and return types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<Type>,
    pub returns: Vec<Type>,
}

impl Signature {
    pub fn new(params: Vec<Type>, returns: Vec<Type>) -> Self {
        Self { params, returns }
    }
}

/// A stack slot allocated in the function's stack frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackSlot {
    /// Size in bytes.
    pub size: u32,
    /// Alignment in bytes (must be power of 2).
    pub align: u32,
}

impl StackSlot {
    pub fn new(size: u32, align: u32) -> Self {
        debug_assert!(align.is_power_of_two(), "alignment must be power of 2");
        Self { size, align }
    }
}

/// A complete machine-level function, ready for register allocation and encoding.
#[derive(Debug, Clone)]
pub struct MachFunction {
    /// Function name (mangled symbol name).
    pub name: String,
    /// Function signature.
    pub signature: Signature,
    /// All instructions, indexed by InstId.
    pub insts: Vec<MachInst>,
    /// All basic blocks, indexed by BlockId.
    pub blocks: Vec<MachBlock>,
    /// Block layout order (for code emission).
    pub block_order: Vec<BlockId>,
    /// Entry block.
    pub entry: BlockId,
    /// Next virtual register ID to allocate.
    pub next_vreg: u32,
    /// Stack slots allocated by this function.
    pub stack_slots: Vec<StackSlot>,
}

impl MachFunction {
    /// Create a new function with the given name and signature.
    pub fn new(name: String, signature: Signature) -> Self {
        // Create entry block (bb0).
        let entry_block = MachBlock::new();
        Self {
            name,
            signature,
            insts: Vec::new(),
            blocks: vec![entry_block],
            block_order: vec![BlockId(0)],
            entry: BlockId(0),
            next_vreg: 0,
            stack_slots: Vec::new(),
        }
    }

    /// Allocate a new virtual register ID.
    pub fn alloc_vreg(&mut self) -> u32 {
        let id = self.next_vreg;
        self.next_vreg += 1;
        id
    }

    /// Add an instruction to the arena and return its InstId.
    pub fn push_inst(&mut self, inst: MachInst) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(inst);
        id
    }

    /// Create a new basic block and return its BlockId.
    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(MachBlock::new());
        self.block_order.push(id);
        id
    }

    /// Append an instruction to a block.
    pub fn append_inst(&mut self, block: BlockId, inst_id: InstId) {
        self.blocks[block.0 as usize].insts.push(inst_id);
    }

    /// Add a stack slot and return its ID.
    pub fn alloc_stack_slot(&mut self, slot: StackSlot) -> StackSlotId {
        let id = StackSlotId(self.stack_slots.len() as u32);
        self.stack_slots.push(slot);
        id
    }

    /// Get an instruction by ID.
    pub fn inst(&self, id: InstId) -> &MachInst {
        &self.insts[id.0 as usize]
    }

    /// Get a mutable instruction by ID.
    pub fn inst_mut(&mut self, id: InstId) -> &mut MachInst {
        &mut self.insts[id.0 as usize]
    }

    /// Get a block by ID.
    pub fn block(&self, id: BlockId) -> &MachBlock {
        &self.blocks[id.0 as usize]
    }

    /// Get a mutable block by ID.
    pub fn block_mut(&mut self, id: BlockId) -> &mut MachBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Returns the total number of instructions.
    pub fn num_insts(&self) -> usize {
        self.insts.len()
    }

    /// Returns the total number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Add a CFG edge from `from` to `to`.
    pub fn add_edge(&mut self, from: BlockId, to: BlockId) {
        self.blocks[from.0 as usize].succs.push(to);
        self.blocks[to.0 as usize].preds.push(from);
    }
}

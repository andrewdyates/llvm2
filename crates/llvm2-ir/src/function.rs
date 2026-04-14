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
    /// Loop nesting depth (0 = not in a loop). Used by regalloc for spill
    /// weight computation and by optimization passes (LICM) for hoisting
    /// decisions. Populated by loop analysis; defaults to 0.
    pub loop_depth: u32,
}

impl MachBlock {
    /// Create a new empty block.
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
            loop_depth: 0,
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
///
/// Scalar types (I8..I128, F32, F64, B1, Ptr) are the original machine-level
/// types. Aggregate types (Struct, Array) were added to support real programs
/// that pass/return structs and allocate arrays on the stack.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// Aggregate structure type with C-like field layout.
    Struct(Vec<Type>),
    /// Fixed-size array type: element type and count.
    Array(Box<Type>, u32),
}

impl Type {
    /// Round `offset` up to the next multiple of `align`.
    fn align_to(offset: u32, align: u32) -> u32 {
        if align <= 1 {
            offset
        } else {
            let rem = offset % align;
            if rem == 0 { offset } else { offset + (align - rem) }
        }
    }

    /// Size of this type in bytes.
    ///
    /// For structs, uses C-like layout with alignment padding between fields
    /// and trailing padding to the struct's alignment.
    /// For arrays, returns element_size * count.
    pub fn bytes(&self) -> u32 {
        match self {
            Self::I8 | Self::B1 => 1,
            Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 | Self::Ptr => 8,
            Self::I128 => 16,
            Self::Struct(fields) => {
                let mut offset: u32 = 0;
                let mut max_align: u32 = 1;
                for field in fields {
                    let a = field.align();
                    max_align = max_align.max(a);
                    offset = Self::align_to(offset, a);
                    offset += field.bytes();
                }
                Self::align_to(offset, max_align)
            }
            Self::Array(elem, count) => elem.bytes() * count,
        }
    }

    /// Alias for `bytes()`.
    pub fn size_of(&self) -> u32 {
        self.bytes()
    }

    /// Natural alignment of this type in bytes.
    ///
    /// Scalars: min(size, 8). Struct: max alignment of fields. Array: element alignment.
    pub fn align(&self) -> u32 {
        match self {
            Self::Struct(fields) => fields.iter().map(|f| f.align()).max().unwrap_or(1),
            Self::Array(elem, _) => elem.align(),
            _ => self.bytes().min(8),
        }
    }

    /// Alias for `align()`.
    pub fn align_of(&self) -> u32 {
        self.align()
    }

    /// Byte offset of a struct field using C-like layout rules.
    ///
    /// Returns `None` if this is not a struct type or the index is out of range.
    pub fn offset_of(&self, field_index: usize) -> Option<u32> {
        let Self::Struct(fields) = self else {
            return None;
        };
        if field_index >= fields.len() {
            return None;
        }
        let mut offset: u32 = 0;
        for (idx, field) in fields.iter().enumerate() {
            offset = Self::align_to(offset, field.align());
            if idx == field_index {
                return Some(offset);
            }
            offset += field.bytes();
        }
        None
    }

    /// Returns true if this is an aggregate (struct or array) type.
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Self::Struct(_) | Self::Array(_, _))
    }

    /// Returns true if this is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128)
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Returns true if this is a scalar (non-aggregate) type.
    pub fn is_scalar(&self) -> bool {
        !self.is_aggregate()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inst::{AArch64Opcode, MachInst};
    use crate::operand::MachOperand;
    use crate::types::BlockId;

    // ---- MachBlock tests ----

    #[test]
    fn machblock_new_is_empty() {
        let block = MachBlock::new();
        assert!(block.is_empty());
        assert_eq!(block.len(), 0);
        assert!(block.preds.is_empty());
        assert!(block.succs.is_empty());
    }

    #[test]
    fn machblock_default_is_new() {
        let a = MachBlock::new();
        let b = MachBlock::default();
        assert_eq!(a.is_empty(), b.is_empty());
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn machblock_add_instructions() {
        let mut block = MachBlock::new();
        block.insts.push(InstId(0));
        assert!(!block.is_empty());
        assert_eq!(block.len(), 1);
        block.insts.push(InstId(1));
        assert_eq!(block.len(), 2);
    }

    #[test]
    fn machblock_predecessors_successors() {
        let mut block = MachBlock::new();
        block.preds.push(BlockId(0));
        block.preds.push(BlockId(1));
        block.succs.push(BlockId(3));
        assert_eq!(block.preds.len(), 2);
        assert_eq!(block.succs.len(), 1);
    }

    // ---- Type tests ----

    #[test]
    fn type_bytes() {
        assert_eq!(Type::I8.bytes(), 1);
        assert_eq!(Type::B1.bytes(), 1);
        assert_eq!(Type::I16.bytes(), 2);
        assert_eq!(Type::I32.bytes(), 4);
        assert_eq!(Type::F32.bytes(), 4);
        assert_eq!(Type::I64.bytes(), 8);
        assert_eq!(Type::F64.bytes(), 8);
        assert_eq!(Type::Ptr.bytes(), 8);
        assert_eq!(Type::I128.bytes(), 16);
    }

    #[test]
    fn type_align() {
        // Alignment is min(bytes, 8)
        assert_eq!(Type::I8.align(), 1);
        assert_eq!(Type::B1.align(), 1);
        assert_eq!(Type::I16.align(), 2);
        assert_eq!(Type::I32.align(), 4);
        assert_eq!(Type::F32.align(), 4);
        assert_eq!(Type::I64.align(), 8);
        assert_eq!(Type::F64.align(), 8);
        assert_eq!(Type::Ptr.align(), 8);
        assert_eq!(Type::I128.align(), 8); // capped at 8
    }

    #[test]
    fn type_is_int() {
        assert!(Type::I8.is_int());
        assert!(Type::I16.is_int());
        assert!(Type::I32.is_int());
        assert!(Type::I64.is_int());
        assert!(Type::I128.is_int());
        assert!(!Type::F32.is_int());
        assert!(!Type::F64.is_int());
        assert!(!Type::B1.is_int());
        assert!(!Type::Ptr.is_int());
    }

    #[test]
    fn type_is_float() {
        assert!(Type::F32.is_float());
        assert!(Type::F64.is_float());
        assert!(!Type::I8.is_float());
        assert!(!Type::I32.is_float());
        assert!(!Type::I64.is_float());
        assert!(!Type::B1.is_float());
        assert!(!Type::Ptr.is_float());
    }

    #[test]
    fn type_b1_is_neither_int_nor_float() {
        assert!(!Type::B1.is_int());
        assert!(!Type::B1.is_float());
    }

    #[test]
    fn type_ptr_is_neither_int_nor_float() {
        assert!(!Type::Ptr.is_int());
        assert!(!Type::Ptr.is_float());
    }

    #[test]
    fn type_equality() {
        assert_eq!(Type::I32, Type::I32);
        assert_ne!(Type::I32, Type::I64);
        assert_ne!(Type::I32, Type::F32);
    }

    #[test]
    fn type_clone_hash() {
        use std::collections::HashSet;
        let a = Type::I64;
        let b = a.clone(); // Clone
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);

        let mut set = HashSet::new();
        set.insert(Type::I32);
        set.insert(Type::I64);
        set.insert(Type::I32); // duplicate
        assert_eq!(set.len(), 2);
    }

    // ---- Signature tests ----

    #[test]
    fn signature_creation() {
        let sig = Signature::new(
            vec![Type::I64, Type::I32],
            vec![Type::I64],
        );
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.params[0], Type::I64);
        assert_eq!(sig.params[1], Type::I32);
        assert_eq!(sig.returns[0], Type::I64);
    }

    #[test]
    fn signature_empty() {
        let sig = Signature::new(vec![], vec![]);
        assert!(sig.params.is_empty());
        assert!(sig.returns.is_empty());
    }

    #[test]
    fn signature_equality() {
        let a = Signature::new(vec![Type::I64], vec![Type::I32]);
        let b = Signature::new(vec![Type::I64], vec![Type::I32]);
        let c = Signature::new(vec![Type::I32], vec![Type::I32]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ---- StackSlot tests ----

    #[test]
    fn stack_slot_creation() {
        let slot = StackSlot::new(8, 8);
        assert_eq!(slot.size, 8);
        assert_eq!(slot.align, 8);
    }

    #[test]
    fn stack_slot_various_sizes() {
        let slot1 = StackSlot::new(1, 1);
        assert_eq!(slot1.size, 1);
        assert_eq!(slot1.align, 1);

        let slot16 = StackSlot::new(16, 16);
        assert_eq!(slot16.size, 16);
        assert_eq!(slot16.align, 16);

        let slot_mixed = StackSlot::new(12, 4);
        assert_eq!(slot_mixed.size, 12);
        assert_eq!(slot_mixed.align, 4);
    }

    #[test]
    fn stack_slot_equality() {
        let a = StackSlot::new(8, 8);
        let b = StackSlot::new(8, 8);
        let c = StackSlot::new(8, 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    #[should_panic(expected = "alignment must be power of 2")]
    fn stack_slot_non_power_of_two_panics() {
        let _ = StackSlot::new(8, 3);
    }

    #[test]
    #[should_panic(expected = "alignment must be power of 2")]
    fn stack_slot_zero_alignment_panics() {
        let _ = StackSlot::new(8, 0);
    }

    // ---- MachFunction tests ----

    #[test]
    fn machfunction_new_has_entry_block() {
        let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
        let func = MachFunction::new("test_fn".to_string(), sig);
        assert_eq!(func.name, "test_fn");
        assert_eq!(func.num_blocks(), 1); // entry block
        assert_eq!(func.entry, BlockId(0));
        assert_eq!(func.num_insts(), 0);
        assert_eq!(func.next_vreg, 0);
        assert!(func.stack_slots.is_empty());
        assert_eq!(func.block_order.len(), 1);
        assert_eq!(func.block_order[0], BlockId(0));
    }

    #[test]
    fn machfunction_alloc_vreg() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        assert_eq!(func.alloc_vreg(), 0);
        assert_eq!(func.alloc_vreg(), 1);
        assert_eq!(func.alloc_vreg(), 2);
        assert_eq!(func.next_vreg, 3);
    }

    #[test]
    fn machfunction_push_inst() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        let inst = MachInst::new(AArch64Opcode::Nop, vec![]);
        let id = func.push_inst(inst);
        assert_eq!(id, InstId(0));
        assert_eq!(func.num_insts(), 1);

        let inst2 = MachInst::new(AArch64Opcode::Ret, vec![]);
        let id2 = func.push_inst(inst2);
        assert_eq!(id2, InstId(1));
        assert_eq!(func.num_insts(), 2);
    }

    #[test]
    fn machfunction_create_block() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        assert_eq!(func.num_blocks(), 1); // entry

        let bb1 = func.create_block();
        assert_eq!(bb1, BlockId(1));
        assert_eq!(func.num_blocks(), 2);

        let bb2 = func.create_block();
        assert_eq!(bb2, BlockId(2));
        assert_eq!(func.num_blocks(), 3);

        // Block order should include all blocks
        assert_eq!(func.block_order.len(), 3);
    }

    #[test]
    fn machfunction_append_inst() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        let inst_id = func.push_inst(MachInst::new(AArch64Opcode::Nop, vec![]));
        func.append_inst(BlockId(0), inst_id);

        let block = func.block(BlockId(0));
        assert_eq!(block.len(), 1);
        assert_eq!(block.insts[0], inst_id);
    }

    #[test]
    fn machfunction_alloc_stack_slot() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        let ss0 = func.alloc_stack_slot(StackSlot::new(8, 8));
        assert_eq!(ss0, crate::types::StackSlotId(0));
        let ss1 = func.alloc_stack_slot(StackSlot::new(16, 16));
        assert_eq!(ss1, crate::types::StackSlotId(1));
        assert_eq!(func.stack_slots.len(), 2);
    }

    #[test]
    fn machfunction_inst_access() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        let id = func.push_inst(MachInst::new(AArch64Opcode::AddRR, vec![MachOperand::Imm(42)]));

        let inst = func.inst(id);
        assert_eq!(inst.opcode, AArch64Opcode::AddRR);
        assert_eq!(inst.operands.len(), 1);

        // Mutable access
        let inst_mut = func.inst_mut(id);
        inst_mut.operands.push(MachOperand::Imm(99));
        assert_eq!(func.inst(id).operands.len(), 2);
    }

    #[test]
    fn machfunction_block_access() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);

        let block = func.block(BlockId(0));
        assert!(block.is_empty());

        // Mutable access
        let block_mut = func.block_mut(BlockId(0));
        block_mut.preds.push(BlockId(1));
        assert_eq!(func.block(BlockId(0)).preds.len(), 1);
    }

    #[test]
    fn machfunction_add_edge() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        let bb1 = func.create_block();

        func.add_edge(BlockId(0), bb1);

        assert_eq!(func.block(BlockId(0)).succs.len(), 1);
        assert_eq!(func.block(BlockId(0)).succs[0], bb1);
        assert_eq!(func.block(bb1).preds.len(), 1);
        assert_eq!(func.block(bb1).preds[0], BlockId(0));
    }

    #[test]
    fn machfunction_add_multiple_edges() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("f".to_string(), sig);
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        // bb0 -> bb1, bb0 -> bb2 (conditional branch)
        func.add_edge(BlockId(0), bb1);
        func.add_edge(BlockId(0), bb2);

        assert_eq!(func.block(BlockId(0)).succs.len(), 2);
        assert_eq!(func.block(bb1).preds.len(), 1);
        assert_eq!(func.block(bb2).preds.len(), 1);
    }

    #[test]
    fn machfunction_diamond_cfg() {
        // bb0 -> bb1, bb0 -> bb2, bb1 -> bb3, bb2 -> bb3
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("diamond".to_string(), sig);
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        func.add_edge(BlockId(0), bb1);
        func.add_edge(BlockId(0), bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        assert_eq!(func.block(BlockId(0)).succs.len(), 2);
        assert_eq!(func.block(bb3).preds.len(), 2);
        assert_eq!(func.num_blocks(), 4);
    }

    #[test]
    fn machfunction_full_workflow() {
        // Build a simple function: entry block with an add and a return
        let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("add_two".to_string(), sig);

        // Allocate virtual registers
        let v0 = func.alloc_vreg();
        let v1 = func.alloc_vreg();
        let v2 = func.alloc_vreg();
        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);

        // Create instructions
        let add_id = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::VReg(crate::regs::VReg::new(v2, crate::regs::RegClass::Gpr64)),
                MachOperand::VReg(crate::regs::VReg::new(v0, crate::regs::RegClass::Gpr64)),
                MachOperand::VReg(crate::regs::VReg::new(v1, crate::regs::RegClass::Gpr64)),
            ],
        ));
        let ret_id = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));

        // Append to entry block
        func.append_inst(BlockId(0), add_id);
        func.append_inst(BlockId(0), ret_id);

        // Allocate a stack slot
        let ss = func.alloc_stack_slot(StackSlot::new(8, 8));

        // Verify
        assert_eq!(func.num_insts(), 2);
        assert_eq!(func.num_blocks(), 1);
        assert_eq!(func.block(BlockId(0)).len(), 2);
        assert_eq!(func.next_vreg, 3);
        assert_eq!(func.stack_slots.len(), 1);
        assert!(!func.inst(add_id).is_return());
        assert!(func.inst(ret_id).is_return());
        assert_eq!(ss, crate::types::StackSlotId(0));
    }
}

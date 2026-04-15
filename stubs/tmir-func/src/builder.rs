// tmir-func/builder.rs - Builder API for constructing tMIR programs
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Provides a fluent builder API for constructing tMIR functions and modules,
// making it easier to write integration tests and programmatically generate
// tMIR programs for LLVM2's adapter and ISel pipeline.

use tmir_instrs::{BinOp, CastOp, CmpOp, Instr, InstrNode, SwitchCase, UnOp};
use tmir_types::{BlockId, FuncId, FuncTy, StructDef, TmirProof, Ty, ValueId};

use crate::{Block, Function, Module};

/// Builder for constructing tMIR modules.
pub struct ModuleBuilder {
    name: String,
    functions: Vec<Function>,
    structs: Vec<StructDef>,
    next_func_id: u32,
}

impl ModuleBuilder {
    /// Create a new module builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            structs: Vec::new(),
            next_func_id: 0,
        }
    }

    /// Add a struct definition to the module.
    pub fn add_struct(&mut self, def: StructDef) -> &mut Self {
        self.structs.push(def);
        self
    }

    /// Start building a new function in this module.
    pub fn function(
        &mut self,
        name: impl Into<String>,
        params: Vec<Ty>,
        returns: Vec<Ty>,
    ) -> FunctionBuilder {
        let id = FuncId(self.next_func_id);
        self.next_func_id += 1;
        FunctionBuilder::new(id, name.into(), params, returns)
    }

    /// Add a completed function to the module.
    pub fn add_function(&mut self, func: Function) -> &mut Self {
        self.functions.push(func);
        self
    }

    /// Build the module.
    pub fn build(self) -> Module {
        Module {
            name: self.name,
            functions: self.functions,
            structs: self.structs,
        }
    }
}

/// Builder for constructing tMIR functions.
pub struct FunctionBuilder {
    id: FuncId,
    name: String,
    ty: FuncTy,
    blocks: Vec<Block>,
    proofs: Vec<TmirProof>,
    next_value_id: u32,
    next_block_id: u32,
    entry: BlockId,
}

impl FunctionBuilder {
    /// Create a new function builder.
    pub fn new(id: FuncId, name: String, params: Vec<Ty>, returns: Vec<Ty>) -> Self {
        Self {
            id,
            name,
            ty: FuncTy { params, returns },
            blocks: Vec::new(),
            proofs: Vec::new(),
            next_value_id: 0,
            next_block_id: 0,
            entry: BlockId(0),
        }
    }

    /// Set the function ID.
    pub fn with_id(mut self, id: FuncId) -> Self {
        self.id = id;
        self
    }

    /// Add a function-level proof annotation.
    pub fn with_proof(mut self, proof: TmirProof) -> Self {
        self.proofs.push(proof);
        self
    }

    /// Allocate a fresh ValueId.
    pub fn fresh_value(&mut self) -> ValueId {
        let v = ValueId(self.next_value_id);
        self.next_value_id += 1;
        v
    }

    /// Allocate N fresh ValueIds.
    pub fn fresh_values(&mut self, n: usize) -> Vec<ValueId> {
        (0..n).map(|_| self.fresh_value()).collect()
    }

    /// Allocate a fresh BlockId.
    pub fn fresh_block(&mut self) -> BlockId {
        let b = BlockId(self.next_block_id);
        self.next_block_id += 1;
        b
    }

    /// Create a new block with parameters and body, then add it.
    pub fn add_block(
        &mut self,
        id: BlockId,
        params: Vec<(ValueId, Ty)>,
        body: Vec<InstrNode>,
    ) {
        self.blocks.push(Block { id, params, body });
    }

    /// Start building the entry block with the function's parameter types.
    /// Returns (BlockId, Vec<ValueId>) for the entry block and its parameter values.
    pub fn entry_block(&mut self) -> (BlockId, Vec<ValueId>) {
        let block_id = self.fresh_block();
        self.entry = block_id;
        let param_count = self.ty.params.len();
        let params: Vec<ValueId> = (0..param_count).map(|_| self.fresh_value()).collect();
        (block_id, params)
    }

    /// Build the function.
    pub fn build(self) -> Function {
        Function {
            id: self.id,
            name: self.name,
            ty: self.ty,
            entry: self.entry,
            blocks: self.blocks,
            proofs: self.proofs,
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction construction helpers
// ---------------------------------------------------------------------------

/// Create an integer constant instruction.
pub fn iconst(ty: Ty, value: i64, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::Const { ty, value }, vec![result])
}

/// Create a float constant instruction.
pub fn fconst(ty: Ty, value: f64, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::FConst { ty, value }, vec![result])
}

/// Create a binary operation instruction.
pub fn binop(op: BinOp, ty: Ty, lhs: ValueId, rhs: ValueId, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::BinOp { op, ty, lhs, rhs }, vec![result])
}

/// Create a unary operation instruction.
pub fn unop(op: UnOp, ty: Ty, operand: ValueId, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::UnOp { op, ty, operand }, vec![result])
}

/// Create a comparison instruction.
pub fn cmp(op: CmpOp, ty: Ty, lhs: ValueId, rhs: ValueId, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::Cmp { op, ty, lhs, rhs }, vec![result])
}

/// Create a type cast instruction.
pub fn cast(
    op: CastOp,
    src_ty: Ty,
    dst_ty: Ty,
    operand: ValueId,
    result: ValueId,
) -> InstrNode {
    InstrNode::new(
        Instr::Cast {
            op,
            src_ty,
            dst_ty,
            operand,
        },
        vec![result],
    )
}

/// Create a load instruction.
pub fn load(ty: Ty, ptr: ValueId, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::Load { ty, ptr }, vec![result])
}

/// Create a store instruction.
pub fn store(ty: Ty, ptr: ValueId, value: ValueId) -> InstrNode {
    InstrNode::new(Instr::Store { ty, ptr, value }, vec![])
}

/// Create a stack allocation instruction.
pub fn alloc(ty: Ty, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::Alloc { ty, count: None }, vec![result])
}

/// Create an unconditional branch.
pub fn br(target: BlockId, args: Vec<ValueId>) -> InstrNode {
    InstrNode::new(Instr::Br { target, args }, vec![])
}

/// Create a conditional branch.
pub fn condbr(
    cond: ValueId,
    then_target: BlockId,
    then_args: Vec<ValueId>,
    else_target: BlockId,
    else_args: Vec<ValueId>,
) -> InstrNode {
    InstrNode::new(
        Instr::CondBr {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        },
        vec![],
    )
}

/// Create a return instruction.
pub fn ret(values: Vec<ValueId>) -> InstrNode {
    InstrNode::new(Instr::Return { values }, vec![])
}

/// Create a direct function call.
pub fn call(func: FuncId, args: Vec<ValueId>, ret_ty: Vec<Ty>, results: Vec<ValueId>) -> InstrNode {
    InstrNode::new(Instr::Call { func, args, ret_ty }, results)
}

/// Create an indirect function call.
pub fn call_indirect(
    callee: ValueId,
    args: Vec<ValueId>,
    ret_ty: Vec<Ty>,
    results: Vec<ValueId>,
) -> InstrNode {
    InstrNode::new(
        Instr::CallIndirect {
            callee,
            args,
            ret_ty,
        },
        results,
    )
}

/// Create a switch instruction.
pub fn switch(value: ValueId, cases: Vec<(i64, BlockId)>, default: BlockId) -> InstrNode {
    let cases = cases
        .into_iter()
        .map(|(v, target)| SwitchCase { value: v, target })
        .collect();
    InstrNode::new(Instr::Switch { value, cases, default }, vec![])
}

/// Create a select (conditional value) instruction.
pub fn select(
    ty: Ty,
    cond: ValueId,
    true_val: ValueId,
    false_val: ValueId,
    result: ValueId,
) -> InstrNode {
    InstrNode::new(
        Instr::Select {
            ty,
            cond,
            true_val,
            false_val,
        },
        vec![result],
    )
}

/// Create a get-element-pointer instruction.
pub fn gep(
    elem_ty: Ty,
    base: ValueId,
    index: ValueId,
    offset: i32,
    result: ValueId,
) -> InstrNode {
    InstrNode::new(
        Instr::GetElementPtr {
            elem_ty,
            base,
            index,
            offset,
        },
        vec![result],
    )
}

/// Create a struct field extraction instruction.
pub fn field(ty: Ty, value: ValueId, index: u32, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::Field { ty, value, index }, vec![result])
}

/// Create a struct construction instruction.
pub fn struct_val(ty: Ty, fields: Vec<ValueId>, result: ValueId) -> InstrNode {
    InstrNode::new(Instr::Struct { ty, fields }, vec![result])
}

/// Create an array/pointer index instruction.
pub fn index(ty: Ty, base: ValueId, index_val: ValueId, result: ValueId) -> InstrNode {
    InstrNode::new(
        Instr::Index {
            ty,
            base,
            index: index_val,
        },
        vec![result],
    )
}

// llvm2-lower/x86_64_isel.rs - x86-64 instruction selection
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: LLVM X86ISelLowering.cpp
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual
// Reference: System V AMD64 ABI (https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)

//! x86-64 instruction selection: tMIR SSA IR -> x86-64 MachIR with virtual registers.
//!
//! This module parallels the AArch64 instruction selector in `isel.rs` but
//! targets the x86-64 ISA with System V AMD64 ABI calling conventions.
//!
//! Phase 1 (this module): Walk tMIR blocks in order, match each instruction,
//! emit x86-64 instructions with VRegs. Covers arithmetic, comparisons,
//! branches, calls, returns, loads, stores, and constants.
//!
//! The x86-64 ISel types (`X86ISelFunction`, `X86ISelInst`, `X86ISelBlock`,
//! `X86ISelOperand`) are ISel-specific intermediates, parallel to but separate
//! from the AArch64 ISel types in `isel.rs`.

use std::collections::HashMap;

use crate::function::Signature;
use crate::instructions::{Block, FloatCC, Instruction, IntCC, Opcode, Value};
use crate::types::Type;
use thiserror::Error;

use llvm2_ir::regs::{RegClass, VReg};
use llvm2_ir::x86_64_ops::{X86CondCode, X86Opcode};
use llvm2_ir::x86_64_regs::{
    self, X86PReg, RSP,
    X86_ARG_GPRS, X86_ARG_XMMS, X86_RET_GPRS,
};

// ---------------------------------------------------------------------------
// Register model helpers
// ---------------------------------------------------------------------------

/// Derive the register class for a given LIR type.
fn reg_class_for_type(ty: &Type) -> RegClass {
    match ty {
        Type::B1 | Type::I8 | Type::I16 | Type::I32 => RegClass::Gpr32,
        Type::I64 | Type::I128 => RegClass::Gpr64,
        Type::F32 => RegClass::Fpr32,
        Type::F64 => RegClass::Fpr64,
        Type::V128 => RegClass::Fpr128,
        Type::Struct(_) | Type::Array(_, _) => RegClass::Gpr64,
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors during x86-64 instruction selection.
#[derive(Debug, Error)]
pub enum X86ISelError {
    #[error("value {0:?} not defined before use")]
    UndefinedValue(Value),
    #[error("unsupported type for Fconst: {0:?} (expected F32 or F64)")]
    UnsupportedFconstType(Type),
    #[error("unsupported ABI location: stack-passed returns are not supported")]
    UnsupportedReturnLocation,
    #[error("malformed instruction: expected at least {expected} args, got {actual} (opcode context: {context})")]
    InsufficientArgs {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    #[error("malformed instruction: expected at least 1 result (opcode context: {0})")]
    MissingResult(&'static str),
    #[error("too many arguments for System V ABI: {0} (max 6 integer + 8 float)")]
    TooManyArgs(usize),
}

// ---------------------------------------------------------------------------
// Condition code mapping
// ---------------------------------------------------------------------------

/// Map tMIR integer comparison condition to x86-64 condition code.
pub fn x86cc_from_intcc(cc: IntCC) -> X86CondCode {
    match cc {
        IntCC::Equal => X86CondCode::E,
        IntCC::NotEqual => X86CondCode::NE,
        IntCC::SignedLessThan => X86CondCode::L,
        IntCC::SignedGreaterThanOrEqual => X86CondCode::GE,
        IntCC::SignedGreaterThan => X86CondCode::G,
        IntCC::SignedLessThanOrEqual => X86CondCode::LE,
        IntCC::UnsignedLessThan => X86CondCode::B,
        IntCC::UnsignedGreaterThanOrEqual => X86CondCode::AE,
        IntCC::UnsignedGreaterThan => X86CondCode::A,
        IntCC::UnsignedLessThanOrEqual => X86CondCode::BE,
    }
}

/// Map tMIR floating-point comparison condition to x86-64 condition code.
///
/// x86-64 UCOMISD sets CF, ZF, PF:
///   Equal: ZF=1, PF=0 (E)
///   LessThan: CF=1 (B)
///   GreaterThan: CF=0, ZF=0 (A)
///   Unordered (NaN): PF=1 (P)
pub fn x86cc_from_floatcc(cc: FloatCC) -> X86CondCode {
    match cc {
        FloatCC::Equal => X86CondCode::E,
        FloatCC::NotEqual => X86CondCode::NE,
        FloatCC::LessThan => X86CondCode::B,
        FloatCC::LessThanOrEqual => X86CondCode::BE,
        FloatCC::GreaterThan => X86CondCode::A,
        FloatCC::GreaterThanOrEqual => X86CondCode::AE,
        FloatCC::Ordered => X86CondCode::NP,
        FloatCC::Unordered => X86CondCode::P,
    }
}

// ---------------------------------------------------------------------------
// x86-64 ISel operand
// ---------------------------------------------------------------------------

/// Operand of an x86-64 ISel-output machine instruction (post-isel, pre-regalloc).
#[derive(Debug, Clone, PartialEq)]
pub enum X86ISelOperand {
    /// Virtual register.
    VReg(VReg),
    /// Physical register (for ABI constraints).
    PReg(X86PReg),
    /// Immediate integer.
    Imm(i64),
    /// Immediate float.
    FImm(f64),
    /// Basic block target (for branches).
    Block(Block),
    /// x86-64 condition code (for Jcc, SETcc, CMOVcc).
    CondCode(X86CondCode),
    /// Global symbol name (for CALL relocations, RIP-relative addressing).
    Symbol(String),
    /// Stack slot index (resolved during frame lowering).
    StackSlot(u32),
    /// Memory address: base register + displacement.
    ///
    /// Used for load/store addressing modes. The base is a register operand
    /// and disp is a signed 32-bit displacement.
    MemAddr {
        base: Box<X86ISelOperand>,
        disp: i32,
    },
}

// ---------------------------------------------------------------------------
// x86-64 ISel instruction
// ---------------------------------------------------------------------------

/// A single x86-64 ISel-output machine instruction (pre-regalloc).
#[derive(Debug, Clone)]
pub struct X86ISelInst {
    pub opcode: X86Opcode,
    pub operands: Vec<X86ISelOperand>,
}

impl X86ISelInst {
    pub fn new(opcode: X86Opcode, operands: Vec<X86ISelOperand>) -> Self {
        Self { opcode, operands }
    }
}

// ---------------------------------------------------------------------------
// x86-64 ISel basic block
// ---------------------------------------------------------------------------

/// An x86-64 ISel-output basic block of machine instructions.
#[derive(Debug, Clone, Default)]
pub struct X86ISelBlock {
    pub insts: Vec<X86ISelInst>,
    pub successors: Vec<Block>,
}

// ---------------------------------------------------------------------------
// x86-64 ISel function
// ---------------------------------------------------------------------------

/// An x86-64 ISel-output function containing X86ISelInsts with VRegs.
#[derive(Debug, Clone)]
pub struct X86ISelFunction {
    pub name: String,
    pub sig: Signature,
    pub blocks: HashMap<Block, X86ISelBlock>,
    pub block_order: Vec<Block>,
    pub next_vreg: u32,
}

impl X86ISelFunction {
    pub fn new(name: String, sig: Signature) -> Self {
        Self {
            name,
            sig,
            blocks: HashMap::new(),
            block_order: Vec::new(),
            next_vreg: 0,
        }
    }

    /// Emit a machine instruction into the given block.
    pub fn push_inst(&mut self, block: Block, inst: X86ISelInst) {
        self.blocks.entry(block).or_default().insts.push(inst);
    }

    /// Add a block to the function (if not already present).
    pub fn ensure_block(&mut self, block: Block) {
        if !self.blocks.contains_key(&block) {
            self.blocks.insert(block, X86ISelBlock::default());
            self.block_order.push(block);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helper enums
// ---------------------------------------------------------------------------

/// Arithmetic operation classification for opcode selection.
#[derive(Debug, Clone, Copy)]
enum X86ArithOp {
    Add,
    Sub,
    Imul,
}

/// Logical operation classification for opcode selection.
#[derive(Debug, Clone, Copy)]
enum X86LogicOp {
    And,
    Or,
    Xor,
}

/// Integer unary operation classification.
#[derive(Debug, Clone, Copy)]
enum X86UnaryOp {
    Neg,
    Not,
}

// ---------------------------------------------------------------------------
// x86-64 instruction selector
// ---------------------------------------------------------------------------

/// x86-64 instruction selector.
///
/// Walks tMIR blocks in order, selects each instruction into one or more
/// x86-64 X86ISelInsts, tracking value -> VReg mappings. After selection,
/// `finalize()` returns the completed X86ISelFunction.
pub struct X86InstructionSelector {
    func: X86ISelFunction,
    /// tMIR Value -> machine operand mapping.
    value_map: HashMap<Value, X86ISelOperand>,
    /// Type of each value, tracked for selecting correct instruction width.
    value_types: HashMap<Value, Type>,
}

impl X86InstructionSelector {
    /// Create a new x86-64 instruction selector for the given function.
    pub fn new(name: String, sig: Signature) -> Self {
        Self {
            func: X86ISelFunction::new(name, sig),
            value_map: HashMap::new(),
            value_types: HashMap::new(),
        }
    }

    /// Require that `inst` has at least `n` arguments.
    fn require_args(inst: &Instruction, n: usize, context: &'static str) -> Result<(), X86ISelError> {
        if inst.args.len() < n {
            return Err(X86ISelError::InsufficientArgs {
                expected: n,
                actual: inst.args.len(),
                context,
            });
        }
        Ok(())
    }

    /// Require that `inst` has at least one result.
    fn require_result(inst: &Instruction, context: &'static str) -> Result<(), X86ISelError> {
        if inst.results.is_empty() {
            return Err(X86ISelError::MissingResult(context));
        }
        Ok(())
    }

    /// Allocate a fresh virtual register.
    pub fn new_vreg(&mut self, class: RegClass) -> VReg {
        let id = self.func.next_vreg;
        self.func.next_vreg += 1;
        VReg { id, class }
    }

    /// Record a mapping from tMIR Value to machine operand.
    pub fn define_value(&mut self, val: Value, operand: X86ISelOperand, ty: Type) {
        self.value_map.insert(val, operand);
        self.value_types.insert(val, ty);
    }

    /// Look up the machine operand for a tMIR Value.
    pub fn use_value(&self, val: &Value) -> Result<X86ISelOperand, X86ISelError> {
        self.value_map
            .get(val)
            .cloned()
            .ok_or_else(|| X86ISelError::UndefinedValue(*val))
    }

    /// Get the type of a tMIR Value.
    fn value_type(&self, val: &Value) -> Type {
        self.value_types
            .get(val)
            .cloned()
            .unwrap_or(Type::I64)
    }

    /// Determine if a type is 32-bit (uses 32-bit registers/operations).
    pub fn is_32bit(ty: &Type) -> bool {
        matches!(ty, Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::F32)
    }

    // -----------------------------------------------------------------------
    // Top-level selection
    // -----------------------------------------------------------------------

    /// Select all instructions in a block.
    pub fn select_block(
        &mut self,
        block: Block,
        instructions: &[Instruction],
    ) -> Result<(), X86ISelError> {
        self.func.ensure_block(block);
        for inst in instructions {
            self.select_instruction(inst, block)?;
        }
        Ok(())
    }

    /// Select a single tMIR instruction, emitting X86ISelInsts into the block.
    fn select_instruction(
        &mut self,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        match &inst.opcode {
            // Constants
            Opcode::Iconst { ty, imm } => self.select_iconst(ty.clone(), *imm, inst, block)?,
            Opcode::Fconst { ty, imm } => self.select_fconst(ty.clone(), *imm, inst, block)?,

            // Arithmetic
            Opcode::Iadd if inst.args.len() == 1 => self.select_move_reg(inst, block)?,
            Opcode::Iadd => self.select_arithmetic(X86ArithOp::Add, inst, block)?,
            Opcode::Isub => self.select_arithmetic(X86ArithOp::Sub, inst, block)?,
            Opcode::Imul => self.select_arithmetic(X86ArithOp::Imul, inst, block)?,

            // Unary operations
            Opcode::Ineg => self.select_unary(X86UnaryOp::Neg, inst, block)?,
            Opcode::Bnot => self.select_unary(X86UnaryOp::Not, inst, block)?,

            // Logical operations
            Opcode::Band => self.select_logic(X86LogicOp::And, inst, block)?,
            Opcode::Bor => self.select_logic(X86LogicOp::Or, inst, block)?,
            Opcode::Bxor => self.select_logic(X86LogicOp::Xor, inst, block)?,

            // Comparison
            Opcode::Icmp { cond } => self.select_comparison(*cond, inst, block)?,

            // Memory
            Opcode::Load { ty } => self.select_memory_load(ty.clone(), inst, block)?,
            Opcode::Store => self.select_memory_store(inst, block)?,

            // Control flow
            Opcode::Jump { dest } => self.select_branch(*dest, block)?,
            Opcode::Brif {
                then_dest,
                else_dest,
                ..
            } => self.select_condbranch(inst, *then_dest, *else_dest, block)?,
            Opcode::Return => self.lower_return(inst, block)?,
            Opcode::Call { name } => self.lower_call(name, inst, block)?,

            // Unsupported opcodes emit a NOP placeholder for now
            _ => {
                self.func.push_inst(
                    block,
                    X86ISelInst::new(X86Opcode::Nop, vec![]),
                );
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Select integer constant materialization via MOV r64, imm64.
    fn select_iconst(
        &mut self,
        ty: Type,
        imm: i64,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_result(inst, "Iconst")?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = &inst.results[0];

        // x86-64 MOV r64, imm64 handles all immediate sizes.
        // The encoder will pick the optimal encoding (MOV r32, imm32 for
        // small positive values, which implicitly zero-extends to 64-bit).
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::MovRI,
                vec![X86ISelOperand::VReg(dst), X86ISelOperand::Imm(imm)],
            ),
        );

        self.define_value(*result, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select float constant materialization.
    ///
    /// For the scaffold, we emit a MOVSD from a constant pool address.
    /// This is simplified: a real implementation would emit a RIP-relative
    /// MOVSD load from .rodata.
    fn select_fconst(
        &mut self,
        ty: Type,
        imm: f64,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_result(inst, "Fconst")?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = &inst.results[0];

        match ty {
            Type::F32 | Type::F64 => {}
            _ => return Err(X86ISelError::UnsupportedFconstType(ty)),
        }

        // Emit as FImm placeholder; the encoder/legalization will handle
        // materializing this via constant pool + RIP-relative MOVSD.
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::MovsdRR,
                vec![X86ISelOperand::VReg(dst), X86ISelOperand::FImm(imm)],
            ),
        );

        self.define_value(*result, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Register copy (single-arg Iadd used as COPY by adapter)
    // -----------------------------------------------------------------------

    /// Select a register-to-register copy (MOV r64, r64).
    pub fn select_move_reg(
        &mut self,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 1, "Copy")?;
        Self::require_result(inst, "Copy")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];
        let ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let dst = if let Some(X86ISelOperand::VReg(v)) = self.value_map.get(&result_val) {
            *v
        } else {
            let class = reg_class_for_type(&ty);
            self.new_vreg(class)
        };

        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::MovRR,
                vec![X86ISelOperand::VReg(dst), src],
            ),
        );

        self.define_value(result_val, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    /// Select binary arithmetic: ADD, SUB, IMUL.
    ///
    /// x86-64 two-operand form: the destination is the same as the first source.
    /// In SSA/pre-regalloc form we emit three-address: dst = op(lhs, rhs),
    /// with an implicit MOV dst, lhs before the operation handled by regalloc.
    fn select_arithmetic(
        &mut self,
        op: X86ArithOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 2, "Arithmetic")?;
        Self::require_result(inst, "Arithmetic")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match op {
            X86ArithOp::Add => X86Opcode::AddRR,
            X86ArithOp::Sub => X86Opcode::SubRR,
            X86ArithOp::Imul => X86Opcode::ImulRR,
        };

        // Emit: dst = op lhs, rhs (three-address pseudo)
        self.func.push_inst(
            block,
            X86ISelInst::new(
                opc,
                vec![X86ISelOperand::VReg(dst), lhs, rhs],
            ),
        );

        self.define_value(result_val, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Logical operations
    // -----------------------------------------------------------------------

    /// Select binary logical: AND, OR, XOR.
    fn select_logic(
        &mut self,
        op: X86LogicOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 2, "Logic")?;
        Self::require_result(inst, "Logic")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match op {
            X86LogicOp::And => X86Opcode::AndRR,
            X86LogicOp::Or => X86Opcode::OrRR,
            X86LogicOp::Xor => X86Opcode::XorRR,
        };

        self.func.push_inst(
            block,
            X86ISelInst::new(
                opc,
                vec![X86ISelOperand::VReg(dst), lhs, rhs],
            ),
        );

        self.define_value(result_val, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unary operations
    // -----------------------------------------------------------------------

    /// Select integer unary operation: NEG (two's complement), NOT (bitwise).
    fn select_unary(
        &mut self,
        op: X86UnaryOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 1, "Unary")?;
        Self::require_result(inst, "Unary")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        let opc = match op {
            X86UnaryOp::Neg => X86Opcode::Neg,
            X86UnaryOp::Not => X86Opcode::Not,
        };

        self.func.push_inst(
            block,
            X86ISelInst::new(opc, vec![X86ISelOperand::VReg(dst), src]),
        );

        self.define_value(result_val, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    /// Select integer comparison: CMP + SETcc pattern.
    ///
    /// tMIR `Icmp(cond, lhs, rhs) -> bool_result` becomes:
    ///   CMP lhs, rhs           (sets RFLAGS)
    ///   MOV dst, 0             (zero-initialize result)
    ///   Jcc (implicit SETcc)   (set byte based on condition)
    ///
    /// For the scaffold, we emit CMP + a flag-setting pair. The actual
    /// SETcc will be handled by later lowering; here we record the condition
    /// code as metadata and use a CMP + MOV-immediate pattern.
    pub fn select_comparison(
        &mut self,
        cond: IntCC,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 2, "Icmp")?;
        Self::require_result(inst, "Icmp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let x86cc = x86cc_from_intcc(cond);

        // CMP lhs, rhs (sets RFLAGS)
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::CmpRR,
                vec![lhs, rhs],
            ),
        );

        // Materialize boolean result: we encode this as a pseudo that
        // carries the condition code. The register allocator / later
        // lowering will expand to SETcc + MOVZX.
        let dst = self.new_vreg(RegClass::Gpr32);
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::MovRI,
                vec![
                    X86ISelOperand::VReg(dst),
                    X86ISelOperand::Imm(0),
                    X86ISelOperand::CondCode(x86cc),
                ],
            ),
        );

        self.define_value(result_val, X86ISelOperand::VReg(dst), Type::B1);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Branches
    // -----------------------------------------------------------------------

    /// Select unconditional branch: JMP target.
    pub fn select_branch(
        &mut self,
        dest: Block,
        block: Block,
    ) -> Result<(), X86ISelError> {
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::Jmp,
                vec![X86ISelOperand::Block(dest)],
            ),
        );
        self.func
            .blocks
            .entry(block)
            .or_default()
            .successors
            .push(dest);
        Ok(())
    }

    /// Select conditional branch: CMP cond, #0 + Jcc + JMP.
    ///
    /// tMIR `Brif(cond, then, else)` becomes:
    ///   CMP cond_vreg, #0
    ///   Jcc NE, then_block     (branch if true)
    ///   JMP else_block         (fallthrough to else)
    pub fn select_condbranch(
        &mut self,
        inst: &Instruction,
        then_dest: Block,
        else_dest: Block,
        block: Block,
    ) -> Result<(), X86ISelError> {
        let cond_val = inst.args[0];
        let cond_op = self.use_value(&cond_val)?;

        // CMP cond, #0 to test boolean
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::CmpRI,
                vec![cond_op, X86ISelOperand::Imm(0)],
            ),
        );

        // Jcc NE, then_block (condition was nonzero = true)
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::Jcc,
                vec![
                    X86ISelOperand::CondCode(X86CondCode::NE),
                    X86ISelOperand::Block(then_dest),
                ],
            ),
        );

        // JMP else_block (unconditional fallthrough)
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::Jmp,
                vec![X86ISelOperand::Block(else_dest)],
            ),
        );

        let mblock = self.func.blocks.entry(block).or_default();
        mblock.successors.push(then_dest);
        mblock.successors.push(else_dest);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Memory operations
    // -----------------------------------------------------------------------

    /// Select memory load: MOV r64, [base + 0].
    ///
    /// tMIR `Load(ty, addr) -> result`:
    ///   MOV dst, [addr_vreg + 0]
    pub fn select_memory_load(
        &mut self,
        ty: Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 1, "Load")?;
        Self::require_result(inst, "Load")?;

        let addr_val = inst.args[0];
        let result_val = inst.results[0];

        let addr = self.use_value(&addr_val)?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        // MovRM: dst = [base + 0]
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::MovRM,
                vec![
                    X86ISelOperand::VReg(dst),
                    X86ISelOperand::MemAddr {
                        base: Box::new(addr),
                        disp: 0,
                    },
                ],
            ),
        );

        self.define_value(result_val, X86ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select memory store: MOV [base + 0], value.
    ///
    /// tMIR `Store(value, addr)`:
    ///   MOV [addr_vreg + 0], value_vreg
    pub fn select_memory_store(
        &mut self,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        Self::require_args(inst, 2, "Store")?;

        let value_val = inst.args[0];
        let addr_val = inst.args[1];

        let value = self.use_value(&value_val)?;
        let addr = self.use_value(&addr_val)?;

        // MovMR: [base + 0] = value
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::MovMR,
                vec![
                    X86ISelOperand::MemAddr {
                        base: Box::new(addr),
                        disp: 0,
                    },
                    value,
                ],
            ),
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Call (System V AMD64 ABI)
    // -----------------------------------------------------------------------

    /// Lower a function call using System V AMD64 ABI.
    ///
    /// Integer arguments: RDI, RSI, RDX, RCX, R8, R9 (in order)
    /// FP arguments: XMM0-XMM7
    /// Return value: RAX (integer), XMM0 (float)
    ///
    /// 1. Move arguments to physical registers
    /// 2. Emit CALL symbol
    /// 3. Copy return value from RAX/XMM0 to vreg
    pub fn lower_call(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        let mut gpr_idx: usize = 0;
        let mut xmm_idx: usize = 0;
        let mut stack_offset: i64 = 0;

        // Move arguments to ABI registers
        for val in &inst.args {
            let src = self.use_value(val)?;
            let ty = self.value_type(val);

            let is_fp = matches!(ty, Type::F32 | Type::F64);

            if is_fp {
                if xmm_idx < X86_ARG_XMMS.len() {
                    // Move to XMM argument register
                    self.func.push_inst(
                        block,
                        X86ISelInst::new(
                            X86Opcode::MovsdRR,
                            vec![
                                X86ISelOperand::PReg(X86_ARG_XMMS[xmm_idx]),
                                src,
                            ],
                        ),
                    );
                    xmm_idx += 1;
                } else {
                    // Stack-passed FP argument
                    self.func.push_inst(
                        block,
                        X86ISelInst::new(
                            X86Opcode::MovMR,
                            vec![
                                X86ISelOperand::MemAddr {
                                    base: Box::new(X86ISelOperand::PReg(RSP)),
                                    disp: stack_offset as i32,
                                },
                                src,
                            ],
                        ),
                    );
                    stack_offset += 8;
                }
            } else {
                if gpr_idx < X86_ARG_GPRS.len() {
                    // Move to GPR argument register
                    self.func.push_inst(
                        block,
                        X86ISelInst::new(
                            X86Opcode::MovRR,
                            vec![
                                X86ISelOperand::PReg(X86_ARG_GPRS[gpr_idx]),
                                src,
                            ],
                        ),
                    );
                    gpr_idx += 1;
                } else {
                    // Stack-passed integer argument
                    self.func.push_inst(
                        block,
                        X86ISelInst::new(
                            X86Opcode::Push,
                            vec![src],
                        ),
                    );
                    stack_offset += 8;
                }
            }
        }

        // Emit CALL with symbol for relocation
        self.func.push_inst(
            block,
            X86ISelInst::new(
                X86Opcode::Call,
                vec![X86ISelOperand::Symbol(name.to_string())],
            ),
        );

        // Copy return values from physical registers to vregs
        for (i, result_val) in inst.results.iter().enumerate() {
            let ret_ty = self.value_type(result_val);
            let is_fp = matches!(ret_ty, Type::F32 | Type::F64);
            let class = reg_class_for_type(&ret_ty);
            let dst = self.new_vreg(class);

            if is_fp {
                // Return in XMM0 (or XMM1 for second return)
                let xmm_ret = if i == 0 {
                    x86_64_regs::XMM0
                } else {
                    x86_64_regs::XMM1
                };
                self.func.push_inst(
                    block,
                    X86ISelInst::new(
                        X86Opcode::MovsdRR,
                        vec![
                            X86ISelOperand::VReg(dst),
                            X86ISelOperand::PReg(xmm_ret),
                        ],
                    ),
                );
            } else {
                // Return in RAX (or RDX for second return)
                let gpr_ret = X86_RET_GPRS[i.min(X86_RET_GPRS.len() - 1)];
                self.func.push_inst(
                    block,
                    X86ISelInst::new(
                        X86Opcode::MovRR,
                        vec![
                            X86ISelOperand::VReg(dst),
                            X86ISelOperand::PReg(gpr_ret),
                        ],
                    ),
                );
            }

            self.define_value(*result_val, X86ISelOperand::VReg(dst), ret_ty);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Return
    // -----------------------------------------------------------------------

    /// Lower return instruction.
    ///
    /// Move return values to RAX (integer) or XMM0 (float), then emit RET.
    pub fn lower_return(
        &mut self,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), X86ISelError> {
        // Move each return value into its designated physical register
        for (i, val) in inst.args.iter().enumerate() {
            let src = self.use_value(val)?;
            let ty = self.value_type(val);
            let is_fp = matches!(ty, Type::F32 | Type::F64);

            if is_fp {
                let xmm_ret = if i == 0 {
                    x86_64_regs::XMM0
                } else {
                    x86_64_regs::XMM1
                };
                self.func.push_inst(
                    block,
                    X86ISelInst::new(
                        X86Opcode::MovsdRR,
                        vec![X86ISelOperand::PReg(xmm_ret), src],
                    ),
                );
            } else {
                let gpr_ret = X86_RET_GPRS[i.min(X86_RET_GPRS.len() - 1)];
                self.func.push_inst(
                    block,
                    X86ISelInst::new(
                        X86Opcode::MovRR,
                        vec![X86ISelOperand::PReg(gpr_ret), src],
                    ),
                );
            }
        }

        // Emit RET
        self.func.push_inst(
            block,
            X86ISelInst::new(X86Opcode::Ret, vec![]),
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Formal arguments (System V AMD64 ABI)
    // -----------------------------------------------------------------------

    /// Lower formal function arguments from physical registers to vregs.
    ///
    /// By convention, formal args are Value(0), Value(1), ..., Value(n-1).
    /// Integer args arrive in RDI, RSI, RDX, RCX, R8, R9.
    /// FP args arrive in XMM0-XMM7.
    pub fn lower_formal_arguments(
        &mut self,
        sig: &Signature,
        entry_block: Block,
    ) -> Result<(), X86ISelError> {
        self.func.ensure_block(entry_block);

        let mut gpr_idx: usize = 0;
        let mut xmm_idx: usize = 0;

        for (i, ty) in sig.params.iter().enumerate() {
            let val = Value(i as u32);
            let class = reg_class_for_type(ty);
            let vreg = self.new_vreg(class);
            let is_fp = matches!(ty, Type::F32 | Type::F64);

            if is_fp {
                if xmm_idx < X86_ARG_XMMS.len() {
                    self.func.push_inst(
                        entry_block,
                        X86ISelInst::new(
                            X86Opcode::MovsdRR,
                            vec![
                                X86ISelOperand::VReg(vreg),
                                X86ISelOperand::PReg(X86_ARG_XMMS[xmm_idx]),
                            ],
                        ),
                    );
                    xmm_idx += 1;
                }
            } else if gpr_idx < X86_ARG_GPRS.len() {
                self.func.push_inst(
                    entry_block,
                    X86ISelInst::new(
                        X86Opcode::MovRR,
                        vec![
                            X86ISelOperand::VReg(vreg),
                            X86ISelOperand::PReg(X86_ARG_GPRS[gpr_idx]),
                        ],
                    ),
                );
                gpr_idx += 1;
            }

            self.define_value(val, X86ISelOperand::VReg(vreg), ty.clone());
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Finalize
    // -----------------------------------------------------------------------

    /// Consume the selector and return the completed X86ISelFunction.
    pub fn finalize(self) -> X86ISelFunction {
        self.func
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::{Block, Instruction, Opcode, Value};
    use crate::types::Type;
    use llvm2_ir::x86_64_regs::{RAX, RCX, RDI, RDX, RSI, R8, R9};

    /// Helper: create an empty instruction selector and entry block.
    fn make_empty_isel() -> (X86InstructionSelector, Block) {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = X86InstructionSelector::new("test_fn".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);
        (isel, entry)
    }

    /// Helper: define a Value with a vreg in the selector.
    fn define_vreg(
        isel: &mut X86InstructionSelector,
        val: Value,
        ty: Type,
    ) -> VReg {
        let class = reg_class_for_type(&ty);
        let vreg = isel.new_vreg(class);
        isel.define_value(val, X86ISelOperand::VReg(vreg), ty);
        vreg
    }

    // -----------------------------------------------------------------------
    // Condition code mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_intcc_to_x86cc() {
        assert_eq!(x86cc_from_intcc(IntCC::Equal), X86CondCode::E);
        assert_eq!(x86cc_from_intcc(IntCC::NotEqual), X86CondCode::NE);
        assert_eq!(x86cc_from_intcc(IntCC::SignedLessThan), X86CondCode::L);
        assert_eq!(x86cc_from_intcc(IntCC::SignedGreaterThan), X86CondCode::G);
        assert_eq!(x86cc_from_intcc(IntCC::SignedGreaterThanOrEqual), X86CondCode::GE);
        assert_eq!(x86cc_from_intcc(IntCC::SignedLessThanOrEqual), X86CondCode::LE);
        assert_eq!(x86cc_from_intcc(IntCC::UnsignedLessThan), X86CondCode::B);
        assert_eq!(x86cc_from_intcc(IntCC::UnsignedGreaterThan), X86CondCode::A);
        assert_eq!(x86cc_from_intcc(IntCC::UnsignedGreaterThanOrEqual), X86CondCode::AE);
        assert_eq!(x86cc_from_intcc(IntCC::UnsignedLessThanOrEqual), X86CondCode::BE);
    }

    #[test]
    fn test_floatcc_to_x86cc() {
        assert_eq!(x86cc_from_floatcc(FloatCC::Equal), X86CondCode::E);
        assert_eq!(x86cc_from_floatcc(FloatCC::NotEqual), X86CondCode::NE);
        assert_eq!(x86cc_from_floatcc(FloatCC::LessThan), X86CondCode::B);
        assert_eq!(x86cc_from_floatcc(FloatCC::LessThanOrEqual), X86CondCode::BE);
        assert_eq!(x86cc_from_floatcc(FloatCC::GreaterThan), X86CondCode::A);
        assert_eq!(x86cc_from_floatcc(FloatCC::GreaterThanOrEqual), X86CondCode::AE);
        assert_eq!(x86cc_from_floatcc(FloatCC::Ordered), X86CondCode::NP);
        assert_eq!(x86cc_from_floatcc(FloatCC::Unordered), X86CondCode::P);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_add() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, X86Opcode::AddRR);
        assert_eq!(insts[0].operands.len(), 3); // dst, lhs, rhs
    }

    #[test]
    fn test_select_sub() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::SubRR);
    }

    #[test]
    fn test_select_imul() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Imul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::ImulRR);
    }

    // -----------------------------------------------------------------------
    // Logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_and() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Band,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::AndRR);
    }

    #[test]
    fn test_select_or() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bor,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::OrRR);
    }

    #[test]
    fn test_select_xor() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bxor,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::XorRR);
    }

    // -----------------------------------------------------------------------
    // Unary
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_neg() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::Neg);
        assert_eq!(inst.operands.len(), 2); // dst, src
    }

    #[test]
    fn test_select_not() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bnot,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::Not);
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_cmp_eq() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp {
                    cond: IntCC::Equal,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        // Should have CMP + MOV (with condition code)
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].opcode, X86Opcode::CmpRR);
        assert_eq!(insts[1].opcode, X86Opcode::MovRI);
        // Condition code should be E (equal)
        assert!(
            insts[1]
                .operands
                .iter()
                .any(|op| matches!(op, X86ISelOperand::CondCode(X86CondCode::E)))
        );
    }

    #[test]
    fn test_select_cmp_signed_lt() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I32);
        define_vreg(&mut isel, Value(1), Type::I32);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp {
                    cond: IntCC::SignedLessThan,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(insts[0].opcode, X86Opcode::CmpRR);
        assert!(
            insts[1]
                .operands
                .iter()
                .any(|op| matches!(op, X86ISelOperand::CondCode(X86CondCode::L)))
        );
    }

    // -----------------------------------------------------------------------
    // Branches
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_jump() {
        let (mut isel, entry) = make_empty_isel();
        let target = Block(1);
        isel.func.ensure_block(target);

        isel.select_branch(target, entry).unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, X86Opcode::Jmp);
        assert_eq!(insts[0].operands[0], X86ISelOperand::Block(target));

        // Check successor edge
        assert!(mfunc.blocks[&entry].successors.contains(&target));
    }

    #[test]
    fn test_select_condbranch() {
        let (mut isel, entry) = make_empty_isel();
        let then_block = Block(1);
        let else_block = Block(2);
        isel.func.ensure_block(then_block);
        isel.func.ensure_block(else_block);
        define_vreg(&mut isel, Value(0), Type::B1);

        isel.select_condbranch(
            &Instruction {
                opcode: Opcode::Brif {
                    cond: Value(0),
                    then_dest: then_block,
                    else_dest: else_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            then_block,
            else_block,
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        // CMP, Jcc, Jmp
        assert_eq!(insts.len(), 3);
        assert_eq!(insts[0].opcode, X86Opcode::CmpRI);
        assert_eq!(insts[1].opcode, X86Opcode::Jcc);
        assert_eq!(insts[2].opcode, X86Opcode::Jmp);

        // Successor edges
        let succs = &mfunc.blocks[&entry].successors;
        assert!(succs.contains(&then_block));
        assert!(succs.contains(&else_block));
    }

    // -----------------------------------------------------------------------
    // Moves / constants
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_mov_reg() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd, // Single-arg Iadd = COPY
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::MovRR);
    }

    #[test]
    fn test_select_iconst() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I64,
                    imm: 42,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::MovRI);
        assert_eq!(inst.operands[1], X86ISelOperand::Imm(42));
    }

    #[test]
    fn test_select_iconst_negative() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: -1,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::MovRI);
        assert_eq!(inst.operands[1], X86ISelOperand::Imm(-1));
    }

    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_load() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64); // address

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::MovRM);
        assert_eq!(inst.operands.len(), 2); // dst, memaddr
    }

    #[test]
    fn test_select_store() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64); // value
        define_vreg(&mut isel, Value(1), Type::I64); // address

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, X86Opcode::MovMR);
        assert_eq!(inst.operands.len(), 2); // memaddr, value
    }

    // -----------------------------------------------------------------------
    // Call / Return
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_call_system_v() {
        let sig = Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = X86InstructionSelector::new("caller".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        // Define arg values
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        isel.lower_call(
            "callee",
            &Instruction {
                opcode: Opcode::Call {
                    name: "callee".to_string(),
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Should have: MOV RDI, arg0 + MOV RSI, arg1 + CALL + MOV result, RAX
        assert_eq!(insts.len(), 4);

        // First arg goes to RDI
        assert_eq!(insts[0].opcode, X86Opcode::MovRR);
        assert_eq!(insts[0].operands[0], X86ISelOperand::PReg(RDI));

        // Second arg goes to RSI
        assert_eq!(insts[1].opcode, X86Opcode::MovRR);
        assert_eq!(insts[1].operands[0], X86ISelOperand::PReg(RSI));

        // CALL callee
        assert_eq!(insts[2].opcode, X86Opcode::Call);
        assert_eq!(
            insts[2].operands[0],
            X86ISelOperand::Symbol("callee".to_string())
        );

        // Result copied from RAX
        assert_eq!(insts[3].opcode, X86Opcode::MovRR);
        assert_eq!(insts[3].operands[1], X86ISelOperand::PReg(RAX));
    }

    #[test]
    fn test_lower_call_six_args() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = X86InstructionSelector::new("caller".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        // Define 6 arg values
        for i in 0..6 {
            define_vreg(&mut isel, Value(i), Type::I64);
        }

        isel.lower_call(
            "target",
            &Instruction {
                opcode: Opcode::Call {
                    name: "target".to_string(),
                },
                args: vec![
                    Value(0),
                    Value(1),
                    Value(2),
                    Value(3),
                    Value(4),
                    Value(5),
                ],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // 6 MOV + CALL = 7
        assert_eq!(insts.len(), 7);

        // Verify System V register order: RDI, RSI, RDX, RCX, R8, R9
        let expected_regs = [RDI, RSI, RDX, RCX, R8, R9];
        for (i, expected) in expected_regs.iter().enumerate() {
            assert_eq!(insts[i].opcode, X86Opcode::MovRR);
            assert_eq!(insts[i].operands[0], X86ISelOperand::PReg(*expected));
        }
    }

    #[test]
    fn test_lower_return() {
        let sig = Signature {
            params: vec![],
            returns: vec![Type::I64],
        };
        let mut isel = X86InstructionSelector::new("test_fn".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);
        define_vreg(&mut isel, Value(0), Type::I64);

        isel.lower_return(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // MOV RAX, value + RET
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].opcode, X86Opcode::MovRR);
        assert_eq!(insts[0].operands[0], X86ISelOperand::PReg(RAX));
        assert_eq!(insts[1].opcode, X86Opcode::Ret);
    }

    #[test]
    fn test_lower_return_void() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = X86InstructionSelector::new("test_fn".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        isel.lower_return(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Just RET, no MOV
        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, X86Opcode::Ret);
    }

    // -----------------------------------------------------------------------
    // Formal arguments
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_formal_arguments() {
        let sig = Signature {
            params: vec![Type::I64, Type::I32],
            returns: vec![],
        };
        let mut isel = X86InstructionSelector::new("test_fn".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Two MOV from RDI, RSI
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].opcode, X86Opcode::MovRR);
        assert_eq!(insts[0].operands[1], X86ISelOperand::PReg(RDI));
        assert_eq!(insts[1].opcode, X86Opcode::MovRR);
        assert_eq!(insts[1].operands[1], X86ISelOperand::PReg(RSI));
    }

    // -----------------------------------------------------------------------
    // Full function selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_block_multiple_instructions() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        let instructions = vec![
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(2), Value(1)],
                results: vec![Value(3)],
            },
        ];

        isel.select_block(entry, &instructions).unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].opcode, X86Opcode::AddRR);
        assert_eq!(insts[1].opcode, X86Opcode::SubRR);
    }

    #[test]
    fn test_select_block_order_preserved() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = X86InstructionSelector::new("test".to_string(), sig);
        let b0 = Block(0);
        let b1 = Block(1);
        let b2 = Block(2);
        isel.func.ensure_block(b0);
        isel.func.ensure_block(b1);
        isel.func.ensure_block(b2);

        let mfunc = isel.finalize();
        assert_eq!(mfunc.block_order, vec![b0, b1, b2]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_undefined_value_error() {
        let (mut isel, entry) = make_empty_isel();

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(99), Value(100)],
                results: vec![Value(101)],
            },
            entry,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_args_error() {
        let (mut isel, entry) = make_empty_isel();

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![], // No args for Iadd
                results: vec![Value(0)],
            },
            entry,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_result_error() {
        let (mut isel, entry) = make_empty_isel();
        define_vreg(&mut isel, Value(0), Type::I64);
        define_vreg(&mut isel, Value(1), Type::I64);

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![], // No results
            },
            entry,
        );

        assert!(result.is_err());
    }
}

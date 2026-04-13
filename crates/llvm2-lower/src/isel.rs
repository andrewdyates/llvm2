// llvm2-lower/isel.rs - AArch64 instruction selection
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: LLVM AArch64ISelLowering.cpp (tree-pattern matching + late combines)
// Reference: designs/2026-04-12-aarch64-backend.md (two-phase ISel architecture)

//! AArch64 instruction selection: tMIR SSA IR -> AArch64 MachIR with virtual registers.
//!
//! Phase 1 (this module): Walk tMIR blocks in reverse postorder, match each
//! instruction bottom-up, emit AArch64 MachInst with VRegs. This covers
//! arithmetic, comparisons, branches, calls, returns, loads, stores, and
//! constants.
//!
//! Phase 2 (llvm2-opt late combines): Address-mode formation, cmp+branch
//! fusing, csel/cset formation. Those depend on one-use analysis not available
//! during tree matching.

use std::collections::HashMap;

use crate::abi::{gpr, AppleAArch64ABI, ArgLocation, PReg};
use crate::function::Signature;
use crate::instructions::{Block, FloatCC, Instruction, IntCC, Opcode, Value};
use crate::types::Type;
use thiserror::Error;

// Import canonical register types from llvm2-ir.
use llvm2_ir::regs::{RegClass, VReg, SP};

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
        // Aggregates are handled via pointers at the machine level.
        Type::Struct(_) | Type::Array(_, _) => RegClass::Gpr64,
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors during instruction selection.
#[derive(Debug, Error)]
pub enum ISelError {
    #[error("value {0:?} not defined before use")]
    UndefinedValue(Value),
    #[error("unsupported type for Fconst: {0:?} (expected F32 or F64)")]
    UnsupportedFconstType(Type),
    #[error("unsupported ABI location for return value: stack-passed returns are not supported")]
    UnsupportedReturnLocation,
    #[error("malformed instruction: expected at least {expected} args, got {actual} (opcode context: {context})")]
    InsufficientArgs {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    #[error("malformed instruction: expected at least 1 result (opcode context: {0})")]
    MissingResult(&'static str),
    #[error("StructGep on non-struct type: {0:?}")]
    StructGepNonStruct(Type),
    #[error("StructGep field index {index} out of range for struct with {field_count} fields")]
    StructGepOutOfRange { index: u32, field_count: usize },
    #[error("aggregate type too large for inline return: {0} bytes")]
    AggregateReturnTooLarge(u32),
}

// ---------------------------------------------------------------------------
// AArch64 opcode — re-exported from llvm2-ir (unified, issue #73)
// ---------------------------------------------------------------------------
//
// ISel now uses the canonical `llvm2_ir::AArch64Opcode` directly instead of
// its own per-width opcode enum. Width information (32-bit vs 64-bit) is
// conveyed by the register class of the operands, not the opcode name.
//
// The pipeline no longer needs `map_isel_opcode()` to translate between
// ISel opcodes and IR opcodes — they are the same type.
// ---------------------------------------------------------------------------

pub use llvm2_ir::AArch64Opcode;

// ---------------------------------------------------------------------------
// Machine operand
// ---------------------------------------------------------------------------

/// Operand of an ISel-output machine instruction (post-isel, pre-regalloc).
///
/// Separate from `llvm2_ir::MachOperand`: includes ISel-specific variants
/// (CondCode, Symbol) not present in the canonical IR, and uses `Block`
/// (LIR block ID) instead of `BlockId` (MachIR block ID).
#[derive(Debug, Clone, PartialEq)]
pub enum MachOperand {
    /// Virtual register.
    VReg(VReg),
    /// Physical register (for ABI constraints, e.g. X0 for return).
    PReg(PReg),
    /// Immediate integer.
    Imm(i64),
    /// Immediate float.
    FImm(f64),
    /// Basic block target (for branches).
    Block(Block),
    /// Condition code (for B.cond, CSET).
    CondCode(AArch64CC),
    /// Global symbol name (for ADRP/ADD relocations).
    Symbol(String),
    /// Stack slot index (resolved during frame lowering).
    StackSlot(u32),
}

/// AArch64 condition codes (NZCV-based) for ISel output.
///
/// Separate from `llvm2_ir::cc::AArch64CC` (which aliases `CondCode`):
/// this version includes `from_intcc`/`from_floatcc` conversion methods
/// for tMIR comparison conditions. The canonical `CondCode` in llvm2-ir
/// is the hardware-level encoding without tMIR conversion logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AArch64CC {
    EQ,  // Equal (Z=1)
    NE,  // Not equal (Z=0)
    HS,  // Unsigned higher or same (C=1) (aka CS)
    LO,  // Unsigned lower (C=0) (aka CC)
    MI,  // Negative (N=1)
    PL,  // Positive or zero (N=0)
    VS,  // Overflow (V=1)
    VC,  // No overflow (V=0)
    HI,  // Unsigned higher (C=1 & Z=0)
    LS,  // Unsigned lower or same (C=0 | Z=1)
    GE,  // Signed greater or equal (N=V)
    LT,  // Signed less than (N!=V)
    GT,  // Signed greater than (Z=0 & N=V)
    LE,  // Signed less or equal (Z=1 | N!=V)
}

impl AArch64CC {
    /// Map tMIR integer comparison condition to AArch64 condition code.
    pub fn from_intcc(cc: IntCC) -> Self {
        match cc {
            IntCC::Equal => AArch64CC::EQ,
            IntCC::NotEqual => AArch64CC::NE,
            IntCC::SignedLessThan => AArch64CC::LT,
            IntCC::SignedGreaterThanOrEqual => AArch64CC::GE,
            IntCC::SignedGreaterThan => AArch64CC::GT,
            IntCC::SignedLessThanOrEqual => AArch64CC::LE,
            IntCC::UnsignedLessThan => AArch64CC::LO,
            IntCC::UnsignedGreaterThanOrEqual => AArch64CC::HS,
            IntCC::UnsignedGreaterThan => AArch64CC::HI,
            IntCC::UnsignedLessThanOrEqual => AArch64CC::LS,
        }
    }

    /// Map tMIR floating-point comparison condition to AArch64 condition code.
    ///
    /// AArch64 FCMP sets NZCV as follows for ordered comparisons:
    ///   Equal: EQ (Z=1)
    ///   LessThan: MI (N=1)
    ///   GreaterThan: GT (Z=0, N=V)
    ///   Unordered (NaN): VS (V=1)
    pub fn from_floatcc(cc: FloatCC) -> Self {
        match cc {
            FloatCC::Equal => AArch64CC::EQ,
            FloatCC::NotEqual => AArch64CC::NE,
            FloatCC::LessThan => AArch64CC::MI,
            FloatCC::LessThanOrEqual => AArch64CC::LS,
            FloatCC::GreaterThan => AArch64CC::GT,
            FloatCC::GreaterThanOrEqual => AArch64CC::GE,
            FloatCC::Ordered => AArch64CC::VC,
            FloatCC::Unordered => AArch64CC::VS,
        }
    }

    /// Return the inverted condition code.
    pub fn invert(self) -> Self {
        match self {
            AArch64CC::EQ => AArch64CC::NE,
            AArch64CC::NE => AArch64CC::EQ,
            AArch64CC::HS => AArch64CC::LO,
            AArch64CC::LO => AArch64CC::HS,
            AArch64CC::MI => AArch64CC::PL,
            AArch64CC::PL => AArch64CC::MI,
            AArch64CC::VS => AArch64CC::VC,
            AArch64CC::VC => AArch64CC::VS,
            AArch64CC::HI => AArch64CC::LS,
            AArch64CC::LS => AArch64CC::HI,
            AArch64CC::GE => AArch64CC::LT,
            AArch64CC::LT => AArch64CC::GE,
            AArch64CC::GT => AArch64CC::LE,
            AArch64CC::LE => AArch64CC::GT,
        }
    }
}

// ---------------------------------------------------------------------------
// ISel-level machine instruction
// ---------------------------------------------------------------------------

/// A single ISel-output machine instruction (pre-regalloc).
///
/// Simpler than `llvm2_ir::MachInst`: no flags, no implicit defs/uses, no
/// proof annotations. Those are added during the ISel-to-MachIR translation.
#[derive(Debug, Clone)]
pub struct MachInst {
    pub opcode: AArch64Opcode,
    pub operands: Vec<MachOperand>,
}

impl MachInst {
    pub fn new(opcode: AArch64Opcode, operands: Vec<MachOperand>) -> Self {
        Self { opcode, operands }
    }
}

// ---------------------------------------------------------------------------
// ISel-level machine basic block
// ---------------------------------------------------------------------------

/// An ISel-output basic block of machine instructions.
///
/// Uses `Vec<MachInst>` inline (not arena-indexed), and `successors` (not
/// `succs`/`preds`). This is the ISel output format; the canonical
/// `llvm2_ir::MachBlock` uses arena-indexed `Vec<InstId>` and explicit
/// predecessor tracking.
#[derive(Debug, Clone, Default)]
pub struct MachBlock {
    pub insts: Vec<MachInst>,
    pub successors: Vec<Block>,
}

// ---------------------------------------------------------------------------
// ISel-level machine function
// ---------------------------------------------------------------------------

/// An ISel-output function containing MachInsts with VRegs.
///
/// Uses `HashMap<Block, MachBlock>` for blocks (convenient for ISel
/// construction), while `llvm2_ir::MachFunction` uses `Vec<MachBlock>`
/// indexed by `BlockId` (cache-friendly for optimization passes).
#[derive(Debug, Clone)]
pub struct MachFunction {
    pub name: String,
    pub sig: Signature,
    pub blocks: HashMap<Block, MachBlock>,
    pub block_order: Vec<Block>,
    pub next_vreg: u32,
}

impl MachFunction {
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
    pub fn push_inst(&mut self, block: Block, inst: MachInst) {
        self.blocks.entry(block).or_default().insts.push(inst);
    }

    /// Add a block to the function (if not already present).
    pub fn ensure_block(&mut self, block: Block) {
        if !self.blocks.contains_key(&block) {
            self.blocks.insert(block, MachBlock::default());
            self.block_order.push(block);
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction selector
// ---------------------------------------------------------------------------

/// AArch64 instruction selector.
///
/// Walks tMIR blocks in order, selects each instruction into one or more
/// AArch64 MachInsts, tracking value -> VReg mappings. After selection,
/// `finalize()` returns the completed MachFunction.
pub struct InstructionSelector {
    func: MachFunction,
    /// tMIR Value -> machine operand mapping.
    value_map: HashMap<Value, MachOperand>,
    /// Type of each value, tracked for selecting correct instruction width.
    value_types: HashMap<Value, Type>,
}

impl InstructionSelector {
    /// Create a new instruction selector for the given function.
    pub fn new(name: String, sig: Signature) -> Self {
        Self {
            func: MachFunction::new(name, sig),
            value_map: HashMap::new(),
            value_types: HashMap::new(),
        }
    }

    /// Require that `inst` has at least `n` arguments.
    fn require_args(inst: &Instruction, n: usize, context: &'static str) -> Result<(), ISelError> {
        if inst.args.len() < n {
            return Err(ISelError::InsufficientArgs {
                expected: n,
                actual: inst.args.len(),
                context,
            });
        }
        Ok(())
    }

    /// Require that `inst` has at least one result.
    fn require_result(inst: &Instruction, context: &'static str) -> Result<(), ISelError> {
        if inst.results.is_empty() {
            return Err(ISelError::MissingResult(context));
        }
        Ok(())
    }

    /// Allocate a fresh virtual register.
    fn new_vreg(&mut self, class: RegClass) -> VReg {
        let id = self.func.next_vreg;
        self.func.next_vreg += 1;
        VReg { id, class }
    }

    /// Record a mapping from tMIR Value to machine operand.
    fn define_value(&mut self, val: Value, operand: MachOperand, ty: Type) {
        self.value_map.insert(val, operand);
        self.value_types.insert(val, ty);
    }

    /// Look up the machine operand for a tMIR Value.
    fn use_value(&self, val: &Value) -> Result<MachOperand, ISelError> {
        self.value_map
            .get(val)
            .cloned()
            .ok_or_else(|| ISelError::UndefinedValue(*val))
    }

    /// Get the type of a tMIR Value.
    fn value_type(&self, val: &Value) -> Type {
        self.value_types
            .get(val)
            .cloned()
            .unwrap_or(Type::I64) // default to i64 if unknown
    }

    /// Determine if a type is 32-bit (uses W registers).
    fn is_32bit(ty: &Type) -> bool {
        matches!(ty, Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::F32)
    }

    // -----------------------------------------------------------------------
    // Top-level selection
    // -----------------------------------------------------------------------

    /// Select all instructions in a block.
    pub fn select_block(&mut self, block: Block, instructions: &[Instruction]) -> Result<(), ISelError> {
        self.func.ensure_block(block);
        for inst in instructions {
            self.select_instruction(inst, block)?;
        }
        Ok(())
    }

    /// Select a single tMIR instruction, emitting MachInsts into the block.
    fn select_instruction(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        match &inst.opcode {
            // Constants
            Opcode::Iconst { ty, imm } => self.select_iconst(ty.clone(), *imm, inst, block)?,
            Opcode::Fconst { ty, imm } => self.select_fconst(ty.clone(), *imm, inst, block)?,

            // Arithmetic
            // Single-arg Iadd is used by the adapter as a COPY placeholder
            // (block argument passing, phi resolution). Emit a MOV instead.
            Opcode::Iadd if inst.args.len() == 1 => self.select_copy(inst, block)?,
            Opcode::Iadd => self.select_binop(AArch64BinOp::Add, inst, block)?,
            Opcode::Isub => self.select_binop(AArch64BinOp::Sub, inst, block)?,
            Opcode::Imul => self.select_binop(AArch64BinOp::Mul, inst, block)?,
            Opcode::Sdiv => self.select_binop(AArch64BinOp::Sdiv, inst, block)?,
            Opcode::Udiv => self.select_binop(AArch64BinOp::Udiv, inst, block)?,
            Opcode::Srem => self.select_remainder(/*signed=*/true, inst, block)?,
            Opcode::Urem => self.select_remainder(/*signed=*/false, inst, block)?,

            // Unary operations
            Opcode::Ineg => self.select_int_unaryop(AArch64IntUnaryOp::Neg, inst, block)?,
            Opcode::Bnot => self.select_int_unaryop(AArch64IntUnaryOp::Mvn, inst, block)?,
            Opcode::Fneg => self.select_fp_unaryop(inst, block)?,

            // Shift operations
            Opcode::Ishl => self.select_shift(AArch64ShiftOp::Lsl, inst, block)?,
            Opcode::Ushr => self.select_shift(AArch64ShiftOp::Lsr, inst, block)?,
            Opcode::Sshr => self.select_shift(AArch64ShiftOp::Asr, inst, block)?,

            // Logical operations
            Opcode::Band => self.select_logic(AArch64LogicOp::And, inst, block)?,
            Opcode::Bor => self.select_logic(AArch64LogicOp::Orr, inst, block)?,
            Opcode::Bxor => self.select_logic(AArch64LogicOp::Eor, inst, block)?,
            Opcode::BandNot => self.select_logic(AArch64LogicOp::Bic, inst, block)?,
            Opcode::BorNot => self.select_logic(AArch64LogicOp::Orn, inst, block)?,

            // Extensions
            Opcode::Sextend { from_ty, to_ty } => {
                self.select_extend(true, from_ty, to_ty, inst, block)?;
            }
            Opcode::Uextend { from_ty, to_ty } => {
                self.select_extend(false, from_ty, to_ty, inst, block)?;
            }

            // Bitfield operations
            Opcode::ExtractBits { lsb, width } => {
                self.select_bitfield_extract(false, *lsb, *width, inst, block)?;
            }
            Opcode::SextractBits { lsb, width } => {
                self.select_bitfield_extract(true, *lsb, *width, inst, block)?;
            }
            Opcode::InsertBits { lsb, width } => {
                self.select_bitfield_insert(*lsb, *width, inst, block)?;
            }

            // Conditional select
            Opcode::Select { cond } => self.select_csel(*cond, inst, block)?,

            // Comparison
            Opcode::Icmp { cond } => self.select_cmp(*cond, inst, block)?,

            // Floating-point arithmetic
            Opcode::Fadd => self.select_fp_binop(AArch64FpBinOp::Fadd, inst, block)?,
            Opcode::Fsub => self.select_fp_binop(AArch64FpBinOp::Fsub, inst, block)?,
            Opcode::Fmul => self.select_fp_binop(AArch64FpBinOp::Fmul, inst, block)?,
            Opcode::Fdiv => self.select_fp_binop(AArch64FpBinOp::Fdiv, inst, block)?,
            Opcode::Fcmp { cond } => self.select_fcmp(*cond, inst, block)?,
            Opcode::FcvtToInt { dst_ty } => {
                self.select_fcvt_to_int(dst_ty.clone(), inst, block)?;
            }
            Opcode::FcvtFromInt { src_ty } => {
                self.select_fcvt_from_int(src_ty.clone(), inst, block)?;
            }

            // Addressing
            Opcode::GlobalRef { name } => {
                self.select_global_ref(name, inst, block)?;
            }
            Opcode::ExternRef { name } => {
                self.select_extern_ref(name, inst, block)?;
            }
            Opcode::TlsRef { name } => {
                self.select_tls_ref(name, inst, block)?;
            }
            Opcode::StackAddr { slot } => {
                self.select_stack_addr(*slot, inst, block)?;
            }

            // Control flow
            Opcode::Jump { dest } => self.select_jump(*dest, block)?,
            Opcode::Brif { then_dest, else_dest, .. } => {
                self.select_brif(inst, *then_dest, *else_dest, block)?;
            }
            Opcode::Return => self.select_return(inst, block)?,
            Opcode::Call { name } => {
                self.select_call_from_lir(name, inst, block)?;
            }
            Opcode::CallIndirect => {
                self.select_call_indirect_from_lir(inst, block)?;
            }
            Opcode::CallVariadic { name, fixed_args } => {
                self.select_variadic_call_from_lir(name, *fixed_args, inst, block)?;
            }
            Opcode::Switch { cases, default } => {
                self.select_switch(cases, *default, inst, block)?;
            }

            // Type conversions (unsigned FP <-> int)
            Opcode::FcvtToUint { dst_ty } => {
                self.select_fcvt_to_uint(dst_ty.clone(), inst, block)?;
            }
            Opcode::FcvtFromUint { src_ty } => {
                self.select_fcvt_from_uint(src_ty.clone(), inst, block)?;
            }
            // Float precision conversion
            Opcode::FPExt => self.select_fp_ext(inst, block)?,
            Opcode::FPTrunc => self.select_fp_trunc(inst, block)?,
            // Integer truncation and bitcast
            Opcode::Trunc { to_ty } => self.select_trunc(&to_ty.clone(), inst, block)?,
            Opcode::Bitcast { to_ty } => self.select_bitcast(&to_ty.clone(), inst, block)?,

            // Memory
            Opcode::Load { ty } => self.select_load(ty.clone(), inst, block)?,
            Opcode::Store => self.select_store(inst, block)?,

            // Aggregate operations
            Opcode::StructGep { struct_ty, field_index } => {
                self.select_struct_gep(struct_ty.clone(), *field_index, inst, block)?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Select integer constant materialization.
    /// Small values (0..65535): MOVZ
    /// Negative small values: MOVN
    /// Large values: MOVZ + MOVK sequence (up to 4 chunks for 64-bit)
    fn select_iconst(&mut self, ty: Type, imm: i64, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_result(inst, "Iconst")?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = &inst.results[0];

        if imm >= 0 && imm <= 0xFFFF {
            // Simple MOVZ
            let opc = if Self::is_32bit(&ty) {
                AArch64Opcode::Movz
            } else {
                AArch64Opcode::Movz
            };
            self.func.push_inst(
                block,
                MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::Imm(imm)]),
            );
        } else if imm < 0 && imm >= -0x10000 {
            // MOVN for small negative values
            let opc = if Self::is_32bit(&ty) {
                AArch64Opcode::Movn
            } else {
                AArch64Opcode::Movn
            };
            // MOVN encodes ~(imm16 << shift), so for -1 we encode MOVN Xd, #0
            let encoded = (!imm) as u64 & 0xFFFF;
            self.func.push_inst(
                block,
                MachInst::new(
                    opc,
                    vec![MachOperand::VReg(dst), MachOperand::Imm(encoded as i64)],
                ),
            );
        } else {
            // Large immediate: MOVZ + MOVK sequence
            // Start with lowest 16 bits via MOVZ
            let opc_z = if Self::is_32bit(&ty) {
                AArch64Opcode::Movz
            } else {
                AArch64Opcode::Movz
            };
            let low16 = (imm as u64) & 0xFFFF;
            self.func.push_inst(
                block,
                MachInst::new(
                    opc_z,
                    vec![MachOperand::VReg(dst), MachOperand::Imm(low16 as i64)],
                ),
            );

            // Insert remaining 16-bit chunks via MOVK
            let chunks = if Self::is_32bit(&ty) { 2 } else { 4 };
            for shift in 1..chunks {
                let chunk = ((imm as u64) >> (shift * 16)) & 0xFFFF;
                if chunk != 0 {
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::Movk,
                            vec![
                                MachOperand::VReg(dst),
                                MachOperand::Imm(chunk as i64),
                                MachOperand::Imm(shift as i64 * 16), // shift amount
                            ],
                        ),
                    );
                }
            }
        }

        self.define_value(*result, MachOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select float constant materialization.
    fn select_fconst(&mut self, ty: Type, imm: f64, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_result(inst, "Fconst")?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = &inst.results[0];

        let opc = match ty {
            Type::F32 => AArch64Opcode::FmovImm,
            Type::F64 => AArch64Opcode::FmovImm,
            _ => return Err(ISelError::UnsupportedFconstType(ty)),
        };

        // FMOV immediate encoding is limited to a small set of values.
        // For the scaffold, emit FMOV for all; legalization will handle
        // values outside the 8-bit FP immediate range via constant pool.
        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::FImm(imm)]),
        );

        self.define_value(*result, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Copy (single-arg Iadd used as COPY by adapter)
    // -----------------------------------------------------------------------

    /// Select a COPY instruction (emitted for single-arg Iadd from the adapter).
    ///
    /// The adapter uses `Iadd` with a single argument as a placeholder for
    /// register copies (block argument passing, phi resolution). We emit a
    /// MOV (register-to-register copy) instruction.
    fn select_copy(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Copy")?;
        Self::require_result(inst, "Copy")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        // If the result value already has a vreg (e.g., a block parameter
        // that was defined by an earlier predecessor's copy), reuse it.
        // This ensures all predecessors write to the same vreg for block
        // parameters, which is essential for correct register allocation
        // across loop back-edges.
        let dst = if let Some(existing) = self.value_map.get(&result_val) {
            if let MachOperand::VReg(v) = existing {
                *v
            } else {
                let class = reg_class_for_type(&ty);
                self.new_vreg(class)
            }
        } else {
            let class = reg_class_for_type(&ty);
            self.new_vreg(class)
        };

        let opc = if Self::is_32bit(&ty) {
            AArch64Opcode::MovR
        } else {
            AArch64Opcode::MovR
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    /// Internal enum for selecting the right opcode variant.
    #[allow(dead_code)]
    fn select_binop(&mut self, op: AArch64BinOp, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "BinOp")?;
        Self::require_result(inst, "BinOp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        // Determine type from LHS
        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match (op, is_32) {
            (AArch64BinOp::Add, true) => AArch64Opcode::AddRR,
            (AArch64BinOp::Add, false) => AArch64Opcode::AddRR,
            (AArch64BinOp::Sub, true) => AArch64Opcode::SubRR,
            (AArch64BinOp::Sub, false) => AArch64Opcode::SubRR,
            (AArch64BinOp::Mul, true) => AArch64Opcode::MulRR,
            (AArch64BinOp::Mul, false) => AArch64Opcode::MulRR,
            (AArch64BinOp::Sdiv, true) => AArch64Opcode::SDiv,
            (AArch64BinOp::Sdiv, false) => AArch64Opcode::SDiv,
            (AArch64BinOp::Udiv, true) => AArch64Opcode::UDiv,
            (AArch64BinOp::Udiv, false) => AArch64Opcode::UDiv,
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select integer remainder: SDIV/UDIV + MSUB.
    ///
    /// AArch64 has no hardware remainder instruction. We compute:
    ///   tmp = a / b         (SDIV or UDIV)
    ///   result = a - tmp*b  (MSUB: Rd = Ra - Rn * Rm)
    ///
    /// MSUB operand order: MSUB Rd, Rn, Rm, Ra  =>  Rd = Ra - Rn * Rm
    /// So: MSUB result, tmp, b, a  =>  result = a - tmp * b
    fn select_remainder(&mut self, signed: bool, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Remainder")?;
        Self::require_result(inst, "Remainder")?;

        let lhs_val = inst.args[0]; // dividend (a)
        let rhs_val = inst.args[1]; // divisor (b)
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // Step 1: tmp = a / b
        let tmp = self.new_vreg(class);
        let div_opc = match (signed, is_32) {
            (true, true) => AArch64Opcode::SDiv,
            (true, false) => AArch64Opcode::SDiv,
            (false, true) => AArch64Opcode::UDiv,
            (false, false) => AArch64Opcode::UDiv,
        };
        self.func.push_inst(
            block,
            MachInst::new(div_opc, vec![MachOperand::VReg(tmp), lhs.clone(), rhs.clone()]),
        );

        // Step 2: result = a - tmp * b  (MSUB Rd, Rn, Rm, Ra)
        // Operands: [dst, Rn=tmp, Rm=b, Ra=a]
        let dst = self.new_vreg(class);
        let msub_opc = if is_32 {
            AArch64Opcode::Msub
        } else {
            AArch64Opcode::Msub
        };
        self.func.push_inst(
            block,
            MachInst::new(
                msub_opc,
                vec![
                    MachOperand::VReg(dst),
                    MachOperand::VReg(tmp),
                    rhs,
                    lhs,
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unary operations
    // -----------------------------------------------------------------------

    /// Select integer unary operation: NEG, MVN (bitwise NOT).
    ///
    /// AArch64 lowering:
    ///   `Ineg(x)` → `SUB Xd, XZR, Xn` (alias: NEG)
    ///   `Bnot(x)` → `ORN Xd, XZR, Xn` (alias: MVN)
    fn select_int_unaryop(&mut self, op: AArch64IntUnaryOp, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "IntUnaryOp")?;
        Self::require_result(inst, "IntUnaryOp")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        let opc = match (op, is_32) {
            (AArch64IntUnaryOp::Neg, true) => AArch64Opcode::Neg,
            (AArch64IntUnaryOp::Neg, false) => AArch64Opcode::Neg,
            (AArch64IntUnaryOp::Mvn, true) => AArch64Opcode::OrnRR,
            (AArch64IntUnaryOp::Mvn, false) => AArch64Opcode::OrnRR,
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select floating-point unary negate: FNEG Sd/Dd, Sn/Dn.
    fn select_fp_unaryop(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "FpUnaryOp")?;
        Self::require_result(inst, "FpUnaryOp")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_f32 = matches!(ty, Type::F32);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        let opc = if is_f32 {
            AArch64Opcode::FnegRR
        } else {
            AArch64Opcode::FnegRR
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    /// Select integer comparison: CMP + CSET.
    ///
    /// tMIR `Icmp(cond, lhs, rhs) -> bool_result` becomes:
    ///   CMP Wn/Xn, Wm/Xm        (sets NZCV flags)
    ///   CSET Wd, <cond_code>     (materializes flag into register)
    fn select_cmp(&mut self, cond: IntCC, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Icmp")?;
        Self::require_result(inst, "Icmp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // CMP (subtract setting flags, discard result)
        let cmp_opc = if is_32 {
            AArch64Opcode::CmpRR
        } else {
            AArch64Opcode::CmpRR
        };
        self.func.push_inst(
            block,
            MachInst::new(cmp_opc, vec![lhs, rhs]),
        );

        // CSET: materialize condition code into a register.
        // We use Gpr64 for the destination to avoid register aliasing issues
        // with the linear scan allocator, which treats W<n> and X<n> as
        // independent physical registers despite them sharing hardware.
        // CSET writes a 32-bit result (0 or 1) that zero-extends to 64 bits,
        // so using an X register is semantically correct.
        let cc = AArch64CC::from_intcc(cond);
        let dst = self.new_vreg(RegClass::Gpr64);
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::CSet,
                vec![MachOperand::VReg(dst), MachOperand::CondCode(cc)],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::B1);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    /// Select unconditional jump.
    fn select_jump(&mut self, dest: Block, block: Block) -> Result<(), ISelError> {
        self.func.push_inst(
            block,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(dest)]),
        );
        self.func
            .blocks
            .entry(block)
            .or_default()
            .successors
            .push(dest);
        Ok(())
    }

    /// Select conditional branch: tMIR `Brif(cond, then, else)`.
    ///
    /// The condition value should already be a comparison result. We emit:
    ///   CMP Wcond, #0          (test boolean value)
    ///   B.NE then_block        (branch if true)
    ///   B else_block           (fallthrough to else)
    fn select_brif(
        &mut self,
        inst: &Instruction,
        then_dest: Block,
        else_dest: Block,
        block: Block,
    ) -> Result<(), ISelError> {
        // The condition is the first argument
        let cond_val = inst.args[0];
        let cond_op = self.use_value(&cond_val)?;

        // CMP cond, #0 to set NZCV from boolean.
        // Use 64-bit CMP to match the Gpr64 class of the CSET result.
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::CmpRI,
                vec![cond_op, MachOperand::Imm(0)],
            ),
        );

        // B.NE then_block (condition was nonzero = true)
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::BCond,
                vec![
                    MachOperand::CondCode(AArch64CC::NE),
                    MachOperand::Block(then_dest),
                ],
            ),
        );

        // B else_block (unconditional fallthrough)
        self.func.push_inst(
            block,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(else_dest)]),
        );

        let mblock = self.func.blocks.entry(block).or_default();
        mblock.successors.push(then_dest);
        mblock.successors.push(else_dest);
        Ok(())
    }

    /// Select return. Moves return values into ABI-specified physical registers
    /// (X0/V0 etc.) and emits RET.
    fn select_return(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        // Classify return types to know which physical registers to use
        let ret_types: Vec<Type> = inst
            .args
            .iter()
            .map(|v| self.value_type(v))
            .collect();
        let ret_locs = AppleAArch64ABI::classify_returns(&ret_types);

        // Move each return value into its designated physical register
        for (i, (val, loc)) in inst.args.iter().zip(ret_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = ret_types[i].clone();
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    // COPY pseudo: move value to physical register
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            opc,
                            vec![MachOperand::PReg(*preg), src],
                        ),
                    );
                }
                ArgLocation::Indirect { .. } => {
                    // Aggregate return: dispatch to aggregate return lowering.
                    // src is a pointer to the aggregate in memory.
                    let agg_ty = ret_types[i].clone();
                    self.select_aggregate_return(src, &agg_ty, block)?;
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }

        // Emit RET (branches to LR)
        self.func.push_inst(
            block,
            MachInst::new(AArch64Opcode::Ret, vec![MachOperand::PReg(gpr::LR)]),
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Call
    // -----------------------------------------------------------------------

    /// Select a function call.
    ///
    /// 1. Move arguments to physical registers (X0-X7, V0-V7) or stack
    /// 2. Emit BL (direct call)
    /// 3. Copy results from physical return registers to vregs
    pub fn select_call(
        &mut self,
        callee_name: &str,
        arg_vals: &[Value],
        result_vals: &[Value],
        result_types: &[Type],
        block: Block,
    ) -> Result<(), ISelError> {
        // Classify argument locations
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params(&arg_types);

        // Move arguments to ABI locations
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = arg_types[i].clone();

            // Dispatch aggregate arguments to specialized handler
            if ty.is_aggregate() {
                self.select_aggregate_arg(src, &ty, loc, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // STR to [SP + offset]
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::StrRI
                    } else {
                        AArch64Opcode::StrRI
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            opc,
                            vec![src, MachOperand::PReg(SP), MachOperand::Imm(*offset)],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Non-aggregate indirect (I128): pass pointer in register
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::PReg(*ptr_reg), src],
                        ),
                    );
                }
            }
        }

        // Emit BL (direct call) with the callee symbol for relocation.
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol(callee_name.to_string())],
            ),
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::PReg(*preg)]),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Aggregate returned via sret pointer (X8).
                    // The caller passed X8 as the sret buffer; after the call
                    // the data is at [X8]. Define the result as the sret
                    // pointer itself so downstream code can access fields.
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::VReg(dst), MachOperand::PReg(*ptr_reg)],
                        ),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }
        Ok(())
    }

    /// Select a call instruction from the LIR `Opcode::Call { name }`.
    ///
    /// This bridges the adapter's `Call` opcode to the existing `select_call`
    /// method by extracting the callee name, argument values, result values,
    /// and result types from the LIR instruction.
    fn select_call_from_lir(&mut self, name: &str, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        let result_types: Vec<Type> = inst
            .results
            .iter()
            .map(|v| self.value_type(v))
            .collect();
        self.select_call(name, &inst.args, &inst.results, &result_types, block)
    }

    // -----------------------------------------------------------------------
    // Variadic calls (Apple AArch64 ABI)
    // -----------------------------------------------------------------------

    /// Select a variadic function call (e.g., printf, NSLog).
    ///
    /// Apple AArch64 ABI variadic convention:
    /// - Fixed args (0..fixed_count) use normal register/stack classification
    /// - ALL variadic args (fixed_count..) go on the stack, 8-byte aligned
    ///
    /// This differs from standard AAPCS64, where variadic args can use registers.
    pub fn select_variadic_call(
        &mut self,
        callee_name: &str,
        fixed_count: usize,
        arg_vals: &[Value],
        result_vals: &[Value],
        result_types: &[Type],
        block: Block,
    ) -> Result<(), ISelError> {
        // Classify argument locations using variadic ABI rules
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params_variadic(fixed_count, &arg_types);

        // Move arguments to ABI locations
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = arg_types[i].clone();

            // Dispatch aggregate arguments to specialized handler
            if ty.is_aggregate() {
                self.select_aggregate_arg(src, &ty, loc, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // STR to [SP + offset] — used for both fixed overflow and
                    // variadic arguments (Apple ABI puts all varargs on stack).
                    let opc = if matches!(ty, Type::F32) {
                        AArch64Opcode::StrRI
                    } else if matches!(ty, Type::F64) {
                        AArch64Opcode::StrRI
                    } else if Self::is_32bit(&ty) {
                        AArch64Opcode::StrRI
                    } else {
                        AArch64Opcode::StrRI
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            opc,
                            vec![src, MachOperand::PReg(SP), MachOperand::Imm(*offset)],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Non-aggregate indirect (I128): pass pointer in register
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::PReg(*ptr_reg), src],
                        ),
                    );
                }
            }
        }

        // Emit BL (direct call) with the callee symbol for relocation.
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol(callee_name.to_string())],
            ),
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::PReg(*preg)]),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg } => {
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::VReg(dst), MachOperand::PReg(*ptr_reg)],
                        ),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }
        Ok(())
    }

    /// Select a variadic call from LIR `Opcode::CallVariadic`.
    fn select_variadic_call_from_lir(
        &mut self,
        name: &str,
        fixed_args: u32,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        let result_types: Vec<Type> = inst
            .results
            .iter()
            .map(|v| self.value_type(v))
            .collect();
        self.select_variadic_call(
            name,
            fixed_args as usize,
            &inst.args,
            &inst.results,
            &result_types,
            block,
        )
    }

    // -----------------------------------------------------------------------
    // Indirect call (CallIndirect)
    // -----------------------------------------------------------------------

    /// Select an indirect function call (BLR).
    ///
    /// `CallIndirect` is like `Call` but the target is a register (function
    /// pointer) instead of a symbol name.
    ///
    /// args[0] = function pointer (I64)
    /// args[1..] = call arguments (classified per ABI)
    /// results = return values
    ///
    /// Lowering:
    /// 1. Move call arguments to ABI registers/stack (same as direct call)
    /// 2. MOV function pointer to a scratch register (X16, intra-procedure-call
    ///    scratch per AArch64 ABI)
    /// 3. BLR scratch_reg
    /// 4. Copy return values from ABI registers to vregs
    pub fn select_call_indirect(
        &mut self,
        fn_ptr_val: &Value,
        arg_vals: &[Value],
        result_vals: &[Value],
        result_types: &[Type],
        block: Block,
    ) -> Result<(), ISelError> {
        // Classify argument locations (excluding the function pointer)
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params(&arg_types);

        // Move arguments to ABI locations (same as direct call)
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = arg_types[i].clone();

            if ty.is_aggregate() {
                self.select_aggregate_arg(src, &ty, loc, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::StrRI
                    } else {
                        AArch64Opcode::StrRI
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            opc,
                            vec![src, MachOperand::PReg(SP), MachOperand::Imm(*offset)],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::PReg(*ptr_reg), src],
                        ),
                    );
                }
            }
        }

        // Move function pointer to X16 (intra-procedure-call scratch register).
        // X16/X17 are designated as IP0/IP1 in AArch64 ABI — used by linker
        // veneers and safe to clobber across calls.
        let fn_ptr = self.use_value(fn_ptr_val)?;
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::MovR,
                vec![MachOperand::PReg(gpr::X16), fn_ptr],
            ),
        );

        // Emit BLR X16 (indirect call via register)
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Blr,
                vec![MachOperand::PReg(gpr::X16)],
            ),
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MovR
                    } else {
                        AArch64Opcode::MovR
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::PReg(*preg)]),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg } => {
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::VReg(dst), MachOperand::PReg(*ptr_reg)],
                        ),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }
        Ok(())
    }

    /// Select a `CallIndirect` from LIR.
    ///
    /// `inst.args[0]` is the function pointer, `inst.args[1..]` are call args.
    fn select_call_indirect_from_lir(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "CallIndirect")?;

        let fn_ptr_val = inst.args[0];
        let call_args = &inst.args[1..];
        let result_types: Vec<Type> = inst
            .results
            .iter()
            .map(|v| self.value_type(v))
            .collect();
        self.select_call_indirect(
            &fn_ptr_val,
            call_args,
            &inst.results,
            &result_types,
            block,
        )
    }

    // -----------------------------------------------------------------------
    // Switch (cascading CMP+B.EQ chain)
    // -----------------------------------------------------------------------

    /// Select a switch statement as a cascading CMP+B.EQ chain.
    ///
    /// For each case `(value, target_block)`:
    ///   CMP selector, #value
    ///   B.EQ target_block
    /// After all cases:
    ///   B default_block
    ///
    /// This is the simplest lowering strategy (linear scan). Jump tables
    /// can be added later for dense switch ranges.
    fn select_switch(
        &mut self,
        cases: &[(i64, Block)],
        default: Block,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Switch")?;

        let selector_val = inst.args[0];
        let selector = self.use_value(&selector_val)?;
        let sel_ty = self.value_type(&selector_val);
        let is_32 = Self::is_32bit(&sel_ty);

        // For each case: CMP selector, #case_val then B.EQ target
        for (case_val, target) in cases {
            // Materialize the case value as an immediate comparison.
            // AArch64 CMP immediate (CMPri) supports 12-bit unsigned values.
            // For values outside that range, materialize into a register first.
            let case_fits_imm12 = *case_val >= 0 && *case_val <= 0xFFF;

            if case_fits_imm12 {
                let cmp_opc = if is_32 {
                    AArch64Opcode::CmpRI
                } else {
                    AArch64Opcode::CmpRI
                };
                self.func.push_inst(
                    block,
                    MachInst::new(
                        cmp_opc,
                        vec![selector.clone(), MachOperand::Imm(*case_val)],
                    ),
                );
            } else {
                // Materialize case value into a register, then CMP reg, reg.
                let class = if is_32 { RegClass::Gpr32 } else { RegClass::Gpr64 };
                let case_vreg = self.new_vreg(class);
                let mov_opc = if is_32 {
                    AArch64Opcode::Movz
                } else {
                    AArch64Opcode::Movz
                };
                // For simplicity, use MOVZi for non-negative values that fit
                // in 16 bits, and the full materialization sequence for larger.
                // TODO: Full 64-bit materialization for very large constants.
                self.func.push_inst(
                    block,
                    MachInst::new(
                        mov_opc,
                        vec![
                            MachOperand::VReg(case_vreg),
                            MachOperand::Imm(*case_val),
                        ],
                    ),
                );

                let cmp_opc = if is_32 {
                    AArch64Opcode::CmpRR
                } else {
                    AArch64Opcode::CmpRR
                };
                self.func.push_inst(
                    block,
                    MachInst::new(
                        cmp_opc,
                        vec![selector.clone(), MachOperand::VReg(case_vreg)],
                    ),
                );
            }

            // B.EQ target_block
            self.func.push_inst(
                block,
                MachInst::new(
                    AArch64Opcode::BCond,
                    vec![
                        MachOperand::CondCode(AArch64CC::EQ),
                        MachOperand::Block(*target),
                    ],
                ),
            );

            // Record successor
            self.func
                .blocks
                .entry(block)
                .or_default()
                .successors
                .push(*target);
        }

        // Fall through to default block
        self.func.push_inst(
            block,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(default)]),
        );
        self.func
            .blocks
            .entry(block)
            .or_default()
            .successors
            .push(default);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Load / Store
    // -----------------------------------------------------------------------

    /// Select a memory load.
    fn select_load(&mut self, ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Load")?;
        Self::require_result(inst, "Load")?;

        let addr_val = inst.args[0];
        let result_val = inst.results[0];
        let addr = self.use_value(&addr_val)?;

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        // Select opcode based on load type. For I8/I16, we default to
        // zero-extending loads (LDRB/LDRH). Sign-extending loads (LDRSB/LDRSH)
        // would be selected when the load feeds into an Sextend, but that
        // optimization is deferred to late combines (Phase 2).
        let opc = match ty {
            Type::I8 => AArch64Opcode::LdrbRI,
            Type::I16 => AArch64Opcode::LdrhRI,
            Type::I32 => AArch64Opcode::LdrRI,
            Type::I64 => AArch64Opcode::LdrRI,
            Type::F32 => AArch64Opcode::LdrRI,
            Type::F64 => AArch64Opcode::LdrRI,
            _ => AArch64Opcode::LdrRI,
        };

        self.func.push_inst(
            block,
            MachInst::new(
                opc,
                vec![MachOperand::VReg(dst), addr, MachOperand::Imm(0)],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select a memory store.
    fn select_store(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Store")?;

        let value_val = inst.args[0];
        let addr_val = inst.args[1];

        let src = self.use_value(&value_val)?;
        let addr = self.use_value(&addr_val)?;
        let ty = self.value_type(&value_val);

        let opc = match ty {
            Type::I8 => AArch64Opcode::StrbRI,
            Type::I16 => AArch64Opcode::StrhRI,
            Type::I32 => AArch64Opcode::StrRI,
            Type::I64 => AArch64Opcode::StrRI,
            Type::F32 => AArch64Opcode::StrRI,
            Type::F64 => AArch64Opcode::StrRI,
            _ => AArch64Opcode::StrRI,
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![src, addr, MachOperand::Imm(0)]),
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Aggregate operations
    // -----------------------------------------------------------------------

    /// Select a struct field address computation (GEP-like).
    ///
    /// Given a pointer to a struct and a field index, compute:
    ///   result = base + offset_of(struct_ty, field_index)
    ///
    /// For zero offset (field 0 with no padding), emits a MOV.
    /// Otherwise emits ADD Xd, base, #offset.
    fn select_struct_gep(
        &mut self,
        struct_ty: Type,
        field_index: u32,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "StructGep")?;
        Self::require_result(inst, "StructGep")?;

        // Validate the struct type
        let field_count = match &struct_ty {
            Type::Struct(fields) => fields.len(),
            _ => return Err(ISelError::StructGepNonStruct(struct_ty)),
        };
        if field_index as usize >= field_count {
            return Err(ISelError::StructGepOutOfRange {
                index: field_index,
                field_count,
            });
        }

        let base_val = inst.args[0];
        let result_val = inst.results[0];
        let base = self.use_value(&base_val)?;

        let offset = struct_ty.offset_of(field_index as usize).unwrap_or(0);

        let dst = self.new_vreg(RegClass::Gpr64);

        if offset == 0 {
            // Field is at base address, just move the pointer
            self.func.push_inst(
                block,
                MachInst::new(
                    AArch64Opcode::MovR,
                    vec![MachOperand::VReg(dst), base],
                ),
            );
        } else {
            // ADD Xd, base, #offset
            self.func.push_inst(
                block,
                MachInst::new(
                    AArch64Opcode::AddRI,
                    vec![
                        MachOperand::VReg(dst),
                        base,
                        MachOperand::Imm(offset as i64),
                    ],
                ),
            );
        }

        // Result is a pointer (I64 on AArch64)
        self.define_value(result_val, MachOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select aggregate return value lowering.
    ///
    /// Apple AArch64 ABI:
    /// - Small (<=8 bytes): pack fields into X0
    /// - Medium (<=16 bytes): pack fields into X0 + X1
    /// - Large (>16 bytes): store to memory pointed to by X8 (sret)
    ///
    /// This is called from `select_return` when it detects an aggregate type.
    fn select_aggregate_return(
        &mut self,
        src: MachOperand,
        agg_ty: &Type,
        block: Block,
    ) -> Result<(), ISelError> {
        let size = agg_ty.bytes();

        if size <= 8 {
            // Small aggregate: load entire struct as a single X0 value.
            // src is a pointer to the aggregate in memory.
            // Emit: LDR X0, [src]
            self.func.push_inst(
                block,
                MachInst::new(
                    AArch64Opcode::LdrRI,
                    vec![MachOperand::PReg(gpr::X0), src, MachOperand::Imm(0)],
                ),
            );
        } else if size <= 16 {
            // Medium aggregate: load into X0 + X1.
            // First 8 bytes -> X0
            self.func.push_inst(
                block,
                MachInst::new(
                    AArch64Opcode::LdrRI,
                    vec![
                        MachOperand::PReg(gpr::X0),
                        src.clone(),
                        MachOperand::Imm(0),
                    ],
                ),
            );
            // Next bytes -> X1
            self.func.push_inst(
                block,
                MachInst::new(
                    AArch64Opcode::LdrRI,
                    vec![MachOperand::PReg(gpr::X1), src, MachOperand::Imm(8)],
                ),
            );
        } else {
            // Large aggregate: store to [X8] (sret pointer).
            // Emit a sequence of stores from the source to the sret buffer.
            // For now, emit word-at-a-time copies. A real implementation would
            // use memcpy or unrolled LDP/STP for large sizes.
            let mut offset: u32 = 0;
            while offset + 8 <= size {
                // Load 8 bytes from source
                let tmp = self.new_vreg(RegClass::Gpr64);
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::LdrRI,
                        vec![
                            MachOperand::VReg(tmp),
                            src.clone(),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                // Store 8 bytes to sret destination
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::StrRI,
                        vec![
                            MachOperand::VReg(tmp),
                            MachOperand::PReg(gpr::X8),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                offset += 8;
            }
            // Handle trailing 4-byte chunk
            if offset + 4 <= size {
                let tmp = self.new_vreg(RegClass::Gpr32);
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::LdrRI,
                        vec![
                            MachOperand::VReg(tmp),
                            src.clone(),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::StrRI,
                        vec![
                            MachOperand::VReg(tmp),
                            MachOperand::PReg(gpr::X8),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                offset += 4;
            }
            // Handle trailing 2-byte chunk
            if offset + 2 <= size {
                let tmp = self.new_vreg(RegClass::Gpr32);
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::LdrhRI,
                        vec![
                            MachOperand::VReg(tmp),
                            src.clone(),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::StrhRI,
                        vec![
                            MachOperand::VReg(tmp),
                            MachOperand::PReg(gpr::X8),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                offset += 2;
            }
            // Handle trailing 1-byte chunk
            if offset < size {
                let tmp = self.new_vreg(RegClass::Gpr32);
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::LdrbRI,
                        vec![
                            MachOperand::VReg(tmp),
                            src.clone(),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::StrbRI,
                        vec![
                            MachOperand::VReg(tmp),
                            MachOperand::PReg(gpr::X8),
                            MachOperand::Imm(offset as i64),
                        ],
                    ),
                );
            }
        }

        Ok(())
    }

    /// Select aggregate argument passing for a call.
    ///
    /// Apple AArch64 ABI:
    /// - Small (<=8 bytes): load into a single GPR
    /// - Medium (<=16 bytes): load into two consecutive GPRs
    /// - Large (>16 bytes): allocate stack space, store aggregate, pass pointer
    ///
    /// `src` is the pointer to the aggregate in memory.
    /// `preg` is the physical register assigned by the ABI classifier.
    fn select_aggregate_arg(
        &mut self,
        src: MachOperand,
        agg_ty: &Type,
        loc: &ArgLocation,
        block: Block,
    ) -> Result<(), ISelError> {
        let size = agg_ty.bytes();

        match loc {
            ArgLocation::Reg(preg) => {
                // Small aggregate (<=8 bytes): load as single value into register
                self.func.push_inst(
                    block,
                    MachInst::new(
                        AArch64Opcode::LdrRI,
                        vec![MachOperand::PReg(*preg), src, MachOperand::Imm(0)],
                    ),
                );
            }
            ArgLocation::Indirect { ptr_reg } => {
                if size <= 16 {
                    // Medium aggregate passed as register pair.
                    // The ABI classifier returns Indirect for medium aggregates,
                    // meaning the first register of the pair. Load first 8 bytes.
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                MachOperand::PReg(*ptr_reg),
                                src.clone(),
                                MachOperand::Imm(0),
                            ],
                        ),
                    );
                    // Load next bytes into the following register.
                    // ptr_reg is XN, next register is X(N+1).
                    let next_reg = PReg::new(ptr_reg.hw_enc() as u16 + 1);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                MachOperand::PReg(next_reg),
                                src,
                                MachOperand::Imm(8),
                            ],
                        ),
                    );
                } else {
                    // Large aggregate: pass pointer to the aggregate.
                    // The caller has already placed the aggregate in memory;
                    // just pass the pointer in the designated register.
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::PReg(*ptr_reg), src],
                        ),
                    );
                }
            }
            ArgLocation::Stack { offset, size: slot_size } => {
                // Aggregate on stack: store field-by-field.
                let mut byte_offset: u32 = 0;
                while byte_offset + 8 <= *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr64);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                MachOperand::VReg(tmp),
                                src.clone(),
                                MachOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::StrRI,
                            vec![
                                MachOperand::VReg(tmp),
                                MachOperand::PReg(SP),
                                MachOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                    byte_offset += 8;
                }
                if byte_offset + 4 <= *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr32);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                MachOperand::VReg(tmp),
                                src.clone(),
                                MachOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::StrRI,
                            vec![
                                MachOperand::VReg(tmp),
                                MachOperand::PReg(SP),
                                MachOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                    byte_offset += 4;
                }
                // Handle trailing 2-byte chunk
                if byte_offset + 2 <= *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr32);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::LdrhRI,
                            vec![
                                MachOperand::VReg(tmp),
                                src.clone(),
                                MachOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::StrhRI,
                            vec![
                                MachOperand::VReg(tmp),
                                MachOperand::PReg(SP),
                                MachOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                    byte_offset += 2;
                }
                // Handle trailing 1-byte chunk
                if byte_offset < *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr32);
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::LdrbRI,
                            vec![
                                MachOperand::VReg(tmp),
                                src.clone(),
                                MachOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::StrbRI,
                            vec![
                                MachOperand::VReg(tmp),
                                MachOperand::PReg(SP),
                                MachOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Function entry: lower formal arguments
    // -----------------------------------------------------------------------

    /// Lower formal arguments at function entry.
    ///
    /// For each parameter, emit a COPY from the physical register (or stack
    /// location) designated by the ABI to a fresh virtual register. This
    /// establishes the initial value_map for the function body.
    pub fn lower_formal_arguments(&mut self, sig: &Signature, entry_block: Block) -> Result<(), ISelError> {
        self.func.ensure_block(entry_block);

        let param_locs = AppleAArch64ABI::classify_params(&sig.params);

        // We need Value ids for the formal arguments. By convention, formal
        // args are Value(0), Value(1), ..., Value(n-1).
        for (i, (ty, loc)) in sig.params.iter().zip(param_locs.iter()).enumerate() {
            let val = Value(i as u32);
            let class = reg_class_for_type(ty);
            let vreg = self.new_vreg(class);

            match loc {
                ArgLocation::Reg(preg) => {
                    // COPY from physical to virtual register
                    self.func.push_inst(
                        entry_block,
                        MachInst::new(
                            AArch64Opcode::Copy,
                            vec![MachOperand::VReg(vreg), MachOperand::PReg(*preg)],
                        ),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // Load from stack: LDR vreg, [SP, #offset]
                    let opc = if Self::is_32bit(ty) {
                        AArch64Opcode::LdrRI
                    } else {
                        AArch64Opcode::LdrRI
                    };
                    // SP is register 31 in the encoding
                    self.func.push_inst(
                        entry_block,
                        MachInst::new(
                            opc,
                            vec![
                                MachOperand::VReg(vreg),
                                MachOperand::PReg(SP),
                                MachOperand::Imm(*offset),
                            ],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Large aggregate: load pointer from register, then load data
                    // For scaffold: just copy the pointer register
                    self.func.push_inst(
                        entry_block,
                        MachInst::new(
                            AArch64Opcode::Copy,
                            vec![MachOperand::VReg(vreg), MachOperand::PReg(*ptr_reg)],
                        ),
                    );
                }
            }

            self.define_value(val, MachOperand::VReg(vreg), ty.clone());
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shift operations
    // -----------------------------------------------------------------------

    /// Select shift operation: LSL, LSR, ASR.
    ///
    /// If the shift amount is a constant (tracked as an Iconst in the value
    /// map), we emit the immediate form; otherwise, the register form.
    fn select_shift(&mut self, op: AArch64ShiftOp, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Shift")?;
        Self::require_result(inst, "Shift")?;

        let src_val = inst.args[0];
        let amt_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;
        let amt = self.use_value(&amt_val)?;

        // Check if shift amount is an immediate
        let is_imm = matches!(amt, MachOperand::Imm(_));

        if is_imm {
            // Immediate shift form
            let opc = match (op, is_32) {
                (AArch64ShiftOp::Lsl, true) => AArch64Opcode::LslRI,
                (AArch64ShiftOp::Lsl, false) => AArch64Opcode::LslRI,
                (AArch64ShiftOp::Lsr, true) => AArch64Opcode::LsrRI,
                (AArch64ShiftOp::Lsr, false) => AArch64Opcode::LsrRI,
                (AArch64ShiftOp::Asr, true) => AArch64Opcode::AsrRI,
                (AArch64ShiftOp::Asr, false) => AArch64Opcode::AsrRI,
            };
            self.func.push_inst(
                block,
                MachInst::new(opc, vec![MachOperand::VReg(dst), src, amt]),
            );
        } else {
            // Register shift form
            let opc = match (op, is_32) {
                (AArch64ShiftOp::Lsl, true) => AArch64Opcode::LslRR,
                (AArch64ShiftOp::Lsl, false) => AArch64Opcode::LslRR,
                (AArch64ShiftOp::Lsr, true) => AArch64Opcode::LsrRR,
                (AArch64ShiftOp::Lsr, false) => AArch64Opcode::LsrRR,
                (AArch64ShiftOp::Asr, true) => AArch64Opcode::AsrRR,
                (AArch64ShiftOp::Asr, false) => AArch64Opcode::AsrRR,
            };
            self.func.push_inst(
                block,
                MachInst::new(opc, vec![MachOperand::VReg(dst), src, amt]),
            );
        }

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Logical operations
    // -----------------------------------------------------------------------

    /// Select logical operation: AND, ORR, EOR, BIC, ORN.
    fn select_logic(&mut self, op: AArch64LogicOp, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "LogicOp")?;
        Self::require_result(inst, "LogicOp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match (op, is_32) {
            (AArch64LogicOp::And, true) => AArch64Opcode::AndRR,
            (AArch64LogicOp::And, false) => AArch64Opcode::AndRR,
            (AArch64LogicOp::Orr, true) => AArch64Opcode::OrrRR,
            (AArch64LogicOp::Orr, false) => AArch64Opcode::OrrRR,
            (AArch64LogicOp::Eor, true) => AArch64Opcode::EorRR,
            (AArch64LogicOp::Eor, false) => AArch64Opcode::EorRR,
            (AArch64LogicOp::Bic, true) => AArch64Opcode::BicRR,
            (AArch64LogicOp::Bic, false) => AArch64Opcode::BicRR,
            (AArch64LogicOp::Orn, true) => AArch64Opcode::OrnRR,
            (AArch64LogicOp::Orn, false) => AArch64Opcode::OrnRR,
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Extension operations
    // -----------------------------------------------------------------------

    /// Select zero/sign extension.
    ///
    /// Maps tMIR Sextend/Uextend to the appropriate AArch64 extension
    /// instruction. The AArch64 extension instructions are actually aliases
    /// of SBFM (sign) or AND (zero) with specific bit patterns.
    fn select_extend(
        &mut self,
        signed: bool,
        from_ty: &Type,
        to_ty: &Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Extend")?;
        Self::require_result(inst, "Extend")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);
        let src = self.use_value(&src_val)?;

        let to_64 = !Self::is_32bit(to_ty);
        let to_ty_owned = to_ty.clone();

        let opc = if signed {
            match (from_ty, to_64) {
                (Type::I8, false) => AArch64Opcode::Sxtb,
                (Type::I8, true) => AArch64Opcode::Sxtb,
                (Type::I16, false) => AArch64Opcode::Sxth,
                (Type::I16, true) => AArch64Opcode::Sxth,
                (Type::I32, true) => AArch64Opcode::Sxtw,
                _ => {
                    // Same-width or unsupported: just copy
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::Copy,
                            vec![MachOperand::VReg(dst), src],
                        ),
                    );
                    self.define_value(result_val, MachOperand::VReg(dst), to_ty_owned);
                    return Ok(());
                }
            }
        } else {
            // Unsigned extension
            match from_ty {
                Type::I8 => AArch64Opcode::Uxtw,
                Type::I16 => AArch64Opcode::Uxtw,
                Type::I32 if to_64 => {
                    // UXTW: zero-extend W to X. On AArch64, writing a W register
                    // implicitly zero-extends to X, so a MOV Wd, Wn suffices.
                    // But we need an explicit instruction for tracking.
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MovR,
                            vec![MachOperand::VReg(dst), src],
                        ),
                    );
                    self.define_value(result_val, MachOperand::VReg(dst), to_ty_owned);
                    return Ok(());
                }
                _ => {
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::Copy,
                            vec![MachOperand::VReg(dst), src],
                        ),
                    );
                    self.define_value(result_val, MachOperand::VReg(dst), to_ty_owned);
                    return Ok(());
                }
            }
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );
        self.define_value(result_val, MachOperand::VReg(dst), to_ty_owned);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bitfield operations
    // -----------------------------------------------------------------------

    /// Select unsigned or signed bitfield extract.
    ///
    /// UBFM Wd, Wn, #immr, #imms  -- unsigned bitfield extract
    /// SBFM Wd, Wn, #immr, #imms  -- signed bitfield extract
    ///
    /// For extract at position `lsb` with `width` bits:
    ///   immr = lsb, imms = lsb + width - 1
    fn select_bitfield_extract(
        &mut self,
        signed: bool,
        lsb: u8,
        width: u8,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "BitfieldExtract")?;
        Self::require_result(inst, "BitfieldExtract")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let src = self.use_value(&src_val)?;

        let immr = lsb as i64;
        let imms = (lsb + width - 1) as i64;

        let opc = match (signed, is_32) {
            (false, true) => AArch64Opcode::Ubfm,
            (false, false) => AArch64Opcode::Ubfm,
            (true, true) => AArch64Opcode::Sbfm,
            (true, false) => AArch64Opcode::Sbfm,
        };

        self.func.push_inst(
            block,
            MachInst::new(
                opc,
                vec![
                    MachOperand::VReg(dst),
                    src,
                    MachOperand::Imm(immr),
                    MachOperand::Imm(imms),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select bitfield insert.
    ///
    /// BFM Wd, Wn, #immr, #imms -- insert `width` bits from Wn starting
    /// at bit `lsb` into Wd (which is also read).
    fn select_bitfield_insert(
        &mut self,
        lsb: u8,
        width: u8,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "BitfieldInsert")?;
        Self::require_result(inst, "BitfieldInsert")?;

        let dst_val = inst.args[0]; // destination (also source of unmodified bits)
        let src_val = inst.args[1]; // source of bits to insert

        let ty = self.value_type(&dst_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let result_val = inst.results[0];

        // BFM reads and writes the destination register. We model this by
        // first copying dst to a fresh vreg, then BFM modifying it in place.
        let result = self.new_vreg(class);
        let dst_op = self.use_value(&dst_val)?;
        let src_op = self.use_value(&src_val)?;

        // Copy dst to result (BFM operates on Wd as both read and write)
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Copy,
                vec![MachOperand::VReg(result), dst_op],
            ),
        );

        // BFM encoding for insert at position lsb with width:
        //   immr = (reg_size - lsb) % reg_size
        //   imms = width - 1
        let reg_size = if is_32 { 32 } else { 64 };
        let immr = ((reg_size - lsb as i64) % reg_size) as i64;
        let imms = (width - 1) as i64;

        let opc = if is_32 {
            AArch64Opcode::Bfm
        } else {
            AArch64Opcode::Bfm
        };

        self.func.push_inst(
            block,
            MachInst::new(
                opc,
                vec![
                    MachOperand::VReg(result),
                    src_op,
                    MachOperand::Imm(immr),
                    MachOperand::Imm(imms),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(result), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Conditional select
    // -----------------------------------------------------------------------

    /// Select conditional select (CSEL).
    ///
    /// tMIR: Select { cond } (cc_val, true_val, false_val) -> result
    /// The cc_val is a comparison result (must have set NZCV flags first).
    /// We emit: CMP cc_val, #0; CSEL Wd/Xd, Wn/Xn, Wm/Xm, NE
    fn select_csel(&mut self, cond: IntCC, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 3, "Select")?;
        Self::require_result(inst, "Select")?;

        let cond_val = inst.args[0];
        let true_val = inst.args[1];
        let false_val = inst.args[2];
        let result_val = inst.results[0];

        let ty = self.value_type(&true_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let cond_op = self.use_value(&cond_val)?;
        let true_op = self.use_value(&true_val)?;
        let false_op = self.use_value(&false_val)?;

        // First, test the condition value (CMP cond, #0)
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::CmpRI,
                vec![cond_op, MachOperand::Imm(0)],
            ),
        );

        // CSEL based on the IntCC condition
        let cc = AArch64CC::from_intcc(cond);
        let opc = if is_32 {
            AArch64Opcode::Csel
        } else {
            AArch64Opcode::Csel
        };

        self.func.push_inst(
            block,
            MachInst::new(
                opc,
                vec![
                    MachOperand::VReg(dst),
                    true_op,
                    false_op,
                    MachOperand::CondCode(cc),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Floating-point arithmetic
    // -----------------------------------------------------------------------

    /// Select floating-point binary operation.
    fn select_fp_binop(&mut self, op: AArch64FpBinOp, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "FpBinOp")?;
        Self::require_result(inst, "FpBinOp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_f32 = matches!(ty, Type::F32);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match (op, is_f32) {
            (AArch64FpBinOp::Fadd, true) => AArch64Opcode::FaddRR,
            (AArch64FpBinOp::Fadd, false) => AArch64Opcode::FaddRR,
            (AArch64FpBinOp::Fsub, true) => AArch64Opcode::FsubRR,
            (AArch64FpBinOp::Fsub, false) => AArch64Opcode::FsubRR,
            (AArch64FpBinOp::Fmul, true) => AArch64Opcode::FmulRR,
            (AArch64FpBinOp::Fmul, false) => AArch64Opcode::FmulRR,
            (AArch64FpBinOp::Fdiv, true) => AArch64Opcode::FdivRR,
            (AArch64FpBinOp::Fdiv, false) => AArch64Opcode::FdivRR,
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select floating-point comparison: FCMP + CSET.
    fn select_fcmp(&mut self, cond: FloatCC, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Fcmp")?;
        Self::require_result(inst, "Fcmp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_f32 = matches!(ty, Type::F32);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // FCMP sets NZCV
        let cmp_opc = if is_f32 {
            AArch64Opcode::Fcmp
        } else {
            AArch64Opcode::Fcmp
        };
        self.func.push_inst(
            block,
            MachInst::new(cmp_opc, vec![lhs, rhs]),
        );

        // CSET to materialize result
        let cc = AArch64CC::from_floatcc(cond);
        let dst = self.new_vreg(RegClass::Gpr32);
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::CSet,
                vec![MachOperand::VReg(dst), MachOperand::CondCode(cc)],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::B1);
        Ok(())
    }

    /// Select float-to-integer conversion (FCVTZS).
    ///
    /// FCVTZS rounds toward zero (truncation), matching C cast semantics.
    fn select_fcvt_to_int(&mut self, dst_ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "FcvtToInt")?;
        Self::require_result(inst, "FcvtToInt")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        // Select based on destination integer width. The source float
        // precision is encoded in the operand register class.
        let opc = if Self::is_32bit(&dst_ty) {
            AArch64Opcode::FcvtzsRR
        } else {
            AArch64Opcode::FcvtzsRR
        };

        // Operands: [dst, src, src_type_hint]
        // The src_type_hint helps the encoder know if source is S or D reg.
        let src_hint = if matches!(src_ty, Type::F32) { 32i64 } else { 64 };
        self.func.push_inst(
            block,
            MachInst::new(
                opc,
                vec![
                    MachOperand::VReg(dst),
                    src,
                    MachOperand::Imm(src_hint),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), dst_ty);
        Ok(())
    }

    /// Select integer-to-float conversion (SCVTF).
    fn select_fcvt_from_int(&mut self, src_ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "FcvtFromInt")?;
        Self::require_result(inst, "FcvtFromInt")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        // Destination type is the result's FP type; we infer from context.
        // The src_ty tells us the integer width of the source.
        let src = self.use_value(&src_val)?;

        // We need to know the destination float type. We default to F64 unless
        // the result type is tracked. For now, use the convention that
        // FcvtFromInt produces a value of the same "size class" as src_ty:
        //   I32 -> F32, I64 -> F64
        let dst_ty = if Self::is_32bit(&src_ty) {
            Type::F32
        } else {
            Type::F64
        };

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        let opc = match (Self::is_32bit(&src_ty), matches!(dst_ty, Type::F32)) {
            (true, true) => AArch64Opcode::ScvtfRR,    // SCVTF Sd, Wn
            (true, false) => AArch64Opcode::ScvtfRR,   // SCVTF Dd, Wn
            (false, true) => AArch64Opcode::ScvtfRR,   // SCVTF Sd, Xn
            (false, false) => AArch64Opcode::ScvtfRR,  // SCVTF Dd, Xn
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), dst_ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unsigned FP conversions
    // -----------------------------------------------------------------------

    /// Select float-to-unsigned-integer conversion (FCVTZU).
    ///
    /// FCVTZU rounds toward zero (truncation), like FCVTZS but for unsigned.
    fn select_fcvt_to_uint(&mut self, dst_ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FcvtToUint must have 1 arg");
        assert!(!inst.results.is_empty(), "FcvtToUint must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        let opc = if Self::is_32bit(&dst_ty) {
            AArch64Opcode::FcvtzuRR
        } else {
            AArch64Opcode::FcvtzuRR
        };

        let src_hint = if matches!(src_ty, Type::F32) { 32i64 } else { 64 };
        self.func.push_inst(
            block,
            MachInst::new(
                opc,
                vec![
                    MachOperand::VReg(dst),
                    src,
                    MachOperand::Imm(src_hint),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), dst_ty);
        Ok(())
    }

    /// Select unsigned-integer-to-float conversion (UCVTF).
    fn select_fcvt_from_uint(&mut self, src_ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FcvtFromUint must have 1 arg");
        assert!(!inst.results.is_empty(), "FcvtFromUint must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        // Same convention as signed: I32 -> F32, I64 -> F64
        let dst_ty = if Self::is_32bit(&src_ty) {
            Type::F32
        } else {
            Type::F64
        };

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        let opc = match (Self::is_32bit(&src_ty), matches!(dst_ty, Type::F32)) {
            (true, true) => AArch64Opcode::UcvtfRR,    // UCVTF Sd, Wn
            (true, false) => AArch64Opcode::UcvtfRR,   // UCVTF Dd, Wn
            (false, true) => AArch64Opcode::UcvtfRR,   // UCVTF Sd, Xn
            (false, false) => AArch64Opcode::UcvtfRR,  // UCVTF Dd, Xn
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), dst_ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Float precision conversion
    // -----------------------------------------------------------------------

    /// Select float precision widening: f32 -> f64 (FCVT Dd, Sn).
    fn select_fp_ext(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FPExt must have 1 arg");
        assert!(!inst.results.is_empty(), "FPExt must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        let dst = self.new_vreg(RegClass::Fpr64);

        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::FcvtSD,
                vec![MachOperand::VReg(dst), src],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::F64);
        Ok(())
    }

    /// Select float precision narrowing: f64 -> f32 (FCVT Ss, Dn).
    fn select_fp_trunc(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FPTrunc must have 1 arg");
        assert!(!inst.results.is_empty(), "FPTrunc must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        let dst = self.new_vreg(RegClass::Fpr32);

        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::FcvtDS,
                vec![MachOperand::VReg(dst), src],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::F32);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integer truncation
    // -----------------------------------------------------------------------

    /// Select integer truncation (narrow: i64->i32, i32->i16, etc.).
    ///
    /// On AArch64, truncation is essentially free for register values: the
    /// hardware uses the lower bits of the wider register. We emit a MOV
    /// to the narrower register class so that subsequent instructions see
    /// the correct width.
    fn select_trunc(&mut self, to_ty: &Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "Trunc must have 1 arg");
        assert!(!inst.results.is_empty(), "Trunc must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);

        // On AArch64, truncating from 64-bit to 32-bit (or narrower) is a
        // simple MOV to the W register — the upper 32 bits are ignored.
        // For sub-32-bit targets (I8, I16), we use a 32-bit MOV since
        // AArch64 W registers naturally zero-extend the upper bits.
        let opc = if Self::is_32bit(to_ty) {
            AArch64Opcode::MovR
        } else {
            AArch64Opcode::MovR
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), to_ty.clone());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bitcast
    // -----------------------------------------------------------------------

    /// Select bitcast: reinterpret bits between same-size types.
    ///
    /// On AArch64, bitcast between GPR and FPR uses FMOV:
    /// - GPR -> FPR: FMOV Sd, Wn (32-bit) or FMOV Dd, Xn (64-bit)
    /// - FPR -> GPR: FMOV Wd, Sn (32-bit) or FMOV Xd, Dn (64-bit)
    /// - Same register class: plain MOV (e.g., i32 -> u32, no-op in our IR)
    fn select_bitcast(&mut self, to_ty: &Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "Bitcast must have 1 arg");
        assert!(!inst.results.is_empty(), "Bitcast must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let src_is_fp = matches!(src_ty, Type::F32 | Type::F64);
        let dst_is_fp = matches!(to_ty, Type::F32 | Type::F64);

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);

        let opc = match (src_is_fp, dst_is_fp) {
            // GPR -> FPR (integer bits reinterpreted as float)
            (false, true) => {
                if Self::is_32bit(to_ty) {
                    AArch64Opcode::FmovGprFpr  // FMOV Sd, Wn
                } else {
                    AArch64Opcode::FmovGprFpr  // FMOV Dd, Xn
                }
            }
            // FPR -> GPR (float bits reinterpreted as integer)
            (true, false) => {
                if Self::is_32bit(&src_ty) {
                    AArch64Opcode::FmovFprGpr  // FMOV Wd, Sn
                } else {
                    AArch64Opcode::FmovFprGpr  // FMOV Xd, Dn
                }
            }
            // Same class: plain MOV
            _ => {
                if Self::is_32bit(to_ty) {
                    AArch64Opcode::MovR
                } else {
                    AArch64Opcode::MovR
                }
            }
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), to_ty.clone());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Addressing
    // -----------------------------------------------------------------------

    /// Select global symbol reference.
    ///
    /// On AArch64, global references use a two-instruction sequence:
    ///   ADRP Xd, #page(symbol)      -- load 4KB-aligned page address
    ///   ADD  Xd, Xd, #pageoff(symbol) -- add 12-bit page offset
    fn select_global_ref(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "GlobalRef")?;

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        // ADRP Xd, symbol@PAGE
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Adrp,
                vec![MachOperand::VReg(dst), MachOperand::Symbol(name.to_string())],
            ),
        );

        // ADD Xd, Xd, symbol@PAGEOFF
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![
                    MachOperand::VReg(dst),
                    MachOperand::VReg(dst),
                    MachOperand::Symbol(name.to_string()),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select external symbol reference via GOT (Global Offset Table).
    ///
    /// External symbols (from shared libraries or other translation units)
    /// require GOT-indirect access on Darwin/AArch64:
    ///   ADRP  Xd, symbol@GOTPAGE       -- load GOT page address
    ///   LDR   Xd, [Xd, symbol@GOTPAGEOFF]  -- load pointer from GOT slot
    ///
    /// This is different from local GlobalRef which uses ADRP+ADD (direct).
    /// The GOT slot contains the actual address of the symbol, resolved by
    /// the dynamic linker at load time.
    fn select_extern_ref(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.results.is_empty(), "ExternRef must have a result");

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        // ADRP Xd, symbol@GOTPAGE
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Adrp,
                vec![MachOperand::VReg(dst), MachOperand::Symbol(name.to_string())],
            ),
        );

        // LDR Xd, [Xd, symbol@GOTPAGEOFF]
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::LdrGot,
                vec![
                    MachOperand::VReg(dst),
                    MachOperand::VReg(dst),
                    MachOperand::Symbol(name.to_string()),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select thread-local variable reference via TLV descriptor.
    ///
    /// Thread-local variables on Darwin/AArch64 use a TLV descriptor pattern:
    ///   ADRP  Xd, symbol@TLVPPAGE        -- load TLV descriptor page address
    ///   LDR   Xd, [Xd, symbol@TLVPPAGEOFF]  -- load TLV descriptor pointer
    ///
    /// The loaded pointer is a TLV descriptor that the runtime uses to resolve
    /// the thread-local address. After this sequence, the caller typically
    /// invokes the TLV resolver function stored in the descriptor.
    fn select_tls_ref(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.results.is_empty(), "TlsRef must have a result");

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        // ADRP Xd, symbol@TLVPPAGE
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Adrp,
                vec![MachOperand::VReg(dst), MachOperand::Symbol(name.to_string())],
            ),
        );

        // LDR Xd, [Xd, symbol@TLVPPAGEOFF]
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::LdrTlvp,
                vec![
                    MachOperand::VReg(dst),
                    MachOperand::VReg(dst),
                    MachOperand::Symbol(name.to_string()),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select stack slot address computation.
    ///
    /// Emits ADD Xd, SP, #offset (the actual offset is resolved during
    /// frame lowering; we emit a placeholder StackSlot operand).
    fn select_stack_addr(&mut self, slot: u32, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_result(inst, "StackAddr")?;

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![
                    MachOperand::VReg(dst),
                    MachOperand::PReg(SP), // SP
                    MachOperand::StackSlot(slot),
                ],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::I64);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Finalize
    // -----------------------------------------------------------------------

    /// Define block parameter Values for a non-entry block.
    ///
    /// tMIR uses block parameters for SSA phi semantics. Before processing
    /// instructions in a block that has parameters, the ISel must have VRegs
    /// allocated for those parameter Values. This method creates a fresh VReg
    /// for each block parameter and records the mapping.
    ///
    /// Entry block parameters should be handled by `lower_formal_arguments`
    /// instead, which also emits COPY instructions from physical registers.
    pub fn define_block_params(&mut self, params: &[(Value, Type)]) {
        for (val, ty) in params {
            // Skip values already defined by copies in predecessor blocks.
            // The adapter emits single-arg Iadd (copy) instructions in
            // predecessor blocks that define block parameter values before
            // the target block is processed.
            if self.value_map.contains_key(val) {
                continue;
            }
            let class = reg_class_for_type(ty);
            let vreg = self.new_vreg(class);
            self.define_value(*val, MachOperand::VReg(vreg), ty.clone());
        }
    }

    /// Consume the selector and return the completed MachFunction.
    pub fn finalize(self) -> MachFunction {
        self.func
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Internal binop classification for opcode selection.
#[derive(Debug, Clone, Copy)]
enum AArch64BinOp {
    Add,
    Sub,
    Mul,
    Sdiv,
    Udiv,
}

/// Shift operation classification.
#[derive(Debug, Clone, Copy)]
enum AArch64ShiftOp {
    Lsl,
    Lsr,
    Asr,
}

/// Logical operation classification.
#[derive(Debug, Clone, Copy)]
enum AArch64LogicOp {
    And,
    Orr,
    Eor,
    Bic,
    Orn,
}

/// Integer unary operation classification.
#[derive(Debug, Clone, Copy)]
enum AArch64IntUnaryOp {
    Neg,
    Mvn,
}

/// FP binop classification.
#[derive(Debug, Clone, Copy)]
enum AArch64FpBinOp {
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Signature;
    use crate::instructions::{Block, FloatCC, Instruction, IntCC, Opcode, Value};

    /// Helper: create a simple isel for fn(i32, i32) -> i32.
    fn make_add_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel for fn(i64, i64) -> i64.
    fn make_i64_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("op64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel for fn(f64, f64) -> f64.
    fn make_f64_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::F64, Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fpop".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel for fn(f32, f32) -> f32.
    fn make_f32_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::F32, Type::F32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fpop32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel with no args, manually defining values.
    fn make_empty_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("test".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);
        (isel, entry)
    }

    // =======================================================================
    // Original tests (preserved from scaffold)
    // =======================================================================

    #[test]
    fn lower_formal_arguments_two_i32() {
        let (isel, entry) = make_add_isel();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::PReg(PReg::new(0))); // X0
        assert_eq!(mblock.insts[1].operands[1], MachOperand::PReg(PReg::new(1))); // X1
    }

    #[test]
    fn select_iadd_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn select_isub_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::SubRR);
    }

    #[test]
    fn select_icmp_and_brif() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp { cond: IntCC::Equal },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Brif {
                    cond: Value(2),
                    then_dest: Block(1),
                    else_dest: Block(2),
                },
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CmpRI);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::BCond);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::B);
        assert_eq!(mblock.successors.len(), 2);
    }

    #[test]
    fn select_return_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].operands[0], MachOperand::PReg(PReg::new(0))); // X0
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_iconst_small() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst { ty: Type::I32, imm: 42 },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
    }

    #[test]
    fn select_iconst_negative() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst { ty: Type::I64, imm: -1 },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[0].opcode, AArch64Opcode::Movn);
    }

    #[test]
    fn condition_code_mapping() {
        assert_eq!(AArch64CC::from_intcc(IntCC::Equal), AArch64CC::EQ);
        assert_eq!(AArch64CC::from_intcc(IntCC::NotEqual), AArch64CC::NE);
        assert_eq!(AArch64CC::from_intcc(IntCC::SignedLessThan), AArch64CC::LT);
        assert_eq!(AArch64CC::from_intcc(IntCC::UnsignedGreaterThan), AArch64CC::HI);
    }

    #[test]
    fn full_add_function() {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MovR);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Ret);
        assert_eq!(mfunc.name, "add");
        assert_eq!(mfunc.next_vreg, 3);
    }

    // =======================================================================
    // New tests: Shift operations
    // =======================================================================

    #[test]
    fn select_lsl_i32_register() {
        let (mut isel, entry) = make_add_isel();
        // LSL: Value(2) = Value(0) << Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY + 1 LSLVWr
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR);
    }

    #[test]
    fn select_lsr_i64_register() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ushr,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::LsrRR);
    }

    #[test]
    fn select_asr_i32_register() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sshr,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::AsrRR);
    }

    #[test]
    fn select_lsl_i32_immediate() {
        // Test immediate shift: define a constant shift amount, then shift
        let (mut isel, entry) = make_empty_isel();

        // Value(0) = iconst i32, 100
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst { ty: Type::I32, imm: 100 },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        // Value(1) = iconst i32, 4 (shift amount)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst { ty: Type::I32, imm: 4 },
                args: vec![],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        // Value(2) = ishl Value(0), Value(1)
        // Note: the shift amount is tracked as a VReg, not an immediate,
        // so this will use the register form. Immediate detection would
        // require peephole optimization (late combine phase).
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 MOVZWi + 1 MOVZWi + 1 LSLVWr (register form because value mapped as vreg)
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR);
    }

    // =======================================================================
    // New tests: Logical operations
    // =======================================================================

    #[test]
    fn select_and_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Band,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::AndRR);
    }

    #[test]
    fn select_orr_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bor,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::OrrRR);
    }

    #[test]
    fn select_eor_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bxor,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::EorRR);
    }

    #[test]
    fn select_bic_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::BandNot,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::BicRR);
    }

    #[test]
    fn select_orn_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::BorNot,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::OrnRR);
    }

    // =======================================================================
    // New tests: Extensions
    // =======================================================================

    #[test]
    fn select_sextb_to_i32() {
        // fn(i8) - sign extend byte to i32
        let sig = Signature {
            params: vec![Type::I8],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("sextb".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I8,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 SXTBWr
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Sxtb);
    }

    #[test]
    fn select_sexth_to_i64() {
        let sig = Signature {
            params: vec![Type::I16],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sexth".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I16,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Sxth);
    }

    #[test]
    fn select_sxtw_to_i64() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sxtw".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I32,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Sxtw);
    }

    #[test]
    fn select_uxtb_to_i32() {
        let sig = Signature {
            params: vec![Type::I8],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("uxtb".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I8,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Uxtw);
    }

    #[test]
    fn select_uxth_to_i32() {
        let sig = Signature {
            params: vec![Type::I16],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("uxth".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I16,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Uxtw);
    }

    #[test]
    fn select_uextend_i32_to_i64() {
        // Zero-extend i32 to i64: on AArch64, writing W reg zero-extends to X,
        // so we emit MOVWrr.
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("uxtw".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I32,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MovR);
    }

    // =======================================================================
    // New tests: Bitfield operations
    // =======================================================================

    #[test]
    fn select_ubfm_extract_i32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("ubfm".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Extract 8 bits starting at bit 4: UBFM Wd, Wn, #4, #11
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ExtractBits { lsb: 4, width: 8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[1];
        assert_eq!(inst.opcode, AArch64Opcode::Ubfm);
        // immr = 4, imms = 11
        assert_eq!(inst.operands[2], MachOperand::Imm(4));
        assert_eq!(inst.operands[3], MachOperand::Imm(11));
    }

    #[test]
    fn select_sbfm_extract_i64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sbfm".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Signed extract 16 bits from bit 0: SBFM Xd, Xn, #0, #15
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::SextractBits { lsb: 0, width: 16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[1];
        assert_eq!(inst.opcode, AArch64Opcode::Sbfm);
        assert_eq!(inst.operands[2], MachOperand::Imm(0));
        assert_eq!(inst.operands[3], MachOperand::Imm(15));
    }

    #[test]
    fn select_bfm_insert_i32() {
        let (mut isel, entry) = make_add_isel();

        // Insert 8 bits from Value(1) into Value(0) at bit 4
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::InsertBits { lsb: 4, width: 8 },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY (args) + 1 COPY (dst to result) + 1 BFMWri
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Bfm);
        // immr = (32 - 4) % 32 = 28, imms = 7
        assert_eq!(mblock.insts[3].operands[2], MachOperand::Imm(28));
        assert_eq!(mblock.insts[3].operands[3], MachOperand::Imm(7));
    }

    // =======================================================================
    // New tests: Conditional select
    // =======================================================================

    #[test]
    fn select_csel_i32() {
        let sig = Signature {
            params: vec![Type::I32, Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("csel".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Value(3) = select(NE, Value(0), Value(1), Value(2))
        // cond_val=Value(0), true=Value(1), false=Value(2)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Select { cond: IntCC::NotEqual },
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![Value(3)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 3 COPY (args) + 1 CMPWri + 1 CSELWr
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CmpRI);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Csel);
        assert_eq!(
            mblock.insts[4].operands[3],
            MachOperand::CondCode(AArch64CC::NE)
        );
    }

    #[test]
    fn select_csel_i64() {
        let sig = Signature {
            params: vec![Type::I32, Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("csel64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Select { cond: IntCC::SignedGreaterThan },
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![Value(3)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 3 COPY + 1 CMPWri + 1 CSELXr (64-bit because true_val is I64)
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Csel);
        assert_eq!(
            mblock.insts[4].operands[3],
            MachOperand::CondCode(AArch64CC::GT)
        );
    }

    // =======================================================================
    // New tests: Move wide (large immediate materialization)
    // =======================================================================

    #[test]
    fn select_iconst_large_i64() {
        let (mut isel, entry) = make_empty_isel();
        // 0x0001_0002_0003_0004: needs MOVZ + 3 MOVK
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I64,
                    imm: 0x0001_0002_0003_0004i64,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // MOVZ (low 16 bits) + 3 MOVK (remaining non-zero chunks)
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Imm(0x0004));

        // Should have MOVK for bits [16:31], [32:47], [48:63]
        let movk_count = mblock.insts.iter().filter(|i| i.opcode == AArch64Opcode::Movk).count();
        assert_eq!(movk_count, 3);
    }

    #[test]
    fn select_iconst_i32_large() {
        let (mut isel, entry) = make_empty_isel();
        // 0x00010002: needs MOVZ + 1 MOVK (2 chunks for 32-bit)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0x00010002i64,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Imm(0x0002));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[1].operands[1], MachOperand::Imm(0x0001));
        assert_eq!(mblock.insts[1].operands[2], MachOperand::Imm(16)); // shift=16
    }

    // =======================================================================
    // New tests: FP operations
    // =======================================================================

    #[test]
    fn select_fadd_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FaddRR);
    }

    #[test]
    fn select_fsub_f32() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FsubRR);
    }

    #[test]
    fn select_fmul_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fmul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FmulRR);
    }

    #[test]
    fn select_fdiv_f32() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fdiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FdivRR);
    }

    #[test]
    fn select_fcmp_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fcmp { cond: FloatCC::LessThan },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY + 1 FCMPDrr + 1 CSETWcc
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Fcmp);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[3].operands[1],
            MachOperand::CondCode(AArch64CC::MI)
        );
    }

    #[test]
    fn select_fcvt_to_int_i32() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("fcvt2i".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToInt { dst_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtzsRR);
    }

    #[test]
    fn select_fcvt_from_int_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("i2f".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromInt { src_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::ScvtfRR);
    }

    #[test]
    fn select_fcvt_from_int_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("i642f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromInt { src_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::ScvtfRR);
    }

    // =======================================================================
    // New tests: Addressing
    // =======================================================================

    #[test]
    fn select_global_ref() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::GlobalRef { name: "my_global".to_string() },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // ADRP + ADD
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Adrp);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Symbol("my_global".to_string()));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::AddPCRel);
        assert_eq!(mblock.insts[1].operands[2], MachOperand::Symbol("my_global".to_string()));
    }

    #[test]
    fn select_extern_ref_got_indirect() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ExternRef { name: "printf".to_string() },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // ADRP + LDR (GOT-indirect, not ADD)
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Adrp);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Symbol("printf".to_string()));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::LdrGot);
        assert_eq!(mblock.insts[1].operands[2], MachOperand::Symbol("printf".to_string()));
    }

    #[test]
    fn select_tls_ref_tlv_indirect() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::TlsRef { name: "thread_local_var".to_string() },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // ADRP + LDR (TLV-indirect)
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Adrp);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Symbol("thread_local_var".to_string()));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::LdrTlvp);
        assert_eq!(mblock.insts[1].operands[2], MachOperand::Symbol("thread_local_var".to_string()));
    }

    #[test]
    fn select_stack_addr() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StackAddr { slot: 3 },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddPCRel);
        // Should reference SP (PReg(31)) and StackSlot(3)
        assert_eq!(mblock.insts[0].operands[1], MachOperand::PReg(SP));
        assert_eq!(mblock.insts[0].operands[2], MachOperand::StackSlot(3));
    }

    // =======================================================================
    // New tests: FloatCC mapping
    // =======================================================================

    #[test]
    fn float_condition_code_mapping() {
        assert_eq!(AArch64CC::from_floatcc(FloatCC::Equal), AArch64CC::EQ);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::NotEqual), AArch64CC::NE);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::LessThan), AArch64CC::MI);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::LessThanOrEqual), AArch64CC::LS);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::GreaterThan), AArch64CC::GT);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::GreaterThanOrEqual), AArch64CC::GE);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::Ordered), AArch64CC::VC);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::Unordered), AArch64CC::VS);
    }

    #[test]
    fn condition_code_inversion() {
        assert_eq!(AArch64CC::EQ.invert(), AArch64CC::NE);
        assert_eq!(AArch64CC::NE.invert(), AArch64CC::EQ);
        assert_eq!(AArch64CC::LT.invert(), AArch64CC::GE);
        assert_eq!(AArch64CC::GE.invert(), AArch64CC::LT);
        assert_eq!(AArch64CC::GT.invert(), AArch64CC::LE);
        assert_eq!(AArch64CC::LE.invert(), AArch64CC::GT);
        assert_eq!(AArch64CC::HI.invert(), AArch64CC::LS);
        assert_eq!(AArch64CC::LS.invert(), AArch64CC::HI);
        assert_eq!(AArch64CC::VS.invert(), AArch64CC::VC);
        assert_eq!(AArch64CC::VC.invert(), AArch64CC::VS);
    }

    // =======================================================================
    // End-to-end: compound pattern
    // =======================================================================

    #[test]
    fn full_shift_and_mask() {
        // fn shift_mask(i32, i32) -> i32 { return (a << b) & 0xFF; }
        // This tests shift followed by logical AND.
        let (mut isel, entry) = make_add_isel();

        // Value(2) = ishl Value(0), Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();

        // Value(3) = iconst i32, 0xFF
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst { ty: Type::I32, imm: 0xFF },
                args: vec![],
                results: vec![Value(3)],
            },
            entry,
        ).unwrap();

        // Value(4) = band Value(2), Value(3)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Band,
                args: vec![Value(2), Value(3)],
                results: vec![Value(4)],
            },
            entry,
        ).unwrap();

        // Return Value(4)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(4)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected:
        // 0: COPY vreg0 <- X0
        // 1: COPY vreg1 <- X1
        // 2: LSLVWr vreg2 <- vreg0, vreg1
        // 3: MOVZWi vreg3 <- 0xFF
        // 4: ANDWrr vreg4 <- vreg2, vreg3
        // 5: MOVWrr X0 <- vreg4
        // 6: RET
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::AndRR);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::MovR);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn full_fp_add_function() {
        // fn fadd(f64, f64) -> f64 { return a + b; }
        let sig = Signature {
            params: vec![Type::F64, Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fadd64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY + 1 FADDDrr + 1 MOVXrr (to V0) + 1 RET
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FaddRR);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Call with Symbol operand (issue #69)
    // =======================================================================

    #[test]
    fn select_call_emits_symbol_operand() {
        // Test that Call { name } from the adapter produces a BL with Symbol operand.
        let (mut isel, entry) = make_empty_isel();

        // Define an arg value (simulating a function parameter).
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I32);

        // Select a Call instruction.
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Call { name: "my_callee".to_string() },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should contain at least: MOV (arg to X0) + BL + MOV (X0 to result)
        // Find the BL instruction and verify it has a Symbol operand.
        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(bl_inst.is_some(), "Expected BL instruction for function call");
        let bl = bl_inst.unwrap();
        assert_eq!(
            bl.operands[0],
            MachOperand::Symbol("my_callee".to_string()),
            "BL should have Symbol operand with callee name"
        );
    }

    // =======================================================================
    // CondCode operand preserved through Brif (issue #69)
    // =======================================================================

    #[test]
    fn select_brif_emits_condcode_operand() {
        // Test that Brif from the adapter produces B.cond with CondCode operand.
        let (mut isel, entry) = make_empty_isel();

        // Define a boolean condition value.
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::B1);

        let then_block = Block(1);
        let else_block = Block(2);
        isel.func.ensure_block(then_block);
        isel.func.ensure_block(else_block);

        // Select a Brif instruction.
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Brif {
                    cond: Value(0),
                    then_dest: then_block,
                    else_dest: else_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected: CMP W, #0 + B.NE then + B else
        assert!(mblock.insts.len() >= 3, "Brif should emit CMP + B.cond + B");

        // Find B.cond and verify CondCode operand.
        let bcc_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::BCond);
        assert!(bcc_inst.is_some(), "Expected B.cond instruction");
        let bcc = bcc_inst.unwrap();
        assert_eq!(
            bcc.operands[0],
            MachOperand::CondCode(AArch64CC::NE),
            "B.cond should have NE condition code (branch if nonzero)"
        );
        assert_eq!(
            bcc.operands[1],
            MachOperand::Block(then_block),
            "B.cond should target the then block"
        );
    }

    // =======================================================================
    // Integer remainder (SRem/URem) tests
    // =======================================================================

    #[test]
    fn select_urem_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Urem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPYs for args + UDIV + MSUB = 4 instructions
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::UDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
        // MSUB operands: [dst, tmp(quotient), divisor, dividend]
        assert_eq!(mblock.insts[3].operands.len(), 4);
    }

    #[test]
    fn select_srem_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Srem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
    }

    #[test]
    fn select_urem_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Urem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::UDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
    }

    #[test]
    fn select_srem_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Srem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
    }

    #[test]
    fn select_srem_msub_operand_order() {
        // Verify MSUB operand order: MSUB dst, tmp(quotient), divisor, dividend
        // This ensures result = dividend - quotient * divisor
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Srem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let div_inst = &mblock.insts[2];
        let msub_inst = &mblock.insts[3];

        // SDIV dst=tmp, lhs=dividend, rhs=divisor
        assert_eq!(div_inst.opcode, AArch64Opcode::SDiv);
        let div_dst = &div_inst.operands[0]; // tmp vreg
        let dividend_op = &div_inst.operands[1]; // dividend
        let divisor_op = &div_inst.operands[2]; // divisor

        // MSUB dst, Rn=tmp, Rm=divisor, Ra=dividend
        assert_eq!(msub_inst.opcode, AArch64Opcode::Msub);
        assert_eq!(&msub_inst.operands[1], div_dst, "MSUB Rn should be quotient from SDIV");
        assert_eq!(&msub_inst.operands[2], divisor_op, "MSUB Rm should be divisor");
        assert_eq!(&msub_inst.operands[3], dividend_op, "MSUB Ra should be dividend");
    }

    // =======================================================================
    // Unary operations: Neg, Not (Bnot), FNeg
    // =======================================================================

    #[test]
    fn select_ineg_i32() {
        // fn(i32) -> i32: return -arg
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("neg32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 NEGWr
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Neg);
        assert_eq!(mblock.insts[1].operands.len(), 2);
    }

    #[test]
    fn select_ineg_i64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("neg64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Neg);
    }

    #[test]
    fn select_bnot_i32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("not32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bnot,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 MVNWr
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::OrnRR);
    }

    #[test]
    fn select_bnot_i64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("not64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bnot,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::OrnRR);
    }

    #[test]
    fn select_fneg_f64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fneg64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fneg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 FNEGDr
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FnegRR);
    }

    #[test]
    fn select_fneg_f32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fneg32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fneg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FnegRR);
    }

    #[test]
    fn full_negate_function() {
        // fn negate(i32) -> i32 { return -arg; }
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("negate".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Value(1) = Ineg(Value(0))
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        // Return Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(1)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // COPY (arg) + NEGWr + MOVWrr (to X0) + RET
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Neg);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::MovR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn full_fneg_function() {
        // fn fneg(f64) -> f64 { return -arg; }
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fneg_fn".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fneg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(1)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // COPY (arg) + FNEGDr + MOV (to V0) + RET
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FnegRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Type conversion: unsigned FP <-> int
    // =======================================================================

    #[test]
    fn select_fcvt_to_uint_i32() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("fcvt2u".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToUint { dst_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtzuRR);
    }

    #[test]
    fn select_fcvt_to_uint_i64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("fcvt2u64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToUint { dst_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtzuRR);
    }

    #[test]
    fn select_fcvt_from_uint_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("u2f".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromUint { src_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UcvtfRR);
    }

    #[test]
    fn select_fcvt_from_uint_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("u642f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromUint { src_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UcvtfRR);
    }

    // =======================================================================
    // Type conversion: FP precision (FPExt, FPTrunc)
    // =======================================================================

    #[test]
    fn select_fp_ext_f32_to_f64() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fpext".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FPExt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtSD);
    }

    #[test]
    fn select_fp_trunc_f64_to_f32() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fptrunc".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FPTrunc,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtDS);
    }

    // =======================================================================
    // Type conversion: integer truncation
    // =======================================================================

    #[test]
    fn select_trunc_i64_to_i32() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("trunc64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        // On AArch64, truncation is a MOV to narrower register class
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn select_trunc_i32_to_i16() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I16],
        };
        let mut isel = InstructionSelector::new("trunc16".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        // I16 is sub-32-bit, still uses MOVWrr
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn select_trunc_i64_to_i8() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I8],
        };
        let mut isel = InstructionSelector::new("trunc8".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MovR);
    }

    // =======================================================================
    // Type conversion: bitcast
    // =======================================================================

    #[test]
    fn select_bitcast_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("bc_i2f".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::F32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        // GPR -> FPR: FMOV Sd, Wn
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FmovGprFpr);
    }

    #[test]
    fn select_bitcast_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("bc_i642f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::F64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        // GPR -> FPR: FMOV Dd, Xn
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FmovGprFpr);
    }

    #[test]
    fn select_bitcast_f32_to_i32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("bc_f2i".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        // FPR -> GPR: FMOV Wd, Sn
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FmovFprGpr);
    }

    #[test]
    fn select_bitcast_f64_to_i64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("bc_f642i64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        // FPR -> GPR: FMOV Xd, Dn
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FmovFprGpr);
    }

    #[test]
    fn select_bitcast_same_class_i32_to_i32() {
        // Same class bitcast (e.g., type punning between integer types)
        // should emit a plain MOV
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I32);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[0].opcode, AArch64Opcode::MovR);
    }

    // =======================================================================
    // Aggregate ISel tests
    // =======================================================================

    #[test]
    fn select_struct_gep_field_0() {
        // struct { I32, I32 } -> field 0 is at offset 0, should emit MOV
        let (mut isel, entry) = make_empty_isel();

        // Define a pointer value
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep { struct_ty, field_index: 0 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // Field 0 at offset 0 -> MOVXrr
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn select_struct_gep_field_1_with_offset() {
        // struct { I32, I32 } -> field 1 is at offset 4, should emit ADD
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep { struct_ty, field_index: 1 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // Field 1 at offset 4 -> ADDXri
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRI);
        // Offset should be 4
        assert_eq!(mblock.insts[0].operands[2], MachOperand::Imm(4));
    }

    #[test]
    fn select_struct_gep_with_padding() {
        // struct { I8, I32 } -> field 1 is at offset 4 (3 bytes padding)
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I8, Type::I32]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep { struct_ty, field_index: 1 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRI);
        assert_eq!(mblock.insts[0].operands[2], MachOperand::Imm(4));
    }

    #[test]
    fn select_struct_gep_out_of_range() {
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I32]);
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep { struct_ty, field_index: 5 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        );
        assert!(result.is_err());
    }

    #[test]
    fn select_struct_gep_non_struct_type() {
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I64);

        // Using I32 as the struct_ty should fail
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep { struct_ty: Type::I32, field_index: 0 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        );
        assert!(result.is_err());
    }

    #[test]
    fn select_small_struct_return() {
        // Return a small struct (8 bytes, two i32 fields) packed in X0.
        // The return value is a pointer to the struct; select_return should
        // load it into X0.
        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_small".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        // Define a pointer to the struct as a vreg
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // ABI classifies Struct return as Indirect{X8}, so aggregate return
        // is invoked. For 8-byte struct: single LDR X0, [src]
        let insts = &mblock.insts;
        // Find the LDR X0 instruction
        let ldr_inst = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(ldr_inst.is_some(), "Expected LDR X0 for small struct return");
        // Must end with RET
        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_medium_struct_return_16bytes() {
        // Return a 16-byte struct -> X0 + X1
        let struct_ty = Type::Struct(vec![Type::I64, Type::I64]);
        assert_eq!(struct_ty.bytes(), 16);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_med".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Should have LDR X0 at offset 0, LDR X1 at offset 8, then RET
        let ldr_x0 = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(ldr_x0.is_some(), "Expected LDR X0 for medium struct return");

        let ldr_x1 = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X1))
        });
        assert!(ldr_x1.is_some(), "Expected LDR X1 for medium struct return");
        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_large_struct_return_via_sret() {
        // Return a large struct (24 bytes) -> store to [X8]
        let struct_ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        assert_eq!(struct_ty.bytes(), 24);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_large".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Should have 3 LDR+STR pairs (24 bytes / 8 = 3 chunks) then RET
        let str_to_x8: Vec<_> = insts.iter().filter(|i| {
            i.opcode == AArch64Opcode::StrRI
                && i.operands.get(1) == Some(&MachOperand::PReg(gpr::X8))
        }).collect();
        assert_eq!(str_to_x8.len(), 3, "Expected 3 stores to X8 for 24-byte struct");
        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_call_with_small_aggregate_arg() {
        // Call a function that takes a small struct (8 bytes).
        // The ABI should pass it in a single register.
        let (mut isel, entry) = make_empty_isel();

        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        // Define a pointer to the struct
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        // Define a result value
        let result_types = vec![Type::I32];
        isel.select_call(
            "callee",
            &[Value(0)],
            &[Value(1)],
            &result_types,
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should have: LDR X0 (aggregate arg), BL, MOV (result)
        let has_bl = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Bl);
        assert!(has_bl, "Expected BL instruction for call");

        // The aggregate arg should be loaded into X0 (small struct -> Reg(X0))
        let ldr_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(ldr_x0.is_some(), "Expected LDR X0 for small aggregate arg");
    }

    #[test]
    fn select_call_with_large_aggregate_arg() {
        // Call a function that takes a large struct (24 bytes).
        // The ABI should pass it indirectly via pointer.
        let (mut isel, entry) = make_empty_isel();

        let struct_ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        let result_types = vec![Type::I32];
        isel.select_call(
            "callee_big",
            &[Value(0)],
            &[Value(1)],
            &result_types,
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Large aggregate -> Indirect{X0}, so should emit MOVXrr to X0
        let mov_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Expected MOV X0 for large aggregate indirect pass");
    }

    #[test]
    fn select_call_returns_aggregate_via_sret() {
        // Call a function that returns an aggregate (via X8 sret)
        let (mut isel, entry) = make_empty_isel();

        let struct_ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);

        let result_types = vec![struct_ty];
        isel.select_call(
            "returns_struct",
            &[],
            &[Value(0)],
            &result_types,
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // After BL, should copy X8 to a vreg for the result
        let has_bl = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Bl);
        assert!(has_bl);

        // Result should be defined (the MOVXrr from X8)
        let mov_from_x8 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.get(1) == Some(&MachOperand::PReg(gpr::X8))
        });
        assert!(mov_from_x8.is_some(), "Expected MOV from X8 for sret result");
    }

    #[test]
    fn struct_gep_three_field_struct() {
        // struct { I8, I32, I64 } -> test all field offsets
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I8, Type::I32, Type::I64]);
        // Verify expected offsets from the type system
        assert_eq!(struct_ty.offset_of(0), Some(0));
        assert_eq!(struct_ty.offset_of(1), Some(4));  // 1 byte + 3 padding
        assert_eq!(struct_ty.offset_of(2), Some(8));  // 4+4=8, already aligned to 8

        // Field 2 at offset 8
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep { struct_ty, field_index: 2 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRI);
        assert_eq!(mblock.insts[0].operands[2], MachOperand::Imm(8));
    }

    // =======================================================================
    // Extending loads and truncating stores (byte/halfword)
    // =======================================================================

    #[test]
    fn select_load_i8_emits_ldrb() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::LdrbRI,
            "Load I8 should emit LDRBui (zero-extending byte load)");
    }

    #[test]
    fn select_load_i16_emits_ldrh() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::LdrhRI,
            "Load I16 should emit LDRHui (zero-extending halfword load)");
    }

    #[test]
    fn select_store_i8_emits_strb() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(val_vreg), Type::I8);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), MachOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::StrbRI,
            "Store I8 should emit STRBui (truncating byte store)");
    }

    #[test]
    fn select_store_i16_emits_strh() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(val_vreg), Type::I16);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), MachOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::StrhRI,
            "Store I16 should emit STRHui (truncating halfword store)");
    }

    #[test]
    fn select_load_i8_add_store_i8_roundtrip() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(addr_vreg), Type::I64);
        let addr_vreg2 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), MachOperand::VReg(addr_vreg2), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I8 },
                args: vec![Value(1)],
                results: vec![Value(3)],
            },
            entry,
        ).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(2), Value(3)],
                results: vec![Value(4)],
            },
            entry,
        ).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(4), Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(insts[0].opcode, AArch64Opcode::LdrbRI,
            "First instruction should be byte load");
        assert_eq!(insts[1].opcode, AArch64Opcode::LdrbRI,
            "Second instruction should be byte load");
        assert_eq!(insts[2].opcode, AArch64Opcode::AddRR,
            "Third instruction should be 32-bit add (I8 uses Gpr32)");
        assert_eq!(insts[3].opcode, AArch64Opcode::StrbRI,
            "Fourth instruction should be byte store");
    }

    #[test]
    fn select_large_struct_return_with_trailing_bytes() {
        // Return a struct with 17 I8 fields (17 bytes, align 1).
        // sret path: 2x 8-byte LDR/STR + 1x 1-byte LDRB/STRB tail.
        let struct_ty = Type::Struct(vec![Type::I8; 17]);
        assert_eq!(struct_ty.bytes(), 17);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_tail".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Count stores to X8
        let str_to_x8: Vec<_> = insts.iter().filter(|i| {
            (i.opcode == AArch64Opcode::StrRI || i.opcode == AArch64Opcode::StrbRI)
                && i.operands.get(1) == Some(&MachOperand::PReg(gpr::X8))
        }).collect();
        // 2x STRXui (8-byte each) + 1x STRBui (1-byte tail) = 3 stores
        assert_eq!(str_to_x8.len(), 3, "Expected 3 stores for 17-byte struct sret: 2x8 + 1x1");

        // Verify the tail store is a byte store
        let tail_store = str_to_x8.last().unwrap();
        assert_eq!(tail_store.opcode, AArch64Opcode::StrbRI,
            "Trailing byte should use STRB");
        assert_eq!(tail_store.operands.get(2), Some(&MachOperand::Imm(16)),
            "Trailing byte store should be at offset 16");

        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_large_struct_return_with_trailing_halfword() {
        // Return a struct with a trailing 2-byte field: 3x I64 + 1x I16 = 26 bytes
        // (padded to 32 by align_to(26, 8) = 32).
        // sret: 4x 8-byte = 32 bytes copied. No tails needed.
        //
        // Better: 9x I16 = 18 bytes, align 2, align_to(18, 2) = 18.
        let struct_ty = Type::Struct(vec![Type::I16; 9]);
        assert_eq!(struct_ty.bytes(), 18);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_hw_tail".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // 18 bytes: 2x 8-byte STRXui (16), then 1x 2-byte STRHui (18)
        // No 4-byte or 1-byte tails.
        let str_to_x8: Vec<_> = insts.iter().filter(|i| {
            (i.opcode == AArch64Opcode::StrRI
                || i.opcode == AArch64Opcode::StrhRI)
                && i.operands.get(1) == Some(&MachOperand::PReg(gpr::X8))
        }).collect();
        assert_eq!(str_to_x8.len(), 3, "Expected 3 stores for 18-byte struct: 2x8 + 1x2");

        let tail_store = str_to_x8.last().unwrap();
        assert_eq!(tail_store.opcode, AArch64Opcode::StrhRI,
            "Trailing halfword should use STRH");
        assert_eq!(tail_store.operands.get(2), Some(&MachOperand::Imm(16)),
            "Trailing halfword store should be at offset 16");

        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_large_struct_return_with_4_2_1_tail() {
        // 23 bytes: 2x 8-byte (16) + 1x 4-byte (20) + 1x 2-byte (22) + 1x 1-byte (23)
        // Use 23x I8 -> bytes=23, align=1
        let struct_ty = Type::Struct(vec![Type::I8; 23]);
        assert_eq!(struct_ty.bytes(), 23);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_421_tail".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Count all stores to X8
        let str_to_x8: Vec<_> = insts.iter().filter(|i| {
            matches!(i.opcode,
                AArch64Opcode::StrRI
                | AArch64Opcode::StrhRI | AArch64Opcode::StrbRI)
                && i.operands.get(1) == Some(&MachOperand::PReg(gpr::X8))
        }).collect();
        // 2x STRXui + 1x STRWui + 1x STRHui + 1x STRBui = 5 stores
        assert_eq!(str_to_x8.len(), 5,
            "Expected 5 stores for 23-byte struct: 2x8 + 1x4 + 1x2 + 1x1");

        // Verify tail opcodes in order
        assert_eq!(str_to_x8[0].opcode, AArch64Opcode::StrRI);
        assert_eq!(str_to_x8[1].opcode, AArch64Opcode::StrRI);
        assert_eq!(str_to_x8[2].opcode, AArch64Opcode::StrRI);
        assert_eq!(str_to_x8[3].opcode, AArch64Opcode::StrhRI);
        assert_eq!(str_to_x8[4].opcode, AArch64Opcode::StrbRI);

        // Verify offsets
        assert_eq!(str_to_x8[0].operands.get(2), Some(&MachOperand::Imm(0)));
        assert_eq!(str_to_x8[1].operands.get(2), Some(&MachOperand::Imm(8)));
        assert_eq!(str_to_x8[2].operands.get(2), Some(&MachOperand::Imm(16)));
        assert_eq!(str_to_x8[3].operands.get(2), Some(&MachOperand::Imm(20)));
        assert_eq!(str_to_x8[4].operands.get(2), Some(&MachOperand::Imm(22)));

        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Variadic call ISel tests (Apple AArch64 ABI, issue #79)
    // =======================================================================

    #[test]
    fn select_variadic_call_printf_like() {
        // printf(const char* fmt, ...) called as printf(fmt, 42, 100)
        // fixed_count=1: fmt -> X0 (register)
        // variadic: 42(I32) -> stack[0], 100(I64) -> stack[8]
        let (mut isel, entry) = make_empty_isel();

        // Define arg values
        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(v0_reg), Type::I64); // fmt
        let v1_reg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), MachOperand::VReg(v1_reg), Type::I32); // 42
        let v2_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), MachOperand::VReg(v2_reg), Type::I64); // 100

        let result_types = vec![Type::I32]; // printf returns int
        isel.select_variadic_call(
            "printf",
            1, // 1 fixed arg (fmt)
            &[Value(0), Value(1), Value(2)],
            &[Value(3)],
            &result_types,
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Should have: MOV X0 (fmt), STR [SP+0] (42), STR [SP+8] (100), BL, MOV (result)
        // First: MOV to X0 for fixed arg
        let mov_x0 = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Fixed arg should go to X0");

        // Variadic args should be STR to stack
        let str_sp: Vec<_> = insts.iter().filter(|i| {
            matches!(i.opcode, AArch64Opcode::StrRI | AArch64Opcode::StrRI)
                && i.operands.get(1) == Some(&MachOperand::PReg(SP))
        }).collect();
        assert_eq!(str_sp.len(), 2, "Two variadic args should be stored to stack");

        // Verify offsets: first at 0, second at 8
        assert_eq!(str_sp[0].operands.get(2), Some(&MachOperand::Imm(0)));
        assert_eq!(str_sp[1].operands.get(2), Some(&MachOperand::Imm(8)));

        // BL should be present
        let bl_inst = insts.iter().find(|i| i.opcode == AArch64Opcode::BL);
        assert!(bl_inst.is_some(), "Expected BL instruction");
        assert_eq!(
            bl_inst.unwrap().operands[0],
            MachOperand::Symbol("printf".to_string()),
        );
    }

    #[test]
    fn select_variadic_call_float_on_stack() {
        // Apple ABI: variadic floats go on stack, NOT in FPR.
        // fn(i64, ...) called with (ptr, 1.0f64)
        let (mut isel, entry) = make_empty_isel();

        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(v0_reg), Type::I64);
        let v1_reg = isel.new_vreg(RegClass::Fpr64);
        isel.define_value(Value(1), MachOperand::VReg(v1_reg), Type::F64);

        let result_types = vec![Type::I32];
        isel.select_variadic_call(
            "my_varfn",
            1,
            &[Value(0), Value(1)],
            &[Value(2)],
            &result_types,
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // The variadic F64 should be stored to stack using STRDui (not MOV to V0)
        let str_d_sp = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::StrRI
                && i.operands.get(1) == Some(&MachOperand::PReg(SP))
        });
        assert!(str_d_sp.is_some(), "Variadic f64 should use STRDui to stack");
        assert_eq!(
            str_d_sp.unwrap().operands.get(2),
            Some(&MachOperand::Imm(0)),
            "Variadic f64 at stack offset 0"
        );

        // V0 should NOT be used (no MOV to V0 for variadic float)
        let mov_v0 = insts.iter().find(|i| {
            i.operands.first() == Some(&MachOperand::PReg(gpr::V0))
        });
        assert!(mov_v0.is_none(), "Variadic float should NOT go in V0 (Apple ABI)");
    }

    #[test]
    fn select_variadic_call_via_opcode() {
        // Test the CallVariadic opcode dispatch through select_instruction.
        let (mut isel, entry) = make_empty_isel();

        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(v0_reg), Type::I64); // fmt
        let v1_reg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), MachOperand::VReg(v1_reg), Type::I32); // vararg

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallVariadic {
                    name: "NSLog".to_string(),
                    fixed_args: 1,
                },
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // MOV X0 (fixed), STR [SP+0] (variadic), BL NSLog
        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::BL);
        assert!(bl_inst.is_some());
        assert_eq!(
            bl_inst.unwrap().operands[0],
            MachOperand::Symbol("NSLog".to_string()),
        );

        let str_sp = mblock.insts.iter().find(|i| {
            matches!(i.opcode, AArch64Opcode::StrRI)
                && i.operands.get(1) == Some(&MachOperand::PReg(SP))
        });
        assert!(str_sp.is_some(), "Variadic i32 arg should be on stack");
    }

    #[test]
    fn select_variadic_call_no_varargs() {
        // Variadic function called with only fixed args (no varargs passed).
        // Should behave identically to a normal call.
        let (mut isel, entry) = make_empty_isel();

        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(v0_reg), Type::I64);

        let result_types = vec![Type::I32];
        isel.select_variadic_call(
            "printf",
            1,
            &[Value(0)],
            &[Value(1)],
            &result_types,
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Fixed arg in X0, BL, result from X0
        let mov_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Fixed arg should go to X0");

        // No stack stores (no varargs)
        let str_sp = mblock.insts.iter().any(|i| {
            i.operands.get(1) == Some(&MachOperand::PReg(SP))
                && matches!(i.opcode, AArch64Opcode::StrRI | AArch64Opcode::StrRI
                    | AArch64Opcode::StrRI | AArch64Opcode::StrRI)
        });
        assert!(!str_sp, "No stack stores when no varargs passed");
    }

    // =======================================================================
    // CallIndirect (indirect call via function pointer)
    // =======================================================================

    #[test]
    fn select_call_indirect_basic() {
        // Test: call_indirect(fn_ptr, arg) -> result
        // fn_ptr is I64, arg is I32, returns I32
        let (mut isel, entry) = make_empty_isel();

        // Define the function pointer value
        let fp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(fp_vreg), Type::I64);

        // Define an argument value
        let arg_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), MachOperand::VReg(arg_vreg), Type::I32);

        // Select CallIndirect: args[0]=fn_ptr, args[1]=arg, results=[retval]
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallIndirect,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should contain: MOV (arg to X0), MOV (fn_ptr to X16), BLR X16, MOV (X0 to result)
        // Find the BLR instruction
        let blr_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::BLR);
        assert!(blr_inst.is_some(), "Expected BLR instruction for indirect call");
        let blr = blr_inst.unwrap();
        assert_eq!(
            blr.operands[0],
            MachOperand::PReg(gpr::X16),
            "BLR should target X16 (IP0 scratch register)"
        );

        // Verify arg was moved to X0
        let mov_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Arg should be moved to X0");

        // Verify fn_ptr was moved to X16
        let mov_x16 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X16))
        });
        assert!(mov_x16.is_some(), "Function pointer should be moved to X16");
    }

    #[test]
    fn select_call_indirect_no_args_no_results() {
        // Test: call_indirect(fn_ptr) with no args and no return
        let (mut isel, entry) = make_empty_isel();

        let fp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(fp_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallIndirect,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should have: MOV fn_ptr -> X16, BLR X16
        assert!(mblock.insts.len() >= 2, "At least MOV + BLR");

        let blr_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::BLR);
        assert!(blr_inst.is_some(), "Expected BLR for indirect call");
    }

    #[test]
    fn select_call_indirect_multiple_args() {
        // Test: call_indirect(fn_ptr, a, b, c) -> result
        let (mut isel, entry) = make_empty_isel();

        let fp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(fp_vreg), Type::I64);

        let a_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), MachOperand::VReg(a_vreg), Type::I64);
        let b_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), MachOperand::VReg(b_vreg), Type::I64);
        let c_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(3), MachOperand::VReg(c_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallIndirect,
                args: vec![Value(0), Value(1), Value(2), Value(3)],
                results: vec![Value(4)],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Verify args go to X0, X1, X2
        let mov_x0 = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X0))
        });
        let mov_x1 = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X1))
        });
        let mov_x2 = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&MachOperand::PReg(gpr::X2))
        });
        assert!(mov_x0, "First arg should go to X0");
        assert!(mov_x1, "Second arg should go to X1");
        assert!(mov_x2, "Third arg should go to X2");

        // BLR should be present
        let blr = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::BLR);
        assert!(blr.is_some(), "Expected BLR");
    }

    // =======================================================================
    // Switch (cascading CMP+B.EQ chain)
    // =======================================================================

    #[test]
    fn select_switch_two_cases() {
        // switch(selector) { 0 => block1, 1 => block2, default => block3 }
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(sel_vreg), Type::I32);

        let block1 = Block(1);
        let block2 = Block(2);
        let block3 = Block(3);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block2);
        isel.func.ensure_block(block3);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(0, block1), (1, block2)],
                    default: block3,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected pattern:
        //   CMP sel, #0; B.EQ block1
        //   CMP sel, #1; B.EQ block2
        //   B block3 (default)
        //
        // That's 2*(CMP+B.cond) + 1*B = 5 instructions

        // Count B.cond instructions
        let bcc_count = mblock.insts.iter()
            .filter(|i| i.opcode == AArch64Opcode::Bcc)
            .count();
        assert_eq!(bcc_count, 2, "Should have 2 conditional branches (one per case)");

        // Each B.cond should use EQ condition
        for bcc in mblock.insts.iter().filter(|i| i.opcode == AArch64Opcode::Bcc) {
            assert_eq!(
                bcc.operands[0],
                MachOperand::CondCode(AArch64CC::EQ),
                "Switch cases use B.EQ"
            );
        }

        // The last instruction should be unconditional B to default
        let last = mblock.insts.last().unwrap();
        assert_eq!(last.opcode, AArch64Opcode::B, "Last inst should be B (default fallthrough)");
        assert_eq!(last.operands[0], MachOperand::Block(block3), "Default should be block3");

        // Verify successors
        let succs = &mfunc.blocks[&entry].successors;
        assert!(succs.contains(&block1), "block1 should be a successor");
        assert!(succs.contains(&block2), "block2 should be a successor");
        assert!(succs.contains(&block3), "block3 (default) should be a successor");
    }

    #[test]
    fn select_switch_large_case_value() {
        // Test case value > 4095 (doesn't fit in CMP immediate)
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), MachOperand::VReg(sel_vreg), Type::I64);

        let block1 = Block(1);
        let block2 = Block(2);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block2);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(0x10000, block1)],
                    default: block2,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // For large case value: MOVZ + CMP reg,reg + B.EQ + B
        // MOVZ materializes the constant, then CMP is register-register
        let movz_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Movz);
        assert!(movz_inst.is_some(), "Large case value should be materialized via MOVZXi");

        let cmp_rr = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::CmpRR);
        assert!(cmp_rr.is_some(), "Large case value should use CMP reg,reg");
    }

    #[test]
    fn select_switch_single_case() {
        // Degenerate switch with one case
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(sel_vreg), Type::I32);

        let block1 = Block(1);
        let block_default = Block(2);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block_default);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(42, block1)],
                    default: block_default,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // CMP + B.EQ + B = 3 instructions
        let bcc_count = mblock.insts.iter()
            .filter(|i| i.opcode == AArch64Opcode::Bcc)
            .count();
        assert_eq!(bcc_count, 1, "Single case should produce 1 B.EQ");

        let last = mblock.insts.last().unwrap();
        assert_eq!(last.opcode, AArch64Opcode::B);
        assert_eq!(last.operands[0], MachOperand::Block(block_default));
    }

    #[test]
    fn select_switch_zero_cases() {
        // Switch with no cases = just jump to default
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), MachOperand::VReg(sel_vreg), Type::I32);

        let block_default = Block(1);
        isel.func.ensure_block(block_default);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![],
                    default: block_default,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        ).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Just one B to default
        assert_eq!(mblock.insts.len(), 1, "Empty switch = just B to default");
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::B);
        assert_eq!(mblock.insts[0].operands[0], MachOperand::Block(block_default));
    }
}

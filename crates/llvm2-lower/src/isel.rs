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
}

// ---------------------------------------------------------------------------
// ISel-level AArch64 opcode enumeration
// ---------------------------------------------------------------------------
//
// NOTE: These ISel-level types (AArch64Opcode, MachOperand, AArch64CC,
// MachInst, MachBlock, MachFunction) intentionally shadow the types in
// `llvm2-ir`. The ISel types represent the *instruction selection* output
// with LLVM-style per-width opcode naming (ADDWrr, ADDXrr) and HashMap-
// based block storage, while `llvm2-ir` types represent the canonical
// machine IR with abstract opcodes (AddRR) and arena-indexed storage.
//
// The pipeline flow is:
//   tMIR -> isel::MachFunction (this module)
//        -> llvm2_ir::MachFunction (canonical IR for opt/regalloc/codegen)
//
// A translation pass (not yet implemented) will lower isel output to the
// canonical llvm2-ir representation. See issue #37 for unification tracking.
// ---------------------------------------------------------------------------

/// AArch64 machine opcodes for instruction selection output.
///
/// Uses LLVM-style per-width naming (ADDWrr = 32-bit ADD, ADDXrr = 64-bit ADD)
/// to distinguish width variants at ISel time. These are separate from
/// `llvm2_ir::AArch64Opcode` which uses abstract names (AddRR, AddRI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AArch64Opcode {
    // Arithmetic (register forms)
    ADDWrr,   // 32-bit add, reg+reg
    ADDXrr,   // 64-bit add, reg+reg
    ADDWri,   // 32-bit add, reg+imm12
    ADDXri,   // 64-bit add, reg+imm12
    SUBWrr,   // 32-bit sub, reg+reg
    SUBXrr,   // 64-bit sub, reg+reg
    SUBWri,   // 32-bit sub, reg+imm12
    SUBXri,   // 64-bit sub, reg+imm12
    MULWrrr,  // 32-bit multiply (MADD Wd, Wn, Wm, WZR)
    MULXrrr,  // 64-bit multiply (MADD Xd, Xn, Xm, XZR)
    SDIVWrr,  // 32-bit signed divide
    SDIVXrr,  // 64-bit signed divide
    UDIVWrr,  // 32-bit unsigned divide
    UDIVXrr,  // 64-bit unsigned divide
    MSUBWrrr, // 32-bit multiply-subtract: Wd = Wa - Wn * Wm
    MSUBXrrr, // 64-bit multiply-subtract: Xd = Xa - Xn * Xm

    // Comparison
    CMPWrr,   // 32-bit compare (SUBS WZR, Wn, Wm)
    CMPXrr,   // 64-bit compare (SUBS XZR, Xn, Xm)
    CMPWri,   // 32-bit compare, reg+imm12
    CMPXri,   // 64-bit compare, reg+imm12
    CSETWcc,  // Conditional set (CSET Wd, cc)

    // Move / materialization
    MOVWrr,   // 32-bit register move (ORR Wd, WZR, Wm)
    MOVXrr,   // 64-bit register move (ORR Xd, XZR, Xm)
    MOVZWi,   // 32-bit move wide with zero
    MOVZXi,   // 64-bit move wide with zero
    MOVNWi,   // 32-bit move wide with NOT
    MOVNXi,   // 64-bit move wide with NOT
    MOVKXi,   // Move keep (16-bit insert)
    FMOVSri,  // Move immediate to FPR (32-bit)
    FMOVDri,  // Move immediate to FPR (64-bit)

    // Load / store
    LDRWui,   // Load 32-bit, unsigned offset
    LDRXui,   // Load 64-bit, unsigned offset
    LDRSui,   // Load 32-bit float, unsigned offset
    LDRDui,   // Load 64-bit float, unsigned offset
    STRWui,   // Store 32-bit, unsigned offset
    STRXui,   // Store 64-bit, unsigned offset
    STRSui,   // Store 32-bit float, unsigned offset
    STRDui,   // Store 64-bit float, unsigned offset

    // Branch
    B,        // Unconditional branch
    Bcc,      // Conditional branch (B.cond)
    BL,       // Branch with link (call)
    BLR,      // Branch with link to register (indirect call)
    RET,      // Return (BR X30)

    // Shift (register forms)
    LSLVWr,   // 32-bit logical shift left by register
    LSLVXr,   // 64-bit logical shift left by register
    LSRVWr,   // 32-bit logical shift right by register
    LSRVXr,   // 64-bit logical shift right by register
    ASRVWr,   // 32-bit arithmetic shift right by register
    ASRVXr,   // 64-bit arithmetic shift right by register

    // Shift (immediate forms - encoded via UBFM/SBFM)
    LSLWi,    // 32-bit LSL immediate (UBFM alias)
    LSLXi,    // 64-bit LSL immediate
    LSRWi,    // 32-bit LSR immediate (UBFM alias)
    LSRXi,    // 64-bit LSR immediate
    ASRWi,    // 32-bit ASR immediate (SBFM alias)
    ASRXi,    // 64-bit ASR immediate

    // Logical (register forms)
    ANDWrr,   // 32-bit bitwise AND
    ANDXrr,   // 64-bit bitwise AND
    ORRWrr,   // 32-bit bitwise OR
    ORRXrr,   // 64-bit bitwise OR
    EORWrr,   // 32-bit bitwise XOR
    EORXrr,   // 64-bit bitwise XOR
    BICWrr,   // 32-bit AND-NOT (bit clear)
    BICXrr,   // 64-bit AND-NOT
    ORNWrr,   // 32-bit OR-NOT
    ORNXrr,   // 64-bit OR-NOT

    // Logical (immediate forms - uses bitmask immediate encoding)
    ANDWri,   // 32-bit AND immediate
    ANDXri,   // 64-bit AND immediate
    ORRWri,   // 32-bit OR immediate
    ORRXri,   // 64-bit OR immediate
    EORWri,   // 32-bit XOR immediate
    EORXri,   // 64-bit XOR immediate

    // Conditional select
    CSELWr,   // 32-bit CSEL Wd, Wn, Wm, cond
    CSELXr,   // 64-bit CSEL Xd, Xn, Xm, cond
    CSINCWr,  // 32-bit CSINC (conditional increment)
    CSINCXr,  // 64-bit CSINC
    CSINVWr,  // 32-bit CSINV (conditional invert)
    CSINVXr,  // 64-bit CSINV
    CSNEGWr,  // 32-bit CSNEG (conditional negate)
    CSNEGXr,  // 64-bit CSNEG

    // Extension instructions
    SXTBWr,   // Sign-extend byte to 32-bit (SBFM Wd, Wn, #0, #7)
    SXTBXr,   // Sign-extend byte to 64-bit (SBFM Xd, Xn, #0, #7)
    SXTHWr,   // Sign-extend halfword to 32-bit (SBFM Wd, Wn, #0, #15)
    SXTHXr,   // Sign-extend halfword to 64-bit
    SXTWXr,   // Sign-extend word to 64-bit (SBFM Xd, Wn, #0, #31)
    UXTBWr,   // Zero-extend byte to 32-bit (AND Wd, Wn, #0xFF)
    UXTHWr,   // Zero-extend halfword to 32-bit (AND Wd, Wn, #0xFFFF)

    // Bitfield operations
    UBFMWri,  // Unsigned bitfield move, 32-bit
    UBFMXri,  // Unsigned bitfield move, 64-bit
    SBFMWri,  // Signed bitfield move, 32-bit
    SBFMXri,  // Signed bitfield move, 64-bit
    BFMWri,   // Bitfield move (insert), 32-bit
    BFMXri,   // Bitfield move (insert), 64-bit

    // Unary integer operations
    NEGWr,    // 32-bit negate (SUB Wd, WZR, Wm)
    NEGXr,    // 64-bit negate (SUB Xd, XZR, Xm)
    MVNWr,    // 32-bit bitwise NOT (ORN Wd, WZR, Wm)
    MVNXr,    // 64-bit bitwise NOT (ORN Xd, XZR, Xm)

    // Floating-point unary operations
    FNEGSr,   // 32-bit float negate (FNEG Sd, Sn)
    FNEGDr,   // 64-bit float negate (FNEG Dd, Dn)

    // Floating-point arithmetic
    FADDSrr,  // 32-bit float add
    FADDDrr,  // 64-bit float add
    FSUBSrr,  // 32-bit float sub
    FSUBDrr,  // 64-bit float sub
    FMULSrr,  // 32-bit float mul
    FMULDrr,  // 64-bit float mul
    FDIVSrr,  // 32-bit float div
    FDIVDrr,  // 64-bit float div

    // Floating-point comparison
    FCMPSrr,  // 32-bit float compare
    FCMPDrr,  // 64-bit float compare

    // Floating-point conversion
    FCVTZSWr, // Float to signed int, 32-bit result (FCVTZS Wd, Sn/Dn)
    FCVTZSXr, // Float to signed int, 64-bit result (FCVTZS Xd, Sn/Dn)
    SCVTFSWr, // Signed int to float, from 32-bit (SCVTF Sd, Wn)
    SCVTFDWr, // Signed int to double, from 32-bit (SCVTF Dd, Wn)
    SCVTFSXr, // Signed int to float, from 64-bit (SCVTF Sd, Xn)
    SCVTFDXr, // Signed int to double, from 64-bit (SCVTF Dd, Xn)

    // Address generation
    ADRP,     // PC-relative page address (ADRP Xd, #page)
    ADDXriPCRel, // Add page offset for global (ADD Xd, Xn, #pageoff)

    // Stack / frame operations
    ADDXriSP, // SP-relative add for stack slot address

    // Load/store with register offset
    LDRWro,   // Load 32-bit, base + register offset
    LDRXro,   // Load 64-bit, base + register offset
    STRWro,   // Store 32-bit, base + register offset
    STRXro,   // Store 64-bit, base + register offset

    // Pseudo-ops (expanded later by frame lowering / regalloc)
    COPY,     // Pseudo: reg-to-reg copy (resolved by regalloc)
    PHI,      // Pseudo: SSA phi (eliminated before regalloc)
}

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

            // Memory
            Opcode::Load { ty } => self.select_load(ty.clone(), inst, block)?,
            Opcode::Store => self.select_store(inst, block)?,
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Select integer constant materialization.
    /// Small values (0..65535): MOVZ
    /// Negative small values: MOVN
    /// Large values: MOVZ + MOVK sequence (TODO: full sequence)
    fn select_iconst(&mut self, ty: Type, imm: i64, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = inst.results.first().expect("Iconst must have a result");

        if imm >= 0 && imm <= 0xFFFF {
            // Simple MOVZ
            let opc = if Self::is_32bit(&ty) {
                AArch64Opcode::MOVZWi
            } else {
                AArch64Opcode::MOVZXi
            };
            self.func.push_inst(
                block,
                MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::Imm(imm)]),
            );
        } else if imm < 0 && imm >= -0x10000 {
            // MOVN for small negative values
            let opc = if Self::is_32bit(&ty) {
                AArch64Opcode::MOVNWi
            } else {
                AArch64Opcode::MOVNXi
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
                AArch64Opcode::MOVZWi
            } else {
                AArch64Opcode::MOVZXi
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
                            AArch64Opcode::MOVKXi,
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
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = inst.results.first().expect("Fconst must have a result");

        let opc = match ty {
            Type::F32 => AArch64Opcode::FMOVSri,
            Type::F64 => AArch64Opcode::FMOVDri,
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
        assert!(inst.args.len() == 1, "copy must have exactly 1 arg");
        assert!(!inst.results.is_empty(), "copy must have a result");

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
            AArch64Opcode::MOVWrr
        } else {
            AArch64Opcode::MOVXrr
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
        assert!(inst.args.len() >= 2, "BinOp must have at least 2 args");
        assert!(!inst.results.is_empty(), "BinOp must have a result");

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
            (AArch64BinOp::Add, true) => AArch64Opcode::ADDWrr,
            (AArch64BinOp::Add, false) => AArch64Opcode::ADDXrr,
            (AArch64BinOp::Sub, true) => AArch64Opcode::SUBWrr,
            (AArch64BinOp::Sub, false) => AArch64Opcode::SUBXrr,
            (AArch64BinOp::Mul, true) => AArch64Opcode::MULWrrr,
            (AArch64BinOp::Mul, false) => AArch64Opcode::MULXrrr,
            (AArch64BinOp::Sdiv, true) => AArch64Opcode::SDIVWrr,
            (AArch64BinOp::Sdiv, false) => AArch64Opcode::SDIVXrr,
            (AArch64BinOp::Udiv, true) => AArch64Opcode::UDIVWrr,
            (AArch64BinOp::Udiv, false) => AArch64Opcode::UDIVXrr,
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
        assert!(inst.args.len() >= 2, "Remainder must have at least 2 args");
        assert!(!inst.results.is_empty(), "Remainder must have a result");

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
            (true, true) => AArch64Opcode::SDIVWrr,
            (true, false) => AArch64Opcode::SDIVXrr,
            (false, true) => AArch64Opcode::UDIVWrr,
            (false, false) => AArch64Opcode::UDIVXrr,
        };
        self.func.push_inst(
            block,
            MachInst::new(div_opc, vec![MachOperand::VReg(tmp), lhs.clone(), rhs.clone()]),
        );

        // Step 2: result = a - tmp * b  (MSUB Rd, Rn, Rm, Ra)
        // Operands: [dst, Rn=tmp, Rm=b, Ra=a]
        let dst = self.new_vreg(class);
        let msub_opc = if is_32 {
            AArch64Opcode::MSUBWrrr
        } else {
            AArch64Opcode::MSUBXrrr
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
        assert!(!inst.args.is_empty(), "Unary op must have 1 arg");
        assert!(!inst.results.is_empty(), "Unary op must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        let opc = match (op, is_32) {
            (AArch64IntUnaryOp::Neg, true) => AArch64Opcode::NEGWr,
            (AArch64IntUnaryOp::Neg, false) => AArch64Opcode::NEGXr,
            (AArch64IntUnaryOp::Mvn, true) => AArch64Opcode::MVNWr,
            (AArch64IntUnaryOp::Mvn, false) => AArch64Opcode::MVNXr,
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
        assert!(!inst.args.is_empty(), "FP unary op must have 1 arg");
        assert!(!inst.results.is_empty(), "FP unary op must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_f32 = matches!(ty, Type::F32);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        let opc = if is_f32 {
            AArch64Opcode::FNEGSr
        } else {
            AArch64Opcode::FNEGDr
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
        assert!(inst.args.len() >= 2, "Icmp must have 2 args");
        assert!(!inst.results.is_empty(), "Icmp must have a result");

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // CMP (subtract setting flags, discard result)
        let cmp_opc = if is_32 {
            AArch64Opcode::CMPWrr
        } else {
            AArch64Opcode::CMPXrr
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
                AArch64Opcode::CSETWcc,
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
                AArch64Opcode::CMPXri,
                vec![cond_op, MachOperand::Imm(0)],
            ),
        );

        // B.NE then_block (condition was nonzero = true)
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::Bcc,
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
                        AArch64Opcode::MOVWrr
                    } else {
                        AArch64Opcode::MOVXrr
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
                    // Large aggregate return via X8 pointer: store to [X8]
                    // TODO: Implement indirect return lowering
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }

        // Emit RET (branches to LR)
        self.func.push_inst(
            block,
            MachInst::new(AArch64Opcode::RET, vec![MachOperand::PReg(gpr::LR)]),
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
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = arg_types[i].clone();
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::MOVWrr
                    } else {
                        AArch64Opcode::MOVXrr
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // STR to [SP + offset]
                    let ty = arg_types[i].clone();
                    let opc = if Self::is_32bit(&ty) {
                        AArch64Opcode::STRWui
                    } else {
                        AArch64Opcode::STRXui
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
                    // Large aggregate: store to memory, pass pointer in register
                    // TODO: Allocate stack space, store aggregate, pass pointer
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MOVXrr,
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
                AArch64Opcode::BL,
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
                        AArch64Opcode::MOVWrr
                    } else {
                        AArch64Opcode::MOVXrr
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::PReg(*preg)]),
                    );
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg: _ } => {
                    // Large aggregate returned via X8 pointer
                    // TODO: Load from sret pointer
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
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
    // Load / Store
    // -----------------------------------------------------------------------

    /// Select a memory load.
    fn select_load(&mut self, ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "Load must have address arg");
        assert!(!inst.results.is_empty(), "Load must have result");

        let addr_val = inst.args[0];
        let result_val = inst.results[0];
        let addr = self.use_value(&addr_val)?;

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let opc = match ty {
            Type::I32 => AArch64Opcode::LDRWui,
            Type::I64 => AArch64Opcode::LDRXui,
            Type::F32 => AArch64Opcode::LDRSui,
            Type::F64 => AArch64Opcode::LDRDui,
            // Smaller types need zero/sign-extending loads (LDRB, LDRH, LDRSB, etc.)
            // TODO: Implement extending loads
            _ => AArch64Opcode::LDRXui,
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
        assert!(
            inst.args.len() >= 2,
            "Store must have value and address args"
        );

        let value_val = inst.args[0];
        let addr_val = inst.args[1];

        let src = self.use_value(&value_val)?;
        let addr = self.use_value(&addr_val)?;
        let ty = self.value_type(&value_val);

        let opc = match ty {
            Type::I32 => AArch64Opcode::STRWui,
            Type::I64 => AArch64Opcode::STRXui,
            Type::F32 => AArch64Opcode::STRSui,
            Type::F64 => AArch64Opcode::STRDui,
            // TODO: Implement truncating stores (STRB, STRH)
            _ => AArch64Opcode::STRXui,
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![src, addr, MachOperand::Imm(0)]),
        );
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
                            AArch64Opcode::COPY,
                            vec![MachOperand::VReg(vreg), MachOperand::PReg(*preg)],
                        ),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // Load from stack: LDR vreg, [SP, #offset]
                    let opc = if Self::is_32bit(ty) {
                        AArch64Opcode::LDRWui
                    } else {
                        AArch64Opcode::LDRXui
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
                            AArch64Opcode::COPY,
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
        assert!(inst.args.len() >= 2, "Shift must have 2 args (value, amount)");
        assert!(!inst.results.is_empty(), "Shift must have a result");

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
                (AArch64ShiftOp::Lsl, true) => AArch64Opcode::LSLWi,
                (AArch64ShiftOp::Lsl, false) => AArch64Opcode::LSLXi,
                (AArch64ShiftOp::Lsr, true) => AArch64Opcode::LSRWi,
                (AArch64ShiftOp::Lsr, false) => AArch64Opcode::LSRXi,
                (AArch64ShiftOp::Asr, true) => AArch64Opcode::ASRWi,
                (AArch64ShiftOp::Asr, false) => AArch64Opcode::ASRXi,
            };
            self.func.push_inst(
                block,
                MachInst::new(opc, vec![MachOperand::VReg(dst), src, amt]),
            );
        } else {
            // Register shift form
            let opc = match (op, is_32) {
                (AArch64ShiftOp::Lsl, true) => AArch64Opcode::LSLVWr,
                (AArch64ShiftOp::Lsl, false) => AArch64Opcode::LSLVXr,
                (AArch64ShiftOp::Lsr, true) => AArch64Opcode::LSRVWr,
                (AArch64ShiftOp::Lsr, false) => AArch64Opcode::LSRVXr,
                (AArch64ShiftOp::Asr, true) => AArch64Opcode::ASRVWr,
                (AArch64ShiftOp::Asr, false) => AArch64Opcode::ASRVXr,
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
        assert!(inst.args.len() >= 2, "Logic op must have 2 args");
        assert!(!inst.results.is_empty(), "Logic op must have a result");

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
            (AArch64LogicOp::And, true) => AArch64Opcode::ANDWrr,
            (AArch64LogicOp::And, false) => AArch64Opcode::ANDXrr,
            (AArch64LogicOp::Orr, true) => AArch64Opcode::ORRWrr,
            (AArch64LogicOp::Orr, false) => AArch64Opcode::ORRXrr,
            (AArch64LogicOp::Eor, true) => AArch64Opcode::EORWrr,
            (AArch64LogicOp::Eor, false) => AArch64Opcode::EORXrr,
            (AArch64LogicOp::Bic, true) => AArch64Opcode::BICWrr,
            (AArch64LogicOp::Bic, false) => AArch64Opcode::BICXrr,
            (AArch64LogicOp::Orn, true) => AArch64Opcode::ORNWrr,
            (AArch64LogicOp::Orn, false) => AArch64Opcode::ORNXrr,
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
        assert!(!inst.args.is_empty(), "Extend must have 1 arg");
        assert!(!inst.results.is_empty(), "Extend must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);
        let src = self.use_value(&src_val)?;

        let to_64 = !Self::is_32bit(to_ty);
        let to_ty_owned = to_ty.clone();

        let opc = if signed {
            match (from_ty, to_64) {
                (Type::I8, false) => AArch64Opcode::SXTBWr,
                (Type::I8, true) => AArch64Opcode::SXTBXr,
                (Type::I16, false) => AArch64Opcode::SXTHWr,
                (Type::I16, true) => AArch64Opcode::SXTHXr,
                (Type::I32, true) => AArch64Opcode::SXTWXr,
                _ => {
                    // Same-width or unsupported: just copy
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::COPY,
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
                Type::I8 => AArch64Opcode::UXTBWr,
                Type::I16 => AArch64Opcode::UXTHWr,
                Type::I32 if to_64 => {
                    // UXTW: zero-extend W to X. On AArch64, writing a W register
                    // implicitly zero-extends to X, so a MOV Wd, Wn suffices.
                    // But we need an explicit instruction for tracking.
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            AArch64Opcode::MOVWrr,
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
                            AArch64Opcode::COPY,
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
        assert!(!inst.args.is_empty(), "BitfieldExtract must have 1 arg");
        assert!(!inst.results.is_empty(), "BitfieldExtract must have a result");

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
            (false, true) => AArch64Opcode::UBFMWri,
            (false, false) => AArch64Opcode::UBFMXri,
            (true, true) => AArch64Opcode::SBFMWri,
            (true, false) => AArch64Opcode::SBFMXri,
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
        assert!(
            inst.args.len() >= 2,
            "BitfieldInsert must have 2 args (dst, src)"
        );
        assert!(!inst.results.is_empty(), "BitfieldInsert must have a result");

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
                AArch64Opcode::COPY,
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
            AArch64Opcode::BFMWri
        } else {
            AArch64Opcode::BFMXri
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
        assert!(
            inst.args.len() >= 3,
            "Select must have 3 args (cond_val, true_val, false_val)"
        );
        assert!(!inst.results.is_empty(), "Select must have a result");

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
                AArch64Opcode::CMPWri,
                vec![cond_op, MachOperand::Imm(0)],
            ),
        );

        // CSEL based on the IntCC condition
        let cc = AArch64CC::from_intcc(cond);
        let opc = if is_32 {
            AArch64Opcode::CSELWr
        } else {
            AArch64Opcode::CSELXr
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
        assert!(inst.args.len() >= 2, "FP binop must have 2 args");
        assert!(!inst.results.is_empty(), "FP binop must have a result");

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
            (AArch64FpBinOp::Fadd, true) => AArch64Opcode::FADDSrr,
            (AArch64FpBinOp::Fadd, false) => AArch64Opcode::FADDDrr,
            (AArch64FpBinOp::Fsub, true) => AArch64Opcode::FSUBSrr,
            (AArch64FpBinOp::Fsub, false) => AArch64Opcode::FSUBDrr,
            (AArch64FpBinOp::Fmul, true) => AArch64Opcode::FMULSrr,
            (AArch64FpBinOp::Fmul, false) => AArch64Opcode::FMULDrr,
            (AArch64FpBinOp::Fdiv, true) => AArch64Opcode::FDIVSrr,
            (AArch64FpBinOp::Fdiv, false) => AArch64Opcode::FDIVDrr,
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
        assert!(inst.args.len() >= 2, "Fcmp must have 2 args");
        assert!(!inst.results.is_empty(), "Fcmp must have a result");

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_f32 = matches!(ty, Type::F32);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // FCMP sets NZCV
        let cmp_opc = if is_f32 {
            AArch64Opcode::FCMPSrr
        } else {
            AArch64Opcode::FCMPDrr
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
                AArch64Opcode::CSETWcc,
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
        assert!(!inst.args.is_empty(), "FcvtToInt must have 1 arg");
        assert!(!inst.results.is_empty(), "FcvtToInt must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        // Select based on destination integer width. The source float
        // precision is encoded in the operand register class.
        let opc = if Self::is_32bit(&dst_ty) {
            AArch64Opcode::FCVTZSWr
        } else {
            AArch64Opcode::FCVTZSXr
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
        assert!(!inst.args.is_empty(), "FcvtFromInt must have 1 arg");
        assert!(!inst.results.is_empty(), "FcvtFromInt must have a result");

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
            (true, true) => AArch64Opcode::SCVTFSWr,    // SCVTF Sd, Wn
            (true, false) => AArch64Opcode::SCVTFDWr,   // SCVTF Dd, Wn
            (false, true) => AArch64Opcode::SCVTFSXr,   // SCVTF Sd, Xn
            (false, false) => AArch64Opcode::SCVTFDXr,  // SCVTF Dd, Xn
        };

        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), src]),
        );

        self.define_value(result_val, MachOperand::VReg(dst), dst_ty);
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
        assert!(!inst.results.is_empty(), "GlobalRef must have a result");

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        // ADRP Xd, symbol@PAGE
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::ADRP,
                vec![MachOperand::VReg(dst), MachOperand::Symbol(name.to_string())],
            ),
        );

        // ADD Xd, Xd, symbol@PAGEOFF
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::ADDXriPCRel,
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
        assert!(!inst.results.is_empty(), "StackAddr must have a result");

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::ADDXriSP,
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::COPY);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::ADDWrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::SUBXrr);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CMPWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSETWcc);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CMPWri);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Bcc);
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
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::RET);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MOVZWi);
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
        assert_eq!(mfunc.blocks[&entry].insts[0].opcode, AArch64Opcode::MOVNXi);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::ADDWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MOVWrr);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::RET);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LSLVWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::LSRVXr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::ASRVWr);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LSLVWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::ANDWrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::ORRXrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::EORWrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::BICXrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::ORNWrr);
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
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::SXTBWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::SXTHXr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::SXTWXr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UXTBWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UXTHWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MOVWrr);
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
        assert_eq!(inst.opcode, AArch64Opcode::UBFMWri);
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
        assert_eq!(inst.opcode, AArch64Opcode::SBFMXri);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::BFMWri);
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
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CMPWri);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CSELWr);
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
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CSELXr);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MOVZXi);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Imm(0x0004));

        // Should have MOVK for bits [16:31], [32:47], [48:63]
        let movk_count = mblock.insts.iter().filter(|i| i.opcode == AArch64Opcode::MOVKXi).count();
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MOVZWi);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Imm(0x0002));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::MOVKXi);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FADDDrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FSUBSrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FMULDrr);
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
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FDIVSrr);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FCMPDrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSETWcc);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FCVTZSWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::SCVTFSWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::SCVTFDXr);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::ADRP);
        assert_eq!(mblock.insts[0].operands[1], MachOperand::Symbol("my_global".to_string()));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::ADDXriPCRel);
        assert_eq!(mblock.insts[1].operands[2], MachOperand::Symbol("my_global".to_string()));
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::ADDXriSP);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LSLVWr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MOVZWi);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::ANDWrr);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::MOVWrr);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::RET);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FADDDrr);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::RET);
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
        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::BL);
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
        let bcc_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bcc);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::UDIVWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MSUBWrrr);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SDIVWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MSUBWrrr);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::UDIVXrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MSUBXrrr);
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
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SDIVXrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MSUBXrrr);
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
        assert_eq!(div_inst.opcode, AArch64Opcode::SDIVWrr);
        let div_dst = &div_inst.operands[0]; // tmp vreg
        let dividend_op = &div_inst.operands[1]; // dividend
        let divisor_op = &div_inst.operands[2]; // divisor

        // MSUB dst, Rn=tmp, Rm=divisor, Ra=dividend
        assert_eq!(msub_inst.opcode, AArch64Opcode::MSUBWrrr);
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
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::NEGWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::NEGXr);
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
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::MVNWr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MVNXr);
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
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FNEGDr);
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
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FNEGSr);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::NEGWr);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::MOVWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::RET);
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
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FNEGDr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::RET);
    }
}

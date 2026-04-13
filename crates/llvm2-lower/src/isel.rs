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
use crate::instructions::{Block, Instruction, IntCC, Opcode, Value};
use crate::types::Type;

// Import canonical register types from llvm2-ir.
use llvm2_ir::regs::{RegClass, VReg};

// ---------------------------------------------------------------------------
// Register model helpers
// ---------------------------------------------------------------------------

/// Derive the register class for a given LIR type.
fn reg_class_for_type(ty: Type) -> RegClass {
    match ty {
        Type::B1 | Type::I8 | Type::I16 | Type::I32 => RegClass::Gpr32,
        Type::I64 | Type::I128 => RegClass::Gpr64,
        Type::F32 => RegClass::Fpr32,
        Type::F64 => RegClass::Fpr64,
    }
}

// ---------------------------------------------------------------------------
// AArch64 opcode enumeration (subset for scaffold)
// ---------------------------------------------------------------------------

/// AArch64 machine opcodes. This is a scaffold subset; the full set will be
/// expanded by other techleads to cover all ~25 tMIR instructions plus
/// pseudo-ops.
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

    // Floating-point arithmetic
    FADDSrr,  // 32-bit float add
    FADDDrr,  // 64-bit float add
    FSUBSrr,  // 32-bit float sub
    FSUBDrr,  // 64-bit float sub
    FMULSrr,  // 32-bit float mul
    FMULDrr,  // 64-bit float mul
    FDIVSrr,  // 32-bit float div
    FDIVDrr,  // 64-bit float div

    // Pseudo-ops (expanded later by frame lowering / regalloc)
    COPY,     // Pseudo: reg-to-reg copy (resolved by regalloc)
    PHI,      // Pseudo: SSA phi (eliminated before regalloc)
}

// ---------------------------------------------------------------------------
// Machine operand
// ---------------------------------------------------------------------------

/// Operand of a machine instruction (post-isel, pre-regalloc).
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
}

/// AArch64 condition codes (NZCV-based).
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
}

// ---------------------------------------------------------------------------
// Machine instruction
// ---------------------------------------------------------------------------

/// A single machine instruction (pre-regalloc).
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
// Machine basic block
// ---------------------------------------------------------------------------

/// A basic block of machine instructions.
#[derive(Debug, Clone, Default)]
pub struct MachBlock {
    pub insts: Vec<MachInst>,
    pub successors: Vec<Block>,
}

// ---------------------------------------------------------------------------
// Machine function
// ---------------------------------------------------------------------------

/// A function after instruction selection, containing MachInsts with VRegs.
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
    fn use_value(&self, val: &Value) -> MachOperand {
        self.value_map
            .get(val)
            .cloned()
            .unwrap_or_else(|| panic!("Value {:?} not defined before use", val))
    }

    /// Get the type of a tMIR Value.
    fn value_type(&self, val: &Value) -> Type {
        self.value_types
            .get(val)
            .copied()
            .unwrap_or(Type::I64) // default to i64 if unknown
    }

    /// Determine if a type is 32-bit (uses W registers).
    fn is_32bit(ty: Type) -> bool {
        matches!(ty, Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::F32)
    }

    // -----------------------------------------------------------------------
    // Top-level selection
    // -----------------------------------------------------------------------

    /// Select all instructions in a block.
    pub fn select_block(&mut self, block: Block, instructions: &[Instruction]) {
        self.func.ensure_block(block);
        for inst in instructions {
            self.select_instruction(inst, block);
        }
    }

    /// Select a single tMIR instruction, emitting MachInsts into the block.
    fn select_instruction(&mut self, inst: &Instruction, block: Block) {
        match &inst.opcode {
            // Constants
            Opcode::Iconst { ty, imm } => self.select_iconst(*ty, *imm, inst, block),
            Opcode::Fconst { ty, imm } => self.select_fconst(*ty, *imm, inst, block),

            // Arithmetic
            Opcode::Iadd => self.select_binop(AArch64BinOp::Add, inst, block),
            Opcode::Isub => self.select_binop(AArch64BinOp::Sub, inst, block),
            Opcode::Imul => self.select_binop(AArch64BinOp::Mul, inst, block),
            Opcode::Sdiv => self.select_binop(AArch64BinOp::Sdiv, inst, block),
            Opcode::Udiv => self.select_binop(AArch64BinOp::Udiv, inst, block),

            // Comparison
            Opcode::Icmp { cond } => self.select_cmp(*cond, inst, block),

            // Control flow
            Opcode::Jump { dest } => self.select_jump(*dest, block),
            Opcode::Brif { then_dest, else_dest, .. } => {
                self.select_brif(inst, *then_dest, *else_dest, block);
            }
            Opcode::Return => self.select_return(inst, block),

            // Memory
            Opcode::Load { ty } => self.select_load(*ty, inst, block),
            Opcode::Store => self.select_store(inst, block),
        }
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Select integer constant materialization.
    /// Small values (0..65535): MOVZ
    /// Negative small values: MOVN
    /// Large values: MOVZ + MOVK sequence (TODO: full sequence)
    fn select_iconst(&mut self, ty: Type, imm: i64, inst: &Instruction, block: Block) {
        let class = reg_class_for_type(ty);
        let dst = self.new_vreg(class);
        let result = inst.results.first().expect("Iconst must have a result");

        if imm >= 0 && imm <= 0xFFFF {
            // Simple MOVZ
            let opc = if Self::is_32bit(ty) {
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
            let opc = if Self::is_32bit(ty) {
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
            let opc_z = if Self::is_32bit(ty) {
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
            let chunks = if Self::is_32bit(ty) { 2 } else { 4 };
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
    }

    /// Select float constant materialization.
    fn select_fconst(&mut self, ty: Type, imm: f64, inst: &Instruction, block: Block) {
        let class = reg_class_for_type(ty);
        let dst = self.new_vreg(class);
        let result = inst.results.first().expect("Fconst must have a result");

        let opc = match ty {
            Type::F32 => AArch64Opcode::FMOVSri,
            Type::F64 => AArch64Opcode::FMOVDri,
            _ => unreachable!("Fconst with non-float type"),
        };

        // FMOV immediate encoding is limited to a small set of values.
        // For the scaffold, emit FMOV for all; legalization will handle
        // values outside the 8-bit FP immediate range via constant pool.
        self.func.push_inst(
            block,
            MachInst::new(opc, vec![MachOperand::VReg(dst), MachOperand::FImm(imm)]),
        );

        self.define_value(*result, MachOperand::VReg(dst), ty);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    /// Internal enum for selecting the right opcode variant.
    #[allow(dead_code)]
    fn select_binop(&mut self, op: AArch64BinOp, inst: &Instruction, block: Block) {
        assert!(inst.args.len() >= 2, "BinOp must have at least 2 args");
        assert!(!inst.results.is_empty(), "BinOp must have a result");

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        // Determine type from LHS
        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(ty);

        let class = reg_class_for_type(ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val);
        let rhs = self.use_value(&rhs_val);

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
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    /// Select integer comparison: CMP + CSET.
    ///
    /// tMIR `Icmp(cond, lhs, rhs) -> bool_result` becomes:
    ///   CMP Wn/Xn, Wm/Xm        (sets NZCV flags)
    ///   CSET Wd, <cond_code>     (materializes flag into register)
    fn select_cmp(&mut self, cond: IntCC, inst: &Instruction, block: Block) {
        assert!(inst.args.len() >= 2, "Icmp must have 2 args");
        assert!(!inst.results.is_empty(), "Icmp must have a result");

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(ty);

        let lhs = self.use_value(&lhs_val);
        let rhs = self.use_value(&rhs_val);

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

        // CSET: materialize condition code into a W register
        let cc = AArch64CC::from_intcc(cond);
        let dst = self.new_vreg(RegClass::Gpr32);
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::CSETWcc,
                vec![MachOperand::VReg(dst), MachOperand::CondCode(cc)],
            ),
        );

        self.define_value(result_val, MachOperand::VReg(dst), Type::B1);
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    /// Select unconditional jump.
    fn select_jump(&mut self, dest: Block, block: Block) {
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
    ) {
        // The condition is the first argument
        let cond_val = inst.args[0];
        let cond_op = self.use_value(&cond_val);

        // CMP cond, #0 to set NZCV from boolean
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::CMPWri,
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
    }

    /// Select return. Moves return values into ABI-specified physical registers
    /// (X0/V0 etc.) and emits RET.
    fn select_return(&mut self, inst: &Instruction, block: Block) {
        // Classify return types to know which physical registers to use
        let ret_types: Vec<Type> = inst
            .args
            .iter()
            .map(|v| self.value_type(v))
            .collect();
        let ret_locs = AppleAArch64ABI::classify_returns(&ret_types);

        // Move each return value into its designated physical register
        for (i, (val, loc)) in inst.args.iter().zip(ret_locs.iter()).enumerate() {
            let src = self.use_value(val);
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = ret_types[i];
                    let opc = if Self::is_32bit(ty) {
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
                    unreachable!("Return values should not be on stack");
                }
            }
        }

        // Emit RET (branches to LR)
        self.func.push_inst(
            block,
            MachInst::new(AArch64Opcode::RET, vec![MachOperand::PReg(gpr::LR)]),
        );
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
    ) {
        // Classify argument locations
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params(&arg_types);

        // Move arguments to ABI locations
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val);
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = arg_types[i];
                    let opc = if Self::is_32bit(ty) {
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
                    let ty = arg_types[i];
                    let opc = if Self::is_32bit(ty) {
                        AArch64Opcode::STRWui
                    } else {
                        AArch64Opcode::STRXui
                    };
                    self.func.push_inst(
                        block,
                        MachInst::new(
                            opc,
                            vec![src, MachOperand::PReg(PReg(31)), MachOperand::Imm(*offset)],
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

        // Emit BL (direct call)
        // The callee name is stored as an immediate placeholder; the linker
        // will resolve it to a BRANCH26 relocation.
        self.func.push_inst(
            block,
            MachInst::new(
                AArch64Opcode::BL,
                vec![MachOperand::Imm(0)], // placeholder, resolved by relocation
            ),
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i];
                    let class = reg_class_for_type(ty);
                    let dst = self.new_vreg(class);
                    let opc = if Self::is_32bit(ty) {
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
                    let ty = result_types[i];
                    let class = reg_class_for_type(ty);
                    let dst = self.new_vreg(class);
                    self.define_value(*val, MachOperand::VReg(dst), ty);
                }
                ArgLocation::Stack { .. } => {
                    unreachable!("Return values should not be on stack");
                }
            }
        }

        // Mark callee_name usage to suppress unused warning
        let _ = callee_name;
    }

    // -----------------------------------------------------------------------
    // Load / Store
    // -----------------------------------------------------------------------

    /// Select a memory load.
    fn select_load(&mut self, ty: Type, inst: &Instruction, block: Block) {
        assert!(!inst.args.is_empty(), "Load must have address arg");
        assert!(!inst.results.is_empty(), "Load must have result");

        let addr_val = inst.args[0];
        let result_val = inst.results[0];
        let addr = self.use_value(&addr_val);

        let class = reg_class_for_type(ty);
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
    }

    /// Select a memory store.
    fn select_store(&mut self, inst: &Instruction, block: Block) {
        assert!(
            inst.args.len() >= 2,
            "Store must have value and address args"
        );

        let value_val = inst.args[0];
        let addr_val = inst.args[1];

        let src = self.use_value(&value_val);
        let addr = self.use_value(&addr_val);
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
    }

    // -----------------------------------------------------------------------
    // Function entry: lower formal arguments
    // -----------------------------------------------------------------------

    /// Lower formal arguments at function entry.
    ///
    /// For each parameter, emit a COPY from the physical register (or stack
    /// location) designated by the ABI to a fresh virtual register. This
    /// establishes the initial value_map for the function body.
    pub fn lower_formal_arguments(&mut self, sig: &Signature, entry_block: Block) {
        self.func.ensure_block(entry_block);

        let param_locs = AppleAArch64ABI::classify_params(&sig.params);

        // We need Value ids for the formal arguments. By convention, formal
        // args are Value(0), Value(1), ..., Value(n-1).
        for (i, (ty, loc)) in sig.params.iter().zip(param_locs.iter()).enumerate() {
            let val = Value(i as u32);
            let class = reg_class_for_type(*ty);
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
                    let opc = if Self::is_32bit(*ty) {
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
                                MachOperand::PReg(PReg(31)),
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

            self.define_value(val, MachOperand::VReg(vreg), *ty);
        }
    }

    // -----------------------------------------------------------------------
    // Finalize
    // -----------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Signature;
    use crate::instructions::{Block, Instruction, IntCC, Opcode, Value};

    /// Helper: create a simple isel for fn(i32, i32) -> i32.
    fn make_add_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry);
        (isel, entry)
    }

    #[test]
    fn lower_formal_arguments_two_i32() {
        let (isel, entry) = make_add_isel();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should have 2 COPY instructions (X0->vreg0, X1->vreg1)
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::COPY);

        // First COPY: dst=vreg0, src=X0
        assert!(matches!(mblock.insts[0].operands[1], MachOperand::PReg(PReg(0))));
        // Second COPY: dst=vreg1, src=X1
        assert!(matches!(mblock.insts[1].operands[1], MachOperand::PReg(PReg(1))));
    }

    #[test]
    fn select_iadd_i32() {
        let (mut isel, entry) = make_add_isel();

        // Iadd: Value(2) = Value(0) + Value(1)
        let add_inst = Instruction {
            opcode: Opcode::Iadd,
            args: vec![Value(0), Value(1)],
            results: vec![Value(2)],
        };
        isel.select_instruction(&add_inst, entry);

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 2 COPY (formal args) + 1 ADDWrr
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::ADDWrr);
    }

    #[test]
    fn select_isub_i64() {
        let sig = Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sub64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry);

        let sub_inst = Instruction {
            opcode: Opcode::Isub,
            args: vec![Value(0), Value(1)],
            results: vec![Value(2)],
        };
        isel.select_instruction(&sub_inst, entry);

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SUBXrr);
    }

    #[test]
    fn select_icmp_and_brif() {
        let (mut isel, entry) = make_add_isel();

        // Icmp: Value(2) = Value(0) == Value(1)
        let cmp_inst = Instruction {
            opcode: Opcode::Icmp {
                cond: IntCC::Equal,
            },
            args: vec![Value(0), Value(1)],
            results: vec![Value(2)],
        };
        isel.select_instruction(&cmp_inst, entry);

        // Brif: if Value(2) goto Block(1) else Block(2)
        let brif_inst = Instruction {
            opcode: Opcode::Brif {
                cond: Value(2),
                then_dest: Block(1),
                else_dest: Block(2),
            },
            args: vec![Value(2)],
            results: vec![],
        };
        isel.select_instruction(&brif_inst, entry);

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 2 COPY + 1 CMP + 1 CSET + 1 CMP (brif test) + 1 B.cc + 1 B
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CMPWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSETWcc);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CMPWri);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Bcc);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::B);

        // Block should have 2 successors
        assert_eq!(mblock.successors.len(), 2);
    }

    #[test]
    fn select_return_i32() {
        let (mut isel, entry) = make_add_isel();

        // Return Value(0)
        let ret_inst = Instruction {
            opcode: Opcode::Return,
            args: vec![Value(0)],
            results: vec![],
        };
        isel.select_instruction(&ret_inst, entry);

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 2 COPY (formal args) + 1 MOV (to X0) + 1 RET
        assert_eq!(mblock.insts.len(), 4);
        // The MOV should target X0
        assert!(matches!(
            mblock.insts[2].operands[0],
            MachOperand::PReg(PReg(0))
        ));
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::RET);
    }

    #[test]
    fn select_iconst_small() {
        let sig = Signature {
            params: vec![],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("const_fn".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let const_inst = Instruction {
            opcode: Opcode::Iconst {
                ty: Type::I32,
                imm: 42,
            },
            args: vec![],
            results: vec![Value(0)],
        };
        isel.select_instruction(&const_inst, entry);

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MOVZWi);
    }

    #[test]
    fn select_iconst_negative() {
        let sig = Signature {
            params: vec![],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("neg_const".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let const_inst = Instruction {
            opcode: Opcode::Iconst {
                ty: Type::I64,
                imm: -1,
            },
            args: vec![],
            results: vec![Value(0)],
        };
        isel.select_instruction(&const_inst, entry);

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MOVNXi);
    }

    #[test]
    fn condition_code_mapping() {
        assert_eq!(AArch64CC::from_intcc(IntCC::Equal), AArch64CC::EQ);
        assert_eq!(AArch64CC::from_intcc(IntCC::NotEqual), AArch64CC::NE);
        assert_eq!(AArch64CC::from_intcc(IntCC::SignedLessThan), AArch64CC::LT);
        assert_eq!(
            AArch64CC::from_intcc(IntCC::UnsignedGreaterThan),
            AArch64CC::HI
        );
    }

    #[test]
    fn full_add_function() {
        // End-to-end: fn add(i32, i32) -> i32 { return a + b; }
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry);

        // Iadd
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        );

        // Return
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        );

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected sequence:
        // COPY vreg0 <- X0
        // COPY vreg1 <- X1
        // ADDWrr vreg2 <- vreg0, vreg1
        // MOVWrr X0 <- vreg2
        // RET
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::COPY);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::ADDWrr);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::MOVWrr);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::RET);

        assert_eq!(mfunc.name, "add");
        assert_eq!(mfunc.next_vreg, 3); // 3 vregs allocated
    }
}

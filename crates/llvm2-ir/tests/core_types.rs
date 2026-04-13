// llvm2-ir integration tests for core types
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Integration tests that exercise the public API of llvm2-ir across modules.

use llvm2_ir::aarch64_regs;
use llvm2_ir::cc::{AArch64CC, FloatSize, OperandSize};
use llvm2_ir::function::{MachBlock, MachFunction, Signature, StackSlot, Type};
use llvm2_ir::inst::{AArch64Opcode, InstFlags, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{PReg, RegClass, SpecialReg, VReg, X0, X1, X29};
use llvm2_ir::types::{BlockId, FrameIdx, InstId, StackSlotId, VRegId};

// ===========================================================================
// Re-export smoke tests — verify crate root re-exports
// ===========================================================================

#[test]
fn crate_root_reexports_types() {
    // These types are re-exported at crate root — verify they're accessible.
    let _cc: llvm2_ir::AArch64CC = AArch64CC::EQ;
    let _os: llvm2_ir::OperandSize = OperandSize::S64;
    let _fs: llvm2_ir::FloatSize = FloatSize::F32;
    let _block: llvm2_ir::MachBlock = MachBlock::new();
    let _ty: llvm2_ir::Type = Type::I64;
    let _opcode: llvm2_ir::AArch64Opcode = AArch64Opcode::AddRR;
    let _flags: llvm2_ir::InstFlags = InstFlags::EMPTY;
    let _preg: llvm2_ir::PReg = X0;
    let _rc: llvm2_ir::RegClass = RegClass::Gpr64;
    let _vreg: llvm2_ir::VReg = VReg::new(0, RegClass::Gpr64);
    let _sp: llvm2_ir::SpecialReg = SpecialReg::SP;
    let _bid: llvm2_ir::BlockId = BlockId(0);
    let _iid: llvm2_ir::InstId = InstId(0);
    let _vid: llvm2_ir::VRegId = VRegId(0);
    let _sid: llvm2_ir::StackSlotId = StackSlotId(0);
    let _fi: llvm2_ir::FrameIdx = FrameIdx(0);
}

// ===========================================================================
// Cross-module integration: building a function end-to-end
// ===========================================================================

#[test]
fn build_simple_function_end_to_end() {
    // Build: fn add(i64, i64) -> i64 { return a + b; }
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("add".to_string(), sig);

    // Allocate virtual registers for params and result
    let v0 = func.alloc_vreg();
    let v1 = func.alloc_vreg();
    let v2 = func.alloc_vreg();

    // Create ADD instruction
    let add_inst = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::VReg(VReg::new(v2, RegClass::Gpr64)),
            MachOperand::VReg(VReg::new(v0, RegClass::Gpr64)),
            MachOperand::VReg(VReg::new(v1, RegClass::Gpr64)),
        ],
    );
    let add_id = func.push_inst(add_inst);
    func.append_inst(BlockId(0), add_id);

    // Create RET instruction
    let ret_inst = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret_inst);
    func.append_inst(BlockId(0), ret_id);

    // Verify structure
    assert_eq!(func.name, "add");
    assert_eq!(func.num_insts(), 2);
    assert_eq!(func.num_blocks(), 1);
    assert_eq!(func.block(BlockId(0)).len(), 2);

    // Verify instruction properties
    let add = func.inst(add_id);
    assert!(!add.is_branch());
    assert!(!add.is_return());
    assert!(!add.has_side_effects());
    assert_eq!(add.operands.len(), 3);

    let ret = func.inst(ret_id);
    assert!(ret.is_return());
    assert!(ret.is_terminator());
}

#[test]
fn build_branching_function() {
    // Build a function with a conditional branch:
    // bb0: cmp, bcond -> bb1 / bb2
    // bb1: nop, b -> bb2
    // bb2: ret
    let sig = Signature::new(vec![Type::I64], vec![]);
    let mut func = MachFunction::new("branch_fn".to_string(), sig);

    let bb1 = func.create_block();
    let bb2 = func.create_block();

    // bb0 -> bb1, bb0 -> bb2
    func.add_edge(BlockId(0), bb1);
    func.add_edge(BlockId(0), bb2);
    // bb1 -> bb2
    func.add_edge(bb1, bb2);

    // bb0: CmpRI + BCond
    let cmp = func.push_inst(MachInst::new(
        AArch64Opcode::CmpRI,
        vec![
            MachOperand::VReg(VReg::new(0, RegClass::Gpr64)),
            MachOperand::Imm(0),
        ],
    ));
    func.append_inst(BlockId(0), cmp);

    let bcond = func.push_inst(MachInst::new(
        AArch64Opcode::BCond,
        vec![MachOperand::Block(bb1)],
    ));
    func.append_inst(BlockId(0), bcond);

    // bb1: Nop + B
    let nop = func.push_inst(MachInst::new(AArch64Opcode::Nop, vec![]));
    func.append_inst(bb1, nop);

    let br = func.push_inst(MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Block(bb2)],
    ));
    func.append_inst(bb1, br);

    // bb2: Ret
    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
    func.append_inst(bb2, ret);

    // Verify CFG
    assert_eq!(func.num_blocks(), 3);
    assert_eq!(func.block(BlockId(0)).succs, vec![bb1, bb2]);
    assert_eq!(func.block(bb1).preds, vec![BlockId(0)]);
    assert_eq!(func.block(bb2).preds, vec![BlockId(0), bb1]);

    // Verify instruction flags
    assert!(func.inst(cmp).has_side_effects());
    assert!(func.inst(bcond).is_branch());
    assert!(func.inst(bcond).is_terminator());
    assert!(func.inst(nop).is_pseudo());
    assert!(func.inst(br).is_branch());
    assert!(func.inst(ret).is_return());
}

#[test]
fn build_function_with_stack_slots() {
    let sig = Signature::new(vec![Type::I64], vec![]);
    let mut func = MachFunction::new("spill".to_string(), sig);

    let ss0 = func.alloc_stack_slot(StackSlot::new(8, 8));
    let _ss1 = func.alloc_stack_slot(StackSlot::new(4, 4));

    // Use stack slot in a store instruction
    let str_inst = MachInst::new(
        AArch64Opcode::StrRI,
        vec![
            MachOperand::VReg(VReg::new(0, RegClass::Gpr64)),
            MachOperand::StackSlot(ss0),
        ],
    );
    let str_id = func.push_inst(str_inst);
    func.append_inst(BlockId(0), str_id);

    assert!(func.inst(str_id).writes_memory());
    assert!(func.inst(str_id).has_side_effects());
    assert_eq!(func.stack_slots.len(), 2);
    assert_eq!(func.stack_slots[0].size, 8);
    assert_eq!(func.stack_slots[1].size, 4);
}

// ===========================================================================
// RegClass.for_type consistency with Type properties
// ===========================================================================

#[test]
fn regclass_for_type_consistent_with_type_properties() {
    // Integer types should map to GPR classes
    let int_types = [Type::I8, Type::I16, Type::I32, Type::I64, Type::I128];
    for ty in &int_types {
        let rc = RegClass::for_type(*ty);
        assert!(
            rc == RegClass::Gpr32 || rc == RegClass::Gpr64,
            "{:?} should map to a GPR class", ty
        );
    }

    // Float types should map to FPR classes
    let float_types = [Type::F32, Type::F64];
    for ty in &float_types {
        let rc = RegClass::for_type(*ty);
        assert!(
            rc == RegClass::Fpr32 || rc == RegClass::Fpr64,
            "{:?} should map to an FPR class", ty
        );
    }

    // Ptr should be 64-bit
    assert_eq!(RegClass::for_type(Type::Ptr), RegClass::Gpr64);

    // B1 should be 32-bit (smallest GPR)
    assert_eq!(RegClass::for_type(Type::B1), RegClass::Gpr32);
}

// ===========================================================================
// MachOperand with all register types
// ===========================================================================

#[test]
fn operand_with_all_register_kinds() {
    // VReg operand
    let v = MachOperand::VReg(VReg::new(0, RegClass::Gpr64));
    assert!(v.is_vreg());

    // PReg operand
    let p = MachOperand::PReg(X0);
    assert!(p.is_preg());

    // Special register operand
    let sp = MachOperand::Special(SpecialReg::SP);
    assert!(!sp.is_preg()); // Special is a separate variant

    // MemOp with PReg base
    let mem = MachOperand::MemOp { base: X29, offset: -16 };
    assert!(mem.is_mem());
}

// ===========================================================================
// AArch64CC integration with operands
// ===========================================================================

#[test]
fn cc_encoding_is_valid_for_bcond() {
    // All condition codes should have valid 4-bit encodings
    let all_cc = [
        AArch64CC::EQ, AArch64CC::NE, AArch64CC::HS, AArch64CC::LO,
        AArch64CC::MI, AArch64CC::PL, AArch64CC::VS, AArch64CC::VC,
        AArch64CC::HI, AArch64CC::LS, AArch64CC::GE, AArch64CC::LT,
        AArch64CC::GT, AArch64CC::LE, AArch64CC::AL, AArch64CC::NV,
    ];
    for cc in &all_cc {
        assert!(cc.encoding() <= 0b1111);
        // Double inversion is identity
        assert_eq!(cc.invert().invert(), *cc);
    }
}

// ===========================================================================
// aarch64_regs integration tests
// ===========================================================================

#[test]
fn aarch64_regs_preg_encoding_scheme() {
    // Verify the encoding scheme documented in aarch64_regs.rs
    // GPR64: 0-31
    assert_eq!(aarch64_regs::X0.encoding(), 0);
    assert_eq!(aarch64_regs::X30.encoding(), 30);
    assert_eq!(aarch64_regs::SP.encoding(), 31);

    // GPR32: 32-63
    assert_eq!(aarch64_regs::W0.encoding(), 32);
    assert_eq!(aarch64_regs::W30.encoding(), 62);
    assert_eq!(aarch64_regs::WSP.encoding(), 63);

    // FPR128: 64-95
    assert_eq!(aarch64_regs::V0.encoding(), 64);
    assert_eq!(aarch64_regs::V31.encoding(), 95);

    // FPR64: 96-127
    assert_eq!(aarch64_regs::D0.encoding(), 96);
    assert_eq!(aarch64_regs::D31.encoding(), 127);

    // FPR32: 128-159
    assert_eq!(aarch64_regs::S0.encoding(), 128);
    assert_eq!(aarch64_regs::S31.encoding(), 159);

    // Special: 160-164
    assert_eq!(aarch64_regs::XZR.encoding(), 160);
    assert_eq!(aarch64_regs::WZR.encoding(), 161);
    assert_eq!(aarch64_regs::NZCV.encoding(), 162);
    assert_eq!(aarch64_regs::FPCR.encoding(), 163);
    assert_eq!(aarch64_regs::FPSR.encoding(), 164);

    // FPR16: 165-196
    assert_eq!(aarch64_regs::H0.encoding(), 165);
    assert_eq!(aarch64_regs::H31.encoding(), 196);

    // FPR8: 197-228
    assert_eq!(aarch64_regs::B0.encoding(), 197);
    assert_eq!(aarch64_regs::B31.encoding(), 228);
}

#[test]
fn aarch64_regs_hw_encoding_correctness() {
    // X0 -> hw 0, X30 -> hw 30, SP -> hw 31
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::X0), 0);
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::X30), 30);
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::SP), 31);

    // W0 -> hw 0, WSP -> hw 31
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::W0), 0);
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::WSP), 31);

    // XZR and WZR both encode as 31
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::XZR), 31);
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::WZR), 31);

    // V0 -> hw 0, V31 -> hw 31
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::V0), 0);
    assert_eq!(aarch64_regs::hw_encoding(aarch64_regs::V31), 31);
}

#[test]
fn aarch64_regs_allocatable_counts() {
    assert_eq!(aarch64_regs::ALLOCATABLE_GPRS.len(), 25);
    assert_eq!(aarch64_regs::ALLOCATABLE_FPRS.len(), 32);
    assert_eq!(aarch64_regs::CALLEE_SAVED_GPRS.len(), 10);
    assert_eq!(aarch64_regs::CALLEE_SAVED_FPRS.len(), 8);
    assert_eq!(aarch64_regs::CALLER_SAVED_GPRS.len(), 15);
    assert_eq!(aarch64_regs::CALLER_SAVED_FPRS.len(), 24);
    assert_eq!(aarch64_regs::ARG_GPRS.len(), 8);
    assert_eq!(aarch64_regs::ARG_FPRS.len(), 8);
    assert_eq!(aarch64_regs::RET_GPRS.len(), 8);
    assert_eq!(aarch64_regs::RET_FPRS.len(), 8);
    assert_eq!(aarch64_regs::TEMP_GPRS.len(), 7);
    assert_eq!(aarch64_regs::ALL_GPRS.len(), 31);
    assert_eq!(aarch64_regs::ALL_FPRS.len(), 32);
}

#[test]
fn aarch64_regs_callee_saved_function() {
    // GPR callee-saved: X19-X28
    for i in 19..=28 {
        assert!(
            aarch64_regs::is_callee_saved(aarch64_regs::PReg::new(i)),
            "X{} should be callee-saved", i
        );
    }
    // X0-X18 are NOT callee-saved (except it's more nuanced, but X0-X15 certainly not)
    for i in 0..=15 {
        assert!(
            !aarch64_regs::is_callee_saved(aarch64_regs::PReg::new(i)),
            "X{} should NOT be callee-saved", i
        );
    }

    // FPR callee-saved: V8-V15 (encoding 72-79)
    for i in 72..=79 {
        assert!(
            aarch64_regs::is_callee_saved(aarch64_regs::PReg::new(i)),
            "V{} should be callee-saved", i - 64
        );
    }
}

#[test]
fn aarch64_regs_gpr64_to_gpr32_roundtrip() {
    // X0-X30 -> W0-W30 -> X0-X30
    for i in 0..=30 {
        let x = aarch64_regs::PReg::new(i);
        let w = aarch64_regs::gpr64_to_gpr32(x).expect("should convert");
        let x_back = aarch64_regs::gpr32_to_gpr64(w).expect("should convert back");
        assert_eq!(x, x_back);
    }

    // SP -> WSP -> SP
    let wsp = aarch64_regs::gpr64_to_gpr32(aarch64_regs::SP).unwrap();
    assert_eq!(wsp, aarch64_regs::WSP);
    let sp_back = aarch64_regs::gpr32_to_gpr64(wsp).unwrap();
    assert_eq!(sp_back, aarch64_regs::SP);

    // XZR -> WZR -> XZR
    let wzr = aarch64_regs::gpr64_to_gpr32(aarch64_regs::XZR).unwrap();
    assert_eq!(wzr, aarch64_regs::WZR);
    let xzr_back = aarch64_regs::gpr32_to_gpr64(wzr).unwrap();
    assert_eq!(xzr_back, aarch64_regs::XZR);
}

#[test]
fn aarch64_regs_fpr_conversion_chain() {
    // V0 -> D0 -> V0
    let d0 = aarch64_regs::fpr128_to_fpr64(aarch64_regs::V0).unwrap();
    assert_eq!(d0, aarch64_regs::D0);
    let v0_back = aarch64_regs::fpr64_to_fpr128(d0).unwrap();
    assert_eq!(v0_back, aarch64_regs::V0);

    // V0 -> S0 -> V0
    let s0 = aarch64_regs::fpr128_to_fpr32(aarch64_regs::V0).unwrap();
    assert_eq!(s0, aarch64_regs::S0);
    let v0_back = aarch64_regs::fpr32_to_fpr128(s0).unwrap();
    assert_eq!(v0_back, aarch64_regs::V0);

    // V0 -> H0
    let h0 = aarch64_regs::fpr128_to_fpr16(aarch64_regs::V0).unwrap();
    assert_eq!(h0, aarch64_regs::H0);

    // V0 -> B0
    let b0 = aarch64_regs::fpr128_to_fpr8(aarch64_regs::V0).unwrap();
    assert_eq!(b0, aarch64_regs::B0);
}

#[test]
fn aarch64_regs_overlap() {
    // X0 and W0 overlap
    assert!(aarch64_regs::regs_overlap(aarch64_regs::X0, aarch64_regs::W0));
    // V0, D0, S0, H0, B0 all overlap
    assert!(aarch64_regs::regs_overlap(aarch64_regs::V0, aarch64_regs::D0));
    assert!(aarch64_regs::regs_overlap(aarch64_regs::V0, aarch64_regs::S0));
    assert!(aarch64_regs::regs_overlap(aarch64_regs::V0, aarch64_regs::H0));
    assert!(aarch64_regs::regs_overlap(aarch64_regs::V0, aarch64_regs::B0));
    assert!(aarch64_regs::regs_overlap(aarch64_regs::D0, aarch64_regs::S0));

    // X0 and X1 do NOT overlap
    assert!(!aarch64_regs::regs_overlap(aarch64_regs::X0, aarch64_regs::X1));
    // X0 and V0 do NOT overlap (different groups)
    assert!(!aarch64_regs::regs_overlap(aarch64_regs::X0, aarch64_regs::V0));
}

#[test]
fn aarch64_regs_condcode_integration() {
    // CondCode encoding and inversion
    assert_eq!(aarch64_regs::CondCode::EQ.encoding(), 0);
    assert_eq!(aarch64_regs::CondCode::EQ.invert(), aarch64_regs::CondCode::NE);
    assert_eq!(aarch64_regs::CondCode::EQ.as_str(), "eq");

    // CC/CS aliases
    assert_eq!(aarch64_regs::CC, aarch64_regs::CondCode::LO);
    assert_eq!(aarch64_regs::CS, aarch64_regs::CondCode::HS);

    // Parse
    assert_eq!(
        aarch64_regs::CondCode::from_str("eq"),
        Some(aarch64_regs::CondCode::EQ)
    );
    assert_eq!(
        aarch64_regs::CondCode::from_str("cs"),
        Some(aarch64_regs::CondCode::HS)
    );
}

#[test]
fn aarch64_regs_preg_name_for_all_gprs() {
    for i in 0..=30 {
        let name = aarch64_regs::preg_name(aarch64_regs::PReg::new(i));
        assert_eq!(name, format!("x{}", i));
    }
    assert_eq!(aarch64_regs::preg_name(aarch64_regs::SP), "sp");

    for i in 0..=30 {
        let name = aarch64_regs::preg_name(aarch64_regs::PReg::new(32 + i));
        assert_eq!(name, format!("w{}", i));
    }
    assert_eq!(aarch64_regs::preg_name(aarch64_regs::WSP), "wsp");
}

#[test]
fn aarch64_regs_convenience_aliases() {
    assert_eq!(aarch64_regs::FP, aarch64_regs::X29);
    assert_eq!(aarch64_regs::LR, aarch64_regs::X30);
}

// ===========================================================================
// InstFlags all-bits test
// ===========================================================================

#[test]
fn instflags_all_individual_bits_are_powers_of_two() {
    let flags = [
        InstFlags::IS_CALL,
        InstFlags::IS_BRANCH,
        InstFlags::IS_RETURN,
        InstFlags::IS_TERMINATOR,
        InstFlags::HAS_SIDE_EFFECTS,
        InstFlags::IS_PSEUDO,
        InstFlags::READS_MEMORY,
        InstFlags::WRITES_MEMORY,
        InstFlags::IS_PHI,
    ];
    for f in &flags {
        let bits = f.bits();
        assert!(bits.is_power_of_two(), "flag {:?} bits={:#x} not power of 2", f, bits);
    }
}

// ===========================================================================
// OperandSize/FloatSize with MachInst
// ===========================================================================

#[test]
fn operand_size_sf_bit_for_encoding() {
    // sf=0 for 32-bit, sf=1 for 64-bit — used in AArch64 instruction encoding
    assert_eq!(OperandSize::S32.sf_bit(), 0);
    assert_eq!(OperandSize::S64.sf_bit(), 1);
}

#[test]
fn float_size_ftype_for_encoding() {
    // ftype=0 for F32, ftype=1 for F64 — used in FP instruction encoding
    assert_eq!(FloatSize::F32.ftype(), 0);
    assert_eq!(FloatSize::F64.ftype(), 1);
}

// ===========================================================================
// MachInst with implicit defs/uses (call convention test)
// ===========================================================================

#[test]
fn machinst_call_with_implicit_clobbers() {
    // A BL instruction clobbers caller-saved registers
    static CALL_CLOBBERS: &[PReg] = &[X0, X1];
    static CALL_USES: &[PReg] = &[X0]; // argument in X0

    let inst = MachInst::new(AArch64Opcode::Bl, vec![MachOperand::Imm(0x1000)])
        .with_implicit_defs(CALL_CLOBBERS)
        .with_implicit_uses(CALL_USES);

    assert!(inst.is_call());
    assert!(inst.has_side_effects());
    assert_eq!(inst.implicit_defs.len(), 2);
    assert_eq!(inst.implicit_uses.len(), 1);
    assert_eq!(inst.implicit_defs[0], X0);
    assert_eq!(inst.implicit_uses[0], X0);
}

// ===========================================================================
// Typed index safety (BlockId vs InstId vs VRegId)
// ===========================================================================

#[test]
fn typed_indices_are_display_distinct() {
    // Even with same inner value, different types display differently
    assert_ne!(format!("{}", InstId(5)), format!("{}", BlockId(5)));
    assert_ne!(format!("{}", BlockId(5)), format!("{}", VRegId(5)));
    assert_ne!(format!("{}", VRegId(5)), format!("{}", StackSlotId(5)));
    assert_ne!(format!("{}", StackSlotId(5)), format!("{}", FrameIdx(5)));
}

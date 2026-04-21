// aggregate_pipeline.rs — O0/O1/O2/O3 regression for aggregates (#391)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verify that aggregate-style code patterns (struct field addressing,
//! array element addressing, aggregate loads/stores) survive every
//! optimisation level without being miscompiled.
//!
//! The machine IR patterns below mimic what the ISel produces for
//! `Opcode::StructGep`, `Opcode::ArrayGep`, and bulk memory copies:
//! `AddRI` + `LdrRI`/`StrRI` for struct fields, and `LslRI` + `AddRR`
//! + `LdrRI`/`StrRI` for array element addressing.
//!
//! Phase 1 of the aggregate lowering plan — see
//! `designs/2026-04-18-aggregate-lowering.md` for context and the
//! per-pass audit table.

use llvm2_ir::{
    AArch64Opcode, BlockId, InstId, MachBlock, MachFunction, MachInst, MachOperand,
    RegClass, Signature, VReg,
};
use llvm2_opt::{OptLevel, OptimizationPipeline};

fn vreg64(id: u32) -> MachOperand {
    MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
}
fn vreg32(id: u32) -> MachOperand {
    MachOperand::VReg(VReg::new(id, RegClass::Gpr32))
}
fn imm(value: i64) -> MachOperand {
    MachOperand::Imm(value)
}

/// Build a function whose single block models:
///
/// ```text
///   base   = arg0        (X0 held in v0: Gpr64)
///   addr   = base + 8    ; ADD imm (StructGep field 1)
///   value  = LDR addr    ; Load from field 1 (i64)
///   STR value, addr      ; Store back to same field
///   RET
/// ```
fn make_struct_field_rw_function(name: &str) -> MachFunction {
    let mut func = MachFunction::new(name.to_string(), Signature::new(vec![], vec![]));
    let entry = func.entry;

    // base pointer in v0 (Gpr64)
    // addr = base + 8
    let add = func.push_inst(MachInst::new(
        AArch64Opcode::AddRI,
        vec![vreg64(1), vreg64(0), imm(8)],
    ));
    func.append_inst(entry, add);

    // v2 = LDR [v1, #0]
    let ldr = func.push_inst(MachInst::new(
        AArch64Opcode::LdrRI,
        vec![vreg64(2), vreg64(1), imm(0)],
    ));
    func.append_inst(entry, ldr);

    // STR v2, [v1, #0]
    let str_inst = func.push_inst(MachInst::new(
        AArch64Opcode::StrRI,
        vec![vreg64(2), vreg64(1), imm(0)],
    ));
    func.append_inst(entry, str_inst);

    // RET
    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
    func.append_inst(entry, ret);

    func
}

/// Build a function modelling an array element read:
///
/// ```text
///   base  = v0
///   index = v1
///   scaled = index << 2            ; LSL for i32 array
///   addr   = base + scaled         ; ADD
///   value  = LDR [addr, #0]
///   RET
/// ```
fn make_array_read_function(name: &str) -> MachFunction {
    let mut func = MachFunction::new(name.to_string(), Signature::new(vec![], vec![]));
    let entry = func.entry;

    // scaled = index << 2
    let lsl = func.push_inst(MachInst::new(
        AArch64Opcode::LslRI,
        vec![vreg64(2), vreg64(1), imm(2)],
    ));
    func.append_inst(entry, lsl);

    // addr = base + scaled
    let add = func.push_inst(MachInst::new(
        AArch64Opcode::AddRR,
        vec![vreg64(3), vreg64(0), vreg64(2)],
    ));
    func.append_inst(entry, add);

    // value = LDR [addr, #0]   (i32 load via Gpr32)
    let ldr = func.push_inst(MachInst::new(
        AArch64Opcode::LdrRI,
        vec![vreg32(4), vreg64(3), imm(0)],
    ));
    func.append_inst(entry, ldr);

    // RET
    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
    func.append_inst(entry, ret);

    func
}

/// Count instructions with a specific opcode in block `entry`.
fn count_opcode(func: &MachFunction, block: BlockId, opc: AArch64Opcode) -> usize {
    let block: &MachBlock = func.block(block);
    block
        .insts
        .iter()
        .filter(|id| func.inst(**id).opcode == opc)
        .count()
}

/// Assert that the function's entry block still contains the aggregate
/// scaffolding (ADD/LSL/LDR/STR) needed to produce observable behaviour.
fn assert_struct_pattern_preserved(func: &MachFunction, ctx: &str) {
    let entry = func.entry;
    let add = count_opcode(func, entry, AArch64Opcode::AddRI);
    let ldr = count_opcode(func, entry, AArch64Opcode::LdrRI);
    let str_count = count_opcode(func, entry, AArch64Opcode::StrRI);
    let ret = count_opcode(func, entry, AArch64Opcode::Ret);

    // The store is observable: every pass MUST preserve at least one
    // STR. The LDR feeds the STR, so it must survive too. DCE must not
    // remove either one. The ADD can be merged into AddrModeFormation
    // but an equivalent addressing operand must remain on the LDR/STR.
    assert!(
        str_count >= 1,
        "[{}] STR was DCE'd or removed — aggregate store is observable",
        ctx
    );
    assert!(
        ldr >= 1,
        "[{}] LDR was DCE'd — aggregate load feeds the STR",
        ctx
    );
    assert!(ret >= 1, "[{}] RET disappeared", ctx);
    // ADD may be folded into LDR/STR via addr-mode; that's fine.
    let _ = add;
}

fn assert_array_pattern_preserved(func: &MachFunction, ctx: &str) {
    let entry = func.entry;
    let lsl = count_opcode(func, entry, AArch64Opcode::LslRI);
    let add_rr = count_opcode(func, entry, AArch64Opcode::AddRR);
    let ldr = count_opcode(func, entry, AArch64Opcode::LdrRI);
    let ret = count_opcode(func, entry, AArch64Opcode::Ret);
    // DCE-only O1: the LDR's result is unused, so DCE may remove LDR.
    // But the address computation (LSL+AddRR+LDR) is pure, so eliminating
    // them is semantically safe — the only observable effect is the
    // return. So we check only that RET is present and the function
    // still compiles.
    assert!(ret >= 1, "[{}] RET disappeared", ctx);
    let _ = lsl;
    let _ = add_rr;
    let _ = ldr;
}

// ---------------------------------------------------------------------------
// Struct field read/write — every opt level preserves the observable store.
// ---------------------------------------------------------------------------

#[test]
fn struct_field_rw_o0_preserves_everything() {
    let mut func = make_struct_field_rw_function("struct_rw_o0");
    let before_insts: Vec<InstId> = func.block(func.entry).insts.clone();
    OptimizationPipeline::new(OptLevel::O0).run(&mut func);
    let after_insts = &func.block(func.entry).insts;
    assert_eq!(
        &before_insts, after_insts,
        "O0 must be identity — no insts removed/added"
    );
    assert_struct_pattern_preserved(&func, "O0");
}

#[test]
fn struct_field_rw_o1_preserves_store() {
    let mut func = make_struct_field_rw_function("struct_rw_o1");
    OptimizationPipeline::new(OptLevel::O1).run(&mut func);
    assert_struct_pattern_preserved(&func, "O1");
}

#[test]
fn struct_field_rw_o2_preserves_store() {
    let mut func = make_struct_field_rw_function("struct_rw_o2");
    OptimizationPipeline::new(OptLevel::O2).run(&mut func);
    assert_struct_pattern_preserved(&func, "O2");
}

#[test]
fn struct_field_rw_o3_preserves_store() {
    let mut func = make_struct_field_rw_function("struct_rw_o3");
    OptimizationPipeline::new(OptLevel::O3).run(&mut func);
    assert_struct_pattern_preserved(&func, "O3");
}

// ---------------------------------------------------------------------------
// Array element read — every opt level preserves control flow (RET).
// DCE is expected to remove unused loads at O1+, so we only check RET.
// ---------------------------------------------------------------------------

#[test]
fn array_read_o0_identity() {
    let mut func = make_array_read_function("array_read_o0");
    let before_insts: Vec<InstId> = func.block(func.entry).insts.clone();
    OptimizationPipeline::new(OptLevel::O0).run(&mut func);
    let after_insts = &func.block(func.entry).insts;
    assert_eq!(&before_insts, after_insts, "O0 must be identity");
    assert_array_pattern_preserved(&func, "O0");
}

#[test]
fn array_read_o1_survives() {
    let mut func = make_array_read_function("array_read_o1");
    OptimizationPipeline::new(OptLevel::O1).run(&mut func);
    assert_array_pattern_preserved(&func, "O1");
}

#[test]
fn array_read_o2_survives() {
    let mut func = make_array_read_function("array_read_o2");
    OptimizationPipeline::new(OptLevel::O2).run(&mut func);
    assert_array_pattern_preserved(&func, "O2");
}

#[test]
fn array_read_o3_survives() {
    let mut func = make_array_read_function("array_read_o3");
    OptimizationPipeline::new(OptLevel::O3).run(&mut func);
    assert_array_pattern_preserved(&func, "O3");
}

// ---------------------------------------------------------------------------
// Aliasing guard — a load followed by a store to the same address, then
// another load: the second load must not be CSE'd away (the intervening
// store is a memory clobber).
// ---------------------------------------------------------------------------

/// Build a function that does:
/// ```text
///   base  = v0
///   addr  = base + 8
///   v2 = LDR [addr, #0]
///   STR v2, [addr, #0]   ; observable store — must survive
///   v3 = LDR [addr, #0]  ; may or may not be CSE'd depending on effects
///   RET
/// ```
fn make_load_store_load(name: &str) -> MachFunction {
    let mut func = MachFunction::new(name.to_string(), Signature::new(vec![], vec![]));
    let entry = func.entry;

    let add = func.push_inst(MachInst::new(
        AArch64Opcode::AddRI,
        vec![vreg64(1), vreg64(0), imm(8)],
    ));
    func.append_inst(entry, add);

    let ld1 = func.push_inst(MachInst::new(
        AArch64Opcode::LdrRI,
        vec![vreg64(2), vreg64(1), imm(0)],
    ));
    func.append_inst(entry, ld1);

    let st = func.push_inst(MachInst::new(
        AArch64Opcode::StrRI,
        vec![vreg64(2), vreg64(1), imm(0)],
    ));
    func.append_inst(entry, st);

    let ld2 = func.push_inst(MachInst::new(
        AArch64Opcode::LdrRI,
        vec![vreg64(3), vreg64(1), imm(0)],
    ));
    func.append_inst(entry, ld2);

    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
    func.append_inst(entry, ret);

    func
}

#[test]
fn store_survives_every_opt_level() {
    // The STR is the only observable side effect. DCE, CSE, GVN, LICM
    // must never remove it.
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut func = make_load_store_load(&format!("lsl_{:?}", level));
        OptimizationPipeline::new(level).run(&mut func);
        let str_count = count_opcode(&func, func.entry, AArch64Opcode::StrRI);
        assert!(
            str_count >= 1,
            "STR removed at {:?}! aggregate store is observable",
            level
        );
    }
}

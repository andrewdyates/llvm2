// sroa_pipeline.rs — O0/O1/O2/O3 regression for SROA (#391 phase 2b)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! End-to-end regression for Scalar Replacement of Aggregates (SROA).
//!
//! The representative pattern is the isel output for a `(i64, i64)` struct
//! local whose address never escapes:
//!
//! ```text
//!   v10 = AddPCRel  SP, StackSlot(0)    ; root address
//!   v11 = AddRI     v10, #8             ; field 1 offset
//!   STR  v0, v10, #0                    ; field0 = arg0
//!   STR  v1, v11, #0                    ; field1 = arg1
//!   v2  = LDR       v10, #0             ; return field 0
//!   RET
//! ```
//!
//! At O0 the pipeline is an identity — SROA does NOT fire.
//! At O1+ SROA must eliminate every LDR/STR and every AddPCRel/AddRI root.
//!
//! Must not regress the Phase 1 array/struct tests: see
//! `crates/llvm2-opt/tests/aggregate_pipeline.rs`.

use llvm2_ir::{
    AArch64Opcode, BlockId, MachBlock, MachFunction, MachInst, MachOperand, RegClass, Signature,
    StackSlot, VReg, regs::SP,
};
use llvm2_opt::{OptLevel, OptimizationPipeline};

fn vreg64(id: u32) -> MachOperand {
    MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
}
fn imm(value: i64) -> MachOperand {
    MachOperand::Imm(value)
}

/// Build the struct-local pattern: AddPCRel + AddRI + STR/STR/LDR.
///
/// Uses vreg ids 10 (root), 11 (derived), 0/1 (inputs), 2 (result).
fn make_sroa_fixture(name: &str) -> MachFunction {
    let mut func = MachFunction::new(name.to_string(), Signature::new(vec![], vec![]));
    let entry = func.entry;
    func.next_vreg = 20;
    // 16-byte, 8-byte aligned slot (matches (i64, i64)).
    let slot = func.alloc_stack_slot(StackSlot::new(16, 8));

    let push = |func: &mut MachFunction, block: BlockId, inst: MachInst| {
        let id = func.push_inst(inst);
        func.append_inst(block, id);
    };

    // v10 = AddPCRel SP, StackSlot(N)
    push(
        &mut func,
        entry,
        MachInst::new(
            AArch64Opcode::AddPCRel,
            vec![vreg64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
        ),
    );
    // v11 = AddRI v10, #8
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::AddRI, vec![vreg64(11), vreg64(10), imm(8)]),
    );
    // STR v0, v10, #0    (store field 0)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::StrRI, vec![vreg64(0), vreg64(10), imm(0)]),
    );
    // STR v1, v11, #0    (store field 1)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::StrRI, vec![vreg64(1), vreg64(11), imm(0)]),
    );
    // v2 = LDR v10, #0   (read back field 0 — kept live by returning it)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::LdrRI, vec![vreg64(2), vreg64(10), imm(0)]),
    );
    // Consume v2 in something DCE cannot kill (pretend-call: opcode that writes memory).
    // Using Bl keeps v2 live across optimizations without introducing a return.
    push(
        &mut func,
        entry,
        MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("use_v2".to_string()), vreg64(2)],
        ),
    );
    // RET
    push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

    func
}

fn opcode_histogram(func: &MachFunction) -> Vec<AArch64Opcode> {
    let mut out = Vec::new();
    for block_id in &func.block_order {
        let block: &MachBlock = func.block(*block_id);
        for id in &block.insts {
            out.push(func.inst(*id).opcode);
        }
    }
    out
}

fn count(opcodes: &[AArch64Opcode], target: AArch64Opcode) -> usize {
    opcodes.iter().filter(|o| **o == target).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// O0: SROA must NOT fire. Every load/store/add survives verbatim.
#[test]
fn sroa_o0_no_change() {
    let mut func = make_sroa_fixture("sroa_o0");
    let before = opcode_histogram(&func);
    OptimizationPipeline::new(OptLevel::O0).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(
        before, after,
        "O0 must be an identity — SROA is not in the O0 pipeline"
    );
    assert!(
        count(&after, AArch64Opcode::AddPCRel) >= 1,
        "O0: root AddPCRel kept"
    );
    assert!(
        count(&after, AArch64Opcode::StrRI) >= 2,
        "O0: both STR instructions kept"
    );
    assert!(
        count(&after, AArch64Opcode::LdrRI) >= 1,
        "O0: LDR kept"
    );
}

/// O1: SROA fires — all stack traffic gone from the fixture.
#[test]
fn sroa_o1_eliminates_stack_traffic() {
    let mut func = make_sroa_fixture("sroa_o1");
    OptimizationPipeline::new(OptLevel::O1).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(
        count(&after, AArch64Opcode::AddPCRel),
        0,
        "O1: root AddPCRel must be SROA'd away"
    );
    assert_eq!(
        count(&after, AArch64Opcode::StrRI),
        0,
        "O1: STR instructions must be SROA'd away"
    );
    assert_eq!(
        count(&after, AArch64Opcode::LdrRI),
        0,
        "O1: LDR instructions must be SROA'd away"
    );
    // Ret and the Bl that consumes v2 should still be present.
    assert!(count(&after, AArch64Opcode::Ret) >= 1);
    assert!(count(&after, AArch64Opcode::Bl) >= 1);
}

/// O2: SROA fires (full bisect pipeline).
#[test]
fn sroa_o2_eliminates_stack_traffic() {
    let mut func = make_sroa_fixture("sroa_o2");
    OptimizationPipeline::new(OptLevel::O2).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(count(&after, AArch64Opcode::AddPCRel), 0, "O2: root removed");
    assert_eq!(count(&after, AArch64Opcode::StrRI), 0, "O2: STR removed");
    assert_eq!(count(&after, AArch64Opcode::LdrRI), 0, "O2: LDR removed");
}

/// O3: SROA fires (fixpoint pipeline).
#[test]
fn sroa_o3_eliminates_stack_traffic() {
    let mut func = make_sroa_fixture("sroa_o3");
    OptimizationPipeline::new(OptLevel::O3).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(count(&after, AArch64Opcode::AddPCRel), 0, "O3: root removed");
    assert_eq!(count(&after, AArch64Opcode::StrRI), 0, "O3: STR removed");
    assert_eq!(count(&after, AArch64Opcode::LdrRI), 0, "O3: LDR removed");
}

/// Negative regression: an escaping slot must stay on the stack at every
/// optimisation level. (Addresses a slot-that-escapes-to-call pattern.)
#[test]
fn sroa_respects_escape_at_all_levels() {
    fn make_escape() -> MachFunction {
        let mut func = MachFunction::new(
            "sroa_escape".to_string(),
            Signature::new(vec![], vec![]),
        );
        let entry = func.entry;
        func.next_vreg = 20;
        let slot = func.alloc_stack_slot(StackSlot::new(8, 8));
        let push = |func: &mut MachFunction, block: BlockId, inst: MachInst| {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        };
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![vreg64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // BL callee, v10   — passing the slot address is an escape.
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("callee".to_string()), vreg64(10)],
            ),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));
        func
    }

    // The key property: SROA must NOT remove the AddPCRel when the slot
    // address escapes (i.e., is passed to a call). Other passes (tail-call
    // opt, DCE) may legitimately transform the Bl at higher levels, but the
    // slot address must still be computed for the transformed call.
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut func = make_escape();
        OptimizationPipeline::new(level).run(&mut func);
        let after = opcode_histogram(&func);
        assert!(
            count(&after, AArch64Opcode::AddPCRel) >= 1,
            "{:?}: escaping AddPCRel must be preserved by SROA",
            level
        );
    }
}

// ---------------------------------------------------------------------------
// Nested-aggregate coverage (#391 Phase 2c / #443 item 1)
//
// Phase 2b adapter lowering for `Constant::Aggregate(..)` recurses into nested
// fields using a SINGLE parent StackAddr + per-leaf `StructGep`/`ArrayGep` +
// `Store` chain (see `adapter.rs::fill_aggregate_at_ptr`). SROA's
// `trace_addr_uses` already recurses through chained `AddRI` — so a nested
// aggregate local whose address never escapes must scalarise end-to-end at
// O1/O2/O3.
//
// The adapter-level regression lives at
// `crates/llvm2-lower/tests/aggregate_constants.rs` (1 StackAddr, per-leaf
// Store shape). That proves the tMIR -> MachIR shape, but NOT that the SROA
// pass erases the stack traffic. This file closes that gap.
//
// Fixture layout models a nested struct local:
//     Outer {
//         inner: Inner { x: i64, y: i64 },   // bytes [0 .. 16)
//         z:     i64,                         // bytes [16 .. 24)
//     }
//
// MachIR pattern (4 distinct byte offsets: 0, 8, 16; 3 stores + 1 load):
//     v10 = AddPCRel SP, Slot              ; root (Outer base, offset 0)
//     v11 = AddRI    v10, #0               ; inner base (chained via identity)
//     v12 = AddRI    v11, #8               ; inner.y  (offset 0 + 8 = 8)
//     v13 = AddRI    v10, #16              ; outer.z  (offset 16)
//     STR  v0, v11, #0                     ; inner.x = arg0
//     STR  v1, v12, #0                     ; inner.y = arg1
//     STR  v2, v13, #0                     ; outer.z = arg2
//     v3  = LDR       v11, #0              ; read inner.x (kept live below)
//     Bl  use_v3 v3
//     Ret
//
// The chained `AddRI v11, #8` exercises the key nested-aggregate property:
// every derived address must resolve to a concrete byte offset via SROA's
// recursive base-offset accumulator. If that recursion ever regresses, the
// nested test fires before any real tMIR input hits the pipeline.
// ---------------------------------------------------------------------------

/// Build the nested-aggregate fixture described above. Uses vreg ids 10
/// (root), 11 (inner base), 12 (inner.y), 13 (outer.z); 0/1/2 (inputs);
/// 3 (result).
fn make_nested_sroa_fixture(name: &str) -> MachFunction {
    let mut func = MachFunction::new(name.to_string(), Signature::new(vec![], vec![]));
    let entry = func.entry;
    func.next_vreg = 20;
    // 24-byte, 8-byte aligned slot (sizeof Outer = 24, align = 8).
    let slot = func.alloc_stack_slot(StackSlot::new(24, 8));

    let push = |func: &mut MachFunction, block: BlockId, inst: MachInst| {
        let id = func.push_inst(inst);
        func.append_inst(block, id);
    };

    // v10 = AddPCRel SP, StackSlot(N)
    push(
        &mut func,
        entry,
        MachInst::new(
            AArch64Opcode::AddPCRel,
            vec![vreg64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
        ),
    );
    // v11 = AddRI v10, #0   (inner base; identity offset as emitted for the
    //                        first field of a nested aggregate literal)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::AddRI, vec![vreg64(11), vreg64(10), imm(0)]),
    );
    // v12 = AddRI v11, #8   (chained: inner.y at absolute offset 8)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::AddRI, vec![vreg64(12), vreg64(11), imm(8)]),
    );
    // v13 = AddRI v10, #16  (outer.z at absolute offset 16)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::AddRI, vec![vreg64(13), vreg64(10), imm(16)]),
    );
    // STR v0, v11, #0   (inner.x = arg0 at offset 0)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::StrRI, vec![vreg64(0), vreg64(11), imm(0)]),
    );
    // STR v1, v12, #0   (inner.y = arg1 at offset 8)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::StrRI, vec![vreg64(1), vreg64(12), imm(0)]),
    );
    // STR v2, v13, #0   (outer.z = arg2 at offset 16)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::StrRI, vec![vreg64(2), vreg64(13), imm(0)]),
    );
    // v3 = LDR v11, #0  (reload inner.x — kept live by the consumer below)
    push(
        &mut func,
        entry,
        MachInst::new(AArch64Opcode::LdrRI, vec![vreg64(3), vreg64(11), imm(0)]),
    );
    // Bl use_v3 v3   (pretend-consumer so DCE cannot remove the read-back)
    push(
        &mut func,
        entry,
        MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("use_v3".to_string()), vreg64(3)],
        ),
    );
    // Ret
    push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

    func
}

/// O0: nested-aggregate SROA must NOT fire. All 1 AddPCRel + 3 AddRI + 3 STR
/// + 1 LDR survive verbatim.
#[test]
fn nested_sroa_o0_no_change() {
    let mut func = make_nested_sroa_fixture("nested_sroa_o0");
    let before = opcode_histogram(&func);
    OptimizationPipeline::new(OptLevel::O0).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(
        before, after,
        "O0 must be an identity — SROA is not in the O0 pipeline"
    );
    assert_eq!(count(&after, AArch64Opcode::AddPCRel), 1, "O0: root kept");
    assert_eq!(
        count(&after, AArch64Opcode::AddRI),
        3,
        "O0: all three AddRI derived defs kept (inner base, inner.y, outer.z)"
    );
    assert_eq!(
        count(&after, AArch64Opcode::StrRI),
        3,
        "O0: all three STRs (x/y/z) kept"
    );
    assert_eq!(
        count(&after, AArch64Opcode::LdrRI),
        1,
        "O0: inner.x read-back LDR kept"
    );
}

/// O1: chained AddRI (nested aggregate access) must scalarise. Every
/// AddPCRel, AddRI-that-derives-from-the-root, STR, and LDR disappears —
/// only the `Bl` consumer and `Ret` should remain.
#[test]
fn nested_sroa_o1_eliminates_nested_stack_traffic() {
    let mut func = make_nested_sroa_fixture("nested_sroa_o1");
    OptimizationPipeline::new(OptLevel::O1).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(
        count(&after, AArch64Opcode::AddPCRel),
        0,
        "O1: root AddPCRel must be SROA'd away (nested-aggregate slot)"
    );
    assert_eq!(
        count(&after, AArch64Opcode::AddRI),
        0,
        "O1: all three derived AddRI must be SROA'd away — including the \
         chained v12 = AddRI v11, #8 (nested field access)"
    );
    assert_eq!(
        count(&after, AArch64Opcode::StrRI),
        0,
        "O1: all three STRs must be SROA'd away"
    );
    assert_eq!(
        count(&after, AArch64Opcode::LdrRI),
        0,
        "O1: LDR must be SROA'd away — scalar replacement folds STR(x)+LDR \
         into a vreg copy"
    );
    assert!(
        count(&after, AArch64Opcode::Ret) >= 1,
        "Ret must survive SROA"
    );
    assert!(
        count(&after, AArch64Opcode::Bl) >= 1,
        "Bl consumer must survive SROA (keeps the reload live)"
    );
}

/// O2: SROA fires through the full bisect-friendly pipeline on a nested slot.
#[test]
fn nested_sroa_o2_eliminates_nested_stack_traffic() {
    let mut func = make_nested_sroa_fixture("nested_sroa_o2");
    OptimizationPipeline::new(OptLevel::O2).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(
        count(&after, AArch64Opcode::AddPCRel),
        0,
        "O2: root removed"
    );
    assert_eq!(count(&after, AArch64Opcode::AddRI), 0, "O2: chained AddRI removed");
    assert_eq!(count(&after, AArch64Opcode::StrRI), 0, "O2: STRs removed");
    assert_eq!(count(&after, AArch64Opcode::LdrRI), 0, "O2: LDR removed");
}

/// O3: SROA fires through the fixpoint pipeline on a nested slot. Guards
/// against a future fixpoint pass re-introducing the stack traffic.
#[test]
fn nested_sroa_o3_eliminates_nested_stack_traffic() {
    let mut func = make_nested_sroa_fixture("nested_sroa_o3");
    OptimizationPipeline::new(OptLevel::O3).run(&mut func);
    let after = opcode_histogram(&func);
    assert_eq!(
        count(&after, AArch64Opcode::AddPCRel),
        0,
        "O3: root removed"
    );
    assert_eq!(count(&after, AArch64Opcode::AddRI), 0, "O3: chained AddRI removed");
    assert_eq!(count(&after, AArch64Opcode::StrRI), 0, "O3: STRs removed");
    assert_eq!(count(&after, AArch64Opcode::LdrRI), 0, "O3: LDR removed");
}

/// Negative regression for nested aggregates: if the nested slot's address
/// escapes (via `Bl` consuming the root), SROA must preserve the AddPCRel at
/// every optimisation level. This is the nested-analogue of
/// `sroa_respects_escape_at_all_levels` — guards against a future SROA change
/// that accidentally treats a chained AddRI base as "non-escaping" even when
/// the root does escape.
#[test]
fn nested_sroa_respects_escape_at_all_levels() {
    fn make_nested_escape() -> MachFunction {
        let mut func = MachFunction::new(
            "nested_sroa_escape".to_string(),
            Signature::new(vec![], vec![]),
        );
        let entry = func.entry;
        func.next_vreg = 20;
        let slot = func.alloc_stack_slot(StackSlot::new(24, 8));
        let push = |func: &mut MachFunction, block: BlockId, inst: MachInst| {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        };
        // v10 = AddPCRel SP, slot
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![vreg64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // v11 = AddRI v10, #0   (inner base)
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::AddRI, vec![vreg64(11), vreg64(10), imm(0)]),
        );
        // v12 = AddRI v11, #8   (chained nested-field derived address)
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::AddRI, vec![vreg64(12), vreg64(11), imm(8)]),
        );
        // STR v0, v12, #0
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![vreg64(0), vreg64(12), imm(0)]),
        );
        // Bl callee, v10    — the root address escapes to a non-pure call.
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("callee".to_string()), vreg64(10)],
            ),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));
        func
    }

    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut func = make_nested_escape();
        OptimizationPipeline::new(level).run(&mut func);
        let after = opcode_histogram(&func);
        assert!(
            count(&after, AArch64Opcode::AddPCRel) >= 1,
            "{:?}: escaping nested-aggregate AddPCRel must be preserved by SROA",
            level
        );
    }
}

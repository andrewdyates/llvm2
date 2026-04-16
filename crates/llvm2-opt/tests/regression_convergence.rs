// llvm2-opt/tests/regression_convergence.rs
//
// Regression tests for optimization convergence and termination.
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests guard against infinite loops in the scheduler (cyclic DAG
// dependencies) and CFG simplification (oscillating sub-passes). The bugs
// were fixed in Wave 35 by:
//   1. Force-scheduling stuck nodes in schedule_list / schedule_list_pressure_aware
//   2. Bounding CfgSimplify iterations to max_iterations = 32
//
// Part of #288

use std::time::{Duration, Instant};

use llvm2_ir::{
    AArch64Opcode, BlockId, InstId, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg,
};
use llvm2_opt::{
    cfg_simplify::CfgSimplify,
    scheduler::{
        schedule_list, schedule_list_pressure_aware, ExecutionPort, ScheduleDAG, ScheduleNode,
    },
    MachinePass, OptLevel, OptimizationPipeline,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn new_func(name: &str) -> MachFunction {
    MachFunction::new(name.to_string(), Signature::new(vec![], vec![]))
}

fn vreg(id: u32) -> MachOperand {
    MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
}

fn imm(value: i64) -> MachOperand {
    MachOperand::Imm(value)
}

fn add_ri(dst: u32, src: u32, value: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::AddRI,
        vec![vreg(dst), vreg(src), imm(value)],
    )
}

fn b(target: BlockId) -> MachInst {
    MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(target)])
}

fn cbz(cond: u32, target: BlockId) -> MachInst {
    MachInst::new(
        AArch64Opcode::Cbz,
        vec![vreg(cond), MachOperand::Block(target)],
    )
}

fn ret() -> MachInst {
    MachInst::new(AArch64Opcode::Ret, vec![])
}

fn append_inst(func: &mut MachFunction, block: BlockId, inst: MachInst) -> InstId {
    let inst_id = func.push_inst(inst);
    func.append_inst(block, inst_id);
    inst_id
}

fn schedule_node(inst_id: InstId, deps: &[usize], rev_deps: &[usize]) -> ScheduleNode {
    ScheduleNode {
        inst_id,
        latency: 1,
        port: ExecutionPort::IntAlu,
        deps: deps.to_vec(),
        rev_deps: rev_deps.to_vec(),
        earliest_start: 0,
        priority: 0,
        scheduled: false,
    }
}

/// Assert that `order` contains exactly the same InstIds as `expected` (order-independent).
fn assert_schedules_all(order: &[InstId], expected: &[InstId]) {
    let mut actual = order.to_vec();
    let mut exp = expected.to_vec();
    actual.sort();
    exp.sort();
    assert_eq!(actual, exp);
}

/// Assert that the operation completed in under 5 seconds.
fn assert_under_five_seconds(start: Instant) {
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "expected completion within 5 seconds, took {:?}",
        elapsed
    );
}

/// Check that at least one block in the function contains a Ret instruction.
fn has_return(func: &MachFunction) -> bool {
    func.block_order.iter().any(|&block_id| {
        func.block(block_id)
            .insts
            .iter()
            .any(|&inst_id| func.inst(inst_id).opcode == AArch64Opcode::Ret)
    })
}

// ---------------------------------------------------------------------------
// Scheduler convergence tests
// ---------------------------------------------------------------------------

/// Regression: schedule_list must terminate when the DAG contains a 2-node
/// dependency cycle (A depends on B, B depends on A). Before the fix, this
/// caused an infinite loop because no node ever became ready.
#[test]
fn test_scheduler_cyclic_deps_terminates() {
    let start = Instant::now();
    let mut dag = ScheduleDAG {
        nodes: vec![
            schedule_node(InstId(0), &[1], &[1]),
            schedule_node(InstId(1), &[0], &[0]),
        ],
    };

    let order = schedule_list(&mut dag);

    assert_under_five_seconds(start);
    assert_eq!(order.len(), 2);
    assert_schedules_all(&order, &[InstId(0), InstId(1)]);
}

/// Regression: schedule_list must terminate on a 3-node cycle
/// (0 depends on 2, 1 depends on 0, 2 depends on 1). The force-schedule
/// mechanism must break the cycle and schedule all three nodes.
#[test]
fn test_scheduler_three_node_cycle_terminates() {
    let start = Instant::now();
    let mut dag = ScheduleDAG {
        nodes: vec![
            schedule_node(InstId(0), &[2], &[1]),
            schedule_node(InstId(1), &[0], &[2]),
            schedule_node(InstId(2), &[1], &[0]),
        ],
    };

    let order = schedule_list(&mut dag);

    assert_under_five_seconds(start);
    assert_eq!(order.len(), 3);
    assert_schedules_all(&order, &[InstId(0), InstId(1), InstId(2)]);
}

/// Regression: when free (no-dep) nodes coexist with cyclic nodes, the
/// scheduler must schedule all of them. The free nodes go first, then the
/// force-schedule mechanism breaks the cycle for the remaining nodes.
#[test]
fn test_scheduler_mixed_cycle_and_free() {
    let start = Instant::now();
    let mut dag = ScheduleDAG {
        nodes: vec![
            schedule_node(InstId(0), &[], &[]),    // free
            schedule_node(InstId(1), &[], &[]),    // free
            schedule_node(InstId(2), &[3], &[3]),  // cycle with node 3
            schedule_node(InstId(3), &[2], &[2]),  // cycle with node 2
        ],
    };

    let order = schedule_list(&mut dag);

    assert_under_five_seconds(start);
    assert_eq!(order.len(), 4);
    assert_schedules_all(
        &order,
        &[InstId(0), InstId(1), InstId(2), InstId(3)],
    );
}

/// Regression: schedule_list_pressure_aware must also terminate on cyclic
/// DAGs. It has its own copy of the force-schedule logic. Test with a
/// real MachFunction so the pressure tracker has instruction data to work with.
#[test]
fn test_scheduler_pressure_aware_cyclic_terminates() {
    let start = Instant::now();
    let mut func = new_func("pressure_cycle");
    let entry = func.entry;
    let inst0 = append_inst(&mut func, entry, add_ri(1, 0, 1));
    let inst1 = append_inst(&mut func, entry, add_ri(2, 1, 2));

    let mut dag = ScheduleDAG {
        nodes: vec![
            schedule_node(inst0, &[1], &[1]),
            schedule_node(inst1, &[0], &[0]),
        ],
    };

    let (order, _tracker) = schedule_list_pressure_aware(&func, &mut dag);

    assert_under_five_seconds(start);
    assert_eq!(order.len(), 2);
    assert_schedules_all(&order, &[inst0, inst1]);
}

// ---------------------------------------------------------------------------
// CFG simplification convergence tests
// ---------------------------------------------------------------------------

/// Regression: CfgSimplify must terminate on a function containing a loop
/// back-edge (bb1 -> bb0). Before the max_iterations cap, oscillating
/// sub-passes could loop forever on such patterns.
///
/// CFG:  bb0 --cbz--> bb1 --B--> bb0  (loop)
///        |
///        +--fallthrough--> bb2 (ret)
#[test]
fn test_cfg_simplify_terminates_with_loop() {
    let mut func = new_func("cfg_loop");
    let bb1 = func.create_block();
    let bb2 = func.create_block();

    // Layout: [entry, bb2, bb1] -- bb2 is the fallthrough from entry's Cbz
    func.block_order = vec![func.entry, bb2, bb1];

    let entry = func.entry;
    append_inst(&mut func, entry, cbz(0, bb1));
    append_inst(&mut func, bb1, b(entry));
    append_inst(&mut func, bb2, ret());

    func.add_edge(entry, bb1);
    func.add_edge(entry, bb2);
    func.add_edge(bb1, entry);

    let mut pass = CfgSimplify;
    let start = Instant::now();
    let _changed = pass.run(&mut func);

    assert_under_five_seconds(start);
    // The function must still contain a Ret instruction after simplification.
    assert!(has_return(&func));
}

/// Regression: a long chain of empty blocks (each containing only an
/// unconditional B to the next) must be fully collapsed without the
/// pass oscillating. The chain bb0->bb1->...->bb9->exit(ret) should
/// reduce to bb0 containing just Ret.
#[test]
fn test_cfg_simplify_chain_of_empty_blocks() {
    const CHAIN_LENGTH: usize = 10;

    let mut func = new_func("empty_chain");
    let mut current = func.entry;

    for _ in 1..CHAIN_LENGTH {
        let next = func.create_block();
        append_inst(&mut func, current, b(next));
        func.add_edge(current, next);
        current = next;
    }

    let exit = func.create_block();
    append_inst(&mut func, current, b(exit));
    func.add_edge(current, exit);
    append_inst(&mut func, exit, ret());

    let mut pass = CfgSimplify;
    let start = Instant::now();
    let _changed = pass.run(&mut func);

    assert_under_five_seconds(start);
    // All intermediate blocks should be eliminated, leaving only entry.
    assert_eq!(func.block_order, vec![func.entry]);

    // The entry block should now contain exactly one instruction: Ret.
    let entry_block = func.block(func.entry);
    assert_eq!(entry_block.insts.len(), 1);
    assert_eq!(
        func.inst(entry_block.insts[0]).opcode,
        AArch64Opcode::Ret
    );
}

// ---------------------------------------------------------------------------
// Full pipeline convergence test
// ---------------------------------------------------------------------------

/// Regression: the full O3 pipeline (which iterates to fixed point with
/// max 10 iterations) must terminate on a multi-block diamond CFG. This
/// exercises the interaction between CFG simplification, DCE, peephole,
/// and all other O3 passes on branching code.
///
/// Diamond CFG:
///   entry --cbz--> bb2
///     |              |
///     v              v
///    bb1 ---B---> bb3 (ret)
///              <---B--- bb2
#[test]
fn test_pipeline_o3_multiblock_terminates() {
    let mut func = new_func("diamond");
    let bb1 = func.create_block();
    let bb2 = func.create_block();
    let bb3 = func.create_block();

    let entry = func.entry;
    // entry: Cbz to bb2, fallthrough to bb1
    append_inst(&mut func, entry, cbz(0, bb2));
    append_inst(&mut func, bb1, b(bb3));
    append_inst(&mut func, bb2, b(bb3));
    append_inst(&mut func, bb3, ret());

    func.add_edge(entry, bb1);
    func.add_edge(entry, bb2);
    func.add_edge(bb1, bb3);
    func.add_edge(bb2, bb3);

    let pipeline = OptimizationPipeline::new(OptLevel::O3);
    let start = Instant::now();
    let stats = pipeline.run(&mut func);

    assert_under_five_seconds(start);
    assert!(stats.iterations >= 1);
    assert!(has_return(&func));
}

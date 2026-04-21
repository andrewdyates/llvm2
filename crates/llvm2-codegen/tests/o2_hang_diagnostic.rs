// Diagnostic test to isolate O2 hang on loop functions
//
// Part of #301 — O2 optimization hangs on loop-based tMIR functions
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

use tmir::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule, FuncTy, Ty};
use tmir::{Inst, InstrNode, BinOp, ICmpOp};
use tmir::{BlockId, FuncId, ValueId, Constant};

/// Build the same `sum_to_n` loop that hangs at O2.
fn build_loop_function() -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "sum_to_n", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(1), ValueId::new(2)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![
                (ValueId::new(10), Ty::I64),
                (ValueId::new(11), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(11),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(12),
                    then_target: BlockId::new(2),
                    then_args: vec![],
                    else_target: BlockId::new(3),
                    else_args: vec![ValueId::new(10)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(11),
                })
                .with_result(ValueId::new(13)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(14)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(11),
                    rhs: ValueId::new(14),
                })
                .with_result(ValueId::new(15)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(13), ValueId::new(15)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(20), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(20)],
            })],
        },
    ];
    module.add_function(func.clone());
    (func, module)
}

fn get_ir_func_via_isel() -> llvm2_ir::MachFunction {
    let (tmir_func, module) = build_loop_function();
    let (lir_func, _proof_ctx) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should succeed");

    use llvm2_lower::isel::InstructionSelector;
    use llvm2_lower::function::Signature;

    let sig = Signature {
        params: lir_func.signature.params.clone(),
        returns: lir_func.signature.returns.clone(),
    };

    let mut isel = InstructionSelector::new(lir_func.name.clone(), sig.clone());
    isel.set_stack_slots(lir_func.stack_slots.clone());
    isel.seed_value_types(&lir_func.value_types);
    isel.seed_pure_callees(&lir_func.pure_callees);
    isel.lower_formal_arguments(&sig, lir_func.entry_block)
        .expect("lower_formal_arguments should succeed");

    let mut block_order: Vec<_> = lir_func.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| b.0);

    for block_ref in &block_order {
        let basic_block = &lir_func.blocks[block_ref];
        if *block_ref != lir_func.entry_block && !basic_block.params.is_empty() {
            isel.define_block_params(&basic_block.params);
        }
        isel.select_block(*block_ref, &basic_block.instructions)
            .expect("select_block should succeed");
    }

    let isel_func = isel.finalize();
    isel_func.to_ir_func()
}

/// Test per-block scheduling to find which block hangs.
#[test]
fn test_scheduler_per_block() {
    use llvm2_opt::scheduler::{build_dag, schedule_list_pressure_aware};

    let func = get_ir_func_via_isel();

    eprintln!("  Function has {} blocks, block_order: {:?}", func.blocks.len(), func.block_order);

    for &block_id in &func.block_order {
        let block = func.block(block_id);
        eprintln!("  Block {:?}: {} instructions, succs={:?}, preds={:?}",
            block_id, block.insts.len(), block.succs, block.preds);

        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            eprintln!("    {:?}: {:?} {:?} flags={:?}",
                inst_id, inst.opcode, inst.operands, inst.flags);
        }

        if block.insts.len() <= 1 {
            eprintln!("  Block {:?}: skipping (<=1 inst)", block_id);
            continue;
        }

        eprintln!("  Building DAG for block {:?} ...", block_id);
        let mut dag = build_dag(&func, block_id);
        eprintln!("  DAG built: {} nodes", dag.nodes.len());

        // Dump DAG structure
        for (i, node) in dag.nodes.iter().enumerate() {
            eprintln!("    node[{}]: inst={:?} lat={} prio={} deps={:?} rev_deps={:?}",
                i, node.inst_id, node.latency, node.priority,
                node.deps, node.rev_deps);
        }

        // Check for cycles
        let mut has_cycle = false;
        for (i, node) in dag.nodes.iter().enumerate() {
            if node.deps.contains(&i) {
                eprintln!("    WARNING: node[{}] depends on itself!", i);
                has_cycle = true;
            }
            for &dep in &node.deps {
                if dag.nodes[dep].deps.contains(&i) {
                    eprintln!("    WARNING: cycle between node[{}] and node[{}]!", i, dep);
                    has_cycle = true;
                }
            }
        }

        if has_cycle {
            eprintln!("  Block {:?}: CYCLE DETECTED, skipping scheduling", block_id);
            continue;
        }

        eprintln!("  Scheduling block {:?} ...", block_id);
        let (new_order, tracker) = schedule_list_pressure_aware(&func, &mut dag);
        eprintln!("  Block {:?}: scheduled OK, {} insts, peak_gpr={}, peak_fpr={}",
            block_id, new_order.len(), tracker.peak_gpr, tracker.peak_fpr);
    }
    eprintln!("  All blocks scheduled!");
}

/// Test the non-pressure-aware scheduler (InstructionScheduler) on loop.
/// If this works but PressureAwareScheduler doesn't, the bug is in
/// the pressure-aware scheduling logic specifically.
#[test]
fn test_non_pressure_scheduler_on_loop() {
    use llvm2_opt::pass_manager::MachinePass;
    use llvm2_opt::scheduler::InstructionScheduler;

    let mut func = get_ir_func_via_isel();
    eprintln!("  Running InstructionScheduler (non-pressure) on loop ...");
    let mut sched = InstructionScheduler;
    sched.run(&mut func);
    eprintln!("  InstructionScheduler completed on loop.");
}

/// Test PressureAwareScheduler on the raw loop IR.
#[test]
fn test_pressure_scheduler_on_raw_loop() {
    use llvm2_opt::pass_manager::MachinePass;
    use llvm2_opt::scheduler::PressureAwareScheduler;

    let mut func = get_ir_func_via_isel();
    eprintln!("  Running PressureAwareScheduler on raw loop ...");
    let mut sched = PressureAwareScheduler;
    sched.run(&mut func);
    eprintln!("  PressureAwareScheduler completed on raw loop.");
}

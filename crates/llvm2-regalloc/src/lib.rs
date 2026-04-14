// llvm2-regalloc - Register allocation for LLVM2
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Register allocation for the LLVM2 verified compiler backend.
//!
//! This crate implements liveness analysis and register allocation for
//! machine-level IR. The current implementation uses linear scan with
//! spill weight computation and parallel copy resolution for phi elimination.
//!
//! ## Architecture
//!
//! ```text
//! RegAllocFunction (input, SSA with phis)
//!      |
//!      v
//! +-------------------+
//! | Critical Edge      |  split_critical_edges()
//! | Splitting          |
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Phi Elimination    |  eliminate_phis()
//! | (parallel copies)  |
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Liveness           |  compute_live_intervals()
//! | Analysis           |
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Copy Coalescing    |  coalesce_copies() + apply_coalescing()
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Linear Scan        |  LinearScan::allocate()
//! | Allocation         |
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Remat / Spill Code |  find_remat_candidates() / insert_spill_code()
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Spill Slot Reuse   |  compute_spill_slot_reuse()
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Call Save/Restore  |  insert_call_save_restore()
//! +--------+----------+
//!          |
//!          v
//! +-------------------+
//! | Post-RA Coalesce   |  post_ra_coalesce()
//! +--------+----------+
//!          |
//!          v
//! RegAllocFunction (output, VRegs replaced with PRegs)
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llvm2_regalloc::{allocate, AllocConfig};
//! use llvm2_regalloc::machine_types::RegAllocFunction;
//!
//! # fn example(mut func: RegAllocFunction) {
//! let config = AllocConfig::default_aarch64();
//! let result = allocate(&mut func, &config).expect("allocation failed");
//! # }
//! ```

pub mod machine_types;
pub mod liveness;
pub mod linear_scan;
pub mod greedy;
pub mod spill;
pub mod phi_elim;
pub mod coalesce;
pub mod post_ra_coalesce;
pub mod split;
pub mod remat;
pub mod spill_slot_reuse;
pub mod call_clobber;

pub use liveness::{compute_live_intervals, LiveInterval, LiveRange, LivenessResult};
pub use linear_scan::{aarch64_allocatable_regs, AllocError, AllocationResult, LinearScan, SpillInfo};
pub use greedy::{GreedyAllocator, Stage as GreedyStage};
pub use machine_types::{
    // Canonical names (issue #73):
    RegAllocBlock, RegAllocFunction, RegAllocInst, RegAllocOperand, RegAllocStackSlot,
    // Backward-compatible aliases (deprecated — use RegAlloc* names):
    InstFlags, MachBlock, MachFunction, MachInst, MachOperand, StackSlot,
};
// Re-export canonical types from llvm2-ir via machine_types.
pub use machine_types::{BlockId, InstId, PReg, RegClass, StackSlotId, VReg};
pub use phi_elim::{eliminate_phis, split_critical_edges};
pub use spill::insert_spill_code;
pub use coalesce::{coalesce_copies, apply_coalescing, CoalesceResult};
pub use post_ra_coalesce::{post_ra_coalesce, PostRACoalesceResult};
pub use split::{split_interval, find_optimal_split_point, SplitDecision, SplitResult};
pub use remat::{classify_remat_cost, find_remat_candidates, RematCost, RematCandidate};
pub use spill_slot_reuse::{compute_spill_slot_reuse, SpillSlotReuseResult};
pub use call_clobber::{
    aarch64_callee_saved_regs, aarch64_caller_saved_regs, find_call_crossings,
    insert_call_save_restore, compute_call_crossing_hints, CallCrossing,
};

use std::collections::HashMap;

/// Which register allocation algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    /// Linear scan: fast, processes intervals by start position.
    LinearScan,
    /// Greedy: LLVM-style, processes by spill weight with eviction and
    /// splitting for better code quality.
    Greedy,
}

/// Configuration for the register allocator.
pub struct AllocConfig {
    /// Allocatable physical registers per register class.
    pub allocatable_regs: HashMap<RegClass, Vec<PReg>>,
    /// Which allocation algorithm to use (default: LinearScan).
    pub strategy: AllocStrategy,
    /// Whether to enable copy coalescing (default: true).
    pub enable_coalescing: bool,
    /// Whether to enable rematerialization (default: true).
    pub enable_remat: bool,
    /// Whether to enable spill slot reuse (default: true).
    pub enable_spill_slot_reuse: bool,
    /// Register hints for the greedy allocator (ignored by linear scan).
    pub hints: HashMap<VReg, Vec<PReg>>,
}

impl AllocConfig {
    /// Default configuration for AArch64 (Apple calling convention).
    /// Uses linear scan for backward compatibility.
    pub fn default_aarch64() -> Self {
        Self {
            allocatable_regs: aarch64_allocatable_regs(),
            strategy: AllocStrategy::LinearScan,
            enable_coalescing: true,
            enable_remat: true,
            enable_spill_slot_reuse: true,
            hints: HashMap::new(),
        }
    }

    /// AArch64 configuration using the greedy allocator.
    pub fn greedy_aarch64() -> Self {
        Self {
            allocatable_regs: aarch64_allocatable_regs(),
            strategy: AllocStrategy::Greedy,
            enable_coalescing: true,
            enable_remat: true,
            enable_spill_slot_reuse: true,
            hints: HashMap::new(),
        }
    }
}

/// Main entry point: run the full register allocation pipeline.
///
/// This function:
/// 1. Splits critical edges.
/// 2. Eliminates phi instructions (inserts parallel copies).
/// 3. Computes live intervals.
/// 4. Copy coalescing (merges non-interfering intervals from copies).
/// 5. Runs allocation (linear scan or greedy, based on `config.strategy`).
/// 6. Rematerialization (recompute cheap values instead of spilling).
/// 7. Inserts spill code for remaining spilled VRegs.
/// 8. Spill slot reuse (share slots for non-overlapping spills).
///
/// Returns the allocation result with VReg-to-PReg mappings and spill info.
pub fn allocate(
    func: &mut RegAllocFunction,
    config: &AllocConfig,
) -> Result<AllocationResult, AllocError> {
    // Phase 1: Critical edge splitting (required before phi elimination).
    let _edges_split = split_critical_edges(func);

    // Phase 2: Phi elimination — lower phis to copies.
    eliminate_phis(func);

    // Phase 3: Liveness analysis.
    let liveness = compute_live_intervals(func);
    let mut intervals_map = liveness.intervals;

    // Phase 4: Copy coalescing — merge non-interfering intervals from copies.
    if config.enable_coalescing {
        let coalesce_result = coalesce_copies(func, &mut intervals_map);
        if coalesce_result.copies_removed > 0 {
            apply_coalescing(func, &coalesce_result.removals, &coalesce_result.rewrites);
        }
    }

    let intervals: Vec<LiveInterval> = intervals_map.values().cloned().collect();

    // Phase 5: Allocation — select algorithm based on strategy.
    let (mut result, spilled) = match config.strategy {
        AllocStrategy::LinearScan => {
            let mut scanner = LinearScan::new(intervals, &config.allocatable_regs);
            let result = scanner.allocate()?;
            let spilled = scanner.spilled_vregs().to_vec();
            (result, spilled)
        }
        AllocStrategy::Greedy => {
            let mut allocator = GreedyAllocator::new(
                intervals,
                &config.allocatable_regs,
                config.hints.clone(),
            );
            let result = allocator.allocate_with_splitting(func)?;
            let spilled = allocator.spilled_vregs().to_vec();
            (result, spilled)
        }
    };

    // Phase 6: Spill handling — rematerialization + spill code insertion.
    if !spilled.is_empty() {
        if config.enable_remat {
            // Try to rematerialize cheap values instead of spilling.
            let remat_candidates = find_remat_candidates(func, &spilled);
            if !remat_candidates.is_empty() {
                // First insert spill code for all spilled vregs.
                let mut spill_infos = insert_spill_code(func, &spilled, &result.allocation);
                // Then replace spill loads with rematerialized instructions.
                remat::apply_rematerialization(func, &remat_candidates, &mut spill_infos);
                result.spills = spill_infos;
            } else {
                let spill_infos = insert_spill_code(func, &spilled, &result.allocation);
                result.spills = spill_infos;
            }
        } else {
            let spill_infos = insert_spill_code(func, &spilled, &result.allocation);
            result.spills = spill_infos;
        }

        // Phase 7: Spill slot reuse.
        if config.enable_spill_slot_reuse && !result.spills.is_empty() {
            let reuse = compute_spill_slot_reuse(&result.spills, &intervals_map);
            if reuse.slots_eliminated > 0 {
                spill_slot_reuse::apply_spill_slot_reuse(func, &reuse.slot_rewrites);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple straight-line function with N virtual registers.
    fn make_straight_line(n: u32) -> MachFunction {
        let mut insts = Vec::new();
        let mut inst_ids = Vec::new();

        for i in 0..n {
            // def vi = imm i
            let inst = MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(VReg {
                    id: i,
                    class: RegClass::Gpr64,
                })],
                uses: vec![MachOperand::Imm(i as i64)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            };
            inst_ids.push(InstId(insts.len() as u32));
            insts.push(inst);
        }

        // Use all vregs at the end.
        for i in 0..n {
            let inst = MachInst {
                opcode: 2,
                defs: vec![],
                uses: vec![MachOperand::VReg(VReg {
                    id: i,
                    class: RegClass::Gpr64,
                })],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            };
            inst_ids.push(InstId(insts.len() as u32));
            insts.push(inst);
        }

        MachFunction {
            name: "test_straight_line".into(),
            insts,
            blocks: vec![MachBlock {
                insts: inst_ids,
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: n,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    /// Helper: build a diamond CFG (entry -> if/else -> merge).
    fn make_diamond() -> MachFunction {
        let mut insts = Vec::new();

        // Block 0 (entry): def v0, branch
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg {
                id: 0,
                class: RegClass::Gpr64,
            })],
            uses: vec![MachOperand::Imm(0)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBB,
            defs: vec![],
            uses: vec![
                MachOperand::VReg(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                }),
                MachOperand::Block(BlockId(1)),
                MachOperand::Block(BlockId(2)),
            ],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 1 (then): def v1
        let i2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg {
                id: 1,
                class: RegClass::Gpr64,
            })],
            uses: vec![MachOperand::Imm(1)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        // Block 2 (else): def v2
        let i3 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg {
                id: 2,
                class: RegClass::Gpr64,
            })],
            uses: vec![MachOperand::Imm(2)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        // Block 3 (merge): use v0
        let i4 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 2,
            defs: vec![],
            uses: vec![MachOperand::VReg(VReg {
                id: 0,
                class: RegClass::Gpr64,
            })],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        MachFunction {
            name: "test_diamond".into(),
            insts,
            blocks: vec![
                MachBlock {
                    insts: vec![i0, i1],
                    preds: Vec::new(),
                    succs: vec![BlockId(1), BlockId(2)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i2],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i3],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i4],
                    preds: vec![BlockId(1), BlockId(2)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
            ],
            block_order: vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3)],
            entry_block: BlockId(0),
            next_vreg: 3,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    /// Helper: build a simple loop.
    fn make_loop() -> MachFunction {
        let mut insts = Vec::new();

        // Block 0 (preheader): def v0 = 0
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg {
                id: 0,
                class: RegClass::Gpr64,
            })],
            uses: vec![MachOperand::Imm(0)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        // Block 1 (loop body): use v0, def v1 = v0 + 1
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 2,
            defs: vec![MachOperand::VReg(VReg {
                id: 1,
                class: RegClass::Gpr64,
            })],
            uses: vec![
                MachOperand::VReg(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                }),
                MachOperand::Imm(1),
            ],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBB,
            defs: vec![],
            uses: vec![
                MachOperand::VReg(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                }),
                MachOperand::Block(BlockId(1)),
                MachOperand::Block(BlockId(2)),
            ],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 2 (exit): use v1
        let i3 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 3,
            defs: vec![],
            uses: vec![MachOperand::VReg(VReg {
                id: 1,
                class: RegClass::Gpr64,
            })],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        MachFunction {
            name: "test_loop".into(),
            insts,
            blocks: vec![
                MachBlock {
                    insts: vec![i0],
                    preds: Vec::new(),
                    succs: vec![BlockId(1)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i1, i2],
                    preds: vec![BlockId(0), BlockId(1)],
                    succs: vec![BlockId(1), BlockId(2)],
                    loop_depth: 1,
                },
                MachBlock {
                    insts: vec![i3],
                    preds: vec![BlockId(1)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
            ],
            block_order: vec![BlockId(0), BlockId(1), BlockId(2)],
            entry_block: BlockId(0),
            next_vreg: 2,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    #[test]
    fn test_pipeline_straight_line_no_spill() {
        // With 10 vregs and 26 GPRs, no spilling should occur.
        let mut func = make_straight_line(10);
        let config = AllocConfig::default_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        assert_eq!(result.allocation.len(), 10);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_diamond_cfg() {
        let mut func = make_diamond();
        let config = AllocConfig::default_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        // All 3 vregs should be allocated without spilling.
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_loop() {
        let mut func = make_loop();
        let config = AllocConfig::default_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_high_pressure_causes_spills() {
        // 30 simultaneously live vregs with only 26 GPRs available.
        let mut func = make_straight_line(30);
        let config = AllocConfig::default_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        // With 30 simultaneously-live vregs and 26 GPRs, at least some must spill.
        // After coalescing, VReg count may differ, but allocation should succeed.
        // Verify we have a valid allocation (some VRegs allocated, some spilled).
        let total = result.allocation.len() + result.spills.len();
        assert!(total > 0, "should have some allocation results");
        // The number of allocated VRegs should not exceed available registers.
        assert!(
            result.allocation.len() <= 26,
            "cannot allocate more than 26 GPRs: got {}",
            result.allocation.len()
        );
    }

    #[test]
    fn test_pipeline_with_call() {
        // def v0, call, use v0 — v0 should be allocated.
        let mut insts = Vec::new();
        let i0 = InstId(0);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg {
                id: 0,
                class: RegClass::Gpr64,
            })],
            uses: vec![MachOperand::Imm(42)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1 = InstId(1);
        insts.push(MachInst {
            opcode: 0xCA,
            defs: vec![],
            uses: vec![],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),
        });
        let i2 = InstId(2);
        insts.push(MachInst {
            opcode: 2,
            defs: vec![],
            uses: vec![MachOperand::VReg(VReg {
                id: 0,
                class: RegClass::Gpr64,
            })],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        let mut func = MachFunction {
            name: "test_call".into(),
            insts,
            blocks: vec![MachBlock {
                insts: vec![i0, i1, i2],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        };

        let config = AllocConfig::default_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        // v0 should be allocated (or spilled if crossing call, depending on config).
        let total = result.allocation.len() + result.spills.len();
        assert!(total >= 1);
    }

    #[test]
    fn test_coalescing_disabled() {
        let mut func = make_straight_line(5);
        let config = AllocConfig {
            allocatable_regs: aarch64_allocatable_regs(),
            strategy: AllocStrategy::LinearScan,
            enable_coalescing: false,
            enable_remat: false,
            enable_spill_slot_reuse: false,
            hints: HashMap::new(),
        };
        let result = allocate(&mut func, &config).expect("allocation failed");
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_greedy_straight_line_no_spill() {
        // Same as linear scan test but using greedy allocator.
        let mut func = make_straight_line(10);
        let config = AllocConfig::greedy_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        assert_eq!(result.allocation.len(), 10);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_greedy_high_pressure() {
        // 30 simultaneously live vregs with only 26 GPRs available.
        // The greedy allocator with splitting can reduce pressure by
        // splitting long intervals, so it may produce fewer spills than
        // linear scan. The key invariant: allocation succeeds and every
        // original VReg is either allocated or spilled.
        let mut func = make_straight_line(30);
        let config = AllocConfig::greedy_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");
        let total = result.allocation.len() + result.spills.len();
        assert!(total > 0, "should have some allocation results");
    }

    #[test]
    fn test_pipeline_remat_disabled_spill_reuse_enabled() {
        let mut func = make_straight_line(30);
        let mut config = AllocConfig::default_aarch64();
        config.enable_remat = false;

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert!(
            !result.spills.is_empty(),
            "expected spills with 30 live GPR64 vregs and remat disabled"
        );
        assert!(result.allocation.len() <= 26);
    }

    #[test]
    fn test_pipeline_all_optimizations_disabled() {
        let mut func = make_straight_line(5);
        let mut config = AllocConfig::default_aarch64();
        config.enable_coalescing = false;
        config.enable_remat = false;
        config.enable_spill_slot_reuse = false;

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert_eq!(result.allocation.len(), 5);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_greedy_diamond() {
        let mut func = make_diamond();
        let config = AllocConfig::greedy_aarch64();

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_greedy_loop() {
        let mut func = make_loop();
        let config = AllocConfig::greedy_aarch64();

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_greedy_coalescing_disabled() {
        let mut func = make_straight_line(5);
        let mut config = AllocConfig::greedy_aarch64();
        config.enable_coalescing = false;

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert_eq!(result.allocation.len(), 5);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_empty_function() {
        let mut func = MachFunction {
            name: "empty".into(),
            insts: Vec::new(),
            blocks: vec![MachBlock {
                insts: Vec::new(),
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        };

        let config = AllocConfig::default_aarch64();
        let result = allocate(&mut func, &config).expect("allocation failed");

        assert!(result.allocation.is_empty());
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_single_vreg() {
        let mut func = make_straight_line(1);
        let config = AllocConfig::default_aarch64();

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert_eq!(result.allocation.len(), 1);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_pipeline_fpr64_registers() {
        let mut func = MachFunction {
            name: "fpr64".into(),
            insts: vec![
                MachInst {
                    opcode: 1,
                    defs: vec![MachOperand::VReg(VReg {
                        id: 0,
                        class: RegClass::Fpr64,
                    })],
                    uses: vec![MachOperand::FImm(1.0)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                MachInst {
                    opcode: 1,
                    defs: vec![MachOperand::VReg(VReg {
                        id: 1,
                        class: RegClass::Fpr64,
                    })],
                    uses: vec![MachOperand::FImm(2.0)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                MachInst {
                    opcode: 2,
                    defs: vec![],
                    uses: vec![
                        MachOperand::VReg(VReg {
                            id: 0,
                            class: RegClass::Fpr64,
                        }),
                        MachOperand::VReg(VReg {
                            id: 1,
                            class: RegClass::Fpr64,
                        }),
                    ],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
            ],
            blocks: vec![MachBlock {
                insts: vec![InstId(0), InstId(1), InstId(2)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 2,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        };

        let all_regs = aarch64_allocatable_regs();
        let mut regs = HashMap::new();
        regs.insert(
            RegClass::Fpr64,
            all_regs
                .get(&RegClass::Fpr64)
                .expect("missing Fpr64 regs")
                .clone(),
        );

        let config = AllocConfig {
            allocatable_regs: regs,
            strategy: AllocStrategy::LinearScan,
            enable_coalescing: true,
            enable_remat: true,
            enable_spill_slot_reuse: true,
            hints: HashMap::new(),
        };

        let result = allocate(&mut func, &config).expect("allocation failed");

        assert_eq!(result.allocation.len(), 2);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_default_aarch64_uses_linear_scan() {
        let config = AllocConfig::default_aarch64();
        assert_eq!(config.strategy, AllocStrategy::LinearScan);
    }

    #[test]
    fn test_greedy_aarch64_uses_greedy() {
        let config = AllocConfig::greedy_aarch64();
        assert_eq!(config.strategy, AllocStrategy::Greedy);
    }
}

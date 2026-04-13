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
//! MachFunction (input, SSA with phis)
//!      │
//!      ▼
//! ┌─────────────────┐
//! │ Critical Edge    │  split_critical_edges()
//! │ Splitting        │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Phi Elimination  │  eliminate_phis()
//! │ (parallel copies)│
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Liveness         │  compute_live_intervals()
//! │ Analysis         │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Linear Scan      │  LinearScan::allocate()
//! │ Allocation       │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Spill Code       │  insert_spill_code()
//! │ Insertion        │
//! └────────┬────────┘
//!          │
//!          ▼
//! MachFunction (output, VRegs replaced with PRegs)
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llvm2_regalloc::{allocate, AllocConfig};
//! use llvm2_regalloc::machine_types::MachFunction;
//!
//! # fn example(mut func: MachFunction) {
//! let config = AllocConfig::default_aarch64();
//! let result = allocate(&mut func, &config).expect("allocation failed");
//! # }
//! ```
//!
//! ## Future work
//!
//! - Interval splitting for better spill placement
//! - Rematerialization for constants and addresses
//! - Spill-slot reuse (share slots for non-overlapping intervals)
//! - Copy coalescing after phi elimination and spill rewrite
//! - LLVM-style greedy allocator (Phase 2)

pub mod machine_types;
pub mod liveness;
pub mod linear_scan;
pub mod spill;
pub mod phi_elim;

pub use liveness::{compute_live_intervals, LiveInterval, LiveRange, LivenessResult};
pub use linear_scan::{aarch64_allocatable_regs, AllocError, AllocationResult, LinearScan, SpillInfo};
pub use machine_types::{
    InstFlags, MachBlock, MachFunction, MachInst, MachOperand, StackSlot,
};
// Re-export canonical types from llvm2-ir via machine_types.
pub use machine_types::{BlockId, InstId, PReg, RegClass, StackSlotId, VReg};
pub use phi_elim::{eliminate_phis, split_critical_edges};
pub use spill::insert_spill_code;

use std::collections::HashMap;

/// Configuration for the register allocator.
pub struct AllocConfig {
    /// Allocatable physical registers per register class.
    pub allocatable_regs: HashMap<RegClass, Vec<PReg>>,
}

impl AllocConfig {
    /// Default configuration for AArch64 (Apple calling convention).
    pub fn default_aarch64() -> Self {
        Self {
            allocatable_regs: aarch64_allocatable_regs(),
        }
    }
}

/// Main entry point: run the full register allocation pipeline.
///
/// This function:
/// 1. Splits critical edges.
/// 2. Eliminates phi instructions (inserts parallel copies).
/// 3. Computes live intervals.
/// 4. Runs linear scan allocation.
/// 5. Inserts spill code for any spilled VRegs.
///
/// Returns the allocation result with VReg-to-PReg mappings and spill info.
pub fn allocate(
    func: &mut MachFunction,
    config: &AllocConfig,
) -> Result<AllocationResult, AllocError> {
    // Phase 1: Critical edge splitting (required before phi elimination).
    let _edges_split = split_critical_edges(func);

    // Phase 2: Phi elimination — lower phis to copies.
    eliminate_phis(func);

    // Phase 3: Liveness analysis.
    let liveness = compute_live_intervals(func);
    let intervals: Vec<LiveInterval> = liveness.intervals.into_values().collect();

    // Phase 4: Linear scan allocation.
    let mut scanner = LinearScan::new(intervals, &config.allocatable_regs);
    let mut result = scanner.allocate()?;

    // Phase 5: Spill code insertion.
    let spilled = scanner.spilled_vregs().to_vec();
    if !spilled.is_empty() {
        let spill_infos = insert_spill_code(func, &spilled, &result.allocation);
        result.spills = spill_infos;
    }

    Ok(result)
}

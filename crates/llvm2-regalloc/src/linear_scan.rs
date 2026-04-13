// llvm2-regalloc/linear_scan.rs - Linear scan register allocator
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Linear scan register allocator.
//!
//! Assigns physical registers to virtual registers by processing live
//! intervals sorted by start position. When no register is available,
//! the interval with the lowest spill weight is spilled.
//!
//! Reference: Poletto & Sarkar, "Linear Scan Register Allocation" (1999)
//! LLVM reference: `~/llvm-project-ref/llvm/lib/CodeGen/RegAllocGreedy.cpp`
//!
//! Current implementation: basic linear scan without interval splitting.
//! Future work (Phase 2): interval splitting, rematerialization, hints.

use crate::liveness::LiveInterval;
use crate::machine_types::{PReg, RegClass, StackSlotId, VReg};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during register allocation.
#[derive(Debug, Error)]
pub enum AllocError {
    #[error("cannot spill fixed interval for {0}")]
    CannotSpillFixed(String),
    #[error("no allocatable registers for class {0:?}")]
    NoRegistersForClass(RegClass),
    #[error("register allocation failed: {0}")]
    Failed(String),
}

/// Result of register allocation.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// VReg -> PReg assignment for successfully allocated registers.
    pub allocation: HashMap<VReg, PReg>,
    /// VRegs that were spilled, with their assigned stack slots.
    pub spills: Vec<SpillInfo>,
}

/// Information about a spilled virtual register.
#[derive(Debug, Clone)]
pub struct SpillInfo {
    /// The spilled virtual register.
    pub vreg: VReg,
    /// The stack slot assigned for this spill.
    pub slot: StackSlotId,
}

/// Linear scan register allocator.
///
/// The algorithm processes intervals sorted by start position:
/// 1. For each interval starting at position `pos`:
///    a. Expire any active intervals that end before `pos`.
///    b. Try to find a free register from the allocatable set.
///    c. If no free register, spill the active interval with the lowest
///       spill weight (or the current interval if it's cheaper).
///
/// Reference: LLVM's RegAllocLinearScan (removed in favor of Greedy,
/// but the core algorithm is a useful starting point).
pub struct LinearScan {
    /// All live intervals, sorted by start position.
    intervals: Vec<LiveInterval>,
    /// Indices into `intervals` of currently active (allocated) intervals,
    /// sorted by end position.
    active: Vec<usize>,
    /// Available physical registers per register class.
    allocatable: HashMap<RegClass, Vec<PReg>>,
    /// Current allocation: VReg -> PReg.
    allocation: HashMap<VReg, PReg>,
    /// VRegs that need spilling.
    spills: Vec<VReg>,
    /// Free register pool per class: tracks which PRegs are currently unassigned.
    free_regs: HashMap<RegClass, Vec<PReg>>,
}

impl LinearScan {
    /// Create a new linear scan allocator.
    ///
    /// `intervals`: computed live intervals for the function.
    /// `target_regs`: allocatable physical registers, organized by class.
    pub fn new(
        mut intervals: Vec<LiveInterval>,
        target_regs: &HashMap<RegClass, Vec<PReg>>,
    ) -> Self {
        // Sort intervals by start position (ascending).
        intervals.sort_by_key(|i| i.start());

        let free_regs = target_regs.clone();

        Self {
            intervals,
            active: Vec::new(),
            allocatable: target_regs.clone(),
            allocation: HashMap::new(),
            spills: Vec::new(),
            free_regs,
        }
    }

    /// Run the linear scan allocation algorithm.
    pub fn allocate(&mut self) -> Result<AllocationResult, AllocError> {
        for i in 0..self.intervals.len() {
            let start = self.intervals[i].start();

            // Step 1: Expire old intervals.
            self.expire_old_intervals(start);

            // Step 2: Skip fixed intervals (already allocated).
            if self.intervals[i].is_fixed {
                continue;
            }

            let class = self.intervals[i].vreg.class;

            // Step 3: Try to allocate a free register.
            if let Some(preg) = self.try_alloc_free_reg(class) {
                self.allocation.insert(self.intervals[i].vreg, preg);
                self.insert_active(i);
            } else {
                // Step 4: No free register — spill something.
                self.allocate_blocked_reg(i)?;
            }
        }

        Ok(AllocationResult {
            allocation: self.allocation.clone(),
            spills: Vec::new(), // Spill info filled in by insert_spill_code
        })
    }

    /// Expire active intervals that end before `pos`.
    fn expire_old_intervals(&mut self, pos: u32) {
        let mut expired = Vec::new();

        for (active_idx, &interval_idx) in self.active.iter().enumerate() {
            if self.intervals[interval_idx].end() <= pos {
                expired.push(active_idx);
                // Return the register to the free pool.
                let vreg = self.intervals[interval_idx].vreg;
                if let Some(preg) = self.allocation.get(&vreg) {
                    self.free_regs
                        .entry(vreg.class)
                        .or_default()
                        .push(*preg);
                }
            }
        }

        // Remove expired entries in reverse order to preserve indices.
        for idx in expired.into_iter().rev() {
            self.active.remove(idx);
        }
    }

    /// Try to allocate a free register from the given class.
    fn try_alloc_free_reg(&mut self, class: RegClass) -> Option<PReg> {
        self.free_regs.get_mut(&class)?.pop()
    }

    /// Handle the case where no free register is available.
    ///
    /// Spill the interval (current or active) with the lowest spill weight.
    fn allocate_blocked_reg(&mut self, current_idx: usize) -> Result<(), AllocError> {
        let class = self.intervals[current_idx].vreg.class;

        // Find the active interval with the lowest spill weight in the same class.
        let mut spill_candidate: Option<(usize, f64)> = None;
        for &active_interval_idx in &self.active {
            let interval = &self.intervals[active_interval_idx];
            if interval.vreg.class == class && !interval.is_fixed {
                match spill_candidate {
                    None => {
                        spill_candidate = Some((active_interval_idx, interval.spill_weight));
                    }
                    Some((_, weight)) if interval.spill_weight < weight => {
                        spill_candidate = Some((active_interval_idx, interval.spill_weight));
                    }
                    _ => {}
                }
            }
        }

        let current_weight = self.intervals[current_idx].spill_weight;

        match spill_candidate {
            Some((spill_idx, spill_weight)) if spill_weight < current_weight => {
                // Spill the active interval and give its register to current.
                let spill_vreg = self.intervals[spill_idx].vreg;
                let preg = self.allocation.remove(&spill_vreg).ok_or_else(|| {
                    AllocError::Failed(format!(
                        "active interval {} has no allocation",
                        spill_vreg
                    ))
                })?;

                self.spills.push(spill_vreg);
                self.allocation.insert(self.intervals[current_idx].vreg, preg);

                // Remove spilled interval from active.
                self.active.retain(|&idx| idx != spill_idx);
                self.insert_active(current_idx);

                Ok(())
            }
            _ => {
                // Spill the current interval.
                self.spills.push(self.intervals[current_idx].vreg);
                Ok(())
            }
        }
    }

    /// Insert an interval index into the active list, maintaining sort by end position.
    fn insert_active(&mut self, interval_idx: usize) {
        let end = self.intervals[interval_idx].end();
        let pos = self
            .active
            .iter()
            .position(|&idx| self.intervals[idx].end() > end)
            .unwrap_or(self.active.len());
        self.active.insert(pos, interval_idx);
    }

    /// Returns the list of VRegs that were spilled during allocation.
    pub fn spilled_vregs(&self) -> &[VReg] {
        &self.spills
    }
}

/// Returns the default allocatable registers for AArch64 (Apple calling convention).
///
/// Caller-saved: X0-X15 (excluding X18 which is reserved on Apple).
/// Callee-saved: X19-X28.
/// X29 = FP, X30 = LR, X31 = SP/ZR — not allocatable.
pub fn aarch64_allocatable_regs() -> HashMap<RegClass, Vec<PReg>> {
    let mut regs = HashMap::new();

    // GPR64: X0-X15, X19-X28 (skip X16-X17 scratch, X18 reserved, X29 FP, X30 LR)
    let gpr64: Vec<PReg> = (0..=15)
        .chain(19..=28)
        .map(PReg)
        .collect();
    regs.insert(RegClass::Gpr64, gpr64.clone());

    // GPR32: same set (W registers are the lower 32 bits of X registers).
    regs.insert(RegClass::Gpr32, gpr64);

    // FPR64 and FPR32: V0-V31 (encoded as 32-63).
    let fpr: Vec<PReg> = (32..=63).map(PReg).collect();
    regs.insert(RegClass::Fpr64, fpr.clone());
    regs.insert(RegClass::Fpr32, fpr.clone());
    regs.insert(RegClass::Vec128, fpr);

    regs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aarch64_allocatable_regs() {
        let regs = aarch64_allocatable_regs();
        // 26 GPRs: X0-X15 (16) + X19-X28 (10) = 26
        assert_eq!(regs[&RegClass::Gpr64].len(), 26);
        // 32 FPRs: V0-V31
        assert_eq!(regs[&RegClass::Fpr64].len(), 32);
    }

    #[test]
    fn test_simple_allocation() {
        let regs = aarch64_allocatable_regs();
        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 10);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(5, 15);
                i.spill_weight = 2.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        // Both should be allocated to different registers (plenty available).
        assert_eq!(result.allocation.len(), 2);
        let preg0 = result.allocation[&VReg { id: 0, class: RegClass::Gpr64 }];
        let preg1 = result.allocation[&VReg { id: 1, class: RegClass::Gpr64 }];
        assert_ne!(preg0, preg1);
    }
}

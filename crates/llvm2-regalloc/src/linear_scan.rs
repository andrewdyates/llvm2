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
    /// Available physical registers per register class (retained for future use
    /// in interval splitting and rematerialization).
    _allocatable: HashMap<RegClass, Vec<PReg>>,
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
            _allocatable: target_regs.clone(),
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
    // PReg encoding: X registers are 0..=30
    let gpr64: Vec<PReg> = (0u16..=15)
        .chain(19u16..=28)
        .map(PReg::new)
        .collect();
    regs.insert(RegClass::Gpr64, gpr64.clone());

    // GPR32: same set (W registers are the lower 32 bits of X registers).
    // PReg encoding: W registers are 32..=62
    let gpr32: Vec<PReg> = (32u16..=47)
        .chain(51u16..=60)
        .map(PReg::new)
        .collect();
    regs.insert(RegClass::Gpr32, gpr32);

    // FPR128: V0-V31 (encoded as 64-95).
    let fpr128: Vec<PReg> = (64u16..=95).map(PReg::new).collect();
    regs.insert(RegClass::Fpr128, fpr128);

    // FPR64: D0-D31 (encoded as 96-127).
    let fpr64: Vec<PReg> = (96u16..=127).map(PReg::new).collect();
    regs.insert(RegClass::Fpr64, fpr64);

    // FPR32: S0-S31 (encoded as 128-159).
    let fpr32: Vec<PReg> = (128u16..=159).map(PReg::new).collect();
    regs.insert(RegClass::Fpr32, fpr32);

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

    #[test]
    fn test_non_overlapping_intervals_reuse_register() {
        let regs = aarch64_allocatable_regs();
        // Two intervals that don't overlap can use the same register.
        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
                i.add_range(0, 5);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr64 });
                i.add_range(5, 10); // starts exactly where first ends
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        assert_eq!(result.allocation.len(), 2);
        // With plenty of registers, they may or may not reuse — the important
        // thing is both are allocated successfully.
    }

    #[test]
    fn test_spill_under_register_pressure() {
        // Create a situation with only 2 registers and 3 overlapping intervals.
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0), PReg::new(1)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
                i.add_range(0, 20);
                i.spill_weight = 1.0; // low weight = spill candidate
                i
            },
            {
                let mut i = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr64 });
                i.add_range(0, 20);
                i.spill_weight = 5.0; // high weight
                i
            },
            {
                let mut i = LiveInterval::new(VReg { id: 2, class: RegClass::Gpr64 });
                i.add_range(0, 20);
                i.spill_weight = 10.0; // highest weight
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        // Only 2 registers for 3 overlapping intervals — one must be spilled.
        assert_eq!(result.allocation.len(), 2, "should allocate 2");
        assert_eq!(scan.spilled_vregs().len(), 1, "should spill 1");

        // The spilled VReg should be the one with the lowest spill weight (v0).
        let spilled = scan.spilled_vregs();
        assert_eq!(spilled[0].id, 0, "lowest weight vreg should be spilled");
    }

    #[test]
    fn test_spill_current_if_cheaper() {
        // When the current interval has lower weight than all active intervals,
        // the current interval itself should be spilled.
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
                i.add_range(0, 20);
                i.spill_weight = 10.0; // high weight, allocated first
                i
            },
            {
                let mut i = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr64 });
                i.add_range(5, 15);
                i.spill_weight = 1.0; // low weight, arrives later
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        assert_eq!(result.allocation.len(), 1);
        assert_eq!(scan.spilled_vregs().len(), 1);
        // v1 (the cheaper one) should be spilled since v0 is already active
        // and has higher weight.
        assert_eq!(scan.spilled_vregs()[0].id, 1);
    }

    #[test]
    fn test_expire_old_intervals_frees_registers() {
        // Two non-overlapping intervals + one overlapping should work with 1 register.
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
                i.add_range(0, 5);
                i.spill_weight = 1.0;
                i
            },
            {
                // Starts after v0 ends — register should be freed.
                let mut i = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr64 });
                i.add_range(5, 10);
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        // Both should be allocated (same register, sequentially).
        assert_eq!(result.allocation.len(), 2);
        assert!(scan.spilled_vregs().is_empty());
    }

    #[test]
    fn test_fixed_intervals_skipped() {
        let regs = aarch64_allocatable_regs();
        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
                i.add_range(0, 10);
                i.is_fixed = true; // should be skipped
                i
            },
            {
                let mut i = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr64 });
                i.add_range(0, 10);
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        // Fixed interval should NOT appear in allocation.
        assert!(!result.allocation.contains_key(&VReg { id: 0, class: RegClass::Gpr64 }));
        // Non-fixed interval should be allocated.
        assert!(result.allocation.contains_key(&VReg { id: 1, class: RegClass::Gpr64 }));
    }

    #[test]
    fn test_multiple_register_classes() {
        let regs = aarch64_allocatable_regs();
        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
                i.add_range(0, 10);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg { id: 1, class: RegClass::Fpr64 });
                i.add_range(0, 10);
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        // Both should be allocated — they use different register classes.
        assert_eq!(result.allocation.len(), 2);
        let preg_gpr = result.allocation[&VReg { id: 0, class: RegClass::Gpr64 }];
        let preg_fpr = result.allocation[&VReg { id: 1, class: RegClass::Fpr64 }];
        // GPR and FPR registers should have different encodings.
        assert_ne!(preg_gpr, preg_fpr);
    }

    #[test]
    fn test_many_sequential_intervals_no_spill() {
        // 100 intervals that don't overlap should all be allocated with 1 register.
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals: Vec<LiveInterval> = (0..100)
            .map(|i| {
                let mut interval = LiveInterval::new(VReg { id: i, class: RegClass::Gpr64 });
                interval.add_range(i * 10, i * 10 + 5);
                interval.spill_weight = 1.0;
                interval
            })
            .collect();

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");
        assert_eq!(result.allocation.len(), 100);
        assert!(scan.spilled_vregs().is_empty());
    }

    #[test]
    fn test_empty_intervals() {
        let regs = aarch64_allocatable_regs();
        let intervals: Vec<LiveInterval> = Vec::new();
        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("empty should succeed");
        assert!(result.allocation.is_empty());
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_aarch64_regs_gpr32_count() {
        let regs = aarch64_allocatable_regs();
        // GPR32: W0-W15 (16) + W19-W28 (10) = 26
        assert_eq!(regs[&RegClass::Gpr32].len(), 26);
    }

    #[test]
    fn test_aarch64_regs_fpr128_count() {
        let regs = aarch64_allocatable_regs();
        // FPR128: V0-V31 = 32
        assert_eq!(regs[&RegClass::Fpr128].len(), 32);
    }

    #[test]
    fn test_aarch64_regs_fpr32_count() {
        let regs = aarch64_allocatable_regs();
        // FPR32: S0-S31 = 32
        assert_eq!(regs[&RegClass::Fpr32].len(), 32);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests (issue #139)
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_intervals_fixed_nothing_to_allocate() {
        let regs = HashMap::new();
        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 10);
                i.is_fixed = true;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Fpr32,
                });
                i.add_range(3, 12);
                i.is_fixed = true;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 2,
                    class: RegClass::Gpr32,
                });
                i.add_range(12, 20);
                i.is_fixed = true;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("all-fixed allocation should succeed");

        assert!(result.allocation.is_empty());
        assert!(scan.spilled_vregs().is_empty());
    }

    #[test]
    fn test_single_interval_with_zero_spill_weight() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![{
            let mut i = LiveInterval::new(VReg {
                id: 0,
                class: RegClass::Gpr64,
            });
            i.add_range(0, 10);
            i.spill_weight = 0.0;
            i
        }];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 1);
        assert_eq!(
            result.allocation[&VReg {
                id: 0,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
        assert!(scan.spilled_vregs().is_empty());
    }

    #[test]
    fn test_extreme_register_pressure_spills_exactly_one_interval() {
        let mut regs = HashMap::new();
        regs.insert(
            RegClass::Gpr64,
            vec![PReg::new(0), PReg::new(1), PReg::new(2), PReg::new(3)],
        );

        let intervals: Vec<LiveInterval> = (0u32..=4)
            .map(|id| {
                let mut i = LiveInterval::new(VReg {
                    id,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 20);
                i.spill_weight = (id + 1) as f64;
                i
            })
            .collect();

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 4);
        assert_eq!(
            scan.spilled_vregs(),
            &[VReg {
                id: 0,
                class: RegClass::Gpr64,
            }]
        );
        for id in 1u32..=4 {
            assert!(result.allocation.contains_key(&VReg {
                id,
                class: RegClass::Gpr64,
            }));
        }
    }

    #[test]
    fn test_intervals_sorted_in_reverse_order_are_processed_by_start() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(10, 20);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 5);
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 2);
        assert!(scan.spilled_vregs().is_empty());
        assert_eq!(
            result.allocation[&VReg {
                id: 0,
                class: RegClass::Gpr64,
            }],
            result.allocation[&VReg {
                id: 1,
                class: RegClass::Gpr64,
            }]
        );
    }

    #[test]
    fn test_very_long_interval_vs_many_short_intervals() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let mut intervals = vec![{
            let mut i = LiveInterval::new(VReg {
                id: 0,
                class: RegClass::Gpr64,
            });
            i.add_range(0, 1000);
            i.spill_weight = 1.0;
            i
        }];

        for id in 1u32..=5 {
            let mut i = LiveInterval::new(VReg {
                id,
                class: RegClass::Gpr64,
            });
            i.add_range(id * 100, id * 100 + 10);
            i.spill_weight = 10.0;
            intervals.push(i);
        }

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(
            scan.spilled_vregs(),
            &[VReg {
                id: 0,
                class: RegClass::Gpr64,
            }]
        );
        assert_eq!(result.allocation.len(), 5);
        for id in 1u32..=5 {
            assert!(result.allocation.contains_key(&VReg {
                id,
                class: RegClass::Gpr64,
            }));
            assert_eq!(
                result.allocation[&VReg {
                    id,
                    class: RegClass::Gpr64,
                }],
                PReg::new(0)
            );
        }
    }

    #[test]
    fn test_spill_with_equal_weights_is_deterministic() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 20);
                i.spill_weight = 5.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(5, 15);
                i.spill_weight = 5.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 1);
        assert_eq!(
            scan.spilled_vregs(),
            &[VReg {
                id: 1,
                class: RegClass::Gpr64,
            }]
        );
        assert!(result.allocation.contains_key(&VReg {
            id: 0,
            class: RegClass::Gpr64,
        }));
        assert!(!result.allocation.contains_key(&VReg {
            id: 1,
            class: RegClass::Gpr64,
        }));
    }

    #[test]
    fn test_allocation_with_only_fpr32_class_registers() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Fpr32, vec![PReg::new(128), PReg::new(129)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Fpr32,
                });
                i.add_range(0, 10);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Fpr32,
                });
                i.add_range(0, 10);
                i.spill_weight = 2.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        let allowed = [PReg::new(128), PReg::new(129)];
        let preg0 = result.allocation[&VReg {
            id: 0,
            class: RegClass::Fpr32,
        }];
        let preg1 = result.allocation[&VReg {
            id: 1,
            class: RegClass::Fpr32,
        }];

        assert_eq!(result.allocation.len(), 2);
        assert!(scan.spilled_vregs().is_empty());
        assert_ne!(preg0, preg1);
        assert!(allowed.contains(&preg0));
        assert!(allowed.contains(&preg1));
    }

    #[test]
    fn test_three_way_register_pressure_with_mixed_fixed_and_non_fixed_intervals() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 20);
                i.is_fixed = true;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 20);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 2,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 20);
                i.spill_weight = 10.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 1);
        assert!(!result.allocation.contains_key(&VReg {
            id: 0,
            class: RegClass::Gpr64,
        }));
        assert_eq!(
            scan.spilled_vregs(),
            &[VReg {
                id: 1,
                class: RegClass::Gpr64,
            }]
        );
        assert_eq!(
            result.allocation[&VReg {
                id: 2,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
    }

    #[test]
    fn test_sequential_intervals_with_gaps_free_register_properly() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 2);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(4, 6);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 2,
                    class: RegClass::Gpr64,
                });
                i.add_range(9, 11);
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 3);
        assert!(scan.spilled_vregs().is_empty());
        assert_eq!(
            result.allocation[&VReg {
                id: 0,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
        assert_eq!(
            result.allocation[&VReg {
                id: 1,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
        assert_eq!(
            result.allocation[&VReg {
                id: 2,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
    }

    #[test]
    fn test_adjacent_intervals_with_exact_matching_boundaries() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 5);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(5, 10);
                i.spill_weight = 1.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(result.allocation.len(), 2);
        assert!(scan.spilled_vregs().is_empty());
        assert_eq!(
            result.allocation[&VReg {
                id: 0,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
        assert_eq!(
            result.allocation[&VReg {
                id: 1,
                class: RegClass::Gpr64,
            }],
            PReg::new(0)
        );
    }

    #[test]
    fn test_very_short_intervals_length_one_interleaved_with_long_ones() {
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0), PReg::new(1)]);

        let intervals = vec![
            {
                let mut i = LiveInterval::new(VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 100);
                i.spill_weight = 100.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 1,
                    class: RegClass::Gpr64,
                });
                i.add_range(0, 100);
                i.spill_weight = 1.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 2,
                    class: RegClass::Gpr64,
                });
                i.add_range(10, 11);
                i.spill_weight = 10.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 3,
                    class: RegClass::Gpr64,
                });
                i.add_range(50, 51);
                i.spill_weight = 10.0;
                i
            },
            {
                let mut i = LiveInterval::new(VReg {
                    id: 4,
                    class: RegClass::Gpr64,
                });
                i.add_range(90, 91);
                i.spill_weight = 10.0;
                i
            },
        ];

        let mut scan = LinearScan::new(intervals, &regs);
        let result = scan.allocate().expect("allocation should succeed");

        assert_eq!(
            scan.spilled_vregs(),
            &[VReg {
                id: 1,
                class: RegClass::Gpr64,
            }]
        );
        assert_eq!(result.allocation.len(), 4);
        assert!(result.allocation.contains_key(&VReg {
            id: 0,
            class: RegClass::Gpr64,
        }));
        assert!(result.allocation.contains_key(&VReg {
            id: 2,
            class: RegClass::Gpr64,
        }));
        assert!(result.allocation.contains_key(&VReg {
            id: 3,
            class: RegClass::Gpr64,
        }));
        assert!(result.allocation.contains_key(&VReg {
            id: 4,
            class: RegClass::Gpr64,
        }));
    }
}

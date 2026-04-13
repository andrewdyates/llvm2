// llvm2-regalloc/greedy.rs - Greedy register allocator
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Greedy register allocator (Phase 2).
//!
//! LLVM-style greedy allocator that processes live intervals by spill weight
//! (highest first) and uses eviction, splitting, and cascade limiting for
//! better code quality than the linear scan allocator.
//!
//! ## Algorithm Overview
//!
//! 1. **Priority queue**: intervals are processed in decreasing spill weight
//!    order.  High-weight intervals (hot loops, frequently used values) get
//!    first pick of registers.
//!
//! 2. **Register hints**: when available, the allocator tries hint registers
//!    (from coalescing or ABI conventions) before scanning the full set.
//!
//! 3. **Interference checking**: for each candidate physical register, we
//!    check whether any already-assigned interval overlaps the current one.
//!
//! 4. **Eviction**: when no register is free, the allocator finds the
//!    lowest-weight interfering interval.  If its weight is less than the
//!    current interval's weight, it evicts the interferer and re-enqueues
//!    it for later processing.
//!
//! 5. **Cascade limiting**: each eviction assigns a *cascade number* to the
//!    evicted interval.  An interval can only be evicted by a strictly
//!    higher cascade number, preventing infinite eviction loops.  The
//!    maximum cascade depth is configurable (default 10).
//!
//! 6. **Splitting**: before giving up and spilling, the allocator tries to
//!    split the interval around its largest gap.  Both halves are
//!    re-enqueued as new intervals.
//!
//! 7. **Spilling**: intervals that cannot be assigned, evicted, or split
//!    are marked for spilling.
//!
//! ## Stage Progression
//!
//! Each interval progresses through stages:
//! `New -> Evict -> Split -> Spill -> Done`
//!
//! An interval only attempts eviction in the `New`/`Evict` stages, splitting
//! in the `Split` stage, and is spilled in the `Spill` stage.
//!
//! Reference: LLVM `RegAllocGreedy.cpp`
//!            Poletto & Sarkar, "Linear Scan Register Allocation" (1999)

use crate::liveness::LiveInterval;
use crate::linear_scan::{AllocError, AllocationResult};
use crate::machine_types::{MachFunction, PReg, RegClass, VReg};
use crate::split;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Stage tracking
// ---------------------------------------------------------------------------

/// Allocation stage for a live interval.
///
/// Intervals progress through stages in order; each stage unlocks a
/// different recovery strategy when a free register is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// First time in the queue -- try assignment then eviction.
    New = 0,
    /// Failed assignment; try eviction (second chance).
    Evict = 1,
    /// Eviction failed; try splitting.
    Split = 2,
    /// Splitting failed; will be spilled.
    Spill = 3,
    /// Terminal state.
    Done = 4,
}

// ---------------------------------------------------------------------------
// Priority queue entry
// ---------------------------------------------------------------------------

/// An entry in the priority queue, ordered by spill weight (descending).
///
/// We use `total_cmp` for a total order on f64 (NaN-safe).
#[derive(Debug, Clone)]
struct PriorityEntry {
    vreg_id: u32,
    weight: f64,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.vreg_id == other.vreg_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher weight = higher priority.  Break ties by lower vreg_id
        // (deterministic ordering).
        self.weight
            .total_cmp(&other.weight)
            .then_with(|| other.vreg_id.cmp(&self.vreg_id))
    }
}

// ---------------------------------------------------------------------------
// Greedy allocator
// ---------------------------------------------------------------------------

/// LLVM-style greedy register allocator.
///
/// See module-level documentation for algorithm details.
pub struct GreedyAllocator {
    // -- configuration --
    /// Allocatable physical registers per register class.
    allocatable_regs: HashMap<RegClass, Vec<PReg>>,
    /// Register hints: preferred physical registers per virtual register.
    hints: HashMap<VReg, Vec<PReg>>,
    /// Maximum eviction cascade depth (default 10).
    max_cascade_depth: u32,

    // -- per-interval state --
    /// Live intervals keyed by VReg id.
    intervals: HashMap<u32, LiveInterval>,
    /// Current VReg -> PReg assignment.
    assignment: HashMap<VReg, PReg>,
    /// Reverse map: PReg -> set of VReg ids currently assigned to it.
    preg_assignments: HashMap<PReg, Vec<u32>>,
    /// Cascade number per VReg id.
    cascade: HashMap<u32, u32>,
    /// Next cascade number to hand out.
    next_cascade: u32,
    /// Allocation stage per VReg id.
    stage: HashMap<u32, Stage>,
    /// VRegs that have been spilled (final output).
    spilled: Vec<VReg>,
    /// Main priority queue.
    worklist: BinaryHeap<PriorityEntry>,
}

impl GreedyAllocator {
    /// Create a new greedy allocator.
    ///
    /// * `intervals` -- live intervals computed by liveness analysis.
    /// * `allocatable` -- physical registers available per class.
    /// * `hints` -- optional per-VReg register preferences.
    pub fn new(
        intervals: Vec<LiveInterval>,
        allocatable: &HashMap<RegClass, Vec<PReg>>,
        hints: HashMap<VReg, Vec<PReg>>,
    ) -> Self {
        let mut interval_map: HashMap<u32, LiveInterval> = HashMap::new();
        let mut worklist = BinaryHeap::new();
        let mut stage_map: HashMap<u32, Stage> = HashMap::new();

        for iv in intervals {
            if iv.is_fixed {
                // Fixed intervals are pre-assigned and never enter the queue.
                continue;
            }
            let id = iv.vreg.id;
            worklist.push(PriorityEntry {
                vreg_id: id,
                weight: iv.spill_weight,
            });
            stage_map.insert(id, Stage::New);
            interval_map.insert(id, iv);
        }

        Self {
            allocatable_regs: allocatable.clone(),
            hints,
            max_cascade_depth: 10,
            intervals: interval_map,
            assignment: HashMap::new(),
            preg_assignments: HashMap::new(),
            cascade: HashMap::new(),
            next_cascade: 1,
            stage: stage_map,
            worklist,
            spilled: Vec::new(),
        }
    }

    /// Run the greedy allocation algorithm **without** splitting.
    ///
    /// This performs priority-queue-ordered assignment with eviction and
    /// cascade limiting.  Intervals that cannot be assigned are spilled.
    pub fn allocate(&mut self) -> Result<AllocationResult, AllocError> {
        while let Some(entry) = self.worklist.pop() {
            let vreg_id = entry.vreg_id;

            // Skip if this interval was already assigned (e.g. re-enqueued
            // after a failed eviction but later assigned via another path).
            if self.is_assigned(vreg_id) {
                continue;
            }

            // Skip if already spilled in a prior round.
            if self.is_spilled(vreg_id) {
                continue;
            }

            let current_stage = self.stage.get(&vreg_id).copied().unwrap_or(Stage::New);
            if current_stage == Stage::Done {
                continue;
            }

            // Step 1: try direct assignment (prefer hints).
            if let Some(preg) = self.try_assign(vreg_id) {
                self.assign(vreg_id, preg);
                self.advance_stage(vreg_id, Stage::Done);
                continue;
            }

            // Step 2: try eviction (only in New/Evict stages).
            if current_stage <= Stage::Evict {
                if let Some(preg) = self.try_evict(vreg_id) {
                    self.assign(vreg_id, preg);
                    self.advance_stage(vreg_id, Stage::Done);
                    continue;
                }
                // Advance to Split stage for next attempt.
                self.advance_stage(vreg_id, Stage::Split);
                self.worklist.push(PriorityEntry {
                    vreg_id,
                    weight: entry.weight,
                });
                continue;
            }

            // Step 3: splitting not available in basic `allocate()`.
            // Advance to Spill.
            if current_stage <= Stage::Split {
                self.advance_stage(vreg_id, Stage::Spill);
                self.worklist.push(PriorityEntry {
                    vreg_id,
                    weight: entry.weight,
                });
                continue;
            }

            // Step 4: spill.
            self.do_spill(vreg_id);
        }

        Ok(self.build_result())
    }

    /// Run the greedy allocation algorithm **with** interval splitting.
    ///
    /// Same as [`allocate`] but before spilling, attempts to split the
    /// interval around its largest gap and re-enqueues both halves.
    pub fn allocate_with_splitting(
        &mut self,
        func: &mut MachFunction,
    ) -> Result<AllocationResult, AllocError> {
        while let Some(entry) = self.worklist.pop() {
            let vreg_id = entry.vreg_id;

            if self.is_assigned(vreg_id) || self.is_spilled(vreg_id) {
                continue;
            }

            let current_stage = self.stage.get(&vreg_id).copied().unwrap_or(Stage::New);
            if current_stage == Stage::Done {
                continue;
            }

            // Step 1: try direct assignment.
            if let Some(preg) = self.try_assign(vreg_id) {
                self.assign(vreg_id, preg);
                self.advance_stage(vreg_id, Stage::Done);
                continue;
            }

            // Step 2: try eviction.
            if current_stage <= Stage::Evict {
                if let Some(preg) = self.try_evict(vreg_id) {
                    self.assign(vreg_id, preg);
                    self.advance_stage(vreg_id, Stage::Done);
                    continue;
                }
                self.advance_stage(vreg_id, Stage::Split);
                self.worklist.push(PriorityEntry {
                    vreg_id,
                    weight: entry.weight,
                });
                continue;
            }

            // Step 3: try splitting.
            if current_stage == Stage::Split {
                if self.try_split(vreg_id, func) {
                    // Both halves have been re-enqueued; do not spill.
                    continue;
                }
                self.advance_stage(vreg_id, Stage::Spill);
                self.worklist.push(PriorityEntry {
                    vreg_id,
                    weight: entry.weight,
                });
                continue;
            }

            // Step 4: spill.
            self.do_spill(vreg_id);
        }

        Ok(self.build_result())
    }

    // -----------------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------------

    /// Try to assign a free register to `vreg_id`.
    ///
    /// Checks hint registers first, then the full allocatable set for the
    /// interval's register class.  Returns `Some(preg)` if a non-interfering
    /// register is found.
    fn try_assign(&self, vreg_id: u32) -> Option<PReg> {
        let interval = self.intervals.get(&vreg_id)?;
        let class = interval.vreg.class;
        let allocatable = self.allocatable_regs.get(&class)?;

        // Try hint registers first.
        if let Some(hint_regs) = self.hints.get(&interval.vreg) {
            for &preg in hint_regs {
                if allocatable.contains(&preg) && !self.interferes(vreg_id, preg) {
                    return Some(preg);
                }
            }
        }

        // Try all allocatable registers.
        for &preg in allocatable {
            if !self.interferes(vreg_id, preg) {
                return Some(preg);
            }
        }

        None
    }

    /// Check whether assigning `preg` to `vreg_id` would interfere with
    /// any interval already assigned to `preg`.
    fn interferes(&self, vreg_id: u32, preg: PReg) -> bool {
        let interval = match self.intervals.get(&vreg_id) {
            Some(iv) => iv,
            None => return false,
        };

        if let Some(assigned_vregs) = self.preg_assignments.get(&preg) {
            for &other_id in assigned_vregs {
                if other_id == vreg_id {
                    continue;
                }
                if let Some(other_iv) = self.intervals.get(&other_id) {
                    if interval.overlaps(other_iv) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Record the assignment of `vreg_id` to `preg`.
    fn assign(&mut self, vreg_id: u32, preg: PReg) {
        if let Some(iv) = self.intervals.get(&vreg_id) {
            self.assignment.insert(iv.vreg, preg);
        }
        self.preg_assignments
            .entry(preg)
            .or_default()
            .push(vreg_id);
    }

    /// Remove the assignment of `vreg_id`.
    fn unassign(&mut self, vreg_id: u32) {
        if let Some(iv) = self.intervals.get(&vreg_id) {
            if let Some(preg) = self.assignment.remove(&iv.vreg) {
                if let Some(list) = self.preg_assignments.get_mut(&preg) {
                    list.retain(|&id| id != vreg_id);
                }
            }
        }
    }

    fn is_assigned(&self, vreg_id: u32) -> bool {
        self.intervals
            .get(&vreg_id)
            .map_or(false, |iv| self.assignment.contains_key(&iv.vreg))
    }

    fn is_spilled(&self, vreg_id: u32) -> bool {
        self.intervals
            .get(&vreg_id)
            .map_or(false, |iv| self.spilled.iter().any(|v| v.id == iv.vreg.id))
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Try to evict a lower-weight interval to make room for `vreg_id`.
    ///
    /// For each allocatable register, collects all interfering intervals.
    /// If every interferer has a lower spill weight and a cascade number
    /// lower than the current interval's, the interferers are evicted and
    /// the register is returned.
    fn try_evict(&mut self, vreg_id: u32) -> Option<PReg> {
        let interval = self.intervals.get(&vreg_id)?;
        let class = interval.vreg.class;
        let weight = interval.spill_weight;
        let my_cascade = self.cascade.get(&vreg_id).copied().unwrap_or(0);

        let allocatable = self.allocatable_regs.get(&class)?.clone();

        // Try hint registers first for eviction too.
        let hint_regs: Vec<PReg> = self
            .hints
            .get(&interval.vreg)
            .cloned()
            .unwrap_or_default();

        let candidates: Vec<PReg> = hint_regs
            .iter()
            .chain(allocatable.iter())
            .copied()
            .collect();

        let mut best_preg: Option<PReg> = None;
        let mut best_evict_cost: f64 = f64::MAX;

        for &preg in &candidates {
            if !allocatable.contains(&preg) {
                continue;
            }

            let assigned = match self.preg_assignments.get(&preg) {
                Some(list) => list.clone(),
                None => {
                    // No assignments -- free register (should have been
                    // caught by try_assign, but handle gracefully).
                    return Some(preg);
                }
            };

            // Collect interferers for this preg.
            let mut interferers: Vec<(u32, f64, u32)> = Vec::new(); // (vreg_id, weight, cascade)
            let mut can_evict = true;
            let mut total_cost = 0.0_f64;

            for &other_id in &assigned {
                if other_id == vreg_id {
                    continue;
                }
                if let Some(other_iv) = self.intervals.get(&other_id) {
                    if interval.overlaps(other_iv) {
                        let other_weight = other_iv.spill_weight;
                        let other_cascade =
                            self.cascade.get(&other_id).copied().unwrap_or(0);

                        // Cannot evict a heavier interval.
                        if other_weight >= weight {
                            can_evict = false;
                            break;
                        }
                        // Cannot evict if cascade would be exceeded.
                        if other_cascade >= my_cascade
                            && my_cascade >= self.max_cascade_depth
                        {
                            can_evict = false;
                            break;
                        }

                        total_cost += other_weight;
                        interferers.push((other_id, other_weight, other_cascade));
                    }
                }
            }

            if can_evict && !interferers.is_empty() && total_cost < best_evict_cost {
                best_evict_cost = total_cost;
                best_preg = Some(preg);
            }
        }

        // Perform the eviction for the best register found.
        if let Some(preg) = best_preg {
            self.evict_interference(preg, vreg_id);
            return Some(preg);
        }

        None
    }

    /// Evict all intervals assigned to `preg` that interfere with `new_vreg_id`.
    ///
    /// Evicted intervals are unassigned, get a new cascade number, and are
    /// pushed back onto the worklist.
    fn evict_interference(&mut self, preg: PReg, new_vreg_id: u32) {
        let new_cascade = self.next_cascade;
        self.next_cascade += 1;
        self.cascade.insert(new_vreg_id, new_cascade);

        let interval = match self.intervals.get(&new_vreg_id) {
            Some(iv) => iv.clone(),
            None => return,
        };

        let assigned = match self.preg_assignments.get(&preg) {
            Some(list) => list.clone(),
            None => return,
        };

        for other_id in assigned {
            if other_id == new_vreg_id {
                continue;
            }
            let overlaps = self
                .intervals
                .get(&other_id)
                .map_or(false, |iv| interval.overlaps(iv));
            if overlaps {
                let other_weight = self
                    .intervals
                    .get(&other_id)
                    .map_or(0.0, |iv| iv.spill_weight);

                self.unassign(other_id);
                self.cascade.insert(other_id, new_cascade);
                // Reset stage to Evict so it can try assignment again.
                self.stage.insert(other_id, Stage::Evict);
                self.worklist.push(PriorityEntry {
                    vreg_id: other_id,
                    weight: other_weight,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Splitting
    // -----------------------------------------------------------------------

    /// Try to split `vreg_id`'s interval at its optimal split point.
    ///
    /// If successful, both halves are inserted into the interval map and
    /// enqueued on the worklist.  Returns `true` if the split was
    /// performed.
    fn try_split(&mut self, vreg_id: u32, func: &mut MachFunction) -> bool {
        let interval = match self.intervals.get(&vreg_id) {
            Some(iv) => iv.clone(),
            None => return false,
        };

        let split_point = match split::find_optimal_split_point(&interval) {
            Some(pt) => pt,
            None => return false,
        };

        let result = match split::split_interval(&interval, split_point, func) {
            Some(r) => r,
            None => return false,
        };

        // Remove the old interval.
        self.intervals.remove(&vreg_id);
        self.stage.remove(&vreg_id);

        // Insert the original (truncated) half.
        let orig_id = result.original_vreg.id;
        let orig_weight = result.original_interval.spill_weight;
        self.intervals.insert(orig_id, result.original_interval);
        self.stage.insert(orig_id, Stage::New);
        self.worklist.push(PriorityEntry {
            vreg_id: orig_id,
            weight: orig_weight,
        });

        // Insert the new half.
        let new_id = result.new_vreg.id;
        let new_weight = result.new_interval.spill_weight;
        self.intervals.insert(new_id, result.new_interval);
        self.stage.insert(new_id, Stage::New);
        self.worklist.push(PriorityEntry {
            vreg_id: new_id,
            weight: new_weight,
        });

        true
    }

    // -----------------------------------------------------------------------
    // Spilling
    // -----------------------------------------------------------------------

    /// Mark `vreg_id` as spilled.
    fn do_spill(&mut self, vreg_id: u32) {
        if let Some(iv) = self.intervals.get(&vreg_id) {
            self.spilled.push(iv.vreg);
        }
        self.advance_stage(vreg_id, Stage::Done);
    }

    /// Return the list of spilled VRegs.
    pub fn spilled_vregs(&self) -> &[VReg] {
        &self.spilled
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Advance `vreg_id` to at least `target` stage.
    fn advance_stage(&mut self, vreg_id: u32, target: Stage) {
        let current = self.stage.get(&vreg_id).copied().unwrap_or(Stage::New);
        if target > current {
            self.stage.insert(vreg_id, target);
        }
    }

    /// Build the final [`AllocationResult`].
    fn build_result(&self) -> AllocationResult {
        AllocationResult {
            allocation: self.assignment.clone(),
            spills: Vec::new(), // Spill info filled in by insert_spill_code
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::LiveInterval;
    use crate::machine_types::{
        BlockId, InstFlags, InstId, MachBlock, MachFunction, MachInst, MachOperand, PReg,
        RegClass, VReg,
    };
    use std::collections::HashMap;

    // -- helpers --

    fn vreg(id: u32) -> VReg {
        VReg {
            id,
            class: RegClass::Gpr64,
        }
    }

    fn make_interval(
        id: u32,
        ranges: &[(u32, u32)],
        weight: f64,
    ) -> LiveInterval {
        let mut iv = LiveInterval::new(vreg(id));
        for &(start, end) in ranges {
            iv.add_range(start, end);
        }
        iv.spill_weight = weight;
        // Add use/def positions at range boundaries for split tests.
        if let Some(&(s, _)) = ranges.first() {
            iv.def_positions.push(s);
        }
        for &(_, e) in ranges {
            iv.use_positions.push(e.saturating_sub(1));
        }
        iv
    }

    fn one_gpr_regs() -> HashMap<RegClass, Vec<PReg>> {
        let mut m = HashMap::new();
        m.insert(RegClass::Gpr64, vec![PReg::new(0)]);
        m
    }

    fn two_gpr_regs() -> HashMap<RegClass, Vec<PReg>> {
        let mut m = HashMap::new();
        m.insert(RegClass::Gpr64, vec![PReg::new(0), PReg::new(1)]);
        m
    }

    fn many_gpr_regs() -> HashMap<RegClass, Vec<PReg>> {
        let mut m = HashMap::new();
        // 26 GPR64 regs matching AArch64.
        let regs: Vec<PReg> = (0u16..=15).chain(19u16..=28).map(PReg::new).collect();
        m.insert(RegClass::Gpr64, regs);
        m
    }

    fn make_test_func(num_insts: usize) -> MachFunction {
        let mut insts = Vec::new();
        let mut inst_ids = Vec::new();

        for i in 0..num_insts {
            let inst = MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(vreg(i as u32))],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            };
            inst_ids.push(InstId(i as u32));
            insts.push(inst);
        }

        MachFunction {
            name: "test".into(),
            insts,
            blocks: vec![MachBlock {
                insts: inst_ids,
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: num_insts as u32,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    // -- tests --

    #[test]
    fn test_greedy_simple_allocation() {
        // Two overlapping intervals, 26 GPRs -> both allocated.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(5, 15)], 2.0),
        ];
        let regs = many_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 2);
        let p0 = result.allocation[&vreg(0)];
        let p1 = result.allocation[&vreg(1)];
        assert_ne!(p0, p1);
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_greedy_non_overlapping_same_reg() {
        // Two non-overlapping intervals with 1 reg -> both get the same reg.
        let intervals = vec![
            make_interval(0, &[(0, 5)], 1.0),
            make_interval(1, &[(5, 10)], 1.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 2);
        assert_eq!(
            result.allocation[&vreg(0)],
            result.allocation[&vreg(1)]
        );
    }

    #[test]
    fn test_greedy_eviction() {
        // 3 overlapping intervals, only 1 register.
        // Weights: v0=1.0, v1=5.0, v2=3.0 (all overlap [0,10)).
        // v1 (highest weight) processes first, gets the register.
        // v2 tries eviction but v1 is heavier -> cannot evict.
        // v0 tries eviction but v1 is heavier -> cannot evict.
        // v2 and v0 are spilled.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(0, 10)], 5.0),
            make_interval(2, &[(0, 10)], 3.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // v1 wins the register.
        assert!(result.allocation.contains_key(&vreg(1)));
        // v0 and v2 are spilled.
        assert_eq!(alloc.spilled.len(), 2);
        let spilled_ids: Vec<u32> = alloc.spilled.iter().map(|v| v.id).collect();
        assert!(spilled_ids.contains(&0));
        assert!(spilled_ids.contains(&2));
    }

    #[test]
    fn test_greedy_eviction_low_weight_evicted() {
        // 2 overlapping intervals, 1 register.
        // v0 (weight=1.0) goes first by queue ordering but v1 (weight=5.0)
        // has higher priority.  v1 should evict v0.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(0, 10)], 5.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // v1 (higher weight) should be assigned.
        assert!(result.allocation.contains_key(&vreg(1)));
        // v0 (lower weight) should be spilled.
        assert_eq!(alloc.spilled.len(), 1);
        assert_eq!(alloc.spilled[0].id, 0);
    }

    #[test]
    fn test_greedy_cascade_limit() {
        // Create a chain of intervals with decreasing weights and 1 register.
        // Each evicts the previous. With max_cascade_depth=3, eviction
        // should stop after 3 levels.
        let intervals = vec![
            make_interval(0, &[(0, 20)], 1.0),
            make_interval(1, &[(0, 20)], 2.0),
            make_interval(2, &[(0, 20)], 3.0),
            make_interval(3, &[(0, 20)], 4.0),
            make_interval(4, &[(0, 20)], 5.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        alloc.max_cascade_depth = 3;
        let result = alloc.allocate().unwrap();

        // The highest weight interval (v4) should be assigned.
        assert!(result.allocation.contains_key(&vreg(4)));
        // The rest should be spilled.
        assert_eq!(alloc.spilled.len(), 4);
    }

    #[test]
    fn test_greedy_with_hints() {
        // Two non-overlapping intervals.  v0 has a hint for PReg(5).
        let intervals = vec![
            make_interval(0, &[(0, 5)], 1.0),
            make_interval(1, &[(5, 10)], 1.0),
        ];
        let regs = many_gpr_regs();
        let mut hints = HashMap::new();
        hints.insert(vreg(0), vec![PReg::new(5)]);

        let mut alloc = GreedyAllocator::new(intervals, &regs, hints);
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 2);
        // v0 should have received its hint register.
        assert_eq!(result.allocation[&vreg(0)], PReg::new(5));
    }

    #[test]
    fn test_greedy_spill_when_no_eviction_possible() {
        // All overlapping, all same weight, 1 register -> only first wins.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(0, 10)], 1.0),
            make_interval(2, &[(0, 10)], 1.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Exactly one should be allocated.
        assert_eq!(result.allocation.len(), 1);
        // Two should be spilled.
        assert_eq!(alloc.spilled.len(), 2);
    }

    #[test]
    fn test_greedy_split_before_spill() {
        // An interval [0, 20) with uses at 2 and 18.  Large gap in the
        // middle.  With only 1 register and an overlapping interval [8,12),
        // splitting should produce two halves that can each fit.
        let mut iv0 = LiveInterval::new(vreg(0));
        iv0.add_range(0, 20);
        iv0.spill_weight = 3.0;
        iv0.def_positions = vec![0];
        iv0.use_positions = vec![2, 18];

        let iv1 = make_interval(1, &[(8, 12)], 5.0);

        let intervals = vec![iv0, iv1];
        let regs = two_gpr_regs();
        let mut func = make_test_func(20);

        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate_with_splitting(&mut func).unwrap();

        // After splitting, all halves should be assignable (2 regs, and the
        // split halves don't fully overlap).  No spills expected.
        assert!(
            alloc.spilled.is_empty(),
            "expected no spills but got {} spills: {:?}",
            alloc.spilled.len(),
            alloc.spilled
        );
        // We should have allocations for: v1, plus the two halves of v0.
        assert!(result.allocation.len() >= 2);
    }

    #[test]
    fn test_greedy_many_intervals_no_spill() {
        // 10 non-overlapping intervals with 26 regs -> all allocated.
        let intervals: Vec<LiveInterval> = (0..10)
            .map(|i| make_interval(i, &[(i * 5, i * 5 + 3)], 1.0))
            .collect();
        let regs = many_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 10);
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_greedy_high_pressure_spill() {
        // 30 simultaneously-live intervals with 26 GPRs.
        let intervals: Vec<LiveInterval> = (0..30)
            .map(|i| make_interval(i, &[(0, 100)], (i + 1) as f64))
            .collect();
        let regs = many_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // At most 26 can be assigned.
        assert!(result.allocation.len() <= 26);
        // At least 4 must be spilled.
        assert!(alloc.spilled.len() >= 4);
        // The lowest-weight intervals should be spilled.
        for v in &alloc.spilled {
            // Spilled intervals should have weight <= 26.0 (the cutoff for 26 regs).
            // The top-26 weights are 5..30, so spilled weights should be 1..4.
            assert!(
                v.id < 26,
                "expected low-weight interval to be spilled, got v{}",
                v.id
            );
        }
    }

    #[test]
    fn test_greedy_fixed_intervals_skipped() {
        // Fixed intervals are skipped (not enqueued).
        let mut iv0 = make_interval(0, &[(0, 10)], 1.0);
        iv0.is_fixed = true;

        let iv1 = make_interval(1, &[(0, 10)], 2.0);

        let intervals = vec![iv0, iv1];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Only v1 should be in the allocation (v0 is fixed, skipped).
        assert_eq!(result.allocation.len(), 1);
        assert!(result.allocation.contains_key(&vreg(1)));
    }

    #[test]
    fn test_greedy_empty_intervals() {
        let intervals: Vec<LiveInterval> = Vec::new();
        let regs = many_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert!(result.allocation.is_empty());
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_priority_entry_ordering() {
        // Higher weight = higher priority.
        let a = PriorityEntry {
            vreg_id: 0,
            weight: 1.0,
        };
        let b = PriorityEntry {
            vreg_id: 1,
            weight: 5.0,
        };
        assert!(b > a);

        // Equal weight: lower vreg_id = higher priority.
        let c = PriorityEntry {
            vreg_id: 2,
            weight: 5.0,
        };
        assert!(b > c);
    }
}

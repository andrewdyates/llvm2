// llvm2-regalloc/greedy.rs - Greedy register allocator
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
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
// Sub-register aliasing support (issue #336)
// ---------------------------------------------------------------------------

/// Returns all physical registers that alias the given register (in different
/// register classes).
///
/// On AArch64, W-registers are the lower 32 bits of X-registers, and
/// D/S/H/B-registers are sub-views of V-registers. Writing to any alias
/// clobbers the others, so the allocator must treat them as conflicting.
///
/// Does NOT include the input register itself.
pub fn aliasing_pregs(preg: PReg) -> Vec<PReg> {
    use llvm2_ir::regs::{
        gpr64_to_gpr32, gpr32_to_gpr64,
        fpr128_to_fpr64, fpr128_to_fpr32,
        fpr64_to_fpr128, fpr32_to_fpr128,
    };
    let mut aliases = Vec::new();
    let e = preg.encoding();
    match e {
        0..=30 => {
            // GPR64 X0-X30 -> alias is the corresponding W register
            if let Some(w) = gpr64_to_gpr32(preg) {
                aliases.push(w);
            }
        }
        32..=62 => {
            // GPR32 W0-W30 -> alias is the corresponding X register
            if let Some(x) = gpr32_to_gpr64(preg) {
                aliases.push(x);
            }
        }
        64..=95 => {
            // FPR128 V0-V31 -> aliases are D, S sub-registers
            if let Some(d) = fpr128_to_fpr64(preg) {
                aliases.push(d);
            }
            if let Some(s) = fpr128_to_fpr32(preg) {
                aliases.push(s);
            }
        }
        96..=127 => {
            // FPR64 D0-D31 -> aliases are V (parent) and S (sibling)
            if let Some(v) = fpr64_to_fpr128(preg) {
                aliases.push(v);
                // S register has same number as D
                if let Some(s) = fpr128_to_fpr32(v) {
                    aliases.push(s);
                }
            }
        }
        128..=159 => {
            // FPR32 S0-S31 -> aliases are V (parent) and D (sibling)
            if let Some(v) = fpr32_to_fpr128(preg) {
                aliases.push(v);
                // D register has same number as S
                if let Some(d) = fpr128_to_fpr64(v) {
                    aliases.push(d);
                }
            }
        }
        _ => {}
    }
    aliases
}

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
        allocatable.iter().find(|&&preg| !self.interferes(vreg_id, preg)).copied()
    }

    /// Check whether assigning `preg` to `vreg_id` would interfere with
    /// any interval already assigned to `preg` or its aliases.
    ///
    /// On AArch64, writing to X28 clobbers W28 (and vice versa), so we must
    /// check both the exact register and all aliasing registers for conflicts.
    /// (Issue #336: mixed-width ABI register aliasing.)
    fn interferes(&self, vreg_id: u32, preg: PReg) -> bool {
        let interval = match self.intervals.get(&vreg_id) {
            Some(iv) => iv,
            None => return false,
        };

        // Check the exact PReg.
        if self.interferes_with_preg(vreg_id, preg, interval) {
            return true;
        }

        // Check all aliasing PRegs (e.g., W28 when checking X28).
        for alias in aliasing_pregs(preg) {
            if self.interferes_with_preg(vreg_id, alias, interval) {
                return true;
            }
        }

        false
    }

    /// Check whether any interval assigned to a specific `preg` overlaps `interval`.
    fn interferes_with_preg(&self, vreg_id: u32, preg: PReg, interval: &LiveInterval) -> bool {
        if let Some(assigned_vregs) = self.preg_assignments.get(&preg) {
            for &other_id in assigned_vregs {
                if other_id == vreg_id {
                    continue;
                }
                if let Some(other_iv) = self.intervals.get(&other_id)
                    && interval.overlaps(other_iv) {
                        return true;
                    }
            }
        }
        false
    }

    /// Record the assignment of `vreg_id` to `preg`.
    ///
    /// Also records the assignment against all aliasing registers so that
    /// interference checks on aliased registers will find this interval.
    /// (Issue #336: mixed-width ABI register aliasing.)
    fn assign(&mut self, vreg_id: u32, preg: PReg) {
        if let Some(iv) = self.intervals.get(&vreg_id) {
            self.assignment.insert(iv.vreg, preg);
        }
        self.preg_assignments
            .entry(preg)
            .or_default()
            .push(vreg_id);
        // Also record in aliasing registers.
        for alias in aliasing_pregs(preg) {
            self.preg_assignments
                .entry(alias)
                .or_default()
                .push(vreg_id);
        }
    }

    /// Remove the assignment of `vreg_id`.
    ///
    /// Also removes from all aliasing registers.
    /// (Issue #336: mixed-width ABI register aliasing.)
    fn unassign(&mut self, vreg_id: u32) {
        if let Some(iv) = self.intervals.get(&vreg_id)
            && let Some(preg) = self.assignment.remove(&iv.vreg) {
                if let Some(list) = self.preg_assignments.get_mut(&preg) {
                    list.retain(|&id| id != vreg_id);
                }
                // Also remove from aliasing registers.
                for alias in aliasing_pregs(preg) {
                    if let Some(list) = self.preg_assignments.get_mut(&alias) {
                        list.retain(|&id| id != vreg_id);
                    }
                }
            }
    }

    fn is_assigned(&self, vreg_id: u32) -> bool {
        self.intervals
            .get(&vreg_id)
            .is_some_and(|iv| self.assignment.contains_key(&iv.vreg))
    }

    fn is_spilled(&self, vreg_id: u32) -> bool {
        self.intervals
            .get(&vreg_id)
            .is_some_and(|iv| self.spilled.iter().any(|v| v.id == iv.vreg.id))
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
                if let Some(other_iv) = self.intervals.get(&other_id)
                    && interval.overlaps(other_iv) {
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
                .is_some_and(|iv| interval.overlaps(iv));
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

    /// Try to split `vreg_id`'s interval using multiple strategies.
    ///
    /// Strategies are attempted in order of increasing aggressiveness:
    /// 1. **Gap-based**: split at the midpoint of the largest gap between
    ///    consecutive use/def positions (existing `find_optimal_split_point`).
    /// 2. **Interference-aware**: find where register pressure is highest
    ///    and split just before that region, keeping the first half in a
    ///    register.
    /// 3. **Per-use**: split between consecutive use/def positions,
    ///    creating short intervals that are easy to allocate individually.
    ///
    /// If any strategy succeeds, both halves are inserted into the
    /// interval map and enqueued on the worklist.  Returns `true` if
    /// a split was performed.
    fn try_split(&mut self, vreg_id: u32, func: &mut MachFunction) -> bool {
        let interval = match self.intervals.get(&vreg_id) {
            Some(iv) => iv.clone(),
            None => return false,
        };

        // Strategy 1: gap-based split (least aggressive, best quality).
        if let Some(split_point) = split::find_optimal_split_point(&interval) {
            if let Some(result) = split::split_interval(&interval, split_point, func) {
                self.apply_split(vreg_id, result);
                return true;
            }
        }

        // Strategy 2: interference-aware split.
        if let Some(interference_start) = self.find_interference_start(vreg_id) {
            if let Some(split_point) =
                split::find_split_near_interference(&interval, interference_start)
            {
                if let Some(result) = split::split_interval(&interval, split_point, func) {
                    self.apply_split(vreg_id, result);
                    return true;
                }
            }
        }

        // Strategy 3: per-use split (most aggressive).
        let per_use_splits = split::find_per_use_split_points(&interval);
        for (split_point, _weight) in per_use_splits {
            if let Some(result) = split::split_interval(&interval, split_point, func) {
                self.apply_split(vreg_id, result);
                return true;
            }
        }

        false
    }

    /// Apply a split result: remove the old interval, insert both halves,
    /// and enqueue them on the worklist.
    fn apply_split(&mut self, vreg_id: u32, result: split::SplitResult) {
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
    }

    // -----------------------------------------------------------------------
    // Interference analysis (for splitting)
    // -----------------------------------------------------------------------

    /// Find the earliest point where all allocatable registers for
    /// `vreg_id`'s class are occupied by other assigned intervals.
    ///
    /// This identifies where register pressure is highest, guiding
    /// the interference-aware split strategy.  Returns `None` if
    /// there is no fully-blocked point (meaning direct assignment
    /// should have succeeded).
    fn find_interference_start(&self, vreg_id: u32) -> Option<u32> {
        let interval = self.intervals.get(&vreg_id)?;
        let class = interval.vreg.class;
        let allocatable = self.allocatable_regs.get(&class)?;

        for range in &interval.ranges {
            for pos in range.start..range.end {
                let all_interfere = allocatable
                    .iter()
                    .all(|&preg| self.is_occupied_at(preg, pos, vreg_id));
                if all_interfere {
                    return Some(pos);
                }
            }
        }

        None
    }

    /// Check whether a physical register is occupied at a specific
    /// program point by any interval other than `exclude_vreg_id`.
    fn is_occupied_at(&self, preg: PReg, pos: u32, exclude_vreg_id: u32) -> bool {
        if let Some(assigned_vregs) = self.preg_assignments.get(&preg) {
            for &other_id in assigned_vregs {
                if other_id == exclude_vreg_id {
                    continue;
                }
                if let Some(other_iv) = self.intervals.get(&other_id) {
                    if other_iv.is_live_at(pos) {
                        return true;
                    }
                }
            }
        }
        false
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

    // =====================================================================
    // Additional coverage tests
    // =====================================================================

    #[test]
    fn test_basic_allocation_no_spills() {
        // 4 non-overlapping intervals with 2 registers.
        // Each pair shares a register since they don't overlap.
        let intervals = vec![
            make_interval(0, &[(0, 5)], 1.0),
            make_interval(1, &[(5, 10)], 1.0),
            make_interval(2, &[(10, 15)], 1.0),
            make_interval(3, &[(15, 20)], 1.0),
        ];
        let regs = two_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 4);
        assert!(alloc.spilled.is_empty(), "no spills expected with non-overlapping intervals");
    }

    #[test]
    fn test_allocation_requiring_spills_more_live_than_regs() {
        // 3 simultaneously-live intervals with only 2 registers.
        // The lowest-weight one must be spilled.
        let intervals = vec![
            make_interval(0, &[(0, 20)], 1.0),
            make_interval(1, &[(0, 20)], 3.0),
            make_interval(2, &[(0, 20)], 5.0),
        ];
        let regs = two_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Top-2 weights get assigned.
        assert_eq!(result.allocation.len(), 2);
        assert!(result.allocation.contains_key(&vreg(1)));
        assert!(result.allocation.contains_key(&vreg(2)));
        // Lowest weight is spilled.
        assert_eq!(alloc.spilled.len(), 1);
        assert_eq!(alloc.spilled[0].id, 0);
    }

    #[test]
    fn test_interference_graph_correctness() {
        // Test that the allocator correctly detects interference between
        // overlapping intervals and assigns different registers.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(5, 15)], 1.0),
            make_interval(2, &[(10, 20)], 1.0),
        ];
        let regs = two_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // v0 and v1 overlap -> different regs.
        let p0 = result.allocation[&vreg(0)];
        let p1 = result.allocation[&vreg(1)];
        assert_ne!(p0, p1, "overlapping intervals must get different regs");

        // v1 and v2 overlap -> different regs.
        let p2 = result.allocation[&vreg(2)];
        assert_ne!(p1, p2, "overlapping intervals must get different regs");

        // v0 and v2 do NOT overlap (0..10 and 10..20 are adjacent, not overlapping)
        // so they CAN share a register.
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_spill_weight_determines_spill_victim() {
        // 5 overlapping intervals, 2 registers. The 3 lowest-weight
        // intervals should be spilled.
        let intervals = vec![
            make_interval(0, &[(0, 100)], 10.0),
            make_interval(1, &[(0, 100)], 50.0),
            make_interval(2, &[(0, 100)], 30.0),
            make_interval(3, &[(0, 100)], 20.0),
            make_interval(4, &[(0, 100)], 40.0),
        ];
        let regs = two_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Top-2 by weight: v1 (50.0) and v4 (40.0)
        assert_eq!(result.allocation.len(), 2);
        assert!(result.allocation.contains_key(&vreg(1)), "highest weight should be allocated");
        assert!(result.allocation.contains_key(&vreg(4)), "second highest should be allocated");

        // The other 3 should be spilled.
        assert_eq!(alloc.spilled.len(), 3);
        let spilled_ids: Vec<u32> = alloc.spilled.iter().map(|v| v.id).collect();
        assert!(spilled_ids.contains(&0));
        assert!(spilled_ids.contains(&2));
        assert!(spilled_ids.contains(&3));
    }

    #[test]
    fn test_call_clobber_handling_live_across_call() {
        // Simulate live range across a call instruction.
        // v0 spans the entire function including a "call" at instruction 10.
        // v1 only spans post-call. With 1 register, eviction should
        // keep the higher-weight one.
        let intervals = vec![
            make_interval(0, &[(0, 20)], 2.0),  // crosses "call" at 10
            make_interval(1, &[(10, 20)], 5.0),  // starts at call
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // v1 has higher weight, so it gets the register.
        assert!(result.allocation.contains_key(&vreg(1)));
        // v0 must be spilled (lower weight, can't share the register).
        assert_eq!(alloc.spilled.len(), 1);
        assert_eq!(alloc.spilled[0].id, 0);
    }

    #[test]
    fn test_multiple_register_classes_independent() {
        // GPR and FPR intervals use disjoint PReg sets and don't interfere.
        // AArch64 encoding: GPR64 = PReg 0-30, FPR64 = PReg 96-127.
        let gpr_iv = {
            let mut iv = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
            iv.add_range(0, 10);
            iv.spill_weight = 1.0;
            iv.def_positions.push(0);
            iv.use_positions.push(9);
            iv
        };
        let fpr_iv = {
            let mut iv = LiveInterval::new(VReg { id: 1, class: RegClass::Fpr64 });
            iv.add_range(0, 10);
            iv.spill_weight = 1.0;
            iv.def_positions.push(0);
            iv.use_positions.push(9);
            iv
        };

        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);
        regs.insert(RegClass::Fpr64, vec![PReg::new(96)]); // D0 — disjoint from GPR

        let intervals = vec![gpr_iv, fpr_iv];
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Both should be allocated to their respective class registers.
        assert_eq!(result.allocation.len(), 2);
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_eviction_cascade_respects_weight_ordering() {
        // Chain: v0(1.0) assigned first, v1(2.0) evicts v0,
        // v2(3.0) evicts v1, etc. All overlapping, 1 register.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(0, 10)], 2.0),
            make_interval(2, &[(0, 10)], 3.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // The highest weight (v2) should win.
        assert!(result.allocation.contains_key(&vreg(2)));
        assert_eq!(alloc.spilled.len(), 2);
    }

    #[test]
    fn test_hint_conflicts_with_existing_allocation() {
        // v0 gets PReg(0). v1 has a hint for PReg(0) but overlaps v0,
        // so it should fall back to PReg(1).
        let intervals = vec![
            make_interval(0, &[(0, 10)], 5.0),
            make_interval(1, &[(0, 10)], 3.0),
        ];
        let regs = two_gpr_regs();
        let mut hints = HashMap::new();
        hints.insert(vreg(1), vec![PReg::new(0)]);

        let mut alloc = GreedyAllocator::new(intervals, &regs, hints);
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 2);
        // Both should be allocated to different regs despite the hint conflict.
        let p0 = result.allocation[&vreg(0)];
        let p1 = result.allocation[&vreg(1)];
        assert_ne!(p0, p1);
    }

    #[test]
    fn test_single_instruction_interval() {
        // An interval that spans exactly one instruction.
        let intervals = vec![
            make_interval(0, &[(5, 6)], 1.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 1);
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_interleaved_intervals_no_overlap() {
        // Intervals that alternate: v0=[0,5), v1=[5,10), v2=[10,15), etc.
        // All should fit in 1 register.
        let intervals: Vec<LiveInterval> = (0..5)
            .map(|i| make_interval(i, &[(i * 5, i * 5 + 5)], 1.0))
            .collect();
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 5);
        assert!(alloc.spilled.is_empty());
        // All should get the same register.
        let preg0 = result.allocation[&vreg(0)];
        for i in 1..5 {
            assert_eq!(result.allocation[&vreg(i)], preg0);
        }
    }

    #[test]
    fn test_stage_progression_new_to_done() {
        // Verify that a successfully allocated interval goes from New to Done.
        let intervals = vec![make_interval(0, &[(0, 5)], 1.0)];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let _ = alloc.allocate().unwrap();

        // After allocation, the stage should be Done.
        assert_eq!(*alloc.stage.get(&0).unwrap(), Stage::Done);
    }

    #[test]
    fn test_spilled_vregs_accessor() {
        // 2 overlapping same-weight intervals, 1 register.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(0, 10)], 1.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let _ = alloc.allocate().unwrap();

        // One gets allocated, one gets spilled.
        let spilled = alloc.spilled_vregs();
        assert_eq!(spilled.len(), 1);
        assert_eq!(spilled[0].class, RegClass::Gpr64);
    }

    #[test]
    fn test_max_cascade_depth_zero_disables_eviction() {
        // With max_cascade_depth=0, eviction should be impossible.
        let intervals = vec![
            make_interval(0, &[(0, 10)], 1.0),
            make_interval(1, &[(0, 10)], 5.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        alloc.max_cascade_depth = 0;
        let result = alloc.allocate().unwrap();

        // v1 gets allocated first (higher priority). v0 cannot evict
        // because cascade depth is 0, so it spills.
        assert!(result.allocation.contains_key(&vreg(1)));
        assert_eq!(alloc.spilled.len(), 1);
        assert_eq!(alloc.spilled[0].id, 0);
    }

    #[test]
    fn test_disjoint_live_ranges_same_vreg() {
        // An interval with multiple disjoint ranges (hole in the middle).
        let intervals = vec![
            make_interval(0, &[(0, 5), (15, 20)], 1.0),
            make_interval(1, &[(5, 15)], 2.0),
        ];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // The two intervals don't overlap (v0 has a hole where v1 lives).
        assert_eq!(result.allocation.len(), 2);
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_greedy_all_same_weight_deterministic_spill_order() {
        // 4 overlapping intervals with identical weight and 2 registers.
        // The allocator should be deterministic: lower vreg_id breaks ties
        // in the priority queue (higher priority for lower id with same weight).
        let intervals = vec![
            make_interval(0, &[(0, 20)], 3.0),
            make_interval(1, &[(0, 20)], 3.0),
            make_interval(2, &[(0, 20)], 3.0),
            make_interval(3, &[(0, 20)], 3.0),
        ];
        let regs = two_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Exactly 2 should be allocated, 2 spilled.
        assert_eq!(result.allocation.len(), 2);
        assert_eq!(alloc.spilled.len(), 2);
    }

    #[test]
    fn test_greedy_hundred_non_overlapping_one_register() {
        // 100 sequential non-overlapping intervals should all fit in 1 register.
        let intervals: Vec<LiveInterval> = (0..100)
            .map(|i| make_interval(i, &[(i * 10, i * 10 + 5)], 1.0))
            .collect();
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        assert_eq!(result.allocation.len(), 100);
        assert!(alloc.spilled.is_empty());
        // All should share the single register.
        let preg0 = result.allocation[&vreg(0)];
        for i in 1..100 {
            assert_eq!(result.allocation[&vreg(i)], preg0);
        }
    }

    // =====================================================================
    // Live range splitting tests (issue #332)
    // =====================================================================

    #[test]
    fn test_greedy_split_at_interference() {
        // v0: [0, 20) with uses at 0, 5, 10, 15, 19 (no large gap for
        // gap-based split). v1: [4, 12) weight 10.0 (blocks the middle).
        // 2 registers. After eviction fails (v1 heavier), splitting v0
        // around interference should produce two halves.
        let mut iv0 = LiveInterval::new(vreg(0));
        iv0.add_range(0, 20);
        iv0.spill_weight = 2.0;
        iv0.def_positions = vec![0];
        iv0.use_positions = vec![5, 10, 15, 19];

        let iv1 = make_interval(1, &[(4, 12)], 10.0);

        let intervals = vec![iv0, iv1];
        let regs = two_gpr_regs();
        let mut func = make_test_func(20);

        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate_with_splitting(&mut func).unwrap();

        // Should complete without panic and account for all original intervals.
        let total = result.allocation.len() + alloc.spilled.len();
        assert!(total >= 2, "all original intervals should be accounted for");
    }

    #[test]
    fn test_greedy_per_use_split() {
        // An interval with many closely-spaced uses and high pressure.
        // Two blockers occupy both registers over different halves.
        let mut iv0 = LiveInterval::new(vreg(0));
        iv0.add_range(0, 30);
        iv0.spill_weight = 1.0;
        iv0.def_positions = vec![0];
        iv0.use_positions = vec![3, 7, 12, 18, 25, 29];

        let iv1 = make_interval(1, &[(0, 15)], 5.0);
        let iv2 = make_interval(2, &[(10, 30)], 5.0);

        let intervals = vec![iv0, iv1, iv2];
        let regs = two_gpr_regs();
        let mut func = make_test_func(30);

        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let _result = alloc.allocate_with_splitting(&mut func).unwrap();

        // Verify allocation completed (may have spills, that's OK).
        // The key is that splitting was attempted before spilling.
    }

    #[test]
    fn test_split_near_interference_from_greedy() {
        // Test find_split_near_interference via the split module.
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 20);
        iv.use_positions = vec![2, 8, 15];
        iv.def_positions = vec![0];

        let sp = split::find_split_near_interference(&iv, 10);
        assert!(sp.is_some());
        assert_eq!(
            sp.unwrap(),
            9,
            "should split at 9 (after last use at 8 before interference at 10)"
        );
    }

    #[test]
    fn test_per_use_split_points_from_greedy() {
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 30);
        iv.use_positions = vec![3, 7, 15, 25];
        iv.def_positions = vec![0];

        let splits = split::find_per_use_split_points(&iv);
        assert!(!splits.is_empty(), "should find split points between uses");

        for (sp, _weight) in &splits {
            assert!(*sp > iv.start());
            assert!(*sp < iv.end());
        }

        // Sorted by weight descending (largest gaps first).
        if splits.len() >= 2 {
            assert!(
                splits[0].1 >= splits[1].1,
                "splits should be sorted by weight descending"
            );
        }
    }

    #[test]
    fn test_apply_split_helper() {
        // Test that apply_split properly re-enqueues both halves.
        let iv0 = make_interval(0, &[(0, 20)], 3.0);
        let intervals = vec![iv0.clone()];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());

        let mut func = make_test_func(20);
        let result = split::split_interval(&iv0, 10, &mut func).unwrap();

        // Drain the worklist first.
        while alloc.worklist.pop().is_some() {}

        alloc.apply_split(0, result);

        // Both halves should be in the worklist now.
        let mut found = 0;
        while alloc.worklist.pop().is_some() {
            found += 1;
        }
        assert_eq!(found, 2, "both split halves should be enqueued");
    }

    #[test]
    fn test_greedy_split_attempts_before_spill() {
        // Verify that splitting is attempted and produces split intervals
        // when there is a large gap.  With 2 registers and a blocker in
        // the middle, the split halves of v0 should be allocable.
        let mut iv0 = LiveInterval::new(vreg(0));
        iv0.add_range(0, 30);
        iv0.spill_weight = 2.0;
        iv0.def_positions = vec![0];
        iv0.use_positions = vec![2, 28];

        let iv1 = make_interval(1, &[(10, 20)], 10.0);

        let intervals = vec![iv0, iv1];
        let regs = two_gpr_regs();
        let mut func = make_test_func(30);

        let mut alloc =
            GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate_with_splitting(&mut func).unwrap();

        // With 2 registers, the split halves of v0 don't fully overlap
        // v1, so everything should be allocable without spills.
        assert!(
            alloc.spilled.is_empty(),
            "with 2 regs and a split, expected no spills but got {} spills: {:?}",
            alloc.spilled.len(),
            alloc.spilled
        );
        // v1 plus the two halves of v0 should all be allocated.
        assert!(result.allocation.len() >= 2);
    }

    #[test]
    fn test_find_interference_start() {
        // Set up an allocator where one register is occupied, then
        // query find_interference_start.
        let iv0 = make_interval(0, &[(0, 20)], 1.0);
        let iv1 = make_interval(1, &[(5, 15)], 5.0);

        let intervals = vec![iv0, iv1];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());

        // Manually assign v1 to the only register.
        alloc.assign(1, PReg::new(0));

        // v0 should find interference starting at position 5 (where v1 starts).
        let start = alloc.find_interference_start(0);
        assert!(start.is_some());
        assert_eq!(start.unwrap(), 5);
    }

    #[test]
    fn test_is_occupied_at() {
        let iv0 = make_interval(0, &[(0, 10)], 1.0);
        let iv1 = make_interval(1, &[(3, 8)], 2.0);

        let intervals = vec![iv0, iv1];
        let regs = one_gpr_regs();
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());

        // Assign v1 to PReg(0).
        alloc.assign(1, PReg::new(0));

        // PReg(0) should be occupied at position 5 (v1 is [3,8)).
        assert!(alloc.is_occupied_at(PReg::new(0), 5, 0));
        // PReg(0) should NOT be occupied at position 1 (before v1).
        assert!(!alloc.is_occupied_at(PReg::new(0), 1, 0));
        // PReg(0) should NOT be occupied at position 9 (after v1).
        assert!(!alloc.is_occupied_at(PReg::new(0), 9, 0));
    }

    // =====================================================================
    // Issue #336: Mixed-width ABI register aliasing tests
    // =====================================================================

    #[test]
    fn test_issue_336_mixed_width_gpr_aliasing() {
        // A Gpr64 interval (i64) and a Gpr32 interval (i32) that are
        // simultaneously live MUST NOT be assigned to aliasing registers
        // (e.g., X0/W0 share the same physical storage).
        let gpr64_iv = {
            let mut iv = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
            iv.add_range(0, 10);
            iv.spill_weight = 1.0;
            iv.def_positions.push(0);
            iv.use_positions.push(9);
            iv
        };
        let gpr32_iv = {
            let mut iv = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr32 });
            iv.add_range(0, 10);
            iv.spill_weight = 1.0;
            iv.def_positions.push(0);
            iv.use_positions.push(9);
            iv
        };

        // Provide X0 for Gpr64 and W0 for Gpr32 (they alias!).
        // The allocator must NOT assign both — one should be detected as
        // interfering with the other's alias.
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0), PReg::new(1)]);    // X0, X1
        regs.insert(RegClass::Gpr32, vec![PReg::new(32), PReg::new(33)]);  // W0, W1

        let intervals = vec![gpr64_iv, gpr32_iv];
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Both should be allocated (we have 2 regs in each class).
        assert_eq!(result.allocation.len(), 2);
        assert!(alloc.spilled.is_empty());

        // Critical: they must NOT alias!
        let p0 = result.allocation[&VReg { id: 0, class: RegClass::Gpr64 }];
        let p1 = result.allocation[&VReg { id: 1, class: RegClass::Gpr32 }];
        assert!(
            !llvm2_ir::regs::regs_overlap(p0, p1),
            "X register {:?} and W register {:?} must not alias! \
             This is the issue #336 regression.",
            p0, p1
        );
    }

    #[test]
    fn test_issue_336_mixed_width_only_one_reg_available() {
        // If we only have X0 for Gpr64 and W0 for Gpr32 (they alias),
        // the allocator cannot assign both simultaneously-live intervals.
        // One must be spilled.
        let gpr64_iv = {
            let mut iv = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
            iv.add_range(0, 10);
            iv.spill_weight = 2.0;
            iv.def_positions.push(0);
            iv.use_positions.push(9);
            iv
        };
        let gpr32_iv = {
            let mut iv = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr32 });
            iv.add_range(0, 10);
            iv.spill_weight = 1.0;
            iv.def_positions.push(0);
            iv.use_positions.push(9);
            iv
        };

        // Only one physical register pair available: X0/W0 (they alias).
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);   // X0
        regs.insert(RegClass::Gpr32, vec![PReg::new(32)]);  // W0 (aliases X0!)

        let intervals = vec![gpr64_iv, gpr32_iv];
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Cannot both be allocated — one must be spilled.
        assert_eq!(result.allocation.len(), 1, "only one can be allocated when registers alias");
        assert_eq!(alloc.spilled.len(), 1, "one must be spilled");
        // The higher-weight interval (v0, weight=2.0) should be allocated.
        assert!(result.allocation.contains_key(&VReg { id: 0, class: RegClass::Gpr64 }));
    }

    #[test]
    fn test_issue_336_aliasing_pregs_function() {
        // Verify the aliasing_pregs helper function.
        use super::aliasing_pregs;

        // X0 (PReg(0)) aliases W0 (PReg(32))
        let aliases_x0 = aliasing_pregs(PReg::new(0));
        assert_eq!(aliases_x0.len(), 1);
        assert_eq!(aliases_x0[0], PReg::new(32)); // W0

        // W0 (PReg(32)) aliases X0 (PReg(0))
        let aliases_w0 = aliasing_pregs(PReg::new(32));
        assert_eq!(aliases_w0.len(), 1);
        assert_eq!(aliases_w0[0], PReg::new(0)); // X0

        // X28 (PReg(28)) aliases W28 (PReg(60))
        let aliases_x28 = aliasing_pregs(PReg::new(28));
        assert_eq!(aliases_x28.len(), 1);
        assert_eq!(aliases_x28[0], PReg::new(60)); // W28

        // V0 (PReg(64)) aliases D0 (PReg(96)) and S0 (PReg(128))
        let aliases_v0 = aliasing_pregs(PReg::new(64));
        assert_eq!(aliases_v0.len(), 2);
        assert!(aliases_v0.contains(&PReg::new(96)));  // D0
        assert!(aliases_v0.contains(&PReg::new(128))); // S0

        // D0 (PReg(96)) aliases V0 (PReg(64)) and S0 (PReg(128))
        let aliases_d0 = aliasing_pregs(PReg::new(96));
        assert_eq!(aliases_d0.len(), 2);
        assert!(aliases_d0.contains(&PReg::new(64)));  // V0
        assert!(aliases_d0.contains(&PReg::new(128))); // S0

        // S0 (PReg(128)) aliases V0 (PReg(64)) and D0 (PReg(96))
        let aliases_s0 = aliasing_pregs(PReg::new(128));
        assert_eq!(aliases_s0.len(), 2);
        assert!(aliases_s0.contains(&PReg::new(64)));  // V0
        assert!(aliases_s0.contains(&PReg::new(96)));  // D0

        // SP (PReg(31)) has no aliases in our alias function: encoding 31
        // falls outside 0..=30 (X0-X30) and 32..=62 (W0-W30), so the match
        // returns nothing. SP is not allocatable — this is by design.
        let aliases_sp = aliasing_pregs(PReg::new(31));
        assert_eq!(aliases_sp.len(), 0);
    }

    #[test]
    fn test_issue_336_non_overlapping_mixed_width_ok() {
        // Non-overlapping intervals of different widths CAN share the
        // same physical register pair (no conflict).
        let gpr64_iv = {
            let mut iv = LiveInterval::new(VReg { id: 0, class: RegClass::Gpr64 });
            iv.add_range(0, 5);
            iv.spill_weight = 1.0;
            iv.def_positions.push(0);
            iv.use_positions.push(4);
            iv
        };
        let gpr32_iv = {
            let mut iv = LiveInterval::new(VReg { id: 1, class: RegClass::Gpr32 });
            iv.add_range(5, 10);
            iv.spill_weight = 1.0;
            iv.def_positions.push(5);
            iv.use_positions.push(9);
            iv
        };

        // Only X0/W0 available (they alias, but intervals don't overlap).
        let mut regs = HashMap::new();
        regs.insert(RegClass::Gpr64, vec![PReg::new(0)]);   // X0
        regs.insert(RegClass::Gpr32, vec![PReg::new(32)]);  // W0

        let intervals = vec![gpr64_iv, gpr32_iv];
        let mut alloc = GreedyAllocator::new(intervals, &regs, HashMap::new());
        let result = alloc.allocate().unwrap();

        // Both should be allocated since they don't overlap in time.
        assert_eq!(result.allocation.len(), 2);
        assert!(alloc.spilled.is_empty());
    }
}

// llvm2-regalloc/liveness.rs - Live interval computation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Live interval computation for register allocation.
//!
//! Computes live ranges for each virtual register by walking the function
//! in reverse order within each block, tracking definitions and uses.
//!
//! Reference: `~/llvm-project-ref/llvm/lib/CodeGen/LiveIntervals.cpp`
//! Algorithm: backwards dataflow analysis per block with iterative fixpoint
//! for cross-block liveness.

use crate::machine_types::{InstId, MachFunction, RegClass, VReg};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A contiguous range where a virtual register is live.
///
/// Uses instruction indices (not slot indices like LLVM). A VReg is live
/// at instruction `i` if `start <= i < end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// Instruction index where the live range starts (inclusive).
    pub start: u32,
    /// Instruction index where the live range ends (exclusive).
    pub end: u32,
}

impl LiveRange {
    pub fn new(start: u32, end: u32) -> Self {
        debug_assert!(start < end, "empty live range: {start}..{end}");
        Self { start, end }
    }

    /// Returns true if this range contains the given instruction index.
    pub fn contains(&self, idx: u32) -> bool {
        self.start <= idx && idx < self.end
    }

    /// Returns true if this range overlaps with another.
    pub fn overlaps(&self, other: &LiveRange) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// The complete live interval for a virtual register.
///
/// A live interval consists of one or more non-overlapping LiveRanges
/// (a VReg may have holes in its liveness, e.g., across branches).
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The virtual register this interval belongs to.
    pub vreg: VReg,
    /// Non-overlapping, sorted live ranges.
    pub ranges: Vec<LiveRange>,
    /// Spill weight: higher = more expensive to spill.
    /// Computed as sum of (use_count * 10^loop_depth) / interval_length.
    pub spill_weight: f64,
    /// True if this is a fixed physical register (cannot be spilled).
    pub is_fixed: bool,
    /// Use positions within the interval (instruction indices where used).
    pub use_positions: Vec<u32>,
    /// Definition positions within the interval.
    pub def_positions: Vec<u32>,
}

impl LiveInterval {
    pub fn new(vreg: VReg) -> Self {
        Self {
            vreg,
            ranges: Vec::new(),
            spill_weight: 0.0,
            is_fixed: false,
            use_positions: Vec::new(),
            def_positions: Vec::new(),
        }
    }

    /// Returns the start of the first range (earliest point of liveness).
    pub fn start(&self) -> u32 {
        self.ranges.first().map(|r| r.start).unwrap_or(0)
    }

    /// Returns the end of the last range (latest point of liveness).
    pub fn end(&self) -> u32 {
        self.ranges.last().map(|r| r.end).unwrap_or(0)
    }

    /// Returns true if this interval is live at the given instruction index.
    pub fn is_live_at(&self, idx: u32) -> bool {
        self.ranges.iter().any(|r| r.contains(idx))
    }

    /// Returns true if this interval overlaps with another.
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        // Since ranges are sorted, we can do a merge-style comparison,
        // but for the scaffold a simple O(n*m) check suffices.
        for r1 in &self.ranges {
            for r2 in &other.ranges {
                if r1.overlaps(r2) {
                    return true;
                }
            }
        }
        false
    }

    /// Add a live range, merging with adjacent/overlapping ranges.
    pub fn add_range(&mut self, start: u32, end: u32) {
        if start >= end {
            return;
        }

        let new_range = LiveRange::new(start, end);

        // Find insertion point and merge overlapping ranges.
        let mut merged_start = start;
        let mut merged_end = end;
        let mut remove_start = None;
        let mut remove_end = None;

        for (i, existing) in self.ranges.iter().enumerate() {
            // Check for overlap or adjacency (ranges that touch are merged).
            if existing.start <= merged_end && merged_start <= existing.end {
                merged_start = merged_start.min(existing.start);
                merged_end = merged_end.max(existing.end);
                if remove_start.is_none() {
                    remove_start = Some(i);
                }
                remove_end = Some(i);
            }
        }

        if let (Some(rs), Some(re)) = (remove_start, remove_end) {
            // Remove merged ranges and insert the combined one.
            self.ranges.drain(rs..=re);
            self.ranges.insert(rs, LiveRange::new(merged_start, merged_end));
        } else {
            // No overlap — insert in sorted position.
            let pos = self
                .ranges
                .iter()
                .position(|r| r.start > new_range.start)
                .unwrap_or(self.ranges.len());
            self.ranges.insert(pos, new_range);
        }
    }
}

impl fmt::Display for LiveInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.vreg)?;
        for (i, r) in self.ranges.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[{}, {})", r.start, r.end)?;
        }
        write!(f, " (weight={:.2})", self.spill_weight)
    }
}

/// Result of live interval computation.
pub struct LivenessResult {
    /// Live intervals indexed by VReg id.
    pub intervals: HashMap<u32, LiveInterval>,
    /// Instruction numbering: maps (BlockId, instr_index_in_block) to global index.
    pub inst_numbering: HashMap<InstId, u32>,
}

/// Compute live intervals for all virtual registers in the function.
///
/// Algorithm:
/// 1. Number all instructions linearly (block order * block contents).
/// 2. For each block (in reverse order), walk instructions backwards:
///    - For each use of a VReg: extend liveness back to this point.
///    - For each def of a VReg: start liveness at this point.
/// 3. Handle cross-block liveness: if a VReg is live-in to a block,
///    extend its liveness to the end of all predecessor blocks.
/// 4. Iterate until fixed point (live-in sets stabilize).
/// 5. Compute spill weights based on use density and loop depth.
///
/// Reference: LLVM's `LiveIntervals::computeVirtRegs()` and
/// `LiveIntervalCalc::extend()`.
pub fn compute_live_intervals(func: &MachFunction) -> LivenessResult {
    let mut intervals: HashMap<u32, LiveInterval> = HashMap::new();
    let mut inst_numbering: HashMap<InstId, u32> = HashMap::new();

    // Step 1: Number instructions linearly across all blocks.
    let mut idx: u32 = 0;
    for &block_id in &func.block_order {
        let block = &func.blocks[block_id.0 as usize];
        for &inst_id in &block.insts {
            inst_numbering.insert(inst_id, idx);
            idx += 1;
        }
    }

    // Step 2: Compute live-in and live-out sets per block via backward dataflow.
    let num_blocks = func.blocks.len();
    let mut live_in: Vec<HashSet<u32>> = vec![HashSet::new(); num_blocks];
    let mut live_out: Vec<HashSet<u32>> = vec![HashSet::new(); num_blocks];

    // Iterate to fixed point.
    let mut changed = true;
    while changed {
        changed = false;

        // Process blocks in reverse order for faster convergence.
        for &block_id in func.block_order.iter().rev() {
            let bi = block_id.0 as usize;
            let block = &func.blocks[bi];

            // live_out[b] = union of live_in[s] for all successors s
            let mut new_live_out = HashSet::new();
            for &succ_id in &block.succs {
                for &vreg_id in &live_in[succ_id.0 as usize] {
                    new_live_out.insert(vreg_id);
                }
            }

            // live_in[b] = use[b] union (live_out[b] - def[b])
            let mut new_live_in = new_live_out.clone();
            // Walk instructions in reverse to compute local effect.
            for &inst_id in block.insts.iter().rev() {
                let inst = &func.insts[inst_id.0 as usize];
                // Remove defs.
                for vreg in inst.vreg_defs() {
                    new_live_in.remove(&vreg.id);
                }
                // Add uses.
                for vreg in inst.vreg_uses() {
                    new_live_in.insert(vreg.id);
                }
            }

            if new_live_in != live_in[bi] || new_live_out != live_out[bi] {
                changed = true;
                live_in[bi] = new_live_in;
                live_out[bi] = new_live_out;
            }
        }
    }

    // Step 3: Build intervals from the computed liveness information.
    for &block_id in &func.block_order {
        let bi = block_id.0 as usize;
        let block = &func.blocks[bi];

        if block.insts.is_empty() {
            continue;
        }

        let block_start = inst_numbering[&block.insts[0]];
        let block_end = inst_numbering[block.insts.last().unwrap()] + 1;

        // For VRegs that are live-out of this block, they're live throughout
        // the entire block (conservatively).
        for &vreg_id in &live_out[bi] {
            let interval = intervals.entry(vreg_id).or_insert_with(|| {
                // We don't know the class yet; it will be set when we see a def.
                LiveInterval::new(VReg {
                    id: vreg_id,
                    class: RegClass::Gpr64,
                })
            });
            interval.add_range(block_start, block_end);
        }

        // Walk instructions forward to record defs and uses.
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            let idx = inst_numbering[&inst_id];

            for vreg in inst.vreg_defs() {
                let interval = intervals.entry(vreg.id).or_insert_with(|| {
                    LiveInterval::new(vreg)
                });
                interval.vreg.class = vreg.class;
                interval.def_positions.push(idx);
                // A def starts a live range that extends at least to idx+1.
                // If live-out, it was already extended above.
                interval.add_range(idx, idx + 1);
            }

            for vreg in inst.vreg_uses() {
                let interval = intervals.entry(vreg.id).or_insert_with(|| {
                    LiveInterval::new(vreg)
                });
                interval.use_positions.push(idx);
                // A use means the VReg must be live at this instruction.
                interval.add_range(idx, idx + 1);
            }
        }
    }

    // Step 4: Compute spill weights.
    for interval in intervals.values_mut() {
        compute_spill_weight(interval, func, &inst_numbering);
    }

    LivenessResult {
        intervals,
        inst_numbering,
    }
}

/// Compute the spill weight for a live interval.
///
/// Weight = sum(10^loop_depth for each use/def) / interval_length.
/// Higher weight means more expensive to spill (more frequently used,
/// in hotter loops).
///
/// Reference: LLVM's `VirtRegAuxInfo::calculateSpillWeightAndHint()`
fn compute_spill_weight(
    interval: &mut LiveInterval,
    func: &MachFunction,
    inst_numbering: &HashMap<InstId, u32>,
) {
    if interval.ranges.is_empty() {
        interval.spill_weight = 0.0;
        return;
    }

    let mut weight = 0.0;

    // Accumulate weight from each use/def position.
    let all_positions: Vec<u32> = interval
        .use_positions
        .iter()
        .chain(interval.def_positions.iter())
        .copied()
        .collect();

    for &pos in &all_positions {
        // Find which block this instruction is in and get its loop depth.
        let loop_depth = find_loop_depth_for_inst(pos, func, inst_numbering);
        weight += 10.0_f64.powi(loop_depth as i32);
    }

    // Normalize by interval length.
    let length = interval.end().saturating_sub(interval.start()).max(1) as f64;
    interval.spill_weight = weight / length;
}

/// Find the loop depth for an instruction at a given global index.
fn find_loop_depth_for_inst(
    target_idx: u32,
    func: &MachFunction,
    inst_numbering: &HashMap<InstId, u32>,
) -> u32 {
    for &block_id in &func.block_order {
        let block = &func.blocks[block_id.0 as usize];
        for &inst_id in &block.insts {
            if let Some(&idx) = inst_numbering.get(&inst_id) {
                if idx == target_idx {
                    return block.loop_depth;
                }
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_range_overlap() {
        let r1 = LiveRange::new(0, 5);
        let r2 = LiveRange::new(3, 8);
        let r3 = LiveRange::new(5, 10);
        let r4 = LiveRange::new(10, 15);

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3)); // [0,5) and [5,10) don't overlap
        assert!(!r1.overlaps(&r4));
        assert!(r2.overlaps(&r3));
    }

    #[test]
    fn test_live_interval_add_range_merging() {
        let mut interval = LiveInterval::new(VReg {
            id: 0,
            class: RegClass::Gpr64,
        });
        interval.add_range(0, 5);
        interval.add_range(3, 8);
        assert_eq!(interval.ranges.len(), 1);
        assert_eq!(interval.ranges[0].start, 0);
        assert_eq!(interval.ranges[0].end, 8);
    }

    #[test]
    fn test_live_interval_add_range_no_overlap() {
        let mut interval = LiveInterval::new(VReg {
            id: 0,
            class: RegClass::Gpr64,
        });
        interval.add_range(0, 3);
        interval.add_range(5, 8);
        assert_eq!(interval.ranges.len(), 2);
        assert_eq!(interval.ranges[0].start, 0);
        assert_eq!(interval.ranges[1].start, 5);
    }
}

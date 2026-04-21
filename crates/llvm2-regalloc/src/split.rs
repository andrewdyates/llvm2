// llvm2-regalloc/split.rs - Live interval splitting
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Live interval splitting for the register allocator.
//!
//! When a long interval is expensive to spill, we can split it around
//! uses instead of spilling the whole thing. This reduces spill code by
//! keeping the value in a register only where it's actually needed.
//!
//! The algorithm finds optimal split points by analyzing gaps between
//! consecutive use/def positions and splitting at the largest gap.
//!
//! Reference: LLVM `SplitKit.cpp` — simplified for our linear-scan context.

use crate::liveness::LiveInterval;
use crate::machine_types::{
    InstFlags, InstId, MachFunction, MachInst, MachOperand, VReg,
};
use crate::phi_elim;

/// Describes where to split a live interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitDecision {
    /// No beneficial split exists.
    NoSplit,
    /// Split just before a use at the given instruction index.
    SplitBeforeUse(u32),
    /// Split just after a def at the given instruction index.
    SplitAfterDef(u32),
    /// Split around a region, creating a hole in [start, end).
    SplitAroundRegion { start: u32, end: u32 },
}

/// Result of splitting a live interval.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// The original virtual register (interval truncated).
    pub original_vreg: VReg,
    /// The new virtual register (covers the split-off portion).
    pub new_vreg: VReg,
    /// The truncated original interval (covers [orig_start, split_point)).
    pub original_interval: LiveInterval,
    /// The new interval (covers [split_point, orig_end)).
    pub new_interval: LiveInterval,
}

/// Analyze an interval for split candidates.
///
/// Returns a list of beneficial split decisions ordered by quality
/// (largest gap first). An empty list means no beneficial split exists.
pub fn analyze_split_candidates(
    interval: &LiveInterval,
    _active_intervals: &[&LiveInterval],
    _allocatable_count: usize,
) -> Vec<SplitDecision> {
    let mut decisions = Vec::new();

    // Collect all use/def positions and sort them.
    let mut positions: Vec<u32> = interval
        .use_positions
        .iter()
        .chain(interval.def_positions.iter())
        .copied()
        .collect();
    positions.sort_unstable();
    positions.dedup();

    if positions.len() < 2 {
        return decisions;
    }

    // Find gaps between consecutive positions and score them.
    let mut gaps: Vec<(u32, u32, u32)> = Vec::new(); // (gap_size, start, end)
    for window in positions.windows(2) {
        let gap_start = window[0] + 1;
        let gap_end = window[1];
        if gap_end > gap_start {
            gaps.push((gap_end - gap_start, gap_start, gap_end));
        }
    }

    // Sort by gap size descending (largest gaps first — best split candidates).
    gaps.sort_by(|a, b| b.0.cmp(&a.0));

    for (gap_size, gap_start, gap_end) in gaps {
        // Only consider splits where the gap is at least 2 instructions wide.
        if gap_size >= 2 {
            decisions.push(SplitDecision::SplitAroundRegion {
                start: gap_start,
                end: gap_end,
            });
        } else {
            decisions.push(SplitDecision::SplitBeforeUse(gap_end));
        }
    }

    if decisions.is_empty() {
        decisions.push(SplitDecision::NoSplit);
    }

    decisions
}

/// Find the optimal split point for a live interval.
///
/// The optimal split point is at the middle of the largest gap between
/// consecutive use/def positions, minimizing the register pressure on
/// both sides.
pub fn find_optimal_split_point(interval: &LiveInterval) -> Option<u32> {
    let mut positions: Vec<u32> = interval
        .use_positions
        .iter()
        .chain(interval.def_positions.iter())
        .copied()
        .collect();
    positions.sort_unstable();
    positions.dedup();

    if positions.len() < 2 {
        return None;
    }

    let mut best_gap = 0u32;
    let mut best_mid = None;

    for window in positions.windows(2) {
        let gap = window[1].saturating_sub(window[0]);
        if gap > best_gap {
            best_gap = gap;
            best_mid = Some(window[0] + gap / 2);
        }
    }

    // Only split if the gap is meaningful (at least 2 instructions).
    if best_gap >= 2 {
        best_mid
    } else {
        None
    }
}

/// Split a live interval at the given split point.
///
/// Creates two intervals:
/// - Original: covers ranges before the split point.
/// - New: covers ranges at and after the split point.
///
/// A `PSEUDO_COPY` instruction is inserted to connect them: the new VReg
/// is defined as a copy of the original at the split point.
///
/// Returns `None` if the split point is outside the interval or would
/// produce empty intervals.
pub fn split_interval(
    interval: &LiveInterval,
    split_point: u32,
    func: &mut MachFunction,
) -> Option<SplitResult> {
    // Validate: split point must be within the interval's overall extent.
    if split_point <= interval.start() || split_point >= interval.end() {
        return None;
    }

    let original_vreg = interval.vreg;
    let new_vreg = func.alloc_vreg(original_vreg.class);

    // Build the two new intervals by partitioning ranges.
    let mut original_interval = LiveInterval::new(original_vreg);
    let mut new_interval = LiveInterval::new(new_vreg);

    for range in &interval.ranges {
        if range.end <= split_point {
            // Entirely before split.
            original_interval.add_range(range.start, range.end);
        } else if range.start >= split_point {
            // Entirely after split.
            new_interval.add_range(range.start, range.end);
        } else {
            // Range spans the split point — split it.
            original_interval.add_range(range.start, split_point);
            new_interval.add_range(split_point, range.end);
        }
    }

    // Don't split if either side would be empty.
    if original_interval.ranges.is_empty() || new_interval.ranges.is_empty() {
        return None;
    }

    // Partition use/def positions.
    for &pos in &interval.use_positions {
        if pos < split_point {
            original_interval.use_positions.push(pos);
        } else {
            new_interval.use_positions.push(pos);
        }
    }
    for &pos in &interval.def_positions {
        if pos < split_point {
            original_interval.def_positions.push(pos);
        } else {
            new_interval.def_positions.push(pos);
        }
    }

    // Distribute spill weight proportionally.
    let total_len = interval.end().saturating_sub(interval.start()).max(1) as f64;
    let orig_len = original_interval
        .end()
        .saturating_sub(original_interval.start())
        .max(1) as f64;
    let new_len = new_interval
        .end()
        .saturating_sub(new_interval.start())
        .max(1) as f64;
    original_interval.spill_weight = interval.spill_weight * (orig_len / total_len);
    new_interval.spill_weight = interval.spill_weight * (new_len / total_len);

    // Insert a PSEUDO_COPY at the split point: new_vreg <- original_vreg.
    let copy_inst = MachInst {
        opcode: phi_elim::PSEUDO_COPY,
        defs: vec![MachOperand::VReg(new_vreg)],
        uses: vec![MachOperand::VReg(original_vreg)],
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        flags: InstFlags::default(),
    };

    let copy_id = InstId(func.insts.len() as u32);
    func.insts.push(copy_inst);

    // Insert the copy at the appropriate block position.
    // Find the block that contains the split point and insert the copy.
    insert_copy_at_point(func, copy_id, split_point);

    Some(SplitResult {
        original_vreg,
        new_vreg,
        original_interval,
        new_interval,
    })
}

/// Insert a copy instruction at the given program point.
///
/// Finds the block containing instructions around `point` and inserts
/// the copy before the instruction at that index.
fn insert_copy_at_point(func: &mut MachFunction, copy_id: InstId, point: u32) {
    // Walk blocks to find where to insert based on instruction numbering.
    let mut global_idx: u32 = 0;
    for block in &mut func.blocks {
        let block_start = global_idx;
        let block_end = global_idx + block.insts.len() as u32;

        if point >= block_start && point < block_end {
            let local_pos = (point - block_start) as usize;
            block.insts.insert(local_pos, copy_id);
            return;
        }

        global_idx = block_end;
    }

    // Fallback: append to the last block.
    if let Some(last_block) = func.blocks.last_mut() {
        last_block.insts.push(copy_id);
    }
}

/// Find the best split point near an interference region.
///
/// Given an interval and the point where interference starts, find
/// the best split point that separates the interval into a part that
/// can be allocated and a part that can be re-enqueued.
///
/// Strategy: split just after the last use/def before the interference
/// point, so the first half is as long as possible while still
/// fitting in a register.
pub fn find_split_near_interference(
    interval: &LiveInterval,
    interference_start: u32,
) -> Option<u32> {
    let mut positions: Vec<u32> = interval
        .use_positions
        .iter()
        .chain(interval.def_positions.iter())
        .copied()
        .filter(|&p| p < interference_start)
        .collect();
    positions.sort_unstable();

    if positions.is_empty() {
        return None;
    }

    let last_before = *positions.last().unwrap();
    let split_point = last_before + 1;

    if split_point <= interval.start() || split_point >= interval.end() {
        return None;
    }

    Some(split_point)
}

/// Find split points between consecutive use/def positions.
///
/// This is the most aggressive splitting strategy. Each resulting
/// interval covers only a small region around its uses, making it
/// easy to allocate. The cost is more spill/reload traffic.
///
/// Returns a list of `(split_point, weight)` pairs for each viable
/// split, sorted by descending weight (most beneficial first).
/// Weight equals the gap size between the consecutive positions,
/// so larger gaps are preferred.
pub fn find_per_use_split_points(interval: &LiveInterval) -> Vec<(u32, f64)> {
    let mut positions: Vec<u32> = interval
        .use_positions
        .iter()
        .chain(interval.def_positions.iter())
        .copied()
        .collect();
    positions.sort_unstable();
    positions.dedup();

    if positions.len() < 2 {
        return Vec::new();
    }

    let mut splits = Vec::new();

    for window in positions.windows(2) {
        let gap = window[1].saturating_sub(window[0]);
        if gap >= 2 {
            let split_point = window[0] + 1;
            if split_point > interval.start() && split_point < interval.end() {
                splits.push((split_point, gap as f64));
            }
        }
    }

    splits.sort_by(|a, b| b.1.total_cmp(&a.1));
    splits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::LiveInterval;
    use crate::machine_types::{
        BlockId, InstFlags, InstId, MachBlock, MachFunction, MachInst, MachOperand, RegClass,
        VReg,
    };
    use std::collections::HashMap;

    fn vreg(id: u32) -> VReg {
        VReg {
            id,
            class: RegClass::Gpr64,
        }
    }

    fn make_interval(id: u32, ranges: &[(u32, u32)], uses: &[u32], defs: &[u32]) -> LiveInterval {
        let mut interval = LiveInterval::new(vreg(id));
        for &(start, end) in ranges {
            interval.add_range(start, end);
        }
        interval.use_positions = uses.to_vec();
        interval.def_positions = defs.to_vec();
        interval.spill_weight = 1.0;
        interval
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

    #[test]
    fn test_find_optimal_split_point_large_gap() {
        // Interval: [0, 20) with uses at 2 and 18.
        // Largest gap: 2..18 (size 16), midpoint = 10.
        let interval = make_interval(0, &[(0, 20)], &[2, 18], &[0]);
        let split = find_optimal_split_point(&interval);
        assert_eq!(split, Some(10));
    }

    #[test]
    fn test_find_optimal_split_point_no_gap() {
        // Interval with consecutive uses — no meaningful gap.
        let interval = make_interval(0, &[(0, 3)], &[0, 1, 2], &[]);
        let split = find_optimal_split_point(&interval);
        assert_eq!(split, None);
    }

    #[test]
    fn test_find_optimal_split_point_single_use() {
        let interval = make_interval(0, &[(0, 10)], &[5], &[]);
        let split = find_optimal_split_point(&interval);
        // Only one position — can't split.
        assert_eq!(split, None);
    }

    #[test]
    fn test_analyze_split_candidates_returns_sorted() {
        // Uses at 0, 5, 15, 20.
        // Gaps: (5,1,5), (10,6,15), (5,16,20).
        // Largest gap is 6..15 (size 10).
        let interval = make_interval(0, &[(0, 25)], &[0, 5, 15, 20], &[]);
        let candidates = analyze_split_candidates(&interval, &[], 10);
        assert!(!candidates.is_empty());
        // The first candidate should be the largest gap.
        match &candidates[0] {
            SplitDecision::SplitAroundRegion { start, end } => {
                assert_eq!(*start, 6);
                assert_eq!(*end, 15);
            }
            other => panic!("Expected SplitAroundRegion, got {:?}", other),
        }
    }

    #[test]
    fn test_split_interval_basic() {
        let mut func = make_test_func(20);
        let interval = make_interval(0, &[(0, 20)], &[2, 18], &[0]);

        let result = split_interval(&interval, 10, &mut func);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.original_vreg, vreg(0));
        assert_ne!(result.new_vreg, vreg(0));

        // Original interval should cover [0, 10).
        assert_eq!(result.original_interval.start(), 0);
        assert_eq!(result.original_interval.end(), 10);

        // New interval should cover [10, 20).
        assert_eq!(result.new_interval.start(), 10);
        assert_eq!(result.new_interval.end(), 20);
    }

    #[test]
    fn test_split_interval_out_of_range() {
        let mut func = make_test_func(10);
        let interval = make_interval(0, &[(0, 10)], &[2, 8], &[0]);

        // Split at 0 (start boundary) should fail.
        assert!(split_interval(&interval, 0, &mut func).is_none());
        // Split at 10 (end boundary) should fail.
        assert!(split_interval(&interval, 10, &mut func).is_none());
        // Split beyond range should fail.
        assert!(split_interval(&interval, 15, &mut func).is_none());
    }

    #[test]
    fn test_split_interval_with_holes() {
        let mut func = make_test_func(20);
        // Interval has a hole: [0,5) and [10,20).
        let interval = make_interval(0, &[(0, 5), (10, 20)], &[2, 15], &[0, 10]);

        let result = split_interval(&interval, 7, &mut func);
        assert!(result.is_some());

        let result = result.unwrap();
        // Original: [0, 5) (the [10,20) range is after split point).
        assert_eq!(result.original_interval.ranges.len(), 1);
        assert_eq!(result.original_interval.start(), 0);
        assert_eq!(result.original_interval.end(), 5);

        // New: [10, 20) (the range starting at 10 is after 7).
        assert_eq!(result.new_interval.ranges.len(), 1);
        assert_eq!(result.new_interval.start(), 10);
        assert_eq!(result.new_interval.end(), 20);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case and correctness tests (issue #139)
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_optimal_split_point_two_uses() {
        // Uses at 2 and 10. Gap = 8, midpoint = 6.
        let interval = make_interval(0, &[(0, 15)], &[2, 10], &[]);
        let split = find_optimal_split_point(&interval);
        assert_eq!(split, Some(6));
    }

    #[test]
    fn test_find_optimal_split_point_equal_gaps() {
        // Uses at 0, 5, 10 — two gaps of size 5 each.
        let interval = make_interval(0, &[(0, 15)], &[0, 5, 10], &[]);
        let split = find_optimal_split_point(&interval);
        // Both gaps are 5, so should pick the first one found: mid of [0,5] = 2.
        assert!(split.is_some());
        let sp = split.unwrap();
        // Either gap midpoint is valid.
        assert!(sp == 2 || sp == 7, "split should be at midpoint of one gap: got {sp}");
    }

    #[test]
    fn test_find_optimal_split_point_gap_size_1() {
        // Uses at 3 and 5. Gap = 2. Midpoint = 4. Gap >= 2 so should split.
        let interval = make_interval(0, &[(0, 10)], &[3, 5], &[]);
        let split = find_optimal_split_point(&interval);
        assert_eq!(split, Some(4));
    }

    #[test]
    fn test_find_optimal_split_point_empty_interval() {
        let interval = make_interval(0, &[], &[], &[]);
        let split = find_optimal_split_point(&interval);
        assert_eq!(split, None);
    }

    #[test]
    fn test_analyze_split_candidates_empty_positions() {
        let interval = make_interval(0, &[(0, 10)], &[], &[]);
        let candidates = analyze_split_candidates(&interval, &[], 10);
        assert!(candidates.is_empty() || candidates[0] == SplitDecision::NoSplit);
    }

    #[test]
    fn test_analyze_split_candidates_small_gap() {
        // Uses at 0 and 2. Gap = 2 but gap_start=1, gap_end=2 -> gap_size=1.
        // gap_size < 2 so should produce SplitBeforeUse.
        let interval = make_interval(0, &[(0, 5)], &[0, 2], &[]);
        let candidates = analyze_split_candidates(&interval, &[], 10);
        assert!(!candidates.is_empty());
        match &candidates[0] {
            SplitDecision::SplitBeforeUse(pos) => {
                assert_eq!(*pos, 2);
            }
            SplitDecision::SplitAroundRegion { .. } => {
                // Also acceptable if gap >= 2.
            }
            other => panic!("Unexpected decision: {:?}", other),
        }
    }

    #[test]
    fn test_analyze_split_candidates_multiple_decisions() {
        // Uses at 0, 10, 30, 35.
        // Gaps: (10, 1, 10) size=9, (20, 11, 30) size=19, (5, 31, 35) size=4.
        // Sorted: size 19 first, then 9, then 4.
        let interval = make_interval(0, &[(0, 40)], &[0, 10, 30, 35], &[]);
        let candidates = analyze_split_candidates(&interval, &[], 10);
        assert!(candidates.len() >= 3, "should have 3 candidates");
        // First should be the largest gap (11..30, size 19).
        match &candidates[0] {
            SplitDecision::SplitAroundRegion { start, end } => {
                assert_eq!(*start, 11);
                assert_eq!(*end, 30);
            }
            other => panic!("Expected largest gap first, got {:?}", other),
        }
    }

    #[test]
    fn test_split_interval_creates_copy_instruction() {
        let mut func = make_test_func(20);
        let interval = make_interval(0, &[(0, 20)], &[2, 18], &[0]);
        let insts_before = func.insts.len();

        let result = split_interval(&interval, 10, &mut func);
        assert!(result.is_some());

        // A PSEUDO_COPY instruction should have been inserted.
        assert!(func.insts.len() > insts_before, "should have added a copy instruction");
        let copy_inst = &func.insts[insts_before];
        assert_eq!(copy_inst.opcode, crate::phi_elim::PSEUDO_COPY);
    }

    #[test]
    fn test_split_interval_allocates_new_vreg() {
        let mut func = make_test_func(20);
        let original_next_vreg = func.next_vreg;
        let interval = make_interval(0, &[(0, 20)], &[2, 18], &[0]);

        let result = split_interval(&interval, 10, &mut func).unwrap();
        assert!(func.next_vreg > original_next_vreg, "should allocate a new vreg");
        assert_eq!(result.new_vreg.id, original_next_vreg);
    }

    #[test]
    fn test_split_interval_use_positions_partitioned() {
        let mut func = make_test_func(20);
        let interval = make_interval(0, &[(0, 20)], &[2, 5, 12, 18], &[0]);

        let result = split_interval(&interval, 10, &mut func).unwrap();

        // Uses before split point go to original.
        assert!(result.original_interval.use_positions.contains(&2));
        assert!(result.original_interval.use_positions.contains(&5));
        // Uses at or after split point go to new.
        assert!(result.new_interval.use_positions.contains(&12));
        assert!(result.new_interval.use_positions.contains(&18));
    }

    #[test]
    fn test_split_interval_spill_weight_distributed() {
        let mut func = make_test_func(20);
        let interval = make_interval(0, &[(0, 20)], &[2, 18], &[0]);

        let result = split_interval(&interval, 10, &mut func).unwrap();

        // Both intervals should have positive spill weight.
        assert!(result.original_interval.spill_weight > 0.0);
        assert!(result.new_interval.spill_weight > 0.0);

        // Combined should roughly equal original.
        let combined = result.original_interval.spill_weight + result.new_interval.spill_weight;
        let diff = (combined - 1.0).abs();
        assert!(diff < 0.01, "combined weight {combined} should be close to 1.0");
    }

    #[test]
    fn test_split_interval_spanning_range() {
        // Split in the middle of a single range.
        let mut func = make_test_func(30);
        let interval = make_interval(0, &[(5, 25)], &[5, 24], &[5]);

        let result = split_interval(&interval, 15, &mut func).unwrap();
        assert_eq!(result.original_interval.start(), 5);
        assert_eq!(result.original_interval.end(), 15);
        assert_eq!(result.new_interval.start(), 15);
        assert_eq!(result.new_interval.end(), 25);
    }

    #[test]
    fn test_split_interval_many_small_ranges() {
        // Multiple small ranges: [0,3), [5,8), [10,13), [15,18).
        // Split at 9 should put first two in original, last two in new.
        let mut func = make_test_func(20);
        let interval = make_interval(
            0,
            &[(0, 3), (5, 8), (10, 13), (15, 18)],
            &[1, 6, 11, 16],
            &[0, 5, 10, 15],
        );

        let result = split_interval(&interval, 9, &mut func).unwrap();

        assert_eq!(result.original_interval.ranges.len(), 2);
        assert_eq!(result.new_interval.ranges.len(), 2);
        assert_eq!(result.original_interval.end(), 8);
        assert_eq!(result.new_interval.start(), 10);
    }

    #[test]
    fn test_split_decision_enum_equality() {
        assert_eq!(SplitDecision::NoSplit, SplitDecision::NoSplit);
        assert_eq!(
            SplitDecision::SplitBeforeUse(5),
            SplitDecision::SplitBeforeUse(5)
        );
        assert_ne!(
            SplitDecision::SplitBeforeUse(5),
            SplitDecision::SplitBeforeUse(10)
        );
        assert_eq!(
            SplitDecision::SplitAroundRegion { start: 3, end: 7 },
            SplitDecision::SplitAroundRegion { start: 3, end: 7 }
        );
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests (issue #404 — TL7 coverage expansion)
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_at_call_boundary() {
        // Simulate splitting an interval at a call boundary.
        // Interval: [0, 30) with uses at 5 and 25, call at position 15.
        // Splitting at 15 should produce two valid halves.
        let mut func = make_test_func(30);
        let interval = make_interval(0, &[(0, 30)], &[5, 25], &[0]);

        let result = split_interval(&interval, 15, &mut func);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.original_interval.start(), 0);
        assert_eq!(result.original_interval.end(), 15);
        assert_eq!(result.new_interval.start(), 15);
        assert_eq!(result.new_interval.end(), 30);

        // Use at 5 should be in original, use at 25 in new.
        assert!(result.original_interval.use_positions.contains(&5));
        assert!(!result.original_interval.use_positions.contains(&25));
        assert!(result.new_interval.use_positions.contains(&25));
        assert!(!result.new_interval.use_positions.contains(&5));
    }

    #[test]
    fn test_split_produces_minimal_length_intervals() {
        // Split a [0, 4) interval at position 2. Both halves should be 2 instructions.
        let mut func = make_test_func(10);
        let interval = make_interval(0, &[(0, 4)], &[0, 3], &[0]);

        let result = split_interval(&interval, 2, &mut func);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.original_interval.start(), 0);
        assert_eq!(result.original_interval.end(), 2);
        assert_eq!(result.new_interval.start(), 2);
        assert_eq!(result.new_interval.end(), 4);
    }

    #[test]
    fn test_split_def_at_exact_split_point_goes_to_new() {
        // A def position exactly at the split point should go to the new interval.
        let mut func = make_test_func(20);
        let interval = make_interval(0, &[(0, 20)], &[2, 15], &[0, 10]);

        let result = split_interval(&interval, 10, &mut func).unwrap();

        // def at 0 < 10 -> original; def at 10 >= 10 -> new
        assert!(result.original_interval.def_positions.contains(&0));
        assert!(!result.original_interval.def_positions.contains(&10));
        assert!(result.new_interval.def_positions.contains(&10));
        assert!(!result.new_interval.def_positions.contains(&0));
    }

    #[test]
    fn test_analyze_split_candidates_single_position_returns_empty() {
        // An interval with only one use/def position cannot be split.
        let interval = make_interval(0, &[(0, 20)], &[10], &[]);
        let candidates = analyze_split_candidates(&interval, &[], 10);
        assert!(candidates.is_empty(), "single position should produce no candidates");
    }

    #[test]
    fn test_split_fpr_interval_preserves_class() {
        // Splitting an FPR interval should produce intervals of the same class.
        let mut func = make_test_func(20);
        let fpr_vreg = VReg { id: 0, class: RegClass::Fpr64 };
        let mut interval = LiveInterval::new(fpr_vreg);
        interval.add_range(0, 20);
        interval.use_positions = vec![2, 18];
        interval.def_positions = vec![0];
        interval.spill_weight = 2.0;

        let result = split_interval(&interval, 10, &mut func).unwrap();

        assert_eq!(result.original_vreg.class, RegClass::Fpr64);
        assert_eq!(result.new_vreg.class, RegClass::Fpr64);
        assert_eq!(result.original_interval.vreg.class, RegClass::Fpr64);
        assert_eq!(result.new_interval.vreg.class, RegClass::Fpr64);
    }

    // -----------------------------------------------------------------------
    // Tests for interference-aware and per-use splitting (issue #332)
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_split_near_interference_basic() {
        // Interval [0, 20) with uses at 2, 8, 15 and def at 0.
        // Interference starts at position 10.
        // Should split after the last use before interference (use at 8) -> position 9.
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 20);
        iv.use_positions = vec![2, 8, 15];
        iv.def_positions = vec![0];

        let split = find_split_near_interference(&iv, 10);
        assert!(split.is_some());
        assert_eq!(split.unwrap(), 9);
    }

    #[test]
    fn test_find_split_near_interference_no_uses_before() {
        // All uses/defs are after the interference point.
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 20);
        iv.use_positions = vec![12, 18];
        iv.def_positions = vec![10];

        let split = find_split_near_interference(&iv, 5);
        assert!(split.is_none(), "no uses before interference, can't split usefully");
    }

    #[test]
    fn test_find_split_near_interference_at_boundary() {
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 10);
        iv.use_positions = vec![0, 9];
        iv.def_positions = vec![0];

        // Interference at position 5.
        let split = find_split_near_interference(&iv, 5);
        assert!(split.is_some());
        let sp = split.unwrap();
        assert!(sp > iv.start() && sp < iv.end());
    }

    #[test]
    fn test_find_split_near_interference_def_only_before() {
        // Only a def before interference, no uses.
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 20);
        iv.use_positions = vec![15];
        iv.def_positions = vec![0];

        let split = find_split_near_interference(&iv, 10);
        assert!(split.is_some());
        // Should split after def at 0 -> position 1.
        assert_eq!(split.unwrap(), 1);
    }

    #[test]
    fn test_find_per_use_split_points_basic() {
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 30);
        iv.use_positions = vec![3, 7, 15, 25];
        iv.def_positions = vec![0];

        let splits = find_per_use_split_points(&iv);
        assert!(!splits.is_empty(), "should find split points between uses");

        // All split points should be within the interval.
        for (sp, _weight) in &splits {
            assert!(*sp > iv.start());
            assert!(*sp < iv.end());
        }

        // The first split should have the highest weight (largest gap).
        if splits.len() >= 2 {
            assert!(
                splits[0].1 >= splits[1].1,
                "splits should be sorted by weight descending"
            );
        }
    }

    #[test]
    fn test_find_per_use_split_points_empty() {
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 10);
        // Only one use -- can't split.
        iv.use_positions = vec![5];

        let splits = find_per_use_split_points(&iv);
        assert!(splits.is_empty());
    }

    #[test]
    fn test_find_per_use_split_points_consecutive_uses() {
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 10);
        // Uses at 3, 4, 5 -- gaps of 1, too small between uses.
        iv.use_positions = vec![3, 4, 5];
        iv.def_positions = vec![0];

        let splits = find_per_use_split_points(&iv);
        // Gap between def at 0 and use at 3 is 3, which is >= 2.
        assert!(
            !splits.is_empty(),
            "should find at least one split from def to first use"
        );
    }

    #[test]
    fn test_find_per_use_split_points_all_adjacent() {
        // All positions are adjacent -- no gaps >= 2.
        let mut iv = LiveInterval::new(vreg(0));
        iv.add_range(0, 5);
        iv.use_positions = vec![0, 1, 2, 3, 4];
        iv.def_positions = vec![];

        let splits = find_per_use_split_points(&iv);
        assert!(splits.is_empty(), "no gaps >= 2 means no viable split points");
    }
}

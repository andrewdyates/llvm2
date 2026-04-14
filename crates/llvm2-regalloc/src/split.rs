// llvm2-regalloc/split.rs - Live interval splitting
//
// Author: Andrew Yates <ayates@dropbox.com>
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
}

// llvm2-regalloc/coalesce.rs - Copy coalescing for register allocation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Copy coalescing for phi-elimination copies.
//!
//! After phi elimination inserts `PSEUDO_COPY` instructions, many of these
//! copies can be eliminated by merging the live intervals of the source and
//! destination when they don't interfere. This pass computes a deferred edit
//! plan rather than mutating the `MachFunction` directly; the returned
//! removals and rewrites can be applied later with [`apply_coalescing`].
//!
//! The algorithm uses union-find for transitive coalescing: if A is coalesced
//! into B and B into C, all references to A resolve to C.
//!
//! Reference: LLVM `RegisterCoalescer.cpp` — simplified to pure virtual
//! register coalescing without sub-register handling.

use crate::liveness::LiveInterval;
use crate::machine_types::{InstId, MachFunction, MachOperand, RegClass, VReg};
use crate::phi_elim;
use std::collections::{HashMap, HashSet};

/// Result of copy coalescing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoalesceResult {
    /// Number of copy instructions that can be removed.
    pub copies_removed: u32,
    /// Number of interval merges that were performed.
    pub intervals_merged: u32,
    /// Copy instructions to remove from their containing blocks.
    pub removals: Vec<InstId>,
    /// VReg id rewrites: old id -> coalesced representative id.
    pub rewrites: HashMap<u32, u32>,
}

/// Scan the function for coalescible `PSEUDO_COPY` instructions.
///
/// For each copy `dst <- src`, if the current representative intervals of
/// `dst` and `src` do not overlap, `src` is coalesced into `dst`.
/// The function mutates the provided interval map but does not mutate
/// the `MachFunction`; instead it returns the copy removals and vreg rewrites
/// needed to apply the coalescing later.
pub fn coalesce_copies(
    func: &MachFunction,
    intervals: &mut HashMap<u32, LiveInterval>,
) -> CoalesceResult {
    let mut parent: HashMap<u32, u32> = HashMap::new();
    let mut result = CoalesceResult::default();

    // Walk blocks in program order.
    let block_indices: Vec<usize> = if func.block_order.is_empty() {
        (0..func.blocks.len()).collect()
    } else {
        func.block_order
            .iter()
            .map(|block_id| block_id.0 as usize)
            .collect()
    };

    for block_idx in block_indices {
        let block = &func.blocks[block_idx];

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.opcode != phi_elim::PSEUDO_COPY {
                continue;
            }

            let Some(dst_vreg) = inst.defs.first().and_then(MachOperand::as_vreg) else {
                continue;
            };
            let Some(src_vreg) = inst.uses.first().and_then(MachOperand::as_vreg) else {
                continue;
            };

            let dst_root = find_root(&mut parent, dst_vreg.id);
            let src_root = find_root(&mut parent, src_vreg.id);

            // Already coalesced through an earlier copy in the chain.
            if dst_root == src_root {
                result.removals.push(inst_id);
                result.copies_removed += 1;
                continue;
            }

            // Only coalesce within the same register class.
            if dst_vreg.class != src_vreg.class {
                continue;
            }

            let dst_interval = intervals
                .get(&dst_root)
                .cloned()
                .unwrap_or_else(|| LiveInterval::new(VReg {
                    id: dst_root,
                    class: dst_vreg.class,
                }));
            let src_interval = intervals
                .get(&src_root)
                .cloned()
                .unwrap_or_else(|| LiveInterval::new(VReg {
                    id: src_root,
                    class: src_vreg.class,
                }));

            if dst_interval.overlaps(&src_interval) {
                continue;
            }

            // Coalesce: merge src into dst in the union-find.
            parent.insert(src_root, dst_root);
            merge_interval(intervals, dst_root, dst_vreg.class, src_root);

            result.removals.push(inst_id);
            result.copies_removed += 1;
            result.intervals_merged += 1;
        }
    }

    // Build final rewrite map with path compression.
    let seen_ids: Vec<u32> = parent.keys().copied().collect();
    for id in seen_ids {
        let root = find_root(&mut parent, id);
        if root != id {
            result.rewrites.insert(id, root);
        }
    }

    result
}

/// Apply the removals and rewrites produced by [`coalesce_copies`].
///
/// Copy instructions are removed from block instruction lists, and all vreg
/// operands are rewritten according to `rewrites`. Only the vreg id changes;
/// the original operand class is preserved.
pub fn apply_coalescing(
    func: &mut MachFunction,
    removals: &[InstId],
    rewrites: &HashMap<u32, u32>,
) {
    let removal_set: HashSet<InstId> = removals.iter().copied().collect();

    for block in &mut func.blocks {
        block.insts.retain(|inst_id| !removal_set.contains(inst_id));
    }

    if rewrites.is_empty() {
        return;
    }

    for inst in &mut func.insts {
        rewrite_operands(&mut inst.defs, rewrites);
        rewrite_operands(&mut inst.uses, rewrites);
    }
}

// --- Union-find helpers ---

fn find_root(parent: &mut HashMap<u32, u32>, id: u32) -> u32 {
    let mut current = id;
    loop {
        let next = *parent.entry(current).or_insert(current);
        if next == current {
            break;
        }
        current = next;
    }

    // Path compression.
    let root = current;
    let mut current = id;
    loop {
        let next = *parent.entry(current).or_insert(current);
        if next == current {
            break;
        }
        parent.insert(current, root);
        current = next;
    }

    root
}

// --- Interval merging ---

fn merge_interval(
    intervals: &mut HashMap<u32, LiveInterval>,
    dst_id: u32,
    dst_class: RegClass,
    src_id: u32,
) {
    if dst_id == src_id {
        return;
    }

    let src_interval = intervals.remove(&src_id);

    match (intervals.get_mut(&dst_id), src_interval) {
        (Some(dst_interval), Some(src_interval)) => {
            merge_interval_contents(dst_interval, src_interval, dst_id, dst_class);
        }
        (None, Some(mut src_interval)) => {
            src_interval.vreg = VReg {
                id: dst_id,
                class: dst_class,
            };
            intervals.insert(dst_id, src_interval);
        }
        (Some(dst_interval), None) => {
            dst_interval.vreg = VReg {
                id: dst_id,
                class: dst_class,
            };
        }
        (None, None) => {}
    }
}

fn merge_interval_contents(
    dst_interval: &mut LiveInterval,
    src_interval: LiveInterval,
    dst_id: u32,
    dst_class: RegClass,
) {
    dst_interval.vreg = VReg {
        id: dst_id,
        class: dst_class,
    };

    for range in src_interval.ranges {
        dst_interval.add_range(range.start, range.end);
    }

    dst_interval.use_positions.extend(src_interval.use_positions);
    dst_interval.use_positions.sort_unstable();
    dst_interval.use_positions.dedup();

    dst_interval.def_positions.extend(src_interval.def_positions);
    dst_interval.def_positions.sort_unstable();
    dst_interval.def_positions.dedup();

    dst_interval.spill_weight += src_interval.spill_weight;
    dst_interval.is_fixed |= src_interval.is_fixed;
}

// --- Operand rewriting ---

fn rewrite_operands(operands: &mut [MachOperand], rewrites: &HashMap<u32, u32>) {
    for operand in operands {
        if let MachOperand::VReg(vreg) = operand {
            vreg.id = resolve_rewrite(vreg.id, rewrites);
        }
    }
}

fn resolve_rewrite(mut id: u32, rewrites: &HashMap<u32, u32>) -> u32 {
    let mut steps = 0usize;
    while let Some(&next) = rewrites.get(&id) {
        if next == id || steps >= rewrites.len() {
            break;
        }
        id = next;
        steps += 1;
    }
    id
}

// ---------------------------------------------------------------------------
// Coalescing mode and stateful coalescer
// ---------------------------------------------------------------------------

/// Coalescing aggressiveness mode.
///
/// Controls how eagerly the coalescer merges live intervals:
/// - **Aggressive:** merges whenever intervals do not interfere (the default
///   and the behavior of [`coalesce_copies`]).
/// - **Conservative:** additionally rejects merges that would increase
///   register pressure by creating a longer combined interval whose total
///   extent exceeds the sum of the individual extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalesceMode {
    /// Coalesce whenever intervals do not overlap.
    Aggressive,
    /// Coalesce only when the merged interval's extent (max end - min start)
    /// does not exceed the sum of the individual extents — a heuristic to
    /// avoid creating live ranges that span wide program regions and increase
    /// register pressure. Adjacent intervals are always accepted; intervals
    /// with a gap between them are rejected proportionally to the gap size.
    Conservative,
}

impl Default for CoalesceMode {
    fn default() -> Self {
        CoalesceMode::Aggressive
    }
}

/// Summary statistics for a coalescing pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoalesceStats {
    /// Number of copy instructions that were eliminated.
    pub copies_eliminated: u32,
    /// Number of copy instructions that could not be eliminated.
    pub copies_remaining: u32,
    /// Number of live-interval merges performed.
    pub intervals_merged: u32,
}

impl CoalesceResult {
    /// Derive summary statistics given the total number of `PSEUDO_COPY`
    /// instructions in the function.
    pub fn stats(&self, total_copies: u32) -> CoalesceStats {
        CoalesceStats {
            copies_eliminated: self.copies_removed,
            copies_remaining: total_copies.saturating_sub(self.copies_removed),
            intervals_merged: self.intervals_merged,
        }
    }
}

/// Compute the total span (sum of range lengths) of a [`LiveInterval`].
#[cfg(test)]
fn interval_span(interval: &LiveInterval) -> u32 {
    interval.ranges.iter().map(|r| r.end - r.start).sum()
}

/// Compute the extent (max_end - min_start) of a [`LiveInterval`].
///
/// Returns 0 for empty intervals. The extent captures how wide the live
/// range is in program order — a merged interval with the same total span
/// but a much larger extent occupies a register across a wider code region,
/// increasing pressure.
fn interval_extent(interval: &LiveInterval) -> u32 {
    match (interval.ranges.first(), interval.ranges.last()) {
        (Some(first), Some(last)) => last.end.saturating_sub(first.start),
        _ => 0,
    }
}

/// Stateful copy coalescer with configurable aggressiveness.
///
/// Wraps the same union-find + overlap-check algorithm as [`coalesce_copies`],
/// but exposes individual operations (`can_coalesce`, `merge_intervals`,
/// `update_uses`) for callers that need finer-grained control.
///
/// # Example
///
/// ```rust,ignore
/// let mut coalescer = CopyCoalescer::new(CoalesceMode::Aggressive);
/// let result = coalescer.coalesce(&func, &mut intervals);
/// apply_coalescing(&mut func, &result.removals, &result.rewrites);
/// ```
pub struct CopyCoalescer {
    /// Aggressiveness mode.
    mode: CoalesceMode,
    /// Union-find parent map for transitive coalescing.
    parent: HashMap<u32, u32>,
}

impl CopyCoalescer {
    /// Create a new coalescer with the given mode.
    pub fn new(mode: CoalesceMode) -> Self {
        Self {
            mode,
            parent: HashMap::new(),
        }
    }

    /// Reset union-find state so the coalescer can be reused on a different
    /// function.
    pub fn reset(&mut self) {
        self.parent.clear();
    }

    /// Return the current coalescing mode.
    pub fn mode(&self) -> CoalesceMode {
        self.mode
    }

    /// Check whether two intervals can be coalesced.
    ///
    /// In **Aggressive** mode, intervals are coalescible when they do not
    /// overlap.  In **Conservative** mode, the merged interval's *extent*
    /// (max end - min start) must also not exceed the sum of the individual
    /// extents. This rejects merges that would create a single live range
    /// spanning a much wider program region than the two originals combined,
    /// which would increase register pressure.
    pub fn can_coalesce(
        &self,
        src_interval: &LiveInterval,
        dst_interval: &LiveInterval,
    ) -> bool {
        if src_interval.overlaps(dst_interval) {
            return false;
        }
        match self.mode {
            CoalesceMode::Aggressive => true,
            CoalesceMode::Conservative => {
                let merged = Self::merge_intervals(src_interval, dst_interval);
                let merged_extent = interval_extent(&merged);
                let sum_extent =
                    interval_extent(src_interval) + interval_extent(dst_interval);
                merged_extent <= sum_extent
            }
        }
    }

    /// Merge two non-overlapping intervals into a single combined interval.
    ///
    /// The result uses the destination interval's VReg identity. Spill
    /// weights are summed and use/def positions are combined.
    pub fn merge_intervals(
        src: &LiveInterval,
        dst: &LiveInterval,
    ) -> LiveInterval {
        let mut merged = dst.clone();
        for range in &src.ranges {
            merged.add_range(range.start, range.end);
        }
        merged.use_positions.extend(&src.use_positions);
        merged.use_positions.sort_unstable();
        merged.use_positions.dedup();
        merged.def_positions.extend(&src.def_positions);
        merged.def_positions.sort_unstable();
        merged.def_positions.dedup();
        merged.spill_weight += src.spill_weight;
        merged.is_fixed |= src.is_fixed;
        merged
    }

    /// Rewrite all occurrences of `old_vreg` to `new_vreg` in the function.
    ///
    /// This updates both defs and uses across all instructions.
    pub fn update_uses(func: &mut MachFunction, old_vreg: u32, new_vreg: u32) {
        let rewrites = HashMap::from([(old_vreg, new_vreg)]);
        for inst in &mut func.insts {
            rewrite_operands(&mut inst.defs, &rewrites);
            rewrite_operands(&mut inst.uses, &rewrites);
        }
    }

    /// Run the coalescing pass over the entire function.
    ///
    /// In **Aggressive** mode this delegates directly to [`coalesce_copies`].
    /// In **Conservative** mode it applies an additional span check before
    /// each merge.
    pub fn coalesce(
        &mut self,
        func: &MachFunction,
        intervals: &mut HashMap<u32, LiveInterval>,
    ) -> CoalesceResult {
        self.parent.clear();

        if self.mode == CoalesceMode::Aggressive {
            return coalesce_copies(func, intervals);
        }

        // Conservative mode: inline the algorithm with the extra check.
        let mut result = CoalesceResult::default();

        let block_indices: Vec<usize> = if func.block_order.is_empty() {
            (0..func.blocks.len()).collect()
        } else {
            func.block_order
                .iter()
                .map(|block_id| block_id.0 as usize)
                .collect()
        };

        for block_idx in block_indices {
            let block = &func.blocks[block_idx];

            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.0 as usize];
                if inst.opcode != phi_elim::PSEUDO_COPY {
                    continue;
                }

                let Some(dst_vreg) = inst.defs.first().and_then(MachOperand::as_vreg) else {
                    continue;
                };
                let Some(src_vreg) = inst.uses.first().and_then(MachOperand::as_vreg) else {
                    continue;
                };

                let dst_root = find_root(&mut self.parent, dst_vreg.id);
                let src_root = find_root(&mut self.parent, src_vreg.id);

                if dst_root == src_root {
                    result.removals.push(inst_id);
                    result.copies_removed += 1;
                    continue;
                }

                if dst_vreg.class != src_vreg.class {
                    continue;
                }

                let dst_interval = intervals
                    .get(&dst_root)
                    .cloned()
                    .unwrap_or_else(|| LiveInterval::new(VReg {
                        id: dst_root,
                        class: dst_vreg.class,
                    }));
                let src_interval = intervals
                    .get(&src_root)
                    .cloned()
                    .unwrap_or_else(|| LiveInterval::new(VReg {
                        id: src_root,
                        class: src_vreg.class,
                    }));

                if !self.can_coalesce(&src_interval, &dst_interval) {
                    continue;
                }

                self.parent.insert(src_root, dst_root);
                merge_interval(intervals, dst_root, dst_vreg.class, src_root);

                result.removals.push(inst_id);
                result.copies_removed += 1;
                result.intervals_merged += 1;
            }
        }

        // Build final rewrite map with path compression.
        let seen_ids: Vec<u32> = self.parent.keys().copied().collect();
        for id in seen_ids {
            let root = find_root(&mut self.parent, id);
            if root != id {
                result.rewrites.insert(id, root);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_types::{
        BlockId, InstFlags, MachBlock, MachFunction, MachInst, MachOperand, RegClass, VReg,
    };
    use std::collections::HashMap;

    fn vreg(id: u32) -> VReg {
        VReg {
            id,
            class: RegClass::Gpr64,
        }
    }

    fn generic_inst(opcode: u16, defs: Vec<VReg>, uses: Vec<VReg>) -> MachInst {
        MachInst {
            opcode,
            defs: defs.into_iter().map(MachOperand::VReg).collect(),
            uses: uses.into_iter().map(MachOperand::VReg).collect(),
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        }
    }

    fn copy_inst(dst: VReg, src: VReg) -> MachInst {
        generic_inst(crate::phi_elim::PSEUDO_COPY, vec![dst], vec![src])
    }

    fn interval(id: u32, ranges: &[(u32, u32)]) -> LiveInterval {
        let mut interval = LiveInterval::new(vreg(id));
        for &(start, end) in ranges {
            interval.add_range(start, end);
        }
        interval
    }

    fn interval_ranges(interval: &LiveInterval) -> Vec<(u32, u32)> {
        interval
            .ranges
            .iter()
            .map(|range| (range.start, range.end))
            .collect()
    }

    fn make_function(blocks_insts: Vec<Vec<MachInst>>) -> MachFunction {
        let mut insts = Vec::new();
        let mut blocks = Vec::new();
        let mut block_order = Vec::new();

        for block_insts in blocks_insts {
            let block_id = BlockId(blocks.len() as u32);
            let mut inst_ids = Vec::new();

            for inst in block_insts {
                let inst_id = InstId(insts.len() as u32);
                insts.push(inst);
                inst_ids.push(inst_id);
            }

            blocks.push(MachBlock {
                insts: inst_ids,
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            });
            block_order.push(block_id);
        }

        MachFunction {
            name: "test".into(),
            insts,
            blocks,
            block_order,
            entry_block: BlockId(0),
            next_vreg: 32,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    #[test]
    fn test_coalesce_non_overlapping() {
        let func = make_function(vec![vec![
            generic_inst(1, vec![vreg(7)], vec![]),
            copy_inst(vreg(1), vreg(0)),
            generic_inst(2, vec![], vec![vreg(1)]),
        ]]);
        let copy_id = func.blocks[0].insts[1];

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.intervals_merged, 1);
        assert_eq!(result.removals, vec![copy_id]);
        assert_eq!(result.rewrites.get(&0), Some(&1));
        assert_eq!(intervals.len(), 1);
        assert_eq!(interval_ranges(intervals.get(&1).unwrap()), vec![(0, 2)]);
    }

    #[test]
    fn test_coalesce_skips_overlapping() {
        let func = make_function(vec![vec![copy_inst(vreg(1), vreg(0))]]);

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 2)])),
            (1, interval(1, &[(1, 3)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 0);
        assert_eq!(result.intervals_merged, 0);
        assert!(result.removals.is_empty());
        assert!(result.rewrites.is_empty());
        assert_eq!(intervals.len(), 2);
    }

    #[test]
    fn test_coalesce_transitive_chain() {
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(2), vreg(1)),
        ]]);
        let copy1 = func.blocks[0].insts[0];
        let copy2 = func.blocks[0].insts[1];

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 2);
        assert_eq!(result.intervals_merged, 2);
        assert_eq!(result.removals, vec![copy1, copy2]);
        assert_eq!(intervals.len(), 1);
    }

    #[test]
    fn test_coalesce_duplicate_copies() {
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(1), vreg(0)),
        ]]);

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 2);
        assert_eq!(result.intervals_merged, 1);
    }

    #[test]
    fn test_apply_coalescing() {
        let mut func = make_function(vec![
            vec![generic_inst(1, vec![vreg(0)], vec![]), copy_inst(vreg(1), vreg(0))],
            vec![
                copy_inst(vreg(2), vreg(1)),
                generic_inst(2, vec![vreg(3)], vec![vreg(0), vreg(1), vreg(2)]),
            ],
        ]);

        let def_id = func.blocks[0].insts[0];
        let copy1_id = func.blocks[0].insts[1];
        let copy2_id = func.blocks[1].insts[0];
        let user_id = func.blocks[1].insts[1];

        let rewrites = HashMap::from([(0, 1), (1, 2)]);
        apply_coalescing(&mut func, &[copy1_id, copy2_id], &rewrites);

        assert_eq!(func.blocks[0].insts, vec![def_id]);
        assert_eq!(func.blocks[1].insts, vec![user_id]);

        let def_vreg = func.insts[def_id.0 as usize].defs[0].as_vreg().unwrap();
        assert_eq!(def_vreg.id, 2);

        let use_ids: Vec<u32> = func.insts[user_id.0 as usize]
            .uses
            .iter()
            .map(|operand| operand.as_vreg().unwrap().id)
            .collect();
        assert_eq!(use_ids, vec![2, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests (issue #139)
    // -----------------------------------------------------------------------

    #[test]
    fn test_coalesce_rejects_cross_class_copy() {
        let dst = vreg(1);
        let src = VReg {
            id: 0,
            class: RegClass::Fpr64,
        };
        let func = make_function(vec![vec![copy_inst(dst, src)]]);

        let mut src_interval = LiveInterval::new(src);
        src_interval.add_range(0, 1);
        let mut intervals = HashMap::from([(0, src_interval), (1, interval(1, &[(1, 2)]))]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 0);
        assert_eq!(result.intervals_merged, 0);
        assert!(result.removals.is_empty());
        assert!(result.rewrites.is_empty());
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals.get(&0).unwrap().vreg.class, RegClass::Fpr64);
        assert_eq!(interval_ranges(intervals.get(&1).unwrap()), vec![(1, 2)]);
    }

    #[test]
    fn test_coalesce_empty_function_no_instructions() {
        let func = make_function(vec![vec![]]);
        let mut intervals = HashMap::from([(0, interval(0, &[(0, 1)]))]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result, CoalesceResult::default());
        assert_eq!(intervals.len(), 1);
        assert_eq!(interval_ranges(intervals.get(&0).unwrap()), vec![(0, 1)]);
    }

    #[test]
    fn test_coalesce_across_multiple_blocks() {
        let func = make_function(vec![
            vec![copy_inst(vreg(1), vreg(0))],
            vec![copy_inst(vreg(2), vreg(1))],
        ]);
        let copy1 = func.blocks[0].insts[0];
        let copy2 = func.blocks[1].insts[0];

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 2);
        assert_eq!(result.intervals_merged, 2);
        assert_eq!(result.removals, vec![copy1, copy2]);
        assert_eq!(result.rewrites, HashMap::from([(0, 2), (1, 2)]));
        assert_eq!(intervals.len(), 1);
        assert_eq!(interval_ranges(intervals.get(&2).unwrap()), vec![(0, 3)]);
    }

    #[test]
    fn test_coalesce_with_missing_intervals_for_src_and_dst() {
        let func = make_function(vec![vec![copy_inst(vreg(1), vreg(0))]]);
        let copy_id = func.blocks[0].insts[0];
        let mut intervals = HashMap::new();

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.intervals_merged, 1);
        assert_eq!(result.removals, vec![copy_id]);
        assert_eq!(result.rewrites.get(&0), Some(&1));
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_apply_coalescing_without_removals() {
        let mut func = make_function(vec![vec![
            generic_inst(1, vec![vreg(0)], vec![]),
            generic_inst(2, vec![vreg(3)], vec![vreg(0), vreg(1)]),
        ]]);
        let def_id = func.blocks[0].insts[0];
        let user_id = func.blocks[0].insts[1];

        let rewrites = HashMap::from([(0, 1), (1, 2)]);
        apply_coalescing(&mut func, &[], &rewrites);

        assert_eq!(func.blocks[0].insts, vec![def_id, user_id]);

        let def_vreg = func.insts[def_id.0 as usize].defs[0].as_vreg().unwrap();
        assert_eq!(def_vreg.id, 2);

        let user_def = func.insts[user_id.0 as usize].defs[0].as_vreg().unwrap();
        assert_eq!(user_def.id, 3);

        let use_ids: Vec<u32> = func.insts[user_id.0 as usize]
            .uses
            .iter()
            .map(|operand| operand.as_vreg().unwrap().id)
            .collect();
        assert_eq!(use_ids, vec![2, 2]);
    }

    #[test]
    fn test_apply_coalescing_without_rewrites() {
        let mut func = make_function(vec![vec![
            generic_inst(1, vec![vreg(0)], vec![]),
            copy_inst(vreg(1), vreg(0)),
            generic_inst(2, vec![vreg(2)], vec![vreg(1)]),
        ]]);
        let def_id = func.blocks[0].insts[0];
        let copy_id = func.blocks[0].insts[1];
        let user_id = func.blocks[0].insts[2];

        apply_coalescing(&mut func, &[copy_id], &HashMap::new());

        assert_eq!(func.blocks[0].insts, vec![def_id, user_id]);

        let def_vreg = func.insts[def_id.0 as usize].defs[0].as_vreg().unwrap();
        assert_eq!(def_vreg.id, 0);

        let use_ids: Vec<u32> = func.insts[user_id.0 as usize]
            .uses
            .iter()
            .map(|operand| operand.as_vreg().unwrap().id)
            .collect();
        assert_eq!(use_ids, vec![1]);
    }

    #[test]
    fn test_coalesce_long_transitive_chain() {
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(2), vreg(1)),
            copy_inst(vreg(3), vreg(2)),
            copy_inst(vreg(4), vreg(3)),
        ]]);
        let copy1 = func.blocks[0].insts[0];
        let copy2 = func.blocks[0].insts[1];
        let copy3 = func.blocks[0].insts[2];
        let copy4 = func.blocks[0].insts[3];

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
            (3, interval(3, &[(3, 4)])),
            (4, interval(4, &[(4, 5)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 4);
        assert_eq!(result.intervals_merged, 4);
        assert_eq!(result.removals, vec![copy1, copy2, copy3, copy4]);
        assert_eq!(
            result.rewrites,
            HashMap::from([(0, 4), (1, 4), (2, 4), (3, 4)])
        );
        assert_eq!(intervals.len(), 1);
        assert_eq!(interval_ranges(intervals.get(&4).unwrap()), vec![(0, 5)]);
    }

    #[test]
    fn test_coalesce_skips_when_all_intervals_overlap() {
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(2), vreg(1)),
        ]]);

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 4)])),
            (1, interval(1, &[(1, 5)])),
            (2, interval(2, &[(2, 6)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 0);
        assert_eq!(result.intervals_merged, 0);
        assert!(result.removals.is_empty());
        assert!(result.rewrites.is_empty());
        assert_eq!(intervals.len(), 3);
        assert_eq!(interval_ranges(intervals.get(&0).unwrap()), vec![(0, 4)]);
        assert_eq!(interval_ranges(intervals.get(&1).unwrap()), vec![(1, 5)]);
        assert_eq!(interval_ranges(intervals.get(&2).unwrap()), vec![(2, 6)]);
    }

    #[test]
    fn test_coalesce_uses_block_indices_when_block_order_is_empty() {
        let mut func = make_function(vec![
            vec![copy_inst(vreg(1), vreg(0))],
            vec![copy_inst(vreg(2), vreg(1))],
        ]);
        let copy1 = func.blocks[0].insts[0];
        let copy2 = func.blocks[1].insts[0];
        func.block_order.clear();

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 2);
        assert_eq!(result.intervals_merged, 2);
        assert_eq!(result.removals, vec![copy1, copy2]);
        assert_eq!(result.rewrites, HashMap::from([(0, 2), (1, 2)]));
        assert_eq!(intervals.len(), 1);
        assert_eq!(interval_ranges(intervals.get(&2).unwrap()), vec![(0, 3)]);
    }

    #[test]
    fn test_apply_coalescing_detects_rewrite_cycles() {
        let rewrites = HashMap::from([(0, 1), (1, 2), (2, 0)]);
        assert_eq!(resolve_rewrite(0, &rewrites), 0);
        assert_eq!(resolve_rewrite(1, &rewrites), 1);
        assert_eq!(resolve_rewrite(2, &rewrites), 2);

        let mut func = make_function(vec![vec![generic_inst(1, vec![vreg(0)], vec![vreg(1), vreg(2)])]]);
        let inst_id = func.blocks[0].insts[0];

        apply_coalescing(&mut func, &[], &rewrites);

        let def_vreg = func.insts[inst_id.0 as usize].defs[0].as_vreg().unwrap();
        assert_eq!(def_vreg.id, 0);

        let use_ids: Vec<u32> = func.insts[inst_id.0 as usize]
            .uses
            .iter()
            .map(|operand| operand.as_vreg().unwrap().id)
            .collect();
        assert_eq!(use_ids, vec![1, 2]);
    }

    #[test]
    fn test_coalesce_merge_preserves_spill_weight() {
        let func = make_function(vec![vec![copy_inst(vreg(1), vreg(0))]]);

        let mut src_interval = interval(0, &[(0, 1)]);
        src_interval.spill_weight = 1.5;

        let mut dst_interval = interval(1, &[(1, 2)]);
        dst_interval.spill_weight = 2.25;

        let mut intervals = HashMap::from([(0, src_interval), (1, dst_interval)]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.intervals_merged, 1);
        assert_eq!(result.rewrites.get(&0), Some(&1));
        assert_eq!(intervals.len(), 1);
        assert_eq!(interval_ranges(intervals.get(&1).unwrap()), vec![(0, 2)]);
        assert_eq!(intervals.get(&1).unwrap().spill_weight, 3.75);
    }

    #[test]
    fn test_coalesce_multiple_copies_across_multiple_blocks() {
        let func = make_function(vec![
            vec![copy_inst(vreg(1), vreg(0)), copy_inst(vreg(3), vreg(2))],
            vec![copy_inst(vreg(5), vreg(4))],
            vec![copy_inst(vreg(7), vreg(6))],
        ]);
        let copy1 = func.blocks[0].insts[0];
        let copy2 = func.blocks[0].insts[1];
        let copy3 = func.blocks[1].insts[0];
        let copy4 = func.blocks[2].insts[0];

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
            (3, interval(3, &[(3, 4)])),
            (4, interval(4, &[(4, 5)])),
            (5, interval(5, &[(5, 6)])),
            (6, interval(6, &[(6, 7)])),
            (7, interval(7, &[(7, 8)])),
        ]);

        let result = coalesce_copies(&func, &mut intervals);

        assert_eq!(result.copies_removed, 4);
        assert_eq!(result.intervals_merged, 4);
        assert_eq!(result.removals, vec![copy1, copy2, copy3, copy4]);
        assert_eq!(
            result.rewrites,
            HashMap::from([(0, 1), (2, 3), (4, 5), (6, 7)])
        );
        assert_eq!(intervals.len(), 4);
        assert_eq!(interval_ranges(intervals.get(&1).unwrap()), vec![(0, 2)]);
        assert_eq!(interval_ranges(intervals.get(&3).unwrap()), vec![(2, 4)]);
        assert_eq!(interval_ranges(intervals.get(&5).unwrap()), vec![(4, 6)]);
        assert_eq!(interval_ranges(intervals.get(&7).unwrap()), vec![(6, 8)]);
    }

    // -----------------------------------------------------------------------
    // CopyCoalescer, CoalesceMode, and CoalesceStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coalescer_aggressive_mode_same_as_functional() {
        // Aggressive mode should produce the same result as coalesce_copies.
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(2), vreg(1)),
        ]]);

        let mut intervals_a = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
        ]);
        let mut intervals_b = intervals_a.clone();

        let result_fn = coalesce_copies(&func, &mut intervals_a);

        let mut coalescer = CopyCoalescer::new(CoalesceMode::Aggressive);
        let result_struct = coalescer.coalesce(&func, &mut intervals_b);

        assert_eq!(result_fn.copies_removed, result_struct.copies_removed);
        assert_eq!(result_fn.intervals_merged, result_struct.intervals_merged);
        assert_eq!(result_fn.removals, result_struct.removals);
        assert_eq!(result_fn.rewrites, result_struct.rewrites);
    }

    #[test]
    fn test_coalescer_conservative_rejects_wide_gap() {
        // Two intervals with a wide gap between them. Conservative mode uses
        // extent (max_end - min_start) to detect that merging would occupy a
        // register across a much wider program region:
        //
        // src: [0, 2)   -> extent = 2
        // dst: [10, 12) -> extent = 2
        // sum_extent = 4
        // merged: [0, 2) + [10, 12) -> extent = 12
        // 12 > 4 -> REJECT
        //
        // Aggressive mode allows this because the intervals don't overlap.

        let func = make_function(vec![vec![copy_inst(vreg(1), vreg(0))]]);
        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 2)])),
            (1, interval(1, &[(10, 12)])),
        ]);

        let mut coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let result = coalescer.coalesce(&func, &mut intervals);

        // Conservative rejects due to extent increase.
        assert_eq!(result.copies_removed, 0);
        assert_eq!(result.intervals_merged, 0);

        // But aggressive would accept:
        let mut intervals2 = HashMap::from([
            (0, interval(0, &[(0, 2)])),
            (1, interval(1, &[(10, 12)])),
        ]);
        let mut aggressive = CopyCoalescer::new(CoalesceMode::Aggressive);
        let result2 = aggressive.coalesce(&func, &mut intervals2);
        assert_eq!(result2.copies_removed, 1);
        assert_eq!(result2.intervals_merged, 1);
    }

    #[test]
    fn test_coalescer_conservative_allows_non_overlapping() {
        let func = make_function(vec![vec![copy_inst(vreg(1), vreg(0))]]);
        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
        ]);

        let mut coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let result = coalescer.coalesce(&func, &mut intervals);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.intervals_merged, 1);
    }

    #[test]
    fn test_coalescer_conservative_chain() {
        // Test that conservative mode handles transitive chains.
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(2), vreg(1)),
            copy_inst(vreg(3), vreg(2)),
        ]]);

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(2, 3)])),
            (3, interval(3, &[(3, 4)])),
        ]);

        let mut coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let result = coalescer.coalesce(&func, &mut intervals);

        assert_eq!(result.copies_removed, 3);
        assert_eq!(result.intervals_merged, 3);
        assert_eq!(intervals.len(), 1);
    }

    #[test]
    fn test_can_coalesce_non_overlapping() {
        let coalescer = CopyCoalescer::new(CoalesceMode::Aggressive);
        let src = interval(0, &[(0, 2)]);
        let dst = interval(1, &[(3, 5)]);

        assert!(coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_can_coalesce_overlapping_rejected() {
        let coalescer = CopyCoalescer::new(CoalesceMode::Aggressive);
        let src = interval(0, &[(0, 4)]);
        let dst = interval(1, &[(2, 6)]);

        assert!(!coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_can_coalesce_conservative_mode() {
        let conservative = CopyCoalescer::new(CoalesceMode::Conservative);
        let aggressive = CopyCoalescer::new(CoalesceMode::Aggressive);

        // Non-overlapping, adjacent — both modes accept (extent equals sum).
        // src: [0, 2) extent=2, dst: [2, 4) extent=2, sum=4, merged=[0,4) extent=4
        let src_adj = interval(0, &[(0, 2)]);
        let dst_adj = interval(1, &[(2, 4)]);
        assert!(conservative.can_coalesce(&src_adj, &dst_adj));
        assert!(aggressive.can_coalesce(&src_adj, &dst_adj));

        // Non-overlapping, wide gap — conservative rejects, aggressive accepts.
        // src: [0, 2) extent=2, dst: [10, 12) extent=2, sum=4
        // merged: [0,2)+[10,12) extent=12 > 4 -> REJECT
        let src_gap = interval(0, &[(0, 2)]);
        let dst_gap = interval(1, &[(10, 12)]);
        assert!(!conservative.can_coalesce(&src_gap, &dst_gap));
        assert!(aggressive.can_coalesce(&src_gap, &dst_gap));

        // Overlapping — both modes reject.
        let src_overlap = interval(0, &[(0, 4)]);
        let dst_overlap = interval(1, &[(2, 6)]);
        assert!(!conservative.can_coalesce(&src_overlap, &dst_overlap));
        assert!(!aggressive.can_coalesce(&src_overlap, &dst_overlap));
    }

    #[test]
    fn test_can_coalesce_empty_intervals() {
        let coalescer = CopyCoalescer::new(CoalesceMode::Aggressive);
        let empty_a = LiveInterval::new(vreg(0));
        let empty_b = LiveInterval::new(vreg(1));

        assert!(coalescer.can_coalesce(&empty_a, &empty_b));
    }

    #[test]
    fn test_merge_intervals_public_api() {
        let src = interval(0, &[(0, 2)]);
        let dst = interval(1, &[(3, 5)]);

        let merged = CopyCoalescer::merge_intervals(&src, &dst);

        // Merged should use dst's VReg identity.
        assert_eq!(merged.vreg.id, 1);
        // Should contain both ranges.
        let ranges: Vec<(u32, u32)> = merged
            .ranges
            .iter()
            .map(|r| (r.start, r.end))
            .collect();
        assert_eq!(ranges, vec![(0, 2), (3, 5)]);
    }

    #[test]
    fn test_merge_intervals_adjacent_ranges() {
        let src = interval(0, &[(0, 3)]);
        let dst = interval(1, &[(3, 6)]);

        let merged = CopyCoalescer::merge_intervals(&src, &dst);

        // Adjacent ranges should be merged into one.
        let ranges: Vec<(u32, u32)> = merged
            .ranges
            .iter()
            .map(|r| (r.start, r.end))
            .collect();
        assert_eq!(ranges, vec![(0, 6)]);
    }

    #[test]
    fn test_merge_intervals_preserves_spill_weight() {
        let mut src = interval(0, &[(0, 2)]);
        src.spill_weight = 1.5;
        let mut dst = interval(1, &[(3, 5)]);
        dst.spill_weight = 2.5;

        let merged = CopyCoalescer::merge_intervals(&src, &dst);

        assert_eq!(merged.spill_weight, 4.0);
    }

    #[test]
    fn test_merge_intervals_preserves_use_def_positions() {
        let mut src = interval(0, &[(0, 2)]);
        src.use_positions = vec![0, 1];
        src.def_positions = vec![0];
        let mut dst = interval(1, &[(3, 5)]);
        dst.use_positions = vec![3, 4];
        dst.def_positions = vec![3];

        let merged = CopyCoalescer::merge_intervals(&src, &dst);

        assert_eq!(merged.use_positions, vec![0, 1, 3, 4]);
        assert_eq!(merged.def_positions, vec![0, 3]);
    }

    #[test]
    fn test_merge_intervals_is_fixed_propagates() {
        let mut src = interval(0, &[(0, 2)]);
        src.is_fixed = true;
        let dst = interval(1, &[(3, 5)]);

        let merged = CopyCoalescer::merge_intervals(&src, &dst);
        assert!(merged.is_fixed);
    }

    #[test]
    fn test_update_uses_public_api() {
        let mut func = make_function(vec![vec![
            generic_inst(1, vec![vreg(0)], vec![]),
            generic_inst(2, vec![vreg(1)], vec![vreg(0)]),
            generic_inst(3, vec![], vec![vreg(0), vreg(1)]),
        ]]);

        CopyCoalescer::update_uses(&mut func, 0, 5);

        // All occurrences of vreg 0 should now be vreg 5.
        let def0 = func.insts[0].defs[0].as_vreg().unwrap();
        assert_eq!(def0.id, 5);

        let use1 = func.insts[1].uses[0].as_vreg().unwrap();
        assert_eq!(use1.id, 5);

        let use2_0 = func.insts[2].uses[0].as_vreg().unwrap();
        assert_eq!(use2_0.id, 5);

        // vreg 1 should be unchanged.
        let def1 = func.insts[1].defs[0].as_vreg().unwrap();
        assert_eq!(def1.id, 1);
        let use2_1 = func.insts[2].uses[1].as_vreg().unwrap();
        assert_eq!(use2_1.id, 1);
    }

    #[test]
    fn test_coalesce_stats_tracking() {
        let result = CoalesceResult {
            copies_removed: 3,
            intervals_merged: 2,
            removals: Vec::new(),
            rewrites: HashMap::new(),
        };

        let stats = result.stats(5);

        assert_eq!(stats.copies_eliminated, 3);
        assert_eq!(stats.copies_remaining, 2);
        assert_eq!(stats.intervals_merged, 2);
    }

    #[test]
    fn test_coalesce_stats_zero_total() {
        let result = CoalesceResult::default();
        let stats = result.stats(0);

        assert_eq!(stats.copies_eliminated, 0);
        assert_eq!(stats.copies_remaining, 0);
        assert_eq!(stats.intervals_merged, 0);
    }

    #[test]
    fn test_coalesce_stats_all_eliminated() {
        let result = CoalesceResult {
            copies_removed: 10,
            intervals_merged: 8,
            removals: Vec::new(),
            rewrites: HashMap::new(),
        };
        let stats = result.stats(10);

        assert_eq!(stats.copies_eliminated, 10);
        assert_eq!(stats.copies_remaining, 0);
        assert_eq!(stats.intervals_merged, 8);
    }

    #[test]
    fn test_coalescer_reset() {
        let mut coalescer = CopyCoalescer::new(CoalesceMode::Aggressive);

        // Run on one function.
        let func = make_function(vec![vec![copy_inst(vreg(1), vreg(0))]]);
        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
        ]);
        let result = coalescer.coalesce(&func, &mut intervals);
        assert_eq!(result.copies_removed, 1);

        // Reset and run again on a fresh function.
        coalescer.reset();

        let func2 = make_function(vec![vec![copy_inst(vreg(3), vreg(2))]]);
        let mut intervals2 = HashMap::from([
            (2, interval(2, &[(0, 1)])),
            (3, interval(3, &[(1, 2)])),
        ]);
        let result2 = coalescer.coalesce(&func2, &mut intervals2);
        assert_eq!(result2.copies_removed, 1);
        assert_eq!(result2.intervals_merged, 1);
    }

    #[test]
    fn test_coalescer_mode_accessor() {
        let aggressive = CopyCoalescer::new(CoalesceMode::Aggressive);
        assert_eq!(aggressive.mode(), CoalesceMode::Aggressive);

        let conservative = CopyCoalescer::new(CoalesceMode::Conservative);
        assert_eq!(conservative.mode(), CoalesceMode::Conservative);
    }

    #[test]
    fn test_coalesce_mode_default_is_aggressive() {
        assert_eq!(CoalesceMode::default(), CoalesceMode::Aggressive);
    }

    #[test]
    fn test_interval_span_helper() {
        let i = interval(0, &[(0, 3), (5, 8)]);
        assert_eq!(interval_span(&i), 6); // 3 + 3
    }

    #[test]
    fn test_interval_span_empty() {
        let i = LiveInterval::new(vreg(0));
        assert_eq!(interval_span(&i), 0);
    }

    // -----------------------------------------------------------------------
    // Extent-based conservative heuristic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interval_extent_single_range() {
        let i = interval(0, &[(3, 7)]);
        assert_eq!(interval_extent(&i), 4);
    }

    #[test]
    fn test_interval_extent_multiple_ranges() {
        let i = interval(0, &[(0, 3), (10, 15)]);
        assert_eq!(interval_extent(&i), 15); // 15 - 0
    }

    #[test]
    fn test_interval_extent_empty() {
        let i = LiveInterval::new(vreg(0));
        assert_eq!(interval_extent(&i), 0);
    }

    #[test]
    fn test_conservative_rejects_distant_intervals() {
        // Intervals far apart: conservative should reject because merged
        // extent far exceeds sum of individual extents.
        // src: [0, 1) extent=1, dst: [100, 101) extent=1
        // sum_extent = 2, merged_extent = 101 -> REJECT
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = interval(0, &[(0, 1)]);
        let dst = interval(1, &[(100, 101)]);
        assert!(!coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_conservative_accepts_adjacent_intervals() {
        // Adjacent intervals: merged extent == sum of extents.
        // src: [0, 5) extent=5, dst: [5, 10) extent=5
        // sum_extent = 10, merged_extent = 10 -> ACCEPT
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = interval(0, &[(0, 5)]);
        let dst = interval(1, &[(5, 10)]);
        assert!(coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_conservative_rejects_small_gap() {
        // Even a small gap causes rejection: merged extent barely exceeds sum.
        // src: [0, 3) extent=3, dst: [4, 7) extent=3
        // sum_extent = 6, merged_extent = 7 -> 7 > 6 -> REJECT
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = interval(0, &[(0, 3)]);
        let dst = interval(1, &[(4, 7)]);
        assert!(!coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_conservative_multi_range_intervals() {
        // Multi-range src with large internal gap: extent already large.
        // src: [0, 2), [8, 10) -> extent = 10
        // dst: [10, 12) -> extent = 2
        // sum_extent = 12
        // merged: [0,2), [8,12) -> extent = 12
        // 12 <= 12 -> ACCEPT
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = interval(0, &[(0, 2), (8, 10)]);
        let dst = interval(1, &[(10, 12)]);
        assert!(coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_conservative_multi_range_rejects_when_far() {
        // src: [0, 2), [3, 5) -> extent = 5
        // dst: [50, 52) -> extent = 2
        // sum_extent = 7
        // merged: [0,2), [3,5), [50,52) -> extent = 52
        // 52 > 7 -> REJECT
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = interval(0, &[(0, 2), (3, 5)]);
        let dst = interval(1, &[(50, 52)]);
        assert!(!coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_conservative_coalesce_partial_chain() {
        // A chain where the first merge is accepted (adjacent) but the second
        // is rejected (wide gap) by conservative mode.
        //
        // copy v1 <- v0 (intervals adjacent, accepted)
        // copy v2 <- v1 (v1 now has extent [0,2), v2 at [20,21) -> rejected)
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(2), vreg(1)),
        ]]);

        let mut intervals = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(20, 21)])),
        ]);

        let mut coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let result = coalescer.coalesce(&func, &mut intervals);

        // First copy: v0 [0,1) and v1 [1,2) are adjacent -> accepted.
        // After merge: v1 has extent [0,2).
        // Second copy: merged v1 [0,2) and v2 [20,21):
        //   sum_extent = 2 + 1 = 3, merged_extent = 21 -> REJECT
        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.intervals_merged, 1);

        // Compare with aggressive which accepts both:
        let mut intervals2 = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(20, 21)])),
        ]);
        let mut aggressive = CopyCoalescer::new(CoalesceMode::Aggressive);
        let result2 = aggressive.coalesce(&func, &mut intervals2);
        assert_eq!(result2.copies_removed, 2);
        assert_eq!(result2.intervals_merged, 2);
    }

    #[test]
    fn test_conservative_empty_src_accepted() {
        // Empty src merged with non-empty dst: extent stays same.
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = LiveInterval::new(vreg(0));
        let dst = interval(1, &[(5, 10)]);
        assert!(coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_conservative_empty_dst_accepted() {
        let coalescer = CopyCoalescer::new(CoalesceMode::Conservative);
        let src = interval(0, &[(5, 10)]);
        let dst = LiveInterval::new(vreg(1));
        assert!(coalescer.can_coalesce(&src, &dst));
    }

    #[test]
    fn test_aggressive_vs_conservative_stats_differ() {
        // Construct a function where aggressive coalesces more than conservative.
        let func = make_function(vec![vec![
            copy_inst(vreg(1), vreg(0)),
            copy_inst(vreg(3), vreg(2)),
        ]]);

        // First pair: adjacent -> both modes accept.
        // Second pair: wide gap -> conservative rejects.
        let mut intervals_agg = HashMap::from([
            (0, interval(0, &[(0, 1)])),
            (1, interval(1, &[(1, 2)])),
            (2, interval(2, &[(3, 4)])),
            (3, interval(3, &[(50, 51)])),
        ]);
        let mut intervals_con = intervals_agg.clone();

        let mut aggressive = CopyCoalescer::new(CoalesceMode::Aggressive);
        let result_agg = aggressive.coalesce(&func, &mut intervals_agg);

        let mut conservative = CopyCoalescer::new(CoalesceMode::Conservative);
        let result_con = conservative.coalesce(&func, &mut intervals_con);

        // Aggressive: both copies coalesced.
        assert_eq!(result_agg.copies_removed, 2);
        assert_eq!(result_agg.intervals_merged, 2);

        // Conservative: only the first (adjacent) copy coalesced.
        assert_eq!(result_con.copies_removed, 1);
        assert_eq!(result_con.intervals_merged, 1);

        // Stats should reflect the difference.
        let stats_agg = result_agg.stats(2);
        let stats_con = result_con.stats(2);
        assert_eq!(stats_agg.copies_eliminated, 2);
        assert_eq!(stats_agg.copies_remaining, 0);
        assert_eq!(stats_con.copies_eliminated, 1);
        assert_eq!(stats_con.copies_remaining, 1);
    }
}

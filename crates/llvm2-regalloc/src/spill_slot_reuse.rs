// llvm2-regalloc/spill_slot_reuse.rs - Spill slot reuse optimization
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Spill slot reuse: share stack slots for non-overlapping spilled intervals.
//!
//! Without this optimization, each spilled VReg gets its own dedicated stack
//! slot. When two spilled VRegs have non-overlapping live intervals, they
//! can safely share the same slot, reducing stack frame size.
//!
//! Reference: LLVM `StackSlotColoring.cpp`

use crate::linear_scan::SpillInfo;
use crate::liveness::LiveInterval;
use crate::machine_types::{MachFunction, RegClass, StackSlotId, VReg};
use std::collections::HashMap;

/// Result of spill slot reuse analysis.
#[derive(Debug, Clone)]
pub struct SpillSlotReuseResult {
    /// Number of slots eliminated through sharing.
    pub slots_eliminated: u32,
    /// Mapping from original slot -> reused slot.
    pub slot_rewrites: HashMap<StackSlotId, StackSlotId>,
}

/// Analyze spilled intervals and identify slot reuse opportunities.
///
/// Two spilled VRegs can share a slot if:
/// 1. They have the same register class (same size/alignment).
/// 2. Their live intervals do not overlap.
///
/// Uses a greedy coloring approach: for each spill, try to reuse an
/// existing slot whose interval doesn't conflict, otherwise allocate new.
pub fn compute_spill_slot_reuse(
    spill_infos: &[SpillInfo],
    intervals: &HashMap<u32, LiveInterval>,
) -> SpillSlotReuseResult {
    let mut result = SpillSlotReuseResult {
        slots_eliminated: 0,
        slot_rewrites: HashMap::new(),
    };

    // Group spills by register class (same class = same slot size).
    let mut groups: HashMap<RegClass, Vec<(VReg, StackSlotId)>> = HashMap::new();
    for si in spill_infos {
        groups
            .entry(si.vreg.class)
            .or_default()
            .push((si.vreg, si.slot));
    }

    for spills in groups.values() {
        // Coloring: assign each spill to a "color" (shared slot).
        // Colors hold the list of vreg intervals assigned to them.
        let mut colors: Vec<(StackSlotId, Vec<u32>)> = Vec::new();

        for &(vreg, original_slot) in spills {
            let vreg_interval = match intervals.get(&vreg.id) {
                Some(iv) => iv,
                None => {
                    // No interval data — can't share.
                    colors.push((original_slot, vec![vreg.id]));
                    continue;
                }
            };

            // Try to find an existing color that doesn't conflict.
            let mut found_color = false;
            for (color_slot, color_vregs) in &mut colors {
                let conflicts = color_vregs.iter().any(|&existing_id| {
                    if let Some(existing_interval) = intervals.get(&existing_id) {
                        existing_interval.overlaps(vreg_interval)
                    } else {
                        true // Conservative: assume conflict if no interval data.
                    }
                });

                if !conflicts {
                    // Reuse this slot.
                    color_vregs.push(vreg.id);
                    if original_slot != *color_slot {
                        result.slot_rewrites.insert(original_slot, *color_slot);
                        result.slots_eliminated += 1;
                    }
                    found_color = true;
                    break;
                }
            }

            if !found_color {
                // Need a new color (keep original slot).
                colors.push((original_slot, vec![vreg.id]));
            }
        }
    }

    result
}

/// Apply spill slot rewrites to spill instructions in the function.
///
/// Updates all StackSlot operands according to the rewrite map.
pub fn apply_spill_slot_reuse(
    func: &mut MachFunction,
    rewrites: &HashMap<StackSlotId, StackSlotId>,
) {
    if rewrites.is_empty() {
        return;
    }

    for inst in &mut func.insts {
        for op in inst.defs.iter_mut().chain(inst.uses.iter_mut()) {
            if let crate::machine_types::MachOperand::StackSlot(slot) = op
                && let Some(&new_slot) = rewrites.get(slot) {
                    *slot = new_slot;
                }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::LiveInterval;
    use crate::machine_types::{RegClass, StackSlotId, VReg};

    fn vreg(id: u32) -> VReg {
        VReg {
            id,
            class: RegClass::Gpr64,
        }
    }

    fn interval(id: u32, ranges: &[(u32, u32)]) -> LiveInterval {
        let mut iv = LiveInterval::new(vreg(id));
        for &(start, end) in ranges {
            iv.add_range(start, end);
        }
        iv
    }

    #[test]
    fn test_non_overlapping_share_slot() {
        let spills = vec![
            SpillInfo {
                vreg: vreg(0),
                slot: StackSlotId(0),
            },
            SpillInfo {
                vreg: vreg(1),
                slot: StackSlotId(1),
            },
        ];

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(10, 15)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 1);
        assert_eq!(result.slot_rewrites.get(&StackSlotId(1)), Some(&StackSlotId(0)));
    }

    #[test]
    fn test_overlapping_need_separate_slots() {
        let spills = vec![
            SpillInfo {
                vreg: vreg(0),
                slot: StackSlotId(0),
            },
            SpillInfo {
                vreg: vreg(1),
                slot: StackSlotId(1),
            },
        ];

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 10)])),
            (1, interval(1, &[(5, 15)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 0);
        assert!(result.slot_rewrites.is_empty());
    }

    #[test]
    fn test_three_way_reuse() {
        let spills = vec![
            SpillInfo {
                vreg: vreg(0),
                slot: StackSlotId(0),
            },
            SpillInfo {
                vreg: vreg(1),
                slot: StackSlotId(1),
            },
            SpillInfo {
                vreg: vreg(2),
                slot: StackSlotId(2),
            },
        ];

        // [0,5), [10,15), [20,25) — all non-overlapping.
        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(10, 15)])),
            (2, interval(2, &[(20, 25)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 2);
    }

    #[test]
    fn test_different_classes_no_reuse() {
        let spills = vec![
            SpillInfo {
                vreg: VReg {
                    id: 0,
                    class: RegClass::Gpr64,
                },
                slot: StackSlotId(0),
            },
            SpillInfo {
                vreg: VReg {
                    id: 1,
                    class: RegClass::Fpr64,
                },
                slot: StackSlotId(1),
            },
        ];

        // Non-overlapping but different classes.
        let intervals = HashMap::from([
            (
                0,
                {
                    let mut iv = LiveInterval::new(VReg {
                        id: 0,
                        class: RegClass::Gpr64,
                    });
                    iv.add_range(0, 5);
                    iv
                },
            ),
            (
                1,
                {
                    let mut iv = LiveInterval::new(VReg {
                        id: 1,
                        class: RegClass::Fpr64,
                    });
                    iv.add_range(10, 15);
                    iv
                },
            ),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        // Different register classes -> different groups -> no sharing.
        assert_eq!(result.slots_eliminated, 0);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case and correctness tests (issue #139)
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_spills() {
        let result = compute_spill_slot_reuse(&[], &HashMap::new());
        assert_eq!(result.slots_eliminated, 0);
        assert!(result.slot_rewrites.is_empty());
    }

    #[test]
    fn test_single_spill_no_reuse() {
        let spills = vec![SpillInfo {
            vreg: vreg(0),
            slot: StackSlotId(0),
        }];
        let intervals = HashMap::from([(0, interval(0, &[(0, 10)]))]);
        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 0);
    }

    #[test]
    fn test_adjacent_intervals_share_slot() {
        // [0,5) and [5,10) are adjacent (non-overlapping) — should share.
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
        ];
        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(5, 10)])),
        ]);
        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 1);
    }

    #[test]
    fn test_partial_overlap_two_of_three() {
        // v0: [0,10), v1: [5,15), v2: [20,25)
        // v0 and v1 overlap (can't share). v2 can share with v0 or v1.
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
            SpillInfo { vreg: vreg(2), slot: StackSlotId(2) },
        ];
        let intervals = HashMap::from([
            (0, interval(0, &[(0, 10)])),
            (1, interval(1, &[(5, 15)])),
            (2, interval(2, &[(20, 25)])),
        ]);
        let result = compute_spill_slot_reuse(&spills, &intervals);
        // v2 can share with v0, so 1 slot eliminated.
        assert_eq!(result.slots_eliminated, 1);
    }

    #[test]
    fn test_all_overlapping_no_reuse() {
        // All three intervals overlap pairwise — no sharing possible.
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
            SpillInfo { vreg: vreg(2), slot: StackSlotId(2) },
        ];
        let intervals = HashMap::from([
            (0, interval(0, &[(0, 10)])),
            (1, interval(1, &[(5, 15)])),
            (2, interval(2, &[(8, 20)])),
        ]);
        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 0);
    }

    #[test]
    fn test_missing_interval_conservative() {
        // When interval data is missing for a vreg, it should conservatively
        // assume conflict and not share.
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
        ];
        // Only provide interval for v0, not v1.
        let intervals = HashMap::from([(0, interval(0, &[(0, 5)]))]);
        let result = compute_spill_slot_reuse(&spills, &intervals);
        // v1 has no interval data — should get its own color, no sharing.
        assert_eq!(result.slots_eliminated, 0);
    }

    #[test]
    fn test_apply_spill_slot_reuse_rewrites_operands() {
        use crate::machine_types::*;
        let mut func = MachFunction {
            name: "rewrite_test".into(),
            insts: vec![
                MachInst {
                    opcode: crate::spill::PSEUDO_SPILL_STORE,
                    defs: vec![],
                    uses: vec![
                        MachOperand::VReg(vreg(0)),
                        MachOperand::StackSlot(StackSlotId(1)),
                    ],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::WRITES_MEMORY,
                },
                MachInst {
                    opcode: crate::spill::PSEUDO_SPILL_LOAD,
                    defs: vec![MachOperand::VReg(vreg(0))],
                    uses: vec![MachOperand::StackSlot(StackSlotId(1))],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::READS_MEMORY,
                },
            ],
            blocks: vec![MachBlock {
                insts: vec![InstId(0), InstId(1)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 2,
            stack_slots: HashMap::new(),
        };

        let rewrites = HashMap::from([(StackSlotId(1), StackSlotId(0))]);
        apply_spill_slot_reuse(&mut func, &rewrites);

        // All StackSlot(1) operands should now be StackSlot(0).
        for inst in &func.insts {
            for op in inst.defs.iter().chain(inst.uses.iter()) {
                if let MachOperand::StackSlot(slot) = op {
                    assert_ne!(*slot, StackSlotId(1), "slot 1 should have been rewritten to slot 0");
                }
            }
        }
    }

    #[test]
    fn test_apply_spill_slot_reuse_empty_rewrites() {
        use crate::machine_types::*;
        let mut func = MachFunction {
            name: "no_rewrite".into(),
            insts: vec![MachInst {
                opcode: crate::spill::PSEUDO_SPILL_LOAD,
                defs: vec![MachOperand::VReg(vreg(0))],
                uses: vec![MachOperand::StackSlot(StackSlotId(0))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::READS_MEMORY,
            }],
            blocks: vec![MachBlock {
                insts: vec![InstId(0)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 1,
            stack_slots: HashMap::new(),
        };

        // Empty rewrites should be a no-op.
        apply_spill_slot_reuse(&mut func, &HashMap::new());
        let slot_op = &func.insts[0].uses[0];
        assert_eq!(*slot_op, MachOperand::StackSlot(StackSlotId(0)));
    }

    #[test]
    fn test_many_non_overlapping_all_share_one_slot() {
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
            SpillInfo { vreg: vreg(2), slot: StackSlotId(2) },
            SpillInfo { vreg: vreg(3), slot: StackSlotId(3) },
            SpillInfo { vreg: vreg(4), slot: StackSlotId(4) },
        ];

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(10, 15)])),
            (2, interval(2, &[(20, 25)])),
            (3, interval(3, &[(30, 35)])),
            (4, interval(4, &[(40, 45)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 4);
        assert_eq!(result.slot_rewrites.len(), 4);
        assert_eq!(result.slot_rewrites.get(&StackSlotId(1)), Some(&StackSlotId(0)));
        assert_eq!(result.slot_rewrites.get(&StackSlotId(2)), Some(&StackSlotId(0)));
        assert_eq!(result.slot_rewrites.get(&StackSlotId(3)), Some(&StackSlotId(0)));
        assert_eq!(result.slot_rewrites.get(&StackSlotId(4)), Some(&StackSlotId(0)));
    }

    #[test]
    fn test_interleaved_overlapping_uses_two_slots() {
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
            SpillInfo { vreg: vreg(2), slot: StackSlotId(2) },
            SpillInfo { vreg: vreg(3), slot: StackSlotId(3) },
        ];

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 10)])),
            (1, interval(1, &[(5, 15)])),
            (2, interval(2, &[(10, 20)])),
            (3, interval(3, &[(15, 25)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 2);
        assert_eq!(result.slot_rewrites.len(), 2);
        assert_eq!(result.slot_rewrites.get(&StackSlotId(2)), Some(&StackSlotId(0)));
        assert_eq!(result.slot_rewrites.get(&StackSlotId(3)), Some(&StackSlotId(1)));
    }

    #[test]
    fn test_same_slot_id_no_rewrite() {
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(7) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(7) },
        ];

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(10, 15)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 0);
        assert!(result.slot_rewrites.is_empty());
    }

    #[test]
    fn test_chain_of_adjacent_intervals_reuse() {
        let spills = vec![
            SpillInfo { vreg: vreg(0), slot: StackSlotId(0) },
            SpillInfo { vreg: vreg(1), slot: StackSlotId(1) },
            SpillInfo { vreg: vreg(2), slot: StackSlotId(2) },
            SpillInfo { vreg: vreg(3), slot: StackSlotId(3) },
            SpillInfo { vreg: vreg(4), slot: StackSlotId(4) },
            SpillInfo { vreg: vreg(5), slot: StackSlotId(5) },
            SpillInfo { vreg: vreg(6), slot: StackSlotId(6) },
            SpillInfo { vreg: vreg(7), slot: StackSlotId(7) },
            SpillInfo { vreg: vreg(8), slot: StackSlotId(8) },
            SpillInfo { vreg: vreg(9), slot: StackSlotId(9) },
        ];

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(5, 10)])),
            (2, interval(2, &[(10, 15)])),
            (3, interval(3, &[(15, 20)])),
            (4, interval(4, &[(20, 25)])),
            (5, interval(5, &[(25, 30)])),
            (6, interval(6, &[(30, 35)])),
            (7, interval(7, &[(35, 40)])),
            (8, interval(8, &[(40, 45)])),
            (9, interval(9, &[(45, 50)])),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 9);
        assert_eq!(result.slot_rewrites.len(), 9);
        for slot in 1..10 {
            assert_eq!(
                result.slot_rewrites.get(&StackSlotId(slot)),
                Some(&StackSlotId(0))
            );
        }
    }

    #[test]
    fn test_mixed_classes_separate_groups() {
        let g0 = VReg { id: 0, class: RegClass::Gpr64 };
        let g1 = VReg { id: 1, class: RegClass::Gpr64 };
        let g2 = VReg { id: 2, class: RegClass::Gpr64 };
        let f3 = VReg { id: 3, class: RegClass::Fpr64 };
        let f4 = VReg { id: 4, class: RegClass::Fpr64 };

        let spills = vec![
            SpillInfo { vreg: g0, slot: StackSlotId(0) },
            SpillInfo { vreg: g1, slot: StackSlotId(1) },
            SpillInfo { vreg: g2, slot: StackSlotId(2) },
            SpillInfo { vreg: f3, slot: StackSlotId(10) },
            SpillInfo { vreg: f4, slot: StackSlotId(11) },
        ];

        let mut f3_iv = LiveInterval::new(f3);
        f3_iv.add_range(100, 105);
        let mut f4_iv = LiveInterval::new(f4);
        f4_iv.add_range(110, 115);

        let intervals = HashMap::from([
            (0, interval(0, &[(0, 5)])),
            (1, interval(1, &[(10, 15)])),
            (2, interval(2, &[(20, 25)])),
            (3, f3_iv),
            (4, f4_iv),
        ]);

        let result = compute_spill_slot_reuse(&spills, &intervals);
        assert_eq!(result.slots_eliminated, 3);
        assert_eq!(result.slot_rewrites.len(), 3);
        assert_eq!(result.slot_rewrites.get(&StackSlotId(1)), Some(&StackSlotId(0)));
        assert_eq!(result.slot_rewrites.get(&StackSlotId(2)), Some(&StackSlotId(0)));
        assert_eq!(result.slot_rewrites.get(&StackSlotId(11)), Some(&StackSlotId(10)));
    }
}

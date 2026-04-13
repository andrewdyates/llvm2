// llvm2-regalloc/spill_slot_reuse.rs - Spill slot reuse optimization
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

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

    for (_class, spills) in &groups {
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
            if let crate::machine_types::MachOperand::StackSlot(slot) = op {
                if let Some(&new_slot) = rewrites.get(slot) {
                    *slot = new_slot;
                }
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
}

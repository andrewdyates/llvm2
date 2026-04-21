// llvm2-lower/overflow_idiom.rs - i128-widened signed-overflow idiom detection
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Pre-ISel detection of the canonical i128-widened signed-overflow idiom.
//!
//! z4 (and other front-ends targeting SMT-style overflow checking) emits
//! signed-overflow checks on `i64 add/sub` in i128-widened form:
//!
//! ```text
//!     sum       = Add(I64, a, b)                  // or Sub for subtraction
//!     sext_a    = SExt(I64 -> I128, a)
//!     sext_b    = SExt(I64 -> I128, b)
//!     true_sum  = Add(I128, sext_a, sext_b)       // or Sub for subtraction
//!     sext_sum  = SExt(I64 -> I128, sum)
//!     overflow  = Icmp(Ne, I128, sext_sum, true_sum)
//! ```
//!
//! This module detects that pattern at the LIR level and emits direct
//! `ADDS/SUBS` + `B.VS` on aarch64. The cold/slow i128 path would otherwise
//! produce ~6 instructions (ADDS+ADC, SXTW, ADDS+ADC, SXTW, CMP+CMP+CSET+…)
//! for the same meaning. The output collapses to:
//!
//! ```text
//!     ADDS Xsum, Xa, Xb          // flag-setting low-half add; Xsum = a + b
//!     B.VS overflow_block        // branch on V flag
//! ```
//!
//! (or `SUBS Xsum, Xa, Xb` for the subtraction variant.)
//!
//! ## Matching constraints
//!
//! - Matching is **SSA-exact**: the `sum` operand inside `sext_sum = SExt(sum)`
//!   must be the exact same [`Value`] as the result of the `Add(I64,a,b)`.
//!   We do not match structurally equivalent additions.
//! - Both the wide and narrow ops must match: add-to-add or sub-to-sub only.
//! - All six operations must live in the same basic block.
//! - The order must be: narrow op, then sexts of both narrow operands (in any
//!   order), then the wide op, then sext of the narrow sum, then the compare.
//! - The compare must be `IntCC::NotEqual` on `I128`.
//!
//! ## Conservative on ambiguity
//!
//! If any of the intermediate Values has extra users outside the idiom's
//! internal chain, we still allow the rewrite: the idiom emits the narrow sum
//! (via ADDS) and keeps that live; other uses of the narrow sum are unaffected
//! (it's the same vreg). If the intermediate i128 values (sexts, true_sum,
//! sext_sum) have external users, the idiom cannot be applied because their
//! i128 representations are needed — in that case we fall back to full i128
//! lowering and leave the block untouched. Similarly if the overflow boolean
//! is consumed by something other than a Brif (or appears more than once in
//! the block's instructions), we still apply the idiom but materialise the
//! V-flag into a vreg via CSET so the result stays usable as a boolean.

use std::collections::{HashMap, HashSet};

use crate::instructions::{Instruction, IntCC, Opcode, Value};
use crate::types::Type;

/// Kind of signed-overflow idiom recognised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowKind {
    /// Signed addition with overflow: `ADDS`-style (flag V set on overflow).
    SignedAdd,
    /// Signed subtraction with overflow: `SUBS`-style (flag V set on overflow).
    SignedSub,
}

/// Information about a detected signed-overflow idiom.
#[derive(Debug, Clone)]
pub struct OverflowIdiom {
    /// Whether this is a signed add or signed sub overflow check.
    pub kind: OverflowKind,
    /// Narrow (I64) lhs operand.
    pub a: Value,
    /// Narrow (I64) rhs operand.
    pub b: Value,
    /// Narrow (I64) sum/diff result (live-out of the idiom).
    pub sum: Value,
    /// Boolean overflow result (consumed by Brif or downstream logic).
    pub overflow: Value,
    /// Index of the narrow arithmetic op in the block's `instructions` vec.
    pub narrow_idx: usize,
    /// Indices of the idiom's intermediate i128 ops that should be skipped
    /// during ISel (the two SExt(a/b), the wide Add/Sub, the SExt(sum),
    /// and the Icmp Ne). Does **not** include `narrow_idx`.
    pub skip_indices: Vec<usize>,
    /// True if the overflow boolean is referenced by more than one LIR op, or
    /// by an op other than the block's terminal `Brif`. When set, the
    /// selector must materialise the V flag into a vreg via `CSET VS`
    /// immediately after the `ADDS`/`SUBS` to keep the value available for
    /// non-branch consumers.
    pub needs_cset_fallback: bool,
}

/// Result of scanning a block for signed-overflow idioms.
#[derive(Debug, Default, Clone)]
pub struct OverflowAnalysis {
    /// All detected idioms, keyed by the overflow boolean's Value.
    pub by_overflow: HashMap<Value, OverflowIdiom>,
    /// All detected idioms, keyed by the narrow sum Value.
    pub by_sum: HashMap<Value, OverflowIdiom>,
    /// Flat set of instruction indices that should be skipped during ISel
    /// (the idiom intermediates — does **not** include narrow arithmetic ops,
    /// which are still needed to produce the sum register).
    pub skip_indices: HashSet<usize>,
}

impl OverflowAnalysis {
    /// Does the given overflow boolean Value belong to a detected idiom?
    pub fn overflow_idiom(&self, v: &Value) -> Option<&OverflowIdiom> {
        self.by_overflow.get(v)
    }

    /// Does the given narrow sum Value belong to a detected idiom?
    pub fn sum_idiom(&self, v: &Value) -> Option<&OverflowIdiom> {
        self.by_sum.get(v)
    }

    /// Should the instruction at the given index be skipped entirely by ISel?
    pub fn is_skipped(&self, idx: usize) -> bool {
        self.skip_indices.contains(&idx)
    }
}

/// Scan a block's instruction list for signed-overflow idioms.
///
/// Matches are SSA-exact: operand Values must match by identity.
pub fn detect_overflow_idioms(instructions: &[Instruction]) -> OverflowAnalysis {
    // Build a fast lookup: for each LIR Value, where is its producing op?
    // We only index ops that carry a single result (all idiom ops do). Phi
    // nodes and multi-result ops are skipped in the map.
    let mut def_site: HashMap<Value, usize> = HashMap::new();
    for (i, inst) in instructions.iter().enumerate() {
        if inst.results.len() == 1 {
            def_site.insert(inst.results[0], i);
        }
    }

    // Count uses per Value across the block (inside LIR instruction operands).
    let mut use_count: HashMap<Value, usize> = HashMap::new();
    for inst in instructions {
        for a in &inst.args {
            *use_count.entry(*a).or_insert(0) += 1;
        }
    }

    // Find the block's terminator index (last instruction) for fast-path
    // decisions about CSET-fallback.
    let terminator_idx = instructions.len().saturating_sub(1);

    let mut analysis = OverflowAnalysis::default();

    // Walk Icmp instructions — they are the "anchor" of the idiom.
    for (icmp_idx, icmp) in instructions.iter().enumerate() {
        let Opcode::Icmp { cond: IntCC::NotEqual } = icmp.opcode else {
            continue;
        };
        if icmp.args.len() != 2 || icmp.results.len() != 1 {
            continue;
        }
        // Both sides of the compare must be I128.
        let lhs = icmp.args[0];
        let rhs = icmp.args[1];

        // One side must be `SExt(I64 -> I128, sum)`, the other must be
        // `Add(I128, sext_a, sext_b)` where sum = Add(I64, a, b) and
        // sext_a/sext_b are sign-extensions of a/b. Try both orderings.
        let Some((sext_sum_idx, wide_idx)) =
            pick_sext_and_wide(&def_site, lhs, rhs, instructions)
        else {
            continue;
        };
        let sext_sum_inst = &instructions[sext_sum_idx];
        let wide_inst = &instructions[wide_idx];

        // sext_sum must be SExt(I64 -> I128, sum).
        let Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 } = &sext_sum_inst.opcode
        else {
            continue;
        };
        if sext_sum_inst.args.len() != 1 {
            continue;
        }
        let sum_val = sext_sum_inst.args[0];

        // The wide op must be Iadd or Isub on I128.
        let (wide_kind, wide_args_ok) = match &wide_inst.opcode {
            Opcode::Iadd => (OverflowKind::SignedAdd, wide_inst.args.len() == 2),
            Opcode::Isub => (OverflowKind::SignedSub, wide_inst.args.len() == 2),
            _ => continue,
        };
        if !wide_args_ok {
            continue;
        }

        // Find the narrow arithmetic op that produces `sum`.
        let Some(&narrow_idx) = def_site.get(&sum_val) else {
            continue;
        };
        let narrow_inst = &instructions[narrow_idx];
        let narrow_kind = match &narrow_inst.opcode {
            Opcode::Iadd => OverflowKind::SignedAdd,
            Opcode::Isub => OverflowKind::SignedSub,
            _ => continue,
        };
        // Narrow and wide kinds must agree (add ↔ add, sub ↔ sub).
        if narrow_kind != wide_kind {
            continue;
        }
        if narrow_inst.args.len() != 2 || narrow_inst.results.len() != 1 {
            continue;
        }
        let a = narrow_inst.args[0];
        let b = narrow_inst.args[1];

        // The two wide operands must be SExt(a) and SExt(b) respectively.
        // For Iadd we accept either ordering; for Isub the order is fixed
        // (the SExt of the minuend must be the first operand of the wide
        // subtract — subtraction is not commutative).
        let wide_lhs = wide_inst.args[0];
        let wide_rhs = wide_inst.args[1];
        let Some(sext_a_idx) = def_site.get(&wide_lhs).copied() else {
            continue;
        };
        let Some(sext_b_idx) = def_site.get(&wide_rhs).copied() else {
            continue;
        };
        // All four intermediates must be distinct from each other (we do not
        // match degenerate aliasing — the idiom must have five distinct
        // intermediate Values: sext_a, sext_b, true_sum, sext_sum).
        if sext_a_idx == sext_b_idx
            || sext_a_idx == sext_sum_idx
            || sext_b_idx == sext_sum_idx
        {
            continue;
        }
        let sext_a_inst = &instructions[sext_a_idx];
        let sext_b_inst = &instructions[sext_b_idx];

        let Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 } = &sext_a_inst.opcode
        else {
            // Try swapping for Iadd (commutative). For Isub we must fail —
            // the order is semantically significant.
            if wide_kind == OverflowKind::SignedSub {
                continue;
            }
            if let Some((sa, sb)) = swap_sext_pair(
                &def_site,
                wide_rhs,
                wide_lhs,
                instructions,
            ) {
                // Recheck with swapped roles.
                if !matches_sext_pair(instructions, sa, sb, a, b) {
                    continue;
                }
                push_idiom(
                    &mut analysis,
                    icmp_idx,
                    sext_sum_idx,
                    wide_idx,
                    sa,
                    sb,
                    narrow_idx,
                    icmp,
                    wide_kind,
                    a,
                    b,
                    sum_val,
                    &use_count,
                    instructions,
                    terminator_idx,
                );
                continue;
            } else {
                continue;
            }
        };
        let Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 } = &sext_b_inst.opcode
        else {
            continue;
        };
        if sext_a_inst.args.len() != 1 || sext_b_inst.args.len() != 1 {
            continue;
        }
        let sext_a_src = sext_a_inst.args[0];
        let sext_b_src = sext_b_inst.args[0];

        // For Isub: sext_a_src must equal a AND sext_b_src must equal b.
        // For Iadd: (sext_a_src, sext_b_src) must equal (a, b) in some order.
        let pair_matches = match wide_kind {
            OverflowKind::SignedSub => sext_a_src == a && sext_b_src == b,
            OverflowKind::SignedAdd => {
                (sext_a_src == a && sext_b_src == b)
                    || (sext_a_src == b && sext_b_src == a)
            }
        };
        if !pair_matches {
            continue;
        }

        push_idiom(
            &mut analysis,
            icmp_idx,
            sext_sum_idx,
            wide_idx,
            sext_a_idx,
            sext_b_idx,
            narrow_idx,
            icmp,
            wide_kind,
            a,
            b,
            sum_val,
            &use_count,
            instructions,
            terminator_idx,
        );
    }

    analysis
}

/// Return the (sext_sum_idx, wide_op_idx) pair if one of `lhs`/`rhs` is a
/// SExt and the other is a binary Add/Sub producing op.
fn pick_sext_and_wide(
    def_site: &HashMap<Value, usize>,
    lhs: Value,
    rhs: Value,
    instructions: &[Instruction],
) -> Option<(usize, usize)> {
    let lhs_idx = def_site.get(&lhs).copied()?;
    let rhs_idx = def_site.get(&rhs).copied()?;

    if is_sext_i64_to_i128(&instructions[lhs_idx]) && is_wide_binop(&instructions[rhs_idx]) {
        Some((lhs_idx, rhs_idx))
    } else if is_sext_i64_to_i128(&instructions[rhs_idx]) && is_wide_binop(&instructions[lhs_idx])
    {
        Some((rhs_idx, lhs_idx))
    } else {
        None
    }
}

fn is_sext_i64_to_i128(inst: &Instruction) -> bool {
    matches!(
        inst.opcode,
        Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 }
    )
}

fn is_wide_binop(inst: &Instruction) -> bool {
    matches!(inst.opcode, Opcode::Iadd | Opcode::Isub)
}

fn swap_sext_pair(
    def_site: &HashMap<Value, usize>,
    sa_val: Value,
    sb_val: Value,
    _instructions: &[Instruction],
) -> Option<(usize, usize)> {
    let sa = def_site.get(&sa_val).copied()?;
    let sb = def_site.get(&sb_val).copied()?;
    Some((sa, sb))
}

fn matches_sext_pair(
    instructions: &[Instruction],
    sa_idx: usize,
    sb_idx: usize,
    a: Value,
    b: Value,
) -> bool {
    let sa = &instructions[sa_idx];
    let sb = &instructions[sb_idx];
    if !is_sext_i64_to_i128(sa) || !is_sext_i64_to_i128(sb) {
        return false;
    }
    if sa.args.len() != 1 || sb.args.len() != 1 {
        return false;
    }
    (sa.args[0] == a && sb.args[0] == b) || (sa.args[0] == b && sb.args[0] == a)
}

#[allow(clippy::too_many_arguments)]
fn push_idiom(
    analysis: &mut OverflowAnalysis,
    icmp_idx: usize,
    sext_sum_idx: usize,
    wide_idx: usize,
    sext_a_idx: usize,
    sext_b_idx: usize,
    narrow_idx: usize,
    icmp: &Instruction,
    kind: OverflowKind,
    a: Value,
    b: Value,
    sum: Value,
    use_count: &HashMap<Value, usize>,
    instructions: &[Instruction],
    terminator_idx: usize,
) {
    let overflow = icmp.results[0];

    // The four i128 intermediates (sext_a, sext_b, true_sum, sext_sum) must
    // each have exactly one use inside the idiom. If any is referenced
    // outside the idiom chain we cannot drop its producer; fall back to the
    // normal i128 lowering path by **not** registering the idiom.
    let intermediate_values: [Option<Value>; 4] = [
        single_result(instructions, sext_a_idx),
        single_result(instructions, sext_b_idx),
        single_result(instructions, wide_idx),
        single_result(instructions, sext_sum_idx),
    ];
    for maybe_v in &intermediate_values {
        let Some(v) = maybe_v else { return };
        if use_count.get(v).copied().unwrap_or(0) != 1 {
            return;
        }
    }

    // Decide whether the V flag can be consumed directly by a terminator
    // Brif (no CSET needed), or whether we must materialise it.
    //
    // Fast path requirements:
    //   * overflow has exactly one use, AND
    //   * that use is the block's terminator, AND
    //   * the terminator is a Brif whose condition operand is `overflow`.
    //
    // Otherwise, fall back to CSET after ADDS.
    let overflow_uses = use_count.get(&overflow).copied().unwrap_or(0);
    let terminator_is_brif_on_overflow = instructions
        .get(terminator_idx)
        .map(|t| match &t.opcode {
            Opcode::Brif { .. } => t.args.first() == Some(&overflow),
            _ => false,
        })
        .unwrap_or(false);
    let needs_cset_fallback = !(overflow_uses == 1 && terminator_is_brif_on_overflow);

    let skip_indices_local: Vec<usize> = {
        let mut v = vec![sext_a_idx, sext_b_idx, wide_idx, sext_sum_idx, icmp_idx];
        v.sort_unstable();
        v
    };

    let idiom = OverflowIdiom {
        kind,
        a,
        b,
        sum,
        overflow,
        narrow_idx,
        skip_indices: skip_indices_local.clone(),
        needs_cset_fallback,
    };

    for i in &skip_indices_local {
        analysis.skip_indices.insert(*i);
    }
    analysis.by_overflow.insert(overflow, idiom.clone());
    analysis.by_sum.insert(sum, idiom);
}

/// Return the single result Value of `instructions[idx]`, or None if the op
/// does not have exactly one result (out-of-range is also None).
fn single_result(instructions: &[Instruction], idx: usize) -> Option<Value> {
    let inst = instructions.get(idx)?;
    if inst.results.len() == 1 {
        Some(inst.results[0])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::{Instruction, Opcode, Value};
    use crate::types::Type;

    fn mk(opcode: Opcode, args: Vec<Value>, results: Vec<Value>) -> Instruction {
        Instruction { opcode, args, results }
    }

    fn sext_i64_to_i128(src: Value, dst: Value) -> Instruction {
        mk(
            Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 },
            vec![src],
            vec![dst],
        )
    }

    #[test]
    fn detects_signed_add_overflow_idiom() {
        // a = v0, b = v1
        // sum      = Iadd(v0, v1) -> v2
        // sext_a   = SExt(v0)     -> v3
        // sext_b   = SExt(v1)     -> v4
        // true_sum = Iadd(v3, v4) -> v5
        // sext_sum = SExt(v2)     -> v6
        // overflow = Icmp Ne(v6, v5) -> v7
        let insts = vec![
            mk(Opcode::Iadd, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(0), Value(3)),
            sext_i64_to_i128(Value(1), Value(4)),
            mk(Opcode::Iadd, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        let idiom = a.overflow_idiom(&Value(7)).expect("idiom detected");
        assert_eq!(idiom.kind, OverflowKind::SignedAdd);
        assert_eq!(idiom.a, Value(0));
        assert_eq!(idiom.b, Value(1));
        assert_eq!(idiom.sum, Value(2));
        assert_eq!(idiom.overflow, Value(7));
        assert_eq!(idiom.narrow_idx, 0);
        // skip indices cover sext_a, sext_b, true_sum, sext_sum, icmp
        assert!(a.is_skipped(1));
        assert!(a.is_skipped(2));
        assert!(a.is_skipped(3));
        assert!(a.is_skipped(4));
        assert!(a.is_skipped(5));
        assert!(!a.is_skipped(0)); // narrow op is NOT skipped
    }

    #[test]
    fn detects_signed_sub_overflow_idiom() {
        let insts = vec![
            mk(Opcode::Isub, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(0), Value(3)),
            sext_i64_to_i128(Value(1), Value(4)),
            mk(Opcode::Isub, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        let idiom = a.overflow_idiom(&Value(7)).expect("idiom detected");
        assert_eq!(idiom.kind, OverflowKind::SignedSub);
        assert_eq!(idiom.a, Value(0));
        assert_eq!(idiom.b, Value(1));
    }

    #[test]
    fn detects_reversed_compare_operand_order() {
        // Icmp Ne(true_sum, sext_sum) — same semantics, just swapped args.
        let insts = vec![
            mk(Opcode::Iadd, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(0), Value(3)),
            sext_i64_to_i128(Value(1), Value(4)),
            mk(Opcode::Iadd, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(5), Value(6)], // swapped
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        assert!(a.overflow_idiom(&Value(7)).is_some());
    }

    #[test]
    fn rejects_sub_with_swapped_operands() {
        // For SUB the order matters: sext(b), sext(a) is NOT the same as
        // sext(a - b).
        let insts = vec![
            mk(Opcode::Isub, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(1), Value(3)), // sext of b, not a
            sext_i64_to_i128(Value(0), Value(4)), // sext of a, not b
            mk(Opcode::Isub, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        assert!(a.overflow_idiom(&Value(7)).is_none());
    }

    #[test]
    fn accepts_add_with_swapped_sext_operands() {
        // For ADD the two SExts are commutative.
        let insts = vec![
            mk(Opcode::Iadd, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(1), Value(3)), // sext of b first
            sext_i64_to_i128(Value(0), Value(4)), // sext of a second
            mk(Opcode::Iadd, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        assert!(a.overflow_idiom(&Value(7)).is_some());
    }

    #[test]
    fn rejects_equal_compare() {
        let insts = vec![
            mk(Opcode::Iadd, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(0), Value(3)),
            sext_i64_to_i128(Value(1), Value(4)),
            mk(Opcode::Iadd, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::Equal }, // Eq, not Ne
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        assert!(a.overflow_idiom(&Value(7)).is_none());
    }

    #[test]
    fn rejects_mismatched_narrow_kind() {
        // Narrow is Iadd, wide is Isub — kinds disagree, idiom rejected.
        let insts = vec![
            mk(Opcode::Iadd, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(0), Value(3)),
            sext_i64_to_i128(Value(1), Value(4)),
            mk(Opcode::Isub, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(2), Value(6)),
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        assert!(a.overflow_idiom(&Value(7)).is_none());
    }

    #[test]
    fn requires_sext_of_same_sum_value() {
        // sext_sum is SExt of a DIFFERENT Value (not the narrow sum).
        let insts = vec![
            mk(Opcode::Iadd, vec![Value(0), Value(1)], vec![Value(2)]),
            sext_i64_to_i128(Value(0), Value(3)),
            sext_i64_to_i128(Value(1), Value(4)),
            mk(Opcode::Iadd, vec![Value(3), Value(4)], vec![Value(5)]),
            sext_i64_to_i128(Value(0), Value(6)), // SExt of v0, not v2 (sum)
            mk(
                Opcode::Icmp { cond: IntCC::NotEqual },
                vec![Value(6), Value(5)],
                vec![Value(7)],
            ),
        ];
        let a = detect_overflow_idioms(&insts);
        assert!(a.overflow_idiom(&Value(7)).is_none());
    }
}

// llvm2-lower/smulh_idiom.rs - AArch64 SMULH idiom detection
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Pre-ISel detection of the canonical i128-widened signed-multiply-high
//! idiom.
//!
//! z4's simplex JIT emits signed `mulhi(i64, i64)` in widened form:
//!
//! ```text
//!     sext_a   = SExt(I64 -> I128, a)
//!     sext_b   = SExt(I64 -> I128, b)
//!     wide     = IMul(I128, sext_a, sext_b)
//!     shifted  = SShr(I128, wide, 64)
//!     hi       = Trunc(I128 -> I64, shifted)
//! ```
//!
//! That final `hi` value is exactly the result of AArch64
//! `SMULH Xd, Xn, Xm`.
//!
//! ## Matching constraints
//!
//! - Matching is **SSA-exact**: each edge in the chain must use the exact
//!   [`Value`] produced by the matched predecessor.
//! - Matching is in-block only: the detector sees a single block's
//!   instruction list and never crosses block boundaries.
//! - The four i128 chain intermediates (`sext_a`, `sext_b`, `wide`,
//!   `shifted`) must each have exactly one use in the block. If any of them
//!   has an external user, we must keep the full widened representation and
//!   the idiom does not fire.
//! - The shift amount must be 64, encoded either as:
//!   - `Iconst { ty: I128, imm: 64 }`, or
//!   - `SExt(I64 -> I128, Iconst { ty: I64, imm: 64 })`.
//! - The matched `Trunc(I128 -> I64)` is **not** skipped: it is the dispatch
//!   point where ISel emits `SMULH` and defines the final narrow result.
//! - The matched shift-amount producer may have extra users. If it is used
//!   only by the idiom's `SShr`, we also skip that producer; otherwise it is
//!   left in place for its other consumers.

use std::collections::{HashMap, HashSet};

use crate::instructions::{Instruction, Opcode, Value};
use crate::types::Type;

/// Information about a detected signed multiply-high idiom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmulhIdiom {
    /// Narrow (I64) lhs operand.
    pub a: Value,
    /// Narrow (I64) rhs operand.
    pub b: Value,
    /// Final narrow high-half result (the matched Trunc result).
    pub hi: Value,
    /// Index of the matched `Trunc(I128 -> I64)` instruction.
    pub trunc_idx: usize,
    /// Indices of the four wide intermediates that ISel should skip:
    /// `sext_a`, `sext_b`, `wide`, `shifted`.
    pub skip_indices: [usize; 4],
}

/// Result of scanning a block for signed multiply-high idioms.
#[derive(Debug, Default, Clone)]
pub struct SmulhAnalysis {
    /// All detected idioms, keyed by the final narrow high-half Value.
    pub by_hi: HashMap<Value, SmulhIdiom>,
    /// All detected idioms, keyed by the matched Trunc instruction index.
    pub by_trunc: HashMap<usize, SmulhIdiom>,
    /// Flat set of instruction indices that should be skipped by ISel.
    ///
    /// This always includes the four wide intermediates from
    /// [`SmulhIdiom::skip_indices`]. It may also include the matched
    /// shift-amount producer when that producer is used only by the idiom.
    pub skip_indices: HashSet<usize>,
}

impl SmulhAnalysis {
    /// Return the detected SMULH idiom for the Trunc instruction at `idx`.
    pub fn smulh_for_trunc(&self, idx: usize) -> Option<&SmulhIdiom> {
        self.by_trunc.get(&idx)
    }

    /// Should the instruction at the given index be skipped entirely by ISel?
    pub fn is_skipped(&self, idx: usize) -> bool {
        self.skip_indices.contains(&idx)
    }
}

/// Scan a block's instruction list for signed multiply-high idioms.
///
/// Matches are SSA-exact: operand Values must match by identity.
pub fn detect_smulh_idioms(instructions: &[Instruction]) -> SmulhAnalysis {
    let mut def_site: HashMap<Value, usize> = HashMap::new();
    for (i, inst) in instructions.iter().enumerate() {
        if inst.results.len() == 1 {
            def_site.insert(inst.results[0], i);
        }
    }

    let mut use_count: HashMap<Value, usize> = HashMap::new();
    for inst in instructions {
        for arg in &inst.args {
            *use_count.entry(*arg).or_insert(0) += 1;
        }
    }

    let mut analysis = SmulhAnalysis::default();

    for (trunc_idx, trunc) in instructions.iter().enumerate() {
        let Opcode::Trunc { to_ty: Type::I64 } = &trunc.opcode else {
            continue;
        };
        if trunc.args.len() != 1 || trunc.results.len() != 1 {
            continue;
        }

        let shifted_val = trunc.args[0];
        let hi = trunc.results[0];

        let Some(shifted_idx) = def_site.get(&shifted_val).copied() else {
            continue;
        };
        let shifted_inst = &instructions[shifted_idx];
        if !matches!(shifted_inst.opcode, Opcode::Sshr)
            || shifted_inst.args.len() != 2
            || shifted_inst.results.len() != 1
        {
            continue;
        }

        let wide_val = shifted_inst.args[0];
        let shift_val = shifted_inst.args[1];
        let Some(extra_skip_idx) =
            match_shift_amount(&def_site, &use_count, instructions, shift_val)
        else {
            continue;
        };

        let Some(wide_idx) = def_site.get(&wide_val).copied() else {
            continue;
        };
        let wide_inst = &instructions[wide_idx];
        if !matches!(wide_inst.opcode, Opcode::Imul)
            || wide_inst.args.len() != 2
            || wide_inst.results.len() != 1
        {
            continue;
        }

        let sext_a_val = wide_inst.args[0];
        let sext_b_val = wide_inst.args[1];

        let Some(sext_a_idx) = def_site.get(&sext_a_val).copied() else {
            continue;
        };
        let Some(sext_b_idx) = def_site.get(&sext_b_val).copied() else {
            continue;
        };
        if sext_a_idx == sext_b_idx
            || sext_a_idx == wide_idx
            || sext_a_idx == shifted_idx
            || sext_b_idx == wide_idx
            || sext_b_idx == shifted_idx
            || wide_idx == shifted_idx
        {
            continue;
        }

        let sext_a_inst = &instructions[sext_a_idx];
        let sext_b_inst = &instructions[sext_b_idx];

        let Opcode::Sextend {
            from_ty: Type::I64,
            to_ty: Type::I128,
        } = &sext_a_inst.opcode
        else {
            continue;
        };
        let Opcode::Sextend {
            from_ty: Type::I64,
            to_ty: Type::I128,
        } = &sext_b_inst.opcode
        else {
            continue;
        };
        if sext_a_inst.args.len() != 1
            || sext_a_inst.results.len() != 1
            || sext_b_inst.args.len() != 1
            || sext_b_inst.results.len() != 1
        {
            continue;
        }

        let a = sext_a_inst.args[0];
        let b = sext_b_inst.args[0];

        let intermediate_values = [sext_a_val, sext_b_val, wide_val, shifted_val];
        if intermediate_values
            .iter()
            .any(|v| use_count.get(v).copied().unwrap_or(0) != 1)
        {
            continue;
        }

        let idiom = SmulhIdiom {
            a,
            b,
            hi,
            trunc_idx,
            skip_indices: [sext_a_idx, sext_b_idx, wide_idx, shifted_idx],
        };

        for idx in idiom.skip_indices {
            analysis.skip_indices.insert(idx);
        }
        if let Some(idx) = extra_skip_idx {
            analysis.skip_indices.insert(idx);
        }
        analysis.by_hi.insert(hi, idiom);
        analysis.by_trunc.insert(trunc_idx, idiom);
    }

    analysis
}

fn match_shift_amount(
    def_site: &HashMap<Value, usize>,
    use_count: &HashMap<Value, usize>,
    instructions: &[Instruction],
    shift_value: Value,
) -> Option<Option<usize>> {
    let shift_idx = def_site.get(&shift_value).copied()?;
    let shift_inst = instructions.get(shift_idx)?;

    match &shift_inst.opcode {
        Opcode::Iconst {
            ty: Type::I128,
            imm,
        } if *imm == 64 => {
            let extra_skip =
                (use_count.get(&shift_value).copied().unwrap_or(0) == 1).then_some(shift_idx);
            Some(extra_skip)
        }
        Opcode::Sextend {
            from_ty: Type::I64,
            to_ty: Type::I128,
        } => {
            if shift_inst.args.len() != 1 || shift_inst.results.len() != 1 {
                return None;
            }

            let narrow_const_val = shift_inst.args[0];
            let narrow_const_idx = def_site.get(&narrow_const_val).copied()?;
            let narrow_const_inst = instructions.get(narrow_const_idx)?;

            match &narrow_const_inst.opcode {
                Opcode::Iconst { ty: Type::I64, imm } if *imm == 64 => {
                    let extra_skip = (use_count.get(&shift_value).copied().unwrap_or(0) == 1)
                        .then_some(shift_idx);
                    Some(extra_skip)
                }
                _ => None,
            }
        }
        _ => None,
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
        Instruction {
            opcode,
            args,
            results,
        }
    }

    fn sext_i64_to_i128(src: Value, dst: Value) -> Instruction {
        mk(
            Opcode::Sextend {
                from_ty: Type::I64,
                to_ty: Type::I128,
            },
            vec![src],
            vec![dst],
        )
    }

    fn iconst_i64(imm: i64, dst: Value) -> Instruction {
        mk(Opcode::Iconst { ty: Type::I64, imm }, vec![], vec![dst])
    }

    fn iconst_i128(imm: i64, dst: Value) -> Instruction {
        mk(
            Opcode::Iconst {
                ty: Type::I128,
                imm,
            },
            vec![],
            vec![dst],
        )
    }

    #[test]
    fn detects_smulh_idiom_with_i128_shift_iconst() {
        let insts = vec![
            sext_i64_to_i128(Value(0), Value(2)),
            sext_i64_to_i128(Value(1), Value(3)),
            mk(Opcode::Imul, vec![Value(2), Value(3)], vec![Value(4)]),
            iconst_i128(64, Value(5)),
            mk(Opcode::Sshr, vec![Value(4), Value(5)], vec![Value(6)]),
            mk(
                Opcode::Trunc { to_ty: Type::I64 },
                vec![Value(6)],
                vec![Value(7)],
            ),
        ];

        let analysis = detect_smulh_idioms(&insts);
        let idiom = analysis.by_hi.get(&Value(7)).expect("idiom detected");
        assert_eq!(idiom.a, Value(0));
        assert_eq!(idiom.b, Value(1));
        assert_eq!(idiom.hi, Value(7));
        assert_eq!(idiom.trunc_idx, 5);
        assert_eq!(idiom.skip_indices, [0, 1, 2, 4]);
        assert_eq!(analysis.smulh_for_trunc(5), Some(idiom));
        assert!(analysis.is_skipped(0));
        assert!(analysis.is_skipped(1));
        assert!(analysis.is_skipped(2));
        assert!(analysis.is_skipped(3));
        assert!(analysis.is_skipped(4));
        assert!(!analysis.is_skipped(5));
    }

    #[test]
    fn detects_smulh_idiom_with_sign_extended_i64_shift_iconst() {
        let insts = vec![
            sext_i64_to_i128(Value(0), Value(2)),
            sext_i64_to_i128(Value(1), Value(3)),
            mk(Opcode::Imul, vec![Value(2), Value(3)], vec![Value(4)]),
            iconst_i64(64, Value(5)),
            sext_i64_to_i128(Value(5), Value(6)),
            mk(Opcode::Sshr, vec![Value(4), Value(6)], vec![Value(7)]),
            mk(
                Opcode::Trunc { to_ty: Type::I64 },
                vec![Value(7)],
                vec![Value(8)],
            ),
        ];

        let analysis = detect_smulh_idioms(&insts);
        let idiom = analysis.by_hi.get(&Value(8)).expect("idiom detected");
        assert_eq!(idiom.a, Value(0));
        assert_eq!(idiom.b, Value(1));
        assert_eq!(idiom.trunc_idx, 6);
        assert_eq!(idiom.skip_indices, [0, 1, 2, 5]);
        assert!(analysis.is_skipped(0));
        assert!(analysis.is_skipped(1));
        assert!(analysis.is_skipped(2));
        assert!(!analysis.is_skipped(3));
        assert!(analysis.is_skipped(4));
        assert!(analysis.is_skipped(5));
        assert!(!analysis.is_skipped(6));
    }

    #[test]
    fn rejects_shift_amount_other_than_64() {
        let insts = vec![
            sext_i64_to_i128(Value(0), Value(2)),
            sext_i64_to_i128(Value(1), Value(3)),
            mk(Opcode::Imul, vec![Value(2), Value(3)], vec![Value(4)]),
            iconst_i128(63, Value(5)),
            mk(Opcode::Sshr, vec![Value(4), Value(5)], vec![Value(6)]),
            mk(
                Opcode::Trunc { to_ty: Type::I64 },
                vec![Value(6)],
                vec![Value(7)],
            ),
        ];

        let analysis = detect_smulh_idioms(&insts);
        assert!(analysis.by_hi.is_empty());
        assert!(analysis.by_trunc.is_empty());
        assert!(analysis.skip_indices.is_empty());
    }

    #[test]
    fn rejects_when_intermediate_has_multiple_uses() {
        let insts = vec![
            sext_i64_to_i128(Value(0), Value(2)),
            sext_i64_to_i128(Value(1), Value(3)),
            mk(Opcode::Imul, vec![Value(2), Value(3)], vec![Value(4)]),
            iconst_i128(64, Value(5)),
            mk(Opcode::Sshr, vec![Value(4), Value(5)], vec![Value(6)]),
            mk(
                Opcode::Trunc { to_ty: Type::I64 },
                vec![Value(6)],
                vec![Value(7)],
            ),
            mk(
                Opcode::Trunc { to_ty: Type::I64 },
                vec![Value(4)],
                vec![Value(8)],
            ),
        ];

        let analysis = detect_smulh_idioms(&insts);
        assert!(analysis.by_hi.is_empty());
        assert!(analysis.by_trunc.is_empty());
        assert!(analysis.skip_indices.is_empty());
    }
}

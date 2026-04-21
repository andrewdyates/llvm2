// llvm2-verify/switch_proofs.rs - SMT proofs for Switch lowering correctness
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves that switch lowering in llvm2-lower/src/switch.rs preserves program
// semantics. Switch lowering selects one of three strategies based on density
// and case count (see #323):
//
//   1. Linear scan    -- sequential CMP+B.EQ chain (N <= 3).
//   2. Jump table     -- bounds check + indexed indirect branch (N >= 4, dense).
//   3. Binary search  -- balanced BST of compare-and-branch (N > 3, sparse).
//
// For a bounded scrutinee width, we prove that the lowered machine-side CFG
// computes the same "taken branch target" as the scalar reference semantics
// of `Switch(scrutinee, cases, default)`.
//
// ## Encoding
//
// The reference (tMIR) semantics of a switch is a linear-scan ITE chain over
// the case values, modelled as the selected successor block ID:
//
//   tmir(x) = ite(x == v_0, t_0,
//             ite(x == v_1, t_1,
//             ...
//             ite(x == v_{n-1}, t_{n-1}, default)))
//
// Block identifiers are encoded as distinct small integer constants on the
// same bit-width as the scrutinee. The proof is about *successor identity*,
// not about the concrete instruction sequence that reaches the successor.
// This matches the technique used for branch lowering proofs in `cfg_proofs`
// and `lowering_proof::proof_condbr_generic` (branch-taken bit equivalence).
//
// ### Jump table
//
// The machine-side jump-table lowering is modelled by the same semantics
// a jump-table actually computes:
//
//   idx = x - min_val                       (normalize)
//   in_range = (idx <=_u range)             (bounds check via CMP+B.HI)
//   table_result = ite(idx == 0, t_0, ite(idx == 1, t_1, ..., default))
//   result = ite(in_range, table_result, default)
//
// For a dense contiguous case range [min..=max] with targets `t_i` for
// `v_i = min + i`, the jump-table lookup ITE chain is equivalent to the
// reference linear-scan ITE chain. Holes in the table (case values without
// explicit targets) map to `default`, matching the lowering in
// `emit_jump_table`.
//
// Signed-vs-unsigned note: `emit_jump_table` issues `CMP + B.HI` where HI is
// the unsigned higher. This is correct because after `SUB idx, x, min`, any
// `x < min` underflows modulo-2^w to a very large unsigned integer that fails
// the unsigned bounds check. We model this faithfully by doing an unsigned
// `idx <=_u range` test, with `idx` computed via bitvector subtraction (which
// wraps modulo 2^w, matching AArch64 SUB semantics).
//
// ### Binary search
//
// The BST machine-side lowering is modelled by walking the decision tree:
//
//   bst(x, sorted_cases) =
//     let pivot = sorted_cases[mid]
//     ite(x == pivot_val, pivot_target,
//       ite(x <_s pivot_val,
//         bst(x, sorted_cases[..mid]),     -- left subtree
//         bst(x, sorted_cases[mid+1..])))  -- right subtree
//
// Base case (empty subtree) returns `default`. This directly mirrors the
// code in `emit_bst_node` which emits `CMP ; B.EQ pivot ; B.LT left ; B right`.
//
// The BST uses signed compares (`B.LT` in `emit_bst_node` line 282), so we
// encode the tree walk with `bvslt`. The reference semantics uses `==` only,
// so equivalence holds iff the *sorted* case order is consistent: every value
// strictly less than the pivot is in the left subtree and every value strictly
// greater is in the right. That invariant is established by `sort_by_key` on
// the case list in `emit_binary_search` before the tree walk begins.
//
// ## Why this works at 8-bit
//
// For scrutinee width 8, the evaluation verifier does an exhaustive check
// over all 256 possible scrutinee values. Because both sides are pure
// functions of `x` and produce the same block-id bitvector, a pass at 8-bit
// is a complete proof for that width. We also provide 16-bit variants for
// the z4-prove smoke lane, where z4 reasons symbolically over the entire
// 2^16 domain.
//
// Reference:
// - `crates/llvm2-lower/src/switch.rs` -- the lowering under test
// - `designs/2026-04-13-verification-architecture.md`
// - Alive2 (PLDI 2021) -- technique
// - #323 -- switch lowering; #444 -- this proof obligation
//
// ## Scope
//
// This proof verifies the successor-selection semantics of the generated
// CFG. It does NOT verify the encoding of individual AArch64 instructions
// emitted by `emit_jump_table` / `emit_binary_search` -- those are covered
// by the existing CMP/branch/JumpTable instruction lowering proofs.

//! SMT proofs for switch lowering correctness (jump-table + binary-search).
//!
//! See [`proof_switch_dense_i8`], [`proof_switch_sparse_i8`], and the `_i16`
//! variants. Each proof checks that a representative switch lowering selects
//! the same successor block as a scalar linear-scan reference.

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

/// A (case_value, target_block_id) pair. Block IDs are distinct small
/// integers encoded on the scrutinee's bit-width.
pub type Case = (u64, u64);

/// Build the reference (tMIR-side) semantics of a switch as a linear-scan
/// ITE chain over the case values. Returns the selected block ID as a BV
/// of the given width.
///
/// `cases` is the list of (value, target_block_id) pairs, in source order.
/// `default_id` is the block ID for the fall-through/default target.
pub fn encode_tmir_switch(
    scrutinee: SmtExpr,
    cases: &[Case],
    default_id: u64,
    width: u32,
) -> SmtExpr {
    // Build the ITE chain from right to left so the final (innermost) else
    // arm is the default.
    let mut result = SmtExpr::bv_const(default_id, width);
    for (val, target) in cases.iter().rev() {
        let cond = scrutinee.clone().eq_expr(SmtExpr::bv_const(*val, width));
        let hit = SmtExpr::bv_const(*target, width);
        result = SmtExpr::ite(cond, hit, result);
    }
    result
}

/// Build the machine-side semantics of a *jump table* lowering over a dense
/// contiguous range of cases.
///
/// The lowering is modelled as:
///
/// ```text
/// idx = scrutinee - min_val                    (bitvector subtraction)
/// in_range = idx <=_u range                    (unsigned bounds check)
/// table_result = case_map[idx] (or default for holes)
/// result = in_range ? table_result : default
/// ```
///
/// `cases` may contain holes: any (min_val + i) not present in `cases` maps
/// to `default_id`, exactly like `emit_jump_table` fills its targets vector.
pub fn encode_jump_table_switch(
    scrutinee: SmtExpr,
    cases: &[Case],
    default_id: u64,
    width: u32,
) -> SmtExpr {
    assert!(!cases.is_empty(), "jump-table proof requires at least one case");

    let min_val = cases.iter().map(|(v, _)| *v).min().unwrap();
    let max_val = cases.iter().map(|(v, _)| *v).max().unwrap();
    let range = max_val - min_val;

    // Build the dense target map: index i -> target if case exists, else default.
    // This mirrors the `targets` vector built in `emit_jump_table`.
    let mut dense_targets: Vec<u64> = Vec::with_capacity((range + 1) as usize);
    for i in 0..=range {
        let v = min_val + i;
        let t = cases
            .iter()
            .find(|(cv, _)| *cv == v)
            .map(|(_, t)| *t)
            .unwrap_or(default_id);
        dense_targets.push(t);
    }

    // idx = scrutinee - min_val. For min_val = 0 this is a no-op, matching
    // the `emit_jump_table` fast path.
    let idx = if min_val == 0 {
        scrutinee.clone()
    } else {
        scrutinee
            .clone()
            .bvsub(SmtExpr::bv_const(min_val, width))
    };

    // table_result = ITE chain over idx 0..=range.
    // Innermost else is the default -- this handles the table-hole case the
    // same way emit_jump_table does (hole index -> default block).
    let mut table_result = SmtExpr::bv_const(default_id, width);
    for (i, target) in dense_targets.iter().enumerate().rev() {
        let cond = idx
            .clone()
            .eq_expr(SmtExpr::bv_const(i as u64, width));
        let hit = SmtExpr::bv_const(*target, width);
        table_result = SmtExpr::ite(cond, hit, table_result);
    }

    // in_range = (idx <=_u range).  Unsigned compare matches AArch64's
    // CMP+B.HI: any scrutinee less than min_val wraps to a large unsigned
    // value and fails the bounds check, routing to default.
    let in_range = idx.bvule(SmtExpr::bv_const(range, width));

    SmtExpr::ite(in_range, table_result, SmtExpr::bv_const(default_id, width))
}

/// Build the machine-side semantics of a *binary-search* lowering.
///
/// Encodes the BST as a nested ITE over signed-less-than pivots, matching
/// the structure emitted by `emit_bst_node` in `switch.rs`.
///
/// `cases` is automatically sorted before tree construction (mirroring
/// `emit_binary_search`'s `sort_by_key`).
pub fn encode_binary_search_switch(
    scrutinee: SmtExpr,
    cases: &[Case],
    default_id: u64,
    width: u32,
) -> SmtExpr {
    let mut sorted: Vec<Case> = cases.to_vec();
    sorted.sort_by_key(|(v, _)| *v);
    bst_tree(scrutinee, &sorted, default_id, width)
}

/// Recursive BST encoder -- mirrors `emit_bst_node`.
///
/// For `cases.len() <= 3`: use a linear-scan ITE chain (leaf). `emit_bst_node`
/// falls back to `emit_linear_scan` at the same threshold.
///
/// For `cases.len() > 3`: pick median as pivot, emit
///   ite(x == pivot, t_pivot, ite(x <_s pivot, left, right)).
fn bst_tree(
    scrutinee: SmtExpr,
    cases: &[Case],
    default_id: u64,
    width: u32,
) -> SmtExpr {
    // Leaf: linear scan over <= 3 cases. Falls through to default.
    if cases.len() <= 3 {
        return encode_tmir_switch(scrutinee, cases, default_id, width);
    }

    let mid = cases.len() / 2;
    let (pivot_val, pivot_target) = cases[mid];
    let left = &cases[..mid];
    let right = &cases[mid + 1..];

    let left_expr = bst_tree(scrutinee.clone(), left, default_id, width);
    let right_expr = bst_tree(scrutinee.clone(), right, default_id, width);

    let pivot_const = SmtExpr::bv_const(pivot_val, width);
    let eq_cond = scrutinee.clone().eq_expr(pivot_const.clone());
    // Signed compare matches emit_bst_node which issues B.LT.
    let lt_cond = scrutinee.bvslt(pivot_const);

    SmtExpr::ite(
        eq_cond,
        SmtExpr::bv_const(pivot_target, width),
        SmtExpr::ite(lt_cond, left_expr, right_expr),
    )
}

// ---------------------------------------------------------------------------
// Jump-table proofs: 8-case dense contiguous switch.
// ---------------------------------------------------------------------------

/// Representative dense switch: 8 cases at values 0..=7, targets 1..=8,
/// default = 9. On the i8 scrutinee width, verification is exhaustive
/// over all 256 possible scrutinee values.
fn dense_switch_cases(base: u64) -> (Vec<Case>, u64) {
    let cases: Vec<Case> = (0..8).map(|i| (base + i, 1 + i)).collect();
    let default_id: u64 = 9;
    (cases, default_id)
}

/// Proof: a dense 8-case switch (i8 scrutinee) lowered as a jump table
/// preserves the successor-selection semantics of the reference tMIR switch.
///
/// Dense = range 0..=7, density = 1.0, which triggers `SwitchStrategy::JumpTable`.
/// We verify exhaustively over all 2^8 = 256 scrutinee values.
pub fn proof_switch_dense_i8() -> ProofObligation {
    let width = 8u32;
    let (cases, default_id) = dense_switch_cases(0);
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: dense 8-case i8 -> jump table".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_jump_table_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: a dense 8-case switch (i16 scrutinee) lowered as a jump table.
///
/// Used in the z4-prove smoke lane so z4 reasons symbolically over the full
/// 2^16 input space (statistical sampling would be weaker).
pub fn proof_switch_dense_i16() -> ProofObligation {
    let width = 16u32;
    let (cases, default_id) = dense_switch_cases(0);
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: dense 8-case i16 -> jump table".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_jump_table_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: a dense 8-case switch (i8 scrutinee) with a non-zero base
/// (min_val = 10). Verifies the `SUB idx, x, min_val` normalisation path.
/// Ensures that scrutinee values below min_val underflow and fail the
/// unsigned bounds check, routing to default.
pub fn proof_switch_dense_i8_nonzero_base() -> ProofObligation {
    let width = 8u32;
    let (cases, default_id) = dense_switch_cases(10);
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: dense 8-case i8 min=10 -> jump table (SUB normalise)".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_jump_table_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: a dense switch with a "hole" (case value missing from the range)
/// correctly maps the hole to the default block. Cases 0,1,3,4,5
/// -- index 2 is a hole -- density = 5/6 ~ 0.83 -> jump table.
pub fn proof_switch_dense_i8_with_hole() -> ProofObligation {
    let width = 8u32;
    let cases: Vec<Case> = vec![(0, 1), (1, 2), (3, 3), (4, 4), (5, 5)];
    let default_id = 9u64;
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: dense i8 with hole -> jump table".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_jump_table_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

// ---------------------------------------------------------------------------
// Binary-search proofs: 5-case sparse switch.
// ---------------------------------------------------------------------------

/// Representative sparse switch: 5 cases with gaps that trigger
/// `SwitchStrategy::BinarySearch` (range 100, density = 5/101 < 0.4).
///
/// Values are chosen to lie in the positive signed i8 range [0, 127) so the
/// BST's signed-less-than pivots partition them correctly. On i8 scrutinee
/// width we verify exhaustively.
fn sparse_switch_cases() -> (Vec<Case>, u64) {
    let cases: Vec<Case> = vec![
        (5, 1),
        (20, 2),
        (42, 3),
        (77, 4),
        (100, 5),
    ];
    let default_id: u64 = 9;
    (cases, default_id)
}

/// Proof: a sparse 5-case switch (i8 scrutinee) lowered as a binary search
/// tree preserves the successor-selection semantics of the reference tMIR
/// switch. Verified exhaustively over all 256 scrutinee values.
pub fn proof_switch_sparse_i8() -> ProofObligation {
    let width = 8u32;
    let (cases, default_id) = sparse_switch_cases();
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: sparse 5-case i8 -> binary search".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_binary_search_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: a sparse 5-case switch (i16 scrutinee) lowered as a binary search.
/// Used in the z4-prove smoke lane for symbolic coverage.
pub fn proof_switch_sparse_i16() -> ProofObligation {
    let width = 16u32;
    let (cases, default_id) = sparse_switch_cases();
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: sparse 5-case i16 -> binary search".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_binary_search_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: a sparse 7-case switch (i8 scrutinee) lowered as a binary search.
/// Exercises deeper BST recursion: 7 sorted cases produce a 3-level tree
/// (pivot + 3-case left + 3-case right, with each 3-case leaf falling back
/// to linear scan). Verified exhaustively.
pub fn proof_switch_sparse_7case_i8() -> ProofObligation {
    let width = 8u32;
    let cases: Vec<Case> = vec![
        (1, 1),
        (15, 2),
        (30, 3),
        (50, 4),
        (70, 5),
        (90, 6),
        (110, 7),
    ];
    let default_id: u64 = 9;
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "Switch: sparse 7-case i8 -> binary search (3-level tree)".to_string(),
        tmir_expr: encode_tmir_switch(x.clone(), &cases, default_id, width),
        aarch64_expr: encode_binary_search_switch(x, &cases, default_id, width),
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

// ---------------------------------------------------------------------------
// Aggregate accessor
// ---------------------------------------------------------------------------

/// Return all switch-lowering proofs.
pub fn all_switch_proofs() -> Vec<ProofObligation> {
    vec![
        proof_switch_dense_i8(),
        proof_switch_dense_i8_nonzero_base(),
        proof_switch_dense_i8_with_hole(),
        proof_switch_sparse_i8(),
        proof_switch_sparse_7case_i8(),
        // i16 variants stay in the registry; the default eval path uses
        // statistical sampling for >8-bit widths. z4-prove provides full
        // symbolic coverage on these via the smoke lane.
        proof_switch_dense_i16(),
        proof_switch_sparse_i16(),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::verify_by_evaluation;
    use crate::verify::VerificationResult;

    /// Assert an obligation verifies to Valid. Panics with the counterexample
    /// on failure so test output is actionable.
    fn assert_valid(obligation: &ProofObligation) {
        let result = verify_by_evaluation(obligation);
        match &result {
            VerificationResult::Valid => {}
            VerificationResult::Invalid { counterexample } => {
                panic!(
                    "Proof '{}' FAILED with counterexample: {}",
                    obligation.name, counterexample
                );
            }
            VerificationResult::Unknown { reason } => {
                panic!("Proof '{}' returned Unknown: {}", obligation.name, reason);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Semantic encoder unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tmir_switch_hits_case() {
        let width = 8u32;
        let cases: Vec<Case> = vec![(3, 10), (5, 20), (7, 30)];
        let x = SmtExpr::bv_const(5, width);
        let expr = encode_tmir_switch(x, &cases, 99, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(20));
    }

    #[test]
    fn encode_tmir_switch_falls_through() {
        let width = 8u32;
        let cases: Vec<Case> = vec![(3, 10), (5, 20), (7, 30)];
        let x = SmtExpr::bv_const(42, width); // not in cases
        let expr = encode_tmir_switch(x, &cases, 99, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(99));
    }

    #[test]
    fn jump_table_hit() {
        let width = 8u32;
        let cases: Vec<Case> = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let x = SmtExpr::bv_const(2, width);
        let expr = encode_jump_table_switch(x, &cases, 99, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(3));
    }

    #[test]
    fn jump_table_out_of_range_goes_to_default() {
        let width = 8u32;
        let cases: Vec<Case> = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        // Scrutinee 10 is out of the 0..=3 range -> default.
        let x = SmtExpr::bv_const(10, width);
        let expr = encode_jump_table_switch(x, &cases, 99, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(99));
    }

    #[test]
    fn jump_table_hole_goes_to_default() {
        let width = 8u32;
        // Hole at index 2.
        let cases: Vec<Case> = vec![(0, 1), (1, 2), (3, 3)];
        let x = SmtExpr::bv_const(2, width);
        let expr = encode_jump_table_switch(x, &cases, 99, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(99));
    }

    #[test]
    fn jump_table_nonzero_base_below_min_goes_to_default() {
        let width = 8u32;
        // Base 10, scrutinee 5 underflows via subtraction and fails bounds.
        let (cases, default_id) = dense_switch_cases(10);
        let x = SmtExpr::bv_const(5, width);
        let expr = encode_jump_table_switch(x, &cases, default_id, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(default_id));
    }

    #[test]
    fn binary_search_hits_pivot() {
        let width = 8u32;
        let (cases, default_id) = sparse_switch_cases();
        // Pivot for sorted [5, 20, 42, 77, 100] is 42.
        let x = SmtExpr::bv_const(42, width);
        let expr = encode_binary_search_switch(x, &cases, default_id, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(3));
    }

    #[test]
    fn binary_search_hits_left_subtree() {
        let width = 8u32;
        let (cases, default_id) = sparse_switch_cases();
        let x = SmtExpr::bv_const(5, width); // first case
        let expr = encode_binary_search_switch(x, &cases, default_id, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(1));
    }

    #[test]
    fn binary_search_hits_right_subtree() {
        let width = 8u32;
        let (cases, default_id) = sparse_switch_cases();
        let x = SmtExpr::bv_const(100, width); // last case
        let expr = encode_binary_search_switch(x, &cases, default_id, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(5));
    }

    #[test]
    fn binary_search_falls_through_to_default() {
        let width = 8u32;
        let (cases, default_id) = sparse_switch_cases();
        let x = SmtExpr::bv_const(50, width); // between cases, not in list
        let expr = encode_binary_search_switch(x, &cases, default_id, width);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, crate::smt::EvalResult::Bv(default_id));
    }

    // -----------------------------------------------------------------------
    // Proof-obligation verification tests (exhaustive 8-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_switch_dense_i8() {
        assert_valid(&proof_switch_dense_i8());
    }

    #[test]
    fn test_proof_switch_dense_i8_nonzero_base() {
        assert_valid(&proof_switch_dense_i8_nonzero_base());
    }

    #[test]
    fn test_proof_switch_dense_i8_with_hole() {
        assert_valid(&proof_switch_dense_i8_with_hole());
    }

    #[test]
    fn test_proof_switch_sparse_i8() {
        assert_valid(&proof_switch_sparse_i8());
    }

    #[test]
    fn test_proof_switch_sparse_7case_i8() {
        assert_valid(&proof_switch_sparse_7case_i8());
    }

    // i16 variants run through the statistical sampler for the default eval
    // path. z4-prove covers them symbolically.
    #[test]
    fn test_proof_switch_dense_i16() {
        assert_valid(&proof_switch_dense_i16());
    }

    #[test]
    fn test_proof_switch_sparse_i16() {
        assert_valid(&proof_switch_sparse_i16());
    }

    #[test]
    fn test_all_switch_proofs() {
        for obligation in all_switch_proofs() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Anti-tautology: if we swap case values, the proof MUST fail.
    //
    // This guards against a silent no-op proof where both sides were built
    // from the same underlying expression.
    // -----------------------------------------------------------------------

    #[test]
    fn switch_jt_anti_tautology_wrong_target() {
        let width = 8u32;
        let cases: Vec<Case> = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let x = SmtExpr::var("x", width);

        // Machine side with INCORRECT target permutation. Must produce a
        // counterexample, otherwise the encoder is doing nothing useful.
        let wrong_cases: Vec<Case> = vec![(0, 4), (1, 3), (2, 2), (3, 1)];
        let bogus = ProofObligation {
            name: "anti-tautology: jump-table with permuted targets".to_string(),
            tmir_expr: encode_tmir_switch(x.clone(), &cases, 99, width),
            aarch64_expr: encode_jump_table_switch(x, &wrong_cases, 99, width),
            inputs: vec![("x".to_string(), width)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&bogus);
        assert!(
            matches!(result, VerificationResult::Invalid { .. }),
            "anti-tautology failed: permuted targets were accepted as equivalent"
        );
    }

    #[test]
    fn switch_bst_anti_tautology_wrong_default() {
        let width = 8u32;
        let (cases, _default_id) = sparse_switch_cases();
        let x = SmtExpr::var("x", width);

        // Machine side uses a different default ID. Any scrutinee outside
        // the case set will disagree.
        let bogus = ProofObligation {
            name: "anti-tautology: BST with wrong default".to_string(),
            tmir_expr: encode_tmir_switch(x.clone(), &cases, 9, width),
            aarch64_expr: encode_binary_search_switch(x, &cases, 7, width),
            inputs: vec![("x".to_string(), width)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&bogus);
        assert!(
            matches!(result, VerificationResult::Invalid { .. }),
            "anti-tautology failed: BST with wrong default accepted as equivalent"
        );
    }
}

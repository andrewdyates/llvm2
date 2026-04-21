// Symbolic execution: null-pointer-dereference detector
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Symbolic execution: null-pointer-dereference detector (Phase 1a).
//!
//! This is the minimum-viable starting point for the `fsym` pass described
//! in designs/2026-04-18-integrated-symbolic-execution.md section 3.1.
//! It consumes a pointer expression and a straight-line path condition
//! and reports whether a concrete witness shows UB via null-deref.

use crate::smt::{EvalResult, SmtExpr};
use std::collections::HashMap;

/// Outcome of a single null-deref check.
#[derive(Debug, Clone, PartialEq)]
pub enum FsymVerdict {
    /// The obligation is unsatisfiable under the given path condition:
    /// the pointer cannot be null, no UB on this path.
    Safe,
    /// A concrete witness makes the pointer null while satisfying the
    /// path condition. `witness` maps variable name -> u64 value.
    Ub { witness: HashMap<String, u64> },
    /// The analyzer could neither prove safety nor produce a witness
    /// within the fast evaluator. Escalate to an SMT solver (future).
    Unknown { reason: String },
}

/// Metadata about a memory operation being checked.
#[derive(Debug, Clone)]
pub struct MemOp {
    /// Human-friendly label, e.g. "load bb3/inst12".
    pub label: String,
    /// Symbolic pointer expression (must be a bitvector).
    pub ptr: SmtExpr,
    /// Width of the pointer in bits (typically 64 on AArch64).
    pub ptr_width: u32,
    /// If true, caller has vouched for non-null via a NotNull
    /// annotation; the check is short-circuited to Safe.
    pub has_not_null_annotation: bool,
}

/// Side-condition obligation: the path assumptions that must hold for
/// the UB to be reachable.
#[derive(Debug, Clone)]
pub struct PathContext {
    /// Boolean SmtExpr path guards; their conjunction is the path
    /// condition.
    pub guards: Vec<SmtExpr>,
    /// Candidate concrete environments to try as witnesses. Each entry
    /// maps variable name -> u64 value. If empty, only the single
    /// empty environment is tried.
    pub witness_candidates: Vec<HashMap<String, u64>>,
}

fn no_witness_found() -> FsymVerdict {
    FsymVerdict::Unknown {
        reason: "no witness found in evaluator; escalate to SMT".to_string(),
    }
}

fn guards_hold(guards: &[SmtExpr], env: &HashMap<String, u64>) -> bool {
    guards
        .iter()
        .all(|guard| matches!(guard.try_eval(env), Ok(EvalResult::Bool(true))))
}

fn is_null_ptr(result: EvalResult) -> bool {
    matches!(result, EvalResult::Bv(0) | EvalResult::Bv128(0))
}

/// Check a single memory op against its path context.
///
/// Algorithm:
/// 1. If `op.has_not_null_annotation`, return `Safe` unconditionally.
/// 2. If `op.ptr` is a `BvConst { value, .. }`:
///    - If `value == 0`: return `Ub` with empty witness.
///    - Else: return `Safe`.
/// 3. Otherwise, try each candidate environment. If no candidates are
///    supplied, try a single empty environment:
///    - Evaluate each path guard; if any guard is not `Bool(true)`, skip
///      this candidate.
///    - If all guards are `Bool(true)` and `op.ptr` evaluates to `Bv(0)`
///      or `Bv128(0)`, return `Ub` with that witness.
/// 4. If no candidate triggered UB, return `Unknown { reason:
///    "no witness found in evaluator; escalate to SMT" }`.
///
/// Path guards or pointer expressions that reference variables missing
/// from the environment cause `try_eval` to return an error; treat
/// such candidates as "not applicable" and continue to the next.
pub fn check_null_deref(op: &MemOp, ctx: &PathContext) -> FsymVerdict {
    if op.has_not_null_annotation {
        return FsymVerdict::Safe;
    }

    if let SmtExpr::BvConst { value, .. } = &op.ptr {
        return if *value == 0 {
            FsymVerdict::Ub {
                witness: HashMap::new(),
            }
        } else {
            FsymVerdict::Safe
        };
    }

    let empty_env = HashMap::new();
    let candidates: &[HashMap<String, u64>] = if ctx.witness_candidates.is_empty() {
        std::slice::from_ref(&empty_env)
    } else {
        ctx.witness_candidates.as_slice()
    };

    for env in candidates {
        if !guards_hold(&ctx.guards, env) {
            continue;
        }

        let Ok(result) = op.ptr.try_eval(env) else {
            continue;
        };

        if is_null_ptr(result) {
            return FsymVerdict::Ub {
                witness: env.clone(),
            };
        }
    }

    no_witness_found()
}

/// Scan a collection of memory ops sharing one path context; return
/// each verdict tagged with its op label.
#[cfg(feature = "fsym")]
pub fn run_null_deref_scan(ops: &[MemOp], ctx: &PathContext) -> Vec<(String, FsymVerdict)> {
    ops.iter()
        .map(|op| (op.label.clone(), check_null_deref(op, ctx)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{check_null_deref, FsymVerdict, MemOp, PathContext};
    use crate::smt::SmtExpr;
    use std::collections::HashMap;

    fn op(ptr: SmtExpr) -> MemOp {
        MemOp {
            label: "memop".to_string(),
            ptr,
            ptr_width: 64,
            has_not_null_annotation: false,
        }
    }

    fn ctx(guards: Vec<SmtExpr>, witness_candidates: Vec<HashMap<String, u64>>) -> PathContext {
        PathContext {
            guards,
            witness_candidates,
        }
    }

    fn unknown() -> FsymVerdict {
        FsymVerdict::Unknown {
            reason: "no witness found in evaluator; escalate to SMT".to_string(),
        }
    }

    #[test]
    fn not_null_annotation_short_circuits() {
        let mut mem_op = op(SmtExpr::bv_const(0, 64));
        mem_op.has_not_null_annotation = true;
        assert_eq!(check_null_deref(&mem_op, &ctx(vec![], vec![])), FsymVerdict::Safe);
    }

    #[test]
    fn concrete_null_is_ub() {
        assert_eq!(
            check_null_deref(&op(SmtExpr::bv_const(0, 64)), &ctx(vec![], vec![])),
            FsymVerdict::Ub {
                witness: HashMap::new(),
            }
        );
    }

    #[test]
    fn concrete_nonnull_is_safe() {
        assert_eq!(
            check_null_deref(&op(SmtExpr::bv_const(0x1000, 64)), &ctx(vec![], vec![])),
            FsymVerdict::Safe
        );
    }

    #[test]
    fn symbolic_ptr_null_witness_found() {
        let witness = HashMap::from([(String::from("p"), 0_u64)]);
        assert_eq!(
            check_null_deref(&op(SmtExpr::var("p", 64)), &ctx(vec![], vec![witness.clone()])),
            FsymVerdict::Ub { witness }
        );
    }

    #[test]
    fn symbolic_ptr_nonnull_candidate_yields_unknown() {
        let witness = HashMap::from([(String::from("p"), 42_u64)]);
        assert_eq!(
            check_null_deref(&op(SmtExpr::var("p", 64)), &ctx(vec![], vec![witness])),
            unknown()
        );
    }

    #[test]
    fn path_guard_blocks_witness() {
        let guard = SmtExpr::var("g", 1).eq_expr(SmtExpr::bv_const(1, 1));
        let witness = HashMap::from([(String::from("p"), 0_u64), (String::from("g"), 0_u64)]);
        assert_eq!(
            check_null_deref(&op(SmtExpr::var("p", 64)), &ctx(vec![guard], vec![witness])),
            unknown()
        );
    }

    #[test]
    fn path_guard_allows_witness() {
        let guard = SmtExpr::var("g", 1).eq_expr(SmtExpr::bv_const(1, 1));
        let witness = HashMap::from([(String::from("p"), 0_u64), (String::from("g"), 1_u64)]);
        assert_eq!(
            check_null_deref(&op(SmtExpr::var("p", 64)), &ctx(vec![guard], vec![witness.clone()])),
            FsymVerdict::Ub { witness }
        );
    }
}

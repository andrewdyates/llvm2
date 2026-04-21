// llvm2-dialect - Pass trait + Legality
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Pass infrastructure: the [`Pass`] trait, [`Legality`] declarations, and a
//! `validate_legality` helper for post-pass assertions.

use llvm2_ir::Type;

use crate::dialect::{Arity, OpDef, TypeConstraint};
use crate::id::{DialectId, DialectOpId};
use crate::module::DialectModule;
use crate::op::DialectOp;

/// Dialect-level pass legality declaration.
///
/// Passes declare which dialects they consume and which they produce so the
/// pipeline can statically verify chain-ability and tests can assert that a
/// pass actually removed every op from an input dialect.
#[derive(Debug, Clone, Default)]
pub struct Legality {
    /// Input ops must belong to one of these dialects.
    pub accepts: Vec<DialectId>,
    /// Output ops must belong to one of these dialects.
    pub produces: Vec<DialectId>,
    /// Specific ops that must not appear in the output.
    pub illegal_ops: Vec<DialectOpId>,
}

impl Legality {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn accepts(mut self, d: DialectId) -> Self {
        self.accepts.push(d);
        self
    }
    pub fn produces(mut self, d: DialectId) -> Self {
        self.produces.push(d);
        self
    }
    pub fn forbid(mut self, op: DialectOpId) -> Self {
        self.illegal_ops.push(op);
        self
    }
}

/// Error returned by a [`Pass::run`] implementation.
#[derive(Debug, Clone)]
pub enum PassError {
    /// The pass encountered an op it does not know how to handle.
    UnsupportedOp(DialectOpId, String),
    /// The pass produced output that violates its declared legality.
    LegalityViolation(String),
    /// Generic error carrying a message.
    Other(String),
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassError::UnsupportedOp(id, ctx) => {
                write!(f, "pass does not support op {:?} ({})", id, ctx)
            }
            PassError::LegalityViolation(m) => write!(f, "legality violation: {}", m),
            PassError::Other(m) => write!(f, "{}", m),
        }
    }
}

impl std::error::Error for PassError {}

/// A dialect-level pass.
pub trait Pass {
    fn name(&self) -> &'static str;
    fn legality(&self) -> Legality;
    fn run(&self, module: &mut DialectModule) -> Result<(), PassError>;
}

/// Boolean predicate form of [`validate_type_constraints_report`].
///
/// Returns `true` iff the op's actual operand and result types satisfy the
/// per-position [`TypeConstraint`] slices declared on `def`. Empty constraint
/// slices are a legacy opt-out and are treated as "no constraint".
///
/// Note: operand types are only checkable when an SSA value->type environment
/// is supplied. Callers outside a module walk (e.g. synthetic tests) can pass
/// a lookup that returns `None` for all ids; in that case operand-side
/// constraints are skipped and only result-side constraints are enforced.
pub fn validate_type_constraints(op: &DialectOp, def: &OpDef) -> bool {
    let empty = std::collections::HashMap::new();
    validate_type_constraints_with_env(op, def, &empty).is_ok()
}

/// Structured version of [`validate_type_constraints`].
///
/// `value_types` maps SSA `ValueId`s to concrete `Type`s (built from function
/// parameters plus prior op results). Positions whose operand type cannot be
/// resolved are skipped on the operand side; the result side is always
/// checked because `DialectOp` carries the result types inline.
pub fn validate_type_constraints_with_env(
    op: &DialectOp,
    def: &OpDef,
    value_types: &std::collections::HashMap<crate::id::ValueId, Type>,
) -> Result<(), PassError> {
    // Arity check first — a mismatch makes positional constraint-slice access
    // ambiguous.
    if !def.num_operands.accepts(op.operands.len()) {
        return Err(PassError::LegalityViolation(format!(
            "op {} has {} operand(s) but arity {:?} required",
            def.name,
            op.operands.len(),
            def.num_operands
        )));
    }
    if !def.num_results.accepts(op.results.len()) {
        return Err(PassError::LegalityViolation(format!(
            "op {} has {} result(s) but arity {:?} required",
            def.name,
            op.results.len(),
            def.num_results
        )));
    }

    // Fixed-arity ops must have constraint slices whose length matches their
    // arity (or be empty — the legacy opt-out). Catching this here makes
    // dialect-definition bugs surface at the first legality check rather than
    // silently mis-matching by index.
    check_slice_shape("operand", def.name, def.operand_types, def.num_operands)?;
    check_slice_shape("result", def.name, def.result_types, def.num_results)?;

    // Build a positional operand-type list. `None` entries mark operand slots
    // whose type could not be resolved from the supplied env. This list is
    // passed directly to `TypeConstraint::accepts` so that `SameAs(idx)`
    // resolves against the *original* operand position — collapsing `None`s
    // out would silently shift indices and either false-positive or, worse,
    // silently accept an op whose `SameAs` target was unresolved (see #410).
    let resolved_operand_tys: Vec<Option<Type>> = op
        .operands
        .iter()
        .map(|v| value_types.get(v).cloned())
        .collect();

    // --- Operand side ---
    if !def.operand_types.is_empty() {
        for (idx, maybe_ty) in resolved_operand_tys.iter().enumerate() {
            let Some(ty) = maybe_ty else { continue };
            let constraint = constraint_for(def.operand_types, def.num_operands, idx);
            if !constraint.accepts(ty, &resolved_operand_tys) {
                return Err(PassError::LegalityViolation(format!(
                    "op {} operand {} ({:?}) does not satisfy constraint {:?}",
                    def.name, idx, ty, constraint
                )));
            }
        }
    }

    // --- Result side ---
    if !def.result_types.is_empty() {
        for (idx, (_, ty)) in op.results.iter().enumerate() {
            let constraint = constraint_for(def.result_types, def.num_results, idx);
            if !constraint.accepts(ty, &resolved_operand_tys) {
                return Err(PassError::LegalityViolation(format!(
                    "op {} result {} ({:?}) does not satisfy constraint {:?}",
                    def.name, idx, ty, constraint
                )));
            }
        }
    }

    Ok(())
}

fn check_slice_shape(
    side: &'static str,
    op_name: &'static str,
    slice: &'static [TypeConstraint],
    arity: Arity,
) -> Result<(), PassError> {
    if slice.is_empty() {
        return Ok(());
    }
    match arity {
        Arity::Fixed(n) => {
            if slice.len() != n as usize {
                return Err(PassError::LegalityViolation(format!(
                    "op {} {}_types slice length {} != fixed arity {}",
                    op_name,
                    side,
                    slice.len(),
                    n
                )));
            }
        }
        Arity::Variadic(_) => {
            // For variadic ops the slice either has length 1 (applied to
            // every operand/result) or length >= the maximum expected count.
            // We can't enforce the latter without seeing the runtime count,
            // so accept any non-empty slice here and let constraint_for
            // handle positional lookup.
        }
    }
    Ok(())
}

fn constraint_for(
    slice: &'static [TypeConstraint],
    arity: Arity,
    idx: usize,
) -> &'static TypeConstraint {
    match arity {
        Arity::Fixed(_) => &slice[idx],
        Arity::Variadic(_) => {
            if slice.len() == 1 {
                &slice[0]
            } else {
                // For multi-element variadic slices, clamp to the last entry
                // if the runtime count exceeds the declared positions. This
                // matches MLIR's "variadic suffix" convention.
                slice.get(idx).unwrap_or(&slice[slice.len() - 1])
            }
        }
    }
}

/// Assert that every op in `module` satisfies the legality declaration
/// symmetrically:
///
/// - every op's dialect must be in `legality.accepts` (if non-empty),
/// - every op's dialect must be in `legality.produces` (if non-empty),
/// - no op may appear in `legality.illegal_ops`,
/// - when the op's `OpDef` declares non-empty `operand_types`/`result_types`
///   slices, its per-position [`TypeConstraint`]s are checked.
///
/// Returns `Err` with a description of the first offender. Design doc §5
/// requires both sides of the accept/produce contract to "connect end-to-end"
/// — so this helper enforces both input and output legality. Passes or tests
/// that only care about one side should populate only that side of the
/// [`Legality`] struct; empty sets are treated as "no constraint on this side."
pub fn validate_legality(
    module: &DialectModule,
    legality: &Legality,
) -> Result<(), PassError> {
    use std::collections::HashMap;

    for func in &module.functions {
        // Per-function value->type environment from parameters + op results.
        let mut value_types: HashMap<crate::id::ValueId, Type> = HashMap::new();
        for (v, ty) in &func.params {
            value_types.insert(*v, ty.clone());
        }
        for op in &func.ops {
            for (v, ty) in &op.results {
                value_types.insert(*v, ty.clone());
            }
        }

        for op in func.iter_ops() {
            if !legality.accepts.is_empty()
                && !legality.accepts.contains(&op.op.dialect)
            {
                let name = module
                    .resolve(op.op)
                    .map(|d| d.name)
                    .unwrap_or("<unregistered>");
                return Err(PassError::LegalityViolation(format!(
                    "op {:?} ({}) belongs to dialect {:?} which is not in accepts set {:?}",
                    op.op, name, op.op.dialect, legality.accepts
                )));
            }
            if !legality.produces.is_empty()
                && !legality.produces.contains(&op.op.dialect)
            {
                let name = module
                    .resolve(op.op)
                    .map(|d| d.name)
                    .unwrap_or("<unregistered>");
                return Err(PassError::LegalityViolation(format!(
                    "op {:?} ({}) belongs to dialect {:?} which is not in produces set {:?}",
                    op.op, name, op.op.dialect, legality.produces
                )));
            }
            if legality.illegal_ops.contains(&op.op) {
                let name = module
                    .resolve(op.op)
                    .map(|d| d.name)
                    .unwrap_or("<unregistered>");
                return Err(PassError::LegalityViolation(format!(
                    "op {:?} ({}) is explicitly forbidden",
                    op.op, name
                )));
            }
            if let Some(def) = module.resolve(op.op) {
                validate_type_constraints_with_env(op, def, &value_types)?;
            }
        }
    }
    Ok(())
}

// llvm2-codegen/dialect_pipeline.rs - Pre-adapter dialect lowering
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: issue #433, tMIR #428. Wires tMIR's dialect framework into
// the LLVM2 compilation pipeline so that upstream dialects (currently the
// canonical `verif` example; future `tla2-verif`) are rewritten out of
// `Inst::DialectOp` before the tMIR->LIR adapter runs.

//! Pre-adapter dialect lowering.
//!
//! Runs `tmir::dialect::lower_module` against a [`DialectRegistry`] with at
//! least one dialect registered (per LLVM2 issue #433 / tMIR #428). The goal
//! is to catch unknown `DialectOp` instances at a known location *before*
//! they reach [`llvm2_lower::translate_module`] — the adapter has no
//! `DialectOp` handler and would otherwise emit a translation error deep in
//! the ISel stack.
//!
//! The registry currently ships with tMIR's canonical `VerifDialect` (the
//! example used by `tla2` and referenced by #428 §"What LLVM2 consumers
//! should watch for"). Additional dialects — notably a dedicated
//! `tla2-verif` dialect — can be registered by future work without touching
//! the rest of the pipeline. See `default_registry()` for the hook.

use thiserror::Error;

/// Error from pre-adapter dialect lowering.
#[derive(Debug, Error)]
pub enum DialectPipelineError {
    /// tMIR dialect framework reported a lowering failure (fixpoint not
    /// reached, pass returned an error, or an op references an unknown
    /// dialect).
    #[error("tMIR dialect lowering failed: {0}")]
    Lowering(String),
}

/// Maximum number of lowering fixpoint iterations.
///
/// tMIR's `lower_module` iterates passes until no rewrites fire, or this
/// limit is hit. Empirically the `verif` example converges in at most 3
/// iterations; 16 is a conservative ceiling that still cuts off runaway
/// rewriters.
pub const MAX_LOWERING_ITERS: usize = 16;

/// Build the default [`tmir::dialect::DialectRegistry`] used by the
/// LLVM2 compile path.
///
/// Includes tMIR's canonical `VerifDialect` example so that modules
/// emitting `verif.bfs_step` / `verif.frontier_drain` /
/// `verif.fingerprint_batch` round-trip through the pipeline.
/// `tla2-verif`, when it lands upstream, will be registered here.
pub fn default_registry() -> tmir::dialect::DialectRegistry {
    let mut reg = tmir::dialect::DialectRegistry::new();
    // Feature-gated on tMIR's `dialect-verif-example`; enabled in
    // `[workspace.dependencies]` via `features = ["dialect-verif-example"]`.
    reg.register(Box::new(
        tmir::dialect::examples::verif::VerifDialect,
    ));
    reg
}

/// Run pre-adapter dialect lowering on `module`.
///
/// Uses [`default_registry`] to progressively rewrite any
/// `Inst::DialectOp` in the module. Modules with no dialect ops are a
/// no-op (the registry's validate + lower paths short-circuit).
///
/// Returns the number of rewrites applied, for tracing / diagnostics.
pub fn lower_dialects(
    module: &mut tmir::Module,
) -> Result<usize, DialectPipelineError> {
    let registry = default_registry();
    // Validate that every DialectOp in the module references a registered
    // dialect. Unknown dialects are the hard error tMIR#428 §"Dialect
    // framework" calls out — this surfaces it at a known pipeline stage
    // rather than letting the adapter skip over an unrecognized op.
    registry
        .validate_module(module)
        .map_err(|e| DialectPipelineError::Lowering(format!("{e}")))?;
    let result = registry
        .lower(module, MAX_LOWERING_ITERS)
        .map_err(|e| DialectPipelineError::Lowering(format!("{e}")))?;
    Ok(result.rewrites_applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tmir::dialect::{examples::verif, DialectInst};
    use tmir::inst::Inst;
    use tmir::node::InstrNode;
    use tmir::ty::Ty;
    use tmir::value::{BlockId, FuncId, FuncTyId, ValueId};
    use tmir::{Block, Function, Module};

    fn mk_func_with_body(body: Vec<InstrNode>) -> Function {
        let mut func = Function::new(
            FuncId::new(0),
            "test",
            FuncTyId::new(0),
            BlockId::new(0),
        );
        let mut block = Block::new(BlockId::new(0));
        block.body = body;
        func.blocks.push(block);
        func
    }

    /// Empty module lowers cleanly with zero rewrites.
    #[test]
    fn empty_module_is_noop() {
        let mut module = Module::new("empty");
        let n = lower_dialects(&mut module).expect("empty lower");
        assert_eq!(n, 0);
    }

    /// Module with a `verif.frontier_drain` op lowers (pass erases it).
    #[test]
    fn frontier_drain_erased() {
        let mut module = Module::new("test");
        let func = mk_func_with_body(vec![InstrNode::new(Inst::DialectOp(
            Box::new(verif::frontier_drain(ValueId::new(0))),
        ))]);
        module.functions.push(func);

        // Before lowering: one DialectOp body entry.
        assert_eq!(module.functions[0].blocks[0].body.len(), 1);
        assert!(matches!(
            module.functions[0].blocks[0].body[0].inst,
            Inst::DialectOp(_)
        ));

        let n = lower_dialects(&mut module).expect("lower");
        // FrontierDrainErase deletes the op.
        assert!(n >= 1, "expected at least one rewrite, got {n}");
        assert_eq!(module.functions[0].blocks[0].body.len(), 0);
    }

    /// Unknown dialect (not registered) is rejected up front.
    #[test]
    fn unknown_dialect_rejected() {
        let mut module = Module::new("bad");
        let func = mk_func_with_body(vec![InstrNode::new(Inst::DialectOp(
            Box::new(
                DialectInst::new("no_such_dialect", "op")
                    .with_result_ty(Ty::I32),
            ),
        ))]);
        module.functions.push(func);

        let err = lower_dialects(&mut module)
            .expect_err("unknown dialect must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("no_such_dialect"),
            "error message missing dialect name: {msg}"
        );
    }
}

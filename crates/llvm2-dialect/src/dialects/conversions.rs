// llvm2-dialect - PoC conversion patterns: verif -> tmir, tmir -> machir
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! PoC [`ConversionPattern`] implementations.
//!
//! * [`VerifToTmir`] — expands `verif.fingerprint_batch_stub(ptr, len)` into
//!   an XOR-based `tmir.*` sequence. The stub semantics are intentionally
//!   trivial; the conversion pipeline is what's being tested.
//! * [`TmirToMachir`] — lowers `tmir.add/xor/const/ret` to the corresponding
//!   `machir.*` ops.

use llvm2_ir::Type;

use crate::conversion::{ConversionDriver, ConversionError, ConversionPattern, Rewriter};
use crate::dialects::{machir, tmir, verif};
use crate::id::{DialectId, DialectOpId};
use crate::op::{Attribute, DialectOp};

/// Magic constant mixed into the stub fingerprint. Only purpose is to make the
/// end-to-end test observable — it's not cryptographically meaningful.
pub const FINGERPRINT_STUB_MAGIC: u64 = 0xA5A5_A5A5_A5A5_A5A5;

// ---------------------------------------------------------------------------
// verif -> tmir
// ---------------------------------------------------------------------------

pub struct VerifToTmir {
    pub verif_id: DialectId,
    pub tmir_id: DialectId,
}

impl ConversionPattern for VerifToTmir {
    fn source_op(&self) -> DialectOpId {
        DialectOpId::new(self.verif_id, verif::FINGERPRINT_BATCH_STUB)
    }

    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError> {
        if op.operands.len() != 2 || op.results.len() != 1 {
            return Err(ConversionError::RewriteFailed(
                "verif.fingerprint_batch_stub expects (ptr, len) -> i64".to_string(),
            ));
        }

        // Operands arriving via `rewriter` are already translated from the
        // source function's value space into the destination function's.
        let ptr = op.operands[0];
        let len = op.operands[1];

        // %magic = tmir.const {value = FINGERPRINT_STUB_MAGIC}
        let magic = rewriter.alloc_value();
        rewriter.emit(
            DialectOpId::new(self.tmir_id, tmir::TMIR_CONST),
            vec![(magic, Type::I64)],
            vec![],
            vec![("value".to_string(), Attribute::U64(FINGERPRINT_STUB_MAGIC))],
            op.source,
        );

        // %t1 = tmir.xor ptr, len
        let t1 = rewriter.alloc_value();
        rewriter.emit(
            DialectOpId::new(self.tmir_id, tmir::TMIR_XOR),
            vec![(t1, Type::I64)],
            vec![ptr, len],
            vec![],
            op.source,
        );

        // %t2 = tmir.xor t1, magic
        let t2 = rewriter.alloc_value();
        rewriter.emit(
            DialectOpId::new(self.tmir_id, tmir::TMIR_XOR),
            vec![(t2, Type::I64)],
            vec![t1, magic],
            vec![],
            op.source,
        );

        // Bind the source result to %t2 so downstream ops (e.g. the return)
        // will reference it.
        let (src_result, _) = op.results[0];
        rewriter.bind_result(src_result, t2);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// tmir -> machir
// ---------------------------------------------------------------------------

pub struct TmirConstToMachir {
    pub tmir_id: DialectId,
    pub machir_id: DialectId,
}

impl ConversionPattern for TmirConstToMachir {
    fn source_op(&self) -> DialectOpId {
        DialectOpId::new(self.tmir_id, tmir::TMIR_CONST)
    }

    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError> {
        if op.results.len() != 1 {
            return Err(ConversionError::RewriteFailed(
                "tmir.const must have exactly one result".to_string(),
            ));
        }
        let (src_result, ty) = op.results[0].clone();
        let dst = rewriter.alloc_value();
        // Attribute passthrough: carry the `value` attribute so the MachIR
        // layer can materialize the constant.
        let mut attrs = vec![];
        if let Some(v) = op.attr("value").cloned() {
            attrs.push(("value".to_string(), v));
        }
        rewriter.emit(
            DialectOpId::new(self.machir_id, machir::MACHIR_MOVZ_I64),
            vec![(dst, ty)],
            vec![],
            attrs,
            op.source,
        );
        rewriter.bind_result(src_result, dst);
        Ok(())
    }
}

pub struct TmirAddToMachir {
    pub tmir_id: DialectId,
    pub machir_id: DialectId,
}

impl ConversionPattern for TmirAddToMachir {
    fn source_op(&self) -> DialectOpId {
        DialectOpId::new(self.tmir_id, tmir::TMIR_ADD)
    }

    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError> {
        binary_to_machir(op, rewriter, self.machir_id, machir::MACHIR_ADD_RR, "tmir.add")
    }
}

pub struct TmirXorToMachir {
    pub tmir_id: DialectId,
    pub machir_id: DialectId,
}

impl ConversionPattern for TmirXorToMachir {
    fn source_op(&self) -> DialectOpId {
        DialectOpId::new(self.tmir_id, tmir::TMIR_XOR)
    }

    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError> {
        binary_to_machir(op, rewriter, self.machir_id, machir::MACHIR_EOR_RR, "tmir.xor")
    }
}

pub struct TmirRetToMachir {
    pub tmir_id: DialectId,
    pub machir_id: DialectId,
}

impl ConversionPattern for TmirRetToMachir {
    fn source_op(&self) -> DialectOpId {
        DialectOpId::new(self.tmir_id, tmir::TMIR_RET)
    }

    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError> {
        rewriter.emit(
            DialectOpId::new(self.machir_id, machir::MACHIR_RET),
            vec![],
            op.operands.clone(),
            vec![],
            op.source,
        );
        Ok(())
    }
}

fn binary_to_machir(
    op: &DialectOp,
    rewriter: &mut Rewriter<'_>,
    machir_id: DialectId,
    machir_op: crate::id::OpCode,
    name: &'static str,
) -> Result<(), ConversionError> {
    if op.operands.len() != 2 || op.results.len() != 1 {
        return Err(ConversionError::RewriteFailed(format!(
            "{} expects (a, b) -> i64",
            name
        )));
    }
    let a = op.operands[0];
    let b = op.operands[1];
    let (src_result, ty) = op.results[0].clone();
    let dst = rewriter.alloc_value();
    rewriter.emit(
        DialectOpId::new(machir_id, machir_op),
        vec![(dst, ty)],
        vec![a, b],
        vec![],
        op.source,
    );
    rewriter.bind_result(src_result, dst);
    Ok(())
}

// Aggregate helpers --------------------------------------------------------

/// Build a driver that runs the full `verif.* -> tmir.*` conversion.
pub fn verif_to_tmir_driver(verif_id: DialectId, tmir_id: DialectId) -> ConversionDriver {
    let mut d = ConversionDriver::new();
    d.register(Box::new(VerifToTmir { verif_id, tmir_id }));
    d
}

/// Build a driver that runs the full `tmir.* -> machir.*` conversion.
pub fn tmir_to_machir_driver(tmir_id: DialectId, machir_id: DialectId) -> ConversionDriver {
    let mut d = ConversionDriver::new();
    d.register(Box::new(TmirConstToMachir { tmir_id, machir_id }));
    d.register(Box::new(TmirAddToMachir { tmir_id, machir_id }));
    d.register(Box::new(TmirXorToMachir { tmir_id, machir_id }));
    d.register(Box::new(TmirRetToMachir { tmir_id, machir_id }));
    d
}

/// Convenience: register all three PoC dialects into `registry` and return
/// their assigned ids as `(verif, tmir, machir)`.
pub fn register_all(
    registry: &mut crate::registry::DialectRegistry,
) -> (DialectId, DialectId, DialectId) {
    let verif_id = registry.register(Box::new(verif::VerifDialect::new()));
    let tmir_id = registry.register(Box::new(tmir::TmirDialect::new()));
    let machir_id = registry.register(Box::new(machir::MachirDialect::new()));
    (verif_id, tmir_id, machir_id)
}

// Re-exports to match the expectation set by `dialects::mod.rs` `pub use`.
pub use TmirConstToMachir as TmirToMachir;

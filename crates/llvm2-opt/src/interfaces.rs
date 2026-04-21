// llvm2-opt - Op interface catalog
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Op interface catalog for optimization passes.
//!
//! Interfaces are boolean / classifier queries on [`MachInst`] or a bare
//! opcode. They let passes say "any pure instruction" or "any reduction
//! op" instead of enumerating every opcode.
//!
//! The interfaces in this module mirror the MLIR op-interface concept and
//! back the [`rewrite`](crate::rewrite) framework's interface-aware
//! constraints. See `designs/2026-04-18-rewrite-and-interfaces.md`.
//!
//! # Interfaces
//!
//! | Interface | Meaning | Source of truth |
//! |-----------|---------|-----------------|
//! | `Pure` | No memory effects, no implicit flag writes | [`effects::opcode_effect`] + `writes_flags` |
//! | `HasParallelism` | Data-parallel (independent sub-work) | Opcode table |
//! | `IsReduction` | Many inputs, one output via associative fold | Opcode table |
//! | `IsFold` | Left/right-fold pattern | Opcode table |
//! | `IsMap` | Per-element function, no cross-element deps | Opcode table |
//! | `HasBoundedLoops` | Instruction represents a bounded loop | [`ProofAnnotation`] (future) |
//! | `DivergenceClass` | Uniform / Divergent / Unknown | Opcode + proof |
//!
//! # Design notes
//!
//! - `is_pure()` is stricter than [`MemoryEffect::is_pure`]: a `Pure`
//!   instruction has both no memory effects *and* does not write implicit
//!   condition flags. A pure-value but flag-writing `ADDS` is not `Pure`
//!   for rewrite purposes because a later flag reader could observe it.
//! - `HasBoundedLoops` is wired to [`ProofAnnotation`]. The enum in
//!   `llvm2-ir` does not yet include a `BoundedLoop` variant (tracked in
//!   #390 item 3); we return `false` for every opcode today. Consumers
//!   can call the API and upgrade when the annotation lands.

use llvm2_ir::{AArch64Opcode, MachInst};

use crate::effects;

// ---------------------------------------------------------------------------
// DivergenceClass
// ---------------------------------------------------------------------------

/// Divergence classification for an instruction.
///
/// Used by heterogeneous-compute passes (GPU / ANE) to decide whether a
/// computation is safe to execute in lock-step across lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DivergenceClass {
    /// All lanes produce the same value (loop-invariant within a warp).
    Uniform,
    /// Different lanes may produce different values.
    Divergent,
    /// Not yet analyzed (safe default; pass should treat as `Divergent`).
    Unknown,
}

// ---------------------------------------------------------------------------
// OpInterfaces trait
// ---------------------------------------------------------------------------

/// Interface queries on an instruction or opcode.
///
/// Implemented for both [`MachInst`] and [`AArch64Opcode`] so that passes
/// can query without materializing a dummy instruction.
pub trait OpInterfaces {
    /// No memory side effects AND does not write implicit flags.
    fn is_pure(&self) -> bool;

    /// The operation can be split into independent sub-work (data-parallel
    /// map/reduce style).
    fn has_parallelism(&self) -> bool;

    /// Single output computed from many inputs via an associative fold
    /// (e.g., horizontal add across a vector).
    fn is_reduction(&self) -> bool;

    /// Left- or right-fold pattern (a reduction is a fold, but not every
    /// fold is a reduction).
    fn is_fold(&self) -> bool;

    /// Per-element function with no cross-element dependency.
    fn is_map(&self) -> bool;

    /// This instruction represents a bounded loop (iteration count proved
    /// by tMIR).
    fn has_bounded_loops(&self) -> bool;

    /// Divergence class for heterogeneous-compute lowering.
    fn divergence_class(&self) -> DivergenceClass;
}

// ---------------------------------------------------------------------------
// Opcode-level implementation (AArch64)
// ---------------------------------------------------------------------------

impl OpInterfaces for AArch64Opcode {
    fn is_pure(&self) -> bool {
        use AArch64Opcode::*;
        let op = *self;

        // Memory-effect gate: loads, stores, calls, and barriers are never pure.
        if !effects::opcode_effect(op).is_pure() {
            return false;
        }
        // Implicit flag writers (CMP/TST/ADDS/SUBS/FCMP) are not pure: a later
        // flag reader would observe them.
        if effects::writes_flags(op) {
            return false;
        }
        // Implicit flag readers (CSEL/CSET/CSINC/CSINV/CSNEG, ADC/SBC) depend
        // on NZCV state that is not in the explicit operand list. Reordering
        // them across a flag writer is a silent miscompile. `reads_flags` is
        // the single source of truth and now covers ADC/SBC as well (#409).
        if effects::reads_flags(op) {
            return false;
        }
        // Tied def-use (MOVK, BFM) reads its own destination register as an
        // implicit source. An SSA "pure" contract would allow free reordering,
        // which corrupts the MOVZ/MOVK chain or BFM insert (see #366, #382,
        // #408). `has_tied_def_use` is the single source of truth and now
        // covers BFM as well.
        if effects::has_tied_def_use(op) {
            return false;
        }
        // Control flow: branches, returns, and indirect branches are never
        // pure regardless of memory-effect classification. `opcode_effect`
        // marks them `Pure` for memory-alias purposes; the interface contract
        // requires control-flow-freedom as well.
        if matches!(op, B | BCond | Bcc | Cbz | Cbnz | Tbz | Tbnz | Br | Ret) {
            return false;
        }
        // Trap pseudo-instructions alter control flow conditionally; they are
        // side-effecting even though `opcode_effect` classifies them `Pure`
        // (they access no memory).
        if matches!(
            op,
            TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange
        ) {
            return false;
        }
        // Trapping FP: FDIV and FSQRT can raise domain/divide-by-zero
        // exceptions when FP exceptions are enabled. Conservative: not pure.
        if matches!(op, FdivRR | FsqrtRR | NeonFdivV) {
            return false;
        }
        // Integer division: SDiv/UDiv trap on divide-by-zero (handled by
        // TrapDivZero in practice, but the raw opcode is still trapping).
        if matches!(op, SDiv | UDiv) {
            return false;
        }
        true
    }

    fn has_parallelism(&self) -> bool {
        use AArch64Opcode::*;
        // NEON element-wise ops distribute trivially across lanes.
        matches!(
            *self,
            NeonAddV
                | NeonSubV
                | NeonMulV
                | NeonFaddV
                | NeonFsubV
                | NeonFmulV
                | NeonFdivV
                | NeonAndV
                | NeonOrrV
                | NeonEorV
                | NeonBicV
                | NeonNotV
                | NeonCmeqV
                | NeonCmgtV
                | NeonCmgeV
                | NeonDupElem
                | NeonDupGen
        )
    }

    fn is_reduction(&self) -> bool {
        // No dedicated AArch64 reduction opcodes in today's model. NEON
        // horizontal add (ADDV, FADDP) will be added by the heterogeneous
        // compute work (#390). The interface exists so that pass code can
        // be written today.
        false
    }

    fn is_fold(&self) -> bool {
        // Every reduction is also a fold. When reduction opcodes land,
        // they will flip this on too.
        self.is_reduction()
    }

    fn is_map(&self) -> bool {
        // NEON element-wise ops are the canonical "map" pattern.
        self.has_parallelism()
    }

    fn has_bounded_loops(&self) -> bool {
        // Today's ProofAnnotation enum has no BoundedLoop variant. Wiring
        // is in place; returns false until the annotation lands.
        false
    }

    fn divergence_class(&self) -> DivergenceClass {
        // Conservative: we have no divergence analysis yet. Return
        // Unknown; GPU lowering will upgrade specific opcodes later.
        DivergenceClass::Unknown
    }
}

// ---------------------------------------------------------------------------
// Instruction-level implementation
// ---------------------------------------------------------------------------

impl OpInterfaces for MachInst {
    fn is_pure(&self) -> bool {
        self.opcode.is_pure()
    }

    fn has_parallelism(&self) -> bool {
        self.opcode.has_parallelism()
    }

    fn is_reduction(&self) -> bool {
        self.opcode.is_reduction()
    }

    fn is_fold(&self) -> bool {
        self.opcode.is_fold()
    }

    fn is_map(&self) -> bool {
        self.opcode.is_map()
    }

    fn has_bounded_loops(&self) -> bool {
        // Proof-driven. The current ProofAnnotation enum does not have a
        // BoundedLoop variant — when tMIR ships one (#390 item 3), adding
        // a match arm here flips the interface on for annotated ops.
        let _ = self.proof; // suppress dead-read warning once populated
        false
    }

    fn divergence_class(&self) -> DivergenceClass {
        self.opcode.divergence_class()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{AArch64Target, MachOperand, TargetInfo};

    #[test]
    fn pure_arithmetic_is_pure() {
        assert!(AArch64Opcode::AddRR.is_pure());
        assert!(AArch64Opcode::SubRR.is_pure());
        assert!(AArch64Opcode::MulRR.is_pure());
        assert!(AArch64Opcode::AndRR.is_pure());
        assert!(AArch64Opcode::Neg.is_pure());
    }

    #[test]
    fn moves_are_pure() {
        assert!(AArch64Opcode::MovR.is_pure());
        assert!(AArch64Opcode::MovI.is_pure());
        assert!(AArch64Opcode::Movz.is_pure());
    }

    #[test]
    fn flag_writers_are_not_pure() {
        // CMP/ADDS write NZCV; a later CSEL/BCond would observe them.
        assert!(!AArch64Opcode::CmpRR.is_pure());
        assert!(!AArch64Opcode::CmpRI.is_pure());
        assert!(!AArch64Opcode::AddsRR.is_pure());
        assert!(!AArch64Opcode::SubsRI.is_pure());
        assert!(!AArch64Opcode::Tst.is_pure());
        assert!(!AArch64Opcode::Fcmp.is_pure());
    }

    #[test]
    fn memory_ops_are_not_pure() {
        assert!(!AArch64Opcode::LdrRI.is_pure());
        assert!(!AArch64Opcode::StrRI.is_pure());
        assert!(!AArch64Opcode::Bl.is_pure());
    }

    #[test]
    fn flag_readers_are_not_pure() {
        // CSEL/CSET-family ops read NZCV implicitly. Reordering across a
        // flag writer would silently miscompile.
        assert!(!AArch64Opcode::Csel.is_pure());
        assert!(!AArch64Opcode::CSet.is_pure());
        assert!(!AArch64Opcode::Csinc.is_pure());
        assert!(!AArch64Opcode::Csinv.is_pure());
        assert!(!AArch64Opcode::Csneg.is_pure());
    }

    #[test]
    fn carry_readers_are_not_pure() {
        // ADC/SBC read the carry flag for multi-precision arithmetic.
        assert!(!AArch64Opcode::Adc.is_pure());
        assert!(!AArch64Opcode::Sbc.is_pure());
    }

    #[test]
    fn movk_is_not_pure() {
        // MOVK has a tied def/use: its destination is also an implicit
        // source. The operand list hides that, so treating MOVK as pure
        // allows it to float past the instruction that set the other
        // half of the constant (see #366, #382).
        assert!(!AArch64Opcode::Movk.is_pure());
    }

    #[test]
    fn bfm_is_not_pure() {
        // BFM is a bitfield *insert* (BFI/BFXIL): it preserves the bits of
        // Rd outside the inserted field, so its prior Rd value is an
        // implicit input just like MOVK. It must fail the purity gate so
        // that passes do not reorder or CSE two BFMs whose prior Rd
        // values differ (see #408).
        assert!(!AArch64Opcode::Bfm.is_pure());
        // UBFM and SBFM fully redefine Rd (uncovered bits zero- or
        // sign-extend) so they remain pure.
        assert!(AArch64Opcode::Ubfm.is_pure());
        assert!(AArch64Opcode::Sbfm.is_pure());
    }

    #[test]
    fn branches_are_not_pure() {
        // Control-flow ops are not pure regardless of memory-effect class.
        assert!(!AArch64Opcode::B.is_pure());
        assert!(!AArch64Opcode::BCond.is_pure());
        assert!(!AArch64Opcode::Bcc.is_pure());
        assert!(!AArch64Opcode::Cbz.is_pure());
        assert!(!AArch64Opcode::Cbnz.is_pure());
        assert!(!AArch64Opcode::Tbz.is_pure());
        assert!(!AArch64Opcode::Tbnz.is_pure());
        assert!(!AArch64Opcode::Br.is_pure());
        assert!(!AArch64Opcode::Ret.is_pure());
    }

    #[test]
    fn calls_are_not_pure() {
        // Calls are `Call` effect in the memory model, which is already
        // excluded by the first gate in is_pure(). Re-assert defensively.
        assert!(!AArch64Opcode::Bl.is_pure());
        assert!(!AArch64Opcode::Blr.is_pure());
    }

    #[test]
    fn traps_are_not_pure() {
        // Trap pseudo-ops alter control flow on a condition; treating them
        // as pure would let DCE delete the check.
        assert!(!AArch64Opcode::TrapOverflow.is_pure());
        assert!(!AArch64Opcode::TrapBoundsCheck.is_pure());
        assert!(!AArch64Opcode::TrapNull.is_pure());
        assert!(!AArch64Opcode::TrapDivZero.is_pure());
        assert!(!AArch64Opcode::TrapShiftRange.is_pure());
    }

    #[test]
    fn trapping_fp_is_not_pure() {
        // FDIV/FSQRT raise FP exceptions on domain errors; integer divide
        // traps on zero divisor. Conservative: not pure.
        assert!(!AArch64Opcode::FdivRR.is_pure());
        assert!(!AArch64Opcode::FsqrtRR.is_pure());
        assert!(!AArch64Opcode::NeonFdivV.is_pure());
        assert!(!AArch64Opcode::SDiv.is_pure());
        assert!(!AArch64Opcode::UDiv.is_pure());
    }

    #[test]
    fn inst_is_pure_false_for_flag_reader() {
        // Sanity: MachInst delegation reflects the tighter classification.
        let inst = MachInst::new(
            AArch64Opcode::Csel,
            vec![
                MachOperand::Imm(0),
                MachOperand::Imm(1),
                MachOperand::Imm(2),
                MachOperand::Imm(13), // LE
            ],
        );
        assert!(!inst.is_pure());
    }

    #[test]
    fn neon_ops_have_parallelism() {
        assert!(AArch64Opcode::NeonAddV.has_parallelism());
        assert!(AArch64Opcode::NeonFmulV.has_parallelism());
        assert!(AArch64Opcode::NeonAndV.has_parallelism());
    }

    #[test]
    fn scalar_ops_do_not_have_parallelism() {
        assert!(!AArch64Opcode::AddRR.has_parallelism());
        assert!(!AArch64Opcode::MulRR.has_parallelism());
        assert!(!AArch64Opcode::MovR.has_parallelism());
    }

    #[test]
    fn neon_ops_are_maps() {
        // Element-wise NEON ops qualify as maps.
        assert!(AArch64Opcode::NeonAddV.is_map());
        assert!(AArch64Opcode::NeonFmulV.is_map());
    }

    #[test]
    fn no_reductions_today() {
        // Until horizontal-add opcodes are added (#390), nothing is a reduction.
        assert!(!AArch64Opcode::AddRR.is_reduction());
        assert!(!AArch64Opcode::NeonAddV.is_reduction());
    }

    #[test]
    fn every_reduction_is_a_fold() {
        // Inductive check: folding is a superset of reducing.
        for op in [
            AArch64Opcode::AddRR,
            AArch64Opcode::NeonAddV,
            AArch64Opcode::MovR,
        ] {
            if op.is_reduction() {
                assert!(op.is_fold(), "{op:?} reduces but isn't a fold?");
            }
        }
    }

    #[test]
    fn no_bounded_loops_today() {
        // No BoundedLoop variant in ProofAnnotation yet.
        assert!(!AArch64Opcode::AddRR.has_bounded_loops());
        let inst = MachInst::new(
            AArch64Target::add_rr(),
            vec![
                MachOperand::Imm(0),
                MachOperand::Imm(1),
                MachOperand::Imm(2),
            ],
        );
        assert!(!inst.has_bounded_loops());
    }

    #[test]
    fn default_divergence_is_unknown() {
        assert_eq!(
            AArch64Opcode::AddRR.divergence_class(),
            DivergenceClass::Unknown
        );
        assert_eq!(
            AArch64Opcode::NeonAddV.divergence_class(),
            DivergenceClass::Unknown
        );
    }

    #[test]
    fn inst_delegates_to_opcode() {
        let inst = MachInst::new(
            AArch64Target::mov_rr(),
            vec![MachOperand::Imm(0), MachOperand::Imm(1)],
        );
        assert!(inst.is_pure());
        assert!(!inst.has_parallelism());
        assert!(!inst.is_reduction());
        assert!(!inst.is_fold());
        assert!(!inst.is_map());
        assert!(!inst.has_bounded_loops());
    }

    // Property: if a proof annotation makes an op "proven pure", future
    // work may override is_pure(). Today, Pure proof alone doesn't flip
    // a memory op into pure; this tests the current, conservative contract.
    #[test]
    fn pure_proof_does_not_currently_override_memory_effect() {
        let mut inst = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![
                MachOperand::Imm(0),
                MachOperand::Imm(1),
                MachOperand::Imm(0),
            ],
        );
        inst.proof = Some(llvm2_ir::ProofAnnotation::Pure);
        assert!(!inst.is_pure());
    }
}

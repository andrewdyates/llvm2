// llvm2-ir/cost_model.rs - AArch64 Apple Silicon instruction cost model
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Instruction latency and throughput cost model for AArch64 Apple Silicon.
//!
//! Provides per-opcode cycle costs for superoptimization candidate ranking
//! and optimization pass profitability analysis.
//!
//! # Data Sources
//!
//! Primary: Dougall Johnson, "Apple M1 Firestorm Microarchitecture",
//! <https://dougallj.github.io/applecpu/firestorm.html>
//!
//! The M1 Firestorm core has:
//! - 8-wide decode, 6 ALU execution units (4 integer + 2 complex integer)
//! - 2 load/store units
//! - 4 FP/NEON units (2 FADD + 2 FMUL/FMA)
//! - 1-cycle integer ALU latency for most simple ops
//! - 3-cycle MUL latency (integer)
//! - 7-10 cycle SDIV/UDIV latency (data-dependent)
//! - 4-cycle load latency (L1 hit)
//!
//! M4 data is estimated from public benchmarks; the core microarchitecture
//! is evolutionary from M1 with similar latencies but wider execution.
//!
//! # Usage
//!
//! ```ignore
//! use llvm2_ir::cost_model::{CostModel, AppleSiliconCostModel, CostModelGen};
//!
//! let model = AppleSiliconCostModel::new(CostModelGen::M1);
//! let lat = model.latency(AArch64Opcode::MulRR);  // 3
//! let tp = model.throughput(AArch64Opcode::AddRR); // 6.0 (6 ALU units)
//! ```

use crate::inst::AArch64Opcode;

// ---------------------------------------------------------------------------
// CostModel trait
// ---------------------------------------------------------------------------

/// Instruction cost model for optimization profitability analysis.
pub trait CostModel {
    /// Execution latency in cycles (result available after N cycles).
    fn latency(&self, opcode: AArch64Opcode) -> u32;

    /// Reciprocal throughput: instructions per cycle that can execute.
    /// Higher = better. E.g., 6.0 means 6 ADD instructions can dispatch
    /// per cycle across the available execution units.
    fn throughput(&self, opcode: AArch64Opcode) -> f64;

    /// Predict throughput of a basic block (instructions per cycle).
    ///
    /// Uses a simple bottleneck model:
    ///   block_throughput = total_instructions / max(critical_path_latency,
    ///                                              sum(1/throughput_per_inst))
    ///
    /// This captures both latency-bound and throughput-bound behavior.
    fn predict_block_throughput(&self, insts: &[AArch64Opcode]) -> f64;
}

// ---------------------------------------------------------------------------
// AppleSiliconCostModel
// ---------------------------------------------------------------------------

/// Which Apple Silicon generation to model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CostModelGen {
    /// M1 Firestorm core (2020). Best-characterized via Dougall Johnson's work.
    M1,
    /// M4 (2024). Evolutionary improvements over M1; similar latencies,
    /// wider execution for some units.
    M4,
}

/// Apple Silicon AArch64 cost model with per-opcode latency and throughput.
///
/// Data sourced from Dougall Johnson's M1 Firestorm microarchitecture
/// research and public M4 benchmark analysis.
#[derive(Debug, Clone, Copy)]
pub struct AppleSiliconCostModel {
    generation: CostModelGen,
}

impl AppleSiliconCostModel {
    /// Create a cost model for the specified Apple Silicon generation.
    pub fn new(generation: CostModelGen) -> Self {
        Self { generation }
    }

    /// Returns (latency, throughput) for the given opcode.
    ///
    /// Throughput is in instructions-per-cycle (IPC). Higher = faster.
    fn cost_pair(&self, opcode: AArch64Opcode) -> (u32, f64) {
        use AArch64Opcode::*;

        // M4 has similar latencies to M1 for most instructions, with
        // slightly wider execution bandwidth for integer ALU.
        let m4_int_tp = if self.generation == CostModelGen::M4 { 7.0 } else { 6.0 };

        match opcode {
            // ===== Integer ALU (1-cycle latency, 6 execution units on M1) =====
            // Source: Firestorm has 4 simple integer ALU + 2 complex integer,
            // all can handle ADD/SUB/AND/ORR/EOR/shifts.
            AddRR | AddRI => (1, m4_int_tp),
            SubRR | SubRI => (1, m4_int_tp),
            Neg           => (1, m4_int_tp),
            AndRR | AndRI => (1, m4_int_tp),
            OrrRR | OrrRI => (1, m4_int_tp),
            EorRR | EorRI => (1, m4_int_tp),
            OrnRR         => (1, m4_int_tp),
            BicRR         => (1, m4_int_tp),

            // ===== Shifts (1-cycle, uses ALU units) =====
            LslRR | LsrRR | AsrRR => (1, m4_int_tp),
            LslRI | LsrRI | AsrRI => (1, m4_int_tp),

            // ===== Integer multiply (3-cycle latency, 2 complex-integer units) =====
            // Source: Firestorm MUL latency is 3 cycles, throughput 2/cycle.
            MulRR => (3, 2.0),
            Msub  => (3, 2.0),
            Smull => (3, 2.0),
            Umull => (3, 2.0),

            // ===== Integer divide (variable latency: ~7-10 cycles, 1 unit) =====
            // Source: Firestorm SDIV/UDIV is data-dependent, ~7 for 32-bit,
            // ~10 for 64-bit. Throughput: 1 divide unit, cannot overlap.
            SDiv => (8, 0.5),
            UDiv => (8, 0.5),

            // ===== Compare and conditional (1-cycle, uses ALU) =====
            CmpRR | CmpRI       => (1, m4_int_tp),
            CMPWrr | CMPXrr     => (1, m4_int_tp),
            CMPWri | CMPXri     => (1, m4_int_tp),
            Tst                 => (1, m4_int_tp),
            Csel                => (1, m4_int_tp),
            CSet                => (1, m4_int_tp),
            Csinc               => (1, m4_int_tp),
            Csinv               => (1, m4_int_tp),
            Csneg               => (1, m4_int_tp),

            // ===== Flag-setting arithmetic (1-cycle, uses ALU) =====
            AddsRR | AddsRI => (1, m4_int_tp),
            SubsRR | SubsRI => (1, m4_int_tp),

            // ===== Move / immediate (1-cycle, uses ALU) =====
            MovR              => (1, m4_int_tp),
            MovI              => (1, m4_int_tp),
            Movz              => (1, m4_int_tp),
            Movn              => (1, m4_int_tp),
            Movk              => (1, m4_int_tp),
            MOVWrr | MOVXrr   => (1, m4_int_tp),
            MOVZWi | MOVZXi   => (1, m4_int_tp),

            // ===== Extension / bitfield (1-cycle, ALU) =====
            // SXTW, UXTW, SXTB, SXTH are aliases for SBFM/UBFM.
            Sxtw | Uxtw | Sxtb | Sxth => (1, m4_int_tp),
            Ubfm | Sbfm | Bfm         => (1, m4_int_tp),

            // ===== Address generation (1-cycle) =====
            Adrp     => (1, m4_int_tp),
            AddPCRel => (1, m4_int_tp),

            // ===== Branches =====
            // Correctly-predicted branches: 0 effective latency for throughput.
            // Misprediction penalty: ~14 cycles on Firestorm, not modeled here.
            // Throughput: 1 branch per cycle (branch predictor is single-ported).
            B    => (1, 1.0),
            BCond | Bcc => (1, 1.0),
            Cbz  => (1, 1.0),
            Cbnz => (1, 1.0),
            Tbz  => (1, 1.0),
            Tbnz => (1, 1.0),
            Br   => (1, 1.0),

            // ===== Calls / Return =====
            // BL/BLR: 1-cycle + link register write. Throughput limited by
            // branch predictor + call return stack.
            Bl | BL   => (1, 1.0),
            Blr | BLR => (1, 1.0),
            Ret       => (1, 1.0),

            // ===== Load (L1 hit: 4 cycles, 2 load/store units) =====
            // Source: Firestorm has 4-cycle load latency for L1 hits.
            // Two load/store units can handle loads in parallel.
            LdrRI      => (4, 2.0),
            LdrbRI     => (4, 2.0),
            LdrhRI     => (4, 2.0),
            LdrsbRI    => (4, 2.0),
            LdrshRI    => (4, 2.0),
            LdrRO      => (4, 2.0),
            LdrLiteral => (4, 2.0),
            LdpRI      => (4, 2.0),
            LdpPostIndex => (4, 2.0),
            LdrGot     => (4, 2.0),
            LdrTlvp    => (4, 2.0),

            // ===== Store (1-cycle dispatch, 2 load/store units) =====
            // Stores complete out of order; effective latency is 1 for dispatch.
            StrRI      => (1, 2.0),
            StrbRI     => (1, 2.0),
            StrhRI     => (1, 2.0),
            StrRO      => (1, 2.0),
            StpRI      => (1, 2.0),
            StpPreIndex  => (1, 2.0),
            STRWui     => (1, 2.0),
            STRXui     => (1, 2.0),
            STRSui     => (1, 2.0),
            STRDui     => (1, 2.0),

            // ===== FP arithmetic =====
            // Source: Firestorm FP latency is typically 3 cycles for FADD/FSUB,
            // 4 cycles for FMUL, with 2 FADD and 2 FMUL/FMA units.
            FaddRR  => (3, 2.0),
            FsubRR  => (3, 2.0),
            FmulRR  => (4, 2.0),
            FdivRR  => (10, 0.5),  // Single: ~10, Double: ~15
            FnegRR  => (1, 2.0),   // Simple bit flip, uses FP unit
            Fcmp    => (3, 2.0),
            FmovImm => (2, 2.0),

            // ===== FP conversion =====
            // FCVTZS/FCVTZU: FP-to-int conversion, 3-4 cycle latency.
            // SCVTF/UCVTF: int-to-FP, 3-4 cycle latency.
            // FCVT (precision change): 3 cycles.
            FcvtzsRR    => (4, 2.0),
            FcvtzuRR    => (4, 2.0),
            ScvtfRR     => (4, 2.0),
            UcvtfRR     => (4, 2.0),
            FcvtSD      => (3, 2.0),
            FcvtDS      => (3, 2.0),
            FmovGprFpr  => (3, 2.0),
            FmovFprGpr  => (3, 2.0),

            // ===== Pseudo-instructions (no execution cost) =====
            Phi | StackAlloc | Copy | Nop => (0, f64::INFINITY),

            // ===== Trap pseudo-instructions =====
            // These are conditional branches to panic blocks; cost is branch cost.
            TrapOverflow | TrapBoundsCheck | TrapNull
            | TrapDivZero | TrapShiftRange => (1, 1.0),

            // ===== Reference counting pseudo-instructions =====
            // Model as load+store pair (atomic RMW in practice).
            Retain  => (6, 0.5),
            Release => (6, 0.5),
        }
    }
}

impl CostModel for AppleSiliconCostModel {
    fn latency(&self, opcode: AArch64Opcode) -> u32 {
        self.cost_pair(opcode).0
    }

    fn throughput(&self, opcode: AArch64Opcode) -> f64 {
        self.cost_pair(opcode).1
    }

    fn predict_block_throughput(&self, insts: &[AArch64Opcode]) -> f64 {
        if insts.is_empty() {
            return 0.0;
        }

        let n = insts.len() as f64;

        // Critical path: sum of all latencies (conservative upper bound;
        // a real model would compute the DAG critical path).
        // For a basic block throughput estimate, we use the maximum single
        // instruction latency as a lower-bound on execution time, plus
        // account for the throughput bottleneck across execution units.
        let max_latency = insts
            .iter()
            .map(|op| self.latency(*op))
            .max()
            .unwrap_or(1) as f64;

        // Throughput bottleneck: sum of per-instruction execution time.
        // Each instruction takes 1/throughput cycles on its unit.
        let throughput_cycles: f64 = insts
            .iter()
            .map(|op| {
                let tp = self.throughput(*op);
                if tp == f64::INFINITY || tp == 0.0 {
                    0.0
                } else {
                    1.0 / tp
                }
            })
            .sum();

        // The block takes at least max(critical_path, throughput_bound) cycles.
        let total_cycles = f64::max(max_latency, throughput_cycles);

        if total_cycles <= 0.0 {
            return n; // All pseudo-instructions
        }

        // Result: instructions per cycle
        n / total_cycles
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inst::AArch64Opcode;

    fn m1() -> AppleSiliconCostModel {
        AppleSiliconCostModel::new(CostModelGen::M1)
    }

    fn m4() -> AppleSiliconCostModel {
        AppleSiliconCostModel::new(CostModelGen::M4)
    }

    // ---- Latency sanity checks ----

    #[test]
    fn integer_alu_latency_is_1() {
        let model = m1();
        let ops = [
            AArch64Opcode::AddRR, AArch64Opcode::AddRI,
            AArch64Opcode::SubRR, AArch64Opcode::SubRI,
            AArch64Opcode::AndRR, AArch64Opcode::OrrRR,
            AArch64Opcode::EorRR, AArch64Opcode::Neg,
        ];
        for op in &ops {
            assert_eq!(model.latency(*op), 1, "{:?} should have 1-cycle latency", op);
        }
    }

    #[test]
    fn multiply_latency_is_3() {
        let model = m1();
        assert_eq!(model.latency(AArch64Opcode::MulRR), 3);
        assert_eq!(model.latency(AArch64Opcode::Msub), 3);
        assert_eq!(model.latency(AArch64Opcode::Smull), 3);
        assert_eq!(model.latency(AArch64Opcode::Umull), 3);
    }

    #[test]
    fn divide_latency_is_high() {
        let model = m1();
        assert!(model.latency(AArch64Opcode::SDiv) >= 7);
        assert!(model.latency(AArch64Opcode::UDiv) >= 7);
    }

    #[test]
    fn load_latency_is_4() {
        let model = m1();
        assert_eq!(model.latency(AArch64Opcode::LdrRI), 4);
        assert_eq!(model.latency(AArch64Opcode::LdpRI), 4);
        assert_eq!(model.latency(AArch64Opcode::LdrbRI), 4);
    }

    #[test]
    fn store_latency_is_low() {
        let model = m1();
        assert_eq!(model.latency(AArch64Opcode::StrRI), 1);
        assert_eq!(model.latency(AArch64Opcode::StpRI), 1);
    }

    #[test]
    fn fp_latencies() {
        let model = m1();
        assert_eq!(model.latency(AArch64Opcode::FaddRR), 3);
        assert_eq!(model.latency(AArch64Opcode::FmulRR), 4);
        assert!(model.latency(AArch64Opcode::FdivRR) >= 10);
    }

    #[test]
    fn pseudo_instructions_zero_latency() {
        let model = m1();
        assert_eq!(model.latency(AArch64Opcode::Phi), 0);
        assert_eq!(model.latency(AArch64Opcode::Nop), 0);
        assert_eq!(model.latency(AArch64Opcode::Copy), 0);
    }

    // ---- Throughput checks ----

    #[test]
    fn integer_alu_high_throughput() {
        let model = m1();
        assert!(model.throughput(AArch64Opcode::AddRR) >= 6.0);
        assert!(model.throughput(AArch64Opcode::SubRR) >= 6.0);
    }

    #[test]
    fn multiply_throughput() {
        let model = m1();
        assert_eq!(model.throughput(AArch64Opcode::MulRR), 2.0);
    }

    #[test]
    fn divide_low_throughput() {
        let model = m1();
        assert!(model.throughput(AArch64Opcode::SDiv) <= 1.0);
    }

    #[test]
    fn branch_single_throughput() {
        let model = m1();
        assert_eq!(model.throughput(AArch64Opcode::B), 1.0);
        assert_eq!(model.throughput(AArch64Opcode::BCond), 1.0);
    }

    #[test]
    fn pseudo_infinite_throughput() {
        let model = m1();
        assert!(model.throughput(AArch64Opcode::Nop).is_infinite());
        assert!(model.throughput(AArch64Opcode::Phi).is_infinite());
    }

    // ---- M4 vs M1 differences ----

    #[test]
    fn m4_wider_integer_throughput() {
        let m1_model = m1();
        let m4_model = m4();
        // M4 should have >= M1 integer throughput
        assert!(m4_model.throughput(AArch64Opcode::AddRR)
            >= m1_model.throughput(AArch64Opcode::AddRR));
    }

    #[test]
    fn m4_same_multiply_latency() {
        let m1_model = m1();
        let m4_model = m4();
        assert_eq!(
            m1_model.latency(AArch64Opcode::MulRR),
            m4_model.latency(AArch64Opcode::MulRR),
        );
    }

    // ---- Block throughput prediction ----

    #[test]
    fn empty_block_throughput() {
        let model = m1();
        assert_eq!(model.predict_block_throughput(&[]), 0.0);
    }

    #[test]
    fn single_add_throughput() {
        let model = m1();
        let tp = model.predict_block_throughput(&[AArch64Opcode::AddRR]);
        // Single ADD: 1 cycle latency, 1/6 throughput cycle.
        // max(1, 1/6) = 1.0 cycles. 1 inst / 1.0 = 1.0 IPC.
        assert!((tp - 1.0).abs() < 0.01);
    }

    #[test]
    fn all_adds_throughput_limited_by_latency() {
        let model = m1();
        // 3 ADDs: max_latency = 1, throughput_cycles = 3 * (1/6) = 0.5
        // max(1, 0.5) = 1.0 → 3/1.0 = 3.0 IPC
        let tp = model.predict_block_throughput(&[
            AArch64Opcode::AddRR,
            AArch64Opcode::AddRR,
            AArch64Opcode::AddRR,
        ]);
        assert!((tp - 3.0).abs() < 0.01);
    }

    #[test]
    fn divide_dominates_block() {
        let model = m1();
        // 1 SDIV + 2 ADDs: max_latency = 8, tp_cycles = 1/0.5 + 2/6 = 2.33
        // max(8, 2.33) = 8.0 → 3/8 = 0.375 IPC
        let tp = model.predict_block_throughput(&[
            AArch64Opcode::SDiv,
            AArch64Opcode::AddRR,
            AArch64Opcode::AddRR,
        ]);
        assert!((tp - 3.0 / 8.0).abs() < 0.01);
    }

    #[test]
    fn throughput_bound_block() {
        let model = m1();
        // 12 ADDs: max_latency = 1, tp_cycles = 12 * (1/6) = 2.0
        // max(1, 2.0) = 2.0 → 12/2.0 = 6.0 IPC
        let ops = vec![AArch64Opcode::AddRR; 12];
        let tp = model.predict_block_throughput(&ops);
        assert!((tp - 6.0).abs() < 0.01);
    }

    #[test]
    fn load_heavy_block() {
        let model = m1();
        // 4 loads: max_latency = 4, tp_cycles = 4 * (1/2) = 2.0
        // max(4, 2.0) = 4.0 → 4/4.0 = 1.0 IPC
        let tp = model.predict_block_throughput(&[
            AArch64Opcode::LdrRI,
            AArch64Opcode::LdrRI,
            AArch64Opcode::LdrRI,
            AArch64Opcode::LdrRI,
        ]);
        assert!((tp - 1.0).abs() < 0.01);
    }

    #[test]
    fn mixed_fp_block() {
        let model = m1();
        // FADD(lat=3,tp=2.0) + FMUL(lat=4,tp=2.0) + FADD(lat=3,tp=2.0)
        // max_latency = 4, tp_cycles = 1/2 + 1/2 + 1/2 = 1.5
        // max(4, 1.5) = 4.0 → 3/4.0 = 0.75 IPC
        let tp = model.predict_block_throughput(&[
            AArch64Opcode::FaddRR,
            AArch64Opcode::FmulRR,
            AArch64Opcode::FaddRR,
        ]);
        assert!((tp - 0.75).abs() < 0.01);
    }

    // ---- All opcodes have defined costs ----

    #[test]
    fn all_opcodes_return_finite_latency_or_zero() {
        let model = m1();
        // Exhaustive test: every opcode in the enum should return a valid cost.
        // We test a representative from each category.
        let all_ops = [
            // Arithmetic
            AArch64Opcode::AddRR, AArch64Opcode::AddRI,
            AArch64Opcode::SubRR, AArch64Opcode::SubRI,
            AArch64Opcode::MulRR, AArch64Opcode::Msub,
            AArch64Opcode::Smull, AArch64Opcode::Umull,
            AArch64Opcode::SDiv, AArch64Opcode::UDiv,
            AArch64Opcode::Neg,
            // Logical
            AArch64Opcode::AndRR, AArch64Opcode::AndRI,
            AArch64Opcode::OrrRR, AArch64Opcode::OrrRI,
            AArch64Opcode::EorRR, AArch64Opcode::EorRI,
            AArch64Opcode::OrnRR, AArch64Opcode::BicRR,
            // Shifts
            AArch64Opcode::LslRR, AArch64Opcode::LsrRR, AArch64Opcode::AsrRR,
            AArch64Opcode::LslRI, AArch64Opcode::LsrRI, AArch64Opcode::AsrRI,
            // Compare
            AArch64Opcode::CmpRR, AArch64Opcode::CmpRI,
            AArch64Opcode::CMPWrr, AArch64Opcode::CMPXrr,
            AArch64Opcode::CMPWri, AArch64Opcode::CMPXri,
            AArch64Opcode::Tst,
            // Conditional
            AArch64Opcode::Csel, AArch64Opcode::CSet,
            AArch64Opcode::Csinc, AArch64Opcode::Csinv, AArch64Opcode::Csneg,
            // Flag-setting
            AArch64Opcode::AddsRR, AArch64Opcode::AddsRI,
            AArch64Opcode::SubsRR, AArch64Opcode::SubsRI,
            // Move
            AArch64Opcode::MovR, AArch64Opcode::MovI,
            AArch64Opcode::Movz, AArch64Opcode::Movn, AArch64Opcode::Movk,
            AArch64Opcode::MOVWrr, AArch64Opcode::MOVXrr,
            AArch64Opcode::MOVZWi, AArch64Opcode::MOVZXi,
            // Extension
            AArch64Opcode::Sxtw, AArch64Opcode::Uxtw,
            AArch64Opcode::Sxtb, AArch64Opcode::Sxth,
            AArch64Opcode::Ubfm, AArch64Opcode::Sbfm, AArch64Opcode::Bfm,
            // Address
            AArch64Opcode::Adrp, AArch64Opcode::AddPCRel,
            // Branch
            AArch64Opcode::B, AArch64Opcode::BCond, AArch64Opcode::Bcc,
            AArch64Opcode::Cbz, AArch64Opcode::Cbnz,
            AArch64Opcode::Tbz, AArch64Opcode::Tbnz,
            AArch64Opcode::Br,
            // Call/Return
            AArch64Opcode::Bl, AArch64Opcode::BL,
            AArch64Opcode::Blr, AArch64Opcode::BLR,
            AArch64Opcode::Ret,
            // Load
            AArch64Opcode::LdrRI, AArch64Opcode::LdrbRI,
            AArch64Opcode::LdrhRI, AArch64Opcode::LdrsbRI,
            AArch64Opcode::LdrshRI, AArch64Opcode::LdrRO,
            AArch64Opcode::LdrLiteral, AArch64Opcode::LdpRI,
            AArch64Opcode::LdpPostIndex,
            AArch64Opcode::LdrGot, AArch64Opcode::LdrTlvp,
            // Store
            AArch64Opcode::StrRI, AArch64Opcode::StrbRI,
            AArch64Opcode::StrhRI, AArch64Opcode::StrRO,
            AArch64Opcode::StpRI, AArch64Opcode::StpPreIndex,
            AArch64Opcode::STRWui, AArch64Opcode::STRXui,
            AArch64Opcode::STRSui, AArch64Opcode::STRDui,
            // FP
            AArch64Opcode::FaddRR, AArch64Opcode::FsubRR,
            AArch64Opcode::FmulRR, AArch64Opcode::FdivRR,
            AArch64Opcode::FnegRR, AArch64Opcode::Fcmp,
            AArch64Opcode::FmovImm,
            // FP conversion
            AArch64Opcode::FcvtzsRR, AArch64Opcode::FcvtzuRR,
            AArch64Opcode::ScvtfRR, AArch64Opcode::UcvtfRR,
            AArch64Opcode::FcvtSD, AArch64Opcode::FcvtDS,
            AArch64Opcode::FmovGprFpr, AArch64Opcode::FmovFprGpr,
            // Pseudo
            AArch64Opcode::Phi, AArch64Opcode::StackAlloc,
            AArch64Opcode::Copy, AArch64Opcode::Nop,
            // Trap
            AArch64Opcode::TrapOverflow, AArch64Opcode::TrapBoundsCheck,
            AArch64Opcode::TrapNull, AArch64Opcode::TrapDivZero,
            AArch64Opcode::TrapShiftRange,
            // Refcount
            AArch64Opcode::Retain, AArch64Opcode::Release,
        ];
        for op in &all_ops {
            let lat = model.latency(*op);
            let tp = model.throughput(*op);
            // Latency should be finite
            assert!(lat <= 100, "{:?} has unreasonable latency {}", op, lat);
            // Throughput should be positive (or infinite for pseudos)
            assert!(tp > 0.0, "{:?} has non-positive throughput {}", op, tp);
        }
    }

    // ---- Relative cost ordering ----

    #[test]
    fn divide_slower_than_multiply() {
        let model = m1();
        assert!(model.latency(AArch64Opcode::SDiv) > model.latency(AArch64Opcode::MulRR));
    }

    #[test]
    fn multiply_slower_than_add() {
        let model = m1();
        assert!(model.latency(AArch64Opcode::MulRR) > model.latency(AArch64Opcode::AddRR));
    }

    #[test]
    fn load_slower_than_alu() {
        let model = m1();
        assert!(model.latency(AArch64Opcode::LdrRI) > model.latency(AArch64Opcode::AddRR));
    }

    #[test]
    fn fp_div_slower_than_fp_add() {
        let model = m1();
        assert!(model.latency(AArch64Opcode::FdivRR) > model.latency(AArch64Opcode::FaddRR));
    }

    #[test]
    fn fp_mul_slower_than_fp_add() {
        let model = m1();
        assert!(model.latency(AArch64Opcode::FmulRR) > model.latency(AArch64Opcode::FaddRR));
    }
}

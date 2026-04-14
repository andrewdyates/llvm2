// llvm2-ir/cost_model.rs - AArch64 Apple Silicon instruction cost model
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Instruction latency and throughput cost model for AArch64 Apple Silicon.
//!
//! Provides per-opcode cycle costs for superoptimization candidate ranking
//! and optimization pass profitability analysis. Supports scalar CPU, NEON
//! vector, GPU, and ANE (Apple Neural Engine) compute targets for unified
//! multi-target cost estimation.
//!
//! # Data Sources
//!
//! Primary: Dougall Johnson, "Apple M1 Firestorm Microarchitecture",
//! <https://dougallj.github.io/applecpu/firestorm.html>
//!
//! Dougall Johnson, "Apple M1 Firestorm SIMD/FP Instructions",
//! <https://dougallj.github.io/applecpu/firestorm-simd.html>
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
//! # Multi-target cost estimation
//!
//! The [`MultiTargetCostModel`] provides unified cost estimation across:
//! - **CPU Scalar**: per-opcode costs from [`AppleSiliconCostModel`]
//! - **NEON**: per-arrangement vector costs (8B/16B/4H/8H/2S/4S/1D/2D)
//! - **GPU**: Metal compute dispatch overhead + kernel throughput
//! - **ANE**: CoreML compilation overhead + inference throughput
//!
//! ```ignore
//! use llvm2_ir::cost_model::{MultiTargetCostModel, ComputeTarget, CostModelGen};
//!
//! let model = MultiTargetCostModel::new(CostModelGen::M1);
//! let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "MUL", 32);
//! let neon = model.estimate_cost(ComputeTarget::Neon, "MUL", 128);
//! let gpu = model.estimate_cost(ComputeTarget::Gpu, "MUL", 128);
//! assert!(neon.latency_cycles < scalar.latency_cycles * 4);
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
// ComputeTarget — unified target for multi-target cost estimation
// ---------------------------------------------------------------------------

/// Compute targets for multi-target cost estimation.
///
/// The synthesis loop generates candidates for multiple execution domains.
/// Each target has fundamentally different cost characteristics:
/// - CPU scalar: per-instruction cycle costs, 1-element width
/// - NEON: SIMD vector costs, 64-bit or 128-bit arrangements
/// - GPU: Metal compute shader dispatch, amortized over large workloads
/// - ANE: Apple Neural Engine, CoreML compilation + batched inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeTarget {
    /// Scalar integer/FP on AArch64 CPU core.
    CpuScalar,
    /// ARM NEON SIMD (Advanced SIMD). 128-bit registers, per-arrangement costs.
    Neon,
    /// Apple GPU via Metal compute shaders.
    Gpu,
    /// Apple Neural Engine via CoreML/BNNS.
    Ane,
}

// ---------------------------------------------------------------------------
// NeonArrangement — NEON vector element arrangement
// ---------------------------------------------------------------------------

/// NEON vector arrangement specifier.
///
/// Determines the element size and count within a 64-bit or 128-bit NEON
/// register. Naming follows ARM convention: `<count><element_type>`.
///
/// Source: ARM Architecture Reference Manual, C7.2 "Advanced SIMD data types".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeonArrangement {
    /// 8 bytes (8x8-bit), 64-bit register (Vd.8B)
    B8,
    /// 16 bytes (16x8-bit), 128-bit register (Vd.16B)
    B16,
    /// 4 halfwords (4x16-bit), 64-bit register (Vd.4H)
    H4,
    /// 8 halfwords (8x16-bit), 128-bit register (Vd.8H)
    H8,
    /// 2 singles (2x32-bit), 64-bit register (Vd.2S)
    S2,
    /// 4 singles (4x32-bit), 128-bit register (Vd.4S)
    S4,
    /// 1 double (1x64-bit), 64-bit register (Vd.1D)
    D1,
    /// 2 doubles (2x64-bit), 128-bit register (Vd.2D)
    D2,
}

impl NeonArrangement {
    /// Total vector width in bits (64 or 128).
    pub fn width_bits(self) -> u32 {
        match self {
            Self::B8 | Self::H4 | Self::S2 | Self::D1 => 64,
            Self::B16 | Self::H8 | Self::S4 | Self::D2 => 128,
        }
    }

    /// Element size in bits (8, 16, 32, or 64).
    pub fn element_bits(self) -> u32 {
        match self {
            Self::B8 | Self::B16 => 8,
            Self::H4 | Self::H8 => 16,
            Self::S2 | Self::S4 => 32,
            Self::D1 | Self::D2 => 64,
        }
    }

    /// Number of elements (lanes) in the arrangement.
    pub fn lane_count(self) -> u32 {
        self.width_bits() / self.element_bits()
    }

    /// Infer arrangement from a total operation width in bits.
    ///
    /// Returns the 128-bit arrangement with 32-bit elements by default for
    /// ambiguous widths. Returns `None` for unsupported widths.
    pub fn from_width(width: u32) -> Option<Self> {
        match width {
            8 => Some(Self::B8),     // single byte in 8B register
            16 => Some(Self::H4),    // single halfword in 4H register
            32 => Some(Self::S2),    // single word in 2S register
            64 => Some(Self::D1),    // single double in 1D register
            128 => Some(Self::S4),   // 4x32 (most common 128-bit arrangement)
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NeonOp — NEON operation category
// ---------------------------------------------------------------------------

/// NEON operation categories for cost lookup.
///
/// Maps to ARM Advanced SIMD instruction classes. Each op has a known
/// per-arrangement latency on Apple Silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeonOp {
    /// Integer ADD (ADD Vd.<T>, Vn.<T>, Vm.<T>)
    Add,
    /// Integer SUB (SUB Vd.<T>, Vn.<T>, Vm.<T>)
    Sub,
    /// Integer MUL (MUL Vd.<T>, Vn.<T>, Vm.<T>)
    Mul,
    /// Integer NEG (NEG Vd.<T>, Vn.<T>)
    Neg,
    /// Bitwise AND (AND Vd.16B, Vn.16B, Vm.16B)
    And,
    /// Bitwise OR (ORR Vd.16B, Vn.16B, Vm.16B)
    Orr,
    /// Bitwise XOR (EOR Vd.16B, Vn.16B, Vm.16B)
    Eor,
    /// Bit clear (BIC Vd.16B, Vn.16B, Vm.16B)
    Bic,
    /// Shift left (SHL Vd.<T>, Vn.<T>, #imm)
    Shl,
    /// Unsigned shift right (USHR Vd.<T>, Vn.<T>, #imm)
    Ushr,
    /// Signed shift right (SSHR Vd.<T>, Vn.<T>, #imm)
    Sshr,
    /// FP ADD (FADD Vd.<T>, Vn.<T>, Vm.<T>)
    Fadd,
    /// FP MUL (FMUL Vd.<T>, Vn.<T>, Vm.<T>)
    Fmul,
    /// FP FMA (FMLA Vd.<T>, Vn.<T>, Vm.<T>)
    Fmla,
}

impl NeonOp {
    /// Parse a NEON operation from a string name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "ADD" => Some(Self::Add),
            "SUB" => Some(Self::Sub),
            "MUL" => Some(Self::Mul),
            "NEG" => Some(Self::Neg),
            "AND" => Some(Self::And),
            "ORR" | "OR" => Some(Self::Orr),
            "EOR" | "XOR" => Some(Self::Eor),
            "BIC" => Some(Self::Bic),
            "SHL" => Some(Self::Shl),
            "USHR" => Some(Self::Ushr),
            "SSHR" => Some(Self::Sshr),
            "FADD" => Some(Self::Fadd),
            "FMUL" => Some(Self::Fmul),
            "FMLA" | "FMA" => Some(Self::Fmla),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CostEstimate — unified cost result
// ---------------------------------------------------------------------------

/// Unified cost estimate for any compute target.
///
/// Returned by [`MultiTargetCostModel::estimate_cost`]. All fields are
/// target-relative: `latency_cycles` is CPU cycles for CPU/NEON targets,
/// and estimated equivalent cycles for GPU/ANE (including dispatch overhead).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    /// Execution latency in cycles (or equivalent units for GPU/ANE).
    ///
    /// For CPU/NEON: pipeline latency from issue to result availability.
    /// For GPU: dispatch overhead + kernel execution (amortized).
    /// For ANE: CoreML compile + inference (amortized).
    pub latency_cycles: f64,

    /// Throughput in operations per cycle (higher = better).
    ///
    /// For CPU/NEON: instructions per cycle across available execution units.
    /// For GPU: operations per cycle at peak throughput (sustained).
    /// For ANE: matrix ops per cycle at peak throughput.
    pub throughput_per_cycle: f64,

    /// Relative energy cost (1.0 = one scalar integer ADD).
    ///
    /// Normalized to scalar integer ALU. Approximate values:
    /// - Scalar ALU: 1.0
    /// - NEON: 1.5-2.0 (wider datapath, same voltage)
    /// - GPU: 0.3-0.5 per-element (amortized over large workloads)
    /// - ANE: 0.1-0.2 per-element (fixed-function, very efficient)
    pub energy_relative: f64,
}

impl CostEstimate {
    /// Cost-effectiveness metric: throughput / energy.
    ///
    /// Higher is better. Useful for ranking targets when both performance
    /// and power efficiency matter.
    pub fn efficiency(&self) -> f64 {
        if self.energy_relative <= 0.0 {
            return 0.0;
        }
        self.throughput_per_cycle / self.energy_relative
    }
}

// ---------------------------------------------------------------------------
// DataTransferCost — cross-domain transfer costs
// ---------------------------------------------------------------------------

/// Cross-domain data transfer costs.
///
/// Moving data between compute domains is expensive and often dominates
/// total cost. The synthesis loop must account for these when evaluating
/// mixed-target strategies.
///
/// Source: Dougall Johnson, M1 Firestorm SIMD/FP instructions:
/// GPR<->SIMD transfer is 12-13 cycles on Apple Silicon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DataTransferCost {
    /// Scalar GPR -> NEON register (DUP from GPR, FMOV GPR->FP).
    /// M1 Firestorm: ~12 cycles (2 uops, cross-domain penalty).
    pub scalar_to_neon_cycles: f64,

    /// NEON register -> Scalar GPR (UMOV, FMOV FP->GPR).
    /// M1 Firestorm: ~13 cycles (2 uops, cross-domain penalty).
    pub neon_to_scalar_cycles: f64,

    /// Memory -> NEON register (LD1, LDR Q-form).
    /// M1 Firestorm: 4 cycles L1 hit, same as scalar load.
    pub memory_to_neon_cycles: f64,

    /// NEON register -> Memory (ST1, STR Q-form).
    /// M1 Firestorm: 1 cycle dispatch (same as scalar store).
    pub neon_to_memory_cycles: f64,

    /// CPU -> GPU buffer transfer overhead in equivalent cycles.
    /// Unified memory on Apple Silicon: ~200-500 cycles for buffer
    /// mapping/synchronization (no actual copy needed).
    pub cpu_to_gpu_cycles: f64,

    /// GPU -> CPU buffer readback overhead in equivalent cycles.
    /// Unified memory: ~300-800 cycles for synchronization fence.
    pub gpu_to_cpu_cycles: f64,

    /// CPU -> ANE input tensor preparation in equivalent cycles.
    /// CoreML input marshalling: ~1000-5000 cycles depending on
    /// tensor size and format conversion.
    pub cpu_to_ane_cycles: f64,

    /// ANE -> CPU output tensor readback in equivalent cycles.
    /// CoreML output unmarshalling: ~500-2000 cycles.
    pub ane_to_cpu_cycles: f64,
}

// ---------------------------------------------------------------------------
// MultiTargetCostModel
// ---------------------------------------------------------------------------

/// Unified multi-target cost model for Apple Silicon.
///
/// Provides cost estimation across CPU scalar, NEON, GPU, and ANE targets.
/// The synthesis loop uses this to rank candidates from different compute
/// domains and decide when vectorization or offloading is profitable.
///
/// # Architecture
///
/// Apple Silicon (M1-M4) has four major compute domains:
///
/// 1. **CPU scalar**: 6 integer ALU pipes, 4 FP/SIMD pipes on Firestorm P-core.
///    Lowest latency, best for small/irregular workloads.
///
/// 2. **NEON**: Same 4 FP/SIMD pipes as scalar FP, but operating on 128-bit
///    vectors. Same latency as scalar (2-4 cycles), 2-16x throughput depending
///    on element width. Amortization threshold: ~4 elements to beat scalar.
///
/// 3. **GPU**: Metal compute shaders on the integrated GPU. High dispatch
///    overhead (~5000-15000 cycles) but massive throughput for large workloads.
///    Amortization threshold: ~10000 elements to beat NEON.
///
/// 4. **ANE**: Apple Neural Engine. Fixed-function matrix/convolution hardware.
///    Very high CoreML compilation overhead (~50000-200000 cycles) but extreme
///    throughput for supported operations (GEMM, conv2d). Amortization
///    threshold: ~100000 elements.
///
/// # Data Sources
///
/// - NEON: Dougall Johnson, "Apple M1 Firestorm SIMD/FP Instructions"
/// - GPU: Apple Metal Performance Shaders documentation, public benchmarks
/// - ANE: Apple CoreML documentation, MLPerf benchmarks (estimated)
#[derive(Debug, Clone, Copy)]
pub struct MultiTargetCostModel {
    scalar: AppleSiliconCostModel,
    generation: CostModelGen,
}

impl MultiTargetCostModel {
    /// Create a multi-target cost model for the specified Apple Silicon generation.
    pub fn new(generation: CostModelGen) -> Self {
        Self {
            scalar: AppleSiliconCostModel::new(generation),
            generation,
        }
    }

    /// Access the underlying scalar CPU cost model.
    pub fn scalar_model(&self) -> &AppleSiliconCostModel {
        &self.scalar
    }

    /// Estimate cost for an operation on a specific compute target.
    ///
    /// # Arguments
    /// - `target`: The compute domain (CpuScalar, Neon, Gpu, Ane).
    /// - `op`: Operation name (e.g., "ADD", "MUL", "FADD", "GEMM").
    ///   Case-insensitive. Unrecognized ops return a conservative estimate.
    /// - `width`: Total operation width in bits. For scalar: element width
    ///   (32 or 64). For NEON: vector width (64 or 128). For GPU/ANE:
    ///   total data width in bits (e.g., 4096 for 128 floats).
    ///
    /// # Returns
    /// A [`CostEstimate`] with latency, throughput, and energy metrics.
    pub fn estimate_cost(&self, target: ComputeTarget, op: &str, width: u32) -> CostEstimate {
        match target {
            ComputeTarget::CpuScalar => self.estimate_scalar(op, width),
            ComputeTarget::Neon => self.estimate_neon(op, width),
            ComputeTarget::Gpu => self.estimate_gpu(op, width),
            ComputeTarget::Ane => self.estimate_ane(op, width),
        }
    }

    /// Get cross-domain data transfer costs.
    ///
    /// Transfer costs are critical for profitability analysis: a NEON
    /// operation that saves 2 cycles but requires 12+13=25 cycles of
    /// GPR<->SIMD transfers is a net loss.
    pub fn transfer_costs(&self) -> DataTransferCost {
        // M4 has slightly improved cross-domain transfer, estimated ~10%
        let cross_domain_factor = match self.generation {
            CostModelGen::M1 => 1.0,
            CostModelGen::M4 => 0.9,
        };

        DataTransferCost {
            scalar_to_neon_cycles: 12.0 * cross_domain_factor,
            neon_to_scalar_cycles: 13.0 * cross_domain_factor,
            memory_to_neon_cycles: 4.0,
            neon_to_memory_cycles: 1.0,
            cpu_to_gpu_cycles: 350.0,
            gpu_to_cpu_cycles: 550.0,
            cpu_to_ane_cycles: 3000.0,
            ane_to_cpu_cycles: 1500.0,
        }
    }

    /// NEON cost for a specific operation and arrangement.
    ///
    /// Returns (latency, throughput_per_cycle) for the given NEON operation
    /// on the specified arrangement. Data from Dougall Johnson's M1 Firestorm
    /// SIMD/FP instruction analysis.
    pub fn neon_cost(&self, op: NeonOp, arr: NeonArrangement) -> (u32, f64) {
        // M1 Firestorm has 4 FP/SIMD execution units (u11-u14).
        // All NEON integer and FP operations can issue to all 4 units
        // (throughput 0.25 c/i = 4 ops/cycle) unless noted otherwise.
        //
        // Source: Dougall Johnson, "Apple M1 Firestorm SIMD/FP Instructions"
        // https://dougallj.github.io/applecpu/firestorm-simd.html
        let fp_tp = 4.0; // 4 FP/SIMD units, all arrangements

        match op {
            // -- Integer NEON (2-cycle latency, 4-way issue) --
            // ADD/SUB/NEG/AND/ORR/EOR/BIC are all simple SIMD integer
            // operations with 2-cycle latency on M1 Firestorm.
            NeonOp::Add | NeonOp::Sub | NeonOp::Neg => (2, fp_tp),
            NeonOp::And | NeonOp::Orr | NeonOp::Eor | NeonOp::Bic => (2, fp_tp),
            NeonOp::Shl | NeonOp::Ushr | NeonOp::Sshr => (2, fp_tp),

            // -- Integer MUL (4-cycle latency for 16/32-bit, 3c for 8-bit) --
            // MUL.4S/MUL.8H: 4 cycles. MUL.16B: 3 cycles.
            // 8-bit multiply is simpler hardware (fewer partial products).
            NeonOp::Mul => {
                let lat = match arr.element_bits() {
                    8 => 3,
                    _ => 4,
                };
                (lat, fp_tp)
            }

            // -- FP operations --
            // FADD: 3-cycle latency, 4-way issue
            NeonOp::Fadd => (3, fp_tp),
            // FMUL: 3-cycle latency, 4-way issue
            NeonOp::Fmul => (3, fp_tp),
            // FMLA (fused multiply-add): 4-cycle latency, 4-way issue
            NeonOp::Fmla => (4, fp_tp),
        }
    }

    /// Recommend the best compute target for a given operation and element count.
    ///
    /// Compares per-element cost across scalar and NEON (including data transfer
    /// overhead amortized across elements). GPU and ANE are only recommended for
    /// very large element counts due to dispatch/compilation overhead.
    ///
    /// # Arguments
    /// - `op`: Operation name (e.g., "ADD", "MUL").
    /// - `element_bits`: Size of each element in bits (8, 16, 32, or 64).
    /// - `element_count`: Number of elements to process.
    ///
    /// # Returns
    /// The [`ComputeTarget`] with the lowest effective per-element cost,
    /// accounting for data transfer overhead.
    pub fn recommend_target(
        &self,
        op: &str,
        element_bits: u32,
        element_count: u32,
    ) -> ComputeTarget {
        // Compare total cost for processing `element_count` elements.
        //
        // Scalar: compute per element. No domain transfer needed (data in GPRs).
        // NEON: per-element compute (amortized across lanes) + domain transfer.
        //
        // For data in memory, both need loads/stores. But NEON additionally
        // requires the data to be in SIMD registers, and results may need to
        // go back to scalar GPRs. We model the minimum overhead: if data is
        // already contiguous in memory, NEON can use LD1/ST1 (cheap). But
        // there's always some setup overhead for entering/exiting the NEON
        // domain.

        let scalar = self.estimate_cost(ComputeTarget::CpuScalar, op, element_bits);
        let tc = self.transfer_costs();
        let n = element_count as f64;

        // Scalar total: just compute cost per element.
        let scalar_total = scalar.latency_cycles * n;

        // NEON cost: pick the best arrangement for the element size.
        let neon_arr = match element_bits {
            8 => NeonArrangement::B16,  // 16 lanes
            16 => NeonArrangement::H8,  // 8 lanes
            32 => NeonArrangement::S4,  // 4 lanes
            64 => NeonArrangement::D2,  // 2 lanes
            _ => return ComputeTarget::CpuScalar, // unsupported element size
        };
        let neon_width = neon_arr.width_bits();
        let neon_est = self.estimate_cost(ComputeTarget::Neon, op, neon_width);
        let lanes = neon_arr.lane_count() as f64;

        // NEON total: per-element compute + data transfer overhead.
        // Transfer overhead includes memory<->NEON vector loads/stores,
        // amortized across lanes within each vector.
        let vectors_needed = (n / lanes).ceil();
        let neon_transfer = vectors_needed * (tc.memory_to_neon_cycles + tc.neon_to_memory_cycles);
        let neon_compute = neon_est.latency_cycles * n;
        let neon_total = neon_compute + neon_transfer;

        if neon_total < scalar_total {
            ComputeTarget::Neon
        } else {
            ComputeTarget::CpuScalar
        }
    }

    // -----------------------------------------------------------------------
    // Private: per-target cost estimation
    // -----------------------------------------------------------------------

    fn estimate_scalar(&self, op: &str, width: u32) -> CostEstimate {
        // Map string op name to approximate scalar latency/throughput.
        let upper = op.to_ascii_uppercase();
        let (lat, tp) = match upper.as_str() {
            "ADD" | "SUB" | "NEG" | "AND" | "ORR" | "OR"
            | "EOR" | "XOR" | "BIC" | "SHL" | "LSL"
            | "USHR" | "LSR" | "SSHR" | "ASR" => (1, 6.0),
            "MUL" => (3, 2.0),
            "SDIV" | "UDIV" | "DIV" => (8, 0.5),
            "FADD" | "FSUB" => (3, 2.0),
            "FMUL" => (4, 2.0),
            "FDIV" => (10, 0.5),
            "FMA" | "FMLA" | "FMADD" => (4, 2.0),
            "CMP" | "TST" => (1, 6.0),
            "MOV" | "FMOV" => (1, 6.0),
            "LDR" | "LOAD" => (4, 2.0),
            "STR" | "STORE" => (1, 2.0),
            _ => (2, 4.0), // conservative default
        };

        // M4 has slightly wider integer throughput
        let tp_adjusted = if self.generation == CostModelGen::M4 {
            match upper.as_str() {
                "ADD" | "SUB" | "NEG" | "AND" | "ORR" | "OR"
                | "EOR" | "XOR" | "BIC" | "SHL" | "LSL"
                | "USHR" | "LSR" | "SSHR" | "ASR"
                | "CMP" | "TST" | "MOV" => 7.0_f64.min(tp + 1.0),
                _ => tp,
            }
        } else {
            tp
        };

        // 64-bit operations have same latency as 32-bit on AArch64 for
        // most instructions (both are single-cycle on the full 64-bit ALU).
        let _ = width; // width doesn't affect scalar cost significantly

        CostEstimate {
            latency_cycles: lat as f64,
            throughput_per_cycle: tp_adjusted,
            energy_relative: 1.0,
        }
    }

    fn estimate_neon(&self, op: &str, width: u32) -> CostEstimate {
        let neon_op = NeonOp::from_name(op);
        let arrangement = NeonArrangement::from_width(width).unwrap_or(NeonArrangement::S4);
        let lanes = arrangement.lane_count() as f64;

        let (lat, tp) = match neon_op {
            Some(nop) => self.neon_cost(nop, arrangement),
            None => {
                // Unknown NEON op: conservative 3-cycle, 4-way issue
                (3, 4.0)
            }
        };

        // Normalize costs per-element so NEON and scalar are comparable.
        //
        // A NEON ADD.4S processes 4 elements in 2 cycles:
        //   per-element latency = 2 / 4 = 0.5 cycles
        //   per-element throughput = 4 units * 4 lanes = 16 elements/cycle
        //
        // A scalar ADD processes 1 element in 1 cycle:
        //   per-element latency = 1.0 cycle
        //   per-element throughput = 6.0 elements/cycle
        //
        // Without this normalization, scalar always wins because NEON
        // costs are compared as whole-vector costs against single-element
        // scalar costs.
        let per_element_latency = lat as f64 / lanes;
        let per_element_throughput = tp * lanes;

        // NEON energy: ~1.5-2.0x scalar ALU (wider datapath, same voltage).
        // 128-bit operations use ~1.8x energy of scalar; 64-bit ~1.4x.
        // Normalize per-element: one NEON instruction's energy is shared
        // across all lanes.
        let total_energy = if arrangement.width_bits() >= 128 { 1.8 } else { 1.4 };
        let per_element_energy = total_energy / lanes;

        CostEstimate {
            latency_cycles: per_element_latency,
            throughput_per_cycle: per_element_throughput,
            energy_relative: per_element_energy,
        }
    }

    fn estimate_gpu(&self, op: &str, width: u32) -> CostEstimate {
        // GPU cost model: high dispatch overhead, massive throughput.
        //
        // Apple M1 GPU: 128 execution units, 1024 ALUs at ~1.3 GHz.
        // Metal compute dispatch overhead: ~5000-15000 CPU-equivalent cycles
        // (includes command buffer encoding, GPU scheduling, fence).
        //
        // For small workloads, dispatch overhead dominates. For large
        // workloads (>10K elements), GPU throughput dominates.

        let dispatch_overhead: f64 = 10000.0; // ~10K CPU-equivalent cycles

        // GPU throughput per operation type (in GPU-ops per CPU-cycle,
        // accounting for GPU's lower clock rate but massive parallelism).
        let upper = op.to_ascii_uppercase();
        let gpu_ops_per_cycle = match upper.as_str() {
            "ADD" | "SUB" | "NEG" | "AND" | "ORR" | "OR"
            | "EOR" | "XOR" | "BIC" | "SHL" | "USHR" | "SSHR" => 512.0,
            "MUL" => 256.0,
            "FADD" | "FSUB" | "FMUL" | "FMA" | "FMLA" => 256.0,
            "FDIV" => 32.0,
            "GEMM" | "MATMUL" => 512.0, // matrix multiply is GPU's strength
            _ => 128.0, // conservative
        };

        // Elements implied by width (assume 32-bit elements)
        let elements = (width / 32).max(1) as f64;

        // Amortized latency: dispatch + elements/throughput
        let compute_cycles = elements / gpu_ops_per_cycle;
        let total_latency = dispatch_overhead + compute_cycles;

        // Effective throughput: elements / total_latency
        let effective_tp = elements / total_latency;

        // GPU energy: ~0.4x per element (amortized) plus fixed overhead.
        let energy = 0.4 * elements + 50.0; // fixed + per-element
        let energy_per_op = energy / elements;

        CostEstimate {
            latency_cycles: total_latency,
            throughput_per_cycle: effective_tp,
            energy_relative: energy_per_op,
        }
    }

    fn estimate_ane(&self, op: &str, width: u32) -> CostEstimate {
        // ANE cost model: very high compile/setup overhead, extreme throughput
        // for supported operations (GEMM, convolution, elementwise).
        //
        // Apple Neural Engine (M1): 16-core, 11 TOPS (int8).
        // CoreML compilation overhead: ~50K-200K CPU-equivalent cycles
        // for model compilation + input tensor preparation.
        //
        // Only certain operations are accelerated. Unsupported ops fall
        // back to CPU with the full compilation overhead wasted.

        let compile_overhead: f64 = 100_000.0; // ~100K CPU-equivalent cycles

        let upper = op.to_ascii_uppercase();
        let (supported, ane_ops_per_cycle) = match upper.as_str() {
            // Matrix operations: ANE's strength
            "GEMM" | "MATMUL" | "CONV2D" | "CONV" => (true, 2048.0),
            // Elementwise operations: supported but not ANE's strength
            "ADD" | "SUB" | "MUL" | "FADD" | "FMUL" | "FMA" | "FMLA" => (true, 512.0),
            "NEG" | "AND" | "ORR" | "OR" | "EOR" | "XOR" | "BIC" => (true, 256.0),
            // Shift operations
            "SHL" | "USHR" | "SSHR" => (true, 256.0),
            // Division: not well-supported on ANE
            "DIV" | "FDIV" | "SDIV" | "UDIV" => (false, 0.0),
            _ => (false, 0.0),
        };

        if !supported {
            // Unsupported op: return worst-case (falls back to CPU scalar)
            return CostEstimate {
                latency_cycles: compile_overhead + 100.0,
                throughput_per_cycle: 0.001,
                energy_relative: 100.0,
            };
        }

        let elements = (width / 32).max(1) as f64;
        let compute_cycles = elements / ane_ops_per_cycle;
        let total_latency = compile_overhead + compute_cycles;
        let effective_tp = elements / total_latency;

        // ANE energy: ~0.15x per element (fixed-function, very efficient)
        let energy_per_op = 0.15;

        CostEstimate {
            latency_cycles: total_latency,
            throughput_per_cycle: effective_tp,
            energy_relative: energy_per_op,
        }
    }
}

// ---------------------------------------------------------------------------
// Precision — data element precision for target legality decisions
// ---------------------------------------------------------------------------

/// Data element precision for target legality and profitability analysis.
///
/// Different compute targets support different precisions. ANE is limited to
/// FP16/INT8 (BF16 on M4+), while CPU and GPU support all precisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    FP16,
    FP32,
    FP64,
    BF16,
    INT8,
    INT16,
    INT32,
    INT64,
}

// ---------------------------------------------------------------------------
// GpuThresholds — minimum data sizes for profitable GPU dispatch
// ---------------------------------------------------------------------------

/// GPU profitability thresholds for Apple Silicon.
///
/// Below these thresholds, NEON is faster than GPU dispatch.
/// Values derived from: dispatch overhead ~10K cycles, NEON 4-way FP/SIMD,
/// GPU 256-512 ALU ops/cycle, transfer overhead ~10K cycles round-trip.
///
/// Source: designs/2026-04-14-profitability-thresholds.md
#[derive(Debug, Clone, Copy)]
pub struct GpuThresholds {
    /// Minimum elements for element-wise operations (add, mul, etc.)
    /// Below this: use NEON. Above: GPU is profitable.
    pub elementwise_min_elements: u64,
    /// Minimum total FLOP count for GEMM dispatch.
    /// GEMM FLOPS = 2*M*N*K. Below this: use NEON FMA loop.
    pub gemm_min_flops: u64,
    /// Minimum elements for reduction operations.
    /// GPU parallel reduce has high overhead; NEON horizontal adds are fast.
    pub reduction_min_elements: u64,
    /// Minimum data size in bytes for ANY GPU dispatch.
    /// Absolute floor below which GPU overhead can never be amortized.
    pub absolute_min_bytes: u64,
    /// GPU dispatch overhead in cycles (command buffer + scheduling + fence).
    pub dispatch_overhead_cycles: u64,
    /// GPU round-trip transfer overhead in cycles (for unified memory).
    pub transfer_overhead_cycles: u64,
}

impl Default for GpuThresholds {
    fn default() -> Self {
        Self {
            elementwise_min_elements: 4096,
            gemm_min_flops: 32768,
            reduction_min_elements: 8192,
            absolute_min_bytes: 4096,
            dispatch_overhead_cycles: 10000,
            transfer_overhead_cycles: 10000,
        }
    }
}

// ---------------------------------------------------------------------------
// AneThresholds — minimum data sizes for profitable ANE dispatch
// ---------------------------------------------------------------------------

/// ANE profitability thresholds for Apple Silicon.
///
/// ANE has very high dispatch overhead (CoreML compilation) but extreme
/// throughput for supported operations. Only profitable for large workloads.
///
/// Source: designs/2026-04-14-profitability-thresholds.md
#[derive(Debug, Clone, Copy)]
pub struct AneThresholds {
    /// Minimum FLOP count for standalone GEMM.
    pub gemm_min_flops: u64,
    /// Minimum FLOP count for fused Conv-BN-ReLU patterns.
    /// Lower than standalone because fusion amortizes overhead.
    pub fused_conv_min_flops: u64,
    /// Minimum elements for standalone element-wise operations.
    /// Very high: ANE dispatch overhead makes small element-wise unprofitable.
    pub elementwise_min_elements: u64,
    /// Minimum elements for reduction operations.
    pub reduction_min_elements: u64,
    /// Minimum data size in bytes for ANY ANE dispatch.
    pub absolute_min_bytes: u64,
    /// ANE dispatch overhead in cycles (CoreML compile + model load).
    pub dispatch_overhead_cycles: u64,
    /// ANE round-trip transfer overhead in cycles.
    pub transfer_overhead_cycles: u64,
    /// Minimum batch size for ANE profitability (batch=1 often slower than GPU).
    pub min_batch_size: u32,
}

impl Default for AneThresholds {
    fn default() -> Self {
        Self {
            gemm_min_flops: 131072,
            fused_conv_min_flops: 65536,
            elementwise_min_elements: 65536,
            reduction_min_elements: 32768,
            absolute_min_bytes: 32768,
            dispatch_overhead_cycles: 100000,
            transfer_overhead_cycles: 100000,
            min_batch_size: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// OperationCategory — classify operations for threshold selection
// ---------------------------------------------------------------------------

/// Operation category for profitability threshold selection.
///
/// Different operation categories have different crossover points where
/// accelerator dispatch becomes profitable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Element-wise arithmetic (add, sub, mul, div, neg).
    Elementwise,
    /// General matrix multiply (GEMM, matmul, batched_matmul).
    Gemm,
    /// Convolution operations (conv1d, conv2d, depthwise, transposed).
    Convolution,
    /// Reduction operations (sum, mean, max, min).
    Reduction,
    /// Activation functions (relu, sigmoid, tanh, gelu, silu).
    Activation,
    /// Normalization (batch_norm, layer_norm, group_norm).
    Normalization,
    /// Bitwise/logical operations (and, or, xor, bic, shifts).
    BitwiseLogic,
    /// Other or unrecognized operations.
    Other,
}

// ---------------------------------------------------------------------------
// ProfitabilityAnalyzer — combines legality, thresholds, and cost model
// ---------------------------------------------------------------------------

/// Profitability analyzer that combines legality, thresholds, and cost model.
///
/// Sits between the ProofAnalyzer (which determines what is LEGAL from a
/// verification perspective) and the cost model (which determines what is
/// CHEAPEST). This analyzer filters out legal-but-unprofitable targets and
/// checks hardware-level operation support.
///
/// Source: designs/2026-04-14-profitability-thresholds.md
pub struct ProfitabilityAnalyzer {
    gpu_thresholds: GpuThresholds,
    ane_thresholds: AneThresholds,
    cost_model: MultiTargetCostModel,
}

impl ProfitabilityAnalyzer {
    /// Create a new profitability analyzer for the given Apple Silicon generation.
    pub fn new(generation: CostModelGen) -> Self {
        Self {
            gpu_thresholds: GpuThresholds::default(),
            ane_thresholds: AneThresholds::default(),
            cost_model: MultiTargetCostModel::new(generation),
        }
    }

    /// Create a profitability analyzer with custom thresholds.
    pub fn with_thresholds(
        generation: CostModelGen,
        gpu: GpuThresholds,
        ane: AneThresholds,
    ) -> Self {
        Self {
            gpu_thresholds: gpu,
            ane_thresholds: ane,
            cost_model: MultiTargetCostModel::new(generation),
        }
    }

    /// Classify an operation string into an [`OperationCategory`].
    pub fn classify_op(&self, op: &str) -> OperationCategory {
        let upper = op.to_ascii_uppercase();
        match upper.as_str() {
            // Elementwise arithmetic
            "ADD" | "SUB" | "MUL" | "DIV" | "NEG"
            | "FADD" | "FSUB" | "FMUL" | "FDIV" | "FMA" | "FMLA" | "FMADD"
            | "FNEG" => OperationCategory::Elementwise,

            // GEMM / matrix multiply
            "GEMM" | "MATMUL" | "BATCHED_MATMUL" => OperationCategory::Gemm,

            // Convolution
            "CONV1D" | "CONV2D" | "DEPTHWISE_CONV2D" | "TRANSPOSED_CONV2D"
            | "CONV" => OperationCategory::Convolution,

            // Reduction
            "REDUCE_SUM" | "REDUCE_MEAN" | "REDUCE_MAX" | "REDUCE_MIN"
            | "SUM" | "MEAN" | "REDUCE" => OperationCategory::Reduction,

            // Activation functions
            "RELU" | "LEAKY_RELU" | "SIGMOID" | "TANH" | "GELU" | "SILU"
            => OperationCategory::Activation,

            // Normalization
            "BATCH_NORM" | "LAYER_NORM" | "GROUP_NORM"
            => OperationCategory::Normalization,

            // Bitwise / logical
            "AND" | "OR" | "ORR" | "XOR" | "EOR" | "BIC" | "ORN"
            | "SHL" | "LSL" | "USHR" | "LSR" | "SSHR" | "ASR"
            | "NOT" => OperationCategory::BitwiseLogic,

            // Integer division (scalar-only in practice)
            "SDIV" | "UDIV" => OperationCategory::Other,

            _ => OperationCategory::Other,
        }
    }

    /// Check whether a compute target supports the given operation.
    ///
    /// This is a hardware-level legality check, independent of proof-based
    /// legality (which is handled by ProofAnalyzer).
    ///
    /// - CPU Scalar: supports all operations (universal fallback).
    /// - NEON: no integer division, no 64-bit integer MUL.
    /// - GPU: supports all standard arithmetic/matrix operations.
    /// - ANE: restricted to tensor operations (GEMM, conv, activations,
    ///   elementwise tensor ops, normalization, reductions, data movement).
    ///   Does NOT support bitwise/logical ops or general integer arithmetic.
    pub fn target_legality(&self, op: &str, target: ComputeTarget) -> bool {
        let upper = op.to_ascii_uppercase();
        match target {
            ComputeTarget::CpuScalar => true,
            ComputeTarget::Neon => {
                // NEON does not support integer division
                !matches!(
                    upper.as_str(),
                    "SDIV" | "UDIV" | "DIV"
                )
            }
            ComputeTarget::Gpu => {
                // GPU supports all standard arithmetic and matrix ops via
                // Metal compute shaders. Only truly illegal operations are
                // things requiring recursion or dynamic allocation, which
                // we don't model at the operation level.
                true
            }
            ComputeTarget::Ane => {
                // ANE: fixed-function tensor operations only.
                matches!(
                    upper.as_str(),
                    "GEMM" | "MATMUL" | "BATCHED_MATMUL"
                    | "CONV1D" | "CONV2D" | "DEPTHWISE_CONV2D" | "TRANSPOSED_CONV2D"
                    | "MAXPOOL" | "AVGPOOL" | "GLOBAL_AVGPOOL" | "GLOBAL_MAXPOOL"
                    | "BATCH_NORM" | "LAYER_NORM" | "GROUP_NORM"
                    | "RELU" | "LEAKY_RELU" | "SIGMOID" | "TANH" | "GELU" | "SILU"
                    | "ADD" | "SUB" | "MUL" | "DIV"
                    | "FADD" | "FSUB" | "FMUL" | "FDIV"
                    | "FMA" | "FMLA" | "FMADD"
                    | "REDUCE_SUM" | "REDUCE_MEAN" | "REDUCE_MAX" | "REDUCE_MIN"
                    | "RESHAPE" | "TRANSPOSE" | "PERMUTE" | "CONCAT" | "SPLIT"
                    | "ATTENTION" | "SCALED_DOT_PRODUCT_ATTENTION"
                )
            }
        }
    }

    /// Check whether GPU dispatch is profitable for the given operation and data size.
    ///
    /// Returns `true` if the data size exceeds the GPU profitability threshold
    /// for the operation category. GPU dispatch has ~10K cycles of overhead,
    /// so small workloads are better served by NEON.
    pub fn is_gpu_profitable(
        &self,
        op: &str,
        data_size_bytes: u64,
        element_count: u64,
    ) -> bool {
        // Absolute minimum data size floor
        if data_size_bytes < self.gpu_thresholds.absolute_min_bytes {
            return false;
        }

        let category = self.classify_op(op);
        match category {
            OperationCategory::Elementwise | OperationCategory::Activation => {
                element_count >= self.gpu_thresholds.elementwise_min_elements
            }
            OperationCategory::Gemm => {
                // For GEMM, we check FLOP count rather than element count.
                // Approximate: flops ~ element_count (caller should pass 2*M*N*K).
                element_count >= self.gpu_thresholds.gemm_min_flops
            }
            OperationCategory::Convolution => {
                // Convolution thresholds similar to GEMM
                element_count >= self.gpu_thresholds.gemm_min_flops
            }
            OperationCategory::Reduction => {
                element_count >= self.gpu_thresholds.reduction_min_elements
            }
            OperationCategory::Normalization => {
                element_count >= self.gpu_thresholds.elementwise_min_elements
            }
            OperationCategory::BitwiseLogic => {
                // Bitwise ops are very cheap on CPU; GPU rarely profitable
                element_count >= self.gpu_thresholds.elementwise_min_elements * 4
            }
            OperationCategory::Other => false,
        }
    }

    /// Check whether ANE dispatch is profitable for the given operation and tensor shape.
    ///
    /// The `tensor_shape` slice represents the tensor dimensions (e.g., [batch, channels, H, W]).
    /// ANE requires large workloads due to ~100K cycle CoreML compilation overhead.
    pub fn is_ane_profitable(
        &self,
        op: &str,
        data_size_bytes: u64,
        tensor_shape: &[u64],
    ) -> bool {
        // Absolute minimum data size floor
        if data_size_bytes < self.ane_thresholds.absolute_min_bytes {
            return false;
        }

        // ANE operation must be legal first
        if !self.target_legality(op, ComputeTarget::Ane) {
            return false;
        }

        // Compute total element count from tensor shape
        let total_elements: u64 = tensor_shape.iter().product();
        if total_elements == 0 {
            return false;
        }

        let category = self.classify_op(op);
        match category {
            OperationCategory::Gemm => {
                // For GEMM, compute FLOP estimate from shape.
                // Shape [M, N] -> FLOPs = 2*M*N (simplified; real GEMM is 2*M*N*K).
                // If shape has 2 dims, treat as M*N matrix; FLOPs ~ 2*M*N.
                let flops = if tensor_shape.len() >= 2 {
                    2 * tensor_shape.iter().product::<u64>()
                } else {
                    2 * total_elements
                };
                flops >= self.ane_thresholds.gemm_min_flops
            }
            OperationCategory::Convolution | OperationCategory::Normalization => {
                // Fused conv patterns have lower thresholds
                total_elements >= self.ane_thresholds.fused_conv_min_flops
            }
            OperationCategory::Elementwise
            | OperationCategory::Activation => {
                total_elements >= self.ane_thresholds.elementwise_min_elements
            }
            OperationCategory::Reduction => {
                total_elements >= self.ane_thresholds.reduction_min_elements
            }
            OperationCategory::BitwiseLogic => {
                // ANE doesn't support bitwise operations
                false
            }
            OperationCategory::Other => false,
        }
    }

    /// Recommend dispatch targets for an operation, ranked by estimated cost.
    ///
    /// Checks legality and profitability for each target, then estimates
    /// cost and returns candidates sorted cheapest-first. CPU scalar is
    /// always included as a fallback.
    ///
    /// # Arguments
    /// - `op`: Operation name (case-insensitive).
    /// - `data_size_bytes`: Total data size in bytes.
    /// - `tensor_shape`: Tensor dimensions (e.g., [batch, C, H, W]). Pass
    ///   `&[element_count]` for flat arrays.
    ///
    /// # Returns
    /// A vector of `(ComputeTarget, CostEstimate)` pairs sorted by
    /// `latency_cycles` (cheapest first). Always contains at least CpuScalar.
    pub fn recommend_dispatch(
        &self,
        op: &str,
        data_size_bytes: u64,
        tensor_shape: &[u64],
    ) -> Vec<(ComputeTarget, CostEstimate)> {
        let total_elements: u64 = tensor_shape.iter().product();
        // Width in bits for the cost model: total data in bits.
        // Use 32-bit elements as default assumption for GPU/ANE width param.
        let width_bits = (data_size_bytes * 8) as u32;

        let mut candidates: Vec<(ComputeTarget, CostEstimate)> = Vec::new();

        // CPU Scalar: always legal, always included
        let scalar_est = self.cost_model.estimate_cost(
            ComputeTarget::CpuScalar,
            op,
            32, // per-element width
        );
        // Scale scalar cost by element count for total cost comparison
        let scalar_total = CostEstimate {
            latency_cycles: scalar_est.latency_cycles * total_elements.max(1) as f64,
            throughput_per_cycle: scalar_est.throughput_per_cycle,
            energy_relative: scalar_est.energy_relative,
        };
        candidates.push((ComputeTarget::CpuScalar, scalar_total));

        // NEON: check legality
        if self.target_legality(op, ComputeTarget::Neon) && total_elements > 1 {
            let neon_est = self.cost_model.estimate_cost(
                ComputeTarget::Neon,
                op,
                128, // 4S arrangement
            );
            let tc = self.cost_model.transfer_costs();
            let lanes = 4.0_f64; // 4S arrangement
            let vectors_needed = (total_elements as f64 / lanes).ceil();
            let transfer = vectors_needed * (tc.memory_to_neon_cycles + tc.neon_to_memory_cycles);
            let neon_total = CostEstimate {
                latency_cycles: neon_est.latency_cycles * total_elements.max(1) as f64 + transfer,
                throughput_per_cycle: neon_est.throughput_per_cycle,
                energy_relative: neon_est.energy_relative,
            };
            candidates.push((ComputeTarget::Neon, neon_total));
        }

        // GPU: check legality + profitability
        if self.target_legality(op, ComputeTarget::Gpu)
            && self.is_gpu_profitable(op, data_size_bytes, total_elements)
        {
            let gpu_est = self.cost_model.estimate_cost(
                ComputeTarget::Gpu,
                op,
                width_bits.max(32),
            );
            candidates.push((ComputeTarget::Gpu, gpu_est));
        }

        // ANE: check legality + profitability
        if self.target_legality(op, ComputeTarget::Ane)
            && self.is_ane_profitable(op, data_size_bytes, tensor_shape)
        {
            let ane_est = self.cost_model.estimate_cost(
                ComputeTarget::Ane,
                op,
                width_bits.max(32),
            );
            candidates.push((ComputeTarget::Ane, ane_est));
        }

        // Sort by latency (cheapest first)
        candidates.sort_by(|a, b| {
            a.1.latency_cycles
                .partial_cmp(&b.1.latency_cycles)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
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

    // ==== Multi-target cost model tests ====

    fn mt_m1() -> MultiTargetCostModel {
        MultiTargetCostModel::new(CostModelGen::M1)
    }

    fn mt_m4() -> MultiTargetCostModel {
        MultiTargetCostModel::new(CostModelGen::M4)
    }

    // ---- NeonArrangement tests ----

    #[test]
    fn neon_arrangement_width_bits() {
        assert_eq!(NeonArrangement::B8.width_bits(), 64);
        assert_eq!(NeonArrangement::B16.width_bits(), 128);
        assert_eq!(NeonArrangement::H4.width_bits(), 64);
        assert_eq!(NeonArrangement::H8.width_bits(), 128);
        assert_eq!(NeonArrangement::S2.width_bits(), 64);
        assert_eq!(NeonArrangement::S4.width_bits(), 128);
        assert_eq!(NeonArrangement::D1.width_bits(), 64);
        assert_eq!(NeonArrangement::D2.width_bits(), 128);
    }

    #[test]
    fn neon_arrangement_element_bits() {
        assert_eq!(NeonArrangement::B8.element_bits(), 8);
        assert_eq!(NeonArrangement::B16.element_bits(), 8);
        assert_eq!(NeonArrangement::H4.element_bits(), 16);
        assert_eq!(NeonArrangement::H8.element_bits(), 16);
        assert_eq!(NeonArrangement::S2.element_bits(), 32);
        assert_eq!(NeonArrangement::S4.element_bits(), 32);
        assert_eq!(NeonArrangement::D1.element_bits(), 64);
        assert_eq!(NeonArrangement::D2.element_bits(), 64);
    }

    #[test]
    fn neon_arrangement_lane_count() {
        assert_eq!(NeonArrangement::B8.lane_count(), 8);
        assert_eq!(NeonArrangement::B16.lane_count(), 16);
        assert_eq!(NeonArrangement::H4.lane_count(), 4);
        assert_eq!(NeonArrangement::H8.lane_count(), 8);
        assert_eq!(NeonArrangement::S2.lane_count(), 2);
        assert_eq!(NeonArrangement::S4.lane_count(), 4);
        assert_eq!(NeonArrangement::D1.lane_count(), 1);
        assert_eq!(NeonArrangement::D2.lane_count(), 2);
    }

    #[test]
    fn neon_arrangement_from_width() {
        assert_eq!(NeonArrangement::from_width(64), Some(NeonArrangement::D1));
        assert_eq!(NeonArrangement::from_width(128), Some(NeonArrangement::S4));
        assert_eq!(NeonArrangement::from_width(256), None);
    }

    // ---- NeonOp parsing ----

    #[test]
    fn neon_op_from_name_known() {
        assert_eq!(NeonOp::from_name("ADD"), Some(NeonOp::Add));
        assert_eq!(NeonOp::from_name("sub"), Some(NeonOp::Sub));
        assert_eq!(NeonOp::from_name("Mul"), Some(NeonOp::Mul));
        assert_eq!(NeonOp::from_name("neg"), Some(NeonOp::Neg));
        assert_eq!(NeonOp::from_name("AND"), Some(NeonOp::And));
        assert_eq!(NeonOp::from_name("orr"), Some(NeonOp::Orr));
        assert_eq!(NeonOp::from_name("OR"), Some(NeonOp::Orr));
        assert_eq!(NeonOp::from_name("eor"), Some(NeonOp::Eor));
        assert_eq!(NeonOp::from_name("XOR"), Some(NeonOp::Eor));
        assert_eq!(NeonOp::from_name("bic"), Some(NeonOp::Bic));
        assert_eq!(NeonOp::from_name("SHL"), Some(NeonOp::Shl));
        assert_eq!(NeonOp::from_name("USHR"), Some(NeonOp::Ushr));
        assert_eq!(NeonOp::from_name("SSHR"), Some(NeonOp::Sshr));
        assert_eq!(NeonOp::from_name("FADD"), Some(NeonOp::Fadd));
        assert_eq!(NeonOp::from_name("FMUL"), Some(NeonOp::Fmul));
        assert_eq!(NeonOp::from_name("FMLA"), Some(NeonOp::Fmla));
        assert_eq!(NeonOp::from_name("FMA"), Some(NeonOp::Fmla));
    }

    #[test]
    fn neon_op_from_name_unknown() {
        assert_eq!(NeonOp::from_name("SDIV"), None);
        assert_eq!(NeonOp::from_name("UNKNOWN"), None);
        assert_eq!(NeonOp::from_name(""), None);
    }

    // ---- NEON cost data ----

    #[test]
    fn neon_int_alu_latency_is_2() {
        let model = mt_m1();
        let ops = [NeonOp::Add, NeonOp::Sub, NeonOp::Neg,
                    NeonOp::And, NeonOp::Orr, NeonOp::Eor, NeonOp::Bic,
                    NeonOp::Shl, NeonOp::Ushr, NeonOp::Sshr];
        for op in &ops {
            let (lat, tp) = model.neon_cost(*op, NeonArrangement::S4);
            assert_eq!(lat, 2, "{:?} should have 2-cycle NEON latency", op);
            assert!((tp - 4.0).abs() < 0.01,
                "{:?} should have 4.0 throughput", op);
        }
    }

    #[test]
    fn neon_mul_latency_varies_by_element_size() {
        let model = mt_m1();
        // 8-bit MUL: 3 cycles
        let (lat_8b, _) = model.neon_cost(NeonOp::Mul, NeonArrangement::B16);
        assert_eq!(lat_8b, 3);
        // 16-bit MUL: 4 cycles
        let (lat_16b, _) = model.neon_cost(NeonOp::Mul, NeonArrangement::H8);
        assert_eq!(lat_16b, 4);
        // 32-bit MUL: 4 cycles
        let (lat_32b, _) = model.neon_cost(NeonOp::Mul, NeonArrangement::S4);
        assert_eq!(lat_32b, 4);
    }

    #[test]
    fn neon_fp_latencies() {
        let model = mt_m1();
        let (fadd_lat, _) = model.neon_cost(NeonOp::Fadd, NeonArrangement::S4);
        let (fmul_lat, _) = model.neon_cost(NeonOp::Fmul, NeonArrangement::S4);
        let (fmla_lat, _) = model.neon_cost(NeonOp::Fmla, NeonArrangement::S4);
        assert_eq!(fadd_lat, 3);
        assert_eq!(fmul_lat, 3);
        assert_eq!(fmla_lat, 4);
    }

    // ---- CostEstimate ----

    #[test]
    fn cost_estimate_efficiency() {
        let est = CostEstimate {
            latency_cycles: 2.0,
            throughput_per_cycle: 4.0,
            energy_relative: 2.0,
        };
        assert!((est.efficiency() - 2.0).abs() < 0.01);
    }

    #[test]
    fn cost_estimate_efficiency_zero_energy() {
        let est = CostEstimate {
            latency_cycles: 1.0,
            throughput_per_cycle: 4.0,
            energy_relative: 0.0,
        };
        assert_eq!(est.efficiency(), 0.0);
    }

    // ---- Cross-target cost comparisons ----

    #[test]
    fn scalar_mul_vs_neon_mul_128bit() {
        // NEON MUL.4S processes 4x32-bit elements in one instruction.
        // Scalar MUL processes 1x32/64-bit element.
        // NEON should have lower per-element latency for 128-bit width.
        let model = mt_m1();
        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "MUL", 32);
        let neon = model.estimate_cost(ComputeTarget::Neon, "MUL", 128);

        // Scalar MUL: 3 cycles for 1 element
        assert!((scalar.latency_cycles - 3.0).abs() < 0.01);
        // NEON MUL.4S: 4 cycles / 4 lanes = 1.0 cycle per element
        assert!((neon.latency_cycles - 1.0).abs() < 0.01,
            "NEON MUL per-element latency should be 1.0, got {}", neon.latency_cycles);
        // NEON per-element latency should be lower than scalar
        assert!(neon.latency_cycles < scalar.latency_cycles,
            "NEON per-element latency {} should beat scalar {}",
            neon.latency_cycles, scalar.latency_cycles);
        // NEON has higher per-element throughput (4 units * 4 lanes = 16 vs 2)
        assert!(neon.throughput_per_cycle > scalar.throughput_per_cycle);
    }

    #[test]
    fn scalar_add_vs_neon_add_128bit() {
        let model = mt_m1();
        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);
        let neon = model.estimate_cost(ComputeTarget::Neon, "ADD", 128);

        // Scalar ADD: 1 cycle per element
        assert!((scalar.latency_cycles - 1.0).abs() < 0.01);
        // NEON ADD.4S: 2 cycles / 4 lanes = 0.5 cycles per element
        assert!((neon.latency_cycles - 0.5).abs() < 0.01,
            "NEON ADD per-element latency should be 0.5, got {}", neon.latency_cycles);
        // NEON per-element latency should be lower than scalar
        assert!(neon.latency_cycles < scalar.latency_cycles,
            "NEON ADD per-element {} should beat scalar ADD {}",
            neon.latency_cycles, scalar.latency_cycles);
    }

    #[test]
    fn gpu_high_dispatch_overhead() {
        // GPU dispatch overhead should make it expensive for small workloads
        let model = mt_m1();
        let gpu_small = model.estimate_cost(ComputeTarget::Gpu, "MUL", 128);
        let neon = model.estimate_cost(ComputeTarget::Neon, "MUL", 128);

        // GPU latency for 4 elements should be much higher than NEON
        // (dominated by ~10K dispatch overhead)
        assert!(gpu_small.latency_cycles > 1000.0);
        assert!(neon.latency_cycles < 10.0);
    }

    #[test]
    fn ane_high_compile_overhead() {
        // ANE has very high compilation overhead
        let model = mt_m1();
        let ane = model.estimate_cost(ComputeTarget::Ane, "MUL", 128);

        // Should be ~100K+ cycles for tiny workload
        assert!(ane.latency_cycles > 50_000.0);
    }

    #[test]
    fn ane_unsupported_op_is_expensive() {
        let model = mt_m1();
        // Use a large workload to show the throughput difference clearly
        let large_width = 100_000 * 32;
        let ane_div = model.estimate_cost(ComputeTarget::Ane, "DIV", large_width);
        let ane_mul = model.estimate_cost(ComputeTarget::Ane, "MUL", large_width);

        // Unsupported ops should have much worse energy and efficiency
        assert!(ane_div.energy_relative > ane_mul.energy_relative,
            "DIV energy {} should exceed MUL energy {}", ane_div.energy_relative, ane_mul.energy_relative);
        assert!(ane_div.efficiency() < ane_mul.efficiency(),
            "DIV efficiency {} should be worse than MUL efficiency {}", ane_div.efficiency(), ane_mul.efficiency());
    }

    #[test]
    fn gpu_amortizes_for_large_workloads() {
        // For large workloads, GPU throughput should dominate dispatch overhead
        let model = mt_m1();
        let gpu_large = model.estimate_cost(ComputeTarget::Gpu, "ADD", 1_000_000 * 32);
        let scalar_large = model.estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);

        // GPU effective throughput for 1M elements should exceed scalar
        assert!(gpu_large.throughput_per_cycle > 0.0);
        // But scalar still has lower latency per individual operation
        assert!(scalar_large.latency_cycles < gpu_large.latency_cycles);
    }

    // ---- Data transfer costs ----

    #[test]
    fn transfer_costs_gpr_simd_expensive() {
        let model = mt_m1();
        let tc = model.transfer_costs();

        // GPR<->SIMD should be 12-13 cycles (very expensive)
        assert!(tc.scalar_to_neon_cycles >= 11.0);
        assert!(tc.neon_to_scalar_cycles >= 12.0);
        // Memory<->NEON should be comparable to scalar load/store
        assert!(tc.memory_to_neon_cycles <= 5.0);
        assert!(tc.neon_to_memory_cycles <= 2.0);
    }

    #[test]
    fn transfer_costs_gpu_moderate() {
        let model = mt_m1();
        let tc = model.transfer_costs();

        // CPU<->GPU: hundreds of cycles (unified memory, no copy)
        assert!(tc.cpu_to_gpu_cycles >= 100.0);
        assert!(tc.gpu_to_cpu_cycles >= 100.0);
    }

    #[test]
    fn transfer_costs_ane_high() {
        let model = mt_m1();
        let tc = model.transfer_costs();

        // CPU<->ANE: thousands of cycles (CoreML marshalling)
        assert!(tc.cpu_to_ane_cycles >= 1000.0);
        assert!(tc.ane_to_cpu_cycles >= 500.0);
    }

    #[test]
    fn m4_slightly_better_cross_domain() {
        let m1_tc = mt_m1().transfer_costs();
        let m4_tc = mt_m4().transfer_costs();

        // M4 should have slightly lower cross-domain transfer costs
        assert!(m4_tc.scalar_to_neon_cycles <= m1_tc.scalar_to_neon_cycles);
        assert!(m4_tc.neon_to_scalar_cycles <= m1_tc.neon_to_scalar_cycles);
    }

    // ---- NEON per-element energy lower than scalar (amortized) ----

    #[test]
    fn neon_per_element_energy_lower_than_scalar() {
        // NEON uses more total energy per instruction (~1.8x) but processes
        // multiple lanes. Per-element energy is 1.8 / 4 = 0.45 for 4S,
        // which is LESS than scalar's 1.0 per element.
        let model = mt_m1();
        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);
        let neon = model.estimate_cost(ComputeTarget::Neon, "ADD", 128);

        assert!(neon.energy_relative < scalar.energy_relative,
            "NEON per-element energy {} should be lower than scalar {} (amortized across 4 lanes)",
            neon.energy_relative, scalar.energy_relative);
        // 128-bit 4S: 1.8 / 4 = 0.45
        assert!((neon.energy_relative - 0.45).abs() < 0.01,
            "NEON 4S per-element energy should be ~0.45, got {}", neon.energy_relative);
    }

    // ---- ANE energy efficient for supported ops ----

    #[test]
    fn ane_energy_efficient_for_supported_ops() {
        let model = mt_m1();
        let ane = model.estimate_cost(ComputeTarget::Ane, "MUL", 128);
        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "MUL", 32);

        // ANE per-element energy should be lower than scalar
        assert!(ane.energy_relative < scalar.energy_relative,
            "ANE should be more energy-efficient per element");
    }

    // ---- All NEON ops return valid costs ----

    #[test]
    fn all_neon_ops_return_valid_costs() {
        let model = mt_m1();
        let ops = [
            NeonOp::Add, NeonOp::Sub, NeonOp::Mul, NeonOp::Neg,
            NeonOp::And, NeonOp::Orr, NeonOp::Eor, NeonOp::Bic,
            NeonOp::Shl, NeonOp::Ushr, NeonOp::Sshr,
            NeonOp::Fadd, NeonOp::Fmul, NeonOp::Fmla,
        ];
        let arrangements = [
            NeonArrangement::B8, NeonArrangement::B16,
            NeonArrangement::H4, NeonArrangement::H8,
            NeonArrangement::S2, NeonArrangement::S4,
            NeonArrangement::D1, NeonArrangement::D2,
        ];
        for op in &ops {
            for arr in &arrangements {
                let (lat, tp) = model.neon_cost(*op, *arr);
                assert!(lat > 0 && lat <= 10,
                    "{:?} on {:?} has unreasonable latency {}", op, arr, lat);
                assert!(tp > 0.0,
                    "{:?} on {:?} has non-positive throughput", op, arr);
            }
        }
    }

    // ---- Unified estimate_cost for all targets ----

    #[test]
    fn estimate_cost_all_targets_return_positive() {
        let model = mt_m1();
        let targets = [
            ComputeTarget::CpuScalar,
            ComputeTarget::Neon,
            ComputeTarget::Gpu,
            ComputeTarget::Ane,
        ];
        let ops = ["ADD", "SUB", "MUL", "NEG", "AND", "ORR", "EOR",
                    "BIC", "SHL", "USHR", "SSHR"];
        for target in &targets {
            for op in &ops {
                let est = model.estimate_cost(*target, op, 128);
                assert!(est.latency_cycles > 0.0,
                    "{:?}/{} has non-positive latency", target, op);
                assert!(est.throughput_per_cycle > 0.0,
                    "{:?}/{} has non-positive throughput", target, op);
                assert!(est.energy_relative > 0.0,
                    "{:?}/{} has non-positive energy", target, op);
            }
        }
    }

    // ---- Scalar model accessible through multi-target ----

    #[test]
    fn scalar_model_accessible() {
        let model = mt_m1();
        let scalar = model.scalar_model();
        assert_eq!(scalar.latency(AArch64Opcode::AddRR), 1);
        assert_eq!(scalar.latency(AArch64Opcode::MulRR), 3);
    }

    // ==== Lane-count normalization tests (issue #164) ====

    #[test]
    fn neon_4s_add_cheaper_per_element_than_scalar_for_4_elements() {
        // Core test for issue #164: NEON 4S ADD should be cheaper per-element
        // than scalar ADD when processing 4 elements.
        let model = mt_m1();
        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);
        let neon_4s = model.estimate_cost(ComputeTarget::Neon, "ADD", 128); // 4S

        // Scalar: 1.0 cycle/element, NEON 4S: 2 cycles / 4 lanes = 0.5 cycles/element
        assert!(neon_4s.latency_cycles < scalar.latency_cycles,
            "NEON 4S ADD per-element latency {} should be cheaper than scalar {}",
            neon_4s.latency_cycles, scalar.latency_cycles);

        // For 4 elements: scalar total = 4 * 1.0 = 4.0, NEON total = 4 * 0.5 = 2.0
        let scalar_total = scalar.latency_cycles * 4.0;
        let neon_total = neon_4s.latency_cycles * 4.0;
        assert!(neon_total < scalar_total,
            "NEON 4S total {} should be less than scalar total {} for 4 elements",
            neon_total, scalar_total);
    }

    #[test]
    fn neon_2d_add_vs_scalar_for_2_elements() {
        // NEON 2D processes 2x64-bit elements in one instruction.
        let model = mt_m1();
        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "ADD", 64);
        let _neon_s4 = model.estimate_cost(ComputeTarget::Neon, "ADD", 128); // maps to S4

        // Use D2 arrangement directly for 64-bit elements
        let neon_op = NeonOp::from_name("ADD").unwrap();
        let (lat, _tp) = model.neon_cost(neon_op, NeonArrangement::D2);
        let d2_lanes = NeonArrangement::D2.lane_count() as f64;
        let per_element_lat = lat as f64 / d2_lanes;

        // Scalar: 1 cycle for 64-bit ADD, NEON D2: 2 cycles / 2 lanes = 1.0
        assert!((per_element_lat - 1.0).abs() < 0.01,
            "NEON D2 per-element latency should be 1.0, got {}", per_element_lat);
        // For 2 elements: scalar = 2 * 1.0 = 2.0, NEON = 2 * 1.0 = 2.0
        // D2 matches scalar (tie) — NEON needs 3+ elements to clearly win for 64-bit
        assert!(per_element_lat <= scalar.latency_cycles,
            "NEON D2 per-element {} should not exceed scalar {}", per_element_lat, scalar.latency_cycles);
    }

    #[test]
    fn scalar_wins_for_single_element() {
        // For a single element, scalar should be at least as good as NEON
        // because NEON has no lane-count advantage for 1 element.
        let model = mt_m1();

        // D1 arrangement: 1 lane, so per-element cost = raw NEON cost
        let neon_op = NeonOp::from_name("ADD").unwrap();
        let (neon_lat, _) = model.neon_cost(neon_op, NeonArrangement::D1);
        let d1_per_element = neon_lat as f64 / NeonArrangement::D1.lane_count() as f64;

        let scalar = model.estimate_cost(ComputeTarget::CpuScalar, "ADD", 64);

        // NEON D1 = 2 cycles / 1 lane = 2.0 per element, scalar = 1.0
        assert!(scalar.latency_cycles < d1_per_element,
            "Scalar {} should beat NEON D1 {} for single element",
            scalar.latency_cycles, d1_per_element);
    }

    #[test]
    fn lane_count_all_arrangements() {
        // Verify lane_count() returns correct values for all arrangements.
        let cases: [(NeonArrangement, u32); 8] = [
            (NeonArrangement::B8,  8),
            (NeonArrangement::B16, 16),
            (NeonArrangement::H4,  4),
            (NeonArrangement::H8,  8),
            (NeonArrangement::S2,  2),
            (NeonArrangement::S4,  4),
            (NeonArrangement::D1,  1),
            (NeonArrangement::D2,  2),
        ];
        for (arr, expected_lanes) in &cases {
            assert_eq!(arr.lane_count(), *expected_lanes,
                "{:?} should have {} lanes", arr, expected_lanes);
            // Verify lane_count = width_bits / element_bits
            assert_eq!(arr.lane_count(), arr.width_bits() / arr.element_bits(),
                "{:?} lane_count should equal width/element", arr);
        }
    }

    #[test]
    fn neon_throughput_scales_with_lanes() {
        // Per-element throughput should scale with lane count.
        // NEON 4S (4 lanes) should have 4x the per-element throughput of D1 (1 lane).
        let model = mt_m1();
        let neon_4s = model.estimate_cost(ComputeTarget::Neon, "ADD", 128); // 4S
        // D1 is width=64 → from_width(64) = D1
        let neon_d1 = model.estimate_cost(ComputeTarget::Neon, "ADD", 64);  // D1

        // 4S: tp = 4.0 * 4 = 16.0, D1: tp = 4.0 * 1 = 4.0
        assert!((neon_4s.throughput_per_cycle / neon_d1.throughput_per_cycle - 4.0).abs() < 0.01,
            "4S throughput {} should be 4x D1 throughput {}",
            neon_4s.throughput_per_cycle, neon_d1.throughput_per_cycle);
    }

    // ---- recommend_target tests ----

    #[test]
    fn recommend_neon_for_mul_batch() {
        // MUL is expensive (3 cycles scalar), so NEON wins with fewer elements.
        // Scalar: 3.0 * 16 = 48.0
        // NEON 4S: per-element = 4/4 = 1.0, compute = 1.0 * 16 = 16.0,
        //   transfer = 4 vectors * 5 = 20.0, total = 36.0
        // 36.0 < 48.0 → NEON wins
        let model = mt_m1();
        let target = model.recommend_target("MUL", 32, 16);
        assert_eq!(target, ComputeTarget::Neon,
            "NEON should be recommended for 16x 32-bit MUL");
    }

    #[test]
    fn recommend_scalar_for_single_element() {
        // For a single element, scalar should win: no NEON amortization benefit,
        // and the NEON transfer overhead (5 cycles for 1 vector load+store)
        // exceeds any compute savings.
        let model = mt_m1();
        let target = model.recommend_target("MUL", 32, 1);
        assert_eq!(target, ComputeTarget::CpuScalar,
            "Scalar should be recommended for 1x 32-bit MUL");
    }

    #[test]
    fn recommend_neon_for_8bit_batch() {
        // 8-bit NEON processes 16 elements per instruction (B16).
        // NEON ADD: per-element latency = 2/16 = 0.125 cycles.
        // Scalar ADD: 1.0 cycle per element.
        // For 64 elements:
        //   Scalar total: 1.0 * 64 = 64.0
        //   NEON: compute = 0.125 * 64 = 8.0, transfer = 4 * 5 = 20.0, total = 28.0
        // 28.0 < 64.0 → NEON wins decisively
        let model = mt_m1();
        let target = model.recommend_target("ADD", 8, 64);
        assert_eq!(target, ComputeTarget::Neon,
            "NEON should be recommended for 64x 8-bit ADD");
    }

    #[test]
    fn recommend_neon_for_large_add_batch() {
        // For large batches, even ADD (cheap scalar op) should favor NEON
        // because transfer overhead is amortized across many elements.
        // Scalar: 1.0 * 256 = 256.0
        // NEON 4S: compute = 0.5 * 256 = 128.0, vectors = 64, transfer = 64 * 5 = 320.0
        // Wait, that's 448 > 256. The crossover for ADD happens when
        // n * 0.5 < n * 1.0 - ceil(n/4) * 5, i.e., 0.5n + 1.25n < n → never for ADD.
        //
        // For 8-bit ADD (16 lanes), transfer is far less per element:
        // 256 elements: compute = 0.125 * 256 = 32.0, vectors = 16, transfer = 80
        // total = 112 < 256 → NEON wins
        let model = mt_m1();
        let target = model.recommend_target("ADD", 8, 256);
        assert_eq!(target, ComputeTarget::Neon,
            "NEON should be recommended for 256x 8-bit ADD");
    }

    // ==== Profitability threshold tests (issue #171) ====

    fn pa_m1() -> ProfitabilityAnalyzer {
        ProfitabilityAnalyzer::new(CostModelGen::M1)
    }

    // ---- GPU profitability ----

    #[test]
    fn gpu_profitable_large_data() {
        let pa = pa_m1();
        // 8192 elements * 4 bytes = 32768 bytes > 4096 threshold
        assert!(
            pa.is_gpu_profitable("ADD", 32768, 8192),
            "GPU should be profitable for 8192 elements of ADD"
        );
    }

    #[test]
    fn gpu_not_profitable_small_data() {
        let pa = pa_m1();
        // 100 elements * 4 bytes = 400 bytes < 4096 threshold
        assert!(
            !pa.is_gpu_profitable("ADD", 400, 100),
            "GPU should NOT be profitable for 100 elements of ADD"
        );
    }

    #[test]
    fn gpu_not_profitable_256_elements() {
        let pa = pa_m1();
        // 256 elements < 4096 elementwise threshold even if bytes > absolute_min
        assert!(
            !pa.is_gpu_profitable("ADD", 4096, 256),
            "GPU should NOT be profitable for 256 elements of ADD (below 4096 threshold)"
        );
    }

    #[test]
    fn gpu_profitable_gemm_large_flops() {
        let pa = pa_m1();
        // GEMM with 32768+ flops -> profitable
        assert!(
            pa.is_gpu_profitable("GEMM", 65536, 32768),
            "GPU should be profitable for GEMM with 32768 flops"
        );
    }

    // ---- ANE profitability ----

    #[test]
    fn ane_profitable_large_gemm() {
        let pa = pa_m1();
        // GEMM with shape [128, 128] -> 2*128*128 = 32768 flops... but threshold
        // is 131072. Need larger shape: [256, 256] -> 2*65536 = 131072
        assert!(
            pa.is_ane_profitable("GEMM", 131072, &[256, 256]),
            "ANE should be profitable for GEMM [256, 256]"
        );
    }

    #[test]
    fn ane_not_profitable_small_gemm() {
        let pa = pa_m1();
        // GEMM with shape [8, 8] -> 2*64 = 128 flops << 131072 threshold
        assert!(
            !pa.is_ane_profitable("GEMM", 128, &[8, 8]),
            "ANE should NOT be profitable for GEMM [8, 8]"
        );
    }

    #[test]
    fn ane_not_profitable_medium_gemm() {
        let pa = pa_m1();
        // GEMM [64, 64] -> 2*4096 = 8192 flops < 131072
        assert!(
            !pa.is_ane_profitable("GEMM", 16384, &[64, 64]),
            "ANE should NOT be profitable for GEMM [64, 64]"
        );
    }

    // ---- Target legality ----

    #[test]
    fn ane_not_legal_bitwise() {
        let pa = pa_m1();
        assert!(
            !pa.target_legality("AND", ComputeTarget::Ane),
            "ANE should NOT support AND (bitwise op)"
        );
        assert!(
            !pa.target_legality("ORR", ComputeTarget::Ane),
            "ANE should NOT support ORR (bitwise op)"
        );
        assert!(
            !pa.target_legality("EOR", ComputeTarget::Ane),
            "ANE should NOT support EOR (bitwise op)"
        );
        assert!(
            !pa.target_legality("SHL", ComputeTarget::Ane),
            "ANE should NOT support SHL (bitwise op)"
        );
    }

    #[test]
    fn ane_legal_matmul() {
        let pa = pa_m1();
        assert!(
            pa.target_legality("MATMUL", ComputeTarget::Ane),
            "ANE should support MATMUL"
        );
        assert!(
            pa.target_legality("GEMM", ComputeTarget::Ane),
            "ANE should support GEMM"
        );
        assert!(
            pa.target_legality("CONV2D", ComputeTarget::Ane),
            "ANE should support CONV2D"
        );
    }

    #[test]
    fn ane_legal_tensor_elementwise() {
        let pa = pa_m1();
        // ANE supports tensor-tensor elementwise (ADD, SUB, MUL, DIV)
        assert!(pa.target_legality("ADD", ComputeTarget::Ane));
        assert!(pa.target_legality("MUL", ComputeTarget::Ane));
        assert!(pa.target_legality("RELU", ComputeTarget::Ane));
        assert!(pa.target_legality("SIGMOID", ComputeTarget::Ane));
    }

    #[test]
    fn gpu_legal_all_arithmetic() {
        let pa = pa_m1();
        for op in &["ADD", "MUL", "FADD", "FMUL", "GEMM", "SDIV", "AND", "SHL"] {
            assert!(
                pa.target_legality(op, ComputeTarget::Gpu),
                "GPU should support {}", op
            );
        }
    }

    #[test]
    fn cpu_always_legal() {
        let pa = pa_m1();
        for op in &["ADD", "MUL", "SDIV", "GEMM", "AND", "UNKNOWN_OP", "FMLA"] {
            assert!(
                pa.target_legality(op, ComputeTarget::CpuScalar),
                "CPU Scalar should support {}", op
            );
        }
    }

    #[test]
    fn neon_not_legal_div() {
        let pa = pa_m1();
        assert!(
            !pa.target_legality("SDIV", ComputeTarget::Neon),
            "NEON should NOT support SDIV"
        );
        assert!(
            !pa.target_legality("UDIV", ComputeTarget::Neon),
            "NEON should NOT support UDIV"
        );
        assert!(
            !pa.target_legality("DIV", ComputeTarget::Neon),
            "NEON should NOT support DIV"
        );
    }

    #[test]
    fn neon_legal_add() {
        let pa = pa_m1();
        assert!(
            pa.target_legality("ADD", ComputeTarget::Neon),
            "NEON should support ADD"
        );
        assert!(
            pa.target_legality("MUL", ComputeTarget::Neon),
            "NEON should support MUL"
        );
        assert!(
            pa.target_legality("FADD", ComputeTarget::Neon),
            "NEON should support FADD"
        );
    }

    // ---- Dispatch recommendations ----

    #[test]
    fn dispatch_recommendation_ordering() {
        let pa = pa_m1();
        // Large GEMM: 512x512 = 262144 elements, data = 1MB+
        let recs = pa.recommend_dispatch("GEMM", 1048576, &[512, 512]);
        assert!(!recs.is_empty(), "Should have at least one recommendation");

        // Verify sorted by latency (cheapest first)
        for w in recs.windows(2) {
            assert!(
                w[0].1.latency_cycles <= w[1].1.latency_cycles,
                "Recommendations should be sorted by cost: {} <= {}",
                w[0].1.latency_cycles,
                w[1].1.latency_cycles,
            );
        }
    }

    #[test]
    fn dispatch_includes_cpu_fallback() {
        let pa = pa_m1();
        // Even for a small op, CPU should always be in the results
        let recs = pa.recommend_dispatch("ADD", 16, &[4]);
        assert!(
            recs.iter().any(|(t, _)| *t == ComputeTarget::CpuScalar),
            "CpuScalar should always be in dispatch recommendations"
        );
    }

    #[test]
    fn dispatch_small_data_cpu_only() {
        let pa = pa_m1();
        // Very small data: only CPU/NEON should appear (no GPU/ANE)
        let recs = pa.recommend_dispatch("ADD", 32, &[8]);
        let targets: Vec<ComputeTarget> = recs.iter().map(|(t, _)| *t).collect();
        assert!(targets.contains(&ComputeTarget::CpuScalar));
        assert!(
            !targets.contains(&ComputeTarget::Gpu),
            "GPU should not appear for 8-element ADD"
        );
        assert!(
            !targets.contains(&ComputeTarget::Ane),
            "ANE should not appear for 8-element ADD"
        );
    }

    #[test]
    fn dispatch_large_elementwise_includes_gpu() {
        let pa = pa_m1();
        // 10000 elements * 4 bytes = 40000 bytes, element_count = 10000 > 4096
        let recs = pa.recommend_dispatch("ADD", 40000, &[10000]);
        let targets: Vec<ComputeTarget> = recs.iter().map(|(t, _)| *t).collect();
        assert!(
            targets.contains(&ComputeTarget::Gpu),
            "GPU should appear for 10000-element ADD, got targets: {:?}",
            targets
        );
    }

    // ---- Operation classification ----

    #[test]
    fn classify_op_elementwise() {
        let pa = pa_m1();
        assert_eq!(pa.classify_op("ADD"), OperationCategory::Elementwise);
        assert_eq!(pa.classify_op("SUB"), OperationCategory::Elementwise);
        assert_eq!(pa.classify_op("MUL"), OperationCategory::Elementwise);
        assert_eq!(pa.classify_op("FADD"), OperationCategory::Elementwise);
        assert_eq!(pa.classify_op("add"), OperationCategory::Elementwise);
    }

    #[test]
    fn classify_op_gemm() {
        let pa = pa_m1();
        assert_eq!(pa.classify_op("GEMM"), OperationCategory::Gemm);
        assert_eq!(pa.classify_op("MATMUL"), OperationCategory::Gemm);
        assert_eq!(pa.classify_op("gemm"), OperationCategory::Gemm);
    }

    #[test]
    fn classify_op_convolution() {
        let pa = pa_m1();
        assert_eq!(pa.classify_op("CONV2D"), OperationCategory::Convolution);
        assert_eq!(pa.classify_op("DEPTHWISE_CONV2D"), OperationCategory::Convolution);
    }

    #[test]
    fn classify_op_reduction() {
        let pa = pa_m1();
        assert_eq!(pa.classify_op("REDUCE_SUM"), OperationCategory::Reduction);
        assert_eq!(pa.classify_op("REDUCE_MAX"), OperationCategory::Reduction);
    }

    #[test]
    fn classify_op_bitwise() {
        let pa = pa_m1();
        assert_eq!(pa.classify_op("AND"), OperationCategory::BitwiseLogic);
        assert_eq!(pa.classify_op("ORR"), OperationCategory::BitwiseLogic);
        assert_eq!(pa.classify_op("SHL"), OperationCategory::BitwiseLogic);
    }

    // ---- GpuThresholds / AneThresholds defaults ----

    #[test]
    fn gpu_thresholds_defaults() {
        let t = GpuThresholds::default();
        assert_eq!(t.elementwise_min_elements, 4096);
        assert_eq!(t.gemm_min_flops, 32768);
        assert_eq!(t.reduction_min_elements, 8192);
        assert_eq!(t.absolute_min_bytes, 4096);
        assert_eq!(t.dispatch_overhead_cycles, 10000);
        assert_eq!(t.transfer_overhead_cycles, 10000);
    }

    #[test]
    fn ane_thresholds_defaults() {
        let t = AneThresholds::default();
        assert_eq!(t.gemm_min_flops, 131072);
        assert_eq!(t.fused_conv_min_flops, 65536);
        assert_eq!(t.elementwise_min_elements, 65536);
        assert_eq!(t.reduction_min_elements, 32768);
        assert_eq!(t.absolute_min_bytes, 32768);
        assert_eq!(t.dispatch_overhead_cycles, 100000);
        assert_eq!(t.transfer_overhead_cycles, 100000);
        assert_eq!(t.min_batch_size, 4);
    }

    // ---- Precision enum ----

    #[test]
    fn precision_variants_distinct() {
        let precisions = [
            Precision::FP16, Precision::FP32, Precision::FP64, Precision::BF16,
            Precision::INT8, Precision::INT16, Precision::INT32, Precision::INT64,
        ];
        for (i, p) in precisions.iter().enumerate() {
            for (j, q) in precisions.iter().enumerate() {
                if i == j {
                    assert_eq!(p, q);
                } else {
                    assert_ne!(p, q);
                }
            }
        }
    }
}

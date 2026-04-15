# AArch64 Cost Model Calibration for Apple M-Series

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Research Complete
**Part of:** LLVM2 optimization pipeline (`llvm2-opt`)

---

## Implementation Status (as of 2026-04-15)

**Overall: Hand-written cost model tables are implemented. No hardware-measured calibration has been performed.**

| Component | Status | Details |
|-----------|--------|---------|
| **Cost model** (`cost_model.rs`) | IMPLEMENTED | 2.7K LOC. Multi-target (CPU/NEON/GPU/ANE) latency and throughput tables. Hand-written from published microarchitecture data. |
| **Profitability analyzer** | IMPLEMENTED | Part of cost model. Evaluates whether moving computation to a different target is profitable. |
| **Hardware calibration** | NOT DONE | No measurements on actual Apple M-series hardware. Tables are from published research (Dougall Johnson et al.), not direct measurement. |
| **Learned cost model** | NOT IMPLEMENTED | No ML-based cost model (Ithemal-style). |

---

## Purpose

LLVM2's optimization passes (peephole, CSE, LICM, instruction selection) need a cost
model to make correct decisions. This document collects microarchitecture data for
Apple M-series (primary target) and ARM Neoverse (server reference) to calibrate
LLVM2's cost estimates.

---

## Data Sources

1. **Dougall Johnson, "Apple M1 Firestorm Microarchitecture Exploration"**
   https://dougallj.github.io/applecpu/firestorm.html
   Reverse-engineered via performance counters. Covers M1 Firestorm (P-core).
   This is the most detailed public Apple Silicon microarchitecture reference.

2. **LLVM AArch64 Scheduling Models**
   `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64SchedCyclone.td` (Apple A8)
   `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64SchedNeoverseV2.td` (ARM V2)
   `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64SchedA64FX.td` (Fujitsu A64FX)
   Apple does not contribute post-Cyclone scheduling models to upstream LLVM.

3. **ARM Neoverse V2 Software Optimization Guide**
   ARM DDI 0602. Public document from ARM.

4. **No published M4 latency data** as of 2026-04-14. Apple does not publish
   microarchitecture details. M4 is expected to be similar to M3 (Ibiza) with
   incremental improvements. Use M1 Firestorm data as baseline; differences
   are expected to be within ~10% for most instruction classes.

---

## Apple M1 Firestorm (P-core) Microarchitecture

### Pipeline Overview

| Parameter | Value | Source |
|-----------|-------|--------|
| Issue width | 8 uops/cycle | Dougall Johnson |
| Reorder buffer | ~600 entries (estimated) | Dougall Johnson |
| Integer units | 6 (u1-u6) | Dougall Johnson |
| Load/store units | 4 (u7-u10, up to 128-bit) | Dougall Johnson |
| FP/SIMD units | 4 (u11-u14) | Dougall Johnson |
| In-flight loads | ~130 | Dougall Johnson |
| In-flight stores | ~60 | Dougall Johnson |
| Branch prediction | Taken branches: 1/cycle | Dougall Johnson |

### Integer Instruction Latencies

| Instruction | Latency | Throughput | Units | Notes |
|-------------|---------|------------|-------|-------|
| ADD/SUB (reg) | 1c | 0.167c/i | u1-6 | 6-way issue |
| ADD/SUB (shifted) | 2c | 0.333c/i | 2x u1-6 | Fused shift+add |
| ADD/SUB (extended) | 2c | 0.333c/i | 2x u1-6 | Fused extend+add |
| AND/ORR/EOR (reg) | 1c | 0.167c/i | u1-6 | 6-way issue |
| AND/ORR/EOR (shifted) | 2c | 0.333c/i | 2x u1-6 | Fused |
| LSL/LSR/ASR (imm/reg) | 1c | 0.167c/i | u1-6 | |
| CLS/CLZ | 1c | 0.167c/i | u1-6 | |
| CMP (reg) | 1c | 0.333c/i | u1-3 | Only 3 units |
| CCMP | 1c | 0.333c/i | u1-3 | |
| CSEL/CSET | 1c | 0.333c/i | u1-3 | |
| ADC/ADCS | 1c | 0.333c/i | u1-3 | |
| MUL | 3c | 0.5c/i | u5/u6 | 2 multiply pipes |
| MADD/MSUB | [1;3]c | 1c/i | u6 | 1c to addend, 3c to other |
| EXTR | [1;2]c | 1c/i | u6 + u1-6 | |
| BFI/BFC/BFXIL | 1c | 1c/i | u6 | |

**Key cost model insight**: Basic ALU (ADD, AND, ORR, shift) are effectively free
-- 6-way issue means they never bottleneck. Multiplies are expensive (3c latency,
limited to 2 pipes). Conditional operations (CMP, CSEL, CCMP) are limited to 3 pipes.

### MADD Special Latency

MADD has a dependency-dependent latency pattern:
- Result -> addend input of another MADD: **1 cycle** (forwarding path)
- Result -> any other instruction: **3 cycles**

This means multiply-accumulate chains `MADD x, a, b, x` have effective 1c latency
per accumulation step. LLVM2's cost model should distinguish these cases.

### Load/Store Latencies

| Instruction | Latency | Throughput | Units | Notes |
|-------------|---------|------------|-------|-------|
| LDR (base+imm) | <=4c | 0.333c/i | u8-10 | L1 hit |
| LDP | <=4c | 0.333c/i | u8-10 | Load pair |
| STR | N/A (no result) | 0.333c/i | u8-10 | |
| STP | N/A | 0.333c/i | u8-10 | |
| Load -> ALU chain | 4c | - | - | Standard load-to-use |

**L1 cache**: 128 KB, 8-way, likely 3-4 cycle latency.
**L2 cache**: 12 MB (M1 shared), ~12-15 cycle latency (estimated from workloads).

### FP/SIMD Instruction Latencies

| Instruction | Latency | Throughput | Units | Notes |
|-------------|---------|------------|-------|-------|
| FADD (scalar/vector) | 3c | 0.25c/i | u11-14 | 4-way issue |
| FMUL (scalar/vector) | 3c | 0.25c/i | u11-14 | |
| FMADD/FMLA | 4c | 0.25c/i | u11-14 | |
| FABS/FNEG | 2c | 0.25c/i | u11-14 | |
| FADDP (pairwise) | 3c | 0.25c/i | u11-14 | |
| FABD (abs diff) | 3c | 0.25c/i | u11-14 | |
| FCMP/FCMEQ/FCMGE/FCMGT | 2c | 0.25c/i | u11-14 | |
| FCVT (format) | 3c | 0.25c/i | u11-14 | |
| FCVT (to GPR) | <=13c | 0.5c/i | 2 uops | Cross-domain penalty |
| DUP (element) | 2c | 0.25c/i | u11-14 | |
| DUP (from GPR) | <=12c | 0.333c/i | 2 uops | Cross-domain |
| EXT | 2c | 0.25c/i | u11-14 | |
| AND/EOR (vector) | 2c | 0.25c/i | u11-14 | |
| BSL/BIT/BIF | 2c | 0.25c/i | u11-14 | |
| ADD (vector integer) | 2c | 0.25c/i | u11-14 | |
| ABS (vector) | 3c | 0.25c/i | u11-14 | |
| ADDP (vector) | 2c | 0.25c/i | u11-14 | |

**Key cost model insight**: FP/SIMD throughput is excellent (4-way issue). The main
bottleneck is latency (3-4 cycles for most ops). Cross-domain transfers (GPR <-> SIMD)
are very expensive (12-13 cycles). LLVM2 should heavily penalize GPR<->SIMD moves.

### Branch Latencies

| Instruction | Throughput | Notes |
|-------------|------------|-------|
| B/BL (direct) | 1c/i | Taken branch limit |
| B.cc (not taken) | 0.5c/i | |
| CBZ/CBNZ | 0.5-1c/i | |
| BR/BLR/RET | 1c/i | Indirect |

**Misprediction penalty**: ~14-19 cycles (from LLVM Cyclone model, likely similar
for M1).

---

## LLVM Cyclone Model (Apple A8 -- Closest Upstream Reference)

Source: `AArch64SchedCyclone.td`

| Parameter | Value | Notes |
|-----------|-------|--------|
| Issue width | 6 | vs M1's 8 |
| Reorder buffer | 192 | vs M1's ~600 |
| Load latency | 4c | Same as M1 |
| Misprediction penalty | 16c | Similar to M1's 14-19c |
| Integer pipes | 4 | vs M1's 6 |
| Load/store pipes | 2 | vs M1's 4 |
| FP/vector pipes | 3 | vs M1's 4 |
| Multiply pipe | 1 | vs M1's 2 |

**Scaling from Cyclone to M1**: M1 has roughly 1.5-2x the execution resources of
Cyclone. Issue width increased from 6 to 8. Every resource class grew. This means
LLVM2 should model more aggressive out-of-order execution and less contention than
LLVM's Cyclone model suggests.

---

## ARM Neoverse V2 (Server Reference)

Source: `AArch64SchedNeoverseV2.td`

| Parameter | Value | Notes |
|-----------|-------|--------|
| Issue width | 6 | |
| Reorder buffer | 320 uops | |
| Load latency | 4c | |
| Misprediction penalty | 10c | Lower than Apple |
| Integer pipes | 4 single-cycle + 2 multi-cycle | |
| FP/SIMD pipes | 4 (V0-V3) | Same count as M1 |
| Load/store units | 2 load + 1 dedicated load + 2 store data | |
| ALU latency | 1c | Same |
| Multiply latency | 2c | Lower than M1's 3c |
| FP add latency | 2c | Lower than M1's 3c |
| FP multiply latency | 3c | Same as M1 |
| DIV (32-bit) | 12c | |
| DIV (64-bit) | 20c | |
| FP divide (double) | 15c | |

---

## Fujitsu A64FX (HPC Reference)

Source: `AArch64SchedA64FX.td`

| Parameter | Value | Notes |
|-----------|-------|--------|
| Issue width | 6 | |
| Reorder buffer | 180 | Smaller |
| Load latency | 5c | Higher |
| Misprediction penalty | 12c | |
| MUL/MADD latency | 5c | Higher than M1 |
| DIV (32-bit) | 39c | Very high |
| DIV (64-bit) | 23c | |
| FP divide (double) | 43c | Very high |

A64FX is optimized for SVE vector throughput, not scalar latency. Not a good model
for Apple Silicon.

---

## Cost Model Recommendations for LLVM2

### Proposed Cost Table

Based on M1 Firestorm data, with conservative rounding for forward-compatibility:

```rust
/// Instruction cost in abstract "cost units" (roughly cycles, but accounting
/// for throughput and resource contention).
pub struct CostModel {
    // Integer ALU
    pub alu_simple: u32,       // ADD, SUB, AND, ORR, EOR, shift: 1
    pub alu_shifted: u32,      // ADD w/ shift, AND w/ shift: 2
    pub alu_extended: u32,     // ADD w/ extend: 2
    pub multiply: u32,         // MUL: 3
    pub madd_chain: u32,       // MADD (accumulate chain): 1
    pub madd_other: u32,       // MADD (non-chain): 3
    pub divide_32: u32,        // SDIV/UDIV W: 10-12
    pub divide_64: u32,        // SDIV/UDIV X: 12-20
    pub compare: u32,          // CMP, CCMP: 1
    pub csel: u32,             // CSEL, CSET: 1
    pub bitfield: u32,         // BFI, UBFX, EXTR: 1-2
    pub clz_rbit: u32,         // CLZ, RBIT, REV: 1

    // Memory
    pub load: u32,             // LDR: 4 (L1 hit)
    pub load_pair: u32,        // LDP: 4
    pub store: u32,            // STR: 1 (no result latency)
    pub store_pair: u32,       // STP: 1

    // FP/SIMD scalar
    pub fp_add: u32,           // FADD: 3
    pub fp_mul: u32,           // FMUL: 3
    pub fp_fma: u32,           // FMADD/FMSUB: 4
    pub fp_div_f32: u32,       // FDIV.S: ~10
    pub fp_div_f64: u32,       // FDIV.D: ~15
    pub fp_abs_neg: u32,       // FABS, FNEG: 2
    pub fp_cmp: u32,           // FCMP, FCCMP: 2
    pub fp_cvt: u32,           // FCVT (FP<->FP): 3

    // NEON vector (same latency as scalar FP, just wider)
    pub neon_int_alu: u32,     // ADD.4S, SUB.4S, AND.16B: 2
    pub neon_int_mul: u32,     // MUL.4S: estimated 4
    pub neon_fp_add: u32,      // FADD.2D, FADD.4S: 3
    pub neon_fp_mul: u32,      // FMUL.2D, FMUL.4S: 3
    pub neon_fp_fma: u32,      // FMLA.2D, FMLA.4S: 4
    pub neon_shuffle: u32,     // TBL, EXT, ZIP, TRN: 2
    pub neon_horizontal: u32,  // ADDP, FADDP: 2-3

    // Cross-domain
    pub gpr_to_simd: u32,      // DUP from GPR, FMOV GPR->FP: 12
    pub simd_to_gpr: u32,      // FMOV FP->GPR, UMOV: 12-13

    // Control flow
    pub branch_taken: u32,     // B, BL: 1
    pub branch_not_taken: u32, // B.cc (not taken): 0 (free)
    pub branch_indirect: u32,  // BR, BLR: 1
    pub misprediction: u32,    // Branch misprediction: 16
}
```

### Key Optimization Rules

1. **Prefer shifted/extended addressing to separate shift+add**: ADD w/ shift (2c) is
   better than LSL + ADD (1c + 1c = 2c, but uses 2 issue slots vs 1).

2. **Heavily penalize GPR<->SIMD transfers**: 12-13 cycles. Never move a value to SIMD
   just for one operation then back. Threshold: at least 3-4 SIMD operations to amortize.

3. **MADD chains are free (1c)**: Always prefer MADD over MUL+ADD for accumulation.
   The forwarding path makes MADD x, a, b, x effectively 1c latency in a chain.

4. **FP/SIMD has 4-way issue**: No throughput bottleneck for straight-line FP code.
   The bottleneck is always latency (3-4 cycles). Optimize for latency, not throughput.

5. **Division is extremely expensive**: 10-43 cycles depending on width and domain.
   Always replace with multiply-by-reciprocal when possible.

6. **Loads are 4c but highly parallel**: M1 can sustain ~3 loads/cycle with 130
   in-flight. Loads are cheap if not on the critical path. Cost model should count
   loads on the critical path at 4, off-critical at 1.

7. **Conditional branches are free when predicted**: Not-taken branches cost 0.
   CSEL is 1c. Prefer branchless (CSEL) for unpredictable branches, branch-based
   for predictable ones. Misprediction (16c) >> CSEL (1c).

---

## M-Series Evolution (M1 -> M4)

No published microarchitecture data exists for M2/M3/M4. Based on public performance
data and die analysis:

| Parameter | M1 (A14) | M2 (A15) | M3 (A17) | M4 (estimated) |
|-----------|----------|----------|----------|----------------|
| Issue width | 8 | 8 | 8-9? | 8-9? |
| Integer pipes | 6 | 6 | 6 | 6 |
| FP/SIMD pipes | 4 | 4 | 4 | 4 |
| Load/store pipes | 4 | 4 | 4 | 4 |
| L1 cache | 128 KB | 128 KB | 128 KB | 128 KB |
| L2 cache | 12 MB | 16 MB | 16 MB | 16 MB |

**Conclusion**: The fundamental microarchitecture has been remarkably stable from M1
through M4. Apple has focused on frequency scaling, cache size, and GPU improvements
rather than radical pipeline changes. M1 Firestorm latency data is a good
approximation for all M-series chips within ~10%.

---

## Recommended Integration Path

1. **Define `CostModel` struct** in `llvm2-opt` with M1-calibrated defaults
2. **Parameterize by target**: Allow override for Neoverse, A64FX, generic
3. **Use in peephole pass**: Cost comparison for rewrite candidates
4. **Use in instruction selection**: Choose between equivalent lowering patterns
5. **Use in LICM**: Estimate loop body cost for hoisting decisions
6. **Future**: Auto-calibrate from performance counters on target hardware

---

## References

1. Dougall Johnson. "Apple M1 Firestorm Microarchitecture Exploration."
   https://dougallj.github.io/applecpu/firestorm.html
2. Dougall Johnson. "Apple M1 Firestorm SIMD/FP Instructions."
   https://dougallj.github.io/applecpu/firestorm-simd.html
3. Dougall Johnson. "Apple M1 Firestorm Integer Instructions."
   https://dougallj.github.io/applecpu/firestorm-int.html
4. LLVM Cyclone Scheduling Model.
   `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64SchedCyclone.td`
5. LLVM Neoverse V2 Scheduling Model.
   `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64SchedNeoverseV2.td`
6. LLVM A64FX Scheduling Model.
   `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64SchedA64FX.td`
7. ARM. "Arm Neoverse V2 Software Optimization Guide." DDI 0602.
8. ARM. "ARM Architecture Reference Manual." DDI 0487.

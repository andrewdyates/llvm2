// llvm2-codegen/riscv/mod.rs - RISC-V target encoding modules
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! RISC-V RV64GC instruction encoding.
//!
//! This module provides the RISC-V backend for LLVM2. Fixed 32-bit
//! instruction encoding for all six standard formats (R, I, S, B, U, J).
//!
//! Covers:
//! - RV64I base integer instructions
//! - M extension (integer multiply/divide)
//! - D extension (double-precision floating-point)
//!
//! # Architecture
//!
//! RISC-V uses fixed 32-bit instruction encoding with standard field positions:
//! - `opcode[6:0]` — operation category
//! - `rd[11:7]` — destination register
//! - `funct3[14:12]` — operation variant
//! - `rs1[19:15]` — first source register
//! - `rs2[24:20]` — second source register
//! - `funct7[31:25]` — operation sub-variant
//!
//! Reference: RISC-V Unprivileged ISA Specification (Volume 1, Version 20191213)

pub mod encode;
pub mod pipeline;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use encode::{
    RiscVEncodeError, RiscVInstOperands,
    encode_instruction,
    encode_r_type, encode_i_type, encode_s_type, encode_b_type,
    encode_u_type, encode_j_type,
};
pub use pipeline::{
    RiscVPipeline, RiscVPipelineConfig, RiscVPipelineError,
    RiscVRegAssignment,
    RiscVISelFunction, RiscVISelInst, RiscVISelOperand, RiscVISelBlock,
    build_riscv_add_test_function, build_riscv_const_test_function,
    riscv_compile_to_bytes, riscv_compile_to_elf,
};

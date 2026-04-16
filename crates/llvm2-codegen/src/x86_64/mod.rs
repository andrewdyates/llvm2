// llvm2-codegen/x86_64/mod.rs - x86-64 target encoding modules
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! x86-64 (AMD64) instruction encoding and compilation pipeline.
//!
//! This module provides the x86-64 backend for LLVM2. It includes:
//!
//! - **Binary encoding** (`encode`): Full instruction encoder supporting 50+
//!   instruction types including ALU (ADD, SUB, AND, OR, XOR), MOV variants
//!   (reg-reg, reg-imm64, reg-mem, mem-reg, SIB addressing), control flow
//!   (RET, CALL, JMP, Jcc), stack (PUSH, POP), shifts, unary ops, IMUL,
//!   DIV/IDIV, MOVZX/MOVSX, CMOVcc, SETcc, bit manipulation (BSF, BSR,
//!   TZCNT, LZCNT, POPCNT), SSE scalar (ADDSD/SS, SUBSD/SS, MULSD/SS,
//!   DIVSD/SS, MOVSD/SS, UCOMISD/SS), SSE conversions, LEA, and
//!   RIP-relative addressing.
//!
//! - **Compilation pipeline** (`pipeline`): End-to-end pipeline from ISel
//!   output to machine code bytes or object files. Includes System V AMD64
//!   ABI frame lowering (prologue/epilogue), simplified linear-scan register
//!   assignment, branch resolution for variable-length instructions, and
//!   output to raw bytes, ELF, or Mach-O object files.
//!
//! # Architecture
//!
//! x86-64 uses variable-length instruction encoding (1-15 bytes) with:
//! - Legacy prefixes (66h, F2h, F3h, etc.)
//! - REX prefix (40h-4Fh) for 64-bit operands and extended registers
//! - Opcode (1-3 bytes)
//! - ModR/M byte (addressing mode)
//! - SIB byte (scaled index addressing)
//! - Displacement (0, 1, 2, or 4 bytes)
//! - Immediate (0, 1, 2, 4, or 8 bytes)
//!
//! Reference: Intel 64 and IA-32 Architectures SDM, Volume 2
//! Reference: ~/llvm-project-ref/llvm/lib/Target/X86/MCTargetDesc/X86MCCodeEmitter.cpp

pub mod encode;
pub mod pipeline;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use encode::{ModRM, RexPrefix, Sib, X86EncodeError, X86Encoder, X86InstOperands};
pub use pipeline::{
    X86OutputFormat, X86Pipeline, X86PipelineConfig, X86PipelineError,
    X86RegAssignment,
    build_x86_add_test_function, build_x86_const_test_function,
    x86_compile_to_bytes, x86_compile_to_elf, x86_compile_to_macho,
};

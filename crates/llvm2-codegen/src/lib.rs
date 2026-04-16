// llvm2-codegen - Verified machine code generation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified machine code generation for LLVM2.
//!
//! This crate generates machine code from LIR with formal verification
//! of correctness. Each lowering step is verified using llvm2-verify.
//!
//! Target: AArch64 macOS (Apple Silicon) for MVP.
//! x86-64 and RISC-V are future targets.

pub mod aarch64;
pub mod compiler;
pub mod coreml_emitter;
pub mod dwarf_cfi;
pub mod dwarf_info;
pub mod elf;
pub mod error;
pub mod exception_handling;
pub mod frame;
pub mod layout;
pub mod lower;
pub mod macho;
pub mod metal_emitter;
pub mod pipeline;
pub mod relax;
pub mod riscv;
pub mod target;
pub mod unwind;
pub mod x86_64;

pub use compiler::{
    CompilationMetrics, CompilationResult, CompileError,
    Compiler, CompilerConfig, CompilerTrace, CompilerTraceLevel, ProofCertificate,
};
pub use error::CodegenError;
pub use metal_emitter::{MetalOutput, MetalEmitError, NamedKernel, emit_metal_kernels};
pub use pipeline::{
    Pipeline, PipelineConfig, PipelineError, DispatchVerifyMode,
    InputFormat, detect_input_format, load_module, load_module_from_bytes,
    compile_to_object, generate_cpu_only_plan, generate_lsda_for_function,
    CoreMLOutput, emit_coreml_program,
};
pub use lower::LowerError;
pub use relax::{BranchRelaxation, RelaxError, RelaxedCode};
pub use target::{CallingConvention, Target};

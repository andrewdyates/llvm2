// llvm2-codegen - Verified machine code generation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified machine code generation for LLVM2.
//!
//! This crate generates machine code from LIR with formal verification
//! of correctness. Each lowering step is verified using llvm2-verify.
//!
//! Target: AArch64 macOS (Apple Silicon) for MVP.
//! x86-64 and RISC-V are future targets.
//!
//! # Calling convention for JIT-compiled symbols
//!
//! Functions JIT-compiled by [`jit::JitCompiler`] follow the host
//! platform's C calling convention (AAPCS64 / Apple DarwinPCS on
//! aarch64, System V AMD64 on x86-64). External callers such as `z4`
//! and `tla2` may `std::mem::transmute` a symbol pointer returned from
//! [`jit::ExecutableBuffer`] into an `extern "C" fn(...)` matching the
//! tMIR signature and invoke it directly. This is a stable, P0
//! contract — any deviation must be opt-in and documented. See the
//! full contract (register assignments, callee-saved sets, sret
//! handling, known gaps) at the top of [`jit`].

pub mod aarch64;
pub mod compiler;
pub mod constant_pool;
pub mod coreml_emitter;
pub mod dialect_pipeline;
pub mod dwarf_cfi;
pub mod dwarf_info;
pub mod elf;
pub mod error;
pub mod exception_handling;
pub mod frame;
pub mod interpreter;
pub mod jit;
pub mod jit_cert;
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
    CompilationMetrics, CompilationResult, CompileError, Compiler, CompilerConfig, CompilerTrace,
    CompilerTraceLevel, FunctionQualityMetrics, JitCompilationResult, ProofCertificate,
};
pub use error::CodegenError;
pub use interpreter::{
    InterpreterConfig, InterpreterError, InterpreterValue, interpret, interpret_with_config,
};
pub use jit::{ExecutableBuffer, JitCompiler, JitConfig, JitError, ProfileHookMode, ProfileStats};
pub use jit_cert::{JitCertificate, TmirPair};
pub use lower::LowerError;
pub use metal_emitter::{MetalEmitError, MetalOutput, NamedKernel, emit_metal_kernels};
pub use pipeline::{
    CoreMLOutput, DispatchVerifyMode, FormatMode, InputFormat, PhaseTimings, Pipeline,
    PipelineConfig, PipelineError, PreparationMetrics, compile_to_object, detect_input_format,
    emit_coreml_program, generate_cpu_only_plan, generate_lsda_for_function, load_module,
    load_module_as, load_module_from_bytes,
};
pub use relax::{BranchRelaxation, RelaxError, RelaxedCode};
pub use target::{CallingConvention, Target};

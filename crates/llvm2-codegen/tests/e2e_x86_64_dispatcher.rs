// llvm2-codegen/tests/e2e_x86_64_dispatcher.rs - #340 x86-64 dispatcher wiring
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Verifies that `Compiler::compile` honors `config.target` and routes to
// the x86-64 backend (X86Pipeline) when `Target::X86_64` is selected.
//
// Part of #340 — x86-64 cross-platform support for tla2.
//
// Prior to #340-A the dispatcher was hard-wired to the AArch64 path and
// the x86-64 pipeline was unreachable from the public Compiler API. These
// tests pin that behavior: for the same tMIR input, AArch64 and x86-64
// dispatch must produce distinct object bytes and distinct Mach-O CPU type
// headers.

use llvm2_codegen::compiler::{CompileError, Compiler, CompilerConfig};
use llvm2_codegen::pipeline::OptLevel;
use llvm2_codegen::target::Target;

use tmir::{
    BinOp, BlockId, FuncId, FuncTy, FuncTyId, Inst, InstrNode, Module as TmirModule,
    Block as TmirBlock, Function as TmirFunction, Ty, ValueId,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a single-function `fn add(a: i64, b: i64) -> i64 { a + b }` module.
///
/// Shape mirrors `build_simple_add` in e2e_pipeline_integration.rs so the
/// tMIR input is identical to the AArch64 golden path; only the dispatch
/// target changes between test variants.
fn build_add_module() -> TmirModule {
    let mut module = TmirModule::new("test_x86_dispatch");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "add", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

/// Mach-O 64-bit magic (little-endian): 0xFEEDFACF.
const MH_MAGIC_64: u32 = 0xFEEDFACF;
/// Mach-O CPU_TYPE_ARM64.
const CPU_TYPE_ARM64: u32 = 0x0100_000C;
/// Mach-O CPU_TYPE_X86_64.
const CPU_TYPE_X86_64: u32 = 0x0100_0007;

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

// ---------------------------------------------------------------------------
// Dispatcher tests
// ---------------------------------------------------------------------------

#[test]
fn dispatcher_x86_64_produces_x86_64_macho() {
    // Sanity: compiling the same module with Target::X86_64 must produce a
    // Mach-O object whose CPU type is X86_64, not ARM64.
    let module = build_add_module();

    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::X86_64,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("x86_64 dispatcher should compile single-function add module");

    // Output must be a valid Mach-O 64-bit object.
    assert!(
        result.object_code.len() >= 32,
        "x86-64 Mach-O must have a full mach_header_64 (got {} bytes)",
        result.object_code.len()
    );
    let magic = read_u32_le(&result.object_code, 0);
    assert_eq!(
        magic, MH_MAGIC_64,
        "x86-64 object should have Mach-O 64-bit magic; got 0x{:08X}",
        magic
    );

    // CPU type is the next 4-byte field in mach_header_64.
    let cpu_type = read_u32_le(&result.object_code, 4);
    assert_eq!(
        cpu_type, CPU_TYPE_X86_64,
        "x86-64 dispatch must emit CPU_TYPE_X86_64 (0x{:08X}); got 0x{:08X}",
        CPU_TYPE_X86_64, cpu_type
    );

    // Metrics sanity: single function, non-empty code.
    assert_eq!(result.metrics.function_count, 1);
    assert!(
        result.metrics.code_size_bytes > 0,
        "x86-64 code size must be non-zero"
    );
    // code_size_bytes for x86 is the raw encoded code length, NOT
    // instruction_count * 4 (variable-length encoding).
    assert_ne!(
        result.metrics.code_size_bytes,
        result.metrics.instruction_count * 4,
        "x86-64 must not assume 4-byte fixed-width encoding; \
         code_size_bytes should come from the encoder"
    );
}

#[test]
fn dispatcher_aarch64_still_produces_arm64_macho() {
    // Regression guard: the x86 dispatch must not break the AArch64 default.
    let module = build_add_module();

    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::Aarch64,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("default AArch64 dispatcher should keep working");

    let magic = read_u32_le(&result.object_code, 0);
    assert_eq!(magic, MH_MAGIC_64);
    let cpu_type = read_u32_le(&result.object_code, 4);
    assert_eq!(
        cpu_type, CPU_TYPE_ARM64,
        "AArch64 dispatch must still emit CPU_TYPE_ARM64"
    );
}

#[test]
fn dispatcher_x86_64_differs_from_aarch64() {
    // The core behavioral pin: same tMIR input + different targets must
    // produce different object bytes. Before #340-A this test would FAIL
    // because both paths ran the AArch64 pipeline regardless of target.
    let module = build_add_module();

    let x86_result = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::X86_64,
        ..CompilerConfig::default()
    })
    .compile(&module)
    .expect("x86_64 dispatch");

    let aarch64_result = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::Aarch64,
        ..CompilerConfig::default()
    })
    .compile(&module)
    .expect("aarch64 dispatch");

    assert_ne!(
        x86_result.object_code, aarch64_result.object_code,
        "x86_64 and aarch64 object code must differ for the same tMIR input"
    );
    // At minimum, the CPU type in the Mach-O header must differ.
    let x86_cpu = read_u32_le(&x86_result.object_code, 4);
    let arm_cpu = read_u32_le(&aarch64_result.object_code, 4);
    assert_ne!(x86_cpu, arm_cpu, "CPU types must differ across targets");
}

#[test]
fn dispatcher_x86_64_multi_function_module_compiles() {
    // #464: multi-function x86-64 modules must compile end-to-end through
    // `X86Pipeline::compile_module`. Prior to #464 the dispatcher rejected
    // any module with more than one function. The fix mirrors the AArch64
    // multi-function path (see `Pipeline::compile_module`): per-function
    // ISel runs independently, code + inline const pools are concatenated
    // into a single __text section, and the resulting Mach-O object carries
    // one global symbol per function.
    let mut module = build_add_module();

    // Add a second function (sub) reusing the same FuncTyId(0) since it
    // has identical signature (i64, i64) -> i64.
    let mut sub_func = TmirFunction::new(
        FuncId::new(1),
        "sub",
        FuncTyId::new(0),
        BlockId::new(0),
    );
    sub_func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(sub_func);

    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::X86_64,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("x86-64 dispatcher must compile multi-function modules (#464)");

    // Mach-O 64-bit header sanity.
    assert!(
        result.object_code.len() >= 32,
        "multi-function x86-64 Mach-O must have a full mach_header_64 (got {} bytes)",
        result.object_code.len()
    );
    let magic = read_u32_le(&result.object_code, 0);
    assert_eq!(
        magic, MH_MAGIC_64,
        "object should have Mach-O 64-bit magic; got 0x{:08X}",
        magic
    );
    let cpu_type = read_u32_le(&result.object_code, 4);
    assert_eq!(
        cpu_type, CPU_TYPE_X86_64,
        "multi-function module must still emit CPU_TYPE_X86_64 (0x{:08X}); got 0x{:08X}",
        CPU_TYPE_X86_64, cpu_type
    );

    // Metrics must reflect both functions, not just the first.
    assert_eq!(
        result.metrics.function_count, 2,
        "multi-function module must report function_count = 2"
    );
    assert!(
        result.metrics.code_size_bytes > 0,
        "multi-function module must have non-zero code size"
    );
    assert!(
        result.metrics.instruction_count > 0,
        "multi-function module must sum instruction counts across all functions"
    );

    // Both mangled symbol names must appear in the object's symbol table.
    // Mach-O writes symbols uncompressed into the string table, so a raw
    // byte-substring search is sufficient here (this mirrors how the
    // AArch64 multi-function tests verify symbol-table contents without
    // depending on a full Mach-O parser).
    let obj = &result.object_code;
    let has_add = obj.windows(4).any(|w| w == b"_add");
    let has_sub = obj.windows(4).any(|w| w == b"_sub");
    assert!(
        has_add && has_sub,
        "multi-function Mach-O must contain both `_add` and `_sub` symbol names \
         (has_add={}, has_sub={})",
        has_add, has_sub
    );
}

#[test]
fn dispatcher_x86_64_emit_proofs_returns_populated_certs() {
    // #465: the x86-64 dispatcher now wires proof certificates through the
    // public Compiler API (mirror of the AArch64 path). A trivial `add`
    // function must produce `Some(certs)` with `certs.len() > 0`.
    //
    // Historical: prior to #465 this test pinned the inverse behavior —
    // `CompileError::ProofsUnsupportedForTarget` — because the x86-64
    // MachFunction verifier did not yet exist. #465 wires
    // `llvm2_verify::x86_64_function_verifier` through
    // `compile_x86_64`, replacing the typed-error early-return with real
    // proof generation. The verified-codegen invariant
    // (`result.proofs.is_some()` when `emit_proofs=true`) is now preserved
    // by successful generation, not by erroring out.
    //
    // Runs on a thread with enlarged stack because the verifier's
    // recursive SMT evaluation can overflow the default 8 MiB test
    // thread stack in debug builds (same pattern as
    // `test_compile_module_to_jit_with_proofs` for AArch64).
    let child = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let module = build_add_module();

            let compiler = Compiler::new(CompilerConfig {
                opt_level: OptLevel::O0,
                target: Target::X86_64,
                emit_proofs: true,
                ..CompilerConfig::default()
            });

            let result = compiler
                .compile(&module)
                .expect("x86-64 + emit_proofs=true must now succeed (#465)");

            let certs = result
                .proofs
                .as_ref()
                .expect("proofs must be Some when emit_proofs=true on x86-64 (#465)");
            assert!(
                !certs.is_empty(),
                "x86-64 add module must produce >=1 proof certificate; got 0"
            );

            // The trivial `add` lowers through an x86-64 ADD instruction;
            // at least one cert must come from the x86-64 lowering proof
            // registry.
            let has_x86_cert = certs.iter().any(|c| c.rule_name.starts_with("x86_64:"));
            assert!(
                has_x86_cert,
                "expected at least one cert from the x86-64 lowering proof registry; \
                 got rule names: {:?}",
                certs.iter().map(|c| c.rule_name.as_str()).collect::<Vec<_>>()
            );
        })
        .expect("failed to spawn thread with larger stack");
    child.join().expect("test thread panicked");
}

#[test]
fn dispatcher_x86_64_emit_proofs_errors_variant_deprecated_for_x86() {
    // Documentation test pinning the #465 behavior change: the
    // `CompileError::ProofsUnsupportedForTarget` variant is still defined
    // (it fires for `Target::Riscv64`), but under the default `verify`
    // feature the x86-64 dispatcher must never return it. This guards
    // against a silent revert of #465 that leaves the variant reachable
    // for x86-64 without anyone noticing.
    //
    // Same stack-size dance as the populated-certs test above — the
    // verifier is recursive and debug builds overflow the default 8 MiB.
    let child = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let module = build_add_module();

            let compiler = Compiler::new(CompilerConfig {
                opt_level: OptLevel::O0,
                target: Target::X86_64,
                emit_proofs: true,
                ..CompilerConfig::default()
            });

            match compiler.compile(&module) {
                Ok(_) => { /* expected */ }
                Err(CompileError::ProofsUnsupportedForTarget { target }) => panic!(
                    "#465 regression: x86-64 + emit_proofs=true should no longer \
                     return ProofsUnsupportedForTarget (got target={:?})",
                    target
                ),
                Err(other) => panic!("unexpected compile error: {other:?}"),
            }
        })
        .expect("failed to spawn thread with larger stack");
    child.join().expect("test thread panicked");
}

#[test]
fn dispatcher_x86_64_emit_proofs_false_still_compiles_cleanly() {
    // Negative control for the regression pin above: when the caller
    // explicitly leaves `emit_proofs=false` (the default), x86-64 dispatch
    // must still succeed. The error in
    // `dispatcher_x86_64_emit_proofs_errors_instead_of_silently_dropping`
    // is conditional on `emit_proofs=true`; the rest of the x86-64 path
    // must not regress.
    let module = build_add_module();

    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::X86_64,
        emit_proofs: false,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("x86-64 dispatch with emit_proofs=false must still succeed");

    assert!(result.proofs.is_none(), "proofs must be None when emit_proofs=false");
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);
}

#[test]
fn dispatcher_x86_64_cross_function_call_compiles() {
    // #464: a tMIR module whose caller invokes a callee in the same module
    // must compile cleanly through the x86-64 dispatcher. The call site
    // becomes an `E8 dd dd dd dd` CALL rel32 whose 4-byte displacement is
    // patched by either an `X86_64_RELOC_BRANCH` relocation (Mach-O) or
    // the linker (ELF). At MVP the placeholder displacement is zero; this
    // test verifies the object emits without error and carries both
    // function symbols.
    //
    // We intentionally do not decode the Mach-O relocation table here —
    // that's validated inside the X86Pipeline unit tests. This test pins
    // the end-to-end dispatcher wiring: multi-function module + intra-
    // module CALL survives the full ISel -> encode -> Mach-O pipeline.
    let mut module = TmirModule::new("test_x86_call");

    // Callee: `fn callee() -> i64 { return 42 }` — i64 constant materialization
    // is already supported, so use it to exercise a non-trivial body.
    let ft_void_i64 = module.add_func_type(FuncTy {
        params: vec![],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut callee = TmirFunction::new(
        FuncId::new(0),
        "callee",
        ft_void_i64,
        BlockId::new(0),
    );
    callee.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: tmir::Constant::Int(42),
            })
            .with_result(ValueId::new(0)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            }),
        ],
    }];
    module.add_function(callee);

    // Caller: `fn caller() -> i64 { return callee() }`.
    // Uses the same signature slot (void -> i64).
    let mut caller = TmirFunction::new(
        FuncId::new(1),
        "caller",
        ft_void_i64,
        BlockId::new(0),
    );
    caller.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![],
        body: vec![
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![],
            })
            .with_result(ValueId::new(0)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            }),
        ],
    }];
    module.add_function(caller);

    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::X86_64,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("x86-64 dispatcher must compile caller + callee in one module (#464)");

    // Mach-O header sanity.
    let magic = read_u32_le(&result.object_code, 0);
    assert_eq!(magic, MH_MAGIC_64);
    let cpu_type = read_u32_le(&result.object_code, 4);
    assert_eq!(cpu_type, CPU_TYPE_X86_64);

    // Both functions must be in the symbol table.
    let obj = &result.object_code;
    let has_callee = obj.windows(7).any(|w| w == b"_callee");
    let has_caller = obj.windows(7).any(|w| w == b"_caller");
    assert!(
        has_callee && has_caller,
        "Mach-O must expose both `_callee` and `_caller` global symbols \
         (has_callee={}, has_caller={})",
        has_callee, has_caller
    );

    // Metrics.
    assert_eq!(result.metrics.function_count, 2);
    assert!(result.metrics.code_size_bytes > 0);
    // A call site encodes to 5 bytes (E8 dd dd dd dd) + return + prologue,
    // so the combined module must be larger than either function alone.
    assert!(
        result.metrics.code_size_bytes >= 10,
        "caller + callee should produce at least 10 bytes of code (got {})",
        result.metrics.code_size_bytes
    );
}

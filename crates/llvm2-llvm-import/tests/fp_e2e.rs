// llvm2-llvm-import / tests / fp_e2e.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// End-to-end test for expansion item #3 (floating-point types + ops):
// take clang-O0-shaped LL snippets that use `float` / `double` and
// drive them through the full codegen pipeline to an AArch64 Mach-O
// object.
//
// Goals:
//   LL -> tMIR (this crate)
//     -> llvm2-lower MachIR (FP BinOp / FCmp / FP casts)
//       -> llvm2-opt / llvm2-regalloc / llvm2-codegen
//         -> Mach-O bytes.
//
// Hermetic: the `.ll` is embedded as a string constant (verified by
// hand against Apple clang 17.0.0 on 2026-04-20, arm64-apple-macosx).
// No dependency on a system `clang` being installed for CI.

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::OptLevel;
use llvm2_codegen::target::Target;
use llvm2_llvm_import::import_text;

/// The f32 program: `float add(float a, float b) { return a + b; }`.
/// Clang -O0 spills params to alloca + reloads before the fadd — we
/// preserve that shape to exercise the real load/store/fadd sequence.
const LL_ADD_F32: &str = r#"
define float @add_f(float %0, float %1) {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, ptr %3, align 4
  store float %1, ptr %4, align 4
  %5 = load float, ptr %3, align 4
  %6 = load float, ptr %4, align 4
  %7 = fadd float %5, %6
  ret float %7
}
"#;

/// The f64 program: `double add(double a, double b) { return a + b; }`.
const LL_ADD_F64: &str = r#"
define double @add_d(double %0, double %1) {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store double %1, ptr %4, align 8
  %5 = load double, ptr %3, align 8
  %6 = load double, ptr %4, align 8
  %7 = fadd double %5, %6
  ret double %7
}
"#;

/// The fcmp-branch program:
///   `int cmp_lt(double x, double y) { if (x < y) return 1; else return 2; }`.
///
/// This is the hand-simplified shape of clang -O0 output with the
/// result alloca dropped: the select between the two return values is
/// instead expressed as distinct basic blocks reached by a `br i1`
/// predicated on `fcmp olt`. That matches the semantics exactly and
/// removes the need for a phi (phis are not yet supported — item #2).
const LL_CMP_LT: &str = r#"
define i32 @cmp_lt(double %x, double %y) {
entry:
  %c = fcmp olt double %x, %y
  br i1 %c, label %lt, label %ge
lt:
  ret i32 1
ge:
  ret i32 2
}
"#;

/// Simple FP-cast round-trip: integer-to-double then double-to-integer
/// via separate instructions. Exercises `sitofp` + `fptosi` in the
/// adapter's cast path.
const LL_INT_DOUBLE_ROUNDTRIP: &str = r#"
define i32 @roundtrip(i32 %x) {
entry:
  %d = sitofp i32 %x to double
  %r = fptosi double %d to i32
  ret i32 %r
}
"#;

/// `fneg` on f64 through the full pipeline.
const LL_FNEG_F64: &str = r#"
define double @neg(double %x) {
entry:
  %n = fneg double %x
  ret double %n
}
"#;

fn compile_to_aarch64(src: &str, module_name: &str) -> Vec<u8> {
    let module = import_text(src, module_name).expect("import");
    let cfg = CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::Aarch64,
        emit_proofs: false,
        trace_level: CompilerTraceLevel::None,
        emit_debug: false,
        parallel: false,
        cegis_superopt_budget_sec: None,
    };
    let compiler = Compiler::new(cfg);
    let result = compiler
        .compile(&module)
        .unwrap_or_else(|e| panic!("compile `{module_name}` failed: {e}"));
    assert!(
        !result.object_code.is_empty(),
        "pipeline returned empty Mach-O buffer for `{module_name}`"
    );
    result.object_code
}

#[test]
fn fadd_f32_import_to_tmir() {
    let m = import_text(LL_ADD_F32, "add_f").expect("import f32 add");
    assert_eq!(m.functions.len(), 1);
    assert_eq!(m.functions[0].name, "add_f");
    // Every block must have a terminator.
    for (i, b) in m.functions[0].blocks.iter().enumerate() {
        assert!(b.terminator().is_some(), "block {} missing terminator", i);
    }
}

#[test]
fn fadd_f32_full_pipeline_aarch64_o0() {
    compile_to_aarch64(LL_ADD_F32, "add_f");
}

#[test]
fn fadd_f64_full_pipeline_aarch64_o0() {
    compile_to_aarch64(LL_ADD_F64, "add_d");
}

#[test]
fn fcmp_olt_branch_full_pipeline_aarch64_o0() {
    compile_to_aarch64(LL_CMP_LT, "cmp_lt");
}

#[test]
fn sitofp_fptosi_roundtrip_full_pipeline_aarch64_o0() {
    compile_to_aarch64(LL_INT_DOUBLE_ROUNDTRIP, "roundtrip");
}

#[test]
fn fneg_f64_full_pipeline_aarch64_o0() {
    compile_to_aarch64(LL_FNEG_F64, "neg");
}

/// Exhaustive smoke test: every FP binop reaches an object file.
#[test]
fn all_fbinops_full_pipeline_aarch64_o0() {
    for (op_ll, op_name) in [
        ("fadd", "fadd"),
        ("fsub", "fsub"),
        ("fmul", "fmul"),
        ("fdiv", "fdiv"),
    ] {
        let src = format!(
            r#"
define double @{op_name}(double %a, double %b) {{
entry:
  %r = {op_ll} double %a, %b
  ret double %r
}}
"#
        );
        compile_to_aarch64(&src, op_name);
    }
}

// llvm2-codegen/tests/e2e_abi_dual_target_linkrun.rs -- link+run ABI fixture (#466)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Companion to `e2e_abi_dual_target.rs`. That sibling file stops at
// disassembly-level assertions ("the ABI surface looks right"). #466 AC-4
// requires the stronger claim: compile through BOTH pipelines, link with a
// C driver, and assert the correct runtime return value. This file closes
// that last AC item.
//
// ## Scope (from #466 / G5 in #445)
//
// The existing e2e_x86_64_link.rs tests only go up to 4 int args (`sum4`)
// and 2 fp args, which stay entirely in SysV's 6-int-reg / 8-fp-reg window.
// They never exercise the stack-spill path for formal arguments.
//
// The dual-target disassembly test (e2e_abi_dual_target.rs) builds a 10-arg
// fixture (8 int + 2 fp), but deliberately `return i0` so the function body
// is ABI-only. We can observe stack-arg loads in the disassembly but we do
// NOT know they compute the right value at runtime.
//
// This file:
//   * Builds a tMIR fixture `sum8_i64(i0..i7: i64) -> i64 { i0+i1+...+i7 }`
//     which forces stack-spill for i6 and i7 on SysV AMD64 (RDI,RSI,RDX,
//     RCX,R8,R9 take args 0-5; args 6-7 come from [RBP+0x10] / [RBP+0x18]).
//     The body touches every argument, so a bug in `lower_formal_arguments`'
//     stack-slot computation would produce a wrong sum at runtime.
//   * Builds a mixed int+fp fixture `sum6i_3d(i0..i5: i64, f0..f2: f64) -> f64`
//     which exercises the INDEPENDENT int/fp counters of SysV AMD64 (all 6
//     ints in GPRs, all 3 fps in XMM0/XMM1/XMM2) and AAPCS64 (X0-X5 + V0-V2).
//   * Compiles each module through BOTH AArch64 (AAPCS64) and x86-64 (SysV)
//     `Compiler` dispatchers. Links with a C driver. Runs. Asserts return.
//
// AArch64 runs natively on the Apple-Silicon host; x86-64 runs under Rosetta 2
// (same path as e2e_x86_64_link.rs tests 1-20).
//
// ## Why these numbers
//
// For sum8_i64: 8 integer args are the smallest count that forces stack-spill
// in SysV AMD64 (6 GPR arg slots). AAPCS64 has 8 GPR arg slots, so the same
// fixture stays entirely in registers on AArch64 -- a good ABI-contrast.
// Running both confirms both pipelines correctly materialize all 8 args.
//
// For sum6i_3d: the int and fp counters are independent in both ABIs.
// Emitting the same fixture on both targets and asserting the correct result
// verifies (a) the fp-arg path in both pipelines, and (b) that the mixed
// int/fp layout does not accidentally overlap in either pipeline.
//
// ## What this does NOT cover
//
// * Struct-by-value / sret. Listed in #466 as "if tMIR uses them" -- current
//   tMIR MVP does not use aggregate args; out of scope.
// * `tla2`-specific instruction mix beyond int+fp arithmetic. That is a
//   separate per-workload benchmark, not an ABI test.
//
// ## Known x86_64 bugs discovered while writing this file
//
// The x86_64 `sum8_i64` and `sum6i_3d` fixtures produce the wrong runtime
// values due to a parallel-copy bug in `lower_formal_arguments`:
//   #497 -- int arg side (RDI/RSI/RDX/RCX/R8/R9 moves clobber each other)
//   #499 -- fp arg side  (XMM0/XMM1/XMM2 moves clobber each other)
// Both tests therefore assert the observed wrong values (known-bug assertions)
// and will need tightening back to `assert_eq!(exit, 0)` when #497 / #499 land.
// The AArch64 variants assert strict correctness because AAPCS64 lowering is
// unaffected.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::OptLevel;
use llvm2_codegen::target::Target;

use tmir::{
    BinOp, Block as TmirBlock, BlockId, CastOp, Constant, FuncId, FuncTy,
    Function as TmirFunction, Inst, InstrNode, Module as TmirModule, Ty, ValueId,
};

// =============================================================================
// Host / toolchain probes
// =============================================================================

/// Check `cc -arch <arch>` can produce and run a trivial binary. On Apple
/// Silicon this means: `arm64` native, `x86_64` via Rosetta 2.
fn has_cc_arch(arch: &str) -> bool {
    let dir = std::env::temp_dir().join(format!("llvm2_cc_check_{}", arch));
    let _ = fs::create_dir_all(&dir);
    let src = dir.join("check.c");
    let out = dir.join("check");
    let _ = fs::write(&src, b"int main(void) { return 0; }\n");

    let compiled = Command::new("cc")
        .args(["-arch", arch, "-o", out.to_str().unwrap(), src.to_str().unwrap()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let ran = compiled
        && Command::new(out.to_str().unwrap())
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    let _ = fs::remove_dir_all(&dir);
    ran
}

fn has_cc_aarch64() -> bool {
    has_cc_arch("arm64")
}
fn has_cc_x86_64() -> bool {
    has_cc_arch("x86_64")
}

// =============================================================================
// Link + run helpers (one per arch; mirror `e2e_x86_64_link.rs` shape).
// =============================================================================

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_abi_linkrun_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir tmp");
    dir
}

fn write_file(dir: &Path, name: &str, data: &[u8]) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, data).expect("write file");
    path
}

/// Link a C driver with an object file under `cc -arch <arch>`, returning
/// the path to the linked binary. On failure, dumps otool / nm so the
/// diagnostic is usable.
fn link(arch: &str, dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .args([
            "-arch", arch,
            "-o", binary.to_str().unwrap(),
            driver_c.to_str().unwrap(),
            obj.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run cc");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        let otool_out = Command::new("otool")
            .args(["-tv", obj.to_str().unwrap()])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let nm_out = Command::new("nm")
            .args([obj.to_str().unwrap()])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        panic!(
            "link failed for {} ({}!):\ncc stdout: {}\ncc stderr: {}\notool -tv:\n{}\nnm:\n{}",
            output_name, arch, stdout, stderr, otool_out, nm_out
        );
    }
    binary
}

/// Compile a tMIR module through the public `Compiler` API for the given
/// target and return the Mach-O object bytes.
fn compile_for(target: Target, module: &TmirModule) -> Vec<u8> {
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0, // minimise optimisation-induced divergence
        target,
        emit_proofs: false,      // x86_64 + proofs is #465, out of scope here
        trace_level: CompilerTraceLevel::None,
        emit_debug: false,
        parallel: false,
        cegis_superopt_budget_sec: None,
    });
    let result = compiler
        .compile(module)
        .unwrap_or_else(|e| panic!("{:?} compile failed: {:?}", target, e));
    assert!(
        !result.object_code.is_empty(),
        "{:?} compile produced empty object",
        target
    );
    result.object_code
}

/// End-to-end: compile `module` for `target`, link with `driver_src`, run,
/// return (exit_code, stdout_string).
fn link_and_run(
    target: Target,
    test_name: &str,
    module: &TmirModule,
    driver_src: &str,
) -> (i32, String) {
    let arch = match target {
        Target::Aarch64 => "arm64",
        Target::X86_64 => "x86_64",
        other => panic!("unsupported target for link+run: {:?}", other),
    };

    let obj = compile_for(target, module);
    let dir = make_test_dir(&format!("{}_{}", arch, test_name));
    let obj_path = write_file(&dir, &format!("{}.o", test_name), &obj);
    let driver_path = write_file(&dir, "driver.c", driver_src.as_bytes());
    let binary = link(arch, &dir, &driver_path, &obj_path, &format!("bin_{}", test_name));

    let run_out = Command::new(binary.to_str().unwrap())
        .output()
        .unwrap_or_else(|e| panic!("running {} binary failed: {}", arch, e));

    let stdout = String::from_utf8_lossy(&run_out.stdout).to_string();
    let exit = run_out.status.code().unwrap_or(-1);

    eprintln!("[{} {}] exit={}, stdout={}", arch, test_name, exit, stdout.trim());

    // Keep dir if LLVM2_KEEP_LINKRUN is set, for post-mortem disassembly.
    if std::env::var_os("LLVM2_KEEP_LINKRUN").is_none() {
        let _ = fs::remove_dir_all(&dir);
    } else {
        eprintln!("[{} {}] kept artifacts in {}", arch, test_name, dir.display());
    }
    (exit, stdout)
}

// =============================================================================
// tMIR fixtures
// =============================================================================

/// `fn _sum8_i64(i0..i7: i64) -> i64 { i0 + i1 + ... + i7 }`
///
/// Eight i64 args. On SysV AMD64, args 6 and 7 spill to the stack at
/// `[RBP+0x10]` and `[RBP+0x18]`. On AAPCS64, all eight fit in X0..X7 --
/// both ABIs exercised, neither should silently drop an arg.
///
/// Body: left-fold add. All eight ValueIds flow into the final sum, so the
/// return value is a sensitive function of every formal-arg load.
fn build_sum8_i64_module() -> TmirModule {
    let mut module = TmirModule::new("sum8_i64_test");
    let ft = module.add_func_type(FuncTy {
        params: vec![Ty::I64; 8],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sum8_i64", ft, BlockId::new(0));

    // Params: v0..v7
    let params: Vec<(ValueId, Ty)> = (0..8).map(|i| (ValueId::new(i), Ty::I64)).collect();

    // Body: acc = v0; for i in 1..8 { acc = acc + vi; } return acc
    // The accumulator reuses a fresh ValueId each time (tMIR is SSA).
    let mut body = Vec::new();
    let mut acc = ValueId::new(0); // v0
    let mut next_id: u32 = 8;
    for i in 1..8u32 {
        let res = ValueId::new(next_id);
        next_id += 1;
        body.push(
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: acc,
                rhs: ValueId::new(i),
            })
            .with_result(res),
        );
        acc = res;
    }
    body.push(InstrNode::new(Inst::Return { values: vec![acc] }));

    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params,
        body,
    }];
    module.add_function(func);
    module
}

/// `fn _sum6i_3d(i0..i5: i64, f0..f2: f64) -> f64 {
///     (i0+i1+i2+i3+i4+i5) as f64 + f0 + f1 + f2
/// }`
///
/// Six int args (all fit in GPRs on both ABIs) and three fp args (all fit
/// in the respective fp arg regs on both ABIs). Exercises the INDEPENDENT
/// int/fp counters on both SysV and AAPCS64 -- a common site for classifier
/// bugs (e.g., accidentally sharing a counter).
///
/// Body shape:
///   int_sum_i64 = v0+v1+v2+v3+v4+v5        (6 i64 adds)
///   int_sum_f64 = SIToFP(int_sum_i64)      (explicit int->float convert)
///   t1          = int_sum_f64 + v6         (f64 add)
///   t2          = t1 + v7
///   t3          = t2 + v8
///   return t3
fn build_sum6i_3d_module() -> TmirModule {
    let mut module = TmirModule::new("sum6i_3d_test");
    let ft = module.add_func_type(FuncTy {
        params: vec![
            Ty::I64, Ty::I64, Ty::I64, Ty::I64, Ty::I64, Ty::I64,
            Ty::F64, Ty::F64, Ty::F64,
        ],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sum6i_3d", ft, BlockId::new(0));

    let params: Vec<(ValueId, Ty)> = vec![
        (ValueId::new(0), Ty::I64),
        (ValueId::new(1), Ty::I64),
        (ValueId::new(2), Ty::I64),
        (ValueId::new(3), Ty::I64),
        (ValueId::new(4), Ty::I64),
        (ValueId::new(5), Ty::I64),
        (ValueId::new(6), Ty::F64),
        (ValueId::new(7), Ty::F64),
        (ValueId::new(8), Ty::F64),
    ];

    let mut body = Vec::new();
    // int_sum_i64 via left-fold add of v0..v5
    let mut acc = ValueId::new(0);
    let mut next_id: u32 = 9;
    for i in 1..6u32 {
        let res = ValueId::new(next_id);
        next_id += 1;
        body.push(
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: acc,
                rhs: ValueId::new(i),
            })
            .with_result(res),
        );
        acc = res;
    }
    // SIToFP: int_sum_i64 -> int_sum_f64
    let int_sum_f64 = ValueId::new(next_id);
    next_id += 1;
    body.push(
        InstrNode::new(Inst::Cast {
            op: CastOp::SIToFP,
            src_ty: Ty::I64,
            dst_ty: Ty::F64,
            operand: acc,
        })
        .with_result(int_sum_f64),
    );
    // Fold in the three f64 args.
    let mut facc = int_sum_f64;
    for i in 6..9u32 {
        let res = ValueId::new(next_id);
        next_id += 1;
        body.push(
            InstrNode::new(Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::F64,
                lhs: facc,
                rhs: ValueId::new(i),
            })
            .with_result(res),
        );
        facc = res;
    }
    body.push(InstrNode::new(Inst::Return { values: vec![facc] }));

    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params,
        body,
    }];
    module.add_function(func);
    module
}

/// A simpler fixture that returns a constant, used as a smoke test before we
/// run the more ambitious ABI tests. If THIS fails, the test infrastructure
/// itself is broken and the more complex failures would be misleading.
fn build_const_sentinel_module(name: &str, value: i64) -> TmirModule {
    let mut module = TmirModule::new("const_sentinel");
    let ft = module.add_func_type(FuncTy {
        params: vec![],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), name, ft, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(value.into()),
            })
            .with_result(ValueId::new(0)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

// =============================================================================
// Sanity: const sentinel round-trips on both targets.
//
// If these two tests fail, the link+run harness itself is broken. Keep them
// first so diagnostic output is unambiguous.
// =============================================================================

#[test]
fn test_abi_linkrun_const_sentinel_aarch64() {
    if !has_cc_aarch64() {
        eprintln!("SKIP: cc -arch arm64 not available");
        return;
    }
    let module = build_const_sentinel_module("_sentinel_aarch64", 12345);
    let driver = r#"
#include <stdio.h>
extern long _sentinel_aarch64(void);
int main(void) {
    long r = _sentinel_aarch64();
    printf("sentinel=%ld\n", r);
    return (r == 12345) ? 0 : 1;
}
"#;
    let (exit, stdout) = link_and_run(Target::Aarch64, "sentinel", &module, driver);
    assert_eq!(exit, 0, "aarch64 sentinel failed: stdout={}", stdout);
    assert!(stdout.contains("12345"));
}

#[test]
fn test_abi_linkrun_const_sentinel_x86_64() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available (no Rosetta 2)");
        return;
    }
    let module = build_const_sentinel_module("_sentinel_x86_64", 12345);
    let driver = r#"
#include <stdio.h>
extern long _sentinel_x86_64(void);
int main(void) {
    long r = _sentinel_x86_64();
    printf("sentinel=%ld\n", r);
    return (r == 12345) ? 0 : 1;
}
"#;
    let (exit, stdout) = link_and_run(Target::X86_64, "sentinel", &module, driver);
    assert_eq!(exit, 0, "x86_64 sentinel failed: stdout={}", stdout);
    assert!(stdout.contains("12345"));
}

// =============================================================================
// Test A: sum8_i64 -- 8 int args, forces SysV stack-spill for args 6 and 7.
//
// Inputs chosen so each arg is distinct, non-zero, and the sum is small
// enough to not overflow i64 but large enough that dropping any single arg
// changes the result by an amount the driver can detect.
//
//   1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
// =============================================================================

const SUM8_ARGS: [i64; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const SUM8_EXPECTED: i64 = 255;

#[test]
fn test_abi_linkrun_sum8_i64_aarch64() {
    if !has_cc_aarch64() {
        eprintln!("SKIP: cc -arch arm64 not available");
        return;
    }
    let module = build_sum8_i64_module();
    let driver = format!(
        r#"
#include <stdio.h>
extern long _sum8_i64(long,long,long,long,long,long,long,long);
int main(void) {{
    long r = _sum8_i64({},{},{},{},{},{},{},{});
    printf("sum8=%ld\n", r);
    return (r == {}) ? 0 : 1;
}}
"#,
        SUM8_ARGS[0], SUM8_ARGS[1], SUM8_ARGS[2], SUM8_ARGS[3],
        SUM8_ARGS[4], SUM8_ARGS[5], SUM8_ARGS[6], SUM8_ARGS[7],
        SUM8_EXPECTED
    );
    let (exit, stdout) = link_and_run(Target::Aarch64, "sum8", &module, &driver);
    assert_eq!(
        exit, 0,
        "aarch64 sum8_i64 failed: expected {}, stdout={}",
        SUM8_EXPECTED, stdout
    );
    assert!(
        stdout.contains(&SUM8_EXPECTED.to_string()),
        "aarch64 stdout should contain {}, got: {}",
        SUM8_EXPECTED, stdout
    );
}

#[test]
fn test_abi_linkrun_sum8_i64_x86_64() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available (no Rosetta 2)");
        return;
    }
    let module = build_sum8_i64_module();
    let driver = format!(
        r#"
#include <stdio.h>
extern long _sum8_i64(long,long,long,long,long,long,long,long);
int main(void) {{
    long r = _sum8_i64({},{},{},{},{},{},{},{});
    printf("sum8=%ld\n", r);
    return (r == {}) ? 0 : 1;
}}
"#,
        SUM8_ARGS[0], SUM8_ARGS[1], SUM8_ARGS[2], SUM8_ARGS[3],
        SUM8_ARGS[4], SUM8_ARGS[5], SUM8_ARGS[6], SUM8_ARGS[7],
        SUM8_EXPECTED
    );
    // Strict correctness: #497 is fixed by the post-regalloc formal-arg
    // parallel-copy resolver in x86_64/pipeline.rs. The GPR arg prologue now
    // resolves cycles using R11 as scratch, so all 8 args arrive intact.
    let (exit, stdout) = link_and_run(Target::X86_64, "sum8", &module, &driver);
    assert_eq!(
        exit, 0,
        "x86_64 sum8_i64 failed: expected {}, stdout={}",
        SUM8_EXPECTED, stdout
    );
    assert!(
        stdout.contains(&SUM8_EXPECTED.to_string()),
        "x86_64 stdout should contain {}, got: {}",
        SUM8_EXPECTED, stdout
    );
}

// =============================================================================
// Test B: sum6i_3d -- 6 int + 3 fp args, independent int/fp arg counters.
//
// Inputs:
//   int args:  10, 20, 30, 40, 50, 60   (sum = 210)
//   fp args:   0.5, 0.25, 0.125          (sum = 0.875)
//   expected: 210.875
// =============================================================================

#[test]
fn test_abi_linkrun_sum6i_3d_aarch64() {
    if !has_cc_aarch64() {
        eprintln!("SKIP: cc -arch arm64 not available");
        return;
    }
    let module = build_sum6i_3d_module();
    let driver = r#"
#include <stdio.h>
#include <math.h>
extern double _sum6i_3d(long,long,long,long,long,long,double,double,double);
int main(void) {
    double r = _sum6i_3d(10,20,30,40,50,60, 0.5,0.25,0.125);
    printf("sum6i_3d=%.3f\n", r);
    // Exact bit-equality: all operands are powers of two fractions, so the
    // result is representable exactly in f64. No tolerance needed.
    return (r == 210.875) ? 0 : 1;
}
"#;
    let (exit, stdout) = link_and_run(Target::Aarch64, "sum6i_3d", &module, driver);
    assert_eq!(
        exit, 0,
        "aarch64 sum6i_3d failed: expected 210.875, stdout={}",
        stdout
    );
    assert!(
        stdout.contains("210.875"),
        "aarch64 stdout should contain 210.875, got: {}",
        stdout
    );
}

#[test]
fn test_abi_linkrun_sum6i_3d_x86_64() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available (no Rosetta 2)");
        return;
    }
    let module = build_sum6i_3d_module();
    let driver = r#"
#include <stdio.h>
#include <math.h>
extern double _sum6i_3d(long,long,long,long,long,long,double,double,double);
int main(void) {
    double r = _sum6i_3d(10,20,30,40,50,60, 0.5,0.25,0.125);
    printf("sum6i_3d=%.3f\n", r);
    return (r == 210.875) ? 0 : 1;
}
"#;
    // Strict correctness: #499 is fixed by the post-regalloc formal-arg
    // parallel-copy resolver in x86_64/pipeline.rs. The XMM arg prologue now
    // resolves cycles using XMM15 as scratch, so all 3 fp args arrive intact.
    let (exit, stdout) = link_and_run(Target::X86_64, "sum6i_3d", &module, driver);
    assert_eq!(
        exit, 0,
        "x86_64 sum6i_3d failed: expected 210.875, stdout={}",
        stdout
    );
    assert!(
        stdout.contains("210.875"),
        "x86_64 stdout should contain 210.875, got: {}",
        stdout
    );
}

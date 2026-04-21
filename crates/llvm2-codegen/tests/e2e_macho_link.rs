// llvm2-codegen/tests/e2e_macho_link.rs - E2E Mach-O linking integration test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves the AArch64 backend works end-to-end:
//   tMIR -> adapter -> ISel -> opt -> regalloc -> frame -> encode -> Mach-O -> otool -> ld -> run
//
// This is the P1 critical integration test for issue #198. It verifies that
// all pipeline stages compose correctly by producing a .o file that:
//   1. Has valid Mach-O magic and structure (in-process binary parsing)
//   2. Disassembles correctly with otool -tv
//   3. Links with the system linker (cc / ld)
//   4. Executes correctly and produces expected output
//
// Part of #198 -- End-to-end Mach-O linking test

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::macho::constants::*;
use llvm2_codegen::pipeline::{Pipeline, PipelineConfig, OptLevel};

use tmir::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule, FuncTy, Ty, Constant};
use tmir::{Inst, InstrNode, BinOp, ICmpOp};
use tmir::{BlockId, FuncId, ValueId};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

fn has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_otool() -> bool {
    Command::new("otool")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_nm() -> bool {
    Command::new("nm")
        .arg("--version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_link_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write .o file");
    path
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("Failed to write C driver");
    path
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn run_binary_with_output(binary: &Path) -> (i32, String) {
    use std::time::Duration;

    let mut child = Command::new(binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn binary");

    let timeout = Duration::from_secs(10);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = {
                    let mut s = String::new();
                    if let Some(mut out) = child.stdout.take() {
                        use std::io::Read;
                        let _ = out.read_to_string(&mut s);
                    }
                    s
                };
                return (status.code().unwrap_or(-1), stdout);
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "Binary {} timed out after {:?}",
                        binary.display(),
                        timeout
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("Error waiting for binary: {}", e),
        }
    }
}

// ---------------------------------------------------------------------------
// Mach-O binary parsing helpers
// ---------------------------------------------------------------------------

fn verify_macho_magic(bytes: &[u8]) {
    assert!(bytes.len() >= 4, "Object file too small: {} bytes", bytes.len());
    assert_eq!(
        &bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "Invalid Mach-O magic: expected 0xFEEDFACF (LE)"
    );
}

fn macho_filetype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]])
}

fn macho_cputype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

fn macho_ncmds(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]])
}

/// Search for a named section in the Mach-O load commands.
/// Returns (file_offset, size) if found.
fn find_section(bytes: &[u8], target_sectname: &[u8]) -> Option<(u32, u64)> {
    let sizeofcmds = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    let mut offset = MACH_HEADER_64_SIZE as usize;
    let lc_end = MACH_HEADER_64_SIZE as usize + sizeofcmds as usize;

    while offset < lc_end && offset + 8 <= bytes.len() {
        let cmd = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]);
        let cmdsize = u32::from_le_bytes([bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]]) as usize;

        if cmd == LC_SEGMENT_64 {
            let nsects = u32::from_le_bytes([
                bytes[offset + 64], bytes[offset + 65],
                bytes[offset + 66], bytes[offset + 67],
            ]);
            let mut sec_offset = offset + SEGMENT_COMMAND_64_SIZE as usize;

            for _ in 0..nsects {
                let sectname = &bytes[sec_offset..sec_offset + 16];
                let name_len = sectname.iter().position(|&b| b == 0).unwrap_or(16);
                let name = &sectname[..name_len];

                if name == target_sectname {
                    let sec_size = u64::from_le_bytes([
                        bytes[sec_offset + 40], bytes[sec_offset + 41],
                        bytes[sec_offset + 42], bytes[sec_offset + 43],
                        bytes[sec_offset + 44], bytes[sec_offset + 45],
                        bytes[sec_offset + 46], bytes[sec_offset + 47],
                    ]);
                    let sec_file_offset = u32::from_le_bytes([
                        bytes[sec_offset + 48], bytes[sec_offset + 49],
                        bytes[sec_offset + 50], bytes[sec_offset + 51],
                    ]);
                    return Some((sec_file_offset, sec_size));
                }

                sec_offset += SECTION_64_SIZE as usize;
            }
        }

        offset += cmdsize;
    }

    None
}

// ---------------------------------------------------------------------------
// External tool runners
// ---------------------------------------------------------------------------

fn run_otool_tv(path: &Path) -> Option<String> {
    let output = Command::new("otool")
        .args(["-tv", path.to_str().unwrap()])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

fn run_otool_l(path: &Path) -> Option<String> {
    let output = Command::new("otool")
        .args(["-l", path.to_str().unwrap()])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

fn run_nm(path: &Path) -> Option<String> {
    let output = Command::new("nm")
        .args([path.to_str().unwrap()])
        .output()
        .ok()?;

    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

fn link_with_cc(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(driver_c)
        .arg(obj)
        .arg("-Wl,-no_pie")
        .output()
        .expect("Failed to run cc");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        panic!(
            "Linking failed!\ncc stdout: {}\ncc stderr: {}\nDriver: {}\nObject: {}",
            stdout, stderr, driver_c.display(), obj.display()
        );
    }

    binary
}

// ---------------------------------------------------------------------------
// Helper: compile a tMIR function through the full pipeline
// ---------------------------------------------------------------------------

fn compile_tmir_function(
    tmir_func: &TmirFunction,
    module: &TmirModule,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    // Phase 0: Translate tMIR -> LIR (adapter)
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(tmir_func, module)
            .map_err(|e| format!("adapter error: {}", e))?;

    // Phase 1-9: Compile LIR through full pipeline
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    let obj_bytes = pipeline
        .compile_function(&lir_func)
        .map_err(|e| format!("pipeline error: {}", e))?;

    Ok(obj_bytes)
}

// ---------------------------------------------------------------------------
// tMIR function builders
// ---------------------------------------------------------------------------

/// Build tMIR for: fn add_i32(a: i32, b: i32) -> i32 { a + b }
///
/// This is the simplest possible two-argument function: adds two i32 values
/// and returns the result. Tests the full pipeline for 32-bit integer types.
fn build_add_i32_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I32, Ty::I32],
            returns: vec![Ty::I32], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "add_i32", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I32),
                (ValueId::new(1), Ty::I32),
            ],
            body: vec![
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I32,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })

                    .with_result(ValueId::new(2)),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(2)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn add_i64(a: i64, b: i64) -> i64 { a + b }
///
/// 64-bit variant of the add function.
fn build_add_i64_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "add_i64", ft_id, BlockId::new(0));
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
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn sub_mul(a: i64, b: i64, c: i64) -> i64 { (a - b) * c }
///
/// Multi-operation function: tests that chained operations produce correct
/// Mach-O output that links and executes correctly.
fn build_sub_mul_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "sub_mul", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64), // a
                (ValueId::new(1), Ty::I64), // b
                (ValueId::new(2), Ty::I64), // c
            ],
            body: vec![
                // tmp = a - b
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })

                    .with_result(ValueId::new(3)),
                // result = tmp * c
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: ValueId::new(3),
                        rhs: ValueId::new(2),
                    })

                    .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(4)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn abs_val(n: i64) -> i64 { if n < 0 { -n } else { n } }
///
/// Tests conditional branching + negation, producing a function with multiple
/// basic blocks that must link correctly (branch offsets must be valid).
fn build_abs_val_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "abs_val", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): cmp n < 0, branch
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // n
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })

                        .with_result(ValueId::new(1)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Slt,
                            ty: Ty::I64,
                            lhs: ValueId::new(0), // n
                            rhs: ValueId::new(1), // 0
                        })

                        .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(2),
                            then_target: BlockId::new(1), // negate
                            then_args: vec![],
                            else_target: BlockId::new(2), // return n
                            else_args: vec![],
                        }),
                ],
            },
            // bb1: return -n (via 0 - n)
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::I64,
                            lhs: ValueId::new(1), // 0
                            rhs: ValueId::new(0), // n
                        })

                        .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::Return {
                            values: vec![ValueId::new(3)],
                        }),
                ],
            },
            // bb2: return n
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(0)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

// ===========================================================================
// TEST 1: add_i32 -- compile, validate .o structure, verify with otool
// ===========================================================================

/// Compiles fn add_i32(i32, i32) -> i32 through the full pipeline and
/// validates the resulting Mach-O object file structure in-process.
#[test]
fn test_e2e_link_add_i32_macho_structure() {
    let (tmir_func, module) = build_add_i32_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile add_i32");

    // -- Mach-O header validation --
    verify_macho_magic(&obj_bytes);
    assert_eq!(macho_filetype(&obj_bytes), MH_OBJECT, "must be MH_OBJECT");
    assert_eq!(macho_cputype(&obj_bytes), CPU_TYPE_ARM64, "must be ARM64");
    assert!(macho_ncmds(&obj_bytes) >= 3, "need LC_SEGMENT_64 + LC_SYMTAB + LC_DYSYMTAB at minimum");

    // -- __text section --
    let text = find_section(&obj_bytes, b"__text");
    assert!(text.is_some(), "__text section must be present");
    let (text_offset, text_size) = text.unwrap();
    assert!(text_offset > 0, "__text offset must be non-zero");
    assert!(text_size > 0, "__text must have code");
    assert_eq!(text_size % 4, 0, "code size must be 4-byte aligned (AArch64)");

    // -- __compact_unwind section --
    let cu = find_section(&obj_bytes, b"__compact_unwind");
    assert!(cu.is_some(), "__compact_unwind section must be present");
    let (_, cu_size) = cu.unwrap();
    assert_eq!(cu_size, 32, "single function = one 32-byte compact unwind entry");

    // -- Verify code contains ADD and RET instructions --
    let code = &obj_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];
    let mut found_add = false;
    let mut found_ret = false;
    for i in (0..code.len()).step_by(4) {
        let word = u32::from_le_bytes([code[i], code[i+1], code[i+2], code[i+3]]);
        // ADD (register, 64-bit): bits [31:21] = 10001011000
        // ADD (register, 32-bit): bits [31:21] = 00001011000
        if (word >> 21) == 0b10001011000 || (word >> 21) == 0b00001011000 {
            found_add = true;
        }
        // ADD (immediate, 64-bit): bits [31:22] = 1001000100
        // ADD (immediate, 32-bit): bits [31:22] = 0001000100
        if (word >> 22) == 0b1001000100 || (word >> 22) == 0b0001000100 {
            found_add = true;
        }
        if word == 0xD65F03C0 {
            found_ret = true;
        }
    }
    assert!(found_add, "code must contain an ADD instruction. Bytes: {:02X?}", code);
    assert!(found_ret, "code must contain a RET instruction. Bytes: {:02X?}", code);
}

/// Validates the add_i32 .o file with otool and nm.
#[test]
fn test_e2e_link_add_i32_otool_validation() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("link_add_i32_otool");
    let (tmir_func, module) = build_add_i32_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile add_i32");
    let obj_path = write_object_file(&dir, "add_i32.o", &obj_bytes);

    // -- otool -l: load commands --
    if let Some(lc_out) = run_otool_l(&obj_path) {
        assert!(lc_out.contains("LC_SEGMENT_64"), "must have LC_SEGMENT_64");
        assert!(lc_out.contains("LC_SYMTAB"), "must have LC_SYMTAB");
        assert!(lc_out.contains("LC_DYSYMTAB"), "must have LC_DYSYMTAB");
        assert!(lc_out.contains("LC_BUILD_VERSION"), "must have LC_BUILD_VERSION");
        assert!(lc_out.contains("__text"), "must have __text section");
        assert!(lc_out.contains("__compact_unwind"), "must have __compact_unwind");
    }

    // -- otool -tv: disassembly --
    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("add_i32 disassembly:\n{}", disasm);

        assert!(
            disasm.contains("add") || disasm.contains("ADD"),
            "disassembly must contain 'add'.\n{}", disasm
        );
        assert!(
            disasm.contains("ret"),
            "disassembly must contain 'ret'.\n{}", disasm
        );
        assert!(
            disasm.contains("add_i32"),
            "disassembly must reference function name.\n{}", disasm
        );
        // Frame lowering should produce STP/LDP (prologue/epilogue)
        assert!(
            disasm.contains("stp") || disasm.contains("STP"),
            "disassembly must contain STP (frame prologue).\n{}", disasm
        );
        assert!(
            disasm.contains("ldp") || disasm.contains("LDP"),
            "disassembly must contain LDP (frame epilogue).\n{}", disasm
        );
    }

    // -- nm: symbol table --
    if has_nm()
        && let Some(nm_out) = run_nm(&obj_path) {
            assert!(
                nm_out.contains("_add_i32"),
                "nm must show _add_i32 symbol.\n{}", nm_out
            );
            assert!(
                nm_out.contains(" T "),
                "add_i32 must be a global text symbol (T).\n{}", nm_out
            );
        }

    cleanup(&dir);
}

// ===========================================================================
// TEST 2: add_i32 -- link with system linker and execute
// ===========================================================================

/// The definitive end-to-end test: compile tMIR fn add_i32(i32, i32) -> i32,
/// produce a .o file, link it with a C driver using cc, and execute.
/// This proves the entire pipeline produces correct, linkable machine code.
#[test]
fn test_e2e_link_add_i32_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("link_add_i32_run");

    let (tmir_func, module) = build_add_i32_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile add_i32");

    let obj_path = write_object_file(&dir, "add_i32.o", &obj_bytes);

    // C driver that calls our compiled function and validates results.
    // Note: add_i32 compiles as i32 operations. On AArch64, i32 values live
    // in the lower 32 bits of X registers. We use int (32-bit) in the C
    // prototype to match, and cast to long for printf to avoid sign issues.
    let driver_src = r#"
#include <stdio.h>
extern int add_i32(int a, int b);
int main(void) {
    int r1 = add_i32(3, 4);
    int r2 = add_i32(0, 0);
    int r3 = add_i32(-5, 10);
    int r4 = add_i32(1000000, 2000000);
    int r5 = add_i32(-100, -200);
    printf("add_i32(3,4)=%d add_i32(0,0)=%d add_i32(-5,10)=%d add_i32(1M,2M)=%d add_i32(-100,-200)=%d\n",
           r1, r2, r3, r4, r5);
    if (r1 != 7) return 1;
    if (r2 != 0) return 2;
    if (r3 != 5) return 3;
    if (r4 != 3000000) return 4;
    if (r5 != -300) return 5;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_add_i32");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("add_i32 link+run stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "add_i32 link+run failed with exit code {} \
         (1=add(3,4)!=7, 2=add(0,0)!=0, 3=add(-5,10)!=5, 4=add(1M,2M)!=3M, 5=add(-100,-200)!=-300). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// TEST 3: add_i64 -- 64-bit variant link and run
// ===========================================================================

/// 64-bit add: compile, link, and run.
#[test]
fn test_e2e_link_add_i64_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("link_add_i64_run");

    let (tmir_func, module) = build_add_i64_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile add_i64");

    let obj_path = write_object_file(&dir, "add_i64.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long add_i64(long a, long b);
int main(void) {
    long r1 = add_i64(3, 4);
    long r2 = add_i64(0, 0);
    long r3 = add_i64(-5, 10);
    long r4 = add_i64(2147483647L, 1L);  /* overflow i32 but not i64 */
    printf("add_i64(3,4)=%ld add_i64(0,0)=%ld add_i64(-5,10)=%ld add_i64(INT_MAX+1)=%ld\n",
           r1, r2, r3, r4);
    if (r1 != 7) return 1;
    if (r2 != 0) return 2;
    if (r3 != 5) return 3;
    if (r4 != 2147483648L) return 4;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_add_i64");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("add_i64 link+run stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "add_i64 link+run failed with exit code {}. stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// TEST 4: sub_mul -- multi-operation function link and run
// ===========================================================================

/// Chained operations: (a - b) * c. Tests that multiple ALU instructions
/// compose correctly through regalloc and produce valid linked output.
#[test]
fn test_e2e_link_sub_mul_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("link_sub_mul_run");

    let (tmir_func, module) = build_sub_mul_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile sub_mul");

    let obj_path = write_object_file(&dir, "sub_mul.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long sub_mul(long a, long b, long c);
int main(void) {
    long r1 = sub_mul(10, 3, 4);   /* (10-3)*4 = 28 */
    long r2 = sub_mul(5, 5, 100);  /* (5-5)*100 = 0 */
    long r3 = sub_mul(1, 0, 1);    /* (1-0)*1 = 1 */
    long r4 = sub_mul(0, 7, 3);    /* (0-7)*3 = -21 */
    printf("sub_mul(10,3,4)=%ld sub_mul(5,5,100)=%ld sub_mul(1,0,1)=%ld sub_mul(0,7,3)=%ld\n",
           r1, r2, r3, r4);
    if (r1 != 28) return 1;
    if (r2 != 0) return 2;
    if (r3 != 1) return 3;
    if (r4 != -21) return 4;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_sub_mul");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("sub_mul link+run stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "sub_mul link+run failed with exit code {}. stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// TEST 5: abs_val -- conditional branch link and run
// ===========================================================================

/// Conditional branch: if n < 0 return -n else return n.
/// Tests that multi-block functions with branches produce correct linked output.
#[test]
fn test_e2e_link_abs_val_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("link_abs_val_run");

    let (tmir_func, module) = build_abs_val_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile abs_val");

    let obj_path = write_object_file(&dir, "abs_val.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long abs_val(long n);
int main(void) {
    long r1 = abs_val(42);
    long r2 = abs_val(-42);
    long r3 = abs_val(0);
    long r4 = abs_val(-1);
    long r5 = abs_val(1);
    printf("abs(42)=%ld abs(-42)=%ld abs(0)=%ld abs(-1)=%ld abs(1)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != 42) return 1;
    if (r2 != 42) return 2;
    if (r3 != 0) return 3;
    if (r4 != 1) return 4;
    if (r5 != 1) return 5;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_abs_val");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("abs_val link+run stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "abs_val link+run failed with exit code {} \
         (1=abs(42)!=42, 2=abs(-42)!=42, 3=abs(0)!=0, 4=abs(-1)!=1, 5=abs(1)!=1). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// TEST 6: optimization levels produce linkable objects
// ===========================================================================

/// Verify that add_i64 compiled at O0, O1, and O2 all produce valid,
/// linkable, correctly-executing binaries.
#[test]
fn test_e2e_link_opt_levels_all_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("link_opt_levels");
    let (tmir_func, module) = build_add_i64_tmir();

    let driver_src = r#"
#include <stdio.h>
extern long add_i64(long a, long b);
int main(void) {
    long r = add_i64(17, 25);
    printf("add_i64(17,25)=%ld\n", r);
    return (r == 42) ? 0 : 1;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);

    for (i, opt_level) in [OptLevel::O0, OptLevel::O1, OptLevel::O2].iter().enumerate() {
        let obj_bytes = compile_tmir_function(&tmir_func, &module, *opt_level)
            .unwrap_or_else(|e| panic!("pipeline at {:?} failed: {}", opt_level, e));

        // Verify Mach-O structure
        verify_macho_magic(&obj_bytes);
        assert_eq!(macho_filetype(&obj_bytes), MH_OBJECT);

        let obj_filename = format!("add_O{}.o", i);
        let obj_path = write_object_file(&dir, &obj_filename, &obj_bytes);

        let bin_name = format!("test_O{}", i);
        let binary = link_with_cc(&dir, &driver_path, &obj_path, &bin_name);

        let (exit_code, stdout) = run_binary_with_output(&binary);
        assert_eq!(
            exit_code, 0,
            "add_i64 at {:?} failed with exit code {}. stdout: {}",
            opt_level, exit_code, stdout
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST 7: multiple functions -- all link independently
// ===========================================================================

/// Verify that all test functions produce valid Mach-O at the binary level,
/// regardless of whether otool/cc are available (pure in-process checks).
#[test]
fn test_e2e_link_all_functions_produce_valid_macho() {
    let functions: Vec<(&str, TmirFunction, TmirModule)> = vec![
        { let (f, m) = build_add_i32_tmir(); ("add_i32", f, m) },
        { let (f, m) = build_add_i64_tmir(); ("add_i64", f, m) },
        { let (f, m) = build_sub_mul_tmir(); ("sub_mul", f, m) },
        { let (f, m) = build_abs_val_tmir(); ("abs_val", f, m) },
    ];

    for (name, tmir_func, module) in &functions {
        let obj_bytes = compile_tmir_function(tmir_func, module, OptLevel::O0)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        verify_macho_magic(&obj_bytes);
        assert_eq!(macho_filetype(&obj_bytes), MH_OBJECT,
            "{}: must be MH_OBJECT", name);
        assert_eq!(macho_cputype(&obj_bytes), CPU_TYPE_ARM64,
            "{}: must be ARM64", name);

        let text = find_section(&obj_bytes, b"__text");
        assert!(text.is_some(), "{}: __text section must exist", name);
        let (_, text_size) = text.unwrap();
        assert!(text_size > 0, "{}: __text must have code", name);
        assert_eq!(text_size % 4, 0, "{}: code must be 4-byte aligned", name);

        let cu = find_section(&obj_bytes, b"__compact_unwind");
        assert!(cu.is_some(), "{}: __compact_unwind must exist", name);

        // Verify object file size is reasonable (not bloated, not empty)
        assert!(
            obj_bytes.len() >= 100 && obj_bytes.len() <= 10_000,
            "{}: object file size {} is out of range [100, 10000]",
            name, obj_bytes.len()
        );
    }
}

// ===========================================================================
// TEST 8: otool validates all functions' disassembly
// ===========================================================================

#[test]
fn test_e2e_link_all_functions_otool_validates() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("link_all_otool");

    let functions: Vec<(&str, TmirFunction, TmirModule)> = vec![
        { let (f, m) = build_add_i32_tmir(); ("add_i32", f, m) },
        { let (f, m) = build_add_i64_tmir(); ("add_i64", f, m) },
        { let (f, m) = build_sub_mul_tmir(); ("sub_mul", f, m) },
        { let (f, m) = build_abs_val_tmir(); ("abs_val", f, m) },
    ];

    for (name, tmir_func, _module) in &functions {
        let obj_bytes = compile_tmir_function(tmir_func, _module, OptLevel::O0)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        let filename = format!("{}.o", name);
        let obj_path = write_object_file(&dir, &filename, &obj_bytes);

        // otool -tv must succeed and show function name + ret
        let disasm = run_otool_tv(&obj_path);
        assert!(disasm.is_some(), "otool -tv must succeed for {}", name);
        let disasm = disasm.unwrap();
        assert!(disasm.contains("ret"),
            "{}: disassembly must contain 'ret'.\n{}", name, disasm);
        assert!(disasm.contains(name),
            "{}: disassembly must reference function name.\n{}", name, disasm);

        // nm must show the symbol
        if has_nm()
            && let Some(nm_out) = run_nm(&obj_path) {
                let symbol = format!("_{}", name);
                assert!(nm_out.contains(&symbol),
                    "{}: nm must show {} symbol.\n{}", name, symbol, nm_out);
            }
    }

    cleanup(&dir);
}

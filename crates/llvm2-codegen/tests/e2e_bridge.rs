// llvm2-codegen/tests/e2e_bridge.rs - trust-llvm2-bridge integration tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// These tests validate that the LIR format produced by trust-llvm2-bridge
// (in tRust) is valid input for LLVM2's compilation pipeline. Rather than
// depending on tRust/trust-types, we construct llvm2_lower::Function instances
// directly, matching exactly what the bridge would produce.
//
// The trust-llvm2-bridge (~/tRust/crates/trust-llvm2-bridge/src/lib.rs)
// converts trust-types VerifiableFunction -> LLVM2 LIR using these conventions:
//   - Value(0)..Value(arg_count-1) = formal arguments
//   - Additional Values for return slot and temporaries
//   - BasicBlock.params is always empty (params: vec![])
//   - entry_block is Block(0)
//   - stack_slots is empty
//   - BinaryOp -> Iadd/Isub/Imul/Udiv/Sdiv etc.
//   - Constants -> Iconst/Fconst
//   - SwitchInt(1 target) -> Iconst + Icmp(Equal) + Brif
//
// The bridge previously emitted Return with no args (#307), but has been
// fixed to include the return value in Return.args when the function has a
// non-Unit/non-Never return type. These tests validate the ISel-compatible
// format (Return with args) that the bridge now correctly produces.
//
// Part of #301 — trust-llvm2-bridge integration
// Part of #307 — Return value fix

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};
use llvm2_lower::function::{BasicBlock, Function, Signature};
use llvm2_lower::instructions::{Block, Instruction, IntCC, Opcode, Value};
use llvm2_lower::types::Type;

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

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_bridge_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("Failed to write C driver");
    path
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write .o file");
    path
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
            stdout,
            stderr,
            driver_c.display(),
            obj.display()
        );
    }

    binary
}

fn run_binary_with_output(binary: &Path) -> (i32, String) {
    use std::io::Read;
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

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Compile a bridge-format LIR function through the LLVM2 pipeline at O0.
fn compile_bridge_lir(func: &Function) -> Result<Vec<u8>, String> {
    let config = PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    pipeline
        .compile_function(func)
        .map_err(|e| format!("pipeline error: {}", e))
}

// ---------------------------------------------------------------------------
// LIR function builders (matching trust-llvm2-bridge output format)
// ---------------------------------------------------------------------------

/// Build LIR for: fn add(a: i64, b: i64) -> i64 { a + b }
///
/// Conventions:
///   Value(0) = arg a, Value(1) = arg b, Value(2) = add result
///   Block(0): Iadd(args=[V0,V1], results=[V2]), Return(args=[V2])
///
/// ISel requires Return.args to include the return value.
fn build_bridge_add() -> Function {
    let mut blocks = HashMap::new();
    blocks.insert(
        Block(0),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iadd,
                    args: vec![Value(0), Value(1)],
                    results: vec![Value(2)],
                },
                Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(2)],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );

    Function {
        name: "add".to_string(),
        signature: Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        },
        blocks,
        entry_block: Block(0),
        stack_slots: vec![],
        value_types: HashMap::new(),
        pure_callees: HashSet::new(),
    }
}

/// Build LIR for: fn const42() -> i64 { 42 }
///
/// Bridge conventions:
///   Value(0) = return slot
///   Block(0): Iconst{I64, 42}(results=[V0]), Return
fn build_bridge_const42() -> Function {
    let mut blocks = HashMap::new();
    blocks.insert(
        Block(0),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iconst {
                        ty: Type::I64,
                        imm: 42,
                    },
                    args: vec![],
                    results: vec![Value(0)],
                },
                Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(0)],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );

    Function {
        name: "const42".to_string(),
        signature: Signature {
            params: vec![],
            returns: vec![Type::I64],
        },
        blocks,
        entry_block: Block(0),
        stack_slots: vec![],
        value_types: HashMap::new(),
        pure_callees: HashSet::new(),
    }
}

/// Build LIR for: fn midpoint(a: u64, b: u64) -> u64 { (a + b) / 2 }
///
/// Bridge conventions (multi-block):
///   Value(0) = arg a, Value(1) = arg b
///   Value(2) = return slot, Value(3) = add result, Value(4) = const 2, Value(5) = div result
///
///   Block(0): Iadd(V0,V1)->V3, Jump(Block(1))
///   Block(1): Iconst{I64,2}->V4, Udiv(V3,V4)->V5, Return
///
/// Note: the bridge maps Rvalue::Use as value alias (no instruction emitted),
/// so _0 = _5 is just a local_values update. The return uses the last assigned
/// value for local 0 which is V5 after the div.
fn build_bridge_midpoint() -> Function {
    let mut blocks = HashMap::new();
    blocks.insert(
        Block(0),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iadd,
                    args: vec![Value(0), Value(1)],
                    results: vec![Value(3)],
                },
                Instruction {
                    opcode: Opcode::Jump { dest: Block(1) },
                    args: vec![],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );
    blocks.insert(
        Block(1),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iconst {
                        ty: Type::I64,
                        imm: 2,
                    },
                    args: vec![],
                    results: vec![Value(4)],
                },
                Instruction {
                    opcode: Opcode::Udiv,
                    args: vec![Value(3), Value(4)],
                    results: vec![Value(5)],
                },
                Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(5)],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );

    Function {
        name: "midpoint".to_string(),
        signature: Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        },
        blocks,
        entry_block: Block(0),
        stack_slots: vec![],
        value_types: HashMap::new(),
        pure_callees: HashSet::new(),
    }
}

/// Build LIR for: fn pick(x: i64) -> i64 { if x == 0 { 1 } else { 2 } }
///
/// Bridge conventions (SwitchInt with 1 target -> Iconst + Icmp + Brif):
///   Value(0) = arg x
///   Value(1) = return slot (pre-allocated but aliased)
///   Value(2) = const 0, Value(3) = cmp result
///
///   Block(0): Iconst{I64,0}->V2, Icmp(Equal)(V0,V2)->V3, Brif(V3, Block(1), Block(2))
///   Block(1): Iconst{I64,1}->V4, Return  (true branch: x==0 -> return 1)
///   Block(2): Iconst{I64,2}->V5, Return  (false branch: x!=0 -> return 2)
fn build_bridge_pick() -> Function {
    let mut blocks = HashMap::new();
    blocks.insert(
        Block(0),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iconst {
                        ty: Type::I64,
                        imm: 0,
                    },
                    args: vec![],
                    results: vec![Value(2)],
                },
                Instruction {
                    opcode: Opcode::Icmp { cond: IntCC::Equal },
                    args: vec![Value(0), Value(2)],
                    results: vec![Value(3)],
                },
                Instruction {
                    opcode: Opcode::Brif {
                        cond: Value(3),
                        then_dest: Block(1),
                        else_dest: Block(2),
                    },
                    args: vec![Value(3)],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );
    blocks.insert(
        Block(1),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iconst {
                        ty: Type::I64,
                        imm: 1,
                    },
                    args: vec![],
                    results: vec![Value(4)],
                },
                Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(4)],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );
    blocks.insert(
        Block(2),
        BasicBlock {
            params: vec![],
            instructions: vec![
                Instruction {
                    opcode: Opcode::Iconst {
                        ty: Type::I64,
                        imm: 2,
                    },
                    args: vec![],
                    results: vec![Value(5)],
                },
                Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(5)],
                    results: vec![],
                },
            ],
            source_locs: vec![],
        },
    );

    Function {
        name: "pick".to_string(),
        signature: Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        },
        blocks,
        entry_block: Block(0),
        stack_slots: vec![],
        value_types: HashMap::new(),
        pure_callees: HashSet::new(),
    }
}

// ---------------------------------------------------------------------------
// Test 1: add function — compile, link, run
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_add_function() {
    if !is_aarch64() || !has_cc() {
        return;
    }

    let dir = make_test_dir("bridge_add");
    let func = build_bridge_add();
    let obj_bytes = compile_bridge_lir(&func).expect("add compilation should succeed");

    let obj_path = write_object_file(&dir, "add.o", &obj_bytes);
    let driver_path = write_c_driver(
        &dir,
        "driver.c",
        r#"
extern long add(long a, long b);
int main(void) { return (int)add(3, 4); }
"#,
    );

    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_add");
    let (exit_code, _stdout) = run_binary_with_output(&binary);
    assert_eq!(exit_code, 7, "add(3, 4) should return 7");

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 2: const return — compile, link, run
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_const_return() {
    if !is_aarch64() || !has_cc() {
        return;
    }

    let dir = make_test_dir("bridge_const");
    let func = build_bridge_const42();
    let obj_bytes = compile_bridge_lir(&func).expect("const42 compilation should succeed");

    let obj_path = write_object_file(&dir, "const42.o", &obj_bytes);
    let driver_path = write_c_driver(
        &dir,
        "driver.c",
        r#"
extern long const42(void);
int main(void) { return (int)const42(); }
"#,
    );

    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_const");
    let (exit_code, _stdout) = run_binary_with_output(&binary);
    assert_eq!(exit_code, 42, "const42() should return 42");

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 3: midpoint — multi-block, compile, link, run
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_midpoint() {
    if !is_aarch64() || !has_cc() {
        return;
    }

    let dir = make_test_dir("bridge_midpoint");
    let func = build_bridge_midpoint();
    let obj_bytes = compile_bridge_lir(&func).expect("midpoint compilation should succeed");

    let obj_path = write_object_file(&dir, "midpoint.o", &obj_bytes);
    let driver_path = write_c_driver(
        &dir,
        "driver.c",
        r#"
extern long midpoint(long a, long b);
int main(void) { return (int)midpoint(10, 20); }
"#,
    );

    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_midpoint");
    let (exit_code, _stdout) = run_binary_with_output(&binary);
    assert_eq!(exit_code, 15, "midpoint(10, 20) should return 15");

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 4: branch — conditional, compile, link, run
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_branch() {
    if !is_aarch64() || !has_cc() {
        return;
    }

    let dir = make_test_dir("bridge_branch");
    let func = build_bridge_pick();
    let obj_bytes = compile_bridge_lir(&func).expect("pick compilation should succeed");

    let obj_path = write_object_file(&dir, "pick.o", &obj_bytes);

    // Test 1: pick(0) should return 1
    let driver1_path = write_c_driver(
        &dir,
        "driver_zero.c",
        r#"
extern long pick(long x);
int main(void) { return (int)pick(0); }
"#,
    );
    let binary1 = link_with_cc(&dir, &driver1_path, &obj_path, "test_pick_zero");
    let (exit_code1, _) = run_binary_with_output(&binary1);
    assert_eq!(exit_code1, 1, "pick(0) should return 1");

    // Test 2: pick(5) should return 2
    let driver2_path = write_c_driver(
        &dir,
        "driver_nonzero.c",
        r#"
extern long pick(long x);
int main(void) { return (int)pick(5); }
"#,
    );
    let binary2 = link_with_cc(&dir, &driver2_path, &obj_path, "test_pick_nonzero");
    let (exit_code2, _) = run_binary_with_output(&binary2);
    assert_eq!(exit_code2, 2, "pick(5) should return 2");

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 5: LIR structure validation (no link required, runs on all platforms)
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_lir_structure_validation() {
    // Validate the add function structure
    let add = build_bridge_add();
    assert_eq!(add.name, "add");
    assert_eq!(add.signature.params, vec![Type::I64, Type::I64]);
    assert_eq!(add.signature.returns, vec![Type::I64]);
    assert_eq!(add.blocks.len(), 1);
    assert_eq!(add.entry_block, Block(0));
    assert!(add.stack_slots.is_empty());
    let bb0 = &add.blocks[&Block(0)];
    assert!(bb0.params.is_empty(), "bridge always uses empty block params");
    assert_eq!(bb0.instructions.len(), 2);
    assert!(matches!(bb0.instructions[0].opcode, Opcode::Iadd));
    assert_eq!(bb0.instructions[0].args, vec![Value(0), Value(1)]);
    assert_eq!(bb0.instructions[0].results, vec![Value(2)]);
    assert!(matches!(bb0.instructions[1].opcode, Opcode::Return));
    assert_eq!(
        bb0.instructions[1].args,
        vec![Value(2)],
        "Return must include the return value in args for ISel"
    );

    // Validate the const42 function structure
    let c42 = build_bridge_const42();
    assert_eq!(c42.name, "const42");
    assert!(c42.signature.params.is_empty());
    assert_eq!(c42.signature.returns, vec![Type::I64]);
    let bb0 = &c42.blocks[&Block(0)];
    assert!(matches!(
        bb0.instructions[0].opcode,
        Opcode::Iconst {
            ty: Type::I64,
            imm: 42
        }
    ));

    // Validate the midpoint function structure (multi-block)
    let mid = build_bridge_midpoint();
    assert_eq!(mid.name, "midpoint");
    assert_eq!(mid.blocks.len(), 2);
    assert!(mid.blocks.contains_key(&Block(0)));
    assert!(mid.blocks.contains_key(&Block(1)));
    // Block 0: iadd + jump
    let bb0 = &mid.blocks[&Block(0)];
    assert!(bb0
        .instructions
        .iter()
        .any(|i| matches!(i.opcode, Opcode::Iadd)));
    assert!(matches!(
        bb0.instructions.last().unwrap().opcode,
        Opcode::Jump { dest: Block(1) }
    ));
    // Block 1: iconst + udiv + return
    let bb1 = &mid.blocks[&Block(1)];
    assert!(bb1
        .instructions
        .iter()
        .any(|i| matches!(i.opcode, Opcode::Udiv)));
    assert!(matches!(
        bb1.instructions.last().unwrap().opcode,
        Opcode::Return
    ));

    // Validate the pick function structure (branching)
    let pick = build_bridge_pick();
    assert_eq!(pick.name, "pick");
    assert_eq!(pick.blocks.len(), 3);
    let bb0 = &pick.blocks[&Block(0)];
    // Should have: Iconst(0), Icmp(Equal), Brif
    assert!(bb0
        .instructions
        .iter()
        .any(|i| matches!(i.opcode, Opcode::Icmp { cond: IntCC::Equal })));
    assert!(bb0.instructions.iter().any(|i| matches!(
        i.opcode,
        Opcode::Brif {
            then_dest: Block(1),
            else_dest: Block(2),
            ..
        }
    )));
    // Both target blocks should end with Return
    assert!(matches!(
        pick.blocks[&Block(1)]
            .instructions
            .last()
            .unwrap()
            .opcode,
        Opcode::Return
    ));
    assert!(matches!(
        pick.blocks[&Block(2)]
            .instructions
            .last()
            .unwrap()
            .opcode,
        Opcode::Return
    ));
}

// ---------------------------------------------------------------------------
// Test 6: Pipeline compilation succeeds for all bridge functions
// ---------------------------------------------------------------------------

/// Verify that all bridge-format LIR functions compile without error.
/// This is a cross-platform test (does not require AArch64 or cc).
#[test]
fn test_bridge_all_functions_compile() {
    // The AArch64 pipeline is always used for compile_function.
    // On non-AArch64 hosts this still tests ISel, optimization, and regalloc.
    // Encoding may produce AArch64 code but we only check it doesn't panic.
    let functions = vec![
        build_bridge_add(),
        build_bridge_const42(),
        build_bridge_midpoint(),
        build_bridge_pick(),
    ];

    for func in &functions {
        let result = compile_bridge_lir(func);
        assert!(
            result.is_ok(),
            "bridge function '{}' failed to compile: {}",
            func.name,
            result.unwrap_err()
        );

        let obj_bytes = result.unwrap();
        // Verify Mach-O magic number (0xFEEDFACF for 64-bit)
        assert!(
            obj_bytes.len() >= 4,
            "object file for '{}' too small: {} bytes",
            func.name,
            obj_bytes.len()
        );
        assert_eq!(
            &obj_bytes[0..4],
            &[0xCF, 0xFA, 0xED, 0xFE],
            "invalid Mach-O magic for '{}'",
            func.name
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Validate bridge produces correct Mach-O object file type
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_macho_filetype() {
    let func = build_bridge_add();
    let obj_bytes = compile_bridge_lir(&func).expect("compilation should succeed");

    // Mach-O header: filetype at offset 12, should be MH_OBJECT (1)
    let filetype = u32::from_le_bytes([
        obj_bytes[12],
        obj_bytes[13],
        obj_bytes[14],
        obj_bytes[15],
    ]);
    assert_eq!(filetype, 1, "Mach-O filetype should be MH_OBJECT (1)");
}

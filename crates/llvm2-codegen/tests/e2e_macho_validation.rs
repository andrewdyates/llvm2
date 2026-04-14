// llvm2-codegen/tests/e2e_macho_validation.rs - End-to-end Mach-O validation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests exercise the COMPLETE compilation pipeline starting from tMIR,
// through ISel, optimization, register allocation, frame lowering, encoding,
// and Mach-O emission. They then validate the resulting .o file using:
//
// 1. In-process binary parsing (Mach-O header, load commands, sections)
// 2. otool -l (load command structure)
// 3. otool -tv (disassembly — verifies expected AArch64 instructions)
// 4. nm (symbol table)
//
// This is the ultimate correctness test for the compiler backend: the output
// must be a valid Mach-O object file that macOS tooling accepts.
//
// Part of #91 -- End-to-end Mach-O validation test suite

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::macho::constants::*;
use llvm2_codegen::pipeline::{Pipeline, PipelineConfig, OptLevel};

use tmir_func::{Block as TmirBlock, Function as TmirFunction};
use tmir_instrs::{BinOp, CmpOp, Instr, InstrNode};
use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_macho_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write .o file");
    path
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Mach-O binary parsing helpers
// ---------------------------------------------------------------------------

/// Verify the Mach-O magic number is correct (64-bit little-endian).
fn verify_macho_magic(bytes: &[u8]) {
    assert!(bytes.len() >= 4, "Object file too small: {} bytes", bytes.len());
    assert_eq!(
        &bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "Invalid Mach-O magic: expected 0xFEEDFACF (LE), got [{:#04X}, {:#04X}, {:#04X}, {:#04X}]",
        bytes[0], bytes[1], bytes[2], bytes[3]
    );
}

/// Extract the filetype field from the Mach-O header.
fn macho_filetype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]])
}

/// Extract the cputype field from the Mach-O header.
fn macho_cputype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

/// Extract ncmds (number of load commands) from the Mach-O header.
fn macho_ncmds(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]])
}

/// Extract flags from the Mach-O header.
fn macho_flags(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]])
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

/// Extract the segment name for a section by searching Mach-O load commands.
/// Returns the segment name string (e.g. "__TEXT") if the section is found.
fn find_section_segment(bytes: &[u8], target_sectname: &[u8]) -> Option<String> {
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
                    // segname is at offset 16..32 in the section header
                    let segname = &bytes[sec_offset + 16..sec_offset + 32];
                    let seg_len = segname.iter().position(|&b| b == 0).unwrap_or(16);
                    return Some(String::from_utf8_lossy(&segname[..seg_len]).to_string());
                }

                sec_offset += SECTION_64_SIZE as usize;
            }
        }

        offset += cmdsize;
    }

    None
}

/// Count the number of load commands of a given type.
fn count_load_commands(bytes: &[u8], target_cmd: u32) -> u32 {
    let sizeofcmds = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    let mut offset = MACH_HEADER_64_SIZE as usize;
    let lc_end = MACH_HEADER_64_SIZE as usize + sizeofcmds as usize;
    let mut count = 0u32;

    while offset < lc_end && offset + 8 <= bytes.len() {
        let cmd = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]);
        let cmdsize = u32::from_le_bytes([bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]]) as usize;

        if cmd == target_cmd {
            count += 1;
        }

        offset += cmdsize;
    }

    count
}

// ---------------------------------------------------------------------------
// External tool runners
// ---------------------------------------------------------------------------

/// Run `otool -l` on an object file and return stdout.
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

/// Run `otool -tv` on an object file and return stdout (disassembly).
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

/// Run `otool -h` on an object file and return stdout (header).
fn run_otool_h(path: &Path) -> Option<String> {
    let output = Command::new("otool")
        .args(["-h", path.to_str().unwrap()])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

/// Run `nm` on an object file and return stdout.
fn run_nm(path: &Path) -> Option<String> {
    let output = Command::new("nm")
        .args([path.to_str().unwrap()])
        .output()
        .ok()?;

    // nm may succeed with output or fail (for empty symbol tables)
    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

// ---------------------------------------------------------------------------
// Helper: compile a tMIR function through the full pipeline
// ---------------------------------------------------------------------------

fn compile_tmir_function(
    tmir_func: &TmirFunction,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(tmir_func, &[])
            .map_err(|e| format!("adapter error: {}", e))?;

    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
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

/// Build tMIR for: fn return_const() -> i64 { return 42; }
///
/// Simplest possible tMIR function: one block, const + return.
fn build_return_const_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "return_const".to_string(),
        ty: FuncTy {
            params: vec![],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![],
            body: vec![
                InstrNode {
                    instr: Instr::Const {
                        ty: Ty::Int(64),
                        value: 42,
                    },
                    results: vec![ValueId(0)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(0)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

/// Build tMIR for: fn simple_add(a: i64, b: i64) -> i64 { a + b }
fn build_simple_add_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "simple_add".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Int(64)),
                (ValueId(1), Ty::Int(64)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::Int(64),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(2)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

/// Build tMIR for: fn max_val(a: i64, b: i64) -> i64 { if a > b { a } else { b } }
///
/// Tests conditional branching with two exits:
///   bb0 (entry): cmp a > b, condbr -> bb1 (return a), bb2 (return b)
///   bb1: return a
///   bb2: return b
fn build_max_val_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "max_val".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): compare and branch
            TmirBlock {
                id: BlockId(0),
                params: vec![
                    (ValueId(0), Ty::Int(64)), // a
                    (ValueId(1), Ty::Int(64)), // b
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(64),
                            lhs: ValueId(0), // a
                            rhs: ValueId(1), // b
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(2),
                            then_target: BlockId(1), // return a
                            then_args: vec![],
                            else_target: BlockId(2), // return b
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1: return a
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(0)],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb2: return b
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(1)],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    }
}

/// Build tMIR for: fn simple_sub(a: i64, b: i64) -> i64 { a - b }
fn build_simple_sub_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "simple_sub".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Int(64)),
                (ValueId(1), Ty::Int(64)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::Int(64),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(2)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

/// Build tMIR for a simple loop: fn count_down(n: i64) -> i64
///
/// Counts from n down to 0, summing each value:
///   sum = 0, i = n
///   loop:
///     if i <= 0: return sum
///     sum = sum + i
///     i = i - 1
///     goto loop
fn build_count_down_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "count_down".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): init sum=0, jump to loop
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // n
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(1)], // sum_init = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(1), ValueId(0)], // sum=0, i=n
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (loop header): params(sum, i)
            TmirBlock {
                id: BlockId(1),
                params: vec![
                    (ValueId(10), Ty::Int(64)), // sum
                    (ValueId(11), Ty::Int(64)), // i
                ],
                body: vec![
                    // cmp i <= 0
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(12)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: ValueId(11), // i
                            rhs: ValueId(12), // 0
                        },
                        results: vec![ValueId(13)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(13),
                            then_target: BlockId(2), // return sum
                            then_args: vec![ValueId(10)],
                            else_target: BlockId(3), // loop body
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2 (exit): return sum
            TmirBlock {
                id: BlockId(2),
                params: vec![(ValueId(20), Ty::Int(64))],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(20)],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb3 (loop body): sum += i, i -= 1, loop back
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![
                    // new_sum = sum + i
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: ValueId(10), // sum
                            rhs: ValueId(11), // i
                        },
                        results: vec![ValueId(14)],
                        proofs: vec![],
                    },
                    // new_i = i - 1
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(15)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::Int(64),
                            lhs: ValueId(11), // i
                            rhs: ValueId(15), // 1
                        },
                        results: vec![ValueId(16)],
                        proofs: vec![],
                    },
                    // loop back
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(14), ValueId(16)], // new_sum, new_i
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
        ],
        proofs: vec![],
    }
}

// ===========================================================================
// TEST: return_const -- minimal full-pipeline Mach-O validation
// ===========================================================================

#[test]
fn test_macho_return_const_binary_structure() {
    let tmir_func = build_return_const_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile return_const");

    // --- In-process Mach-O header validation ---
    verify_macho_magic(&obj_bytes);
    assert_eq!(
        macho_filetype(&obj_bytes),
        MH_OBJECT,
        "filetype should be MH_OBJECT (1)"
    );
    assert_eq!(
        macho_cputype(&obj_bytes),
        CPU_TYPE_ARM64,
        "cputype should be ARM64 (0x{:08X})",
        CPU_TYPE_ARM64,
    );
    assert_ne!(
        macho_ncmds(&obj_bytes),
        0,
        "load command count should be non-zero"
    );
    assert_eq!(
        macho_flags(&obj_bytes) & MH_SUBSECTIONS_VIA_SYMBOLS,
        MH_SUBSECTIONS_VIA_SYMBOLS,
        "MH_SUBSECTIONS_VIA_SYMBOLS flag should be set"
    );

    // --- Required load commands ---
    assert!(
        count_load_commands(&obj_bytes, LC_SEGMENT_64) >= 1,
        "must have at least one LC_SEGMENT_64"
    );
    assert_eq!(
        count_load_commands(&obj_bytes, LC_SYMTAB),
        1,
        "must have exactly one LC_SYMTAB"
    );
    assert_eq!(
        count_load_commands(&obj_bytes, LC_DYSYMTAB),
        1,
        "must have exactly one LC_DYSYMTAB"
    );
    assert_eq!(
        count_load_commands(&obj_bytes, LC_BUILD_VERSION),
        1,
        "must have exactly one LC_BUILD_VERSION"
    );

    // --- Section validation ---
    let text_section = find_section(&obj_bytes, b"__text");
    assert!(text_section.is_some(), "__text section must be present");

    let (text_offset, text_size) = text_section.unwrap();
    assert!(text_offset > 0, "__text section offset must be non-zero");
    assert!(text_size > 0, "__text section must have non-zero size");
    // Each AArch64 instruction is 4 bytes, so text size must be a multiple of 4.
    assert_eq!(
        text_size % 4,
        0,
        "__text section size ({}) must be a multiple of 4 (AArch64 instruction size)",
        text_size
    );

    // __text should be in __TEXT segment
    let text_seg = find_section_segment(&obj_bytes, b"__text");
    assert_eq!(
        text_seg.as_deref(),
        Some("__TEXT"),
        "__text section should be in __TEXT segment"
    );

    // --- Compact unwind section ---
    let cu_section = find_section(&obj_bytes, b"__compact_unwind");
    assert!(
        cu_section.is_some(),
        "__compact_unwind section must be present"
    );
    let (_cu_offset, cu_size) = cu_section.unwrap();
    assert_eq!(
        cu_size, 32,
        "single-function object should have exactly one 32-byte compact unwind entry"
    );

    // __compact_unwind should be in __LD segment
    let cu_seg = find_section_segment(&obj_bytes, b"__compact_unwind");
    assert_eq!(
        cu_seg.as_deref(),
        Some("__LD"),
        "__compact_unwind section should be in __LD segment"
    );
}

#[test]
fn test_macho_return_const_otool_validation() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("return_const_otool");
    let tmir_func = build_return_const_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile return_const");
    let obj_path = write_object_file(&dir, "return_const.o", &obj_bytes);

    // --- otool -h: header ---
    if let Some(header_out) = run_otool_h(&obj_path) {
        // cputype for ARM64 = 0x0100000C = 16777228.
        // Different otool versions may show "ARM64", hex, or decimal.
        assert!(
            header_out.contains("ARM64")
                || header_out.contains("0x0100000c")
                || header_out.contains("16777228"),
            "otool -h should show ARM64 architecture.\nOutput:\n{}",
            header_out
        );
    }

    // --- otool -l: load commands ---
    if let Some(lc_out) = run_otool_l(&obj_path) {
        assert!(
            lc_out.contains("LC_SEGMENT_64"),
            "otool -l should show LC_SEGMENT_64.\nOutput:\n{}",
            lc_out
        );
        assert!(
            lc_out.contains("LC_SYMTAB"),
            "otool -l should show LC_SYMTAB.\nOutput:\n{}",
            lc_out
        );
        assert!(
            lc_out.contains("LC_DYSYMTAB"),
            "otool -l should show LC_DYSYMTAB.\nOutput:\n{}",
            lc_out
        );
        assert!(
            lc_out.contains("LC_BUILD_VERSION"),
            "otool -l should show LC_BUILD_VERSION.\nOutput:\n{}",
            lc_out
        );
        assert!(
            lc_out.contains("__text"),
            "otool -l should show __text section.\nOutput:\n{}",
            lc_out
        );
        assert!(
            lc_out.contains("__compact_unwind"),
            "otool -l should show __compact_unwind section.\nOutput:\n{}",
            lc_out
        );
        // Section flags for __text: PURE_INSTRUCTIONS | SOME_INSTRUCTIONS
        assert!(
            lc_out.contains("0x80000400"),
            "otool -l should show __text section flags (S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS).\nOutput:\n{}",
            lc_out
        );
    }

    // --- otool -tv: disassembly ---
    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("return_const disassembly:\n{}", disasm);
        // return_const should contain a RET instruction
        assert!(
            disasm.contains("ret"),
            "Disassembly should contain 'ret' instruction.\nOutput:\n{}",
            disasm
        );
        // Should also reference the function name
        assert!(
            disasm.contains("return_const"),
            "Disassembly should reference function name 'return_const'.\nOutput:\n{}",
            disasm
        );
    }

    // --- nm: symbol table ---
    if let Some(nm_out) = run_nm(&obj_path) {
        assert!(
            nm_out.contains("_return_const"),
            "nm should show _return_const symbol.\nOutput:\n{}",
            nm_out
        );
        assert!(
            nm_out.contains(" T ") || nm_out.contains(" t "),
            "_return_const should be a text section symbol (T or t).\nOutput:\n{}",
            nm_out
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: simple_add -- validates ADD instruction in disassembly
// ===========================================================================

#[test]
fn test_macho_simple_add_binary_structure() {
    let tmir_func = build_simple_add_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");

    verify_macho_magic(&obj_bytes);
    assert_eq!(macho_filetype(&obj_bytes), MH_OBJECT);
    assert_eq!(macho_cputype(&obj_bytes), CPU_TYPE_ARM64);

    // Text section should exist and be non-empty
    let text = find_section(&obj_bytes, b"__text");
    assert!(text.is_some(), "__text section must exist");
    let (_, text_size) = text.unwrap();
    assert!(text_size >= 8, "add function needs at least 2 instructions (8 bytes), got {}", text_size);
}

#[test]
fn test_macho_simple_add_disassembly() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("simple_add_disasm");
    let tmir_func = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");
    let obj_path = write_object_file(&dir, "simple_add.o", &obj_bytes);

    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("simple_add disassembly:\n{}", disasm);

        // Should contain an ADD instruction
        assert!(
            disasm.contains("add") || disasm.contains("ADD"),
            "simple_add disassembly should contain 'add' instruction.\nOutput:\n{}",
            disasm
        );
        // Should contain a RET instruction
        assert!(
            disasm.contains("ret"),
            "simple_add disassembly should contain 'ret' instruction.\nOutput:\n{}",
            disasm
        );
        // Function name should appear
        assert!(
            disasm.contains("simple_add"),
            "Disassembly should reference 'simple_add'.\nOutput:\n{}",
            disasm
        );
    }

    if let Some(nm_out) = run_nm(&obj_path) {
        assert!(
            nm_out.contains("_simple_add"),
            "nm should show _simple_add symbol.\nOutput:\n{}",
            nm_out
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: simple_sub -- validates SUB instruction in disassembly
// ===========================================================================

#[test]
fn test_macho_simple_sub_disassembly() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("simple_sub_disasm");
    let tmir_func = build_simple_sub_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_sub");
    let obj_path = write_object_file(&dir, "simple_sub.o", &obj_bytes);

    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("simple_sub disassembly:\n{}", disasm);

        assert!(
            disasm.contains("sub") || disasm.contains("SUB"),
            "simple_sub disassembly should contain 'sub' instruction.\nOutput:\n{}",
            disasm
        );
        assert!(
            disasm.contains("ret"),
            "simple_sub disassembly should contain 'ret' instruction.\nOutput:\n{}",
            disasm
        );
    }

    if let Some(nm_out) = run_nm(&obj_path) {
        assert!(
            nm_out.contains("_simple_sub"),
            "nm should show _simple_sub symbol.\nOutput:\n{}",
            nm_out
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: max_val -- branch (if/else) with CMP + conditional branch
// ===========================================================================

#[test]
fn test_macho_max_val_binary_structure() {
    let tmir_func = build_max_val_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile max_val");

    verify_macho_magic(&obj_bytes);
    assert_eq!(macho_filetype(&obj_bytes), MH_OBJECT);

    let text = find_section(&obj_bytes, b"__text");
    assert!(text.is_some(), "__text section must exist for max_val");
    let (_, text_size) = text.unwrap();
    // max_val has a branch, so it should be larger than a straight-line function.
    // At minimum: prologue + cmp + b.cond + ret + ret + epilogue = many instructions.
    assert!(
        text_size >= 12,
        "max_val with branch should be at least 12 bytes, got {}",
        text_size
    );
}

#[test]
fn test_macho_max_val_disassembly() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("max_val_disasm");
    let tmir_func = build_max_val_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile max_val");
    let obj_path = write_object_file(&dir, "max_val.o", &obj_bytes);

    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("max_val disassembly:\n{}", disasm);

        // Should contain a compare instruction
        assert!(
            disasm.contains("cmp") || disasm.contains("CMP")
                || disasm.contains("subs") || disasm.contains("SUBS"),
            "max_val disassembly should contain a compare instruction (cmp/subs).\nOutput:\n{}",
            disasm
        );
        // Should contain a conditional branch
        assert!(
            disasm.contains("b.") || disasm.contains("B."),
            "max_val disassembly should contain a conditional branch (b.xx).\nOutput:\n{}",
            disasm
        );
        // Should contain RET
        assert!(
            disasm.contains("ret"),
            "max_val disassembly should contain 'ret' instruction.\nOutput:\n{}",
            disasm
        );
    }

    if let Some(nm_out) = run_nm(&obj_path) {
        assert!(
            nm_out.contains("_max_val"),
            "nm should show _max_val symbol.\nOutput:\n{}",
            nm_out
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: count_down -- loop with backward branch
// ===========================================================================

#[test]
fn test_macho_count_down_binary_structure() {
    let tmir_func = build_count_down_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile count_down");

    verify_macho_magic(&obj_bytes);
    assert_eq!(macho_filetype(&obj_bytes), MH_OBJECT);

    let text = find_section(&obj_bytes, b"__text");
    assert!(text.is_some(), "__text section must exist for count_down");
    let (_, text_size) = text.unwrap();
    // Loop function with compare, conditional branch, add, sub, unconditional branch.
    assert!(
        text_size >= 16,
        "count_down with loop should be at least 16 bytes, got {}",
        text_size
    );
}

#[test]
fn test_macho_count_down_disassembly() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("count_down_disasm");
    let tmir_func = build_count_down_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile count_down");
    let obj_path = write_object_file(&dir, "count_down.o", &obj_bytes);

    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("count_down disassembly:\n{}", disasm);

        // Loop must contain an unconditional branch (backward jump) or conditional branch
        let has_branch = disasm.contains("b ") || disasm.contains("b\t")
            || disasm.contains("b.") || disasm.contains("B.");
        assert!(
            has_branch,
            "count_down disassembly should contain a branch instruction.\nOutput:\n{}",
            disasm
        );
        assert!(
            disasm.contains("add") || disasm.contains("ADD"),
            "count_down disassembly should contain 'add' instruction.\nOutput:\n{}",
            disasm
        );
        assert!(
            disasm.contains("sub") || disasm.contains("SUB")
                || disasm.contains("subs") || disasm.contains("SUBS"),
            "count_down disassembly should contain 'sub' instruction.\nOutput:\n{}",
            disasm
        );
        assert!(
            disasm.contains("ret"),
            "count_down disassembly should contain 'ret' instruction.\nOutput:\n{}",
            disasm
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: optimization levels -- same function at O0, O1, O2 all produce valid Mach-O
// ===========================================================================

#[test]
fn test_macho_opt_levels_produce_valid_objects() {
    let tmir_func = build_simple_add_tmir();

    for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        let obj_bytes = compile_tmir_function(&tmir_func, *opt_level)
            .unwrap_or_else(|e| panic!("pipeline at {:?} failed: {}", opt_level, e));

        verify_macho_magic(&obj_bytes);
        assert_eq!(
            macho_filetype(&obj_bytes),
            MH_OBJECT,
            "filetype should be MH_OBJECT at {:?}",
            opt_level
        );
        assert_eq!(
            macho_cputype(&obj_bytes),
            CPU_TYPE_ARM64,
            "cputype should be ARM64 at {:?}",
            opt_level
        );

        let text = find_section(&obj_bytes, b"__text");
        assert!(
            text.is_some(),
            "__text section must exist at {:?}",
            opt_level
        );
        let (_, text_size) = text.unwrap();
        assert!(
            text_size > 0,
            "__text section must have non-zero size at {:?}",
            opt_level
        );
    }
}

#[test]
fn test_macho_opt_levels_otool_validates_all() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("opt_levels_otool");
    let tmir_func = build_simple_add_tmir();

    for (i, opt_level) in [OptLevel::O0, OptLevel::O1, OptLevel::O2].iter().enumerate() {
        let obj_bytes = compile_tmir_function(&tmir_func, *opt_level)
            .unwrap_or_else(|e| panic!("pipeline at {:?} failed: {}", opt_level, e));

        let filename = format!("add_O{}.o", i);
        let obj_path = write_object_file(&dir, &filename, &obj_bytes);

        // otool -l should succeed
        if let Some(lc_out) = run_otool_l(&obj_path) {
            assert!(
                lc_out.contains("LC_SEGMENT_64"),
                "otool -l at {:?} should show LC_SEGMENT_64",
                opt_level
            );
        }

        // otool -tv should succeed and show ret
        if let Some(disasm) = run_otool_tv(&obj_path) {
            assert!(
                disasm.contains("ret"),
                "Disassembly at {:?} should contain 'ret'.\nOutput:\n{}",
                opt_level,
                disasm
            );
        }
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: multiple functions -- each produces independently valid Mach-O
// ===========================================================================

#[test]
fn test_macho_multiple_functions_all_valid() {
    let functions: Vec<(&str, TmirFunction)> = vec![
        ("return_const", build_return_const_tmir()),
        ("simple_add", build_simple_add_tmir()),
        ("simple_sub", build_simple_sub_tmir()),
        ("max_val", build_max_val_tmir()),
        ("count_down", build_count_down_tmir()),
    ];

    for (name, tmir_func) in &functions {
        let obj_bytes = compile_tmir_function(tmir_func, OptLevel::O0)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        verify_macho_magic(&obj_bytes);
        assert_eq!(
            macho_filetype(&obj_bytes),
            MH_OBJECT,
            "{}: filetype should be MH_OBJECT",
            name
        );

        let text = find_section(&obj_bytes, b"__text");
        assert!(
            text.is_some(),
            "{}: __text section must exist",
            name
        );

        let cu = find_section(&obj_bytes, b"__compact_unwind");
        assert!(
            cu.is_some(),
            "{}: __compact_unwind section must exist",
            name
        );
    }
}

#[test]
fn test_macho_multiple_functions_otool_all_valid() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("multi_func_otool");

    let functions: Vec<(&str, TmirFunction)> = vec![
        ("return_const", build_return_const_tmir()),
        ("simple_add", build_simple_add_tmir()),
        ("simple_sub", build_simple_sub_tmir()),
        ("max_val", build_max_val_tmir()),
        ("count_down", build_count_down_tmir()),
    ];

    for (name, tmir_func) in &functions {
        let obj_bytes = compile_tmir_function(tmir_func, OptLevel::O0)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        let filename = format!("{}.o", name);
        let obj_path = write_object_file(&dir, &filename, &obj_bytes);

        // otool -l must succeed
        let lc_out = run_otool_l(&obj_path);
        assert!(
            lc_out.is_some(),
            "otool -l should succeed for {}",
            name
        );

        // otool -tv must succeed
        let disasm = run_otool_tv(&obj_path);
        assert!(
            disasm.is_some(),
            "otool -tv should succeed for {}",
            name
        );

        // nm must show the function symbol
        if let Some(nm_out) = run_nm(&obj_path) {
            let symbol = format!("_{}", name);
            assert!(
                nm_out.contains(&symbol),
                "nm should show {} symbol for {}.\nOutput:\n{}",
                symbol,
                name,
                nm_out
            );
        }
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: text section code bytes match expected instruction encoding
// ===========================================================================

#[test]
fn test_macho_return_const_code_bytes() {
    let tmir_func = build_return_const_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile return_const");

    let (text_offset, text_size) = find_section(&obj_bytes, b"__text")
        .expect("__text section must exist");

    let code = &obj_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];

    // Verify each instruction is a valid 4-byte AArch64 word.
    assert_eq!(code.len() % 4, 0, "Code must be 4-byte aligned");

    // The last non-prologue/epilogue instruction before epilogue should be RET (0xD65F03C0).
    // Find RET in the code bytes.
    let mut found_ret = false;
    for i in (0..code.len()).step_by(4) {
        let word = u32::from_le_bytes([code[i], code[i+1], code[i+2], code[i+3]]);
        if word == 0xD65F03C0 {
            found_ret = true;
        }
    }
    assert!(
        found_ret,
        "Code should contain RET instruction (0xD65F03C0). Code bytes: {:02X?}",
        code
    );
}

#[test]
fn test_macho_simple_add_code_bytes() {
    let tmir_func = build_simple_add_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");

    let (text_offset, text_size) = find_section(&obj_bytes, b"__text")
        .expect("__text section must exist");

    let code = &obj_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];

    // Find ADD instruction in the code bytes.
    // ADD Xd, Xn, Xm = 0x8B_xx_xx_xx (sf=1, op=0, S=0, shift=00)
    let mut found_add = false;
    let mut found_ret = false;
    for i in (0..code.len()).step_by(4) {
        let word = u32::from_le_bytes([code[i], code[i+1], code[i+2], code[i+3]]);
        // Check for ADD (register): bits [31:21] = 10001011000
        if (word >> 21) == 0b10001011000 {
            found_add = true;
        }
        if word == 0xD65F03C0 {
            found_ret = true;
        }
    }
    assert!(
        found_add,
        "simple_add code should contain an ADD (register) instruction. Code: {:02X?}",
        code
    );
    assert!(
        found_ret,
        "simple_add code should contain a RET instruction. Code: {:02X?}",
        code
    );
}

// ===========================================================================
// TEST: compact unwind entry contents
// ===========================================================================

#[test]
fn test_macho_compact_unwind_entry_valid() {
    let tmir_func = build_simple_add_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");

    let (cu_offset, cu_size) = find_section(&obj_bytes, b"__compact_unwind")
        .expect("__compact_unwind section must exist");

    assert_eq!(cu_size, 32, "Compact unwind entry should be 32 bytes");

    let cu_data = &obj_bytes[cu_offset as usize..(cu_offset as usize + cu_size as usize)];

    // function_length at bytes 8..12 should be non-zero
    let func_len = u32::from_le_bytes(cu_data[8..12].try_into().unwrap());
    assert!(
        func_len > 0,
        "Function length in compact unwind should be non-zero"
    );

    // compact_encoding at bytes 12..16
    let encoding = u32::from_le_bytes(cu_data[12..16].try_into().unwrap());
    // Mode should be FRAME (0x04xxxxxx) for functions with frame lowering
    let mode = encoding & 0x0F00_0000;
    assert_eq!(
        mode, 0x0400_0000,
        "Compact unwind encoding mode should be FRAME (0x04), got 0x{:08X}",
        encoding
    );

    // personality (8 bytes) and lsda (8 bytes) should be zero (no exception handling)
    let personality = u64::from_le_bytes(cu_data[16..24].try_into().unwrap());
    let lsda = u64::from_le_bytes(cu_data[24..32].try_into().unwrap());
    assert_eq!(personality, 0, "Personality should be 0 for non-exception code");
    assert_eq!(lsda, 0, "LSDA should be 0 for non-exception code");
}

// ===========================================================================
// TEST: build version platform
// ===========================================================================

#[test]
fn test_macho_build_version_is_macos() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("build_version");
    let tmir_func = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");
    let obj_path = write_object_file(&dir, "add.o", &obj_bytes);

    if let Some(lc_out) = run_otool_l(&obj_path) {
        assert!(
            lc_out.contains("platform 1") || lc_out.contains("MACOS"),
            "Build version should target macOS (platform 1).\nOutput:\n{}",
            lc_out
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: prologue/epilogue in disassembly
// ===========================================================================

#[test]
fn test_macho_frame_lowering_in_disassembly() {
    if !is_aarch64() || !has_otool() {
        eprintln!("Skipping: not AArch64 or otool not available");
        return;
    }

    let dir = make_test_dir("frame_lowering_disasm");
    let tmir_func = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");
    let obj_path = write_object_file(&dir, "add_framed.o", &obj_bytes);

    if let Some(disasm) = run_otool_tv(&obj_path) {
        eprintln!("Frame lowering disassembly:\n{}", disasm);

        // Frame lowering produces STP (save frame pointer + link register) in prologue
        let has_stp = disasm.contains("stp") || disasm.contains("STP");
        // And LDP (restore) in epilogue
        let has_ldp = disasm.contains("ldp") || disasm.contains("LDP");

        // Functions going through full pipeline get frame lowering (prologue/epilogue).
        // Both STP and LDP should be present.
        assert!(
            has_stp,
            "Full pipeline function should have STP in prologue.\nOutput:\n{}",
            disasm
        );
        assert!(
            has_ldp,
            "Full pipeline function should have LDP in epilogue.\nOutput:\n{}",
            disasm
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: symbol attributes
// ===========================================================================

#[test]
fn test_macho_symbol_is_global_text() {
    if !is_aarch64() || !has_nm() {
        eprintln!("Skipping: not AArch64 or nm not available");
        return;
    }

    let dir = make_test_dir("symbol_attrs");
    let tmir_func = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");
    let obj_path = write_object_file(&dir, "add.o", &obj_bytes);

    if let Some(nm_out) = run_nm(&obj_path) {
        // Global text symbol shows as "T" (uppercase = global, text section)
        assert!(
            nm_out.contains(" T _simple_add"),
            "simple_add should be a global text symbol (T).\nnm output:\n{}",
            nm_out
        );
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST: object file sizes are reasonable
// ===========================================================================

#[test]
fn test_macho_object_sizes_reasonable() {
    let functions: Vec<(&str, TmirFunction, usize, usize)> = vec![
        // (name, tmir_func, min_bytes, max_bytes)
        ("return_const", build_return_const_tmir(), 100, 10_000),
        ("simple_add", build_simple_add_tmir(), 100, 10_000),
        ("max_val", build_max_val_tmir(), 100, 10_000),
        ("count_down", build_count_down_tmir(), 100, 10_000),
    ];

    for (name, tmir_func, min_size, max_size) in &functions {
        let obj_bytes = compile_tmir_function(tmir_func, OptLevel::O0)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        assert!(
            obj_bytes.len() >= *min_size,
            "{}: object file too small ({} bytes, minimum {})",
            name,
            obj_bytes.len(),
            min_size
        );
        assert!(
            obj_bytes.len() <= *max_size,
            "{}: object file suspiciously large ({} bytes, maximum {})",
            name,
            obj_bytes.len(),
            max_size
        );
    }
}

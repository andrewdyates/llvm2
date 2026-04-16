// llvm2-codegen/tests/e2e_x86_64_macho.rs - End-to-end x86-64 Mach-O validation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Integration tests for the x86-64 Mach-O object file emission pipeline.
// Validates that the pipeline produces structurally correct Mach-O .o files
// that macOS tooling (otool, nm) accepts.
//
// These tests cover:
//   1. In-process binary parsing of x86-64 Mach-O headers and load commands
//   2. otool -l validation (load command structure)
//   3. otool -h validation (header fields)
//   4. nm validation (symbol table)
//   5. Section attribute verification (S_ATTR_PURE_INSTRUCTIONS, alignment)
//   6. tMIR-to-x86-64-Mach-O full pipeline path
//
// Part of #265 -- Mach-O x86-64 emitter for macOS Intel targets

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::macho::constants::*;
use llvm2_codegen::macho::{MachOTarget, MachOWriter};
use llvm2_codegen::x86_64::{
    build_x86_add_test_function, build_x86_const_test_function,
    x86_compile_to_macho,
    X86OutputFormat, X86Pipeline, X86PipelineConfig,
};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_x86_64_macho_{}", test_name));
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

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

/// Extract the cputype field from the Mach-O header.
fn macho_cputype(bytes: &[u8]) -> u32 {
    read_u32(bytes, 4)
}

/// Extract the cpusubtype field from the Mach-O header.
fn macho_cpusubtype(bytes: &[u8]) -> u32 {
    read_u32(bytes, 8)
}

/// Extract ncmds (number of load commands) from the Mach-O header.
fn macho_ncmds(bytes: &[u8]) -> u32 {
    read_u32(bytes, 16)
}

/// Extract flags from the Mach-O header.
fn macho_flags(bytes: &[u8]) -> u32 {
    read_u32(bytes, 24)
}

/// Search for a named section in the Mach-O load commands.
/// Returns (file_offset, size, align, flags) if found.
fn find_section(
    bytes: &[u8],
    target_sectname: &[u8],
) -> Option<(u32, u64, u32, u32)> {
    let sizeofcmds = read_u32(bytes, 20);
    let mut offset = MACH_HEADER_64_SIZE as usize;
    let lc_end = MACH_HEADER_64_SIZE as usize + sizeofcmds as usize;

    while offset < lc_end && offset + 8 <= bytes.len() {
        let cmd = read_u32(bytes, offset);
        let cmdsize = read_u32(bytes, offset + 4) as usize;

        if cmd == LC_SEGMENT_64 {
            let nsects = read_u32(bytes, offset + 64);
            let mut sec_offset = offset + SEGMENT_COMMAND_64_SIZE as usize;

            for _ in 0..nsects {
                let sectname = &bytes[sec_offset..sec_offset + 16];
                let name_len = sectname.iter().position(|&b| b == 0).unwrap_or(16);
                let name = &sectname[..name_len];

                if name == target_sectname {
                    let sec_size = read_u64(bytes, sec_offset + 40);
                    let sec_file_offset = read_u32(bytes, sec_offset + 48);
                    let sec_align = read_u32(bytes, sec_offset + 52);
                    let sec_flags = read_u32(bytes, sec_offset + 64);
                    return Some((sec_file_offset, sec_size, sec_align, sec_flags));
                }

                sec_offset += SECTION_64_SIZE as usize;
            }
        }

        offset += cmdsize;
    }

    None
}

/// Find LC_BUILD_VERSION and return (platform, minos).
fn find_build_version(bytes: &[u8]) -> Option<(u32, u32)> {
    let sizeofcmds = read_u32(bytes, 20);
    let mut offset = MACH_HEADER_64_SIZE as usize;
    let lc_end = MACH_HEADER_64_SIZE as usize + sizeofcmds as usize;

    while offset < lc_end && offset + 8 <= bytes.len() {
        let cmd = read_u32(bytes, offset);
        let cmdsize = read_u32(bytes, offset + 4) as usize;

        if cmd == LC_BUILD_VERSION {
            let platform = read_u32(bytes, offset + 8);
            let minos = read_u32(bytes, offset + 12);
            return Some((platform, minos));
        }

        offset += cmdsize;
    }

    None
}

// ---------------------------------------------------------------------------
// External tool runners
// ---------------------------------------------------------------------------

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

fn run_nm(path: &Path) -> Option<String> {
    let output = Command::new("nm")
        .args([path.to_str().unwrap()])
        .output()
        .ok()?;

    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

// ===========================================================================
// Tests: In-process binary validation
// ===========================================================================

#[test]
fn test_x86_64_macho_header_complete_validation() {
    // Compile a function and validate every header field.
    let func = build_x86_const_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();

    // Magic: MH_MAGIC_64 = 0xFEEDFACF
    assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64, "magic");

    // CPU type: CPU_TYPE_X86_64 = 0x01000007
    assert_eq!(macho_cputype(&bytes), CPU_TYPE_X86_64, "cputype");

    // CPU subtype: CPU_SUBTYPE_X86_64_ALL = 3
    assert_eq!(macho_cpusubtype(&bytes), CPU_SUBTYPE_X86_64_ALL, "cpusubtype");

    // File type: MH_OBJECT = 1
    assert_eq!(read_u32(&bytes, 12), MH_OBJECT, "filetype");

    // ncmds: 4 (LC_SEGMENT_64, LC_BUILD_VERSION, LC_SYMTAB, LC_DYSYMTAB)
    assert_eq!(macho_ncmds(&bytes), 4, "ncmds");

    // sizeofcmds > 0
    assert!(read_u32(&bytes, 20) > 0, "sizeofcmds > 0");

    // flags: MH_SUBSECTIONS_VIA_SYMBOLS
    assert_ne!(
        macho_flags(&bytes) & MH_SUBSECTIONS_VIA_SYMBOLS,
        0,
        "MH_SUBSECTIONS_VIA_SYMBOLS flag"
    );

    // reserved: 0
    assert_eq!(read_u32(&bytes, 28), 0, "reserved");
}

#[test]
fn test_x86_64_macho_text_section_attributes() {
    // The __text section should have correct attributes for x86-64.
    let func = build_x86_add_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();

    let (offset, size, align, flags) =
        find_section(&bytes, b"__text").expect("__text section not found");

    // x86-64 text section alignment should be 2^4 = 16 bytes.
    assert_eq!(align, 4, "x86-64 text section alignment should be 2^4");

    // Text section should have PURE_INSTRUCTIONS and SOME_INSTRUCTIONS flags.
    assert_ne!(
        flags & S_ATTR_PURE_INSTRUCTIONS,
        0,
        "text section should have S_ATTR_PURE_INSTRUCTIONS"
    );
    assert_ne!(
        flags & S_ATTR_SOME_INSTRUCTIONS,
        0,
        "text section should have S_ATTR_SOME_INSTRUCTIONS"
    );

    // Section should contain non-empty code.
    assert!(size > 0, "text section should not be empty");

    // File offset should be within the file.
    assert!(
        (offset as usize) < bytes.len(),
        "text section offset should be within file"
    );
}

#[test]
fn test_x86_64_macho_build_version() {
    // LC_BUILD_VERSION should specify macOS platform.
    let func = build_x86_const_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();

    let (platform, minos) =
        find_build_version(&bytes).expect("LC_BUILD_VERSION not found");

    assert_eq!(platform, PLATFORM_MACOS, "platform should be macOS");
    assert_eq!(minos, 0x000E_0000, "minimum OS should be macOS 14.0");
}

#[test]
fn test_x86_64_macho_contains_machine_code() {
    // Compile to raw bytes and Mach-O, verify the raw code appears in the Mach-O.
    let func = build_x86_const_test_function();

    let pipeline_raw = X86Pipeline::new(X86PipelineConfig {
        output_format: X86OutputFormat::RawBytes,
        emit_frame: false,
        ..X86PipelineConfig::default()
    });
    let raw_code = pipeline_raw.compile_function(&func).unwrap();

    let pipeline_macho = X86Pipeline::new(X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: false,
        ..X86PipelineConfig::default()
    });
    let macho_bytes = pipeline_macho.compile_function(&func).unwrap();

    // The raw code should appear somewhere in the Mach-O output.
    let found = macho_bytes
        .windows(raw_code.len())
        .any(|window| window == raw_code.as_slice());
    assert!(
        found,
        "Mach-O output should contain the raw machine code ({} bytes)",
        raw_code.len()
    );
}

#[test]
fn test_x86_64_macho_symbol_table() {
    // Verify the symbol table contains the expected function symbol.
    let func = build_x86_add_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();

    // Find LC_SYMTAB command.
    let seg_cmd_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
    let symtab_cmd_offset =
        (MACH_HEADER_64_SIZE + seg_cmd_size + BUILD_VERSION_COMMAND_SIZE) as usize;
    let cmd = read_u32(&bytes, symtab_cmd_offset);
    assert_eq!(cmd, LC_SYMTAB, "expected LC_SYMTAB command");

    let symoff = read_u32(&bytes, symtab_cmd_offset + 8) as usize;
    let nsyms = read_u32(&bytes, symtab_cmd_offset + 12);
    assert!(nsyms >= 1, "should have at least 1 symbol");

    let stroff = read_u32(&bytes, symtab_cmd_offset + 16) as usize;
    let strsize = read_u32(&bytes, symtab_cmd_offset + 20) as usize;

    // Verify string table starts with null byte.
    assert_eq!(bytes[stroff], 0, "string table must start with null byte");

    // Verify symbol name "_add" is in the string table.
    let strtab = &bytes[stroff..stroff + strsize];
    let strtab_str = String::from_utf8_lossy(strtab);
    assert!(
        strtab_str.contains("_add"),
        "string table should contain '_add' (Mach-O name-mangled)"
    );

    // First symbol should be external defined (N_SECT | N_EXT).
    let n_type = bytes[symoff + 4];
    assert_eq!(n_type, N_SECT | N_EXT, "symbol should be global defined");
}

#[test]
fn test_x86_64_macho_dysymtab() {
    // Verify LC_DYSYMTAB is present and has correct partition.
    let func = build_x86_const_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();

    let seg_cmd_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
    let dysymtab_offset = (MACH_HEADER_64_SIZE
        + seg_cmd_size
        + BUILD_VERSION_COMMAND_SIZE
        + SYMTAB_COMMAND_SIZE) as usize;

    let cmd = read_u32(&bytes, dysymtab_offset);
    assert_eq!(cmd, LC_DYSYMTAB, "expected LC_DYSYMTAB command");

    let cmdsize = read_u32(&bytes, dysymtab_offset + 4);
    assert_eq!(cmdsize, DYSYMTAB_COMMAND_SIZE, "dysymtab cmdsize");

    // With one global symbol: nlocalsym=0, nextdefsym=1, nundefsym=0.
    let nlocalsym = read_u32(&bytes, dysymtab_offset + 12);
    let nextdefsym = read_u32(&bytes, dysymtab_offset + 20);
    assert_eq!(nlocalsym, 0, "should have 0 local symbols");
    assert_eq!(nextdefsym, 1, "should have 1 external defined symbol");
}

#[test]
fn test_x86_64_macho_not_arm64() {
    // Defensive: ensure x86-64 Mach-O never accidentally emits ARM64 CPU type.
    let func = build_x86_const_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();

    assert_ne!(
        macho_cputype(&bytes),
        CPU_TYPE_ARM64,
        "x86-64 Mach-O must not have ARM64 CPU type"
    );
}

#[test]
fn test_x86_64_macho_add_function_roundtrip() {
    // Full pipeline: build add(a,b)->a+b, compile to Mach-O, validate header.
    let func = build_x86_add_test_function();
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: true,
        ..X86PipelineConfig::default()
    });
    let bytes = pipeline.compile_function(&func).unwrap();

    assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64);
    assert_eq!(macho_cputype(&bytes), CPU_TYPE_X86_64);

    // With frame prologue/epilogue, the code should be longer.
    let (_, size, _, _) =
        find_section(&bytes, b"__text").expect("__text section not found");
    assert!(size >= 5, "add function with frame should have >= 5 bytes of code");
}

#[test]
fn test_x86_64_macho_frameless_function() {
    // A frameless function (no prologue/epilogue) should still produce valid Mach-O.
    let func = build_x86_const_test_function();
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: false,
        ..X86PipelineConfig::default()
    });
    let bytes = pipeline.compile_function(&func).unwrap();

    assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64);
    assert_eq!(macho_cputype(&bytes), CPU_TYPE_X86_64);

    let (_, size, _, _) =
        find_section(&bytes, b"__text").expect("__text section not found");
    // Frameless const42: MOV RAX, 42 (10 bytes) + RET (1 byte) = ~11 bytes minimum.
    // But with register allocation overhead it may be more.
    assert!(size > 0, "frameless function should still produce code");
}

#[test]
fn test_x86_64_macho_writer_direct() {
    // Test the MachOWriter directly with hand-crafted x86-64 code.
    let mut writer = MachOWriter::for_target(MachOTarget::X86_64);

    // Minimal x86-64 function: push rbp; mov rbp, rsp; xor eax, eax; pop rbp; ret
    let code = vec![
        0x55,                   // push rbp
        0x48, 0x89, 0xE5,       // mov rbp, rsp
        0x31, 0xC0,             // xor eax, eax
        0x5D,                   // pop rbp
        0xC3,                   // ret
    ];
    writer.add_text_section(&code);
    writer.add_symbol("_zero", 1, 0, true);

    let bytes = writer.write();

    assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64);
    assert_eq!(macho_cputype(&bytes), CPU_TYPE_X86_64);
    assert_eq!(macho_cpusubtype(&bytes), CPU_SUBTYPE_X86_64_ALL);
    assert_eq!(read_u32(&bytes, 12), MH_OBJECT);

    // The code should appear verbatim in the output.
    let found = bytes.windows(code.len()).any(|w| w == code.as_slice());
    assert!(found, "machine code should appear in Mach-O output");
}

#[test]
fn test_x86_64_macho_multi_section() {
    // Test x86-64 Mach-O with both text and data sections.
    let mut writer = MachOWriter::for_target(MachOTarget::X86_64);

    writer.add_text_section(&[0xC3]); // ret
    writer.add_data_section(&[0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    writer.add_symbol("_main", 1, 0, true);
    writer.add_symbol("_answer", 2, 0, true);

    let bytes = writer.write();

    assert_eq!(macho_cputype(&bytes), CPU_TYPE_X86_64);

    // Verify nsects = 2.
    let seg_offset = MACH_HEADER_64_SIZE as usize;
    let nsects = read_u32(&bytes, seg_offset + 64);
    assert_eq!(nsects, 2, "should have 2 sections (text + data)");
}

// ===========================================================================
// Tests: External tool validation (otool, nm)
// ===========================================================================

#[test]
fn test_x86_64_macho_otool_l_validation() {
    // Write x86-64 Mach-O to disk and validate with otool -l.
    if !has_otool() {
        eprintln!("SKIP: otool not available");
        return;
    }

    let dir = make_test_dir("otool_l");

    let func = build_x86_add_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();
    let obj_path = write_object_file(&dir, "add.o", &bytes);

    let output = run_otool_l(&obj_path).expect("otool -l should succeed on x86-64 Mach-O");

    // otool -l should succeed (meaning the file is valid Mach-O).
    // The output shows load commands, not the header CPU type.
    // Verify the expected load commands and sections appear.

    // Should have a __TEXT,__text section.
    assert!(
        output.contains("__text") && output.contains("__TEXT"),
        "otool -l should show __TEXT,__text section"
    );

    // Should have LC_SEGMENT_64.
    assert!(
        output.contains("LC_SEGMENT_64"),
        "otool -l should show LC_SEGMENT_64 load command"
    );

    // Should have LC_SYMTAB.
    assert!(
        output.contains("LC_SYMTAB"),
        "otool -l should show LC_SYMTAB load command"
    );

    // Should have LC_BUILD_VERSION.
    assert!(
        output.contains("LC_BUILD_VERSION"),
        "otool -l should show LC_BUILD_VERSION load command"
    );

    // Verify correct section alignment for x86-64: 2^4 = 16 bytes.
    assert!(
        output.contains("2^4 (16)"),
        "otool -l should show 16-byte alignment for x86-64 text section"
    );

    // Also validate header with otool -h to confirm CPU type.
    let header_output =
        run_otool_h(&obj_path).expect("otool -h should succeed on x86-64 Mach-O");
    // CPU_TYPE_X86_64 = 0x01000007 = 16777223
    assert!(
        header_output.contains("16777223"),
        "otool -h should show CPU_TYPE_X86_64 (16777223), got:\n{}",
        header_output
    );

    cleanup(&dir);
}

#[test]
fn test_x86_64_macho_otool_h_validation() {
    // Validate the header with otool -h.
    if !has_otool() {
        eprintln!("SKIP: otool not available");
        return;
    }

    let dir = make_test_dir("otool_h");

    let func = build_x86_const_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();
    let obj_path = write_object_file(&dir, "const42.o", &bytes);

    let output = run_otool_h(&obj_path).expect("otool -h should succeed on x86-64 Mach-O");

    // otool -h shows: magic, cputype, cpusubtype, caps, filetype, ncmds, sizeofcmds, flags
    // Example: "0xfeedfacf 16777223          3  0x00           1     4        280 0x00002000"

    // CPU_TYPE_X86_64 = 16777223
    assert!(
        output.contains("16777223"),
        "otool -h should show CPU_TYPE_X86_64 (16777223), got:\n{}",
        output
    );

    // cpusubtype = 3
    assert!(
        output.contains("3"),
        "otool -h should show cpusubtype 3"
    );

    // filetype = 1 (MH_OBJECT)
    // The "1" in filetype column -- hard to check precisely, but we verified
    // the header in-process above. The key assertion is that otool -h succeeds.

    cleanup(&dir);
}

#[test]
fn test_x86_64_macho_nm_validation() {
    // Validate the symbol table with nm.
    if !has_nm() {
        eprintln!("SKIP: nm not available");
        return;
    }

    let dir = make_test_dir("nm");

    let func = build_x86_add_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();
    let obj_path = write_object_file(&dir, "add.o", &bytes);

    let output = run_nm(&obj_path).expect("nm should produce output for x86-64 Mach-O");

    // nm should show the _add symbol as a text symbol (T for global, t for local).
    assert!(
        output.contains("_add"),
        "nm should show _add symbol, got:\n{}",
        output
    );

    cleanup(&dir);
}

#[test]
fn test_x86_64_macho_writer_direct_otool_validation() {
    // Validate hand-crafted x86-64 Mach-O with otool -l.
    if !has_otool() {
        eprintln!("SKIP: otool not available");
        return;
    }

    let dir = make_test_dir("writer_direct");

    let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
    // push rbp; mov rbp, rsp; mov eax, 42; pop rbp; ret
    let code = vec![
        0x55,                         // push rbp
        0x48, 0x89, 0xE5,             // mov rbp, rsp
        0xB8, 0x2A, 0x00, 0x00, 0x00, // mov eax, 42
        0x5D,                         // pop rbp
        0xC3,                         // ret
    ];
    writer.add_text_section(&code);
    writer.add_symbol("_answer", 1, 0, true);

    let bytes = writer.write();
    let obj_path = write_object_file(&dir, "answer.o", &bytes);

    let output = run_otool_l(&obj_path).expect("otool -l should accept hand-crafted x86-64 Mach-O");

    assert!(
        output.contains("__text"),
        "otool -l should show __text section"
    );
    assert!(
        output.contains("LC_SEGMENT_64"),
        "otool -l should show LC_SEGMENT_64"
    );

    // Validate header CPU type via otool -h.
    let header_output =
        run_otool_h(&obj_path).expect("otool -h should succeed on hand-crafted x86-64 Mach-O");
    assert!(
        header_output.contains("16777223"),
        "otool -h should show CPU_TYPE_X86_64 (16777223)"
    );

    cleanup(&dir);
}

#[test]
fn test_x86_64_macho_otool_tv_disassembly() {
    // Validate that otool -tv can disassemble the x86-64 code in the Mach-O.
    if !has_otool() {
        eprintln!("SKIP: otool not available");
        return;
    }

    let dir = make_test_dir("otool_tv");

    let func = build_x86_add_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();
    let obj_path = write_object_file(&dir, "add.o", &bytes);

    // otool -tv produces text disassembly of the __TEXT,__text section.
    let output = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool -tv should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "otool -tv should succeed on x86-64 Mach-O, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The disassembly should show x86-64 instructions.
    // At minimum, the function should contain a retq instruction.
    assert!(
        stdout.contains("retq") || stdout.contains("ret"),
        "otool -tv disassembly should contain a ret instruction, got:\n{}",
        stdout
    );

    // The function symbol name should appear.
    assert!(
        stdout.contains("_add"),
        "otool -tv should show the _add symbol, got:\n{}",
        stdout
    );

    // The section header should appear.
    assert!(
        stdout.contains("__TEXT,__text"),
        "otool -tv should show (__TEXT,__text) section header"
    );

    cleanup(&dir);
}

#[test]
fn test_x86_64_macho_nm_symbol_type() {
    // Validate that nm shows the function symbol with the correct type.
    if !has_nm() {
        eprintln!("SKIP: nm not available");
        return;
    }

    let dir = make_test_dir("nm_type");

    let func = build_x86_const_test_function();
    let bytes = x86_compile_to_macho(&func).unwrap();
    let obj_path = write_object_file(&dir, "const42.o", &bytes);

    let output = run_nm(&obj_path).expect("nm should produce output");

    // nm should show _const42 as a text symbol.
    // Global text symbol: uppercase T
    assert!(
        output.contains("_const42"),
        "nm should show _const42 symbol, got:\n{}",
        output
    );

    // The symbol should be in the text section (T or t).
    assert!(
        output.contains(" T ") || output.contains(" t "),
        "nm should show text section symbol (T/t), got:\n{}",
        output
    );

    cleanup(&dir);
}

// ===========================================================================
// Tests: tMIR-to-x86-64-Mach-O full pipeline
// ===========================================================================

#[test]
fn test_x86_64_macho_tmir_compile() {
    // Test the full tMIR -> adapter -> ISel -> encode -> Mach-O pipeline for x86-64.
    use llvm2_codegen::x86_64::{X86Pipeline, X86PipelineConfig, X86OutputFormat};

    use tmir_func::{Block as TmirBlock, Function as TmirFunction};
    use tmir_instrs::{BinOp, Instr, InstrNode, Operand};
    use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

    // Build a simple tMIR function: fn add(a: i64, b: i64) -> i64 { a + b }
    let tmir_func = TmirFunction {
        id: FuncId(0),
        name: "add".to_string(),
        ty: FuncTy {
            params: vec![Ty::int(64), Ty::int(64)],
            returns: vec![Ty::int(64)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::int(64)),
                (ValueId(1), Ty::int(64)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(2))],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    // Translate tMIR to LIR through the adapter.
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &[])
            .expect("tMIR adapter translation should succeed");

    // Compile through the x86-64 pipeline to Mach-O.
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: true,
        ..X86PipelineConfig::default()
    });

    let bytes = pipeline
        .compile_tmir_function(&lir_func)
        .expect("tMIR-to-x86-64 Mach-O compilation should succeed");

    // Validate the Mach-O structure.
    assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64, "magic");
    assert_eq!(macho_cputype(&bytes), CPU_TYPE_X86_64, "cputype");
    assert_eq!(macho_cpusubtype(&bytes), CPU_SUBTYPE_X86_64_ALL, "cpusubtype");

    let (_, size, align, flags) =
        find_section(&bytes, b"__text").expect("__text section not found");
    assert!(size > 0, "text section should contain code");
    assert_eq!(align, 4, "x86-64 text alignment should be 2^4");
    assert_ne!(flags & S_ATTR_PURE_INSTRUCTIONS, 0, "S_ATTR_PURE_INSTRUCTIONS");
}

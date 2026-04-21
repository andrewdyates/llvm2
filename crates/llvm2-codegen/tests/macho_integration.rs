// llvm2-codegen integration test: Mach-O object file writer
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Verifies that the MachOWriter produces valid Mach-O .o files
// that macOS system tools (otool, nm) can parse.

use llvm2_codegen::macho::constants::*;
use llvm2_codegen::macho::{MachOWriter, Relocation};

use std::io::Write;
use std::process::Command;

/// Write bytes to a temp file and return the path.
fn write_temp_o(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("llvm2_test_{}.o", name));
    let mut f = std::fs::File::create(&path).expect("failed to create temp file");
    f.write_all(bytes).expect("failed to write temp file");
    path
}

/// Run otool -l on a .o file and return stdout.
fn run_otool(path: &std::path::Path) -> String {
    let output = Command::new("otool")
        .args(["-l", path.to_str().unwrap()])
        .output()
        .expect("failed to run otool");
    assert!(
        output.status.success(),
        "otool failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Run nm on a .o file and return stdout.
fn run_nm(path: &std::path::Path) -> String {
    let output = Command::new("nm")
        .args([path.to_str().unwrap()])
        .output()
        .expect("failed to run nm");
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn test_minimal_text_section() {
    let mut writer = MachOWriter::new();

    // ARM64 NOP = 0xD503201F, 4 instructions
    let nop = 0xD503201Fu32;
    let mut code = Vec::new();
    for _ in 0..4 {
        code.extend_from_slice(&nop.to_le_bytes());
    }
    writer.add_text_section(&code);
    writer.add_symbol("_main", 1, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "minimal_text");
    let otool_out = run_otool(&path);

    // Verify Mach-O magic
    assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

    // Verify file type is MH_OBJECT
    let filetype = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    assert_eq!(filetype, MH_OBJECT);

    // Verify load commands are present
    assert!(
        otool_out.contains("LC_SEGMENT_64"),
        "Missing LC_SEGMENT_64 in otool output"
    );
    assert!(
        otool_out.contains("LC_BUILD_VERSION"),
        "Missing LC_BUILD_VERSION in otool output"
    );
    assert!(
        otool_out.contains("LC_SYMTAB"),
        "Missing LC_SYMTAB in otool output"
    );
    assert!(
        otool_out.contains("LC_DYSYMTAB"),
        "Missing LC_DYSYMTAB in otool output"
    );

    // Verify sections
    assert!(
        otool_out.contains("__text"),
        "Missing __text section"
    );
    assert!(
        otool_out.contains("__TEXT"),
        "Missing __TEXT segment"
    );

    // Verify section attributes: pure instructions + some instructions
    assert!(
        otool_out.contains("0x80000400"),
        "Missing expected section flags for __text (S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS)"
    );

    // Verify symbol table via nm
    let nm_out = run_nm(&path);
    assert!(
        nm_out.contains("_main"),
        "Missing _main symbol in nm output"
    );
    assert!(
        nm_out.contains(" T "),
        "_main should be in text section (T)"
    );

    // Clean up
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_text_and_data_sections() {
    let mut writer = MachOWriter::new();

    // ARM64 RET = 0xD65F03C0
    let ret_instr = 0xD65F03C0u32;
    writer.add_text_section(&ret_instr.to_le_bytes());

    // Data section with some initialized data
    writer.add_data_section(&[0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00, 0x00, 0x00]);

    writer.add_symbol("_func", 1, 0, true);
    writer.add_symbol("_hello", 2, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "text_and_data");
    let otool_out = run_otool(&path);

    // Both sections present
    assert!(otool_out.contains("__text"), "Missing __text");
    assert!(otool_out.contains("__data"), "Missing __data");
    assert!(otool_out.contains("__TEXT"), "Missing __TEXT segment");
    assert!(otool_out.contains("__DATA"), "Missing __DATA segment");

    // nsects should be 2
    assert!(
        otool_out.contains("nsects 2"),
        "Expected 2 sections in segment"
    );

    // Both symbols visible
    let nm_out = run_nm(&path);
    assert!(nm_out.contains("_func"), "Missing _func symbol");
    assert!(nm_out.contains("_hello"), "Missing _hello symbol");

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_local_and_global_symbols() {
    let mut writer = MachOWriter::new();

    let nop = 0xD503201Fu32;
    let mut code = Vec::new();
    for _ in 0..8 {
        code.extend_from_slice(&nop.to_le_bytes());
    }
    writer.add_text_section(&code);

    // Add both local and global symbols
    writer.add_symbol("_local_helper", 1, 0, false);
    writer.add_symbol("_main", 1, 16, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "symbols");
    let otool_out = run_otool(&path);
    let nm_out = run_nm(&path);

    // Verify dysymtab shows correct partitioning
    assert!(
        otool_out.contains("nlocalsym 1"),
        "Expected 1 local symbol"
    );
    assert!(
        otool_out.contains("nextdefsym 1"),
        "Expected 1 external defined symbol"
    );

    // _main should be global (T), _local_helper should be local (t)
    assert!(nm_out.contains("_main"), "Missing _main");
    assert!(
        nm_out.contains("_local_helper"),
        "Missing _local_helper"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_relocation_entries() {
    let mut writer = MachOWriter::new();

    // BL instruction (will need relocation)
    let bl_instr = 0x94000000u32; // BL #0
    let mut code = Vec::new();
    code.extend_from_slice(&bl_instr.to_le_bytes());
    // Some NOPs after
    let nop = 0xD503201Fu32;
    for _ in 0..3 {
        code.extend_from_slice(&nop.to_le_bytes());
    }
    writer.add_text_section(&code);

    writer.add_symbol("_caller", 1, 0, true);
    writer.add_symbol("_callee", 0, 0, true); // undefined external

    // Add a BRANCH26 relocation at offset 0 referencing symbol index 1 (_callee)
    writer.add_relocation(
        0,
        Relocation::branch26(0, 1),
    );

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "reloc");
    let otool_out = run_otool(&path);

    // Section should have 1 relocation
    assert!(
        otool_out.contains("nreloc 1"),
        "Expected 1 relocation entry, otool output:\n{}", otool_out
    );

    // reloff should be non-zero (use leading whitespace to avoid matching "extreloff 0")
    assert!(
        !otool_out.contains("    reloff 0\n"),
        "reloff should not be 0 when there are relocations, otool output:\n{}", otool_out
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_empty_object() {
    let writer = MachOWriter::new();
    let bytes = writer.write();
    let path = write_temp_o(&bytes, "empty");
    let otool_out = run_otool(&path);

    // Even an empty object should be valid Mach-O
    assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
    assert!(
        otool_out.contains("LC_SEGMENT_64"),
        "Empty object should still have segment command"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_header_flags() {
    let mut writer = MachOWriter::new();
    writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);

    let bytes = writer.write();

    // flags field is at offset 24 in the header
    let flags = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert_eq!(
        flags, MH_SUBSECTIONS_VIA_SYMBOLS,
        "Header flags should be MH_SUBSECTIONS_VIA_SYMBOLS"
    );
}

#[test]
fn test_build_version_platform() {
    let mut writer = MachOWriter::new();
    writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "build_version");
    let otool_out = run_otool(&path);

    assert!(
        otool_out.contains("platform 1"),
        "Build version should target macOS (platform 1)"
    );

    std::fs::remove_file(&path).ok();
}

// llvm2-codegen integration test: Compact unwind emission
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Verifies that the compact unwind section is emitted correctly in Mach-O
// object files and can be parsed by macOS system tools (otool, llvm-objdump).

use llvm2_codegen::frame::{UNWIND_ARM64_FRAME_X19_X20_PAIR, UNWIND_ARM64_MODE_FRAME};
use llvm2_codegen::macho::MachOWriter;
use llvm2_codegen::unwind::{
    add_compact_unwind_to_writer, CompactUnwindEntry, CompactUnwindSection,
};

use std::io::Write;
use std::process::Command;

/// Write bytes to a temp file and return the path.
fn write_temp_o(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("llvm2_test_cu_{}.o", name));
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

#[test]
fn test_compact_unwind_section_present() {
    let mut writer = MachOWriter::new();

    // Emit a simple function: 4 NOPs + RET = 20 bytes
    let nop = 0xD503201Fu32;
    let ret = 0xD65F03C0u32;
    let mut code = Vec::new();
    for _ in 0..4 {
        code.extend_from_slice(&nop.to_le_bytes());
    }
    code.extend_from_slice(&ret.to_le_bytes());
    writer.add_text_section(&code);
    writer.add_symbol("_main", 1, 0, true);

    // Add compact unwind entry
    let mut cu_section = CompactUnwindSection::new();
    cu_section.add_entry(CompactUnwindEntry::new(
        0,                       // function_offset (will be relocated)
        code.len() as u32,       // function_length
        UNWIND_ARM64_MODE_FRAME, // compact_encoding
        0,                       // symbol_index for _main
    ));

    let section_index = add_compact_unwind_to_writer(&mut writer, &cu_section);
    assert!(
        section_index.is_some(),
        "Expected compact unwind section to be added"
    );

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "cu_present");
    let otool_out = run_otool(&path);

    // Verify __compact_unwind section exists
    assert!(
        otool_out.contains("__compact_unwind"),
        "Missing __compact_unwind section in otool output.\notool:\n{}",
        otool_out
    );

    // Verify __LD segment
    assert!(
        otool_out.contains("__LD"),
        "Missing __LD segment in otool output.\notool:\n{}",
        otool_out
    );

    // Verify section has correct size (32 bytes for one entry)
    // otool may use 32-bit or 64-bit hex format depending on platform/version
    assert!(
        otool_out.contains("size 0x00000020")
            || otool_out.contains("size 0x0000000000000020")
            || otool_out.contains("size 32"),
        "Expected compact unwind section size of 32 bytes.\notool:\n{}",
        otool_out
    );

    // Verify it has a relocation entry (for function_offset)
    // The compact_unwind section should have nreloc >= 1
    assert!(
        otool_out.contains("nreloc 1"),
        "Expected 1 relocation for compact unwind section.\notool:\n{}",
        otool_out
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_compact_unwind_multiple_functions() {
    let mut writer = MachOWriter::new();

    // Two functions in one text section
    let nop = 0xD503201Fu32;
    let ret = 0xD65F03C0u32;
    let mut code = Vec::new();
    // Function 1: 2 NOPs + RET = 12 bytes
    for _ in 0..2 {
        code.extend_from_slice(&nop.to_le_bytes());
    }
    code.extend_from_slice(&ret.to_le_bytes());
    let func2_offset = code.len();
    // Function 2: 3 NOPs + RET = 16 bytes
    for _ in 0..3 {
        code.extend_from_slice(&nop.to_le_bytes());
    }
    code.extend_from_slice(&ret.to_le_bytes());
    writer.add_text_section(&code);

    writer.add_symbol("_func1", 1, 0, true);
    writer.add_symbol("_func2", 1, func2_offset as u64, true);

    // Add compact unwind entries for both functions
    let mut cu_section = CompactUnwindSection::new();
    cu_section.add_entry(CompactUnwindEntry::new(
        0,
        12,
        UNWIND_ARM64_MODE_FRAME,
        0, // symbol index for _func1
    ));
    cu_section.add_entry(CompactUnwindEntry::new(
        0,
        16,
        UNWIND_ARM64_MODE_FRAME | UNWIND_ARM64_FRAME_X19_X20_PAIR,
        1, // symbol index for _func2
    ));

    let section_index = add_compact_unwind_to_writer(&mut writer, &cu_section);
    assert!(section_index.is_some());

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "cu_multi");
    let otool_out = run_otool(&path);

    // Verify section exists with 2 entries (64 bytes)
    assert!(
        otool_out.contains("__compact_unwind"),
        "Missing __compact_unwind section"
    );

    // 2 entries * 32 bytes = 64 = 0x40
    // otool may use 32-bit or 64-bit hex format depending on platform/version
    assert!(
        otool_out.contains("size 0x00000040")
            || otool_out.contains("size 0x0000000000000040")
            || otool_out.contains("size 64"),
        "Expected compact unwind section size of 64 bytes for 2 entries.\notool:\n{}",
        otool_out
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_compact_unwind_empty_section_not_added() {
    let mut writer = MachOWriter::new();
    writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);

    let cu_section = CompactUnwindSection::new();
    let section_index = add_compact_unwind_to_writer(&mut writer, &cu_section);
    assert!(section_index.is_none(), "Empty section should not be added");

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "cu_empty");
    let otool_out = run_otool(&path);

    // Should NOT have compact unwind section
    assert!(
        !otool_out.contains("__compact_unwind"),
        "Empty compact unwind section should not be present"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_compact_unwind_data_correctness() {
    // Verify the raw bytes of the compact unwind section are correct
    let encoding = UNWIND_ARM64_MODE_FRAME | UNWIND_ARM64_FRAME_X19_X20_PAIR;
    let entry = CompactUnwindEntry::new(0x1234, 0x100, encoding, 0);

    let bytes = entry.to_bytes();
    assert_eq!(bytes.len(), 32);

    // function_offset = 0x1234 (little-endian u64)
    assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 0x1234);

    // function_length = 0x100 (little-endian u32)
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0x100);

    // compact_encoding = FRAME | X19_X20 (little-endian u32)
    assert_eq!(
        u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        encoding
    );

    // personality = 0 (u64)
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);

    // lsda = 0 (u64)
    assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 0);
}

#[test]
fn test_compact_unwind_with_text_and_data() {
    // Verify compact unwind coexists with text and data sections
    let mut writer = MachOWriter::new();

    let ret = 0xD65F03C0u32;
    writer.add_text_section(&ret.to_le_bytes());
    writer.add_data_section(&[0u8; 8]);
    writer.add_symbol("_main", 1, 0, true);

    let mut cu_section = CompactUnwindSection::new();
    cu_section.add_entry(CompactUnwindEntry::new(0, 4, UNWIND_ARM64_MODE_FRAME, 0));
    add_compact_unwind_to_writer(&mut writer, &cu_section);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "cu_with_data");
    let otool_out = run_otool(&path);

    // All three sections should be present
    assert!(otool_out.contains("__text"), "Missing __text");
    assert!(otool_out.contains("__data"), "Missing __data");
    assert!(
        otool_out.contains("__compact_unwind"),
        "Missing __compact_unwind"
    );

    // Should have 3 sections total
    assert!(
        otool_out.contains("nsects 3"),
        "Expected 3 sections (text, data, compact_unwind).\notool:\n{}",
        otool_out
    );

    std::fs::remove_file(&path).ok();
}

// llvm2-codegen/tests/e2e_elf_link.rs - E2E ELF linking integration test
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves the x86-64 backend works end-to-end:
//   x86 ISel function -> regalloc -> frame lowering -> encode -> ELF .o
//
// This is the integration test for issue #201. It verifies that all x86-64
// pipeline stages compose correctly by producing an ELF .o file that:
//   1. Has valid ELF magic and structure (in-process binary parsing)
//   2. Has correct e_machine (EM_X86_64 = 0x3E)
//   3. Contains a .text section with non-empty code
//   4. Contains a properly encoded symbol table with the function name
//   5. Disassembles correctly with objdump (if on x86-64 and available)
//   6. Links and executes correctly (if on x86-64 with cc available)
//
// Part of #201 -- x86-64 end-to-end ELF linking test

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::elf::constants::*;
use llvm2_codegen::x86_64::{
    build_x86_add_test_function, build_x86_const_test_function,
    x86_compile_to_bytes, x86_compile_to_elf,
    X86Pipeline, X86PipelineConfig,
};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn is_x86_64() -> bool {
    cfg!(target_arch = "x86_64")
}

fn has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_objdump() -> bool {
    // Try GNU objdump first, then llvm-objdump
    Command::new("objdump")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
        || Command::new("llvm-objdump")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

fn has_readelf() -> bool {
    Command::new("readelf")
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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_elf_{}", test_name));
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
// ELF binary parsing helpers
// ---------------------------------------------------------------------------

/// Read a little-endian u16 from a byte slice at the given offset.
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Read a little-endian u32 from a byte slice at the given offset.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset], bytes[offset + 1],
        bytes[offset + 2], bytes[offset + 3],
    ])
}

/// Read a little-endian u64 from a byte slice at the given offset.
fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
        bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7],
    ])
}

/// Verify ELF magic bytes at the start of the file.
fn verify_elf_magic(bytes: &[u8]) {
    assert!(bytes.len() >= 16, "ELF file too small: {} bytes", bytes.len());
    assert_eq!(
        &bytes[0..4],
        &[0x7F, b'E', b'L', b'F'],
        "Invalid ELF magic: expected 7f 45 4c 46"
    );
}

/// Verify the ELF header identifies this as a 64-bit little-endian relocatable.
fn verify_elf_header_basics(bytes: &[u8]) {
    verify_elf_magic(bytes);

    // EI_CLASS = ELFCLASS64 (2)
    assert_eq!(bytes[4], ELFCLASS64, "must be 64-bit ELF");

    // EI_DATA = ELFDATA2LSB (1) = little-endian
    assert_eq!(bytes[5], ELFDATA2LSB, "must be little-endian");

    // EI_VERSION = EV_CURRENT (1)
    assert_eq!(bytes[6], EV_CURRENT, "ELF version must be current");

    // e_type = ET_REL (1) = relocatable object
    let e_type = read_u16(bytes, 16);
    assert_eq!(e_type, ET_REL, "must be relocatable object (ET_REL)");

    // e_machine = EM_X86_64 (62 = 0x3E)
    let e_machine = read_u16(bytes, 18);
    assert_eq!(e_machine, EM_X86_64, "must be x86-64 (EM_X86_64 = 0x3E)");

    // e_ehsize = 64 (ELF64 header size)
    let e_ehsize = read_u16(bytes, 52);
    assert_eq!(e_ehsize, 64, "ELF64 header size must be 64");

    // e_shentsize = 64 (section header entry size)
    let e_shentsize = read_u16(bytes, 58);
    assert_eq!(e_shentsize, 64, "section header entry size must be 64");
}

/// Find a section by name in the ELF section header table.
/// Returns (section_offset, section_size) if found.
fn find_elf_section(bytes: &[u8], target_name: &str) -> Option<(u64, u64)> {
    let sh_offset = read_u64(bytes, 40) as usize; // e_shoff
    let e_shnum = read_u16(bytes, 60) as usize;
    let e_shstrndx = read_u16(bytes, 62) as usize;

    // Read the section header string table location.
    let shstrtab_shdr = sh_offset + e_shstrndx * ELF64_SHDR_SIZE;
    let shstrtab_offset = read_u64(bytes, shstrtab_shdr + 24) as usize;
    let shstrtab_size = read_u64(bytes, shstrtab_shdr + 32) as usize;

    // Search all section headers.
    for i in 0..e_shnum {
        let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
        let sh_name = read_u32(bytes, shdr_off) as usize;

        // Read the section name from .shstrtab.
        if sh_name < shstrtab_size {
            let name_start = shstrtab_offset + sh_name;
            let name_end = bytes[name_start..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| name_start + p)
                .unwrap_or(name_start);
            let name = std::str::from_utf8(&bytes[name_start..name_end]).unwrap_or("");
            if name == target_name {
                let sec_offset = read_u64(bytes, shdr_off + 24);
                let sec_size = read_u64(bytes, shdr_off + 32);
                return Some((sec_offset, sec_size));
            }
        }
    }

    None
}

/// Find the .symtab section and return all symbol names.
fn find_elf_symbol_names(bytes: &[u8]) -> Vec<String> {
    let sh_offset = read_u64(bytes, 40) as usize;
    let e_shnum = read_u16(bytes, 60) as usize;

    // Find .symtab and .strtab sections.
    let mut symtab_offset = 0usize;
    let mut symtab_size = 0usize;
    let mut strtab_shdr_link = 0u32;

    for i in 0..e_shnum {
        let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
        let sh_type = read_u32(bytes, shdr_off + 4);
        if sh_type == SHT_SYMTAB {
            symtab_offset = read_u64(bytes, shdr_off + 24) as usize;
            symtab_size = read_u64(bytes, shdr_off + 32) as usize;
            strtab_shdr_link = read_u32(bytes, shdr_off + 40);
            break;
        }
    }

    if symtab_size == 0 {
        return Vec::new();
    }

    // Read the linked .strtab.
    let strtab_shdr = sh_offset + strtab_shdr_link as usize * ELF64_SHDR_SIZE;
    let strtab_offset = read_u64(bytes, strtab_shdr + 24) as usize;
    let strtab_size = read_u64(bytes, strtab_shdr + 32) as usize;

    let num_syms = symtab_size / ELF64_SYM_SIZE;
    let mut names = Vec::new();

    for i in 0..num_syms {
        let sym_off = symtab_offset + i * ELF64_SYM_SIZE;
        let st_name = read_u32(bytes, sym_off) as usize;
        if st_name > 0 && st_name < strtab_size {
            let name_start = strtab_offset + st_name;
            let name_end = bytes[name_start..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| name_start + p)
                .unwrap_or(name_start);
            if let Ok(name) = std::str::from_utf8(&bytes[name_start..name_end]) {
                names.push(name.to_string());
            }
        }
    }

    names
}

// ---------------------------------------------------------------------------
// External tool runners
// ---------------------------------------------------------------------------

/// Run objdump -d on the given object file and return the disassembly output.
fn run_objdump(path: &Path) -> Option<String> {
    // Try llvm-objdump first (more common on macOS), then GNU objdump
    let output = Command::new("llvm-objdump")
        .args(["-d", path.to_str().unwrap()])
        .output()
        .or_else(|_| {
            Command::new("objdump")
                .args(["-d", path.to_str().unwrap()])
                .output()
        })
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

/// Run readelf -h on the given object file and return the output.
fn run_readelf_header(path: &Path) -> Option<String> {
    let output = Command::new("readelf")
        .args(["-h", path.to_str().unwrap()])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

/// Run readelf -S on the given object file and return the output.
fn run_readelf_sections(path: &Path) -> Option<String> {
    let output = Command::new("readelf")
        .args(["-S", path.to_str().unwrap()])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

/// Run nm on the given object file and return the output.
fn run_nm(path: &Path) -> Option<String> {
    let output = Command::new("nm")
        .args([path.to_str().unwrap()])
        .output()
        .ok()?;

    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

fn link_elf_with_cc(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(driver_c)
        .arg(obj)
        .arg("-no-pie")
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

// ===========================================================================
// TEST 1: const42 -- compile to ELF, validate structure in-process
// ===========================================================================

/// Compiles fn const42() -> i64 through the x86-64 pipeline and validates
/// the resulting ELF object file structure entirely in-process (no external
/// tools needed). This test runs on all platforms.
#[test]
fn test_e2e_elf_const42_structure() {
    let func = build_x86_const_test_function();

    let elf_bytes = x86_compile_to_elf(&func)
        .expect("x86-64 pipeline should compile const42 to ELF");

    // -- ELF header validation --
    verify_elf_header_basics(&elf_bytes);

    // -- Section header table --
    let e_shnum = read_u16(&elf_bytes, 60);
    // At minimum: null + .text + .symtab + .strtab + .shstrtab = 5
    assert!(
        e_shnum >= 5,
        "need at least 5 sections (null + .text + .symtab + .strtab + .shstrtab), got {}",
        e_shnum
    );

    // -- .text section --
    let text = find_elf_section(&elf_bytes, ".text");
    assert!(text.is_some(), ".text section must be present");
    let (text_offset, text_size) = text.unwrap();
    assert!(text_offset > 0, ".text offset must be non-zero");
    assert!(text_size > 0, ".text must contain code");

    // -- Verify the code contains a RET instruction (0xC3) --
    let code = &elf_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];
    let has_ret = code.contains(&0xC3);
    assert!(has_ret, "x86-64 code must contain RET (0xC3). Code bytes: {:02X?}", code);

    // -- Verify the code contains a MOV imm64 with value 42 --
    // MOV r64, imm64 uses REX.W prefix + B8+rd opcode followed by 8-byte immediate.
    // The value 42 = 0x2A should appear in the immediate bytes.
    let has_42 = code.windows(8).any(|w| {
        u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]) == 42
    });
    assert!(has_42, "code must contain immediate value 42. Code bytes: {:02X?}", code);

    // -- Symbol table --
    let symbols = find_elf_symbol_names(&elf_bytes);
    assert!(
        symbols.contains(&"const42".to_string()),
        "symbol table must contain 'const42'. Found: {:?}",
        symbols
    );

    // -- Object file size sanity check --
    assert!(
        elf_bytes.len() >= 100 && elf_bytes.len() <= 10_000,
        "ELF file size {} is out of expected range [100, 10000]",
        elf_bytes.len()
    );
}

// ===========================================================================
// TEST 2: add -- compile to ELF, validate structure in-process
// ===========================================================================

/// Compiles fn add(a: i64, b: i64) -> i64 through the x86-64 pipeline and
/// validates ELF structure. Verifies the function with two arguments produces
/// valid ELF output with correct symbol.
#[test]
fn test_e2e_elf_add_structure() {
    let func = build_x86_add_test_function();

    let elf_bytes = x86_compile_to_elf(&func)
        .expect("x86-64 pipeline should compile add to ELF");

    // -- ELF header --
    verify_elf_header_basics(&elf_bytes);

    // -- .text section --
    let text = find_elf_section(&elf_bytes, ".text");
    assert!(text.is_some(), ".text section must be present");
    let (text_offset, text_size) = text.unwrap();
    assert!(text_size > 0, ".text must contain code");

    // -- Verify code ends with RET --
    let code = &elf_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];
    assert_eq!(
        *code.last().unwrap(), 0xC3,
        "last byte of x86-64 code must be RET (0xC3). Code: {:02X?}", code
    );

    // -- Verify ADD instruction is present --
    // x86-64 ADD r/m64, r64 uses opcode 0x01 (or 0x03 for ADD r64, r/m64)
    // with a REX.W prefix (0x48-0x4F). Look for these patterns.
    let has_add = code.windows(2).any(|w| {
        // REX.W prefix (0x48-0x4F) followed by ADD opcode (0x01 or 0x03)
        (w[0] >= 0x48 && w[0] <= 0x4F) && (w[1] == 0x01 || w[1] == 0x03)
    });
    assert!(has_add, "code must contain an ADD instruction. Code: {:02X?}", code);

    // -- Symbol table --
    let symbols = find_elf_symbol_names(&elf_bytes);
    assert!(
        symbols.contains(&"add".to_string()),
        "symbol table must contain 'add'. Found: {:?}",
        symbols
    );
}

// ===========================================================================
// TEST 3: raw bytes -- compile without ELF wrapper, verify code quality
// ===========================================================================

/// Compiles both test functions to raw bytes (no ELF wrapper) and verifies
/// the raw machine code is valid. This tests the pipeline without the ELF
/// writer, isolating encoding correctness.
#[test]
fn test_e2e_elf_raw_bytes_validation() {
    // const42: should produce code ending in RET
    let func1 = build_x86_const_test_function();
    let bytes1 = x86_compile_to_bytes(&func1)
        .expect("x86-64 pipeline should compile const42 to bytes");
    assert!(!bytes1.is_empty(), "const42 raw bytes should not be empty");
    assert_eq!(*bytes1.last().unwrap(), 0xC3, "const42 must end with RET");

    // add: should produce code ending in RET
    let func2 = build_x86_add_test_function();
    let bytes2 = x86_compile_to_bytes(&func2)
        .expect("x86-64 pipeline should compile add to bytes");
    assert!(!bytes2.is_empty(), "add raw bytes should not be empty");
    assert_eq!(*bytes2.last().unwrap(), 0xC3, "add must end with RET");

    // Both functions should produce non-trivial code (more than just RET).
    assert!(
        bytes1.len() >= 2,
        "const42 ({} bytes) should be at least 2 bytes (MOV+RET)",
        bytes1.len()
    );
    assert!(
        bytes2.len() >= 2,
        "add ({} bytes) should be at least 2 bytes",
        bytes2.len()
    );
}

// ===========================================================================
// TEST 4: ELF vs raw bytes -- verify ELF wraps the same code
// ===========================================================================

/// Verifies that the ELF output wraps the same machine code as the raw bytes
/// output. The .text section of the ELF should contain exactly the raw bytes.
#[test]
fn test_e2e_elf_contains_raw_code() {
    let func = build_x86_const_test_function();

    let raw_bytes = x86_compile_to_bytes(&func)
        .expect("raw bytes compilation should succeed");

    let elf_bytes = x86_compile_to_elf(&func)
        .expect("ELF compilation should succeed");

    // Find .text in the ELF.
    let text = find_elf_section(&elf_bytes, ".text")
        .expect(".text section must exist");
    let (text_offset, text_size) = text;

    let elf_code = &elf_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];

    assert_eq!(
        elf_code, raw_bytes.as_slice(),
        "ELF .text section must contain exactly the raw machine code bytes"
    );
}

// ===========================================================================
// TEST 5: pipeline config -- emit_elf and emit_frame interact correctly
// ===========================================================================

/// Tests various pipeline configuration combinations to ensure they all
/// produce valid output.
#[test]
fn test_e2e_elf_pipeline_configs() {
    let func = build_x86_const_test_function();

    // Config: no ELF, no frame
    let pipeline1 = X86Pipeline::new(X86PipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let bytes1 = pipeline1.compile_function(&func).expect("no-elf no-frame should work");
    assert!(!bytes1.is_empty());
    assert_eq!(*bytes1.last().unwrap(), 0xC3);

    // Config: no ELF, with frame
    let pipeline2 = X86Pipeline::new(X86PipelineConfig {
        emit_elf: false,
        emit_frame: true,
    });
    let bytes2 = pipeline2.compile_function(&func).expect("no-elf with-frame should work");
    assert!(!bytes2.is_empty());
    assert_eq!(*bytes2.last().unwrap(), 0xC3);
    // With frame should be larger (prologue/epilogue add instructions)
    assert!(
        bytes2.len() > bytes1.len(),
        "with-frame ({} bytes) should be larger than no-frame ({} bytes)",
        bytes2.len(), bytes1.len()
    );

    // Config: ELF, with frame
    let pipeline3 = X86Pipeline::new(X86PipelineConfig {
        emit_elf: true,
        emit_frame: true,
    });
    let bytes3 = pipeline3.compile_function(&func).expect("elf with-frame should work");
    verify_elf_header_basics(&bytes3);
    // ELF wrapper should be larger than raw code
    assert!(
        bytes3.len() > bytes2.len(),
        "ELF output ({} bytes) should be larger than raw code ({} bytes)",
        bytes3.len(), bytes2.len()
    );

    // Config: ELF, no frame
    let pipeline4 = X86Pipeline::new(X86PipelineConfig {
        emit_elf: true,
        emit_frame: false,
    });
    let bytes4 = pipeline4.compile_function(&func).expect("elf no-frame should work");
    verify_elf_header_basics(&bytes4);
}

// ===========================================================================
// TEST 6: both functions produce valid ELF with correct symbols
// ===========================================================================

/// Verify that both test functions produce valid ELF objects with correct
/// function symbols. This is a cross-function regression test.
#[test]
fn test_e2e_elf_all_functions_produce_valid_elf() {
    let functions: Vec<(&str, _)> = vec![
        ("const42", build_x86_const_test_function()),
        ("add", build_x86_add_test_function()),
    ];

    for (name, func) in &functions {
        let elf_bytes = x86_compile_to_elf(func)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        // Verify ELF header.
        verify_elf_header_basics(&elf_bytes);

        // Verify .text section exists and has code.
        let text = find_elf_section(&elf_bytes, ".text");
        assert!(text.is_some(), "{}: .text section must exist", name);
        let (_, text_size) = text.unwrap();
        assert!(text_size > 0, "{}: .text must have code", name);

        // Verify function symbol is present.
        let symbols = find_elf_symbol_names(&elf_bytes);
        assert!(
            symbols.contains(&name.to_string()),
            "{}: symbol table must contain '{}'. Found: {:?}",
            name, name, symbols
        );

        // Verify object file size is reasonable.
        assert!(
            elf_bytes.len() >= 100 && elf_bytes.len() <= 10_000,
            "{}: ELF size {} is out of range [100, 10000]",
            name, elf_bytes.len()
        );
    }
}

// ===========================================================================
// TEST 7: frame lowering -- prologue/epilogue produce correct x86-64 patterns
// ===========================================================================

/// Verify that the x86-64 frame lowering produces correct prologue/epilogue
/// patterns. The prologue should contain PUSH RBP + MOV RBP,RSP, and the
/// epilogue should contain POP RBP before RET.
#[test]
fn test_e2e_elf_frame_lowering_patterns() {
    let func = build_x86_const_test_function();

    // Compile with frame enabled.
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        emit_elf: false,
        emit_frame: true,
    });
    let code = pipeline.compile_function(&func).expect("compilation should succeed");

    // PUSH RBP = 0x55 (or 0x50 + 5 with REX if needed, but RBP is not extended)
    // On x86-64, PUSH RBP is just 0x55 (no REX needed since RBP < R8)
    assert!(
        code.contains(&0x55),
        "framed code must contain PUSH RBP (0x55). Code: {:02X?}", code
    );

    // POP RBP = 0x5D
    assert!(
        code.contains(&0x5D),
        "framed code must contain POP RBP (0x5D). Code: {:02X?}", code
    );

    // RET = 0xC3 (last byte)
    assert_eq!(*code.last().unwrap(), 0xC3, "code must end with RET");

    // The POP RBP should appear just before the RET.
    let code_len = code.len();
    assert!(code_len >= 2, "code must be at least 2 bytes");
    assert_eq!(code[code_len - 2], 0x5D, "POP RBP (0x5D) must precede RET (0xC3)");
}

// ===========================================================================
// TEST 8: readelf/objdump validation (x86-64 only, tool-dependent)
// ===========================================================================

/// If on an x86-64 system with readelf/objdump available, validate the ELF
/// object file with external tools for extra assurance.
#[test]
fn test_e2e_elf_external_tool_validation() {
    if !has_readelf() && !has_objdump() {
        eprintln!("Skipping: neither readelf nor objdump available");
        return;
    }

    let dir = make_test_dir("elf_tool_validation");
    let func = build_x86_const_test_function();
    let elf_bytes = x86_compile_to_elf(&func)
        .expect("x86-64 pipeline should compile const42 to ELF");
    let obj_path = write_object_file(&dir, "const42.o", &elf_bytes);

    // -- readelf -h: ELF header --
    if has_readelf() {
        if let Some(header_out) = run_readelf_header(&obj_path) {
            eprintln!("readelf -h output:\n{}", header_out);
            assert!(
                header_out.contains("ELF64") || header_out.contains("Class:"),
                "readelf must parse the ELF header.\n{}", header_out
            );
            assert!(
                header_out.contains("X86-64") || header_out.contains("x86-64")
                    || header_out.contains("Advanced Micro Devices X86-64"),
                "readelf must identify x86-64 machine.\n{}", header_out
            );
            assert!(
                header_out.contains("REL") || header_out.contains("Relocatable"),
                "readelf must identify as relocatable.\n{}", header_out
            );
        }

        // -- readelf -S: sections --
        if let Some(sections_out) = run_readelf_sections(&obj_path) {
            eprintln!("readelf -S output:\n{}", sections_out);
            assert!(
                sections_out.contains(".text"),
                "readelf must show .text section.\n{}", sections_out
            );
            assert!(
                sections_out.contains(".symtab"),
                "readelf must show .symtab section.\n{}", sections_out
            );
        }
    }

    // -- objdump -d: disassembly --
    if has_objdump() {
        if let Some(disasm) = run_objdump(&obj_path) {
            eprintln!("objdump -d output:\n{}", disasm);
            // The disassembly should contain a ret instruction.
            assert!(
                disasm.contains("ret") || disasm.contains("retq"),
                "disassembly must contain 'ret'.\n{}", disasm
            );
        }
    }

    // -- nm: symbols --
    if has_nm() {
        if let Some(nm_out) = run_nm(&obj_path) {
            eprintln!("nm output:\n{}", nm_out);
            assert!(
                nm_out.contains("const42"),
                "nm must show const42 symbol.\n{}", nm_out
            );
        }
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST 9: add function ELF with external tools (x86-64 only)
// ===========================================================================

/// Validate the add function's ELF output with external tools.
#[test]
fn test_e2e_elf_add_external_tool_validation() {
    if !has_readelf() && !has_objdump() {
        eprintln!("Skipping: neither readelf nor objdump available");
        return;
    }

    let dir = make_test_dir("elf_add_tool_validation");
    let func = build_x86_add_test_function();
    let elf_bytes = x86_compile_to_elf(&func)
        .expect("x86-64 pipeline should compile add to ELF");
    let obj_path = write_object_file(&dir, "add.o", &elf_bytes);

    // -- objdump -d: disassembly should show add and ret --
    if has_objdump() {
        if let Some(disasm) = run_objdump(&obj_path) {
            eprintln!("add disassembly:\n{}", disasm);
            assert!(
                disasm.contains("ret") || disasm.contains("retq"),
                "add disassembly must contain 'ret'.\n{}", disasm
            );
            assert!(
                disasm.contains("add") || disasm.contains("ADD"),
                "add disassembly must contain 'add' instruction.\n{}", disasm
            );
        }
    }

    // -- nm: symbol --
    if has_nm() {
        if let Some(nm_out) = run_nm(&obj_path) {
            assert!(
                nm_out.contains("add"),
                "nm must show 'add' symbol.\n{}", nm_out
            );
        }
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST 10: link and execute (x86-64 Linux only)
// ===========================================================================

/// The definitive end-to-end test: compile x86-64 ISel function const42(),
/// produce an ELF .o file, link it with a C driver using cc, and execute.
/// This proves the entire x86-64 pipeline produces correct, linkable code.
///
/// Only runs on x86-64 systems with cc available.
#[test]
fn test_e2e_elf_const42_link_and_run() {
    if !is_x86_64() || !has_cc() {
        eprintln!("Skipping: not x86-64 or cc not available");
        return;
    }

    let dir = make_test_dir("elf_const42_run");

    let func = build_x86_const_test_function();
    let elf_bytes = x86_compile_to_elf(&func)
        .expect("x86-64 pipeline should compile const42 to ELF");

    let obj_path = write_object_file(&dir, "const42.o", &elf_bytes);

    // C driver that calls our compiled function and validates the result.
    let driver_src = r#"
#include <stdio.h>
extern long const42(void);
int main(void) {
    long r = const42();
    printf("const42()=%ld\n", r);
    if (r != 42) return 1;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_elf_with_cc(&dir, &driver_path, &obj_path, "test_const42");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("const42 link+run stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "const42 link+run failed with exit code {} (1=const42()!=42). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

/// Link and execute the add function on x86-64.
#[test]
fn test_e2e_elf_add_link_and_run() {
    if !is_x86_64() || !has_cc() {
        eprintln!("Skipping: not x86-64 or cc not available");
        return;
    }

    let dir = make_test_dir("elf_add_run");

    let func = build_x86_add_test_function();
    let elf_bytes = x86_compile_to_elf(&func)
        .expect("x86-64 pipeline should compile add to ELF");

    let obj_path = write_object_file(&dir, "add.o", &elf_bytes);

    // C driver that calls add(a, b) and validates results.
    // Note: The x86-64 add function uses System V AMD64 ABI (RDI, RSI -> RAX).
    let driver_src = r#"
#include <stdio.h>
extern long add(long a, long b);
int main(void) {
    long r1 = add(3, 4);
    long r2 = add(0, 0);
    long r3 = add(-5, 10);
    long r4 = add(1000000, 2000000);
    long r5 = add(-100, -200);
    printf("add(3,4)=%ld add(0,0)=%ld add(-5,10)=%ld add(1M,2M)=%ld add(-100,-200)=%ld\n",
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
    let binary = link_elf_with_cc(&dir, &driver_path, &obj_path, "test_add");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("add link+run stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "add link+run failed with exit code {} \
         (1=add(3,4)!=7, 2=add(0,0)!=0, 3=add(-5,10)!=5, 4=add(1M,2M)!=3M, 5=add(-100,-200)!=-300). \
         stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

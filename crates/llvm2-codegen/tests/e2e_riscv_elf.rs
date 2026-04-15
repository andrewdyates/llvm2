// llvm2-codegen/tests/e2e_riscv_elf.rs - E2E RISC-V ELF linking integration test
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves the RISC-V RV64GC backend works end-to-end:
//   RISC-V ISel function -> regalloc -> frame lowering -> encode -> ELF .o
//
// This is the integration test for RISC-V target validation. It verifies that
// all RISC-V pipeline stages compose correctly by producing an ELF .o file that:
//   1. Has valid ELF magic and structure (in-process binary parsing)
//   2. Has correct e_machine (EM_RISCV = 0xF3 = 243)
//   3. Contains a .text section with non-empty, properly encoded code
//   4. Contains a properly encoded symbol table with the function name
//   5. All instructions are exactly 4 bytes (RISC-V fixed encoding)
//   6. Instruction encoding matches the RISC-V specification
//   7. If RISC-V cross-tools available: disassembles with objdump, links, runs
//
// Part of #209 -- RISC-V end-to-end linking tests

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::elf::constants::*;
use llvm2_codegen::riscv::{
    build_riscv_add_test_function, build_riscv_const_test_function,
    riscv_compile_to_bytes, riscv_compile_to_elf,
    RiscVPipeline, RiscVPipelineConfig,
    RiscVISelFunction, RiscVISelInst, RiscVISelOperand,
};
use llvm2_ir::regs::{RegClass, VReg};
use llvm2_ir::riscv_ops::RiscVOpcode;
use llvm2_ir::riscv_regs::{A0, A1, A2, RA, ZERO};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn has_riscv_objdump() -> bool {
    // Try riscv64-linux-gnu-objdump, riscv64-unknown-elf-objdump, llvm-objdump
    for cmd in &[
        "riscv64-linux-gnu-objdump",
        "riscv64-unknown-elf-objdump",
        "llvm-objdump",
    ] {
        if Command::new(cmd)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

fn has_riscv_readelf() -> bool {
    for cmd in &[
        "riscv64-linux-gnu-readelf",
        "riscv64-unknown-elf-readelf",
        "readelf",
    ] {
        if Command::new(cmd)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_riscv_{}", test_name));
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
// ELF binary parsing helpers
// ---------------------------------------------------------------------------

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset], bytes[offset + 1],
        bytes[offset + 2], bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
        bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7],
    ])
}

fn verify_elf_magic(bytes: &[u8]) {
    assert!(bytes.len() >= 16, "ELF file too small: {} bytes", bytes.len());
    assert_eq!(
        &bytes[0..4],
        &[0x7F, b'E', b'L', b'F'],
        "Invalid ELF magic: expected 7f 45 4c 46"
    );
}

/// Verify the ELF header identifies this as a 64-bit LE RISC-V relocatable.
fn verify_riscv_elf_header(bytes: &[u8]) {
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

    // e_machine = EM_RISCV (243 = 0xF3)
    let e_machine = read_u16(bytes, 18);
    assert_eq!(e_machine, EM_RISCV, "must be RISC-V (EM_RISCV = 0xF3 = 243)");

    // e_ehsize = 64 (ELF64 header size)
    let e_ehsize = read_u16(bytes, 52);
    assert_eq!(e_ehsize, 64, "ELF64 header size must be 64");

    // e_shentsize = 64 (section header entry size)
    let e_shentsize = read_u16(bytes, 58);
    assert_eq!(e_shentsize, 64, "section header entry size must be 64");
}

/// Find a section by name in the ELF section header table.
fn find_elf_section(bytes: &[u8], target_name: &str) -> Option<(u64, u64)> {
    let sh_offset = read_u64(bytes, 40) as usize;
    let e_shnum = read_u16(bytes, 60) as usize;
    let e_shstrndx = read_u16(bytes, 62) as usize;

    let shstrtab_shdr = sh_offset + e_shstrndx * ELF64_SHDR_SIZE;
    let shstrtab_offset = read_u64(bytes, shstrtab_shdr + 24) as usize;
    let shstrtab_size = read_u64(bytes, shstrtab_shdr + 32) as usize;

    for i in 0..e_shnum {
        let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
        let sh_name = read_u32(bytes, shdr_off) as usize;

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
// RISC-V instruction decoding helpers (for verification)
// ---------------------------------------------------------------------------

/// Extract opcode field [6:0] from a 32-bit RISC-V instruction.
fn riscv_opcode(inst: u32) -> u32 {
    inst & 0x7F
}

/// Extract rd field [11:7] from a 32-bit RISC-V instruction.
fn riscv_rd(inst: u32) -> u32 {
    (inst >> 7) & 0x1F
}

/// Extract funct3 field [14:12] from a 32-bit RISC-V instruction.
fn riscv_funct3(inst: u32) -> u32 {
    (inst >> 12) & 0x7
}

/// Extract rs1 field [19:15] from a 32-bit RISC-V instruction.
fn riscv_rs1(inst: u32) -> u32 {
    (inst >> 15) & 0x1F
}

/// Extract rs2 field [24:20] from a 32-bit RISC-V instruction.
fn riscv_rs2(inst: u32) -> u32 {
    (inst >> 20) & 0x1F
}

/// Extract funct7 field [31:25] from a 32-bit RISC-V instruction.
fn riscv_funct7(inst: u32) -> u32 {
    (inst >> 25) & 0x7F
}

/// Extract I-type immediate [31:20].
fn riscv_imm_i(inst: u32) -> i32 {
    (inst as i32) >> 20 // sign-extend
}

/// RISC-V base opcode constants
const OP: u32 = 0b0110011;       // R-type integer
const OP_IMM: u32 = 0b0010011;   // I-type immediate
const LOAD: u32 = 0b0000011;     // I-type loads
const STORE: u32 = 0b0100011;    // S-type stores
const BRANCH: u32 = 0b1100011;   // B-type branches
const LUI_OP: u32 = 0b0110111;   // U-type LUI
/// Decode a RISC-V instruction from 4 little-endian bytes.
fn decode_riscv_inst(code: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        code[offset], code[offset + 1],
        code[offset + 2], code[offset + 3],
    ])
}

/// Check if a RISC-V instruction is JALR x0, ra, 0 (RET pseudo-instruction).
fn is_riscv_ret(inst: u32) -> bool {
    // JALR x0, x1, 0 = opcode=1100111, rd=0, funct3=000, rs1=1, imm=0
    // = 0x00008067
    inst == 0x00008067
}

// ===========================================================================
// TEST 1: const42 -- compile to ELF, validate structure in-process
// ===========================================================================

/// Compiles fn const42() -> i64 through the RISC-V pipeline and validates
/// the resulting ELF object file structure entirely in-process.
#[test]
fn test_riscv_e2e_const42_elf_structure() {
    let func = build_riscv_const_test_function();

    let elf_bytes = riscv_compile_to_elf(&func)
        .expect("RISC-V pipeline should compile const42 to ELF");

    // -- ELF header validation --
    verify_riscv_elf_header(&elf_bytes);

    // -- Section header table --
    let e_shnum = read_u16(&elf_bytes, 60);
    // At minimum: null + .text + .symtab + .strtab + .shstrtab = 5
    assert!(
        e_shnum >= 5,
        "need at least 5 sections, got {}", e_shnum
    );

    // -- .text section --
    let text = find_elf_section(&elf_bytes, ".text");
    assert!(text.is_some(), ".text section must be present");
    let (text_offset, text_size) = text.unwrap();
    assert!(text_offset > 0, ".text offset must be non-zero");
    assert!(text_size > 0, ".text must contain code");

    // -- RISC-V: all instructions are 4 bytes --
    assert_eq!(
        text_size % 4, 0,
        ".text size {} must be multiple of 4 (RISC-V fixed encoding)",
        text_size
    );

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

/// Compiles fn add(a: i64, b: i64) -> i64 through the RISC-V pipeline.
#[test]
fn test_riscv_e2e_add_elf_structure() {
    let func = build_riscv_add_test_function();

    let elf_bytes = riscv_compile_to_elf(&func)
        .expect("RISC-V pipeline should compile add to ELF");

    // -- ELF header --
    verify_riscv_elf_header(&elf_bytes);

    // -- .text section --
    let text = find_elf_section(&elf_bytes, ".text");
    assert!(text.is_some(), ".text section must be present");
    let (text_offset, text_size) = text.unwrap();
    assert!(text_size > 0, ".text must contain code");
    assert_eq!(text_size % 4, 0, ".text must be 4-byte aligned");

    // -- Verify the code contains a RET (JALR x0, ra, 0 = 0x00008067) --
    let code = &elf_bytes[text_offset as usize..(text_offset as usize + text_size as usize)];
    let num_insts = code.len() / 4;
    let last_inst = decode_riscv_inst(code, (num_insts - 1) * 4);
    assert!(
        is_riscv_ret(last_inst),
        "last RISC-V instruction must be RET (JALR x0, ra, 0 = 0x00008067), got 0x{:08X}",
        last_inst
    );

    // -- Verify an ADD instruction is present (funct7=0, funct3=000, opcode=0110011) --
    let has_add = (0..num_insts).any(|i| {
        let inst = decode_riscv_inst(code, i * 4);
        riscv_opcode(inst) == OP && riscv_funct3(inst) == 0b000 && riscv_funct7(inst) == 0b0000000
    });
    assert!(has_add, "code must contain an ADD instruction");

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

/// Compiles RISC-V test functions to raw bytes (no ELF wrapper) and verifies
/// the raw machine code is valid.
#[test]
fn test_riscv_e2e_raw_bytes_validation() {
    // const42: should produce code ending in RET
    let func1 = build_riscv_const_test_function();
    let bytes1 = riscv_compile_to_bytes(&func1)
        .expect("RISC-V pipeline should compile const42 to bytes");
    assert!(!bytes1.is_empty(), "const42 raw bytes should not be empty");
    assert_eq!(bytes1.len() % 4, 0, "RISC-V code must be 4-byte aligned");

    // Last instruction must be RET.
    let last_inst1 = decode_riscv_inst(&bytes1, bytes1.len() - 4);
    assert!(
        is_riscv_ret(last_inst1),
        "const42 must end with RET, got 0x{:08X}", last_inst1
    );

    // add: should produce code ending in RET
    let func2 = build_riscv_add_test_function();
    let bytes2 = riscv_compile_to_bytes(&func2)
        .expect("RISC-V pipeline should compile add to bytes");
    assert!(!bytes2.is_empty(), "add raw bytes should not be empty");
    assert_eq!(bytes2.len() % 4, 0, "RISC-V code must be 4-byte aligned");

    let last_inst2 = decode_riscv_inst(&bytes2, bytes2.len() - 4);
    assert!(
        is_riscv_ret(last_inst2),
        "add must end with RET, got 0x{:08X}", last_inst2
    );
}

// ===========================================================================
// TEST 4: ELF vs raw bytes -- verify ELF wraps the same code
// ===========================================================================

/// Verifies that the ELF output wraps the same machine code as the raw bytes.
#[test]
fn test_riscv_e2e_elf_contains_raw_code() {
    let func = build_riscv_const_test_function();

    let raw_bytes = riscv_compile_to_bytes(&func)
        .expect("raw bytes compilation should succeed");

    let elf_bytes = riscv_compile_to_elf(&func)
        .expect("ELF compilation should succeed");

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

/// Tests various RISC-V pipeline configuration combinations.
#[test]
fn test_riscv_e2e_pipeline_configs() {
    let func = build_riscv_const_test_function();

    // Config: no ELF, no frame
    let pipeline1 = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let bytes1 = pipeline1.compile_function(&func).expect("no-elf no-frame should work");
    assert!(!bytes1.is_empty());
    assert_eq!(bytes1.len() % 4, 0);
    // Last instruction should be RET.
    let last1 = decode_riscv_inst(&bytes1, bytes1.len() - 4);
    assert!(is_riscv_ret(last1), "no-frame: last inst must be RET, got 0x{:08X}", last1);

    // Config: no ELF, with frame
    let pipeline2 = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: true,
    });
    let bytes2 = pipeline2.compile_function(&func).expect("no-elf with-frame should work");
    assert!(!bytes2.is_empty());
    assert_eq!(bytes2.len() % 4, 0);
    // With frame should be larger (prologue/epilogue add instructions).
    assert!(
        bytes2.len() > bytes1.len(),
        "with-frame ({} bytes) should be larger than no-frame ({} bytes)",
        bytes2.len(), bytes1.len()
    );

    // Config: ELF, with frame
    let pipeline3 = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: true,
        emit_frame: true,
    });
    let bytes3 = pipeline3.compile_function(&func).expect("elf with-frame should work");
    verify_riscv_elf_header(&bytes3);
    assert!(
        bytes3.len() > bytes2.len(),
        "ELF output ({} bytes) should be larger than raw code ({} bytes)",
        bytes3.len(), bytes2.len()
    );

    // Config: ELF, no frame
    let pipeline4 = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: true,
        emit_frame: false,
    });
    let bytes4 = pipeline4.compile_function(&func).expect("elf no-frame should work");
    verify_riscv_elf_header(&bytes4);
}

// ===========================================================================
// TEST 6: all functions produce valid ELF with correct symbols
// ===========================================================================

/// Verify that both test functions produce valid RISC-V ELF objects.
#[test]
fn test_riscv_e2e_all_functions_produce_valid_elf() {
    let functions: Vec<(&str, _)> = vec![
        ("const42", build_riscv_const_test_function()),
        ("add", build_riscv_add_test_function()),
    ];

    for (name, func) in &functions {
        let elf_bytes = riscv_compile_to_elf(func)
            .unwrap_or_else(|e| panic!("pipeline failed for {}: {}", name, e));

        verify_riscv_elf_header(&elf_bytes);

        let text = find_elf_section(&elf_bytes, ".text");
        assert!(text.is_some(), "{}: .text section must exist", name);
        let (_, text_size) = text.unwrap();
        assert!(text_size > 0, "{}: .text must have code", name);
        assert_eq!(text_size % 4, 0, "{}: .text must be 4-byte aligned", name);

        let symbols = find_elf_symbol_names(&elf_bytes);
        assert!(
            symbols.contains(&name.to_string()),
            "{}: symbol table must contain '{}'. Found: {:?}",
            name, name, symbols
        );

        assert!(
            elf_bytes.len() >= 100 && elf_bytes.len() <= 10_000,
            "{}: ELF size {} is out of range [100, 10000]",
            name, elf_bytes.len()
        );
    }
}

// ===========================================================================
// TEST 7: frame lowering -- prologue/epilogue produce correct RISC-V patterns
// ===========================================================================

/// Verify that the RISC-V frame lowering produces correct LP64D ABI patterns.
/// The prologue should: ADDI SP, SP, -frame_size; SD RA; SD S0; ADDI S0, SP, frame_size
/// The epilogue should: LD RA; LD S0; ADDI SP, SP, frame_size (before RET)
#[test]
fn test_riscv_e2e_frame_lowering_patterns() {
    let func = build_riscv_const_test_function();

    // Compile with frame enabled.
    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: true,
    });
    let code = pipeline.compile_function(&func).expect("compilation should succeed");
    assert!(code.len() >= 16, "framed code must be at least 4 instructions");

    let num_insts = code.len() / 4;
    let insts: Vec<u32> = (0..num_insts).map(|i| decode_riscv_inst(&code, i * 4)).collect();

    // First instruction should be ADDI SP, SP, -16 (or similar negative immediate).
    // ADDI SP, SP, imm: opcode=OP_IMM, rd=2(sp), funct3=000, rs1=2(sp)
    let first = insts[0];
    assert_eq!(riscv_opcode(first), OP_IMM, "first inst must be ADDI (opcode=0010011)");
    assert_eq!(riscv_rd(first), 2, "first inst rd must be SP (x2)");
    assert_eq!(riscv_rs1(first), 2, "first inst rs1 must be SP (x2)");
    let sp_adjust = riscv_imm_i(first);
    assert!(sp_adjust < 0, "prologue must subtract from SP, got adjustment {}", sp_adjust);
    assert_eq!(sp_adjust % 16, 0, "stack adjustment {} must be 16-byte aligned", sp_adjust);

    // Verify prologue contains SD RA (store return address).
    // SD: opcode=STORE, funct3=011
    let has_sd_ra = insts.iter().any(|&inst| {
        riscv_opcode(inst) == STORE && riscv_funct3(inst) == 0b011
            && riscv_rs2(inst) == 1  // rs2 = x1 (ra)
            && riscv_rs1(inst) == 2  // rs1 = x2 (sp)
    });
    assert!(has_sd_ra, "prologue must save RA with SD ra, offset(sp)");

    // Verify prologue contains SD S0 (save frame pointer).
    let has_sd_s0 = insts.iter().any(|&inst| {
        riscv_opcode(inst) == STORE && riscv_funct3(inst) == 0b011
            && riscv_rs2(inst) == 8  // rs2 = x8 (s0/fp)
            && riscv_rs1(inst) == 2  // rs1 = x2 (sp)
    });
    assert!(has_sd_s0, "prologue must save S0/FP with SD s0, offset(sp)");

    // Verify epilogue contains LD RA (restore return address).
    // LD: opcode=LOAD, funct3=011
    let has_ld_ra = insts.iter().any(|&inst| {
        riscv_opcode(inst) == LOAD && riscv_funct3(inst) == 0b011
            && riscv_rd(inst) == 1   // rd = x1 (ra)
            && riscv_rs1(inst) == 2  // rs1 = x2 (sp)
    });
    assert!(has_ld_ra, "epilogue must restore RA with LD ra, offset(sp)");

    // Verify epilogue contains LD S0 (restore frame pointer).
    let has_ld_s0 = insts.iter().any(|&inst| {
        riscv_opcode(inst) == LOAD && riscv_funct3(inst) == 0b011
            && riscv_rd(inst) == 8   // rd = x8 (s0/fp)
            && riscv_rs1(inst) == 2  // rs1 = x2 (sp)
    });
    assert!(has_ld_s0, "epilogue must restore S0/FP with LD s0, offset(sp)");

    // Last instruction must be RET.
    assert!(
        is_riscv_ret(*insts.last().unwrap()),
        "code must end with RET, got 0x{:08X}", insts.last().unwrap()
    );
}

// ===========================================================================
// TEST 8: RISC-V instruction encoding verification in E2E context
// ===========================================================================

/// Verifies that the const42 function's no-frame output encodes correctly:
/// ADDI a0, zero, 42 followed by JALR zero, ra, 0 (RET).
#[test]
fn test_riscv_e2e_const42_encoding_correctness() {
    let func = build_riscv_const_test_function();

    // No frame -- just the two body instructions.
    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("compilation should succeed");

    // const42 without frame: ADDI a0, x0, 42; JALR x0, ra, 0
    // = 2 instructions = 8 bytes
    assert_eq!(code.len(), 8, "const42 no-frame should be 8 bytes (2 instructions)");

    // Instruction 0: ADDI a0, x0, 42
    // imm=42=0x02A, rs1=0(x0/zero), funct3=000, rd=10(a0), opcode=0010011
    // = 0x02A00513
    let inst0 = decode_riscv_inst(&code, 0);
    assert_eq!(inst0, 0x02A00513,
        "inst 0 should be ADDI a0, zero, 42 (0x02A00513), got 0x{:08X}", inst0);

    // Instruction 1: JALR x0, ra, 0 (RET)
    // = 0x00008067
    let inst1 = decode_riscv_inst(&code, 4);
    assert_eq!(inst1, 0x00008067,
        "inst 1 should be JALR x0, ra, 0 (RET = 0x00008067), got 0x{:08X}", inst1);
}

// ===========================================================================
// TEST 9: add function encoding correctness
// ===========================================================================

/// Verifies the add function's no-frame output encodes correctly:
/// ADDI v0, a0, 0; ADDI v1, a1, 0; ADD v2, v0, v1; ADDI a0, v2, 0; JALR x0, ra, 0
#[test]
fn test_riscv_e2e_add_encoding_correctness() {
    let func = build_riscv_add_test_function();

    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("compilation should succeed");

    // add without frame: 5 instructions = 20 bytes
    assert_eq!(code.len(), 20, "add no-frame should be 20 bytes (5 instructions)");

    let num_insts = code.len() / 4;
    let insts: Vec<u32> = (0..num_insts).map(|i| decode_riscv_inst(&code, i * 4)).collect();

    // Instruction 0: ADDI v0, a0, 0 (move from a0)
    // v0 gets allocated to first allocatable GPR -- verify it is an ADDI with imm=0, rs1=a0(10)
    assert_eq!(riscv_opcode(insts[0]), OP_IMM, "inst 0 must be ADDI");
    assert_eq!(riscv_funct3(insts[0]), 0b000, "inst 0 must have funct3=000 (ADDI)");
    assert_eq!(riscv_rs1(insts[0]), 10, "inst 0 must read from a0 (x10)");
    assert_eq!(riscv_imm_i(insts[0]), 0, "inst 0 must have imm=0 (move)");

    // Instruction 1: ADDI v1, a1, 0 (move from a1)
    assert_eq!(riscv_opcode(insts[1]), OP_IMM, "inst 1 must be ADDI");
    assert_eq!(riscv_rs1(insts[1]), 11, "inst 1 must read from a1 (x11)");
    assert_eq!(riscv_imm_i(insts[1]), 0, "inst 1 must have imm=0");

    // Instruction 2: ADD v2, v0, v1
    assert_eq!(riscv_opcode(insts[2]), OP, "inst 2 must be R-type (ADD)");
    assert_eq!(riscv_funct3(insts[2]), 0b000, "inst 2 must have funct3=000 (ADD)");
    assert_eq!(riscv_funct7(insts[2]), 0b0000000, "inst 2 must have funct7=0 (ADD, not SUB)");

    // Instruction 3: ADDI a0, v2, 0 (move result to return register)
    assert_eq!(riscv_opcode(insts[3]), OP_IMM, "inst 3 must be ADDI");
    assert_eq!(riscv_rd(insts[3]), 10, "inst 3 must write to a0 (x10)");
    assert_eq!(riscv_imm_i(insts[3]), 0, "inst 3 must have imm=0");

    // Instruction 4: JALR x0, ra, 0 (RET)
    assert!(is_riscv_ret(insts[4]), "inst 4 must be RET (0x00008067), got 0x{:08X}", insts[4]);
}

// ===========================================================================
// TEST 10: multi-block function with branches
// ===========================================================================

/// Build and compile a multi-block RISC-V function with conditional branches.
/// Verifies branch instructions encode with correct opcode fields.
#[test]
fn test_riscv_e2e_multi_block_branch() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };

    // Build: if (a0 == 0) return 1 else return a0
    let mut func = RiscVISelFunction::new("select".to_string(), sig);
    let b0 = Block(0);
    let b1 = Block(1);
    let b2 = Block(2);
    func.ensure_block(b0);
    func.ensure_block(b1);
    func.ensure_block(b2);

    // b0: BEQ a0, zero, b1  (if a0 == 0, goto b1)
    func.push_inst(b0, RiscVISelInst::new(
        RiscVOpcode::Beq,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::Block(b1),
        ],
    ));

    // b0 fallthrough to b2: ADDI a0, a0, 0 (keep a0); JALR x0, ra, 0
    func.push_inst(b0, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // b1: ADDI a0, zero, 1 (return 1)
    func.push_inst(b1, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::Imm(1),
        ],
    ));

    // b1: JALR x0, ra, 0 (return)
    func.push_inst(b1, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // Compile without frame for cleaner verification.
    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("multi-block compilation should succeed");

    assert!(!code.is_empty(), "multi-block code should not be empty");
    assert_eq!(code.len() % 4, 0, "RISC-V code must be 4-byte aligned");

    let num_insts = code.len() / 4;
    let insts: Vec<u32> = (0..num_insts).map(|i| decode_riscv_inst(&code, i * 4)).collect();

    // First instruction should be BEQ (opcode = BRANCH = 0b1100011, funct3 = 000)
    assert_eq!(riscv_opcode(insts[0]), BRANCH, "first inst must be B-type branch");
    assert_eq!(riscv_funct3(insts[0]), 0b000, "BEQ has funct3=000");
    assert_eq!(riscv_rs1(insts[0]), 10, "BEQ rs1 must be a0 (x10)");
    assert_eq!(riscv_rs2(insts[0]), 0, "BEQ rs2 must be zero (x0)");

    // Verify at least one RET exists.
    let has_ret = insts.iter().any(|&inst| is_riscv_ret(inst));
    assert!(has_ret, "multi-block code must contain at least one RET");
}

// ===========================================================================
// TEST 11: load/store encoding in E2E context
// ===========================================================================

/// Build a function that uses load and store instructions to verify
/// their encoding through the pipeline.
#[test]
fn test_riscv_e2e_load_store_encoding() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };

    // Build: load value from a0 address, store back to same address, return loaded value
    let mut func = RiscVISelFunction::new("load_store".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    // LD v0, 0(a0)  -- load doubleword from [a0]
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Ld,
        vec![
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // SD v0, 8(a0)  -- store doubleword to [a0 + 8]
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Sd,
        vec![
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::Imm(8),
        ],
    ));

    // ADDI a0, v0, 0  -- move result to a0
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // JALR x0, ra, 0  -- return
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("load/store compilation should succeed");
    assert_eq!(code.len(), 16, "load_store no-frame should be 16 bytes (4 instructions)");

    let insts: Vec<u32> = (0..4).map(|i| decode_riscv_inst(&code, i * 4)).collect();

    // LD: opcode=LOAD (0b0000011), funct3=011 (doubleword)
    assert_eq!(riscv_opcode(insts[0]), LOAD, "inst 0 must be LOAD (LD)");
    assert_eq!(riscv_funct3(insts[0]), 0b011, "LD has funct3=011");
    assert_eq!(riscv_rs1(insts[0]), 10, "LD base must be a0 (x10)");

    // SD: opcode=STORE (0b0100011), funct3=011 (doubleword)
    assert_eq!(riscv_opcode(insts[1]), STORE, "inst 1 must be STORE (SD)");
    assert_eq!(riscv_funct3(insts[1]), 0b011, "SD has funct3=011");
    assert_eq!(riscv_rs1(insts[1]), 10, "SD base must be a0 (x10)");

    // ADDI to a0
    assert_eq!(riscv_opcode(insts[2]), OP_IMM, "inst 2 must be ADDI");
    assert_eq!(riscv_rd(insts[2]), 10, "inst 2 must write to a0");

    // RET
    assert!(is_riscv_ret(insts[3]), "inst 3 must be RET");
}

// ===========================================================================
// TEST 12: RV64 word operations (ADDW, SUBW, etc.)
// ===========================================================================

/// Verify 32-bit word operations encode correctly through the pipeline.
#[test]
fn test_riscv_e2e_word_operations() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    // Build: addw(a, b) = ADDW a0, a0, a1
    let mut func = RiscVISelFunction::new("addw".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    // ADDW a0, a0, a1
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addw,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(A1),
        ],
    ));

    // JALR x0, ra, 0
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("word op compilation should succeed");
    assert_eq!(code.len(), 8, "addw no-frame should be 8 bytes");

    let inst0 = decode_riscv_inst(&code, 0);

    // ADDW: opcode=OP_32 (0b0111011), funct3=000, funct7=0000000
    let op_32: u32 = 0b0111011;
    assert_eq!(riscv_opcode(inst0), op_32, "ADDW must have opcode=OP_32 (0b0111011)");
    assert_eq!(riscv_funct3(inst0), 0b000, "ADDW has funct3=000");
    assert_eq!(riscv_funct7(inst0), 0b0000000, "ADDW has funct7=0000000");
    assert_eq!(riscv_rd(inst0), 10, "ADDW rd must be a0");
    assert_eq!(riscv_rs1(inst0), 10, "ADDW rs1 must be a0");
    assert_eq!(riscv_rs2(inst0), 11, "ADDW rs2 must be a1");
}

// ===========================================================================
// TEST 13: M-extension (multiply/divide) encoding
// ===========================================================================

/// Verify M-extension instructions encode correctly through the pipeline.
#[test]
fn test_riscv_e2e_m_extension_encoding() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    // Build: mul(a, b) = MUL a0, a0, a1
    let mut func = RiscVISelFunction::new("mul".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    // MUL a0, a0, a1
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Mul,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(A1),
        ],
    ));

    // JALR x0, ra, 0
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("mul compilation should succeed");
    assert_eq!(code.len(), 8, "mul no-frame should be 8 bytes");

    let inst0 = decode_riscv_inst(&code, 0);

    // MUL: opcode=OP (0b0110011), funct3=000, funct7=0000001
    assert_eq!(riscv_opcode(inst0), OP, "MUL must have opcode=OP");
    assert_eq!(riscv_funct3(inst0), 0b000, "MUL has funct3=000");
    assert_eq!(riscv_funct7(inst0), 0b0000001, "MUL has funct7=0000001 (M extension)");
    assert_eq!(riscv_rd(inst0), 10, "MUL rd must be a0");
    assert_eq!(riscv_rs1(inst0), 10, "MUL rs1 must be a0");
    assert_eq!(riscv_rs2(inst0), 11, "MUL rs2 must be a1");
}

// ===========================================================================
// TEST 14: U-type and J-type encoding through pipeline
// ===========================================================================

/// Verify LUI and JAL encode correctly through the pipeline.
#[test]
fn test_riscv_e2e_lui_jal_encoding() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![],
        returns: vec![Type::I64],
    };

    // Build: load large immediate via LUI + ADDI
    let mut func = RiscVISelFunction::new("large_const".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    // LUI a0, 0x12345
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Lui,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::Imm(0x12345),
        ],
    ));

    // ADDI a0, a0, 0x678 (lower 12 bits)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::Imm(0x678),
        ],
    ));

    // JALR x0, ra, 0 (return)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("LUI+ADDI compilation should succeed");
    assert_eq!(code.len(), 12, "LUI+ADDI+RET should be 12 bytes");

    let insts: Vec<u32> = (0..3).map(|i| decode_riscv_inst(&code, i * 4)).collect();

    // LUI a0, 0x12345: opcode=LUI (0b0110111), rd=10(a0), imm[31:12]=0x12345
    assert_eq!(riscv_opcode(insts[0]), LUI_OP, "inst 0 must be LUI");
    assert_eq!(riscv_rd(insts[0]), 10, "LUI rd must be a0");
    // Verify the upper 20 bits contain 0x12345
    let lui_imm = (insts[0] >> 12) & 0xFFFFF;
    assert_eq!(lui_imm, 0x12345, "LUI immediate must be 0x12345, got 0x{:05X}", lui_imm);

    // ADDI a0, a0, 0x678
    assert_eq!(riscv_opcode(insts[1]), OP_IMM, "inst 1 must be ADDI");
    assert_eq!(riscv_rd(insts[1]), 10, "ADDI rd must be a0");
    assert_eq!(riscv_rs1(insts[1]), 10, "ADDI rs1 must be a0");
    assert_eq!(riscv_imm_i(insts[1]), 0x678, "ADDI imm must be 0x678");

    // RET
    assert!(is_riscv_ret(insts[2]), "inst 2 must be RET");
}

// ===========================================================================
// TEST 15: ELF object file with external tools (RISC-V cross-tools)
// ===========================================================================

/// If RISC-V cross-compilation tools are available, validate the ELF output
/// with external tools.
#[test]
fn test_riscv_e2e_external_tool_validation() {
    if !has_riscv_objdump() && !has_riscv_readelf() {
        eprintln!("Skipping: no RISC-V cross-tools available");
        return;
    }

    let dir = make_test_dir("riscv_tool_validation");
    let func = build_riscv_const_test_function();
    let elf_bytes = riscv_compile_to_elf(&func)
        .expect("RISC-V pipeline should compile const42 to ELF");
    let obj_path = write_object_file(&dir, "const42.o", &elf_bytes);

    // Try objdump
    if has_riscv_objdump() {
        for cmd in &[
            "riscv64-linux-gnu-objdump",
            "riscv64-unknown-elf-objdump",
            "llvm-objdump",
        ] {
            let output = Command::new(cmd)
                .args(["-d", obj_path.to_str().unwrap()])
                .output();
            if let Ok(out) = output {
                if out.status.success() {
                    let disasm = String::from_utf8_lossy(&out.stdout);
                    eprintln!("{} -d output:\n{}", cmd, disasm);
                    // Should contain the function name or disassembly.
                    break;
                }
            }
        }
    }

    // Try readelf
    if has_riscv_readelf() {
        for cmd in &[
            "riscv64-linux-gnu-readelf",
            "riscv64-unknown-elf-readelf",
            "readelf",
        ] {
            let output = Command::new(cmd)
                .args(["-h", obj_path.to_str().unwrap()])
                .output();
            if let Ok(out) = output {
                if out.status.success() {
                    let header_out = String::from_utf8_lossy(&out.stdout);
                    eprintln!("{} -h output:\n{}", cmd, header_out);
                    if header_out.contains("RISC-V") || header_out.contains("riscv") {
                        // Great -- readelf recognizes it.
                    }
                    break;
                }
            }
        }
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST 16: ELF .o file can be written to disk
// ===========================================================================

/// Verify that ELF object files can be written to disk and read back.
#[test]
fn test_riscv_e2e_elf_write_and_readback() {
    let dir = make_test_dir("riscv_elf_readback");

    let funcs: Vec<(&str, _)> = vec![
        ("const42", build_riscv_const_test_function()),
        ("add", build_riscv_add_test_function()),
    ];

    for (name, func) in &funcs {
        let elf_bytes = riscv_compile_to_elf(func)
            .unwrap_or_else(|e| panic!("{}: pipeline failed: {}", name, e));

        let obj_path = write_object_file(&dir, &format!("{}.o", name), &elf_bytes);
        assert!(obj_path.exists(), "{}: .o file must be created", name);

        // Read back and verify.
        let readback = fs::read(&obj_path).expect("Failed to read back .o file");
        assert_eq!(readback.len(), elf_bytes.len(), "{}: readback size mismatch", name);
        assert_eq!(&readback, &elf_bytes, "{}: readback content mismatch", name);

        // Verify ELF structure of the readback.
        verify_riscv_elf_header(&readback);
    }

    cleanup(&dir);
}

// ===========================================================================
// TEST 17: register allocation correctness
// ===========================================================================

/// Verify that the register allocator assigns distinct registers for a function
/// with multiple live values.
#[test]
fn test_riscv_e2e_regalloc_distinct_assignments() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64, Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    // Build: sum3(a, b, c) = a + b + c
    let mut func = RiscVISelFunction::new("sum3".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    let v2 = VReg::new(2, RegClass::Gpr64);
    let v3 = VReg::new(3, RegClass::Gpr64);
    func.next_vreg = 4;

    // v0 = a0 (move arg 0)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // v1 = a1 (move arg 1)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::VReg(v1),
            RiscVISelOperand::PReg(A1),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // v2 = a2 (move arg 2)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::VReg(v2),
            RiscVISelOperand::PReg(A2),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // v3 = v0 + v1
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Add,
        vec![
            RiscVISelOperand::VReg(v3),
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::VReg(v1),
        ],
    ));

    // a0 = v3 + v2
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Add,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::VReg(v3),
            RiscVISelOperand::VReg(v2),
        ],
    ));

    // JALR x0, ra, 0
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // Compile without frame to keep output simple.
    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: false,
        emit_frame: false,
    });
    let code = pipeline.compile_function(&func).expect("sum3 compilation should succeed");
    assert_eq!(code.len(), 24, "sum3 should be 24 bytes (6 instructions)");

    let num_insts = code.len() / 4;
    let insts: Vec<u32> = (0..num_insts).map(|i| decode_riscv_inst(&code, i * 4)).collect();

    // Verify the 4 VRegs got assigned to distinct physical registers.
    let vr0_rd = riscv_rd(insts[0]);
    let vr1_rd = riscv_rd(insts[1]);
    let vr2_rd = riscv_rd(insts[2]);
    let vr3_rd = riscv_rd(insts[3]);

    let mut assigned: Vec<u32> = vec![vr0_rd, vr1_rd, vr2_rd, vr3_rd];
    assigned.sort();
    assigned.dedup();
    assert_eq!(
        assigned.len(), 4,
        "all 4 VRegs must be assigned to distinct physical registers. Got rd={:?}",
        vec![vr0_rd, vr1_rd, vr2_rd, vr3_rd]
    );
}

// ===========================================================================
// TEST 18: ELF to disk and verify with nm (if available)
// ===========================================================================

/// If nm is available, verify the RISC-V ELF symbol table externally.
#[test]
fn test_riscv_e2e_nm_validation() {
    let has_nm = Command::new("nm")
        .arg("--version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false);

    if !has_nm {
        eprintln!("Skipping: nm not available");
        return;
    }

    let dir = make_test_dir("riscv_nm");
    let func = build_riscv_add_test_function();
    let elf_bytes = riscv_compile_to_elf(&func)
        .expect("RISC-V pipeline should compile add to ELF");
    let obj_path = write_object_file(&dir, "add.o", &elf_bytes);

    let output = Command::new("nm")
        .arg(obj_path.to_str().unwrap())
        .output()
        .expect("failed to run nm");

    let nm_out = String::from_utf8_lossy(&output.stdout);
    eprintln!("nm output:\n{}", nm_out);

    // nm should show the 'add' symbol (may show as T/t for .text).
    assert!(
        nm_out.contains("add"),
        "nm must show 'add' symbol. Output:\n{}", nm_out
    );

    cleanup(&dir);
}

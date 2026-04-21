// llvm2-codegen/elf/constants.rs - ELF64 format constants
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: System V ABI, ELF-64 Object File Format
// Reference: AArch64 ELF ABI (ARM IHI 0056B)
// Reference: AMD64 ABI (System V AMD64 ABI)

//! ELF64 binary format constants.
//!
//! All magic numbers, machine types, section types, symbol bindings,
//! and relocation types needed to emit valid .o files for Linux
//! AArch64 and x86-64 targets.

// --- ELF identification ---

/// ELF magic byte 0.
pub const ELFMAG0: u8 = 0x7f;
/// ELF magic byte 1.
pub const ELFMAG1: u8 = b'E';
/// ELF magic byte 2.
pub const ELFMAG2: u8 = b'L';
/// ELF magic byte 3.
pub const ELFMAG3: u8 = b'F';

/// 64-bit ELF class.
pub const ELFCLASS64: u8 = 2;

/// Little-endian data encoding.
pub const ELFDATA2LSB: u8 = 1;

/// Current ELF version.
pub const EV_CURRENT: u8 = 1;

/// OS/ABI: UNIX System V.
pub const ELFOSABI_NONE: u8 = 0;

/// Size of e_ident array in ELF header.
pub const EI_NIDENT: usize = 16;

// --- ELF file types ---

/// Relocatable object file (.o).
pub const ET_REL: u16 = 1;

// --- Machine types ---

/// AMD x86-64 architecture.
pub const EM_X86_64: u16 = 62;

/// ARM AARCH64 architecture.
pub const EM_AARCH64: u16 = 183;

/// RISC-V architecture.
pub const EM_RISCV: u16 = 243;

// --- Special section indices ---

/// Undefined section index.
pub const SHN_UNDEF: u16 = 0;

/// Absolute symbol (not relative to any section).
pub const SHN_ABS: u16 = 0xFFF1;

/// Common symbol (allocated at link time).
pub const SHN_COMMON: u16 = 0xFFF2;

// --- Section types ---

/// Inactive section header.
pub const SHT_NULL: u32 = 0;

/// Program-defined data.
pub const SHT_PROGBITS: u32 = 1;

/// Symbol table.
pub const SHT_SYMTAB: u32 = 2;

/// String table.
pub const SHT_STRTAB: u32 = 3;

/// Relocation entries with addends.
pub const SHT_RELA: u32 = 4;

/// Uninitialized data (BSS).
pub const SHT_NOBITS: u32 = 8;

// --- Section flags ---

/// Section is writable at runtime.
pub const SHF_WRITE: u64 = 0x1;

/// Section occupies memory during execution.
pub const SHF_ALLOC: u64 = 0x2;

/// Section contains executable machine instructions.
pub const SHF_EXECINSTR: u64 = 0x4;

/// Section contains information for the linker (e.g., .rela.text has SHF_INFO_LINK).
pub const SHF_INFO_LINK: u64 = 0x40;

// --- Symbol binding (upper 4 bits of st_info) ---

/// Local symbol — not visible outside the object file.
pub const STB_LOCAL: u8 = 0;

/// Global symbol — visible to all object files being combined.
pub const STB_GLOBAL: u8 = 1;

/// Weak symbol — like global, but may be overridden.
pub const STB_WEAK: u8 = 2;

// --- Symbol types (lower 4 bits of st_info) ---

/// Symbol type not specified.
pub const STT_NOTYPE: u8 = 0;

/// Symbol is a data object (variable, array, etc.).
pub const STT_OBJECT: u8 = 1;

/// Symbol is a function entry point.
pub const STT_FUNC: u8 = 2;

/// Symbol is a section.
pub const STT_SECTION: u8 = 3;

/// Symbol is a file name.
pub const STT_FILE: u8 = 4;

// --- Symbol visibility (lower 2 bits of st_other) ---

/// Default visibility — determined by binding type.
pub const STV_DEFAULT: u8 = 0;

/// Hidden visibility — not visible to other components.
pub const STV_HIDDEN: u8 = 2;

// --- AArch64 ELF relocation types ---
// Reference: ARM IHI 0056B, "ELF for the Arm 64-bit Architecture"

/// No relocation.
pub const R_AARCH64_NONE: u32 = 0;

/// S + A (64-bit absolute address).
pub const R_AARCH64_ABS64: u32 = 257;

/// S + A (32-bit absolute address).
pub const R_AARCH64_ABS32: u32 = 258;

/// Page(S + A) - Page(P) (ADRP instruction).
pub const R_AARCH64_ADR_PREL_PG_HI21: u32 = 275;

/// (S + A) & 0xFFF (ADD instruction, 12-bit page offset).
pub const R_AARCH64_ADD_ABS_LO12_NC: u32 = 277;

/// S + A - P (B/BL 26-bit PC-relative branch).
pub const R_AARCH64_JUMP26: u32 = 282;

/// S + A - P (BL 26-bit PC-relative call).
pub const R_AARCH64_CALL26: u32 = 283;

/// (S + A) & 0xFFF (LDR/STR 8-bit unsigned offset).
pub const R_AARCH64_LDST8_ABS_LO12_NC: u32 = 278;

/// (S + A) & 0xFFF >> 1 (LDR/STR 16-bit unsigned offset).
pub const R_AARCH64_LDST16_ABS_LO12_NC: u32 = 284;

/// (S + A) & 0xFFF >> 2 (LDR/STR 32-bit unsigned offset).
pub const R_AARCH64_LDST32_ABS_LO12_NC: u32 = 285;

/// (S + A) & 0xFFF >> 3 (LDR/STR 64-bit unsigned offset).
pub const R_AARCH64_LDST64_ABS_LO12_NC: u32 = 286;

/// (S + A) & 0xFFF >> 4 (LDR/STR 128-bit unsigned offset).
pub const R_AARCH64_LDST128_ABS_LO12_NC: u32 = 299;

/// Page(G(S)) - Page(P) (ADRP to GOT entry).
pub const R_AARCH64_ADR_GOT_PAGE: u32 = 311;

/// G(S) & 0xFFF (LD64 from GOT entry, 12-bit page offset).
pub const R_AARCH64_LD64_GOT_LO12_NC: u32 = 312;

/// Page(G(S)) - Page(P) (ADRP to TLS IE GOT entry).
pub const R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21: u32 = 0x21d;

/// G(S) & 0xFFF (LD64 from TLS IE GOT entry).
pub const R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC: u32 = 0x21e;

// --- x86-64 ELF relocation types ---
// Reference: System V AMD64 ABI, "Relocation Types"

/// No relocation.
pub const R_X86_64_NONE: u32 = 0;

/// S + A (64-bit absolute address).
pub const R_X86_64_64: u32 = 1;

/// S + A - P (32-bit PC-relative).
pub const R_X86_64_PC32: u32 = 2;

/// G + A (32-bit GOT offset).
pub const R_X86_64_GOT32: u32 = 3;

/// L + A - P (32-bit PLT-relative).
pub const R_X86_64_PLT32: u32 = 4;

/// (zero-extended) S + A (32-bit unsigned absolute).
pub const R_X86_64_32: u32 = 10;

/// (sign-extended) S + A (32-bit signed absolute).
pub const R_X86_64_32S: u32 = 11;

/// S + A (16-bit absolute).
pub const R_X86_64_16: u32 = 12;

/// S + A - P (16-bit PC-relative).
pub const R_X86_64_PC16: u32 = 13;

/// S + A (8-bit absolute).
pub const R_X86_64_8: u32 = 14;

/// S + A - P (8-bit PC-relative).
pub const R_X86_64_PC8: u32 = 15;

/// G + GOT + A - P (32-bit GOT-relative PC offset).
pub const R_X86_64_GOTPCREL: u32 = 9;

/// Relaxable R_X86_64_GOTPCREL.
pub const R_X86_64_GOTPCRELX: u32 = 41;

/// Relaxable R_X86_64_GOTPCREL with REX prefix.
pub const R_X86_64_REX_GOTPCRELX: u32 = 42;

// --- Struct sizes ---

/// Size of the ELF64 file header in bytes.
pub const ELF64_EHDR_SIZE: usize = 64;

/// Size of an ELF64 section header entry in bytes.
pub const ELF64_SHDR_SIZE: usize = 64;

/// Size of an ELF64 symbol table entry in bytes.
pub const ELF64_SYM_SIZE: usize = 24;

/// Size of an ELF64 relocation entry with addend in bytes.
pub const ELF64_RELA_SIZE: usize = 24;

/// Encode st_info from binding and type.
pub const fn elf64_st_info(bind: u8, typ: u8) -> u8 {
    (bind << 4) | (typ & 0xf)
}

/// Extract binding from st_info.
pub const fn elf64_st_bind(info: u8) -> u8 {
    info >> 4
}

/// Extract type from st_info.
pub const fn elf64_st_type(info: u8) -> u8 {
    info & 0xf
}

/// Encode r_info from symbol index and relocation type.
pub const fn elf64_r_info(sym: u32, typ: u32) -> u64 {
    ((sym as u64) << 32) | (typ as u64)
}

/// Extract symbol index from r_info.
pub const fn elf64_r_sym(info: u64) -> u32 {
    (info >> 32) as u32
}

/// Extract relocation type from r_info.
pub const fn elf64_r_type(info: u64) -> u32 {
    info as u32
}

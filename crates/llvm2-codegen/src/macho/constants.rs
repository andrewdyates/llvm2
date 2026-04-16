// llvm2-codegen/macho/constants.rs - Mach-O format constants
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/mach-o/loader.h
// Reference: /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/mach-o/arm64/reloc.h

//! Mach-O binary format constants.
//!
//! All magic numbers, CPU types, load command types, section types, and
//! ARM64 relocation types needed to emit valid .o files for macOS.

// --- Mach-O magic numbers ---

/// 64-bit Mach-O magic number (little-endian).
pub const MH_MAGIC_64: u32 = 0xFEED_FACF;

// --- CPU types ---

/// Mask for 64-bit ABI flag.
pub const CPU_ARCH_ABI64: u32 = 0x0100_0000;

/// ARM CPU type (without ABI64 flag).
pub const CPU_TYPE_ARM: u32 = 12;

/// ARM64 CPU type = CPU_TYPE_ARM | CPU_ARCH_ABI64.
pub const CPU_TYPE_ARM64: u32 = CPU_TYPE_ARM | CPU_ARCH_ABI64;

/// x86 CPU type (without ABI64 flag).
pub const CPU_TYPE_X86: u32 = 7;

/// x86-64 CPU type = CPU_TYPE_X86 | CPU_ARCH_ABI64.
pub const CPU_TYPE_X86_64: u32 = CPU_TYPE_X86 | CPU_ARCH_ABI64;

/// ARM64 all subtypes.
pub const CPU_SUBTYPE_ARM64_ALL: u32 = 0;

/// x86-64 all subtypes.
pub const CPU_SUBTYPE_X86_64_ALL: u32 = 3;

// --- File types ---

/// Relocatable object file (.o).
pub const MH_OBJECT: u32 = 0x1;

// --- Mach-O header flags ---

/// Safe to divide sections into sub-sections via symbols for dead code stripping.
pub const MH_SUBSECTIONS_VIA_SYMBOLS: u32 = 0x2000;

// --- Load command types ---

/// 64-bit segment load command.
pub const LC_SEGMENT_64: u32 = 0x19;

/// Symbol table load command.
pub const LC_SYMTAB: u32 = 0x02;

/// Dynamic symbol table load command.
pub const LC_DYSYMTAB: u32 = 0x0B;

/// Build version load command.
pub const LC_BUILD_VERSION: u32 = 0x32;

// --- Build version platforms ---

/// macOS platform identifier.
pub const PLATFORM_MACOS: u32 = 1;

// --- Section types (lower 8 bits of section flags) ---

/// Regular section.
pub const S_REGULAR: u32 = 0x0;

/// Zero-fill on demand section.
pub const S_ZEROFILL: u32 = 0x1;

/// Section with only literal C strings.
pub const S_CSTRING_LITERALS: u32 = 0x2;

// --- Section attributes (upper 24 bits of section flags) ---

/// Section contains only true machine instructions.
pub const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x8000_0000;

/// Section contains some machine instructions.
pub const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x0000_0400;

// --- VM protection flags ---

/// Read permission.
pub const VM_PROT_READ: i32 = 0x01;

/// Write permission.
pub const VM_PROT_WRITE: i32 = 0x02;

/// Execute permission.
pub const VM_PROT_EXECUTE: i32 = 0x04;

/// Read + Write + Execute.
pub const VM_PROT_ALL: i32 = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;

// --- Symbol table constants ---

/// External symbol bit.
pub const N_EXT: u8 = 0x01;

/// Defined in section number n_sect.
pub const N_SECT: u8 = 0x0E;

/// Undefined symbol.
pub const N_UNDF: u8 = 0x00;

/// Private external symbol bit.
pub const N_PEXT: u8 = 0x10;

/// Mask for the type bits.
pub const N_TYPE: u8 = 0x0E;

// --- ARM64 relocation types ---
// Reference: enum reloc_type_arm64 in mach-o/arm64/reloc.h

/// Pointer-sized fixup (absolute address).
pub const ARM64_RELOC_UNSIGNED: u32 = 0;

/// Must be followed by ARM64_RELOC_UNSIGNED.
pub const ARM64_RELOC_SUBTRACTOR: u32 = 1;

/// B/BL instruction with 26-bit displacement.
pub const ARM64_RELOC_BRANCH26: u32 = 2;

/// PC-relative distance to page of target (ADRP).
pub const ARM64_RELOC_PAGE21: u32 = 3;

/// Offset within page, scaled by r_length (LDR/STR/ADD).
pub const ARM64_RELOC_PAGEOFF12: u32 = 4;

/// PC-relative distance to page of GOT slot.
pub const ARM64_RELOC_GOT_LOAD_PAGE21: u32 = 5;

/// Offset within page of GOT slot, scaled by r_length.
pub const ARM64_RELOC_GOT_LOAD_PAGEOFF12: u32 = 6;

/// Pointer to GOT slot.
pub const ARM64_RELOC_POINTER_TO_GOT: u32 = 7;

/// PC-relative distance to page of TLVP slot.
pub const ARM64_RELOC_TLVP_LOAD_PAGE21: u32 = 8;

/// Offset within page of TLVP slot, scaled by r_length.
pub const ARM64_RELOC_TLVP_LOAD_PAGEOFF12: u32 = 9;

/// Addend for PAGE21/PAGEOFF12/BRANCH26.
pub const ARM64_RELOC_ADDEND: u32 = 10;

/// Authenticated pointer (arm64e).
pub const ARM64_RELOC_AUTHENTICATED_POINTER: u32 = 11;

// --- x86-64 relocation types ---
// Reference: enum reloc_type_x86_64 in mach-o/x86_64/reloc.h

/// For absolute addresses (64-bit or 32-bit absolute).
pub const X86_64_RELOC_UNSIGNED: u32 = 0;

/// For signed 32-bit displacement (RIP-relative data access).
pub const X86_64_RELOC_SIGNED: u32 = 1;

/// A CALL/JMP instruction with 32-bit displacement.
pub const X86_64_RELOC_BRANCH: u32 = 2;

/// A MOVQ load of a GOT entry.
pub const X86_64_RELOC_GOT_LOAD: u32 = 3;

/// Other GOT references.
pub const X86_64_RELOC_GOT: u32 = 4;

/// Must be followed by X86_64_RELOC_UNSIGNED (used for symbol difference).
pub const X86_64_RELOC_SUBTRACTOR: u32 = 5;

/// For signed 32-bit displacement with a -1 addend.
pub const X86_64_RELOC_SIGNED_1: u32 = 6;

/// For signed 32-bit displacement with a -2 addend.
pub const X86_64_RELOC_SIGNED_2: u32 = 7;

/// For signed 32-bit displacement with a -4 addend.
pub const X86_64_RELOC_SIGNED_4: u32 = 8;

/// For thread-local variables.
pub const X86_64_RELOC_TLV: u32 = 9;

// --- Relocation info encoding helpers ---

/// Relocation length: byte (1 byte).
pub const RELOC_LENGTH_BYTE: u32 = 0;

/// Relocation length: word (2 bytes).
pub const RELOC_LENGTH_WORD: u32 = 1;

/// Relocation length: long (4 bytes).
pub const RELOC_LENGTH_LONG: u32 = 2;

/// Relocation length: quad (8 bytes).
pub const RELOC_LENGTH_QUAD: u32 = 3;

// --- Struct sizes ---

/// Size of mach_header_64 in bytes.
pub const MACH_HEADER_64_SIZE: u32 = 32;

/// Size of segment_command_64 in bytes (without sections).
pub const SEGMENT_COMMAND_64_SIZE: u32 = 72;

/// Size of section_64 in bytes.
pub const SECTION_64_SIZE: u32 = 80;

/// Size of symtab_command in bytes.
pub const SYMTAB_COMMAND_SIZE: u32 = 24;

/// Size of dysymtab_command in bytes.
pub const DYSYMTAB_COMMAND_SIZE: u32 = 80;

/// Size of build_version_command in bytes (without tool entries).
pub const BUILD_VERSION_COMMAND_SIZE: u32 = 24;

/// Size of nlist_64 in bytes.
pub const NLIST_64_SIZE: u32 = 16;

/// Size of relocation_info in bytes.
pub const RELOCATION_INFO_SIZE: u32 = 8;

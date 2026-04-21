// llvm2-codegen/elf/mod.rs - ELF64 object file support
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! ELF64 object file writer for Linux AArch64 and x86-64.
//!
//! This module emits valid ELF .o files that the Linux system linker (ld)
//! can consume. It handles the complete object file format:
//!
//! - File header (Elf64_Ehdr)
//! - Section data (.text, .data, .bss)
//! - Symbol table (.symtab) with Elf64_Sym entries
//! - String tables (.strtab, .shstrtab)
//! - Relocation entries with addends (.rela.text) using Elf64_Rela
//! - Section header table (Elf64_Shdr entries)
//!
//! Supports both AArch64 and x86-64 targets with architecture-specific
//! relocation types.
//!
//! All data is written in little-endian byte order (both targets are LE).

pub mod constants;
pub mod debug;
pub mod header;
pub mod reloc;
pub mod section;
pub mod symbol;
pub mod writer;

pub use debug::{
    DwarfDebugStubs, NoteGnuStack, ProgramHeaderStub, SectionGroup,
};
pub use header::ElfMachine;
pub use reloc::{AArch64RelocType, Elf64Rela, X86_64RelocType};
pub use symbol::{Elf64Sym, ElfStringTable};
pub use writer::{ElfSymbol, ElfWriter};

// llvm2-codegen/macho/mod.rs - Mach-O object file support
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Mach-O 64-bit object file writer for macOS (AArch64 and x86-64).
//!
//! This module emits valid Mach-O .o files that the macOS system linker (ld)
//! can consume. It handles the complete object file format:
//!
//! - File header (mach_header_64) with target-aware CPU type
//! - Load commands (LC_SEGMENT_64, LC_BUILD_VERSION, LC_SYMTAB, LC_DYSYMTAB)
//! - Section data (__text, __data, custom sections)
//! - Relocation entries (ARM64 and x86-64 relocation types)
//! - Symbol table (nlist_64) with proper local/global partitioning
//! - String table
//! - Deferred fixup layer for late-bound address resolution
//!
//! All data is written in little-endian byte order (native for both ARM64
//! and x86-64 on macOS).

pub mod constants;
pub mod fixup;
pub mod header;
pub mod linker;
pub mod reloc;
pub mod section;
pub mod symbol;
pub mod writer;
pub mod x86_64_reloc;

pub use fixup::{Fixup, FixupError, FixupList, FixupTarget};
pub use reloc::{encode_relocation, AArch64RelocKind, Relocation};
pub use symbol::{DysymtabParams, NList64, SymbolTable};
pub use writer::{MachORelocation, MachOTarget, MachOWriter, Symbol};
pub use x86_64_reloc::{
    encode_x86_64_relocation, decode_x86_64_relocation,
    X86_64RelocKind, X86_64Relocation,
};

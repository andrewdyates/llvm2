// llvm2-codegen/macho/mod.rs - Mach-O object file support
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Mach-O 64-bit object file writer for macOS/ARM64.
//!
//! This module emits valid Mach-O .o files that the macOS system linker (ld)
//! can consume. It handles the complete object file format:
//!
//! - File header (mach_header_64)
//! - Load commands (LC_SEGMENT_64, LC_BUILD_VERSION, LC_SYMTAB, LC_DYSYMTAB)
//! - Section data (__text, __data, custom sections)
//! - Relocation entries (ARM64 relocation types)
//! - Symbol table (nlist_64) with proper local/global partitioning
//! - String table
//! - Deferred fixup layer for late-bound address resolution
//!
//! All data is written in little-endian byte order (ARM64 native).

pub mod constants;
pub mod fixup;
pub mod header;
pub mod reloc;
pub mod section;
pub mod symbol;
pub mod writer;

pub use fixup::{Fixup, FixupList, FixupTarget};
pub use reloc::{encode_relocation, AArch64RelocKind, Relocation};
pub use symbol::{DysymtabParams, NList64, SymbolTable};
pub use writer::{MachOWriter, Symbol};

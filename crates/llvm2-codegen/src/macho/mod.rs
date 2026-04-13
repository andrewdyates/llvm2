// llvm2-codegen/macho/mod.rs - Mach-O object file writer
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
//!
//! All data is written in little-endian byte order (ARM64 native).

pub mod constants;
pub mod header;
pub mod section;
pub mod writer;

pub use writer::{MachOWriter, Relocation, Symbol};

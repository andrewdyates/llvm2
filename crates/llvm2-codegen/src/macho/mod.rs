// llvm2-codegen/macho/mod.rs - Mach-O object file support
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Mach-O object file format support for ARM64 (Apple Silicon).
//!
//! This module provides the building blocks for emitting Mach-O `.o` files:
//!
//! - [`reloc`] — ARM64 relocation encoding (`relocation_info` structs)
//! - [`symbol`] — Symbol table (`nlist_64`) and string table emission
//! - [`fixup`] — Deferred fixup layer for late-bound address resolution

pub mod fixup;
pub mod reloc;
pub mod symbol;

pub use fixup::{Fixup, FixupList, FixupTarget};
pub use reloc::{encode_relocation, AArch64RelocKind, Relocation};
pub use symbol::{DysymtabParams, NList64, SymbolTable};

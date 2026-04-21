// llvm2-codegen/macho/linker.rs - Mach-O linker: read .o files, resolve symbols, emit executables
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Minimal Mach-O linker for LLVM2-generated object files.
//!
//! Reads Mach-O MH_OBJECT (.o) files, resolves symbols across objects, applies
//! relocations (BRANCH26, PAGE21, PAGEOFF12), and emits a Mach-O MH_EXECUTE
//! binary.
//!
//! This is an MVP linker for the LLVM2 pipeline. It handles the common case of
//! linking a small number of .o files into a static executable for AArch64 macOS.
//!
//! # Layout of emitted MH_EXECUTE
//!
//! ```text
//! __PAGEZERO  vmaddr=0x0            vmsize=0x1_0000_0000  (no file data)
//! __TEXT      vmaddr=0x1_0000_0000  (rx) contains __text
//! __DATA      vmaddr=aligned after __TEXT  (rw) contains __data
//! ```

use std::collections::{HashMap, HashSet};

use super::constants::*;
use super::reloc::{decode_relocation, AArch64RelocKind, Relocation};
use super::section::padded_name;
use super::symbol::NList64;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants for executable emission
// ---------------------------------------------------------------------------

/// Mach-O executable file type.
const MH_EXECUTE: u32 = 0x2;

/// Position-independent executable flag.
const MH_PIE: u32 = 0x0020_0000;

/// LC_MAIN load command type (entry point for executables).
const LC_MAIN: u32 = 0x8000_0028;

/// Size of the LC_MAIN (entry_point_command) load command in bytes.
const LC_MAIN_SIZE: u32 = 24;

/// Default base virtual address for macOS AArch64 executables.
const DEFAULT_BASE_ADDR: u64 = 0x1_0000_0000;

/// Page size for AArch64 macOS (16 KiB).
const PAGE_SIZE: u64 = 0x4000;

/// LC_LOAD_DYLIB load command type.
const LC_LOAD_DYLIB: u32 = 0x0C;

/// Size of the dylib_command header (without the name string).
/// cmd(4) + cmdsize(4) + name_offset(4) + timestamp(4) + current_version(4) + compat_version(4)
const LC_LOAD_DYLIB_HEADER_SIZE: u32 = 24;

/// LC_LOAD_DYLINKER load command type (modern macOS requires this for dyld invocation).
const LC_LOAD_DYLINKER: u32 = 0x0E;

/// Size of the dylinker_command header (without the name string).
/// cmd(4) + cmdsize(4) + name_offset(4)
const LC_LOAD_DYLINKER_HEADER_SIZE: u32 = 12;

/// Standard path to macOS dynamic linker.
const DYLD_PATH: &str = "/usr/lib/dyld";

/// Section type for lazy symbol pointers (__la_symbol_ptr).
/// Used for lazy-binding stubs (future: dyld_stub_binder integration).
#[allow(dead_code)]
const S_LAZY_SYMBOL_POINTERS: u32 = 0x7;

/// Section type for non-lazy symbol pointers (__got / __nl_symbol_ptr).
const S_NON_LAZY_SYMBOL_POINTERS: u32 = 0x6;

/// Section type for symbol stubs (__stubs).
const S_SYMBOL_STUBS: u32 = 0x8;

/// Size of a single AArch64 stub entry (3 instructions: ADRP + LDR + BR = 12 bytes).
const STUB_SIZE: u32 = 12;

/// Weak definition flag in n_desc field.
const N_WEAK_DEF: u16 = 0x0080;

/// Weak reference flag in n_desc field.
const N_WEAK_REF: u16 = 0x0040;

/// No dead strip flag in n_desc field.
const N_NO_DEAD_STRIP: u16 = 0x0020;

/// Section type mask (lower 8 bits of section flags).
const SECTION_TYPE_MASK: u32 = 0x0000_00FF;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during linking.
#[derive(Debug, Error)]
pub enum LinkerError {
    /// The input data is too short to contain a valid Mach-O header.
    #[error("data too short: need at least {expected} bytes, got {actual}")]
    TooShort { expected: usize, actual: usize },

    /// The Mach-O magic number is incorrect.
    #[error("bad magic: expected 0xFEEDFACF, got 0x{0:08X}")]
    BadMagic(u32),

    /// The file type is not MH_OBJECT.
    #[error("expected MH_OBJECT (0x1), got 0x{0:X}")]
    NotObject(u32),

    /// A load command extends beyond the declared sizeofcmds.
    #[error("load command at offset {offset} extends beyond load command area")]
    LoadCommandOverflow { offset: usize },

    /// Failed to decode a relocation entry.
    #[error("relocation decode error in section {section}: {detail}")]
    RelocDecode { section: String, detail: String },

    /// An undefined symbol could not be resolved.
    #[error("undefined symbol: {0}")]
    UndefinedSymbol(String),

    /// Duplicate symbol definition.
    #[error("duplicate symbol: {0}")]
    DuplicateSymbol(String),

    /// A relocation type is not yet supported by the linker.
    #[error("unsupported relocation type: {0:?}")]
    UnsupportedRelocation(AArch64RelocKind),

    /// A relocation target is out of range for the instruction encoding.
    #[error("relocation overflow: {detail}")]
    RelocationOverflow { detail: String },

    /// No _main entry point found.
    #[error("no _main entry point found")]
    NoEntryPoint,

    /// Input file is malformed (general parsing error with context).
    #[error("malformed input '{file}': {detail}")]
    MalformedInput { file: String, detail: String },

    /// Section data extends beyond file bounds.
    #[error("section '{section}' data at offset {offset:#x} extends beyond file (size {file_size:#x})")]
    SectionDataOverflow {
        section: String,
        offset: usize,
        file_size: usize,
    },

    /// Multiple strong definitions of the same symbol (detailed variant).
    #[error("duplicate symbol '{name}' (first defined in object {first_obj}, also in object {second_obj})")]
    DuplicateSymbolDetailed {
        name: String,
        first_obj: usize,
        second_obj: usize,
    },
}

// ---------------------------------------------------------------------------
// Parsed object file structures
// ---------------------------------------------------------------------------

/// A parsed Mach-O section from an object file.
#[derive(Debug, Clone)]
pub struct ParsedSection {
    /// Section name (e.g., "__text").
    pub name: String,
    /// Segment name (e.g., "__TEXT").
    pub segment: String,
    /// Raw section data bytes.
    pub data: Vec<u8>,
    /// Virtual address in the object file (usually 0-based for .o files).
    pub addr: u64,
    /// Alignment as power of 2.
    pub align: u32,
    /// Section flags.
    pub flags: u32,
    /// Relocations that apply to this section.
    pub relocations: Vec<Relocation>,
    /// Virtual size (for zerofill sections, may exceed data.len()).
    pub vmsize: u64,
}

impl ParsedSection {
    /// Returns the section type (lower 8 bits of flags).
    pub fn section_type(&self) -> u32 {
        self.flags & SECTION_TYPE_MASK
    }

    /// Returns true if this is a zerofill section (__bss).
    pub fn is_zerofill(&self) -> bool {
        self.section_type() == S_ZEROFILL
    }

    /// Returns the effective size of this section in virtual memory.
    /// For zerofill sections, this is vmsize; for regular sections, data length.
    pub fn effective_size(&self) -> u64 {
        if self.is_zerofill() {
            self.vmsize
        } else {
            self.data.len() as u64
        }
    }
}

/// A parsed symbol from a Mach-O object file.
#[derive(Debug, Clone)]
pub struct ParsedSymbol {
    /// Symbol name from the string table.
    pub name: String,
    /// n_type field from nlist_64.
    pub n_type: u8,
    /// Section number (1-based, 0 = undefined).
    pub section: u8,
    /// n_desc field.
    pub desc: u16,
    /// Symbol value (address/offset).
    pub value: u64,
}

impl ParsedSymbol {
    /// Returns true if this symbol is defined (in a section).
    pub fn is_defined(&self) -> bool {
        (self.n_type & N_TYPE) == N_SECT
    }

    /// Returns true if this symbol is undefined.
    pub fn is_undefined(&self) -> bool {
        (self.n_type & N_TYPE) == N_UNDF && self.section == 0
    }

    /// Returns true if this symbol is external.
    pub fn is_external(&self) -> bool {
        (self.n_type & N_EXT) != 0
    }

    /// Returns true if this symbol is a weak definition.
    pub fn is_weak_def(&self) -> bool {
        self.desc & N_WEAK_DEF != 0
    }

    /// Returns true if this symbol is a weak reference (undefined weak).
    pub fn is_weak_ref(&self) -> bool {
        self.desc & N_WEAK_REF != 0
    }

    /// Returns true if this symbol should not be dead-stripped.
    pub fn is_no_dead_strip(&self) -> bool {
        self.desc & N_NO_DEAD_STRIP != 0
    }
}

/// A fully parsed Mach-O object file.
#[derive(Debug, Clone)]
pub struct ParsedObject {
    /// CPU type from the header.
    pub cputype: u32,
    /// CPU subtype from the header.
    pub cpusubtype: u32,
    /// Header flags.
    pub flags: u32,
    /// Parsed sections.
    pub sections: Vec<ParsedSection>,
    /// Parsed symbols.
    pub symbols: Vec<ParsedSymbol>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Read a little-endian u32 from a byte slice at the given offset.
fn read_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Read a little-endian u64 from a byte slice at the given offset.
fn read_u64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
        data[off + 4],
        data[off + 5],
        data[off + 6],
        data[off + 7],
    ])
}

/// Read a NUL-terminated string from a byte slice starting at `off`.
fn read_cstring(data: &[u8], off: usize) -> String {
    let mut end = off;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    String::from_utf8_lossy(&data[off..end]).into_owned()
}

/// Read a fixed-size name field (16 bytes, NUL-padded) and return a trimmed string.
fn read_name16(data: &[u8], off: usize) -> String {
    let raw = &data[off..off + 16];
    let end = raw.iter().position(|&b| b == 0).unwrap_or(16);
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// Parser for Mach-O MH_OBJECT files.
pub struct MachOParser;

impl MachOParser {
    /// Parse a Mach-O .o file from raw bytes.
    pub fn parse(data: &[u8]) -> Result<ParsedObject, LinkerError> {
        // --- Validate header ---
        let hdr_size = MACH_HEADER_64_SIZE as usize;
        if data.len() < hdr_size {
            return Err(LinkerError::TooShort {
                expected: hdr_size,
                actual: data.len(),
            });
        }

        let magic = read_u32(data, 0);
        if magic != MH_MAGIC_64 {
            return Err(LinkerError::BadMagic(magic));
        }

        let cputype = read_u32(data, 4);
        let cpusubtype = read_u32(data, 8);
        let filetype = read_u32(data, 12);
        if filetype != MH_OBJECT {
            return Err(LinkerError::NotObject(filetype));
        }

        let ncmds = read_u32(data, 16);
        let sizeofcmds = read_u32(data, 20);
        let flags = read_u32(data, 24);

        // --- Walk load commands ---
        let mut sections = Vec::new();
        let mut symbols = Vec::new();

        let lc_start = hdr_size;
        let lc_end = lc_start + sizeofcmds as usize;
        let mut offset = lc_start;

        for _ in 0..ncmds {
            if offset + 8 > lc_end || offset + 8 > data.len() {
                return Err(LinkerError::LoadCommandOverflow { offset });
            }

            let cmd = read_u32(data, offset);
            let cmdsize = read_u32(data, offset + 4) as usize;

            if cmdsize < 8 || offset + cmdsize > data.len() {
                return Err(LinkerError::LoadCommandOverflow { offset });
            }

            match cmd {
                LC_SEGMENT_64 => {
                    // Parse segment command header to find nsects.
                    // segment_command_64 layout:
                    //   cmd(4) + cmdsize(4) + segname(16) + vmaddr(8) + vmsize(8) +
                    //   fileoff(8) + filesize(8) + maxprot(4) + initprot(4) + nsects(4) + flags(4)
                    let nsects = read_u32(data, offset + 64);

                    // Parse each section_64 header.
                    let mut sec_offset = offset + SEGMENT_COMMAND_64_SIZE as usize;
                    for _ in 0..nsects {
                        let sec_size = SECTION_64_SIZE as usize;
                        if sec_offset + sec_size > data.len() {
                            return Err(LinkerError::LoadCommandOverflow {
                                offset: sec_offset,
                            });
                        }

                        let sec_name = read_name16(data, sec_offset);
                        let seg_name = read_name16(data, sec_offset + 16);
                        let sec_addr = read_u64(data, sec_offset + 32);
                        let sec_data_size = read_u64(data, sec_offset + 40) as usize;
                        let sec_file_offset = read_u32(data, sec_offset + 48) as usize;
                        let sec_align = read_u32(data, sec_offset + 52);
                        let sec_reloff = read_u32(data, sec_offset + 56) as usize;
                        let sec_nreloc = read_u32(data, sec_offset + 60);
                        let sec_flags = read_u32(data, sec_offset + 64);

                        // Read section data.
                        let sec_data = if sec_data_size > 0
                            && sec_file_offset + sec_data_size <= data.len()
                        {
                            data[sec_file_offset..sec_file_offset + sec_data_size].to_vec()
                        } else {
                            vec![0u8; sec_data_size]
                        };

                        // Read relocations for this section.
                        let mut relocations = Vec::new();
                        let reloc_size = RELOCATION_INFO_SIZE as usize;
                        for r in 0..sec_nreloc as usize {
                            let roff = sec_reloff + r * reloc_size;
                            if roff + reloc_size <= data.len() {
                                let reloc_bytes: [u8; 8] = [
                                    data[roff],
                                    data[roff + 1],
                                    data[roff + 2],
                                    data[roff + 3],
                                    data[roff + 4],
                                    data[roff + 5],
                                    data[roff + 6],
                                    data[roff + 7],
                                ];
                                match decode_relocation(&reloc_bytes) {
                                    Ok(reloc) => relocations.push(reloc),
                                    Err(e) => {
                                        return Err(LinkerError::RelocDecode {
                                            section: sec_name.clone(),
                                            detail: e.to_string(),
                                        });
                                    }
                                }
                            }
                        }

                        // For zerofill sections (BSS), vmsize is the declared
                        // size but data is empty (no file backing).
                        let vmsize = sec_data_size as u64;

                        sections.push(ParsedSection {
                            name: sec_name,
                            segment: seg_name,
                            data: sec_data,
                            addr: sec_addr,
                            align: sec_align,
                            flags: sec_flags,
                            relocations,
                            vmsize,
                        });

                        sec_offset += sec_size;
                    }
                }
                LC_SYMTAB => {
                    // symtab_command layout:
                    //   cmd(4) + cmdsize(4) + symoff(4) + nsyms(4) + stroff(4) + strsize(4)
                    let symoff = read_u32(data, offset + 8) as usize;
                    let nsyms = read_u32(data, offset + 12) as usize;
                    let stroff = read_u32(data, offset + 16) as usize;
                    let _strsize = read_u32(data, offset + 20) as usize;

                    let nlist_size = NLIST_64_SIZE as usize;
                    for i in 0..nsyms {
                        let sym_off = symoff + i * nlist_size;
                        if sym_off + nlist_size <= data.len() {
                            let nlist_bytes: [u8; 16] = data[sym_off..sym_off + 16]
                                .try_into()
                                .expect("nlist_64 slice");
                            let nlist = NList64::decode(&nlist_bytes);

                            // Read the symbol name from the string table.
                            let name = if (stroff + nlist.strx as usize) < data.len() {
                                read_cstring(data, stroff + nlist.strx as usize)
                            } else {
                                String::new()
                            };

                            symbols.push(ParsedSymbol {
                                name,
                                n_type: nlist.typ,
                                section: nlist.sect,
                                desc: nlist.desc,
                                value: nlist.value,
                            });
                        }
                    }
                }
                _ => {
                    // Skip unknown load commands (LC_BUILD_VERSION, LC_DYSYMTAB, etc.)
                }
            }

            offset += cmdsize;
        }

        Ok(ParsedObject {
            cputype,
            cpusubtype,
            flags,
            sections,
            symbols,
        })
    }
}

// ---------------------------------------------------------------------------
// Symbol resolution
// ---------------------------------------------------------------------------

/// A resolved symbol with its final virtual address.
#[derive(Debug, Clone)]
pub struct ResolvedSymbol {
    /// Final virtual address of the symbol.
    pub address: u64,
    /// Object index the symbol was defined in.
    pub object_index: usize,
    /// Section index within that object.
    pub section_index: usize,
    /// Whether this is a weak definition (can be overridden).
    pub is_weak: bool,
}

/// Resolves symbols across multiple parsed object files.
pub struct SymbolResolver {
    /// Map from symbol name to its definition.
    defined: HashMap<String, ResolvedSymbol>,
    /// List of (object_index, symbol_index, name) for undefined references.
    undefined: Vec<(usize, usize, String)>,
    /// Set of symbol names that are weak references (can remain unresolved).
    weak_refs: HashSet<String>,
}

impl SymbolResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self {
            defined: HashMap::new(),
            undefined: Vec::new(),
            weak_refs: HashSet::new(),
        }
    }

    /// Register all symbols from a parsed object. The `layout` provides the
    /// section base addresses for computing final symbol addresses.
    ///
    /// Weak symbol semantics:
    /// - Strong definition overrides any existing weak definition
    /// - Weak definition is silently skipped if a strong definition exists
    /// - Duplicate strong definitions produce an error
    pub fn add_object(
        &mut self,
        obj_index: usize,
        obj: &ParsedObject,
        section_addrs: &[u64],
    ) -> Result<(), LinkerError> {
        for (sym_idx, sym) in obj.symbols.iter().enumerate() {
            if sym.is_defined() && sym.is_external() {
                let sec_idx = (sym.section as usize).saturating_sub(1);
                let base = if sec_idx < section_addrs.len() {
                    section_addrs[sec_idx]
                } else {
                    0
                };
                let address = base + sym.value;
                let new_is_weak = sym.is_weak_def();

                if let Some(existing) = self.defined.get(&sym.name) {
                    if existing.is_weak && !new_is_weak {
                        // Strong definition overrides existing weak - replace.
                        self.defined.insert(
                            sym.name.clone(),
                            ResolvedSymbol {
                                address,
                                object_index: obj_index,
                                section_index: sec_idx,
                                is_weak: false,
                            },
                        );
                    } else if new_is_weak {
                        // New is weak, existing is strong (or also weak) - skip.
                        continue;
                    } else {
                        // Both strong - duplicate symbol error.
                        return Err(LinkerError::DuplicateSymbolDetailed {
                            name: sym.name.clone(),
                            first_obj: existing.object_index,
                            second_obj: obj_index,
                        });
                    }
                } else {
                    self.defined.insert(
                        sym.name.clone(),
                        ResolvedSymbol {
                            address,
                            object_index: obj_index,
                            section_index: sec_idx,
                            is_weak: new_is_weak,
                        },
                    );
                }
            } else if sym.is_undefined() && sym.is_external() {
                // Track weak references separately.
                if sym.is_weak_ref() {
                    self.weak_refs.insert(sym.name.clone());
                }
                self.undefined
                    .push((obj_index, sym_idx, sym.name.clone()));
            }
        }
        Ok(())
    }

    /// Resolve all undefined symbols. Returns a map from symbol name to address.
    ///
    /// Weak references that remain unresolved are bound to address 0 (null).
    /// Strong undefined references that have no definition produce an error.
    pub fn resolve(&self) -> Result<HashMap<String, u64>, LinkerError> {
        let mut result: HashMap<String, u64> = HashMap::new();

        // Copy all defined symbols.
        for (name, sym) in &self.defined {
            result.insert(name.clone(), sym.address);
        }

        // Verify all undefined symbols have definitions.
        for (_obj_idx, _sym_idx, name) in &self.undefined {
            if !result.contains_key(name) {
                if self.weak_refs.contains(name) {
                    // Weak references resolve to 0 if not defined.
                    result.insert(name.clone(), 0);
                } else {
                    return Err(LinkerError::UndefinedSymbol(name.clone()));
                }
            }
        }

        Ok(result)
    }

    /// Look up a symbol's resolved address by name.
    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.defined.get(name).map(|s| s.address)
    }
}

impl Default for SymbolResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Section layout
// ---------------------------------------------------------------------------

/// Result of laying out sections across multiple objects.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Base addresses for each section, in order: all sections from obj 0,
    /// then all sections from obj 1, etc.
    pub section_addrs: Vec<Vec<u64>>,
    /// Total size of the __TEXT segment content.
    pub text_size: u64,
    /// Total size of the __DATA segment content.
    pub data_size: u64,
    /// File offset where __TEXT segment data starts.
    pub text_file_offset: u64,
    /// File offset where __DATA segment data starts.
    pub data_file_offset: u64,
    /// Virtual address of __TEXT segment.
    pub text_vmaddr: u64,
    /// Virtual address of __DATA segment.
    pub data_vmaddr: u64,
}

/// Assign final virtual addresses to sections from multiple objects.
///
/// Handles zerofill (BSS) sections: they occupy virtual address space but not
/// file space. BSS sections are placed after regular data sections in the
/// __DATA segment's virtual address space.
pub fn lay_out_sections(objects: &[ParsedObject], base_addr: u64) -> LayoutResult {
    let mut text_offset: u64 = 0;
    let mut data_offset: u64 = 0;
    let mut bss_offset: u64 = 0;

    let mut section_addrs: Vec<Vec<u64>> = Vec::new();

    // First pass: compute sizes and assign addresses for regular sections.
    // BSS sections are deferred to a second pass (they go after regular data).
    for obj in objects {
        let mut addrs = Vec::new();
        for sec in &obj.sections {
            let is_text = sec.segment == "__TEXT";
            let alignment = 1u64 << sec.align;

            if is_text {
                // Align text_offset.
                let misalign = text_offset % alignment;
                if misalign != 0 {
                    text_offset += alignment - misalign;
                }
                addrs.push(base_addr + text_offset);
                text_offset += sec.data.len() as u64;
            } else if sec.is_zerofill() {
                // Zerofill (BSS) - placeholder, will be fixed up in second pass.
                addrs.push(bss_offset);
                let misalign = bss_offset % alignment;
                if misalign != 0 {
                    bss_offset += alignment - misalign;
                    // Re-store with aligned offset.
                    *addrs.last_mut().unwrap() = bss_offset;
                }
                bss_offset += sec.effective_size();
            } else {
                // Regular __DATA section.
                let misalign = data_offset % alignment;
                if misalign != 0 {
                    data_offset += alignment - misalign;
                }
                // Data address will be computed after we know total text size.
                // Store as a relative offset for now.
                addrs.push(data_offset);
                data_offset += sec.data.len() as u64;
            }
        }
        section_addrs.push(addrs);
    }

    // Align text_size to page boundary.
    let text_size = text_offset;
    let text_size_aligned = align_to(text_size, PAGE_SIZE);

    let data_vmaddr = base_addr + text_size_aligned;

    // Fix up data and BSS section addresses.
    // BSS sections are placed after regular data sections in virtual space.
    let bss_base = data_vmaddr + data_offset;
    for (obj_idx, obj) in objects.iter().enumerate() {
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            if sec.segment == "__TEXT" {
                // Already has absolute addresses from the first pass.
            } else if sec.is_zerofill() {
                // BSS sections: relative offset was stored; add bss_base.
                section_addrs[obj_idx][sec_idx] += bss_base;
            } else {
                // Regular data sections: add data_vmaddr.
                section_addrs[obj_idx][sec_idx] += data_vmaddr;
            }
        }
    }

    // Total data segment VM size includes both regular data and BSS.
    let total_data_vmsize = data_offset + bss_offset;

    // Compute file offsets. For the MVP, we'll compute these during emission.
    // The text file offset comes after all load commands.
    LayoutResult {
        section_addrs,
        text_size,
        data_size: total_data_vmsize,
        text_file_offset: 0, // Will be set during emission.
        data_file_offset: 0, // Will be set during emission.
        text_vmaddr: base_addr,
        data_vmaddr,
    }
}

/// Align `value` up to the next multiple of `alignment`.
fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + alignment - remainder
    }
}

/// Compute the size of the LC_LOAD_DYLINKER load command (8-byte aligned).
fn dylinker_command_size() -> u32 {
    let name_len = DYLD_PATH.len() as u32 + 1; // +1 for NUL
    let raw = LC_LOAD_DYLINKER_HEADER_SIZE + name_len;
    align_to(raw as u64, 8) as u32
}

/// Append an LC_LOAD_DYLINKER load command pointing at `/usr/lib/dyld`.
///
/// Format:
///   cmd       u32 = LC_LOAD_DYLINKER
///   cmdsize   u32 (8-byte aligned)
///   name.offset u32 = 12 (immediately after header)
///   name      NUL-terminated string
///   padding   0..7 bytes to reach cmdsize
fn write_lc_load_dylinker(buf: &mut Vec<u8>) {
    let cmd_size = dylinker_command_size();
    buf.extend_from_slice(&LC_LOAD_DYLINKER.to_le_bytes());
    buf.extend_from_slice(&cmd_size.to_le_bytes());
    buf.extend_from_slice(&LC_LOAD_DYLINKER_HEADER_SIZE.to_le_bytes()); // name.offset
    let name_bytes = DYLD_PATH.as_bytes();
    buf.extend_from_slice(name_bytes);
    buf.push(0); // NUL terminator
    let written = LC_LOAD_DYLINKER_HEADER_SIZE as usize + name_bytes.len() + 1;
    let padding = cmd_size as usize - written;
    for _ in 0..padding {
        buf.push(0);
    }
}

// ---------------------------------------------------------------------------
// Relocation application
// ---------------------------------------------------------------------------

/// Apply relocations to mutable section data.
pub struct RelocationApplicator;

impl RelocationApplicator {
    /// Apply all relocations for a section, patching the section data in place.
    ///
    /// - `section_data`: mutable section bytes to patch.
    /// - `section_addr`: virtual address of this section.
    /// - `relocations`: relocations for this section.
    /// - `symbols`: symbol table from the source object.
    /// - `symbol_addrs`: map from symbol name to resolved address.
    pub fn apply(
        section_data: &mut [u8],
        section_addr: u64,
        relocations: &[Relocation],
        symbols: &[ParsedSymbol],
        symbol_addrs: &HashMap<String, u64>,
    ) -> Result<(), LinkerError> {
        for reloc in relocations {
            let target_addr = if reloc.is_extern {
                let sym_idx = reloc.symbol_index as usize;
                if sym_idx >= symbols.len() {
                    continue;
                }
                let sym = &symbols[sym_idx];
                match symbol_addrs.get(&sym.name) {
                    Some(&addr) => addr,
                    None => {
                        return Err(LinkerError::UndefinedSymbol(sym.name.clone()));
                    }
                }
            } else {
                // Section-relative relocation: symbol_index is a 1-based section ordinal.
                // For the MVP, treat the address as absolute.
                reloc.symbol_index as u64
            };

            let pc = section_addr + reloc.offset as u64;
            let patch_offset = reloc.offset as usize;

            match reloc.kind {
                AArch64RelocKind::Branch26 => {
                    Self::apply_branch26(section_data, patch_offset, pc, target_addr)?;
                }
                AArch64RelocKind::Page21 => {
                    Self::apply_page21(section_data, patch_offset, pc, target_addr)?;
                }
                AArch64RelocKind::Pageoff12 => {
                    Self::apply_pageoff12(section_data, patch_offset, target_addr)?;
                }
                other => {
                    return Err(LinkerError::UnsupportedRelocation(other));
                }
            }
        }
        Ok(())
    }

    /// Apply ARM64_RELOC_BRANCH26.
    ///
    /// B/BL instructions encode a signed 26-bit word offset in bits [25:0].
    /// The actual byte displacement is imm26 << 2.
    fn apply_branch26(
        data: &mut [u8],
        offset: usize,
        pc: u64,
        target: u64,
    ) -> Result<(), LinkerError> {
        if offset + 4 > data.len() {
            return Err(LinkerError::RelocationOverflow {
                detail: "BRANCH26 patch offset out of bounds".into(),
            });
        }

        let displacement = target as i64 - pc as i64;
        let imm26 = displacement >> 2;

        // Check range: signed 26-bit = +/- 128 MiB.
        if imm26 < -(1 << 25) || imm26 >= (1 << 25) {
            return Err(LinkerError::RelocationOverflow {
                detail: format!(
                    "BRANCH26 displacement {:#x} out of +/-128MiB range",
                    displacement
                ),
            });
        }

        let mut inst = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        // Clear existing imm26 field and set new value.
        inst = (inst & !0x03FF_FFFF) | ((imm26 as u32) & 0x03FF_FFFF);

        let bytes = inst.to_le_bytes();
        data[offset..offset + 4].copy_from_slice(&bytes);

        Ok(())
    }

    /// Apply ARM64_RELOC_PAGE21.
    ///
    /// ADRP encodes a signed 21-bit page offset. The page delta is:
    ///   (target_page - pc_page) >> 12
    /// where page = addr & ~0xFFF.
    ///
    /// ADRP encoding: immhi[23:5] in bits [23:5], immlo[1:0] in bits [30:29].
    fn apply_page21(
        data: &mut [u8],
        offset: usize,
        pc: u64,
        target: u64,
    ) -> Result<(), LinkerError> {
        if offset + 4 > data.len() {
            return Err(LinkerError::RelocationOverflow {
                detail: "PAGE21 patch offset out of bounds".into(),
            });
        }

        let pc_page = pc & !0xFFF;
        let target_page = target & !0xFFF;
        let page_delta = (target_page as i64 - pc_page as i64) >> 12;

        // Check range: signed 21-bit = +/- 4 GiB.
        if page_delta < -(1 << 20) || page_delta >= (1 << 20) {
            return Err(LinkerError::RelocationOverflow {
                detail: format!(
                    "PAGE21 page delta {:#x} out of +/-4GiB range",
                    page_delta
                ),
            });
        }

        let imm21 = (page_delta as u32) & 0x001F_FFFF;
        let immlo = imm21 & 0x3;
        let immhi = (imm21 >> 2) & 0x7FFFF;

        let mut inst = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        // Clear immhi (bits 23:5) and immlo (bits 30:29), then set.
        inst &= !(0x7FFFF << 5); // Clear immhi
        inst &= !(0x3 << 29); // Clear immlo
        inst |= immhi << 5;
        inst |= immlo << 29;

        let bytes = inst.to_le_bytes();
        data[offset..offset + 4].copy_from_slice(&bytes);

        Ok(())
    }

    /// Apply ARM64_RELOC_PAGEOFF12.
    ///
    /// ADD/LDR instructions encode a 12-bit page offset in bits [21:10].
    /// The offset is target & 0xFFF.
    fn apply_pageoff12(
        data: &mut [u8],
        offset: usize,
        target: u64,
    ) -> Result<(), LinkerError> {
        if offset + 4 > data.len() {
            return Err(LinkerError::RelocationOverflow {
                detail: "PAGEOFF12 patch offset out of bounds".into(),
            });
        }

        let page_offset = (target & 0xFFF) as u32;

        let mut inst = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        // Determine if this is a load/store (needs scaling) or ADD (no scaling).
        // LDR/STR instructions have bit 27 = 1 and bit 26 = 0 (load/store class).
        // ADD has opc = 0b00x at bits [30:29] and op=0 at bit [30].
        let is_load_store = (inst >> 27) & 0x1 == 1 && (inst >> 24) & 0x3 == 0x1;

        let imm12 = if is_load_store {
            // For load/store, the offset is scaled by the access size.
            // size field is bits [31:30]: 00=1B, 01=2B, 10=4B, 11=8B
            let size = (inst >> 30) & 0x3;
            let scale = 1u32 << size;
            page_offset / scale
        } else {
            // For ADD, the immediate is unscaled.
            page_offset
        };

        // Clear imm12 field (bits 21:10) and set new value.
        inst = (inst & !(0xFFF << 10)) | ((imm12 & 0xFFF) << 10);

        let bytes = inst.to_le_bytes();
        data[offset..offset + 4].copy_from_slice(&bytes);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Executable emission
// ---------------------------------------------------------------------------

/// Emits a Mach-O MH_EXECUTE binary from linked sections.
pub struct ExecutableEmitter;

impl ExecutableEmitter {
    /// Emit a complete MH_EXECUTE Mach-O file.
    ///
    /// - `text_sections`: concatenated, relocated __TEXT section data.
    /// - `data_sections`: concatenated, relocated __DATA section data.
    /// - `text_vmaddr`: virtual address of the __TEXT segment.
    /// - `data_vmaddr`: virtual address of the __DATA segment.
    /// - `entry_offset`: offset of _main within the __TEXT segment (from text_vmaddr).
    pub fn emit(
        text_data: &[u8],
        data_data: &[u8],
        text_vmaddr: u64,
        data_vmaddr: u64,
        entry_offset: u64,
    ) -> Vec<u8> {
        // Compute sizes.
        let text_size = text_data.len() as u64;
        let data_size = data_data.len() as u64;
        let text_size_aligned = align_to(text_size, PAGE_SIZE);
        let data_size_aligned = if data_size > 0 {
            align_to(data_size, PAGE_SIZE)
        } else {
            0
        };

        // Count load commands.
        let has_data = !data_data.is_empty();
        // __PAGEZERO + __TEXT + LC_MAIN + LC_SYMTAB + LC_LOAD_DYLINKER [+ __DATA]
        let ncmds: u32 = if has_data { 6 } else { 5 };

        let pagezero_seg_size = SEGMENT_COMMAND_64_SIZE;
        let text_seg_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE; // 1 section
        let data_seg_size = if has_data {
            SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE
        } else {
            0
        };
        let main_cmd_size = LC_MAIN_SIZE;
        let symtab_cmd_size = SYMTAB_COMMAND_SIZE;
        let dylinker_cmd_size = dylinker_command_size();

        let total_lc_size =
            pagezero_seg_size + text_seg_size + data_seg_size + main_cmd_size + symtab_cmd_size
                + dylinker_cmd_size;

        let header_and_lc = MACH_HEADER_64_SIZE + total_lc_size;

        // Text segment starts right after header + load commands, page-aligned.
        let text_file_offset = align_to(header_and_lc as u64, PAGE_SIZE);
        let data_file_offset = text_file_offset + text_size_aligned;

        let total_file_size = if has_data {
            data_file_offset + data_size_aligned
        } else {
            text_file_offset + text_size_aligned
        };

        let mut buf = Vec::with_capacity(total_file_size as usize);

        // --- Header ---
        buf.extend_from_slice(&MH_MAGIC_64.to_le_bytes()); // magic
        buf.extend_from_slice(&CPU_TYPE_ARM64.to_le_bytes()); // cputype
        buf.extend_from_slice(&CPU_SUBTYPE_ARM64_ALL.to_le_bytes()); // cpusubtype
        buf.extend_from_slice(&MH_EXECUTE.to_le_bytes()); // filetype
        buf.extend_from_slice(&ncmds.to_le_bytes()); // ncmds
        buf.extend_from_slice(&total_lc_size.to_le_bytes()); // sizeofcmds
        buf.extend_from_slice(&(MH_PIE).to_le_bytes()); // flags
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // --- __PAGEZERO segment ---
        Self::write_segment(
            &mut buf,
            b"__PAGEZERO",
            0,                   // vmaddr
            DEFAULT_BASE_ADDR,   // vmsize = 4GB
            0,                   // fileoff
            0,                   // filesize
            0,                   // maxprot
            0,                   // initprot
            0,                   // nsects
            0,                   // flags
        );

        // --- __TEXT segment ---
        let text_seg_vmsize = text_size_aligned;
        Self::write_segment(
            &mut buf,
            b"__TEXT",
            text_vmaddr,
            text_seg_vmsize,
            text_file_offset,
            text_size_aligned,
            VM_PROT_READ | VM_PROT_EXECUTE,
            VM_PROT_READ | VM_PROT_EXECUTE,
            1, // 1 section
            0,
        );

        // __text section header
        Self::write_section_header(
            &mut buf,
            b"__text",
            b"__TEXT",
            text_vmaddr,
            text_size,
            text_file_offset as u32,
            2, // align = 2^2 = 4
            0, // reloff
            0, // nreloc
            S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
        );

        // --- __DATA segment (if needed) ---
        if has_data {
            let data_seg_vmsize = data_size_aligned;
            Self::write_segment(
                &mut buf,
                b"__DATA",
                data_vmaddr,
                data_seg_vmsize,
                data_file_offset,
                data_size_aligned,
                VM_PROT_READ | VM_PROT_WRITE,
                VM_PROT_READ | VM_PROT_WRITE,
                1,
                0,
            );

            // __data section header
            Self::write_section_header(
                &mut buf,
                b"__data",
                b"__DATA",
                data_vmaddr,
                data_size,
                data_file_offset as u32,
                3, // align = 2^3 = 8
                0,
                0,
                S_REGULAR,
            );
        }

        // --- LC_MAIN ---
        buf.extend_from_slice(&LC_MAIN.to_le_bytes()); // cmd
        buf.extend_from_slice(&LC_MAIN_SIZE.to_le_bytes()); // cmdsize
        buf.extend_from_slice(&entry_offset.to_le_bytes()); // entryoff
        buf.extend_from_slice(&0u64.to_le_bytes()); // stacksize (0 = default)

        // --- LC_SYMTAB (empty, for format compliance) ---
        buf.extend_from_slice(&LC_SYMTAB.to_le_bytes());
        buf.extend_from_slice(&SYMTAB_COMMAND_SIZE.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // symoff
        buf.extend_from_slice(&0u32.to_le_bytes()); // nsyms
        buf.extend_from_slice(&0u32.to_le_bytes()); // stroff
        buf.extend_from_slice(&0u32.to_le_bytes()); // strsize

        // --- LC_LOAD_DYLINKER (tells macOS kernel to use /usr/lib/dyld) ---
        write_lc_load_dylinker(&mut buf);

        // --- Pad to text_file_offset ---
        while (buf.len() as u64) < text_file_offset {
            buf.push(0);
        }

        // --- Write text data ---
        buf.extend_from_slice(text_data);
        while (buf.len() as u64) < text_file_offset + text_size_aligned {
            buf.push(0);
        }

        // --- Write data data ---
        if has_data {
            buf.extend_from_slice(data_data);
            while (buf.len() as u64) < total_file_size {
                buf.push(0);
            }
        }

        buf
    }

    /// Write a segment_command_64 to the buffer.
    #[allow(clippy::too_many_arguments)]
    fn write_segment(
        buf: &mut Vec<u8>,
        name: &[u8],
        vmaddr: u64,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
        maxprot: i32,
        initprot: i32,
        nsects: u32,
        flags: u32,
    ) {
        let cmdsize = SEGMENT_COMMAND_64_SIZE + nsects * SECTION_64_SIZE;
        buf.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        buf.extend_from_slice(&cmdsize.to_le_bytes());
        buf.extend_from_slice(&padded_name(name));
        buf.extend_from_slice(&vmaddr.to_le_bytes());
        buf.extend_from_slice(&vmsize.to_le_bytes());
        buf.extend_from_slice(&fileoff.to_le_bytes());
        buf.extend_from_slice(&filesize.to_le_bytes());
        buf.extend_from_slice(&maxprot.to_le_bytes());
        buf.extend_from_slice(&initprot.to_le_bytes());
        buf.extend_from_slice(&nsects.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
    }

    /// Write a section_64 header to the buffer.
    #[allow(clippy::too_many_arguments)]
    fn write_section_header(
        buf: &mut Vec<u8>,
        sectname: &[u8],
        segname: &[u8],
        addr: u64,
        size: u64,
        offset: u32,
        align: u32,
        reloff: u32,
        nreloc: u32,
        flags: u32,
    ) {
        buf.extend_from_slice(&padded_name(sectname));
        buf.extend_from_slice(&padded_name(segname));
        buf.extend_from_slice(&addr.to_le_bytes());
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        buf.extend_from_slice(&align.to_le_bytes());
        buf.extend_from_slice(&reloff.to_le_bytes());
        buf.extend_from_slice(&nreloc.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved1
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved2
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved3
    }
}

// ---------------------------------------------------------------------------
// Dead code stripping
// ---------------------------------------------------------------------------

/// Perform basic dead code stripping by removing unreferenced sections.
///
/// This is a conservative implementation: a section is kept if it contains a
/// symbol that is transitively referenced from the entry point. Sections in
/// __DATA segments are always kept (conservative: data may be referenced by
/// address without a relocation). Symbols with N_NO_DEAD_STRIP are also kept.
///
/// Returns a filtered set of objects with unreferenced __TEXT sections removed.
pub fn dead_strip_sections(objects: &[ParsedObject], entry_symbol: &str) -> Vec<ParsedObject> {
    // Phase 1: Collect all initially referenced symbols.
    let mut referenced_symbols: HashSet<String> = HashSet::new();
    referenced_symbols.insert(entry_symbol.to_string());

    // Walk all sections and collect extern relocation targets from ALL sections
    // (initial seed: everything that is referenced from anywhere, then we prune).
    for obj in objects {
        for sec in &obj.sections {
            for reloc in &sec.relocations {
                if reloc.is_extern {
                    let sym_idx = reloc.symbol_index as usize;
                    if sym_idx < obj.symbols.len() {
                        referenced_symbols.insert(obj.symbols[sym_idx].name.clone());
                    }
                }
            }
        }
    }

    // Phase 2: Iteratively compute transitive closure of referenced symbols.
    // A section is live if it defines a referenced symbol. If live, all symbols
    // referenced by its relocations are also referenced.
    let mut changed = true;
    while changed {
        changed = false;
        for obj in objects {
            for (sec_idx, sec) in obj.sections.iter().enumerate() {
                let sec_ordinal = (sec_idx + 1) as u8;
                let section_referenced = obj.symbols.iter().any(|sym| {
                    sym.section == sec_ordinal && referenced_symbols.contains(&sym.name)
                });

                if section_referenced {
                    for reloc in &sec.relocations {
                        if reloc.is_extern {
                            let sym_idx = reloc.symbol_index as usize;
                            if sym_idx < obj.symbols.len() {
                                let name = &obj.symbols[sym_idx].name;
                                if !referenced_symbols.contains(name) {
                                    referenced_symbols.insert(name.clone());
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Filter out unreferenced __TEXT sections.
    objects
        .iter()
        .map(|obj| {
            let sections: Vec<ParsedSection> = obj
                .sections
                .iter()
                .enumerate()
                .filter(|(sec_idx, sec)| {
                    let sec_ordinal = (*sec_idx + 1) as u8;

                    // Keep if any symbol in it is referenced.
                    let has_referenced_sym = obj.symbols.iter().any(|sym| {
                        sym.section == sec_ordinal && referenced_symbols.contains(&sym.name)
                    });

                    // Keep if any symbol has N_NO_DEAD_STRIP.
                    let has_no_dead_strip = obj.symbols.iter().any(|sym| {
                        sym.section == sec_ordinal && sym.is_no_dead_strip()
                    });

                    // Always keep non-TEXT sections (conservative: data may be
                    // referenced by address).
                    let is_data = sec.segment != "__TEXT";

                    has_referenced_sym || has_no_dead_strip || is_data
                })
                .map(|(_, sec)| sec.clone())
                .collect();

            ParsedObject {
                cputype: obj.cputype,
                cpusubtype: obj.cpusubtype,
                flags: obj.flags,
                sections,
                symbols: obj.symbols.clone(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// High-level link API
// ---------------------------------------------------------------------------

/// Link multiple parsed object files into an executable.
///
/// Returns the raw bytes of a Mach-O MH_EXECUTE file.
pub fn link(objects: &[ParsedObject]) -> Result<Vec<u8>, LinkerError> {
    // 1. Lay out sections.
    let layout = lay_out_sections(objects, DEFAULT_BASE_ADDR);

    // 2. Resolve symbols.
    let mut resolver = SymbolResolver::new();
    for (obj_idx, obj) in objects.iter().enumerate() {
        resolver.add_object(obj_idx, obj, &layout.section_addrs[obj_idx])?;
    }
    let symbol_addrs = resolver.resolve()?;

    // 3. Build concatenated section data and apply relocations.
    let mut text_data = Vec::new();
    let mut data_data = Vec::new();

    for (obj_idx, obj) in objects.iter().enumerate() {
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            let sec_addr = layout.section_addrs[obj_idx][sec_idx];
            let mut sec_data = sec.data.clone();

            // Apply relocations.
            if !sec.relocations.is_empty() {
                RelocationApplicator::apply(
                    &mut sec_data,
                    sec_addr,
                    &sec.relocations,
                    &obj.symbols,
                    &symbol_addrs,
                )?;
            }

            if sec.segment == "__TEXT" {
                // Pad to alignment.
                let alignment = 1usize << sec.align;
                let misalign = text_data.len() % alignment;
                if misalign != 0 {
                    text_data.resize(text_data.len() + alignment - misalign, 0);
                }
                text_data.extend_from_slice(&sec_data);
            } else {
                let alignment = 1usize << sec.align;
                let misalign = data_data.len() % alignment;
                if misalign != 0 {
                    data_data.resize(data_data.len() + alignment - misalign, 0);
                }
                data_data.extend_from_slice(&sec_data);
            }
        }
    }

    // 4. Find _main entry point.
    let entry_addr = symbol_addrs
        .get("_main")
        .ok_or(LinkerError::NoEntryPoint)?;
    let entry_offset = entry_addr - layout.text_vmaddr;

    // 5. Emit executable.
    let text_size_aligned = align_to(text_data.len() as u64, PAGE_SIZE);
    let data_vmaddr = layout.text_vmaddr + text_size_aligned;

    Ok(ExecutableEmitter::emit(
        &text_data,
        &data_data,
        layout.text_vmaddr,
        data_vmaddr,
        entry_offset,
    ))
}

// ---------------------------------------------------------------------------
// Dylib linking support
// ---------------------------------------------------------------------------

/// Configuration for dynamic library linking.
///
/// Specifies which dylibs to link against and which symbols they provide.
/// The linker uses this to:
/// 1. Allow undefined symbols that will be resolved at load time
/// 2. Emit LC_LOAD_DYLIB commands for each required dylib
/// 3. Generate stub/GOT entries so code can call dylib functions
#[derive(Debug, Clone)]
pub struct DylibConfig {
    /// Dylib entries: (install_name, set of provided symbols).
    pub dylibs: Vec<DylibEntry>,
}

/// A single dynamic library entry with its symbols.
#[derive(Debug, Clone)]
pub struct DylibEntry {
    /// The install name path (e.g., "/usr/lib/libSystem.B.dylib").
    pub install_name: String,
    /// Symbols exported by this dylib.
    pub symbols: HashSet<String>,
}

impl DylibConfig {
    /// Create a new empty dylib config.
    pub fn new() -> Self {
        Self { dylibs: Vec::new() }
    }

    /// Create a config with libSystem.dylib providing common symbols.
    pub fn with_libsystem() -> Self {
        let mut symbols = HashSet::new();
        // Common libSystem symbols needed by most executables.
        for sym in &[
            "_exit", "_printf", "_puts", "_malloc", "_free", "_write",
            "_read", "_open", "_close", "_mmap", "_munmap", "_memcpy",
            "_memset", "_strlen", "_abort", "___stack_chk_fail",
            "___stack_chk_guard", "_atexit",
        ] {
            symbols.insert(sym.to_string());
        }

        Self {
            dylibs: vec![DylibEntry {
                install_name: "/usr/lib/libSystem.B.dylib".to_string(),
                symbols,
            }],
        }
    }

    /// Add a dylib entry.
    pub fn add_dylib(&mut self, install_name: &str, symbols: HashSet<String>) {
        self.dylibs.push(DylibEntry {
            install_name: install_name.to_string(),
            symbols,
        });
    }

    /// Check if a symbol name is provided by any configured dylib.
    pub fn is_dylib_symbol(&self, name: &str) -> bool {
        self.dylibs.iter().any(|d| d.symbols.contains(name))
    }

    /// Get the indices of dylibs that are actually needed (have symbols referenced).
    pub fn needed_dylibs(&self, undefined_symbols: &[String]) -> Vec<usize> {
        let mut needed = Vec::new();
        for (idx, dylib) in self.dylibs.iter().enumerate() {
            if undefined_symbols.iter().any(|s| dylib.symbols.contains(s)) {
                needed.push(idx);
            }
        }
        needed
    }
}

impl Default for DylibConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve all undefined symbols, allowing dylib symbols to remain unresolved
/// in the object graph (they will be resolved via stubs at runtime).
impl SymbolResolver {
    /// Resolve symbols with dylib support. Symbols in the dylib config
    /// are assigned addresses in the stub region rather than requiring
    /// a definition in the object files.
    pub fn resolve_with_dylibs(
        &self,
        dylib_config: &DylibConfig,
        stub_base_addr: u64,
    ) -> Result<(HashMap<String, u64>, Vec<String>), LinkerError> {
        let mut result: HashMap<String, u64> = HashMap::new();
        let mut dylib_symbols: Vec<String> = Vec::new();

        // Copy all defined symbols.
        for (name, sym) in &self.defined {
            result.insert(name.clone(), sym.address);
        }

        // Process undefined symbols.
        let mut stub_offset = 0u64;
        for (_obj_idx, _sym_idx, name) in &self.undefined {
            if result.contains_key(name) {
                continue;
            }
            if dylib_config.is_dylib_symbol(name) {
                // Assign a stub address for this dylib symbol.
                let stub_addr = stub_base_addr + stub_offset;
                result.insert(name.clone(), stub_addr);
                if !dylib_symbols.contains(name) {
                    dylib_symbols.push(name.clone());
                    stub_offset += STUB_SIZE as u64;
                }
            } else if self.weak_refs.contains(name) {
                // Weak references resolve to 0 if not in any dylib.
                result.insert(name.clone(), 0);
            } else {
                return Err(LinkerError::UndefinedSymbol(name.clone()));
            }
        }

        Ok((result, dylib_symbols))
    }
}

/// Link multiple parsed object files into an executable with dylib support.
///
/// This is the dylib-aware version of `link()`. It handles:
/// - Multi-file linking (any number of .o files)
/// - External dylib symbol resolution via stubs
/// - LC_LOAD_DYLIB emission for required dylibs
/// - GOT entries for indirect symbol access
///
/// Returns the raw bytes of a Mach-O MH_EXECUTE file.
pub fn link_with_dylibs(
    objects: &[ParsedObject],
    dylib_config: &DylibConfig,
) -> Result<Vec<u8>, LinkerError> {
    // 1. Lay out sections.
    let layout = lay_out_sections(objects, DEFAULT_BASE_ADDR);

    // 2. Resolve symbols (first pass to find which are dylib symbols).
    let mut resolver = SymbolResolver::new();
    for (obj_idx, obj) in objects.iter().enumerate() {
        resolver.add_object(obj_idx, obj, &layout.section_addrs[obj_idx])?;
    }

    // Compute text data size to figure out where stubs go.
    let mut text_size_estimate: u64 = 0;
    for obj in objects {
        for sec in &obj.sections {
            if sec.segment == "__TEXT" {
                let alignment = 1u64 << sec.align;
                let misalign = text_size_estimate % alignment;
                if misalign != 0 {
                    text_size_estimate += alignment - misalign;
                }
                text_size_estimate += sec.data.len() as u64;
            }
        }
    }

    // Stubs go at the end of the __TEXT segment content, 4-byte aligned.
    let stubs_offset_in_text = align_to(text_size_estimate, 4);
    let stub_base_addr = DEFAULT_BASE_ADDR + stubs_offset_in_text;

    // Resolve with dylib support.
    let (symbol_addrs, dylib_symbols) =
        resolver.resolve_with_dylibs(dylib_config, stub_base_addr)?;

    let has_dylib_symbols = !dylib_symbols.is_empty();

    // 3. Build concatenated section data and apply relocations.
    let mut text_data = Vec::new();
    let mut data_data = Vec::new();

    for (obj_idx, obj) in objects.iter().enumerate() {
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            let sec_addr = layout.section_addrs[obj_idx][sec_idx];
            let mut sec_data = sec.data.clone();

            if !sec.relocations.is_empty() {
                RelocationApplicator::apply(
                    &mut sec_data,
                    sec_addr,
                    &sec.relocations,
                    &obj.symbols,
                    &symbol_addrs,
                )?;
            }

            if sec.segment == "__TEXT" {
                let alignment = 1usize << sec.align;
                let misalign = text_data.len() % alignment;
                if misalign != 0 {
                    text_data.resize(text_data.len() + alignment - misalign, 0);
                }
                text_data.extend_from_slice(&sec_data);
            } else {
                let alignment = 1usize << sec.align;
                let misalign = data_data.len() % alignment;
                if misalign != 0 {
                    data_data.resize(data_data.len() + alignment - misalign, 0);
                }
                data_data.extend_from_slice(&sec_data);
            }
        }
    }

    // 4. Generate stub code for dylib symbols.
    // Each stub is: ADRP Xip0, _got_slot@PAGE; LDR Xip0, [Xip0, _got_slot@PAGEOFF]; BR Xip0
    // We use placeholder stubs that the dynamic linker patches via the GOT.
    let stubs_data = generate_stubs(&dylib_symbols, stub_base_addr, &layout, &text_data);

    // Pad text_data to stub alignment, then append stubs.
    let stubs_padding = stubs_offset_in_text as usize - text_data.len();
    text_data.resize(text_data.len() + stubs_padding, 0);
    text_data.extend_from_slice(&stubs_data);

    // 5. Generate GOT entries (8 bytes each, in __DATA segment).
    // Each GOT entry will hold the runtime address of a dylib symbol.
    let got_data: Vec<u8> = vec![0u8; dylib_symbols.len() * 8];

    // Append GOT to data section.
    if has_dylib_symbols {
        let alignment = 8usize;
        let misalign = data_data.len() % alignment;
        if misalign != 0 {
            data_data.resize(data_data.len() + alignment - misalign, 0);
        }
        data_data.extend_from_slice(&got_data);
    }

    // 6. Find _main entry point.
    let entry_addr = symbol_addrs
        .get("_main")
        .ok_or(LinkerError::NoEntryPoint)?;
    let entry_offset = entry_addr - layout.text_vmaddr;

    // 7. Emit executable with dylib support.
    let text_size_aligned = align_to(text_data.len() as u64, PAGE_SIZE);
    let data_vmaddr = layout.text_vmaddr + text_size_aligned;

    // Determine which dylibs are needed.
    let needed_dylib_indices = dylib_config.needed_dylibs(&dylib_symbols);
    let needed_dylibs: Vec<&DylibEntry> = needed_dylib_indices
        .iter()
        .map(|&i| &dylib_config.dylibs[i])
        .collect();

    Ok(emit_executable_with_dylibs(
        &text_data,
        &data_data,
        layout.text_vmaddr,
        data_vmaddr,
        entry_offset,
        &needed_dylibs,
        &dylib_symbols,
        stubs_offset_in_text,
    ))
}

/// Generate stub instructions for dylib symbols.
///
/// Each stub is a sequence of AArch64 instructions that loads the target
/// address from the GOT and branches to it:
/// ```text
///   ADRP  X16, _got_entry@PAGE
///   LDR   X16, [X16, _got_entry@PAGEOFF]
///   BR    X16
/// ```
fn generate_stubs(
    _dylib_symbols: &[String],
    _stub_base_addr: u64,
    _layout: &LayoutResult,
    _text_data: &[u8],
) -> Vec<u8> {
    let num_stubs = _dylib_symbols.len();
    let mut stubs = Vec::with_capacity(num_stubs * STUB_SIZE as usize);

    for _i in 0..num_stubs {
        // Generate: ADRP X16, #0; LDR X16, [X16]; BR X16
        // These are placeholder encodings. The GOT address will be patched
        // by the dynamic linker at load time.
        let adrp_x16 = 0x9000_0010u32; // ADRP X16, #0
        let ldr_x16 = 0xF940_0210u32;  // LDR X16, [X16, #0]
        let br_x16 = 0xD61F_0200u32;   // BR X16

        stubs.extend_from_slice(&adrp_x16.to_le_bytes());
        stubs.extend_from_slice(&ldr_x16.to_le_bytes());
        stubs.extend_from_slice(&br_x16.to_le_bytes());
    }

    stubs
}

/// Emit a Mach-O executable with LC_LOAD_DYLIB commands.
#[allow(clippy::too_many_arguments)]
fn emit_executable_with_dylibs(
    text_data: &[u8],
    data_data: &[u8],
    text_vmaddr: u64,
    data_vmaddr: u64,
    entry_offset: u64,
    needed_dylibs: &[&DylibEntry],
    dylib_symbols: &[String],
    stubs_offset: u64,
) -> Vec<u8> {
    let text_size = text_data.len() as u64;
    let data_size = data_data.len() as u64;
    let text_size_aligned = align_to(text_size, PAGE_SIZE);
    let data_size_aligned = if data_size > 0 {
        align_to(data_size, PAGE_SIZE)
    } else {
        0
    };

    let has_data = !data_data.is_empty();
    let has_stubs = !dylib_symbols.is_empty();

    // Count load commands:
    // __PAGEZERO + __TEXT + [__DATA] + LC_MAIN + LC_SYMTAB + LC_LOAD_DYLINKER + N * LC_LOAD_DYLIB
    let num_dylib_cmds = needed_dylibs.len() as u32;
    let base_cmds: u32 = if has_data { 6 } else { 5 };
    let ncmds = base_cmds + num_dylib_cmds;

    // Compute load command sizes.
    let pagezero_seg_size = SEGMENT_COMMAND_64_SIZE;
    // __TEXT segment: __text section + optional __stubs section
    let text_nsects: u32 = if has_stubs { 2 } else { 1 };
    let text_seg_size = SEGMENT_COMMAND_64_SIZE + text_nsects * SECTION_64_SIZE;
    let data_seg_size = if has_data {
        // __data section + optional __got section
        let data_nsects: u32 = if has_stubs { 2 } else { 1 };
        SEGMENT_COMMAND_64_SIZE + data_nsects * SECTION_64_SIZE
    } else {
        0
    };
    let main_cmd_size = LC_MAIN_SIZE;
    let symtab_cmd_size = SYMTAB_COMMAND_SIZE;
    let dylinker_cmd_size = dylinker_command_size();

    // Compute each LC_LOAD_DYLIB size (must be 8-byte aligned).
    let dylib_cmd_sizes: Vec<u32> = needed_dylibs
        .iter()
        .map(|d| {
            let name_len = d.install_name.len() as u32 + 1; // +1 for NUL
            let raw_size = LC_LOAD_DYLIB_HEADER_SIZE + name_len;
            align_to(raw_size as u64, 8) as u32
        })
        .collect();
    let total_dylib_size: u32 = dylib_cmd_sizes.iter().sum();

    let total_lc_size =
        pagezero_seg_size + text_seg_size + data_seg_size + main_cmd_size + symtab_cmd_size
            + dylinker_cmd_size + total_dylib_size;

    let header_and_lc = MACH_HEADER_64_SIZE + total_lc_size;
    let text_file_offset = align_to(header_and_lc as u64, PAGE_SIZE);
    let data_file_offset = text_file_offset + text_size_aligned;

    let total_file_size = if has_data {
        data_file_offset + data_size_aligned
    } else {
        text_file_offset + text_size_aligned
    };

    let mut buf = Vec::with_capacity(total_file_size as usize);

    // --- Header ---
    buf.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
    buf.extend_from_slice(&CPU_TYPE_ARM64.to_le_bytes());
    buf.extend_from_slice(&CPU_SUBTYPE_ARM64_ALL.to_le_bytes());
    buf.extend_from_slice(&MH_EXECUTE.to_le_bytes());
    buf.extend_from_slice(&ncmds.to_le_bytes());
    buf.extend_from_slice(&total_lc_size.to_le_bytes());
    buf.extend_from_slice(&MH_PIE.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // --- __PAGEZERO segment ---
    ExecutableEmitter::write_segment(
        &mut buf,
        b"__PAGEZERO",
        0,
        DEFAULT_BASE_ADDR,
        0,
        0,
        0,
        0,
        0,
        0,
    );

    // --- __TEXT segment ---
    let text_seg_vmsize = text_size_aligned;
    ExecutableEmitter::write_segment(
        &mut buf,
        b"__TEXT",
        text_vmaddr,
        text_seg_vmsize,
        text_file_offset,
        text_size_aligned,
        VM_PROT_READ | VM_PROT_EXECUTE,
        VM_PROT_READ | VM_PROT_EXECUTE,
        text_nsects,
        0,
    );

    // __text section header
    let text_section_size = if has_stubs { stubs_offset } else { text_size };
    ExecutableEmitter::write_section_header(
        &mut buf,
        b"__text",
        b"__TEXT",
        text_vmaddr,
        text_section_size,
        text_file_offset as u32,
        2,
        0,
        0,
        S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
    );

    // __stubs section header (if dylib symbols present)
    if has_stubs {
        let stubs_vmaddr = text_vmaddr + stubs_offset;
        let stubs_size = (dylib_symbols.len() as u64) * (STUB_SIZE as u64);
        let stubs_file_offset = text_file_offset + stubs_offset;

        ExecutableEmitter::write_section_header(
            &mut buf,
            b"__stubs",
            b"__TEXT",
            stubs_vmaddr,
            stubs_size,
            stubs_file_offset as u32,
            2, // 4-byte aligned
            0,
            0,
            S_SYMBOL_STUBS | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
        );
    }

    // --- __DATA segment (if needed) ---
    if has_data {
        let data_seg_vmsize = data_size_aligned;
        let data_nsects: u32 = if has_stubs { 2 } else { 1 };
        ExecutableEmitter::write_segment(
            &mut buf,
            b"__DATA",
            data_vmaddr,
            data_seg_vmsize,
            data_file_offset,
            data_size_aligned,
            VM_PROT_READ | VM_PROT_WRITE,
            VM_PROT_READ | VM_PROT_WRITE,
            data_nsects,
            0,
        );

        // __data section header
        let user_data_size = if has_stubs {
            data_size - (dylib_symbols.len() as u64 * 8)
        } else {
            data_size
        };
        ExecutableEmitter::write_section_header(
            &mut buf,
            b"__data",
            b"__DATA",
            data_vmaddr,
            user_data_size,
            data_file_offset as u32,
            3,
            0,
            0,
            S_REGULAR,
        );

        // __got section header (if dylib symbols present)
        if has_stubs {
            let got_size = (dylib_symbols.len() as u64) * 8;
            let got_offset_in_data = data_size - got_size;
            let got_vmaddr = data_vmaddr + got_offset_in_data;
            let got_file_offset = data_file_offset + got_offset_in_data;

            ExecutableEmitter::write_section_header(
                &mut buf,
                b"__got",
                b"__DATA",
                got_vmaddr,
                got_size,
                got_file_offset as u32,
                3, // 8-byte aligned
                0,
                0,
                S_NON_LAZY_SYMBOL_POINTERS,
            );
        }
    }

    // --- LC_MAIN ---
    buf.extend_from_slice(&LC_MAIN.to_le_bytes());
    buf.extend_from_slice(&LC_MAIN_SIZE.to_le_bytes());
    buf.extend_from_slice(&entry_offset.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // stacksize

    // --- LC_SYMTAB (empty) ---
    buf.extend_from_slice(&LC_SYMTAB.to_le_bytes());
    buf.extend_from_slice(&SYMTAB_COMMAND_SIZE.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // symoff
    buf.extend_from_slice(&0u32.to_le_bytes()); // nsyms
    buf.extend_from_slice(&0u32.to_le_bytes()); // stroff
    buf.extend_from_slice(&0u32.to_le_bytes()); // strsize

    // --- LC_LOAD_DYLINKER (tells macOS kernel to use /usr/lib/dyld) ---
    write_lc_load_dylinker(&mut buf);

    // --- LC_LOAD_DYLIB commands ---
    for (idx, dylib) in needed_dylibs.iter().enumerate() {
        let cmd_size = dylib_cmd_sizes[idx];

        buf.extend_from_slice(&LC_LOAD_DYLIB.to_le_bytes()); // cmd
        buf.extend_from_slice(&cmd_size.to_le_bytes()); // cmdsize
        // name offset: starts at offset 24 within the command (right after the header fields)
        buf.extend_from_slice(&LC_LOAD_DYLIB_HEADER_SIZE.to_le_bytes()); // name.offset
        buf.extend_from_slice(&2u32.to_le_bytes()); // timestamp (conventional: 2)
        // current_version: encode as 1.0.0 = 0x00010000
        buf.extend_from_slice(&0x0001_0000u32.to_le_bytes());
        // compatibility_version: 1.0.0
        buf.extend_from_slice(&0x0001_0000u32.to_le_bytes());

        // Name string (NUL-terminated, padded to alignment).
        let name_bytes = dylib.install_name.as_bytes();
        buf.extend_from_slice(name_bytes);
        buf.push(0); // NUL terminator

        // Pad to cmd_size.
        let written = LC_LOAD_DYLIB_HEADER_SIZE as usize + name_bytes.len() + 1;
        let padding = cmd_size as usize - written;
        for _ in 0..padding {
            buf.push(0);
        }
    }

    // --- Pad to text_file_offset ---
    while (buf.len() as u64) < text_file_offset {
        buf.push(0);
    }

    // --- Write text data ---
    buf.extend_from_slice(text_data);
    while (buf.len() as u64) < text_file_offset + text_size_aligned {
        buf.push(0);
    }

    // --- Write data ---
    if has_data {
        buf.extend_from_slice(data_data);
        while (buf.len() as u64) < total_file_size {
            buf.push(0);
        }
    }

    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::writer::MachOWriter;

    // Helper to read a u32 from bytes.
    fn rd_u32(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }

    fn rd_u64(bytes: &[u8], off: usize) -> u64 {
        u64::from_le_bytes([
            bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
            bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7],
        ])
    }

    // =======================================================================
    // Parser tests
    // =======================================================================

    #[test]
    fn test_parse_round_trip() {
        // Create a .o with MachOWriter, parse with MachOParser.
        let mut writer = MachOWriter::new();
        let nop = 0xD503201Fu32;
        let code: Vec<u8> = (0..4).flat_map(|_| nop.to_le_bytes()).collect();
        writer.add_text_section(&code);
        writer.add_symbol("_main", 1, 0, true);

        let obj_bytes = writer.write();
        let parsed = MachOParser::parse(&obj_bytes).expect("parse failed");

        assert_eq!(parsed.cputype, CPU_TYPE_ARM64);
        assert_eq!(parsed.sections.len(), 1);
        assert_eq!(parsed.sections[0].name, "__text");
        assert_eq!(parsed.sections[0].segment, "__TEXT");
        assert_eq!(parsed.sections[0].data.len(), 16);

        // Verify symbol.
        let main_sym = parsed
            .symbols
            .iter()
            .find(|s| s.name == "_main")
            .expect("_main not found");
        assert!(main_sym.is_defined());
        assert!(main_sym.is_external());
    }

    #[test]
    fn test_parse_header_fields() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0xC0, 0x03, 0x5F, 0xD6]); // RET
        let obj_bytes = writer.write();

        let parsed = MachOParser::parse(&obj_bytes).unwrap();
        assert_eq!(parsed.cputype, CPU_TYPE_ARM64);
        assert_eq!(parsed.cpusubtype, CPU_SUBTYPE_ARM64_ALL);
        assert_eq!(parsed.flags & MH_SUBSECTIONS_VIA_SYMBOLS, MH_SUBSECTIONS_VIA_SYMBOLS);
    }

    #[test]
    fn test_parse_multiple_sections() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
        writer.add_data_section(&[1, 2, 3, 4, 5, 6, 7, 8]);
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_data", 2, 0, true);

        let obj_bytes = writer.write();
        let parsed = MachOParser::parse(&obj_bytes).unwrap();

        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[0].name, "__text");
        assert_eq!(parsed.sections[1].name, "__data");
        assert_eq!(parsed.sections[1].data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_parse_relocations() {
        let mut writer = MachOWriter::new();
        let bl = 0x94000000u32; // BL #0
        let nop = 0xD503201Fu32;
        let mut code = Vec::new();
        code.extend_from_slice(&bl.to_le_bytes());
        for _ in 0..3 {
            code.extend_from_slice(&nop.to_le_bytes());
        }
        writer.add_text_section(&code);
        writer.add_symbol("_caller", 1, 0, true);
        writer.add_symbol("_callee", 0, 0, true);
        writer.add_relocation(0, Relocation::branch26(0, 1));

        let obj_bytes = writer.write();
        let parsed = MachOParser::parse(&obj_bytes).unwrap();

        assert_eq!(parsed.sections[0].relocations.len(), 1);
        let reloc = &parsed.sections[0].relocations[0];
        assert_eq!(reloc.kind, AArch64RelocKind::Branch26);
        assert_eq!(reloc.offset, 0);
        assert!(reloc.is_extern);
        assert!(reloc.pc_relative);
    }

    #[test]
    fn test_parse_bad_magic() {
        let data = vec![0u8; 64]; // All zeros, magic is 0x00000000.
        let err = MachOParser::parse(&data).unwrap_err();
        assert!(matches!(err, LinkerError::BadMagic(0)));
    }

    #[test]
    fn test_parse_too_short() {
        let data = vec![0u8; 16]; // Too short for header.
        let err = MachOParser::parse(&data).unwrap_err();
        assert!(matches!(err, LinkerError::TooShort { .. }));
    }

    // =======================================================================
    // Symbol resolution tests
    // =======================================================================

    #[test]
    fn test_symbol_resolution() {
        // Object 1 defines _callee, Object 2 references it.
        let mut writer1 = MachOWriter::new();
        let ret = 0xD65F03C0u32; // RET
        writer1.add_text_section(&ret.to_le_bytes());
        writer1.add_symbol("_callee", 1, 0, true);
        let obj1_bytes = writer1.write();
        let obj1 = MachOParser::parse(&obj1_bytes).unwrap();

        let mut writer2 = MachOWriter::new();
        let bl = 0x94000000u32;
        writer2.add_text_section(&bl.to_le_bytes());
        writer2.add_symbol("_main", 1, 0, true);
        writer2.add_symbol("_callee", 0, 0, true); // undefined
        writer2.add_relocation(0, Relocation::branch26(0, 1));
        let obj2_bytes = writer2.write();
        let obj2 = MachOParser::parse(&obj2_bytes).unwrap();

        let objects = vec![obj1, obj2];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);

        let mut resolver = SymbolResolver::new();
        resolver
            .add_object(0, &objects[0], &layout.section_addrs[0])
            .unwrap();
        resolver
            .add_object(1, &objects[1], &layout.section_addrs[1])
            .unwrap();

        let addrs = resolver.resolve().unwrap();
        assert!(addrs.contains_key("_callee"));
        assert!(addrs.contains_key("_main"));
    }

    #[test]
    fn test_symbol_resolution_undefined_error() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0; 4]);
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_missing", 0, 0, true); // undefined
        let obj_bytes = writer.write();
        let obj = MachOParser::parse(&obj_bytes).unwrap();

        let objects = vec![obj];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);

        let mut resolver = SymbolResolver::new();
        resolver
            .add_object(0, &objects[0], &layout.section_addrs[0])
            .unwrap();

        let err = resolver.resolve().unwrap_err();
        assert!(matches!(err, LinkerError::UndefinedSymbol(ref s) if s == "_missing"));
    }

    // =======================================================================
    // Section layout tests
    // =======================================================================

    #[test]
    fn test_section_layout() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0u8; 32]);
        writer.add_data_section(&[0u8; 16]);
        let obj_bytes = writer.write();
        let obj = MachOParser::parse(&obj_bytes).unwrap();

        let objects = vec![obj];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);

        // Text section starts at base.
        assert_eq!(layout.section_addrs[0][0], DEFAULT_BASE_ADDR);
        // Data section starts after text, page-aligned.
        let text_aligned = align_to(32, PAGE_SIZE);
        assert_eq!(layout.section_addrs[0][1], DEFAULT_BASE_ADDR + text_aligned);
        assert_eq!(layout.text_vmaddr, DEFAULT_BASE_ADDR);
        assert_eq!(layout.data_vmaddr, DEFAULT_BASE_ADDR + text_aligned);
    }

    // =======================================================================
    // Relocation application tests
    // =======================================================================

    #[test]
    fn test_relocation_branch26() {
        // BL instruction at address 0x100, target at 0x200.
        // displacement = 0x100, imm26 = 0x100 >> 2 = 0x40
        let bl = 0x94000000u32; // BL #0
        let mut data = bl.to_le_bytes().to_vec();

        let pc = 0x1_0000_0000u64;
        let target = 0x1_0000_0100u64;

        let relocs = vec![Relocation::branch26(0, 0)];
        let symbols = vec![ParsedSymbol {
            name: "_callee".into(),
            n_type: N_UNDF | N_EXT,
            section: 0,
            desc: 0,
            value: 0,
        }];
        let mut addrs = HashMap::new();
        addrs.insert("_callee".into(), target);

        RelocationApplicator::apply(&mut data, pc, &relocs, &symbols, &addrs).unwrap();

        let patched = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let imm26 = patched & 0x03FF_FFFF;
        let expected_imm26 = ((target as i64 - pc as i64) >> 2) as u32 & 0x03FF_FFFF;
        assert_eq!(imm26, expected_imm26);
        // Opcode bits should be preserved.
        assert_eq!(patched & 0xFC00_0000, 0x94000000);
    }

    #[test]
    fn test_relocation_page21() {
        // ADRP instruction at address 0x1_0000_0000, target at 0x1_0000_1234.
        let adrp = 0x90000000u32; // ADRP X0, #0
        let mut data = adrp.to_le_bytes().to_vec();

        let pc = 0x1_0000_0000u64;
        let target = 0x1_0000_1234u64;

        let relocs = vec![Relocation::page21(0, 0)];
        let symbols = vec![ParsedSymbol {
            name: "_sym".into(),
            n_type: N_UNDF | N_EXT,
            section: 0,
            desc: 0,
            value: 0,
        }];
        let mut addrs = HashMap::new();
        addrs.insert("_sym".into(), target);

        RelocationApplicator::apply(&mut data, pc, &relocs, &symbols, &addrs).unwrap();

        let patched = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // page_delta = (0x1_0000_1000 - 0x1_0000_0000) >> 12 = 1
        let immlo = (patched >> 29) & 0x3;
        let immhi = (patched >> 5) & 0x7FFFF;
        let imm21 = (immhi << 2) | immlo;
        assert_eq!(imm21, 1);
    }

    #[test]
    fn test_relocation_pageoff12() {
        // ADD instruction at some address, target at 0x1_0000_1234.
        // Page offset = 0x234.
        let add_inst = 0x91000000u32; // ADD X0, X0, #0
        let mut data = add_inst.to_le_bytes().to_vec();

        let pc = 0x1_0000_0000u64;
        let target = 0x1_0000_1234u64;

        let relocs = vec![Relocation::pageoff12(0, 0)];
        let symbols = vec![ParsedSymbol {
            name: "_sym".into(),
            n_type: N_UNDF | N_EXT,
            section: 0,
            desc: 0,
            value: 0,
        }];
        let mut addrs = HashMap::new();
        addrs.insert("_sym".into(), target);

        RelocationApplicator::apply(&mut data, pc, &relocs, &symbols, &addrs).unwrap();

        let patched = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let imm12 = (patched >> 10) & 0xFFF;
        assert_eq!(imm12, 0x234);
    }

    // =======================================================================
    // Executable emission tests
    // =======================================================================

    #[test]
    fn test_executable_emission() {
        let nop = 0xD503201Fu32;
        let text_data: Vec<u8> = (0..4).flat_map(|_| nop.to_le_bytes()).collect();

        let exe = ExecutableEmitter::emit(
            &text_data,
            &[],
            DEFAULT_BASE_ADDR,
            DEFAULT_BASE_ADDR + PAGE_SIZE,
            0, // entry at start
        );

        // Verify header.
        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 4), CPU_TYPE_ARM64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);
        assert_eq!(rd_u32(&exe, 24) & MH_PIE, MH_PIE);

        // Find __PAGEZERO segment.
        let seg_name_off = MACH_HEADER_64_SIZE as usize + 8; // after cmd + cmdsize
        let pagezero_name = read_name16(&exe, seg_name_off);
        assert_eq!(pagezero_name, "__PAGEZERO");

        // Verify __PAGEZERO vmaddr=0, vmsize=4GB.
        let pz_vmaddr = rd_u64(&exe, MACH_HEADER_64_SIZE as usize + 24);
        let pz_vmsize = rd_u64(&exe, MACH_HEADER_64_SIZE as usize + 32);
        assert_eq!(pz_vmaddr, 0);
        assert_eq!(pz_vmsize, DEFAULT_BASE_ADDR);
    }

    #[test]
    fn test_executable_with_data() {
        let nop = 0xD503201Fu32;
        let text_data: Vec<u8> = (0..4).flat_map(|_| nop.to_le_bytes()).collect();
        let data_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

        let text_aligned = align_to(text_data.len() as u64, PAGE_SIZE);
        let data_vmaddr = DEFAULT_BASE_ADDR + text_aligned;

        let exe = ExecutableEmitter::emit(
            &text_data,
            &data_data,
            DEFAULT_BASE_ADDR,
            data_vmaddr,
            0,
        );

        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);

        // Should have 6 load commands when data is present
        // (__PAGEZERO + __TEXT + __DATA + LC_MAIN + LC_SYMTAB + LC_LOAD_DYLINKER).
        assert_eq!(rd_u32(&exe, 16), 6);

        // Verify data is in the file.
        let found = exe.windows(4).any(|w| w == &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(found, "data section content should be in the executable");
    }

    // =======================================================================
    // End-to-end link test
    // =======================================================================

    #[test]
    fn test_link_two_objects() {
        // Object 1: _callee returns (RET instruction).
        let mut writer1 = MachOWriter::new();
        let ret = 0xD65F03C0u32;
        writer1.add_text_section(&ret.to_le_bytes());
        writer1.add_symbol("_callee", 1, 0, true);
        let obj1_bytes = writer1.write();
        let obj1 = MachOParser::parse(&obj1_bytes).unwrap();

        // Object 2: _main calls _callee (BL) then returns (RET).
        let mut writer2 = MachOWriter::new();
        let bl = 0x94000000u32;
        let ret2 = 0xD65F03C0u32;
        let mut code = Vec::new();
        code.extend_from_slice(&bl.to_le_bytes());
        code.extend_from_slice(&ret2.to_le_bytes());
        writer2.add_text_section(&code);
        writer2.add_symbol("_main", 1, 0, true);
        writer2.add_symbol("_callee", 0, 0, true);

        // NOTE: the relocation symbol_index refers to the symbol's position
        // in the writer's symbol list. After parsing, the MachOWriter reorders
        // symbols (locals, extdef, undef). _callee is external-undefined so it
        // ends up last. In the parsed object, _main is at index 0 and _callee
        // is at index 1 (both are external, _main is defined, _callee undefined).
        // The writer encodes the relocation with the original pre-reorder index.
        // After round-tripping through write + parse, we need to check which
        // index _callee ended up at.
        writer2.add_relocation(0, Relocation::branch26(0, 1));
        let obj2_bytes = writer2.write();
        let obj2 = MachOParser::parse(&obj2_bytes).unwrap();

        // Find _callee symbol index in the parsed object2.
        let callee_idx = obj2
            .symbols
            .iter()
            .position(|s| s.name == "_callee")
            .expect("_callee not found in parsed obj2");

        // Reconstruct with correct symbol index.
        let mut obj2_fixed = obj2.clone();
        if !obj2_fixed.sections[0].relocations.is_empty() {
            obj2_fixed.sections[0].relocations[0].symbol_index = callee_idx as u32;
        }

        let exe = link(&[obj1, obj2_fixed]).unwrap();

        // Verify executable header.
        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);
        assert!(exe.len() > 100); // Non-trivial output.
    }

    #[test]
    fn test_link_single_object() {
        // Single object with just _main.
        let mut writer = MachOWriter::new();
        let ret = 0xD65F03C0u32;
        writer.add_text_section(&ret.to_le_bytes());
        writer.add_symbol("_main", 1, 0, true);
        let obj_bytes = writer.write();
        let obj = MachOParser::parse(&obj_bytes).unwrap();

        let exe = link(&[obj]).unwrap();
        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);
    }

    #[test]
    fn test_link_no_main_error() {
        let mut writer = MachOWriter::new();
        let ret = 0xD65F03C0u32;
        writer.add_text_section(&ret.to_le_bytes());
        writer.add_symbol("_foo", 1, 0, true);
        let obj_bytes = writer.write();
        let obj = MachOParser::parse(&obj_bytes).unwrap();

        let err = link(&[obj]).unwrap_err();
        assert!(matches!(err, LinkerError::NoEntryPoint));
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 4096), 0);
        assert_eq!(align_to(1, 4096), 4096);
        assert_eq!(align_to(4096, 4096), 4096);
        assert_eq!(align_to(4097, 4096), 8192);
        assert_eq!(align_to(100, 0), 100);
    }

    // =======================================================================
    // Multi-file linking tests
    // =======================================================================

    #[test]
    fn test_link_three_objects() {
        // Object 1: _add function (ADD X0, X0, X1; RET)
        let mut writer1 = MachOWriter::new();
        let add_inst = 0x8B010000u32; // ADD X0, X0, X1
        let ret1 = 0xD65F03C0u32;    // RET
        let mut code1 = Vec::new();
        code1.extend_from_slice(&add_inst.to_le_bytes());
        code1.extend_from_slice(&ret1.to_le_bytes());
        writer1.add_text_section(&code1);
        writer1.add_symbol("_add", 1, 0, true);
        let obj1_bytes = writer1.write();
        let obj1 = MachOParser::parse(&obj1_bytes).unwrap();

        // Object 2: _sub function (SUB X0, X0, X1; RET)
        let mut writer2 = MachOWriter::new();
        let sub_inst = 0xCB010000u32; // SUB X0, X0, X1
        let ret2 = 0xD65F03C0u32;
        let mut code2 = Vec::new();
        code2.extend_from_slice(&sub_inst.to_le_bytes());
        code2.extend_from_slice(&ret2.to_le_bytes());
        writer2.add_text_section(&code2);
        writer2.add_symbol("_sub", 1, 0, true);
        let obj2_bytes = writer2.write();
        let obj2 = MachOParser::parse(&obj2_bytes).unwrap();

        // Object 3: _main calls _add then _sub (BL _add; BL _sub; RET)
        let mut writer3 = MachOWriter::new();
        let bl1 = 0x94000000u32; // BL #0 (placeholder)
        let bl2 = 0x94000000u32; // BL #0 (placeholder)
        let ret3 = 0xD65F03C0u32;
        let mut code3 = Vec::new();
        code3.extend_from_slice(&bl1.to_le_bytes());
        code3.extend_from_slice(&bl2.to_le_bytes());
        code3.extend_from_slice(&ret3.to_le_bytes());
        writer3.add_text_section(&code3);
        writer3.add_symbol("_main", 1, 0, true);
        writer3.add_symbol("_add", 0, 0, true);  // undefined
        writer3.add_symbol("_sub", 0, 0, true);  // undefined
        writer3.add_relocation(0, Relocation::branch26(0, 1)); // BL _add
        writer3.add_relocation(0, Relocation::branch26(4, 2)); // BL _sub
        let obj3_bytes = writer3.write();
        let obj3 = MachOParser::parse(&obj3_bytes).unwrap();

        // Fix up relocation symbol indices after parsing.
        let mut obj3_fixed = obj3.clone();
        let add_idx = obj3_fixed.symbols.iter().position(|s| s.name == "_add").unwrap();
        let sub_idx = obj3_fixed.symbols.iter().position(|s| s.name == "_sub").unwrap();
        obj3_fixed.sections[0].relocations[0].symbol_index = add_idx as u32;
        obj3_fixed.sections[0].relocations[1].symbol_index = sub_idx as u32;

        // Link all three objects.
        let exe = link(&[obj1, obj2, obj3_fixed]).unwrap();

        // Verify executable header.
        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);
        assert!(exe.len() > 100);

        // Verify that both _add and _sub code are in the text segment.
        // ADD X0, X0, X1 = 0x8B010000
        let has_add = exe.windows(4).any(|w| {
            u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == 0x8B010000
        });
        assert!(has_add, "ADD instruction should be in executable");

        // SUB X0, X0, X1 = 0xCB010000
        let has_sub = exe.windows(4).any(|w| {
            u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == 0xCB010000
        });
        assert!(has_sub, "SUB instruction should be in executable");
    }

    #[test]
    fn test_link_three_objects_with_data() {
        // Object 1: defines _data1 (constant bytes)
        let mut writer1 = MachOWriter::new();
        writer1.add_text_section(&0xD503201Fu32.to_le_bytes()); // NOP
        writer1.add_data_section(&[0x11, 0x22, 0x33, 0x44]);
        writer1.add_symbol("_func1", 1, 0, true);
        writer1.add_symbol("_data1", 2, 0, true);
        let obj1_bytes = writer1.write();
        let obj1 = MachOParser::parse(&obj1_bytes).unwrap();

        // Object 2: defines _data2
        let mut writer2 = MachOWriter::new();
        writer2.add_text_section(&0xD503201Fu32.to_le_bytes()); // NOP
        writer2.add_data_section(&[0xAA, 0xBB, 0xCC, 0xDD]);
        writer2.add_symbol("_func2", 1, 0, true);
        writer2.add_symbol("_data2", 2, 0, true);
        let obj2_bytes = writer2.write();
        let obj2 = MachOParser::parse(&obj2_bytes).unwrap();

        // Object 3: _main
        let mut writer3 = MachOWriter::new();
        writer3.add_text_section(&0xD65F03C0u32.to_le_bytes()); // RET
        writer3.add_symbol("_main", 1, 0, true);
        let obj3_bytes = writer3.write();
        let obj3 = MachOParser::parse(&obj3_bytes).unwrap();

        let exe = link(&[obj1, obj2, obj3]).unwrap();

        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);

        // Verify both data sections appear in the output.
        let has_data1 = exe.windows(4).any(|w| w == &[0x11, 0x22, 0x33, 0x44]);
        let has_data2 = exe.windows(4).any(|w| w == &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert!(has_data1, "data1 should be in executable");
        assert!(has_data2, "data2 should be in executable");
    }

    // =======================================================================
    // Dylib linking tests
    // =======================================================================

    #[test]
    fn test_dylib_config_basic() {
        let config = DylibConfig::with_libsystem();
        assert!(config.is_dylib_symbol("_exit"));
        assert!(config.is_dylib_symbol("_printf"));
        assert!(!config.is_dylib_symbol("_my_custom_func"));
    }

    #[test]
    fn test_dylib_config_needed_dylibs() {
        let config = DylibConfig::with_libsystem();
        let needed = config.needed_dylibs(&["_exit".to_string(), "_printf".to_string()]);
        assert_eq!(needed.len(), 1);
        assert_eq!(needed[0], 0);

        // No undefined symbols -> no dylibs needed.
        let needed_empty = config.needed_dylibs(&[]);
        assert!(needed_empty.is_empty());
    }

    #[test]
    fn test_link_with_dylib_symbols() {
        // Object that calls _exit (a libSystem symbol).
        // _main: MOV X0, #0; BL _exit
        let mut writer = MachOWriter::new();
        let mov_x0_0 = 0xD2800000u32; // MOV X0, #0
        let bl_exit = 0x94000000u32;   // BL #0 (placeholder, will be resolved to stub)
        let mut code = Vec::new();
        code.extend_from_slice(&mov_x0_0.to_le_bytes());
        code.extend_from_slice(&bl_exit.to_le_bytes());
        writer.add_text_section(&code);
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_exit", 0, 0, true); // undefined external
        writer.add_relocation(0, Relocation::branch26(4, 1)); // BL _exit at offset 4

        let obj_bytes = writer.write();
        let obj = MachOParser::parse(&obj_bytes).unwrap();

        // Fix relocation symbol index.
        let mut obj_fixed = obj.clone();
        let exit_idx = obj_fixed.symbols.iter().position(|s| s.name == "_exit").unwrap();
        obj_fixed.sections[0].relocations[0].symbol_index = exit_idx as u32;

        let config = DylibConfig::with_libsystem();
        let exe = link_with_dylibs(&[obj_fixed], &config).unwrap();

        // Verify executable header.
        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);

        // Verify LC_LOAD_DYLIB is present by scanning for the command type.
        let mut found_dylib_cmd = false;
        let mut offset = MACH_HEADER_64_SIZE as usize;
        let ncmds = rd_u32(&exe, 16);
        let sizeofcmds = rd_u32(&exe, 20) as usize;
        let lc_end = offset + sizeofcmds;

        for _ in 0..ncmds {
            if offset + 8 > lc_end {
                break;
            }
            let cmd = rd_u32(&exe, offset);
            let cmdsize = rd_u32(&exe, offset + 4) as usize;

            if cmd == LC_LOAD_DYLIB {
                found_dylib_cmd = true;
                // Verify the dylib name is present.
                let name_offset = rd_u32(&exe, offset + 8) as usize;
                let name_start = offset + name_offset;
                // Read NUL-terminated string.
                let mut name_end = name_start;
                while name_end < exe.len() && exe[name_end] != 0 {
                    name_end += 1;
                }
                let name = String::from_utf8_lossy(&exe[name_start..name_end]);
                assert_eq!(name, "/usr/lib/libSystem.B.dylib");
            }

            offset += cmdsize;
        }
        assert!(found_dylib_cmd, "LC_LOAD_DYLIB should be present");

        // Verify stubs are in the text segment (ADRP X16 = 0x90000010).
        let has_stub = exe.windows(4).any(|w| {
            u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == 0x9000_0010
        });
        assert!(has_stub, "stub ADRP X16 instruction should be present");
    }

    #[test]
    fn test_link_with_dylib_undefined_non_dylib_error() {
        // Object that references a symbol not in any dylib.
        let mut writer = MachOWriter::new();
        writer.add_text_section(&0x94000000u32.to_le_bytes()); // BL
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_unknown_func", 0, 0, true); // undefined, not in any dylib
        writer.add_relocation(0, Relocation::branch26(0, 1));

        let obj_bytes = writer.write();
        let obj = MachOParser::parse(&obj_bytes).unwrap();

        let mut obj_fixed = obj.clone();
        let idx = obj_fixed.symbols.iter().position(|s| s.name == "_unknown_func").unwrap();
        obj_fixed.sections[0].relocations[0].symbol_index = idx as u32;

        let config = DylibConfig::with_libsystem();
        let err = link_with_dylibs(&[obj_fixed], &config).unwrap_err();
        assert!(matches!(err, LinkerError::UndefinedSymbol(ref s) if s == "_unknown_func"));
    }

    #[test]
    fn test_link_with_dylib_multiple_objects_and_exit() {
        // Object 1: _helper function (just RET)
        let mut writer1 = MachOWriter::new();
        writer1.add_text_section(&0xD65F03C0u32.to_le_bytes()); // RET
        writer1.add_symbol("_helper", 1, 0, true);
        let obj1_bytes = writer1.write();
        let obj1 = MachOParser::parse(&obj1_bytes).unwrap();

        // Object 2: _main calls _helper then _exit
        let mut writer2 = MachOWriter::new();
        let bl_helper = 0x94000000u32;
        let mov_x0 = 0xD2800000u32;
        let bl_exit = 0x94000000u32;
        let mut code = Vec::new();
        code.extend_from_slice(&bl_helper.to_le_bytes());
        code.extend_from_slice(&mov_x0.to_le_bytes());
        code.extend_from_slice(&bl_exit.to_le_bytes());
        writer2.add_text_section(&code);
        writer2.add_symbol("_main", 1, 0, true);
        writer2.add_symbol("_helper", 0, 0, true);
        writer2.add_symbol("_exit", 0, 0, true);
        writer2.add_relocation(0, Relocation::branch26(0, 1));  // BL _helper
        writer2.add_relocation(0, Relocation::branch26(8, 2));  // BL _exit
        let obj2_bytes = writer2.write();
        let obj2 = MachOParser::parse(&obj2_bytes).unwrap();

        // Fix up symbol indices.
        let mut obj2_fixed = obj2.clone();
        let helper_idx = obj2_fixed.symbols.iter().position(|s| s.name == "_helper").unwrap();
        let exit_idx = obj2_fixed.symbols.iter().position(|s| s.name == "_exit").unwrap();
        obj2_fixed.sections[0].relocations[0].symbol_index = helper_idx as u32;
        obj2_fixed.sections[0].relocations[1].symbol_index = exit_idx as u32;

        let config = DylibConfig::with_libsystem();
        let exe = link_with_dylibs(&[obj1, obj2_fixed], &config).unwrap();

        // Basic validity checks.
        assert_eq!(rd_u32(&exe, 0), MH_MAGIC_64);
        assert_eq!(rd_u32(&exe, 12), MH_EXECUTE);
        assert!(exe.len() > PAGE_SIZE as usize); // Non-trivial output.
    }

    // =======================================================================
    // Weak symbol tests
    // =======================================================================

    #[test]
    fn test_weak_symbol_override() {
        // Object 1 defines _foo as weak.
        let obj1 = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![0xC0, 0x03, 0x5F, 0xD6], // RET
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 4,
            }],
            symbols: vec![ParsedSymbol {
                name: "_foo".into(),
                n_type: N_SECT | N_EXT,
                section: 1,
                desc: N_WEAK_DEF, // weak definition
                value: 0,
            }],
        };

        // Object 2 defines _foo as strong + _main.
        let obj2 = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![
                    0xC0, 0x03, 0x5F, 0xD6, // RET (_foo)
                    0xC0, 0x03, 0x5F, 0xD6, // RET (_main)
                ],
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 8,
            }],
            symbols: vec![
                ParsedSymbol {
                    name: "_foo".into(),
                    n_type: N_SECT | N_EXT,
                    section: 1,
                    desc: 0, // strong definition
                    value: 0,
                },
                ParsedSymbol {
                    name: "_main".into(),
                    n_type: N_SECT | N_EXT,
                    section: 1,
                    desc: 0,
                    value: 4,
                },
            ],
        };

        // Link should succeed (strong overrides weak).
        let objects = vec![obj1, obj2];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);
        let mut resolver = SymbolResolver::new();
        resolver.add_object(0, &objects[0], &layout.section_addrs[0]).unwrap();
        resolver.add_object(1, &objects[1], &layout.section_addrs[1]).unwrap();
        let addrs = resolver.resolve().unwrap();

        // _foo should resolve to object 2's address (the strong one).
        let foo_addr = addrs["_foo"];
        // Object 2's section starts after object 1's 4-byte section.
        assert_eq!(foo_addr, layout.section_addrs[1][0]);
    }

    #[test]
    fn test_weak_symbol_duplicate_weak() {
        // Two objects both define _foo as weak. No error, first wins.
        let obj1 = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![0xC0, 0x03, 0x5F, 0xD6],
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 4,
            }],
            symbols: vec![
                ParsedSymbol {
                    name: "_foo".into(),
                    n_type: N_SECT | N_EXT,
                    section: 1,
                    desc: N_WEAK_DEF,
                    value: 0,
                },
                ParsedSymbol {
                    name: "_main".into(),
                    n_type: N_SECT | N_EXT,
                    section: 1,
                    desc: 0,
                    value: 0,
                },
            ],
        };

        let obj2 = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![0xC0, 0x03, 0x5F, 0xD6],
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 4,
            }],
            symbols: vec![ParsedSymbol {
                name: "_foo".into(),
                n_type: N_SECT | N_EXT,
                section: 1,
                desc: N_WEAK_DEF,
                value: 0,
            }],
        };

        let objects = vec![obj1, obj2];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);
        let mut resolver = SymbolResolver::new();
        resolver.add_object(0, &objects[0], &layout.section_addrs[0]).unwrap();
        resolver.add_object(1, &objects[1], &layout.section_addrs[1]).unwrap();
        let addrs = resolver.resolve().unwrap();

        // First wins: _foo should be from object 0.
        assert_eq!(addrs["_foo"], layout.section_addrs[0][0]);
    }

    #[test]
    fn test_weak_symbol_strong_duplicate_error() {
        // Two objects both define _foo as strong. Should error.
        let obj1 = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![0xC0, 0x03, 0x5F, 0xD6],
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 4,
            }],
            symbols: vec![ParsedSymbol {
                name: "_foo".into(),
                n_type: N_SECT | N_EXT,
                section: 1,
                desc: 0, // strong
                value: 0,
            }],
        };

        let obj2 = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![0xC0, 0x03, 0x5F, 0xD6],
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 4,
            }],
            symbols: vec![ParsedSymbol {
                name: "_foo".into(),
                n_type: N_SECT | N_EXT,
                section: 1,
                desc: 0, // strong
                value: 0,
            }],
        };

        let objects = vec![obj1, obj2];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);
        let mut resolver = SymbolResolver::new();
        resolver.add_object(0, &objects[0], &layout.section_addrs[0]).unwrap();
        let err = resolver.add_object(1, &objects[1], &layout.section_addrs[1]).unwrap_err();
        assert!(matches!(err, LinkerError::DuplicateSymbolDetailed {
            ref name, first_obj: 0, second_obj: 1
        } if name == "_foo"));
    }

    #[test]
    fn test_weak_reference_unresolved() {
        // Object with a weak reference to _optional_func that has no definition.
        let obj = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![ParsedSection {
                name: "__text".into(),
                segment: "__TEXT".into(),
                data: vec![0xC0, 0x03, 0x5F, 0xD6],
                addr: 0,
                align: 2,
                flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                relocations: vec![],
                vmsize: 4,
            }],
            symbols: vec![
                ParsedSymbol {
                    name: "_main".into(),
                    n_type: N_SECT | N_EXT,
                    section: 1,
                    desc: 0,
                    value: 0,
                },
                ParsedSymbol {
                    name: "_optional_func".into(),
                    n_type: N_UNDF | N_EXT,
                    section: 0,
                    desc: N_WEAK_REF, // weak reference
                    value: 0,
                },
            ],
        };

        let objects = vec![obj];
        let layout = lay_out_sections(&objects, DEFAULT_BASE_ADDR);
        let mut resolver = SymbolResolver::new();
        resolver.add_object(0, &objects[0], &layout.section_addrs[0]).unwrap();

        // resolve() should succeed, with _optional_func at address 0.
        let addrs = resolver.resolve().unwrap();
        assert_eq!(addrs["_optional_func"], 0);
    }

    // =======================================================================
    // Dead code stripping tests
    // =======================================================================

    #[test]
    fn test_dead_strip_basic() {
        // Object with three text sections:
        // - Section 0: _main (referenced as entry)
        // - Section 1: _helper (referenced by _main via relocation)
        // - Section 2: _unused (unreferenced)
        let obj = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![
                ParsedSection {
                    name: "__text".into(),
                    segment: "__TEXT".into(),
                    data: vec![0x94, 0x00, 0x00, 0x00], // BL placeholder
                    addr: 0,
                    align: 2,
                    flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                    relocations: vec![Relocation::branch26(0, 2)], // refs _helper (sym idx 2)
                    vmsize: 4,
                },
                ParsedSection {
                    name: "__text".into(),
                    segment: "__TEXT".into(),
                    data: vec![0xC0, 0x03, 0x5F, 0xD6], // RET
                    addr: 4,
                    align: 2,
                    flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                    relocations: vec![],
                    vmsize: 4,
                },
                ParsedSection {
                    name: "__text".into(),
                    segment: "__TEXT".into(),
                    data: vec![0xC0, 0x03, 0x5F, 0xD6], // RET
                    addr: 8,
                    align: 2,
                    flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                    relocations: vec![],
                    vmsize: 4,
                },
            ],
            symbols: vec![
                ParsedSymbol {
                    name: "_main".into(),
                    n_type: N_SECT | N_EXT,
                    section: 1, // section ordinal 1 (first section)
                    desc: 0,
                    value: 0,
                },
                ParsedSymbol {
                    name: "_unused".into(),
                    n_type: N_SECT | N_EXT,
                    section: 3, // section ordinal 3 (third section)
                    desc: 0,
                    value: 0,
                },
                ParsedSymbol {
                    name: "_helper".into(),
                    n_type: N_SECT | N_EXT,
                    section: 2, // section ordinal 2 (second section)
                    desc: 0,
                    value: 0,
                },
            ],
        };

        let stripped = dead_strip_sections(&[obj], "_main");
        assert_eq!(stripped.len(), 1);
        // _unused's section should be removed (it was the third section).
        // We should have 2 sections left: _main's and _helper's.
        assert_eq!(stripped[0].sections.len(), 2);
    }

    #[test]
    fn test_dead_strip_keeps_data() {
        // Data section without any symbol reference should still be kept.
        let obj = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![
                ParsedSection {
                    name: "__text".into(),
                    segment: "__TEXT".into(),
                    data: vec![0xC0, 0x03, 0x5F, 0xD6],
                    addr: 0,
                    align: 2,
                    flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                    relocations: vec![],
                    vmsize: 4,
                },
                ParsedSection {
                    name: "__data".into(),
                    segment: "__DATA".into(),
                    data: vec![0xDE, 0xAD, 0xBE, 0xEF],
                    addr: 0,
                    align: 2,
                    flags: S_REGULAR,
                    relocations: vec![],
                    vmsize: 4,
                },
            ],
            symbols: vec![ParsedSymbol {
                name: "_main".into(),
                n_type: N_SECT | N_EXT,
                section: 1,
                desc: 0,
                value: 0,
            }],
        };

        let stripped = dead_strip_sections(&[obj], "_main");
        // Both sections should be kept (data is always preserved).
        assert_eq!(stripped[0].sections.len(), 2);
    }

    // =======================================================================
    // BSS and section method tests
    // =======================================================================

    #[test]
    fn test_bss_section_layout() {
        // Object with text, data, and bss sections.
        let obj = ParsedObject {
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            flags: 0,
            sections: vec![
                ParsedSection {
                    name: "__text".into(),
                    segment: "__TEXT".into(),
                    data: vec![0xC0, 0x03, 0x5F, 0xD6],
                    addr: 0,
                    align: 2,
                    flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                    relocations: vec![],
                    vmsize: 4,
                },
                ParsedSection {
                    name: "__data".into(),
                    segment: "__DATA".into(),
                    data: vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
                    addr: 0,
                    align: 3, // 8-byte aligned
                    flags: S_REGULAR,
                    relocations: vec![],
                    vmsize: 8,
                },
                ParsedSection {
                    name: "__bss".into(),
                    segment: "__DATA".into(),
                    data: vec![], // zerofill: no file data
                    addr: 0,
                    align: 3,
                    flags: S_ZEROFILL,
                    relocations: vec![],
                    vmsize: 64, // 64 bytes of zero-initialized memory
                },
            ],
            symbols: vec![ParsedSymbol {
                name: "_main".into(),
                n_type: N_SECT | N_EXT,
                section: 1,
                desc: 0,
                value: 0,
            }],
        };

        let layout = lay_out_sections(&[obj], DEFAULT_BASE_ADDR);

        // Text section at base.
        assert_eq!(layout.section_addrs[0][0], DEFAULT_BASE_ADDR);
        // Data section after text (page-aligned gap).
        let text_aligned = align_to(4, PAGE_SIZE);
        let data_vmaddr = DEFAULT_BASE_ADDR + text_aligned;
        assert_eq!(layout.section_addrs[0][1], data_vmaddr);
        // BSS section after data.
        let bss_addr = layout.section_addrs[0][2];
        assert!(bss_addr >= data_vmaddr + 8, "BSS should be after data");
    }

    #[test]
    fn test_parsed_section_methods() {
        let regular = ParsedSection {
            name: "__text".into(),
            segment: "__TEXT".into(),
            data: vec![0u8; 32],
            addr: 0,
            align: 2,
            flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
            relocations: vec![],
            vmsize: 32,
        };
        assert_eq!(regular.section_type(), S_REGULAR);
        assert!(!regular.is_zerofill());
        assert_eq!(regular.effective_size(), 32);

        let bss = ParsedSection {
            name: "__bss".into(),
            segment: "__DATA".into(),
            data: vec![],
            addr: 0,
            align: 3,
            flags: S_ZEROFILL,
            relocations: vec![],
            vmsize: 256,
        };
        assert_eq!(bss.section_type(), S_ZEROFILL);
        assert!(bss.is_zerofill());
        assert_eq!(bss.effective_size(), 256);
    }

    #[test]
    fn test_parsed_symbol_weak_methods() {
        let strong_def = ParsedSymbol {
            name: "_foo".into(),
            n_type: N_SECT | N_EXT,
            section: 1,
            desc: 0,
            value: 0,
        };
        assert!(!strong_def.is_weak_def());
        assert!(!strong_def.is_weak_ref());
        assert!(!strong_def.is_no_dead_strip());

        let weak_def = ParsedSymbol {
            name: "_bar".into(),
            n_type: N_SECT | N_EXT,
            section: 1,
            desc: N_WEAK_DEF,
            value: 0,
        };
        assert!(weak_def.is_weak_def());
        assert!(!weak_def.is_weak_ref());

        let weak_ref = ParsedSymbol {
            name: "_baz".into(),
            n_type: N_UNDF | N_EXT,
            section: 0,
            desc: N_WEAK_REF,
            value: 0,
        };
        assert!(!weak_ref.is_weak_def());
        assert!(weak_ref.is_weak_ref());

        let no_dead_strip = ParsedSymbol {
            name: "_keep".into(),
            n_type: N_SECT | N_EXT,
            section: 1,
            desc: N_NO_DEAD_STRIP,
            value: 0,
        };
        assert!(no_dead_strip.is_no_dead_strip());
    }
}

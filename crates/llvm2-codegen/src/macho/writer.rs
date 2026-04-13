// llvm2-codegen/macho/writer.rs - Mach-O object file writer
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Assembles a complete Mach-O 64-bit relocatable object file (.o).
//!
//! Layout of a typical MH_OBJECT file:
//!
//! ```text
//! ┌──────────────────────────┐  offset 0
//! │    mach_header_64        │  32 bytes
//! ├──────────────────────────┤
//! │    Load commands:        │
//! │      LC_SEGMENT_64       │  72 + 80*nsects bytes
//! │      LC_BUILD_VERSION    │  24 bytes
//! │      LC_SYMTAB           │  24 bytes
//! │      LC_DYSYMTAB         │  80 bytes
//! ├──────────────────────────┤
//! │    Section data          │  (text, data, etc.)
//! ├──────────────────────────┤
//! │    Relocation entries    │  8 bytes each
//! ├──────────────────────────┤
//! │    Symbol table (nlist)  │  16 bytes each
//! ├──────────────────────────┤
//! │    String table          │  variable
//! └──────────────────────────┘
//! ```

use super::constants::*;
use super::header::MachHeader;
use super::reloc::{encode_relocation, Relocation};
use super::section::{Section64, SegmentCommand64};

/// A symbol to be emitted in the object file.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name (will be prefixed with '_' per Mach-O convention).
    pub name: String,
    /// Section index (1-based; 0 = N_UNDF).
    pub section: u8,
    /// Offset within the section.
    pub value: u64,
    /// Whether the symbol is externally visible.
    pub is_global: bool,
}

/// Internal section data held by the writer.
#[derive(Debug, Clone)]
struct SectionData {
    /// Section name (e.g., b"__text").
    sectname: Vec<u8>,
    /// Segment name (e.g., b"__TEXT").
    segname: Vec<u8>,
    /// Section content bytes.
    data: Vec<u8>,
    /// Alignment as power of 2.
    align: u32,
    /// Section flags.
    flags: u32,
    /// Relocations for this section.
    relocations: Vec<Relocation>,
}

/// Assembles a complete Mach-O 64-bit relocatable object file.
///
/// # Example
///
/// ```
/// use llvm2_codegen::macho::MachOWriter;
///
/// let mut writer = MachOWriter::new();
/// // ARM64 NOP = 0xD503201F
/// writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
/// writer.add_symbol("_main", 1, 0, true);
/// let bytes = writer.write();
/// // bytes now contains a valid Mach-O .o file
/// ```
pub struct MachOWriter {
    sections: Vec<SectionData>,
    symbols: Vec<Symbol>,
}

impl MachOWriter {
    /// Create a new empty Mach-O writer.
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Add a text section (__text in __TEXT) with the given machine code bytes.
    pub fn add_text_section(&mut self, code: &[u8]) {
        self.sections.push(SectionData {
            sectname: b"__text".to_vec(),
            segname: b"__TEXT".to_vec(),
            data: code.to_vec(),
            align: 2, // 2^2 = 4-byte (ARM64 instruction alignment)
            flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
            relocations: Vec::new(),
        });
    }

    /// Add a data section (__data in __DATA) with the given data bytes.
    pub fn add_data_section(&mut self, data: &[u8]) {
        self.sections.push(SectionData {
            sectname: b"__data".to_vec(),
            segname: b"__DATA".to_vec(),
            data: data.to_vec(),
            align: 3, // 2^3 = 8-byte alignment
            flags: S_REGULAR,
            relocations: Vec::new(),
        });
    }

    /// Add a custom section with the given name, segment, data, alignment, and flags.
    ///
    /// This is used for sections like `__LD,__compact_unwind` that don't fit
    /// the standard `__TEXT/__text` or `__DATA/__data` patterns.
    ///
    /// - `sectname`: Section name (e.g., b"__compact_unwind"), max 16 bytes.
    /// - `segname`: Segment name (e.g., b"__LD"), max 16 bytes.
    /// - `data`: Section content bytes.
    /// - `align`: Alignment as power of 2 (e.g., 3 means 8-byte aligned).
    /// - `flags`: Section flags (e.g., S_ATTR_DEBUG).
    ///
    /// Returns the 0-based section index.
    pub fn add_custom_section(
        &mut self,
        sectname: &[u8],
        segname: &[u8],
        data: &[u8],
        align: u32,
        flags: u32,
    ) -> usize {
        let index = self.sections.len();
        self.sections.push(SectionData {
            sectname: sectname.to_vec(),
            segname: segname.to_vec(),
            data: data.to_vec(),
            align,
            flags,
            relocations: Vec::new(),
        });
        index
    }

    /// Add a symbol to the object file.
    ///
    /// - `name`: Symbol name (Mach-O convention adds '_' prefix; caller should
    ///   include it if desired, e.g., "_main").
    /// - `section`: 1-based section index (order of add_text_section / add_data_section calls).
    /// - `offset`: Byte offset within the section.
    /// - `is_global`: Whether the symbol is externally visible.
    pub fn add_symbol(&mut self, name: &str, section: usize, offset: u64, is_global: bool) {
        self.symbols.push(Symbol {
            name: name.to_string(),
            section: section as u8,
            value: offset,
            is_global,
        });
    }

    /// Add a relocation entry to the specified section.
    ///
    /// - `section`: 0-based section index.
    /// - `reloc`: The relocation entry.
    pub fn add_relocation(&mut self, section: usize, reloc: Relocation) {
        if section < self.sections.len() {
            self.sections[section].relocations.push(reloc);
        }
    }

    /// Produce the complete .o file as a byte vector.
    ///
    /// This assembles the entire Mach-O object file in memory:
    /// header, load commands, section data, relocations, symbol table,
    /// and string table.
    pub fn write(&self) -> Vec<u8> {
        let nsects = self.sections.len() as u32;

        // --- Compute load command sizes ---
        let segment_cmd_size = SEGMENT_COMMAND_64_SIZE + nsects * SECTION_64_SIZE;
        let build_version_size = BUILD_VERSION_COMMAND_SIZE;
        let symtab_size = SYMTAB_COMMAND_SIZE;
        let dysymtab_size = DYSYMTAB_COMMAND_SIZE;
        let total_lc_size = segment_cmd_size + build_version_size + symtab_size + dysymtab_size;

        // --- Compute section data offsets ---
        // Section data starts right after header + load commands.
        let header_plus_lc = MACH_HEADER_64_SIZE + total_lc_size;
        let mut section_offsets: Vec<u32> = Vec::new();
        let mut current_offset = header_plus_lc;

        for sec in &self.sections {
            // Align the offset for this section
            let alignment = 1u32 << sec.align;
            let misalign = current_offset % alignment;
            if misalign != 0 {
                current_offset += alignment - misalign;
            }
            section_offsets.push(current_offset);
            current_offset += sec.data.len() as u32;
        }

        let section_data_end = current_offset;

        // --- Compute relocation offsets ---
        let mut reloc_offsets: Vec<u32> = Vec::new();
        let mut reloc_offset = section_data_end;
        for sec in &self.sections {
            if sec.relocations.is_empty() {
                reloc_offsets.push(0);
            } else {
                reloc_offsets.push(reloc_offset);
                reloc_offset += sec.relocations.len() as u32 * RELOCATION_INFO_SIZE;
            }
        }
        let relocs_end = reloc_offset;

        // --- Build string table ---
        // String table starts with a null byte, then each symbol name
        // (null-terminated).
        let mut strtab = vec![0u8]; // index 0 = empty string
        let mut str_offsets: Vec<u32> = Vec::new();
        for sym in &self.symbols {
            str_offsets.push(strtab.len() as u32);
            strtab.extend_from_slice(sym.name.as_bytes());
            strtab.push(0);
        }

        // --- Symbol table offset ---
        let symtab_off = relocs_end;
        let nsyms = self.symbols.len() as u32;
        let strtab_off = symtab_off + nsyms * NLIST_64_SIZE;
        let strtab_size_val = strtab.len() as u32;

        // --- Compute vmsize (total virtual size of all sections) ---
        let vmsize = if self.sections.is_empty() {
            0u64
        } else {
            // vmsize = end of last section (addr + size)
            let mut vm_end = 0u64;
            let mut vm_addr = 0u64;
            for sec in &self.sections {
                let alignment = 1u64 << sec.align;
                let misalign = vm_addr % alignment;
                if misalign != 0 {
                    vm_addr += alignment - misalign;
                }
                vm_addr += sec.data.len() as u64;
                vm_end = vm_addr;
            }
            vm_end
        };

        // --- Compute filesize (total bytes of section data in file) ---
        let fileoff = if self.sections.is_empty() {
            0u64
        } else {
            section_offsets[0] as u64
        };
        let filesize = if self.sections.is_empty() {
            0u64
        } else {
            (section_data_end as u64) - fileoff
        };

        // --- Partition symbols: locals, external defined, undefined ---
        // Mach-O dysymtab requires this exact ordering:
        //   1. Local symbols (not external)
        //   2. External defined symbols (external + defined in a section)
        //   3. Undefined symbols (external + section == 0)
        let mut local_indices: Vec<usize> = Vec::new();
        let mut extdef_indices: Vec<usize> = Vec::new();
        let mut undef_indices: Vec<usize> = Vec::new();
        for (i, sym) in self.symbols.iter().enumerate() {
            if !sym.is_global {
                local_indices.push(i);
            } else if sym.section == 0 {
                undef_indices.push(i);
            } else {
                extdef_indices.push(i);
            }
        }

        // Build the symbol ordering: locals, then extdef, then undef.
        let ordered_indices: Vec<usize> = local_indices
            .iter()
            .chain(extdef_indices.iter())
            .chain(undef_indices.iter())
            .copied()
            .collect();

        let nlocalsym = local_indices.len() as u32;
        let nextdefsym = extdef_indices.len() as u32;
        let nundefsym = undef_indices.len() as u32;

        // --- Write the file ---
        let mut buf = Vec::with_capacity((strtab_off + strtab_size_val) as usize);

        // 1. Header
        let ncmds = 4u32; // LC_SEGMENT_64, LC_BUILD_VERSION, LC_SYMTAB, LC_DYSYMTAB
        let header = MachHeader::new_arm64_object(ncmds, total_lc_size);
        header.write(&mut buf);

        // 2. LC_SEGMENT_64 command
        let segment = SegmentCommand64::new_object(nsects, vmsize, fileoff, filesize);
        segment.write(&mut buf);

        // 3. Section headers (part of the LC_SEGMENT_64 command)
        let mut vm_addr = 0u64;
        for (i, sec) in self.sections.iter().enumerate() {
            let alignment = 1u64 << sec.align;
            let misalign = vm_addr % alignment;
            if misalign != 0 {
                vm_addr += alignment - misalign;
            }

            let section_header = Section64::new(
                &sec.sectname,
                &sec.segname,
                vm_addr,
                sec.data.len() as u64,
                section_offsets[i],
                sec.align,
                reloc_offsets[i],
                sec.relocations.len() as u32,
                sec.flags,
            );
            section_header.write(&mut buf);

            vm_addr += sec.data.len() as u64;
        }

        // 4. LC_BUILD_VERSION command
        self.write_build_version(&mut buf);

        // 5. LC_SYMTAB command
        self.write_symtab_command(&mut buf, symtab_off, nsyms, strtab_off, strtab_size_val);

        // 6. LC_DYSYMTAB command
        self.write_dysymtab_command(&mut buf, nlocalsym, nextdefsym, nundefsym);

        // 7. Section data (with alignment padding)
        for (i, sec) in self.sections.iter().enumerate() {
            let target = section_offsets[i] as usize;
            while buf.len() < target {
                buf.push(0);
            }
            buf.extend_from_slice(&sec.data);
        }

        // 8. Relocation entries
        for sec in &self.sections {
            for reloc in &sec.relocations {
                buf.extend_from_slice(&encode_relocation(reloc));
            }
        }

        // 9. Symbol table (nlist_64 entries)
        for &idx in &ordered_indices {
            let sym = &self.symbols[idx];
            self.write_nlist64(&mut buf, &str_offsets, idx, sym);
        }

        // 10. String table
        buf.extend_from_slice(&strtab);

        buf
    }

    /// Write the LC_BUILD_VERSION load command.
    ///
    /// Specifies macOS 14.0 as the minimum deployment target with no tool entries.
    fn write_build_version(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&LC_BUILD_VERSION.to_le_bytes()); // cmd
        buf.extend_from_slice(&BUILD_VERSION_COMMAND_SIZE.to_le_bytes()); // cmdsize
        buf.extend_from_slice(&PLATFORM_MACOS.to_le_bytes()); // platform
                                                              // minos: 14.0.0 encoded as 0x000E0000
        buf.extend_from_slice(&0x000E_0000u32.to_le_bytes());
        // sdk: 14.0.0
        buf.extend_from_slice(&0x000E_0000u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // ntools = 0
    }

    /// Write the LC_SYMTAB load command.
    fn write_symtab_command(
        &self,
        buf: &mut Vec<u8>,
        symoff: u32,
        nsyms: u32,
        stroff: u32,
        strsize: u32,
    ) {
        buf.extend_from_slice(&LC_SYMTAB.to_le_bytes()); // cmd
        buf.extend_from_slice(&SYMTAB_COMMAND_SIZE.to_le_bytes()); // cmdsize
        buf.extend_from_slice(&symoff.to_le_bytes());
        buf.extend_from_slice(&nsyms.to_le_bytes());
        buf.extend_from_slice(&stroff.to_le_bytes());
        buf.extend_from_slice(&strsize.to_le_bytes());
    }

    /// Write the LC_DYSYMTAB load command.
    fn write_dysymtab_command(
        &self,
        buf: &mut Vec<u8>,
        nlocalsym: u32,
        nextdefsym: u32,
        nundefsym: u32,
    ) {
        buf.extend_from_slice(&LC_DYSYMTAB.to_le_bytes()); // cmd
        buf.extend_from_slice(&DYSYMTAB_COMMAND_SIZE.to_le_bytes()); // cmdsize
        buf.extend_from_slice(&0u32.to_le_bytes()); // ilocalsym = 0
        buf.extend_from_slice(&nlocalsym.to_le_bytes()); // nlocalsym
        buf.extend_from_slice(&nlocalsym.to_le_bytes()); // iextdefsym = nlocalsym
        buf.extend_from_slice(&nextdefsym.to_le_bytes()); // nextdefsym
        let iundefsym = nlocalsym + nextdefsym;
        buf.extend_from_slice(&iundefsym.to_le_bytes()); // iundefsym
        buf.extend_from_slice(&nundefsym.to_le_bytes()); // nundefsym

        // Remaining fields are all zero for simple object files:
        // tocoff, ntoc, modtaboff, nmodtab, extrefsymoff, nextrefsyms,
        // indirectsymoff, nindirectsyms, extreloff, nextrel, locreloff, nlocrel
        for _ in 0..12 {
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
    }

    /// Write a single nlist_64 entry.
    fn write_nlist64(&self, buf: &mut Vec<u8>, str_offsets: &[u32], idx: usize, sym: &Symbol) {
        // n_strx: offset into string table (4 bytes)
        buf.extend_from_slice(&str_offsets[idx].to_le_bytes());

        // n_type: 1 byte
        let n_type = if sym.section == 0 {
            if sym.is_global {
                N_UNDF | N_EXT
            } else {
                N_UNDF
            }
        } else if sym.is_global {
            N_SECT | N_EXT
        } else {
            N_SECT
        };
        buf.push(n_type);

        // n_sect: 1 byte (1-based section ordinal, or 0 for N_UNDF)
        buf.push(sym.section);

        // n_desc: 2 bytes (0 for simple symbols)
        buf.extend_from_slice(&0u16.to_le_bytes());

        // n_value: 8 bytes
        // For defined symbols, this is the address (section base + offset).
        // We compute the virtual address based on section ordering.
        let value = if sym.section == 0 {
            0u64
        } else {
            let sec_idx = (sym.section - 1) as usize;
            let mut addr = 0u64;
            for (i, sec) in self.sections.iter().enumerate() {
                let alignment = 1u64 << sec.align;
                let misalign = addr % alignment;
                if misalign != 0 {
                    addr += alignment - misalign;
                }
                if i == sec_idx {
                    break;
                }
                addr += sec.data.len() as u64;
            }
            addr + sym.value
        };
        buf.extend_from_slice(&value.to_le_bytes());
    }
}

impl Default for MachOWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_writer() {
        let writer = MachOWriter::new();
        let bytes = writer.write();
        // Should at least have a header
        assert!(bytes.len() >= 32);
        // Check magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
    }

    #[test]
    fn test_writer_with_text() {
        let mut writer = MachOWriter::new();
        // 4 ARM64 NOPs
        let nop = 0xD503201Fu32;
        let mut code = Vec::new();
        for _ in 0..4 {
            code.extend_from_slice(&nop.to_le_bytes());
        }
        writer.add_text_section(&code);
        writer.add_symbol("_main", 1, 0, true);

        let bytes = writer.write();
        // Check magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
        // Check file type = MH_OBJECT
        let filetype = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(filetype, MH_OBJECT);
    }

    #[test]
    fn test_relocation_encoding() {
        use super::super::reloc::encode_relocation;

        let reloc = Relocation::branch26(0x10, 1);
        let encoded = encode_relocation(&reloc);
        // r_address = 0x10
        assert_eq!(&encoded[0..4], &0x10u32.to_le_bytes());
        // Packed: symbolnum=1, pcrel=1, length=2, extern=1, type=2
        // = 1 | (1<<24) | (2<<25) | (1<<27) | (2<<28)
        let expected: u32 = 1 | (1 << 24) | (2 << 25) | (1 << 27) | (2 << 28);
        assert_eq!(&encoded[4..8], &expected.to_le_bytes());
    }

    #[test]
    fn test_writer_with_both_sections() {
        let mut writer = MachOWriter::new();
        let nop = 0xD503201Fu32;
        let mut code = Vec::new();
        code.extend_from_slice(&nop.to_le_bytes());
        writer.add_text_section(&code);
        writer.add_data_section(&[1, 2, 3, 4, 5, 6, 7, 8]);
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_data", 2, 0, true);

        let bytes = writer.write();
        // Check ncmds = 4
        let ncmds = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(ncmds, 4);
    }

    #[test]
    fn test_symbol_ordering() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
        // Add a local then a global — dysymtab should sort locals before globals
        writer.add_symbol("_local_func", 1, 0, false);
        writer.add_symbol("_main", 1, 0, true);

        let bytes = writer.write();
        // Just verify it produces a valid-looking file
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
    }
}

// llvm2-codegen/macho/writer.rs - Mach-O object file writer
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
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
use super::x86_64_reloc::{encode_x86_64_relocation, X86_64Relocation};
use super::section::{Section64, SegmentCommand64};

/// Target CPU for the Mach-O object file.
///
/// Determines the CPU type in the Mach-O header and which relocation
/// encoding is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachOTarget {
    /// AArch64 (Apple Silicon).
    AArch64,
    /// x86-64 (Intel).
    X86_64,
}

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

/// A target-independent relocation that holds either AArch64 or x86-64 data.
#[derive(Debug, Clone)]
pub enum MachORelocation {
    /// AArch64 relocation.
    AArch64(Relocation),
    /// x86-64 relocation.
    X86_64(X86_64Relocation),
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
    /// AArch64 relocations for this section.
    relocations: Vec<Relocation>,
    /// x86-64 relocations for this section.
    x86_64_relocations: Vec<X86_64Relocation>,
}

/// Assembles a complete Mach-O 64-bit relocatable object file.
///
/// Supports both AArch64 and x86-64 targets. The target is selected at
/// construction time and determines the CPU type in the header, the text
/// section alignment, and which relocation encoding is used.
///
/// # Example
///
/// ```
/// use llvm2_codegen::macho::{MachOWriter, MachOTarget};
///
/// // AArch64 (default)
/// let mut writer = MachOWriter::new();
/// writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
/// writer.add_symbol("_main", 1, 0, true);
/// let bytes = writer.write();
///
/// // x86-64
/// let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
/// writer.add_text_section(&[0xC3]); // RET
/// writer.add_symbol("_main", 1, 0, true);
/// let bytes = writer.write();
/// ```
pub struct MachOWriter {
    /// Target CPU for this object file.
    target: MachOTarget,
    sections: Vec<SectionData>,
    symbols: Vec<Symbol>,
}

impl MachOWriter {
    /// Create a new empty Mach-O writer for AArch64 (default target).
    pub fn new() -> Self {
        Self {
            target: MachOTarget::AArch64,
            sections: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Create a new empty Mach-O writer for the specified target.
    pub fn for_target(target: MachOTarget) -> Self {
        Self {
            target,
            sections: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Returns the target CPU for this writer.
    pub fn target(&self) -> MachOTarget {
        self.target
    }

    /// Add a text section (__text in __TEXT) with the given machine code bytes.
    ///
    /// Alignment is chosen based on the target:
    /// - AArch64: 4-byte aligned (2^2) for fixed-width instructions
    /// - x86-64: 16-byte aligned (2^4) per System V ABI convention
    pub fn add_text_section(&mut self, code: &[u8]) {
        let align = match self.target {
            MachOTarget::AArch64 => 2, // 2^2 = 4-byte
            MachOTarget::X86_64 => 4,  // 2^4 = 16-byte
        };
        self.sections.push(SectionData {
            sectname: b"__text".to_vec(),
            segname: b"__TEXT".to_vec(),
            data: code.to_vec(),
            align,
            flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
            relocations: Vec::new(),
            x86_64_relocations: Vec::new(),
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
            x86_64_relocations: Vec::new(),
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
            x86_64_relocations: Vec::new(),
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

    /// Add an AArch64 relocation entry to the specified section.
    ///
    /// - `section`: 0-based section index.
    /// - `reloc`: The AArch64 relocation entry.
    pub fn add_relocation(&mut self, section: usize, reloc: Relocation) {
        if section < self.sections.len() {
            self.sections[section].relocations.push(reloc);
        }
    }

    /// Add an x86-64 relocation entry to the specified section.
    ///
    /// - `section`: 0-based section index.
    /// - `reloc`: The x86-64 relocation entry.
    pub fn add_x86_64_relocation(&mut self, section: usize, reloc: X86_64Relocation) {
        if section < self.sections.len() {
            self.sections[section].x86_64_relocations.push(reloc);
        }
    }

    /// Returns the total number of relocations for a section (across both targets).
    fn section_reloc_count(&self, sec: &SectionData) -> u32 {
        (sec.relocations.len() + sec.x86_64_relocations.len()) as u32
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
            let nreloc = self.section_reloc_count(sec);
            if nreloc == 0 {
                reloc_offsets.push(0);
            } else {
                reloc_offsets.push(reloc_offset);
                reloc_offset += nreloc * RELOCATION_INFO_SIZE;
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

        // 1. Header — target-aware
        let ncmds = 4u32; // LC_SEGMENT_64, LC_BUILD_VERSION, LC_SYMTAB, LC_DYSYMTAB
        let header = match self.target {
            MachOTarget::AArch64 => MachHeader::new_arm64_object(ncmds, total_lc_size),
            MachOTarget::X86_64 => MachHeader::new_x86_64_object(ncmds, total_lc_size),
        };
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
                self.section_reloc_count(sec),
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

        // 8. Relocation entries (AArch64 + x86-64)
        for sec in &self.sections {
            for reloc in &sec.relocations {
                buf.extend_from_slice(&encode_relocation(reloc));
            }
            for reloc in &sec.x86_64_relocations {
                buf.extend_from_slice(&encode_x86_64_relocation(reloc));
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

    #[test]
    fn test_got_relocations_in_object() {
        use super::super::reloc::{encode_relocation, Relocation};

        let mut writer = MachOWriter::new();
        // Two ARM64 instructions: ADRP + LDR (GOT-indirect pattern)
        let adrp = 0x9000_0000u32; // ADRP X0, #0
        let ldr = 0xF940_0000u32;  // LDR X0, [X0, #0]
        let mut code = Vec::new();
        code.extend_from_slice(&adrp.to_le_bytes());
        code.extend_from_slice(&ldr.to_le_bytes());
        writer.add_text_section(&code);

        // External symbol for GOT access
        writer.add_symbol("_printf", 0, 0, true); // undefined external

        // GOT relocations
        writer.add_relocation(0, Relocation::got_load_page21(0x00, 0));
        writer.add_relocation(0, Relocation::got_load_pageoff12(0x04, 0));

        let bytes = writer.write();
        // Verify valid Mach-O
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

        // Verify the GOT relocations encode correctly
        let got_page_encoded = encode_relocation(&Relocation::got_load_page21(0x00, 0));
        let r_word1 = u32::from_le_bytes([got_page_encoded[4], got_page_encoded[5],
                                           got_page_encoded[6], got_page_encoded[7]]);
        assert_eq!((r_word1 >> 28) & 0xF, 5, "GOT_LOAD_PAGE21 type = 5");
        assert_eq!((r_word1 >> 24) & 1, 1, "GOT_LOAD_PAGE21 is PC-relative");

        let got_off_encoded = encode_relocation(&Relocation::got_load_pageoff12(0x04, 0));
        let r_word1 = u32::from_le_bytes([got_off_encoded[4], got_off_encoded[5],
                                           got_off_encoded[6], got_off_encoded[7]]);
        assert_eq!((r_word1 >> 28) & 0xF, 6, "GOT_LOAD_PAGEOFF12 type = 6");
        assert_eq!((r_word1 >> 24) & 1, 0, "GOT_LOAD_PAGEOFF12 is not PC-relative");
    }

    #[test]
    fn test_tlvp_relocations_in_object() {
        use super::super::reloc::{encode_relocation, Relocation};

        let mut writer = MachOWriter::new();
        // Two ARM64 instructions: ADRP + LDR (TLV pattern)
        let adrp = 0x9000_0000u32;
        let ldr = 0xF940_0000u32;
        let mut code = Vec::new();
        code.extend_from_slice(&adrp.to_le_bytes());
        code.extend_from_slice(&ldr.to_le_bytes());
        writer.add_text_section(&code);

        // TLV symbol
        writer.add_symbol("_thread_var", 0, 0, true);

        // TLV relocations
        writer.add_relocation(0, Relocation::tlvp_load_page21(0x00, 0));
        writer.add_relocation(0, Relocation::tlvp_load_pageoff12(0x04, 0));

        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

        // Verify TLV relocation types
        let tlvp_page_encoded = encode_relocation(&Relocation::tlvp_load_page21(0x00, 0));
        let r_word1 = u32::from_le_bytes([tlvp_page_encoded[4], tlvp_page_encoded[5],
                                           tlvp_page_encoded[6], tlvp_page_encoded[7]]);
        assert_eq!((r_word1 >> 28) & 0xF, 8, "TLVP_LOAD_PAGE21 type = 8");
        assert_eq!((r_word1 >> 24) & 1, 1, "TLVP_LOAD_PAGE21 is PC-relative");

        let tlvp_off_encoded = encode_relocation(&Relocation::tlvp_load_pageoff12(0x04, 0));
        let r_word1 = u32::from_le_bytes([tlvp_off_encoded[4], tlvp_off_encoded[5],
                                           tlvp_off_encoded[6], tlvp_off_encoded[7]]);
        assert_eq!((r_word1 >> 28) & 0xF, 9, "TLVP_LOAD_PAGEOFF12 type = 9");
        assert_eq!((r_word1 >> 24) & 1, 0, "TLVP_LOAD_PAGEOFF12 is not PC-relative");
    }

    // =====================================================================
    // Additional coverage tests
    // =====================================================================

    // -- Helper to read a little-endian u32 from a byte slice --
    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn read_u64(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ])
    }

    #[test]
    fn test_empty_function_valid_macho_header() {
        // An empty writer (no sections, no symbols) should still produce a
        // structurally valid Mach-O header.
        let writer = MachOWriter::new();
        let bytes = writer.write();

        // Mach-O magic
        assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64);
        // CPU type = ARM64
        assert_eq!(read_u32(&bytes, 4), CPU_TYPE_ARM64);
        // CPU subtype = ALL
        assert_eq!(read_u32(&bytes, 8), CPU_SUBTYPE_ARM64_ALL);
        // File type = MH_OBJECT
        assert_eq!(read_u32(&bytes, 12), MH_OBJECT);
        // ncmds = 4 (segment, build_version, symtab, dysymtab)
        assert_eq!(read_u32(&bytes, 16), 4);
        // sizeofcmds should be non-zero
        let sizeofcmds = read_u32(&bytes, 20);
        assert!(sizeofcmds > 0);
        // flags
        let flags = read_u32(&bytes, 24);
        assert_eq!(flags & MH_SUBSECTIONS_VIA_SYMBOLS, MH_SUBSECTIONS_VIA_SYMBOLS);
    }

    #[test]
    fn test_single_text_section_alignment() {
        let mut writer = MachOWriter::new();
        // 16 bytes of ARM64 code (4 NOPs)
        let nop = 0xD503201Fu32;
        let code: Vec<u8> = (0..4).flat_map(|_| nop.to_le_bytes()).collect();
        writer.add_text_section(&code);

        let bytes = writer.write();
        let header_plus_lc = MACH_HEADER_64_SIZE
            + SEGMENT_COMMAND_64_SIZE
            + SECTION_64_SIZE
            + BUILD_VERSION_COMMAND_SIZE
            + SYMTAB_COMMAND_SIZE
            + DYSYMTAB_COMMAND_SIZE;

        // Section data should be aligned to 4 bytes (2^2).
        // Since header+lc is already a multiple of 4, offset = header_plus_lc.
        assert_eq!(header_plus_lc % 4, 0, "header+lc should be 4-byte aligned");

        // Verify the section data is present at the expected offset.
        let offset = header_plus_lc as usize;
        assert_eq!(
            read_u32(&bytes, offset),
            nop,
            "first instruction at section offset should be NOP"
        );
    }

    #[test]
    fn test_symbol_table_local_vs_external() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]); // 1 NOP

        // Add symbols: 2 locals, 1 global defined, 1 global undefined
        writer.add_symbol("_local1", 1, 0, false);
        writer.add_symbol("_local2", 1, 4, false);
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_extern_undef", 0, 0, true);

        let bytes = writer.write();

        // Find the LC_SYMTAB command to locate the symbol table.
        // It comes after LC_SEGMENT_64 + sections + LC_BUILD_VERSION.
        let seg_cmd_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
        let symtab_cmd_offset = (MACH_HEADER_64_SIZE + seg_cmd_size + BUILD_VERSION_COMMAND_SIZE) as usize;
        let symtab_cmd = read_u32(&bytes, symtab_cmd_offset);
        assert_eq!(symtab_cmd, LC_SYMTAB, "expected LC_SYMTAB command");

        let symoff = read_u32(&bytes, symtab_cmd_offset + 8) as usize;
        let nsyms = read_u32(&bytes, symtab_cmd_offset + 12);
        assert_eq!(nsyms, 4, "should have 4 symbols");

        let stroff = read_u32(&bytes, symtab_cmd_offset + 16) as usize;

        // Read the 4 nlist_64 entries (16 bytes each).
        // Mach-O requires locals first, then extdef, then undef.
        // Expected ordering: _local1, _local2, _main, _extern_undef.

        // Symbol 0: should be local (n_type = N_SECT, no N_EXT)
        let n_type_0 = bytes[symoff + 4]; // offset 4 in nlist_64
        assert_eq!(n_type_0, N_SECT, "symbol 0 should be local (N_SECT, no N_EXT)");

        // Symbol 1: should also be local
        let n_type_1 = bytes[symoff + 16 + 4];
        assert_eq!(n_type_1, N_SECT, "symbol 1 should be local");

        // Symbol 2: should be external defined (N_SECT | N_EXT)
        let n_type_2 = bytes[symoff + 32 + 4];
        assert_eq!(n_type_2, N_SECT | N_EXT, "symbol 2 should be global defined");

        // Symbol 3: should be undefined external (N_UNDF | N_EXT)
        let n_type_3 = bytes[symoff + 48 + 4];
        assert_eq!(n_type_3, N_UNDF | N_EXT, "symbol 3 should be undefined external");

        // Verify string table offsets point to valid strings.
        let str_offset_0 = read_u32(&bytes, symoff) as usize;
        assert!(stroff + str_offset_0 < bytes.len());
    }

    #[test]
    fn test_string_table_correctness() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
        writer.add_symbol("_foo", 1, 0, true);
        writer.add_symbol("_bar", 1, 0, false);

        let bytes = writer.write();

        // Find string table via LC_SYMTAB.
        let seg_cmd_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
        let symtab_cmd_offset = (MACH_HEADER_64_SIZE + seg_cmd_size + BUILD_VERSION_COMMAND_SIZE) as usize;
        let stroff = read_u32(&bytes, symtab_cmd_offset + 16) as usize;
        let strsize = read_u32(&bytes, symtab_cmd_offset + 20) as usize;

        // String table starts with a null byte.
        assert_eq!(bytes[stroff], 0, "string table must start with null byte");

        // Verify both symbol names appear in the string table.
        let strtab = &bytes[stroff..stroff + strsize];
        let strtab_str = String::from_utf8_lossy(strtab);
        assert!(strtab_str.contains("_foo"), "string table should contain _foo");
        assert!(strtab_str.contains("_bar"), "string table should contain _bar");
    }

    #[test]
    fn test_relocation_emission_in_section() {
        use super::super::reloc::Relocation;

        let mut writer = MachOWriter::new();
        // 3 ARM64 instructions
        let nop = 0xD503201Fu32;
        let code: Vec<u8> = (0..3).flat_map(|_| nop.to_le_bytes()).collect();
        writer.add_text_section(&code);
        writer.add_symbol("_callee", 0, 0, true); // undefined external

        // Add a BRANCH26 relocation at offset 4 (second instruction).
        writer.add_relocation(0, Relocation::branch26(0x04, 0));

        let bytes = writer.write();

        // The section header should record 1 relocation.
        // Section header is at: header(32) + segment_cmd(72) = offset 104.
        // section_64 layout: sectname(16) + segname(16) + addr(8) + size(8)
        //   + offset(4) + align(4) + reloff(4) + nreloc(4) + flags(4) + reserved(12) = 80
        let section_hdr_offset = (MACH_HEADER_64_SIZE + SEGMENT_COMMAND_64_SIZE) as usize;

        // reloff at offset 56 in section_64 (16+16+8+8+4+4=56)
        let reloff = read_u32(&bytes, section_hdr_offset + 56) as usize;
        assert!(reloff > 0, "relocation offset should be non-zero");

        // nreloc at offset 60 in section_64 (56+4=60)
        let nreloc = read_u32(&bytes, section_hdr_offset + 60);
        assert_eq!(nreloc, 1, "section should have 1 relocation");

        // Verify relocation address field (first 4 bytes of relocation_info).
        let r_address = read_u32(&bytes, reloff);
        assert_eq!(r_address, 0x04, "relocation should be at offset 0x04");
    }

    #[test]
    fn test_multi_section_output() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]); // 4 bytes text
        writer.add_data_section(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE]); // 8 bytes data

        // Add a custom section (e.g., compact unwind)
        writer.add_custom_section(
            b"__compact_unwind",
            b"__LD",
            &[0; 32], // 32 bytes of zero (compact unwind entry)
            3,         // 8-byte aligned
            S_REGULAR,
        );

        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_data_sym", 2, 0, true);

        let bytes = writer.write();

        // Header should show nsects = 3 in the segment command.
        let seg_cmd_offset = MACH_HEADER_64_SIZE as usize;
        // nsects is at offset 64 of segment_command_64:
        // cmd(4) + cmdsize(4) + segname(16) + vmaddr(8) + vmsize(8) +
        // fileoff(8) + filesize(8) + maxprot(4) + initprot(4) + nsects(4)
        let nsects = read_u32(&bytes, seg_cmd_offset + 64);
        assert_eq!(nsects, 3, "should have 3 sections");

        // Verify all section data is present in the file.
        // The text section data (NOP) should be somewhere in the file.
        let nop_bytes = &[0x1F, 0x20, 0x03, 0xD5];
        let found_nop = bytes.windows(4).any(|w| w == nop_bytes);
        assert!(found_nop, "text section NOP should be in the output");

        // The data section bytes should be somewhere in the file.
        let data_bytes = &[0xDE, 0xAD, 0xBE, 0xEF];
        let found_data = bytes.windows(4).any(|w| w == data_bytes);
        assert!(found_data, "data section content should be in the output");
    }

    #[test]
    fn test_dysymtab_partition_counts() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);

        // 2 locals, 3 globals defined, 1 undefined
        writer.add_symbol("_local_a", 1, 0, false);
        writer.add_symbol("_local_b", 1, 4, false);
        writer.add_symbol("_global_a", 1, 0, true);
        writer.add_symbol("_global_b", 1, 4, true);
        writer.add_symbol("_global_c", 1, 8, true);
        writer.add_symbol("_undef", 0, 0, true);

        let bytes = writer.write();

        // Find LC_DYSYMTAB command.
        let seg_cmd_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
        let dysymtab_offset = (MACH_HEADER_64_SIZE
            + seg_cmd_size
            + BUILD_VERSION_COMMAND_SIZE
            + SYMTAB_COMMAND_SIZE) as usize;
        let cmd = read_u32(&bytes, dysymtab_offset);
        assert_eq!(cmd, LC_DYSYMTAB, "should be LC_DYSYMTAB");

        // ilocalsym = 0
        let ilocalsym = read_u32(&bytes, dysymtab_offset + 8);
        assert_eq!(ilocalsym, 0);

        // nlocalsym = 2
        let nlocalsym = read_u32(&bytes, dysymtab_offset + 12);
        assert_eq!(nlocalsym, 2);

        // iextdefsym = nlocalsym = 2
        let iextdefsym = read_u32(&bytes, dysymtab_offset + 16);
        assert_eq!(iextdefsym, 2);

        // nextdefsym = 3
        let nextdefsym = read_u32(&bytes, dysymtab_offset + 20);
        assert_eq!(nextdefsym, 3);

        // iundefsym = nlocalsym + nextdefsym = 5
        let iundefsym = read_u32(&bytes, dysymtab_offset + 24);
        assert_eq!(iundefsym, 5);

        // nundefsym = 1
        let nundefsym = read_u32(&bytes, dysymtab_offset + 28);
        assert_eq!(nundefsym, 1);
    }

    #[test]
    fn test_symbol_value_computation() {
        // Symbol value should be the section base VM address + offset.
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0u8; 16]); // 16 bytes of code
        writer.add_symbol("_func_at_8", 1, 8, true);

        let bytes = writer.write();

        // Find symbol table.
        let seg_cmd_size = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
        let symtab_cmd_offset = (MACH_HEADER_64_SIZE + seg_cmd_size + BUILD_VERSION_COMMAND_SIZE) as usize;
        let symoff = read_u32(&bytes, symtab_cmd_offset + 8) as usize;

        // n_value is at offset 8 in nlist_64 (8 bytes).
        let n_value = read_u64(&bytes, symoff + 8);
        // Section 1 starts at vmaddr=0, so symbol value = 0 + 8.
        assert_eq!(n_value, 8, "symbol value should be section base + offset");
    }

    #[test]
    fn test_build_version_command() {
        let writer = MachOWriter::new();
        let bytes = writer.write();

        // LC_BUILD_VERSION is the second load command (after LC_SEGMENT_64).
        let seg_cmd_size = SEGMENT_COMMAND_64_SIZE; // no sections for empty writer
        let bv_offset = (MACH_HEADER_64_SIZE + seg_cmd_size) as usize;

        let cmd = read_u32(&bytes, bv_offset);
        assert_eq!(cmd, LC_BUILD_VERSION, "expected LC_BUILD_VERSION");

        let cmdsize = read_u32(&bytes, bv_offset + 4);
        assert_eq!(cmdsize, BUILD_VERSION_COMMAND_SIZE);

        let platform = read_u32(&bytes, bv_offset + 8);
        assert_eq!(platform, PLATFORM_MACOS, "platform should be macOS");

        // minos: 14.0.0 = 0x000E0000
        let minos = read_u32(&bytes, bv_offset + 12);
        assert_eq!(minos, 0x000E_0000, "minimum OS should be macOS 14.0");

        // ntools = 0
        let ntools = read_u32(&bytes, bv_offset + 20);
        assert_eq!(ntools, 0, "no tool entries");
    }

    #[test]
    fn test_default_impl() {
        // Verify the Default implementation works.
        let writer: MachOWriter = MachOWriter::default();
        let bytes = writer.write();
        assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64);
    }

    #[test]
    fn test_custom_section_index_returned() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0; 4]);
        let idx = writer.add_custom_section(b"__cstring", b"__TEXT", b"hello\0", 0, S_CSTRING_LITERALS);
        assert_eq!(idx, 1, "custom section should be at index 1 (after text at index 0)");
    }

    #[test]
    fn test_relocation_to_out_of_range_section_ignored() {
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0; 4]);

        // Adding a relocation to section 99 (out of range) should not crash.
        use super::super::reloc::Relocation;
        writer.add_relocation(99, Relocation::branch26(0, 0));

        // Should still produce valid output.
        let bytes = writer.write();
        assert_eq!(read_u32(&bytes, 0), MH_MAGIC_64);
    }

    #[test]
    fn test_segment_vmsize_and_filesize() {
        let mut writer = MachOWriter::new();
        // 8 bytes text + 16 bytes data
        writer.add_text_section(&[0u8; 8]);
        writer.add_data_section(&[0u8; 16]);

        let bytes = writer.write();

        // Segment command layout: cmd(4) + cmdsize(4) + segname(16) +
        // vmaddr(8, offset 24) + vmsize(8, offset 32) +
        // fileoff(8, offset 40) + filesize(8, offset 48)
        let seg_offset = MACH_HEADER_64_SIZE as usize;
        let vmsize = read_u64(&bytes, seg_offset + 32);
        let filesize = read_u64(&bytes, seg_offset + 48);

        // vmsize: text(8) aligned to data alignment (8-byte, so 8 is ok) + data(16) = 24
        assert!(vmsize >= 24, "vmsize should cover all sections: got {}", vmsize);
        // filesize should also cover all section data.
        assert!(filesize >= 24, "filesize should cover all sections: got {}", filesize);
    }

    // =====================================================================
    // x86-64 Mach-O writer tests
    // =====================================================================

    #[test]
    fn test_x86_64_empty_writer() {
        let writer = MachOWriter::for_target(MachOTarget::X86_64);
        let bytes = writer.write();
        // Should at least have a header
        assert!(bytes.len() >= 32);
        // Check magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
        // CPU type = x86-64
        assert_eq!(read_u32(&bytes, 4), CPU_TYPE_X86_64);
        // CPU subtype = ALL
        assert_eq!(read_u32(&bytes, 8), CPU_SUBTYPE_X86_64_ALL);
    }

    #[test]
    fn test_x86_64_writer_with_text() {
        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        // x86-64: push rbp; mov rbp, rsp; pop rbp; ret
        let code = vec![
            0x55,             // push rbp
            0x48, 0x89, 0xE5, // mov rbp, rsp
            0x5D,             // pop rbp
            0xC3,             // ret
        ];
        writer.add_text_section(&code);
        writer.add_symbol("_main", 1, 0, true);

        let bytes = writer.write();
        // Check magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
        // CPU type = x86-64
        assert_eq!(read_u32(&bytes, 4), CPU_TYPE_X86_64);
        // File type = MH_OBJECT
        assert_eq!(read_u32(&bytes, 12), MH_OBJECT);
    }

    #[test]
    fn test_x86_64_writer_with_relocation() {
        use super::super::x86_64_reloc::X86_64Relocation;

        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        // CALL rel32 (E8 + 4 bytes displacement)
        let code = vec![0xE8, 0x00, 0x00, 0x00, 0x00]; // call +0 (placeholder)
        writer.add_text_section(&code);
        writer.add_symbol("_callee", 0, 0, true); // undefined external

        // Add a BRANCH relocation at offset 1 (the displacement field)
        writer.add_x86_64_relocation(0, X86_64Relocation::branch(0x01, 0));

        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
        assert_eq!(read_u32(&bytes, 4), CPU_TYPE_X86_64);

        // Find section header to verify nreloc = 1
        let section_hdr_offset = (MACH_HEADER_64_SIZE + SEGMENT_COMMAND_64_SIZE) as usize;
        // nreloc at offset 60 in section_64
        let nreloc = read_u32(&bytes, section_hdr_offset + 60);
        assert_eq!(nreloc, 1, "section should have 1 relocation");
    }

    #[test]
    fn test_x86_64_writer_with_data_section() {
        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        writer.add_text_section(&[0xC3]); // ret
        writer.add_data_section(&[0xDE, 0xAD, 0xBE, 0xEF]);
        writer.add_symbol("_main", 1, 0, true);
        writer.add_symbol("_data", 2, 0, true);

        let bytes = writer.write();
        assert_eq!(read_u32(&bytes, 4), CPU_TYPE_X86_64);

        // Verify ncmds
        let ncmds = read_u32(&bytes, 16);
        assert_eq!(ncmds, 4);

        // Verify data appears in the output
        let found = bytes.windows(4).any(|w| w == &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(found, "data section content should be in the output");
    }

    #[test]
    fn test_x86_64_target_accessor() {
        let writer_arm = MachOWriter::new();
        assert_eq!(writer_arm.target(), MachOTarget::AArch64);

        let writer_x86 = MachOWriter::for_target(MachOTarget::X86_64);
        assert_eq!(writer_x86.target(), MachOTarget::X86_64);
    }

    #[test]
    fn test_x86_64_got_relocation() {
        use super::super::x86_64_reloc::X86_64Relocation;

        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        // mov rax, [rip + disp32] (GOT load pattern)
        let code = vec![0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00];
        writer.add_text_section(&code);
        writer.add_symbol("_extern_sym", 0, 0, true);

        writer.add_x86_64_relocation(0, X86_64Relocation::got_load(0x03, 0));

        let bytes = writer.write();
        assert_eq!(read_u32(&bytes, 4), CPU_TYPE_X86_64);

        // Verify relocation was emitted
        let section_hdr_offset = (MACH_HEADER_64_SIZE + SEGMENT_COMMAND_64_SIZE) as usize;
        let nreloc = read_u32(&bytes, section_hdr_offset + 60);
        assert_eq!(nreloc, 1);
    }

    #[test]
    fn test_x86_64_multiple_relocations() {
        use super::super::x86_64_reloc::X86_64Relocation;

        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        // Two calls + RIP-relative load
        let code = vec![0u8; 20];
        writer.add_text_section(&code);
        writer.add_symbol("_func1", 0, 0, true);
        writer.add_symbol("_func2", 0, 0, true);
        writer.add_symbol("_data", 0, 0, true);

        writer.add_x86_64_relocation(0, X86_64Relocation::branch(0x01, 0));
        writer.add_x86_64_relocation(0, X86_64Relocation::branch(0x06, 1));
        writer.add_x86_64_relocation(0, X86_64Relocation::signed(0x0C, 2));

        let bytes = writer.write();
        let section_hdr_offset = (MACH_HEADER_64_SIZE + SEGMENT_COMMAND_64_SIZE) as usize;
        let nreloc = read_u32(&bytes, section_hdr_offset + 60);
        assert_eq!(nreloc, 3, "section should have 3 relocations");
    }
}

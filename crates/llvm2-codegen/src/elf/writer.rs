// llvm2-codegen/elf/writer.rs - ELF64 object file writer
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Assembles a complete ELF64 relocatable object file (.o).
//!
//! Layout of a typical ET_REL file:
//!
//! ```text
//! ┌──────────────────────────┐  offset 0
//! │    ELF64 header          │  64 bytes
//! ├──────────────────────────┤
//! │    .text section data    │  (machine code)
//! ├──────────────────────────┤
//! │    .data section data    │  (initialized data)
//! ├──────────────────────────┤
//! │    .rela.text entries    │  24 bytes each
//! ├──────────────────────────┤
//! │    .symtab entries       │  24 bytes each
//! ├──────────────────────────┤
//! │    .strtab data          │  (symbol name strings)
//! ├──────────────────────────┤
//! │    .shstrtab data        │  (section name strings)
//! ├──────────────────────────┤
//! │    Section header table  │  64 bytes per entry
//! └──────────────────────────┘
//! ```
//!
//! Note: .bss (SHT_NOBITS) has a section header but occupies no file space.

use super::constants::*;
use super::debug::{DwarfDebugStubs, NoteGnuStack, SectionGroup};
use super::header::{Elf64Header, ElfMachine};
use super::reloc::Elf64Rela;
use super::section::Elf64Shdr;
use super::symbol::{Elf64Sym, ElfStringTable};

/// A symbol to be emitted in the ELF object file.
#[derive(Debug, Clone)]
pub struct ElfSymbol {
    /// Symbol name.
    pub name: String,
    /// Section index (0 = undefined, matches section header table index).
    pub section: u16,
    /// Offset within the section.
    pub value: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Whether the symbol is globally visible.
    pub is_global: bool,
    /// Symbol type (STT_NOTYPE, STT_FUNC, STT_OBJECT, etc.).
    pub sym_type: u8,
}

/// Internal section data held by the writer.
#[derive(Debug, Clone)]
struct SectionData {
    /// Section name (e.g., ".text").
    name: String,
    /// Section content bytes (empty for BSS).
    data: Vec<u8>,
    /// Section type (SHT_PROGBITS, SHT_NOBITS).
    sh_type: u32,
    /// Section flags.
    sh_flags: u64,
    /// Alignment as power of 2.
    align: u64,
    /// Relocations for this section.
    relocations: Vec<Elf64Rela>,
}

/// Assembles a complete ELF64 relocatable object file.
///
/// # Example
///
/// ```
/// use llvm2_codegen::elf::{ElfWriter, ElfMachine};
///
/// let mut writer = ElfWriter::new(ElfMachine::AArch64);
/// // ARM64 NOP = 0xD503201F
/// writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
/// writer.add_symbol("main", 1, 0, 0, true, 2); // STT_FUNC = 2
/// let bytes = writer.write();
/// // bytes now contains a valid ELF .o file
/// ```
pub struct ElfWriter {
    /// Target architecture.
    machine: ElfMachine,
    /// User-defined sections (.text, .data, .bss).
    sections: Vec<SectionData>,
    /// Symbols to emit.
    symbols: Vec<ElfSymbol>,
}

impl ElfWriter {
    /// Create a new empty ELF writer for the specified architecture.
    pub fn new(machine: ElfMachine) -> Self {
        Self {
            machine,
            sections: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Add a `.text` section (executable code) with the given machine code bytes.
    ///
    /// Returns the 1-based section index in the section header table.
    /// (Section 0 is always the null section.)
    pub fn add_text_section(&mut self, code: &[u8]) -> u16 {
        let align = match self.machine {
            ElfMachine::AArch64 => 4,  // ARM64 instructions are 4-byte aligned
            ElfMachine::X86_64 => 16,  // x86-64 typically 16-byte aligned
            ElfMachine::Riscv64 => 4,  // RISC-V instructions are 4-byte aligned
        };
        let idx = self.sections.len();
        self.sections.push(SectionData {
            name: ".text".to_string(),
            data: code.to_vec(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            align,
            relocations: Vec::new(),
        });
        // +1 for null section header at index 0
        (idx + 1) as u16
    }

    /// Add a `.data` section (initialized data) with the given data bytes.
    ///
    /// Returns the 1-based section index.
    pub fn add_data_section(&mut self, data: &[u8]) -> u16 {
        let idx = self.sections.len();
        self.sections.push(SectionData {
            name: ".data".to_string(),
            data: data.to_vec(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            align: 8,
            relocations: Vec::new(),
        });
        (idx + 1) as u16
    }

    /// Add a `.bss` section (uninitialized data) with the given size.
    ///
    /// BSS sections occupy no space in the file; only the section header
    /// records the size for runtime allocation.
    ///
    /// Returns the 1-based section index.
    pub fn add_bss_section(&mut self, size: u64) -> u16 {
        let idx = self.sections.len();
        // For NOBITS, we store a phantom vec whose length equals the BSS
        // logical size. The section header uses data.len() for sh_size, but
        // the data is never written to the file (NOBITS has no file content).
        let bss_data = if size > 0 {
            vec![0u8; size as usize]
        } else {
            Vec::new()
        };
        self.sections.push(SectionData {
            name: ".bss".to_string(),
            data: bss_data,
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            align: 8,
            relocations: Vec::new(),
        });
        (idx + 1) as u16
    }

    /// Add a symbol to the object file.
    ///
    /// - `name`: Symbol name.
    /// - `section`: 1-based section index (0 = undefined/external).
    /// - `offset`: Byte offset within the section.
    /// - `size`: Symbol size in bytes.
    /// - `is_global`: Whether the symbol is globally visible.
    /// - `sym_type`: Symbol type (STT_NOTYPE=0, STT_OBJECT=1, STT_FUNC=2).
    pub fn add_symbol(
        &mut self,
        name: &str,
        section: u16,
        offset: u64,
        size: u64,
        is_global: bool,
        sym_type: u8,
    ) {
        self.symbols.push(ElfSymbol {
            name: name.to_string(),
            section,
            value: offset,
            size,
            is_global,
            sym_type,
        });
    }

    /// Add a relocation entry to the specified section.
    ///
    /// - `section_idx`: 0-based index into the user sections (NOT the section
    ///   header table index). For example, if .text was the first section added,
    ///   use 0.
    /// - `rela`: The relocation entry.
    pub fn add_relocation(&mut self, section_idx: usize, rela: Elf64Rela) {
        if section_idx < self.sections.len() {
            self.sections[section_idx].relocations.push(rela);
        }
    }

    /// Add a generic named section with explicit type, flags, and data.
    ///
    /// Returns the 1-based section header table index.
    pub fn add_section(
        &mut self,
        name: &str,
        data: &[u8],
        sh_type: u32,
        sh_flags: u64,
        align: u64,
    ) -> u16 {
        let idx = self.sections.len();
        self.sections.push(SectionData {
            name: name.to_string(),
            data: data.to_vec(),
            sh_type,
            sh_flags,
            align,
            relocations: Vec::new(),
        });
        (idx + 1) as u16
    }

    /// Add minimal DWARF v5 debug section stubs (.debug_info, .debug_abbrev, .debug_line).
    ///
    /// These are minimal but structurally valid headers that tools like `readelf`
    /// can parse. Returns the 1-based section indices as `(debug_info, debug_abbrev, debug_line)`.
    pub fn add_debug_sections(&mut self) -> (u16, u16, u16) {
        let stubs = DwarfDebugStubs::new();
        let info_idx = self.add_section(
            DwarfDebugStubs::DEBUG_INFO_NAME,
            &stubs.debug_info_bytes(),
            SHT_PROGBITS,
            0,
            1,
        );
        let abbrev_idx = self.add_section(
            DwarfDebugStubs::DEBUG_ABBREV_NAME,
            &stubs.debug_abbrev_bytes(),
            SHT_PROGBITS,
            0,
            1,
        );
        let line_idx = self.add_section(
            DwarfDebugStubs::DEBUG_LINE_NAME,
            &stubs.debug_line_bytes(),
            SHT_PROGBITS,
            0,
            1,
        );
        (info_idx, abbrev_idx, line_idx)
    }

    /// Add a `.note.GNU-stack` section to mark the stack as non-executable.
    ///
    /// Returns the 1-based section header table index.
    pub fn add_note_gnu_stack(&mut self) -> u16 {
        let note = NoteGnuStack::new();
        self.add_section(
            NoteGnuStack::NAME,
            &note.section_bytes(),
            note.section_type(),
            note.section_flags(),
            note.section_align(),
        )
    }

    /// Add a section group (`.group`) for COMDAT deduplication.
    ///
    /// The `member_section_indices` are 1-based section header table indices
    /// of the sections that belong to this group.
    ///
    /// Returns the 1-based section header table index of the `.group` section.
    pub fn add_section_group(&mut self, member_section_indices: &[u32]) -> u16 {
        let group = SectionGroup::new(0, 0, member_section_indices.to_vec());
        let data = group.section_bytes();
        let idx = self.sections.len();
        self.sections.push(SectionData {
            name: SectionGroup::NAME.to_string(),
            data,
            sh_type: super::debug::SHT_GROUP,
            sh_flags: 0,
            align: SectionGroup::ALIGN,
            relocations: Vec::new(),
        });
        (idx + 1) as u16
    }

    /// Produce the complete ELF .o file as a byte vector.
    ///
    /// This assembles the entire ELF object file in memory: header,
    /// section data, relocations, symbol table, string tables, and
    /// the section header table (at the end).
    pub fn write(&self) -> Vec<u8> {
        // --- Build section header string table (.shstrtab) ---
        let mut shstrtab = ElfStringTable::new();

        // Pre-register all section names we'll need.
        // Collect name indices for each user section.
        let mut user_section_name_indices: Vec<u32> = Vec::new();
        for sec in &self.sections {
            let idx = shstrtab.add(&sec.name);
            user_section_name_indices.push(idx);
        }

        // Also register names for generated sections.
        // We need: .rela.X for each section with relocations, .symtab, .strtab, .shstrtab
        let mut rela_name_indices: Vec<Option<u32>> = Vec::new();
        for sec in &self.sections {
            if sec.relocations.is_empty() {
                rela_name_indices.push(None);
            } else {
                let rela_name = format!(".rela{}", sec.name);
                let idx = shstrtab.add(&rela_name);
                rela_name_indices.push(Some(idx));
            }
        }

        let symtab_name_idx = shstrtab.add(".symtab");
        let strtab_name_idx = shstrtab.add(".strtab");
        let shstrtab_name_idx = shstrtab.add(".shstrtab");

        // --- Build symbol string table (.strtab) ---
        let mut strtab = ElfStringTable::new();

        // Partition symbols: locals first, then globals (ELF requirement).
        let mut local_syms: Vec<&ElfSymbol> = Vec::new();
        let mut global_syms: Vec<&ElfSymbol> = Vec::new();
        for sym in &self.symbols {
            if sym.is_global {
                global_syms.push(sym);
            } else {
                local_syms.push(sym);
            }
        }

        // Build Elf64_Sym entries: null symbol + locals + globals.
        let mut sym_entries: Vec<Elf64Sym> = Vec::new();
        sym_entries.push(Elf64Sym::null()); // Index 0 is always the null symbol.

        // Map from original symbol index to final symbol table index.
        // (We need this for relocations, but since the user provides symbol
        //  indices directly in Elf64Rela, we assume they account for the
        //  null symbol at index 0.)

        for sym in local_syms.iter().chain(global_syms.iter()) {
            let name_idx = strtab.add(&sym.name);
            let binding = if sym.is_global { STB_GLOBAL } else { STB_LOCAL };
            sym_entries.push(Elf64Sym::new(
                name_idx,
                binding,
                sym.sym_type,
                STV_DEFAULT,
                sym.section,
                sym.value,
                sym.size,
            ));
        }

        // Number of local symbols (including null symbol): for .symtab sh_info.
        let num_local_plus_null = 1 + local_syms.len() as u32;

        // --- Compute section header table layout ---
        //
        // Section header table indices:
        //   0: null
        //   1..N: user sections (.text, .data, .bss, ...)
        //   N+1..M: .rela.X sections (one per user section with relocs)
        //   M+1: .symtab
        //   M+2: .strtab
        //   M+3: .shstrtab
        //
        // Total: 1 + num_user_sections + num_rela_sections + 3

        let num_user_sections = self.sections.len();
        let num_rela_sections = self.sections.iter().filter(|s| !s.relocations.is_empty()).count();
        let total_sections = 1 + num_user_sections + num_rela_sections + 3;

        // Compute .symtab and .strtab section header indices.
        let symtab_shdr_idx = (1 + num_user_sections + num_rela_sections) as u32;
        let strtab_shdr_idx = symtab_shdr_idx + 1;
        let shstrtab_shdr_idx = strtab_shdr_idx + 1;

        // --- Compute file offsets ---
        let mut offset: u64 = ELF64_EHDR_SIZE as u64;

        // User section data offsets (skip NOBITS sections).
        let mut section_offsets: Vec<u64> = Vec::new();
        for sec in &self.sections {
            if sec.sh_type == SHT_NOBITS {
                section_offsets.push(0); // BSS has no file offset
            } else {
                // Align to section alignment.
                let align = sec.align.max(1);
                let misalign = offset % align;
                if misalign != 0 {
                    offset += align - misalign;
                }
                section_offsets.push(offset);
                offset += sec.data.len() as u64;
            }
        }

        // .rela.X section offsets.
        // Align to 8 bytes.
        let align8 = |off: &mut u64| {
            let m = *off % 8;
            if m != 0 { *off += 8 - m; }
        };
        align8(&mut offset);

        let mut rela_offsets: Vec<Option<u64>> = Vec::new();
        for sec in &self.sections {
            if sec.relocations.is_empty() {
                rela_offsets.push(None);
            } else {
                rela_offsets.push(Some(offset));
                offset += (sec.relocations.len() * ELF64_RELA_SIZE) as u64;
            }
        }

        // .symtab offset.
        align8(&mut offset);
        let symtab_offset = offset;
        let symtab_size = (sym_entries.len() * ELF64_SYM_SIZE) as u64;
        offset += symtab_size;

        // .strtab offset.
        let strtab_offset = offset;
        let strtab_size = strtab.size() as u64;
        offset += strtab_size;

        // .shstrtab offset.
        let shstrtab_offset = offset;
        let shstrtab_size = shstrtab.size() as u64;
        offset += shstrtab_size;

        // Section header table offset (aligned to 8 bytes).
        align8(&mut offset);
        let sh_offset = offset;

        // --- Write the file ---
        let total_size = (sh_offset + (total_sections * ELF64_SHDR_SIZE) as u64) as usize;
        let mut buf = Vec::with_capacity(total_size);

        // 1. ELF header.
        let header = Elf64Header::new(
            self.machine,
            sh_offset,
            total_sections as u16,
            shstrtab_shdr_idx as u16,
        );
        header.write(&mut buf);

        // 2. User section data.
        for (i, sec) in self.sections.iter().enumerate() {
            if sec.sh_type == SHT_NOBITS {
                continue; // BSS has no file data
            }
            // Pad to section offset.
            let target = section_offsets[i] as usize;
            while buf.len() < target {
                buf.push(0);
            }
            buf.extend_from_slice(&sec.data);
        }

        // 3. .rela.X entries.
        for (i, sec) in self.sections.iter().enumerate() {
            if sec.relocations.is_empty() {
                continue;
            }
            if let Some(rela_off) = rela_offsets[i] {
                let target = rela_off as usize;
                while buf.len() < target {
                    buf.push(0);
                }
                for rela in &sec.relocations {
                    buf.extend_from_slice(&rela.encode());
                }
            }
        }

        // 4. .symtab entries.
        while buf.len() < symtab_offset as usize {
            buf.push(0);
        }
        for sym in &sym_entries {
            buf.extend_from_slice(&sym.encode());
        }

        // 5. .strtab data.
        debug_assert_eq!(buf.len(), strtab_offset as usize);
        buf.extend_from_slice(strtab.as_bytes());

        // 6. .shstrtab data.
        debug_assert_eq!(buf.len(), shstrtab_offset as usize);
        buf.extend_from_slice(shstrtab.as_bytes());

        // 7. Section header table.
        while buf.len() < sh_offset as usize {
            buf.push(0);
        }

        // Section 0: null.
        Elf64Shdr::null().write(&mut buf);

        // User sections.
        for (i, sec) in self.sections.iter().enumerate() {
            let shdr = match sec.sh_type {
                SHT_NOBITS => {
                    let mut shdr = Elf64Shdr::bss(
                        user_section_name_indices[i],
                        sec.data.len() as u64,
                        sec.align,
                    );
                    // Set the offset to the current file position for BSS
                    // (technically irrelevant since sh_type is NOBITS, but
                    // some tools expect it to be "reasonable").
                    shdr.sh_offset = section_offsets[i];
                    shdr
                }
                _ => {
                    let mut shdr = Elf64Shdr {
                        sh_name: user_section_name_indices[i],
                        sh_type: sec.sh_type,
                        sh_flags: sec.sh_flags,
                        sh_addr: 0,
                        sh_offset: section_offsets[i],
                        sh_size: sec.data.len() as u64,
                        sh_link: 0,
                        sh_info: 0,
                        sh_addralign: sec.align,
                        sh_entsize: 0,
                    };
                    let _ = &mut shdr; // suppress unused warning
                    shdr
                }
            };
            shdr.write(&mut buf);
        }

        // .rela.X sections.
        // We need to track the section header index for each user section
        // to fill in sh_info on the .rela sections.
        for (i, sec) in self.sections.iter().enumerate() {
            if sec.relocations.is_empty() {
                continue;
            }
            if let (Some(name_idx), Some(rela_off)) =
                (rela_name_indices[i], rela_offsets[i])
            {
                let rela_shdr = Elf64Shdr::rela(
                    name_idx,
                    rela_off,
                    (sec.relocations.len() * ELF64_RELA_SIZE) as u64,
                    symtab_shdr_idx,        // sh_link → .symtab
                    (i + 1) as u32,         // sh_info → user section index (1-based)
                );
                rela_shdr.write(&mut buf);
            }
        }

        // .symtab section.
        Elf64Shdr::symtab(
            symtab_name_idx,
            symtab_offset,
            symtab_size,
            strtab_shdr_idx,     // sh_link → .strtab
            num_local_plus_null, // sh_info → one past last local
        ).write(&mut buf);

        // .strtab section.
        Elf64Shdr::strtab(strtab_name_idx, strtab_offset, strtab_size)
            .write(&mut buf);

        // .shstrtab section.
        Elf64Shdr::strtab(shstrtab_name_idx, shstrtab_offset, shstrtab_size)
            .write(&mut buf);

        buf
    }
}

impl Default for ElfWriter {
    fn default() -> Self {
        Self::new(ElfMachine::AArch64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::reloc::{AArch64RelocType, X86_64RelocType};

    /// Helper: read a little-endian u16 from a byte slice.
    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    }

    /// Helper: read a little-endian u32 from a byte slice.
    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset], bytes[offset + 1],
            bytes[offset + 2], bytes[offset + 3],
        ])
    }

    /// Helper: read a little-endian u64 from a byte slice.
    fn read_u64(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
            bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7],
        ])
    }

    #[test]
    fn test_elf_magic_bytes() {
        let writer = ElfWriter::new(ElfMachine::AArch64);
        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F'],
            "ELF magic must be 7f 45 4c 46");
    }

    #[test]
    fn test_elf_header_class_and_endian() {
        let writer = ElfWriter::new(ElfMachine::AArch64);
        let bytes = writer.write();
        assert_eq!(bytes[4], ELFCLASS64, "must be 64-bit");
        assert_eq!(bytes[5], ELFDATA2LSB, "must be little-endian");
        assert_eq!(bytes[6], EV_CURRENT, "ELF version 1");
    }

    #[test]
    fn test_elf_type_is_rel() {
        let writer = ElfWriter::new(ElfMachine::AArch64);
        let bytes = writer.write();
        let e_type = read_u16(&bytes, 16);
        assert_eq!(e_type, ET_REL, "must be relocatable object");
    }

    #[test]
    fn test_elf_machine_aarch64() {
        let writer = ElfWriter::new(ElfMachine::AArch64);
        let bytes = writer.write();
        let e_machine = read_u16(&bytes, 18);
        assert_eq!(e_machine, EM_AARCH64);
    }

    #[test]
    fn test_elf_machine_x86_64() {
        let writer = ElfWriter::new(ElfMachine::X86_64);
        let bytes = writer.write();
        let e_machine = read_u16(&bytes, 18);
        assert_eq!(e_machine, EM_X86_64);
    }

    #[test]
    fn test_empty_writer_section_count() {
        // Empty writer should have: null + .symtab + .strtab + .shstrtab = 4 sections
        let writer = ElfWriter::new(ElfMachine::AArch64);
        let bytes = writer.write();
        let e_shnum = read_u16(&bytes, 60);
        assert_eq!(e_shnum, 4, "empty writer: null + symtab + strtab + shstrtab");
    }

    #[test]
    fn test_text_section_count() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]); // NOP
        let bytes = writer.write();
        let e_shnum = read_u16(&bytes, 60);
        // null + .text + .symtab + .strtab + .shstrtab = 5
        assert_eq!(e_shnum, 5);
    }

    #[test]
    fn test_text_and_data_section_count() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
        writer.add_data_section(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let bytes = writer.write();
        let e_shnum = read_u16(&bytes, 60);
        // null + .text + .data + .symtab + .strtab + .shstrtab = 6
        assert_eq!(e_shnum, 6);
    }

    #[test]
    fn test_text_with_relocs_section_count() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        let nop = 0xD503201Fu32;
        let code: Vec<u8> = (0..4).flat_map(|_| nop.to_le_bytes()).collect();
        writer.add_text_section(&code);
        writer.add_symbol("callee", 0, 0, 0, true, STT_NOTYPE);
        writer.add_relocation(0, Elf64Rela::aarch64(
            0x04, 1, AArch64RelocType::Call26, 0,
        ));
        let bytes = writer.write();
        let e_shnum = read_u16(&bytes, 60);
        // null + .text + .rela.text + .symtab + .strtab + .shstrtab = 6
        assert_eq!(e_shnum, 6);
    }

    #[test]
    fn test_bss_section() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 16]);
        let bss_idx = writer.add_bss_section(1024);
        assert_eq!(bss_idx, 2); // .text=1, .bss=2

        let bytes = writer.write();
        let e_shnum = read_u16(&bytes, 60);
        // null + .text + .bss + .symtab + .strtab + .shstrtab = 6
        assert_eq!(e_shnum, 6);

        // Find the BSS section header and verify it's SHT_NOBITS.
        let sh_offset = read_u64(&bytes, 40) as usize;
        // Section 2 (index 2) is .bss — at sh_offset + 2*64
        let bss_shdr_off = sh_offset + 2 * ELF64_SHDR_SIZE;
        let bss_type = read_u32(&bytes, bss_shdr_off + 4);
        assert_eq!(bss_type, SHT_NOBITS, ".bss must be SHT_NOBITS");
        let bss_size = read_u64(&bytes, bss_shdr_off + 32);
        assert_eq!(bss_size, 1024, ".bss size must be 1024");
    }

    #[test]
    fn test_symbol_table_entries() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 16]);
        writer.add_symbol("main", 1, 0, 16, true, STT_FUNC);
        writer.add_symbol("local_var", 1, 0, 4, false, STT_OBJECT);

        let bytes = writer.write();

        // Find .symtab section header.
        let sh_offset = read_u64(&bytes, 40) as usize;
        let e_shnum = read_u16(&bytes, 60) as usize;

        // Search for the .symtab section (sh_type == SHT_SYMTAB).
        let mut symtab_off = 0u64;
        let mut symtab_size = 0u64;
        let mut symtab_info = 0u32;
        for i in 0..e_shnum {
            let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
            let sh_type = read_u32(&bytes, shdr_off + 4);
            if sh_type == SHT_SYMTAB {
                symtab_off = read_u64(&bytes, shdr_off + 24);
                symtab_size = read_u64(&bytes, shdr_off + 32);
                symtab_info = read_u32(&bytes, shdr_off + 44);
                break;
            }
        }

        // We have: null symbol + local_var + main = 3 entries.
        let num_syms = symtab_size / ELF64_SYM_SIZE as u64;
        assert_eq!(num_syms, 3, "null + local_var + main = 3 symbols");

        // sh_info = one past last local = 2 (null + local_var).
        assert_eq!(symtab_info, 2, "sh_info should be 2 (null + 1 local)");

        // Verify null symbol at index 0.
        let sym0_off = symtab_off as usize;
        let sym0_info = bytes[sym0_off + 4];
        assert_eq!(sym0_info, 0, "null symbol st_info must be 0");

        // Verify local symbol at index 1 (local_var comes before main).
        let sym1_off = symtab_off as usize + ELF64_SYM_SIZE;
        let sym1_info = bytes[sym1_off + 4];
        assert_eq!(elf64_st_bind(sym1_info), STB_LOCAL, "symbol 1 should be local");

        // Verify global symbol at index 2.
        let sym2_off = symtab_off as usize + 2 * ELF64_SYM_SIZE;
        let sym2_info = bytes[sym2_off + 4];
        assert_eq!(elf64_st_bind(sym2_info), STB_GLOBAL, "symbol 2 should be global");
    }

    #[test]
    fn test_relocation_entries() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        // 16 bytes of ARM64 code (4 instructions).
        let nop = 0xD503201Fu32;
        let code: Vec<u8> = (0..4).flat_map(|_| nop.to_le_bytes()).collect();
        writer.add_text_section(&code);

        writer.add_symbol("callee", 0, 0, 0, true, STT_FUNC);

        // Add a CALL26 relocation at offset 4 (second instruction).
        writer.add_relocation(0, Elf64Rela::aarch64(
            4, 1, AArch64RelocType::Call26, 0,
        ));

        // Add an ADRP + ADD pair.
        writer.add_relocation(0, Elf64Rela::aarch64(
            8, 1, AArch64RelocType::AdrPrelPgHi21, 0,
        ));
        writer.add_relocation(0, Elf64Rela::aarch64(
            12, 1, AArch64RelocType::AddAbsLo12Nc, 0,
        ));

        let bytes = writer.write();

        // Find .rela.text section.
        let sh_offset = read_u64(&bytes, 40) as usize;
        let e_shnum = read_u16(&bytes, 60) as usize;

        let mut rela_off = 0u64;
        let mut rela_size = 0u64;
        for i in 0..e_shnum {
            let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
            let sh_type = read_u32(&bytes, shdr_off + 4);
            if sh_type == SHT_RELA {
                rela_off = read_u64(&bytes, shdr_off + 24);
                rela_size = read_u64(&bytes, shdr_off + 32);
                break;
            }
        }

        let num_relas = rela_size / ELF64_RELA_SIZE as u64;
        assert_eq!(num_relas, 3, "should have 3 relocation entries");

        // Verify first relocation (CALL26 at offset 4).
        let rela0_off = rela_off as usize;
        let r_offset = read_u64(&bytes, rela0_off);
        assert_eq!(r_offset, 4, "first reloc at offset 4");
        let r_info = read_u64(&bytes, rela0_off + 8);
        assert_eq!(elf64_r_type(r_info), R_AARCH64_CALL26);
        assert_eq!(elf64_r_sym(r_info), 1, "symbol index 1");
    }

    #[test]
    fn test_x86_64_relocation_entries() {
        let mut writer = ElfWriter::new(ElfMachine::X86_64);
        // 16 bytes of x86-64 NOPs.
        writer.add_text_section(&[0x90; 16]);

        writer.add_symbol("puts", 0, 0, 0, true, STT_FUNC);

        // PLT32 relocation (typical for function calls).
        writer.add_relocation(0, Elf64Rela::x86_64(
            5, 1, X86_64RelocType::Plt32, -4,
        ));

        // GOTPCREL relocation.
        writer.add_relocation(0, Elf64Rela::x86_64(
            10, 1, X86_64RelocType::GotPcRel, -4,
        ));

        let bytes = writer.write();

        // Verify machine type.
        let e_machine = read_u16(&bytes, 18);
        assert_eq!(e_machine, EM_X86_64);

        // Find .rela.text.
        let sh_offset = read_u64(&bytes, 40) as usize;
        let e_shnum = read_u16(&bytes, 60) as usize;

        let mut rela_off = 0u64;
        let mut rela_size = 0u64;
        for i in 0..e_shnum {
            let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
            let sh_type = read_u32(&bytes, shdr_off + 4);
            if sh_type == SHT_RELA {
                rela_off = read_u64(&bytes, shdr_off + 24);
                rela_size = read_u64(&bytes, shdr_off + 32);
                break;
            }
        }

        let num_relas = rela_size / ELF64_RELA_SIZE as u64;
        assert_eq!(num_relas, 2, "should have 2 relocation entries");

        // Verify PLT32 relocation.
        let r_info = read_u64(&bytes, rela_off as usize + 8);
        assert_eq!(elf64_r_type(r_info), R_X86_64_PLT32);

        // Verify addend = -4.
        let r_addend_bytes = &bytes[(rela_off as usize + 16)..(rela_off as usize + 24)];
        let r_addend = i64::from_le_bytes(r_addend_bytes.try_into().unwrap());
        assert_eq!(r_addend, -4);
    }

    #[test]
    fn test_shstrtab_index() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 4]);
        let bytes = writer.write();

        let e_shstrndx = read_u16(&bytes, 62);
        let e_shnum = read_u16(&bytes, 60);
        // .shstrtab is the last section.
        assert_eq!(e_shstrndx, e_shnum - 1);
    }

    #[test]
    fn test_null_section_is_all_zeros() {
        let writer = ElfWriter::new(ElfMachine::AArch64);
        let bytes = writer.write();

        let sh_offset = read_u64(&bytes, 40) as usize;
        // First section header (64 bytes) should be all zeros.
        let null_section = &bytes[sh_offset..sh_offset + ELF64_SHDR_SIZE];
        assert!(null_section.iter().all(|&b| b == 0), "null section must be all zeros");
    }

    #[test]
    fn test_text_section_data_present() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        let nop = 0xD503201Fu32;
        let code: Vec<u8> = nop.to_le_bytes().to_vec();
        writer.add_text_section(&code);

        let bytes = writer.write();

        // The NOP bytes should appear in the output.
        let nop_bytes = &[0x1F, 0x20, 0x03, 0xD5];
        let found = bytes.windows(4).any(|w| w == nop_bytes);
        assert!(found, "text section NOP data should be in the output");
    }

    #[test]
    fn test_default_impl() {
        let writer = ElfWriter::default();
        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);
        // Default is AArch64.
        let e_machine = read_u16(&bytes, 18);
        assert_eq!(e_machine, EM_AARCH64);
    }

    #[test]
    fn test_multiple_sections_with_symbols() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        let text_idx = writer.add_text_section(&[0; 32]);
        let data_idx = writer.add_data_section(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let bss_idx = writer.add_bss_section(256);

        writer.add_symbol("main", text_idx, 0, 32, true, STT_FUNC);
        writer.add_symbol("global_data", data_idx, 0, 4, true, STT_OBJECT);
        writer.add_symbol("bss_var", bss_idx, 0, 256, true, STT_OBJECT);
        writer.add_symbol("local_helper", text_idx, 16, 16, false, STT_FUNC);

        let bytes = writer.write();

        // Verify valid ELF header.
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);

        // Should have: null + .text + .data + .bss + .symtab + .strtab + .shstrtab = 7.
        let e_shnum = read_u16(&bytes, 60);
        assert_eq!(e_shnum, 7);

        // Find .symtab and check symbol count.
        let sh_offset = read_u64(&bytes, 40) as usize;
        for i in 0..e_shnum as usize {
            let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
            let sh_type = read_u32(&bytes, shdr_off + 4);
            if sh_type == SHT_SYMTAB {
                let size = read_u64(&bytes, shdr_off + 32);
                let num_syms = size / ELF64_SYM_SIZE as u64;
                // null + local_helper + main + global_data + bss_var = 5
                assert_eq!(num_syms, 5);
                break;
            }
        }
    }

    #[test]
    fn test_aarch64_got_relocations() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 8]);
        writer.add_symbol("extern_sym", 0, 0, 0, true, STT_NOTYPE);

        writer.add_relocation(0, Elf64Rela::aarch64(
            0, 1, AArch64RelocType::AdrGotPage, 0,
        ));
        writer.add_relocation(0, Elf64Rela::aarch64(
            4, 1, AArch64RelocType::Ld64GotLo12Nc, 0,
        ));

        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);

        // Find .rela.text and verify types.
        let sh_offset = read_u64(&bytes, 40) as usize;
        let e_shnum = read_u16(&bytes, 60) as usize;

        for i in 0..e_shnum {
            let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
            let sh_type = read_u32(&bytes, shdr_off + 4);
            if sh_type == SHT_RELA {
                let rela_off = read_u64(&bytes, shdr_off + 24) as usize;

                let r_info0 = read_u64(&bytes, rela_off + 8);
                assert_eq!(elf64_r_type(r_info0), R_AARCH64_ADR_GOT_PAGE);

                let r_info1 = read_u64(&bytes, rela_off + ELF64_RELA_SIZE + 8);
                assert_eq!(elf64_r_type(r_info1), R_AARCH64_LD64_GOT_LO12_NC);
                break;
            }
        }
    }

    #[test]
    fn test_aarch64_tls_relocations() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 8]);
        writer.add_symbol("tls_var", 0, 0, 0, true, STT_NOTYPE);

        writer.add_relocation(0, Elf64Rela::aarch64(
            0, 1, AArch64RelocType::TlsieAdrGottprelPage21, 0,
        ));
        writer.add_relocation(0, Elf64Rela::aarch64(
            4, 1, AArch64RelocType::TlsieLd64GottprelLo12Nc, 0,
        ));

        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);

        let sh_offset = read_u64(&bytes, 40) as usize;
        let e_shnum = read_u16(&bytes, 60) as usize;

        for i in 0..e_shnum {
            let shdr_off = sh_offset + i * ELF64_SHDR_SIZE;
            let sh_type = read_u32(&bytes, shdr_off + 4);
            if sh_type == SHT_RELA {
                let rela_off = read_u64(&bytes, shdr_off + 24) as usize;

                let r_info0 = read_u64(&bytes, rela_off + 8);
                assert_eq!(elf64_r_type(r_info0), R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21);

                let r_info1 = read_u64(&bytes, rela_off + ELF64_RELA_SIZE + 8);
                assert_eq!(elf64_r_type(r_info1), R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC);
                break;
            }
        }
    }

    #[test]
    fn test_relocation_to_out_of_range_section_ignored() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 4]);
        // Adding a relocation to section 99 (out of range) should not crash.
        writer.add_relocation(99, Elf64Rela::aarch64(
            0, 0, AArch64RelocType::Call26, 0,
        ));
        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_ehsize_field() {
        let writer = ElfWriter::new(ElfMachine::X86_64);
        let bytes = writer.write();
        let e_ehsize = read_u16(&bytes, 52);
        assert_eq!(e_ehsize, 64, "ELF64 header size must be 64");
    }

    #[test]
    fn test_shentsize_field() {
        let writer = ElfWriter::new(ElfMachine::X86_64);
        let bytes = writer.write();
        let e_shentsize = read_u16(&bytes, 58);
        assert_eq!(e_shentsize, 64, "section header entry size must be 64");
    }

    // --- Tests for new debug/metadata section writer methods ---

    #[test]
    fn test_add_debug_sections() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 4]);
        let (info_idx, abbrev_idx, line_idx) = writer.add_debug_sections();
        // .text=1, .debug_info=2, .debug_abbrev=3, .debug_line=4
        assert_eq!(info_idx, 2);
        assert_eq!(abbrev_idx, 3);
        assert_eq!(line_idx, 4);

        let bytes = writer.write();
        // null + .text + .debug_info + .debug_abbrev + .debug_line + .symtab + .strtab + .shstrtab = 8
        let e_shnum = read_u16(&bytes, 60);
        assert_eq!(e_shnum, 8);

        // Verify .debug_info section data is present in the output.
        // DWARF v5 debug_info starts with unit_length=8 as a u32 LE.
        let debug_info_marker = &[8u8, 0, 0, 0, 5, 0]; // unit_length=8, version=5
        let found = bytes.windows(6).any(|w| w == debug_info_marker);
        assert!(found, ".debug_info DWARF v5 header should be in output");
    }

    #[test]
    fn test_add_note_gnu_stack() {
        let mut writer = ElfWriter::new(ElfMachine::X86_64);
        writer.add_text_section(&[0x90; 16]);
        let stack_idx = writer.add_note_gnu_stack();
        assert_eq!(stack_idx, 2); // .text=1, .note.GNU-stack=2

        let bytes = writer.write();
        // null + .text + .note.GNU-stack + .symtab + .strtab + .shstrtab = 6
        let e_shnum = read_u16(&bytes, 60);
        assert_eq!(e_shnum, 6);

        // Find the .note.GNU-stack section header and verify its type.
        let sh_offset = read_u64(&bytes, 40) as usize;
        // Section 2 is .note.GNU-stack
        let note_shdr_off = sh_offset + 2 * ELF64_SHDR_SIZE;
        let note_type = read_u32(&bytes, note_shdr_off + 4);
        assert_eq!(note_type, SHT_PROGBITS, ".note.GNU-stack must be SHT_PROGBITS");
        // Flags must be 0 (no SHF_EXECINSTR)
        let note_flags = read_u64(&bytes, note_shdr_off + 8);
        assert_eq!(note_flags, 0, ".note.GNU-stack must have no executable flag");
        // Size must be 0 (empty marker section)
        let note_size = read_u64(&bytes, note_shdr_off + 32);
        assert_eq!(note_size, 0, ".note.GNU-stack must be empty");
    }

    #[test]
    fn test_add_section_group() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        writer.add_text_section(&[0; 8]);
        writer.add_data_section(&[1, 2, 3, 4]);
        // Group sections 1 and 2
        let group_idx = writer.add_section_group(&[1, 2]);
        assert_eq!(group_idx, 3); // .text=1, .data=2, .group=3

        let bytes = writer.write();
        // null + .text + .data + .group + .symtab + .strtab + .shstrtab = 7
        let e_shnum = read_u16(&bytes, 60);
        assert_eq!(e_shnum, 7);

        // Find the .group section header and verify its type.
        let sh_offset = read_u64(&bytes, 40) as usize;
        let group_shdr_off = sh_offset + 3 * ELF64_SHDR_SIZE;
        let group_type = read_u32(&bytes, group_shdr_off + 4);
        assert_eq!(group_type, super::super::debug::SHT_GROUP, ".group must be SHT_GROUP");

        // Verify the section data contains GRP_COMDAT flag + member indices
        let group_offset = read_u64(&bytes, group_shdr_off + 24) as usize;
        let group_size = read_u64(&bytes, group_shdr_off + 32) as usize;
        // flag(4) + 2 members(8) = 12 bytes
        assert_eq!(group_size, 12);
        let grp_flag = read_u32(&bytes, group_offset);
        assert_eq!(grp_flag, super::super::debug::GRP_COMDAT);
        let member0 = read_u32(&bytes, group_offset + 4);
        assert_eq!(member0, 1, "first member should be section 1");
        let member1 = read_u32(&bytes, group_offset + 8);
        assert_eq!(member1, 2, "second member should be section 2");
    }

    #[test]
    fn test_add_generic_section() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        let rodata = b"Hello, LLVM2!";
        let idx = writer.add_section(".rodata", rodata, SHT_PROGBITS, SHF_ALLOC, 4);
        assert_eq!(idx, 1);

        let bytes = writer.write();
        // The rodata string should appear in the output.
        let found = bytes.windows(rodata.len()).any(|w| w == rodata);
        assert!(found, ".rodata data should be in the output");
    }

    #[test]
    fn test_full_object_with_debug_and_stack() {
        let mut writer = ElfWriter::new(ElfMachine::AArch64);
        // ARM64 NOP = 0xD503201F
        let nop: Vec<u8> = (0..4).flat_map(|_| 0xD503201Fu32.to_le_bytes()).collect();
        writer.add_text_section(&nop);
        writer.add_data_section(&[0xCA, 0xFE]);
        writer.add_debug_sections();
        writer.add_note_gnu_stack();
        writer.add_symbol("main", 1, 0, 16, true, STT_FUNC);

        let bytes = writer.write();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);
        // null + .text + .data + .debug_info + .debug_abbrev + .debug_line
        //   + .note.GNU-stack + .symtab + .strtab + .shstrtab = 10
        let e_shnum = read_u16(&bytes, 60);
        assert_eq!(e_shnum, 10);
    }
}

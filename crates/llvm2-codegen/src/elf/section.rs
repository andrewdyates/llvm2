// llvm2-codegen/elf/section.rs - ELF64 section header table entries
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! ELF64 section header structures and serialization.
//!
//! Each section in an ELF file has a corresponding entry in the section header
//! table. The section header describes the section's name, type, flags, size,
//! and file offset.
//!
//! Section header entry layout (Elf64_Shdr, 64 bytes):
//! ```text
//! Offset  Size  Field
//! 0        4    sh_name      (index into .shstrtab)
//! 4        4    sh_type      (SHT_PROGBITS, SHT_SYMTAB, etc.)
//! 8        8    sh_flags     (SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR, etc.)
//! 16       8    sh_addr      (virtual address, 0 for relocatable)
//! 24       8    sh_offset    (file offset of section data)
//! 32       8    sh_size      (section size in bytes)
//! 40       4    sh_link      (section index link, type-dependent)
//! 44       4    sh_info      (extra info, type-dependent)
//! 48       8    sh_addralign (alignment constraint)
//! 56       8    sh_entsize   (entry size for fixed-size entries, else 0)
//! ```

use super::constants::*;

/// An ELF64 section header entry (Elf64_Shdr).
#[derive(Debug, Clone)]
pub struct Elf64Shdr {
    /// Index into the section header string table (.shstrtab) for the section name.
    pub sh_name: u32,
    /// Section type (SHT_NULL, SHT_PROGBITS, SHT_SYMTAB, etc.).
    pub sh_type: u32,
    /// Section flags (SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR, etc.).
    pub sh_flags: u64,
    /// Virtual address of the section (0 for relocatable objects).
    pub sh_addr: u64,
    /// File offset of the section data.
    pub sh_offset: u64,
    /// Size of the section in bytes.
    pub sh_size: u64,
    /// Section index link (interpretation depends on sh_type).
    pub sh_link: u32,
    /// Extra information (interpretation depends on sh_type).
    pub sh_info: u32,
    /// Alignment constraint (must be a power of 2, or 0/1 for no alignment).
    pub sh_addralign: u64,
    /// Size of each entry for sections with fixed-size entries (e.g., .symtab), else 0.
    pub sh_entsize: u64,
}

impl Elf64Shdr {
    /// Create the null section header (index 0, all zeros).
    pub fn null() -> Self {
        Self {
            sh_name: 0,
            sh_type: SHT_NULL,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: 0,
            sh_size: 0,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 0,
            sh_entsize: 0,
        }
    }

    /// Create a `.text` section header (executable code).
    pub fn text(name_idx: u32, offset: u64, size: u64, align: u64) -> Self {
        Self {
            sh_name: name_idx,
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            sh_addr: 0,
            sh_offset: offset,
            sh_size: size,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: align,
            sh_entsize: 0,
        }
    }

    /// Create a `.data` section header (initialized data).
    pub fn data(name_idx: u32, offset: u64, size: u64, align: u64) -> Self {
        Self {
            sh_name: name_idx,
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            sh_addr: 0,
            sh_offset: offset,
            sh_size: size,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: align,
            sh_entsize: 0,
        }
    }

    /// Create a `.bss` section header (uninitialized data).
    pub fn bss(name_idx: u32, size: u64, align: u64) -> Self {
        Self {
            sh_name: name_idx,
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            sh_addr: 0,
            sh_offset: 0, // BSS takes no file space
            sh_size: size,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: align,
            sh_entsize: 0,
        }
    }

    /// Create a `.symtab` section header.
    ///
    /// - `link`: section header index of the associated string table (.strtab).
    /// - `info`: one greater than the symbol table index of the last local symbol.
    pub fn symtab(name_idx: u32, offset: u64, size: u64, link: u32, info: u32) -> Self {
        Self {
            sh_name: name_idx,
            sh_type: SHT_SYMTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: offset,
            sh_size: size,
            sh_link: link,
            sh_info: info,
            sh_addralign: 8,
            sh_entsize: ELF64_SYM_SIZE as u64,
        }
    }

    /// Create a string table section header (.strtab or .shstrtab).
    pub fn strtab(name_idx: u32, offset: u64, size: u64) -> Self {
        Self {
            sh_name: name_idx,
            sh_type: SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: offset,
            sh_size: size,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 1,
            sh_entsize: 0,
        }
    }

    /// Create a `.rela.text` section header (relocations with addends).
    ///
    /// - `link`: section header index of the associated symbol table (.symtab).
    /// - `info`: section header index of the section to which the relocations apply (.text).
    pub fn rela(name_idx: u32, offset: u64, size: u64, link: u32, info: u32) -> Self {
        Self {
            sh_name: name_idx,
            sh_type: SHT_RELA,
            sh_flags: SHF_INFO_LINK,
            sh_addr: 0,
            sh_offset: offset,
            sh_size: size,
            sh_link: link,
            sh_info: info,
            sh_addralign: 8,
            sh_entsize: ELF64_RELA_SIZE as u64,
        }
    }

    /// Serialize the section header to bytes (little-endian, 64 bytes).
    pub fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.sh_name.to_le_bytes());
        buf.extend_from_slice(&self.sh_type.to_le_bytes());
        buf.extend_from_slice(&self.sh_flags.to_le_bytes());
        buf.extend_from_slice(&self.sh_addr.to_le_bytes());
        buf.extend_from_slice(&self.sh_offset.to_le_bytes());
        buf.extend_from_slice(&self.sh_size.to_le_bytes());
        buf.extend_from_slice(&self.sh_link.to_le_bytes());
        buf.extend_from_slice(&self.sh_info.to_le_bytes());
        buf.extend_from_slice(&self.sh_addralign.to_le_bytes());
        buf.extend_from_slice(&self.sh_entsize.to_le_bytes());
    }

    /// Size of the serialized section header entry in bytes.
    pub fn size() -> usize {
        ELF64_SHDR_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shdr_size() {
        let shdr = Elf64Shdr::null();
        let mut buf = Vec::new();
        shdr.write(&mut buf);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_null_section() {
        let shdr = Elf64Shdr::null();
        let mut buf = Vec::new();
        shdr.write(&mut buf);
        // All bytes should be zero
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_text_section_flags() {
        let shdr = Elf64Shdr::text(1, 0x40, 16, 4);
        assert_eq!(shdr.sh_type, SHT_PROGBITS);
        assert_eq!(shdr.sh_flags, SHF_ALLOC | SHF_EXECINSTR);
    }

    #[test]
    fn test_data_section_flags() {
        let shdr = Elf64Shdr::data(2, 0x50, 8, 8);
        assert_eq!(shdr.sh_type, SHT_PROGBITS);
        assert_eq!(shdr.sh_flags, SHF_ALLOC | SHF_WRITE);
    }

    #[test]
    fn test_bss_section() {
        let shdr = Elf64Shdr::bss(3, 256, 16);
        assert_eq!(shdr.sh_type, SHT_NOBITS);
        assert_eq!(shdr.sh_flags, SHF_ALLOC | SHF_WRITE);
        assert_eq!(shdr.sh_offset, 0, "BSS should not occupy file space");
    }

    #[test]
    fn test_symtab_section() {
        let shdr = Elf64Shdr::symtab(4, 0x100, 48, 5, 2);
        assert_eq!(shdr.sh_type, SHT_SYMTAB);
        assert_eq!(shdr.sh_entsize, ELF64_SYM_SIZE as u64);
        assert_eq!(shdr.sh_link, 5, "link to .strtab");
        assert_eq!(shdr.sh_info, 2, "one past last local");
    }

    #[test]
    fn test_rela_section() {
        let shdr = Elf64Shdr::rela(6, 0x200, 72, 4, 1);
        assert_eq!(shdr.sh_type, SHT_RELA);
        assert_eq!(shdr.sh_entsize, ELF64_RELA_SIZE as u64);
        assert_eq!(shdr.sh_flags, SHF_INFO_LINK);
        assert_eq!(shdr.sh_link, 4, "link to .symtab");
        assert_eq!(shdr.sh_info, 1, "applies to .text");
    }

    #[test]
    fn test_shdr_serialization_roundtrip() {
        let shdr = Elf64Shdr::text(1, 0x100, 32, 4);
        let mut buf = Vec::new();
        shdr.write(&mut buf);

        // Verify key fields by reading back
        let sh_name = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert_eq!(sh_name, 1);

        let sh_type = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(sh_type, SHT_PROGBITS);

        let sh_offset = u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27],
            buf[28], buf[29], buf[30], buf[31],
        ]);
        assert_eq!(sh_offset, 0x100);

        let sh_size = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35],
            buf[36], buf[37], buf[38], buf[39],
        ]);
        assert_eq!(sh_size, 32);
    }
}

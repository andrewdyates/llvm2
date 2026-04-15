// llvm2-codegen/elf/symbol.rs - ELF64 symbol table and string table
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: System V ABI, "Symbol Table" section
// Elf64_Sym layout (24 bytes):
//   st_name:  u32  — index into string table
//   st_info:  u8   — symbol type and binding (ELF64_ST_INFO)
//   st_other: u8   — symbol visibility
//   st_shndx: u16  — section header index
//   st_value: u64  — symbol value
//   st_size:  u64  — symbol size

//! ELF64 symbol table (Elf64_Sym) and string table management.
//!
//! The symbol table stores information about symbols defined or referenced
//! in the object file. Each entry is 24 bytes. The first entry (index 0) is
//! always the null symbol (all zeros).
//!
//! The string table is a sequence of null-terminated strings, starting with
//! a null byte at index 0 (the empty string).

use super::constants::*;

/// A single ELF64 symbol table entry (Elf64_Sym).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elf64Sym {
    /// Index into the string table for this symbol's name.
    pub st_name: u32,
    /// Symbol type and binding attributes (use `elf64_st_info`).
    pub st_info: u8,
    /// Symbol visibility.
    pub st_other: u8,
    /// Section header table index for the section this symbol is defined in.
    pub st_shndx: u16,
    /// Symbol value (address or offset).
    pub st_value: u64,
    /// Symbol size in bytes (0 if unknown).
    pub st_size: u64,
}

impl Elf64Sym {
    /// Create the null symbol entry (index 0, all zeros).
    pub fn null() -> Self {
        Self {
            st_name: 0,
            st_info: 0,
            st_other: 0,
            st_shndx: 0,
            st_value: 0,
            st_size: 0,
        }
    }

    /// Create a symbol with the given attributes.
    pub fn new(
        st_name: u32,
        binding: u8,
        sym_type: u8,
        visibility: u8,
        section_index: u16,
        value: u64,
        size: u64,
    ) -> Self {
        Self {
            st_name,
            st_info: elf64_st_info(binding, sym_type),
            st_other: visibility,
            st_shndx: section_index,
            st_value: value,
            st_size: size,
        }
    }

    /// Encode this symbol entry to its 24-byte little-endian representation.
    pub fn encode(&self) -> [u8; ELF64_SYM_SIZE] {
        let mut buf = [0u8; ELF64_SYM_SIZE];
        buf[0..4].copy_from_slice(&self.st_name.to_le_bytes());
        buf[4] = self.st_info;
        buf[5] = self.st_other;
        buf[6..8].copy_from_slice(&self.st_shndx.to_le_bytes());
        buf[8..16].copy_from_slice(&self.st_value.to_le_bytes());
        buf[16..24].copy_from_slice(&self.st_size.to_le_bytes());
        buf
    }

    /// Decode a symbol entry from its 24-byte little-endian representation.
    pub fn decode(bytes: &[u8; ELF64_SYM_SIZE]) -> Self {
        Self {
            st_name: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            st_info: bytes[4],
            st_other: bytes[5],
            st_shndx: u16::from_le_bytes([bytes[6], bytes[7]]),
            st_value: u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
                bytes[12], bytes[13], bytes[14], bytes[15],
            ]),
            st_size: u64::from_le_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19],
                bytes[20], bytes[21], bytes[22], bytes[23],
            ]),
        }
    }

    /// Returns the binding attribute (STB_LOCAL, STB_GLOBAL, STB_WEAK).
    pub fn binding(&self) -> u8 {
        elf64_st_bind(self.st_info)
    }

    /// Returns the type attribute (STT_NOTYPE, STT_FUNC, STT_OBJECT, etc.).
    pub fn sym_type(&self) -> u8 {
        elf64_st_type(self.st_info)
    }

    /// Returns true if this is a local symbol.
    pub fn is_local(&self) -> bool {
        self.binding() == STB_LOCAL
    }

    /// Returns true if this is a global symbol.
    pub fn is_global(&self) -> bool {
        self.binding() == STB_GLOBAL
    }

    /// Returns true if this symbol is undefined (not defined in this object).
    pub fn is_undefined(&self) -> bool {
        self.st_shndx == SHN_UNDEF && self.st_name != 0
    }
}

/// An ELF string table.
///
/// Stores null-terminated strings with the first byte being null (index 0 is
/// the empty string). Used for both `.strtab` (symbol names) and `.shstrtab`
/// (section names).
#[derive(Debug, Clone)]
pub struct ElfStringTable {
    /// The raw string table bytes.
    data: Vec<u8>,
}

impl ElfStringTable {
    /// Create a new string table with the leading null byte.
    pub fn new() -> Self {
        Self { data: vec![0] }
    }

    /// Add a string to the table and return its index.
    ///
    /// The string is null-terminated in the table. Duplicate strings are not
    /// deduplicated (for simplicity; the linker handles dedup).
    pub fn add(&mut self, s: &str) -> u32 {
        let idx = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);
        idx
    }

    /// Return the raw bytes of the string table.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Return the size of the string table in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl Default for ElfStringTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_symbol() {
        let sym = Elf64Sym::null();
        let encoded = sym.encode();
        assert!(encoded.iter().all(|&b| b == 0));
        assert_eq!(encoded.len(), 24);
    }

    #[test]
    fn test_symbol_encode_decode_roundtrip() {
        let sym = Elf64Sym::new(
            5,           // st_name
            STB_GLOBAL,
            STT_FUNC,
            STV_DEFAULT,
            1,           // section index
            0x1000,      // value
            64,          // size
        );

        let encoded = sym.encode();
        let decoded = Elf64Sym::decode(&encoded);
        assert_eq!(sym, decoded);
    }

    #[test]
    fn test_symbol_binding_and_type() {
        let sym = Elf64Sym::new(1, STB_GLOBAL, STT_FUNC, STV_DEFAULT, 1, 0, 0);
        assert_eq!(sym.binding(), STB_GLOBAL);
        assert_eq!(sym.sym_type(), STT_FUNC);
        assert!(sym.is_global());
        assert!(!sym.is_local());
    }

    #[test]
    fn test_symbol_local() {
        let sym = Elf64Sym::new(1, STB_LOCAL, STT_OBJECT, STV_DEFAULT, 2, 0, 8);
        assert!(sym.is_local());
        assert!(!sym.is_global());
        assert_eq!(sym.sym_type(), STT_OBJECT);
    }

    #[test]
    fn test_symbol_undefined() {
        let sym = Elf64Sym::new(1, STB_GLOBAL, STT_NOTYPE, STV_DEFAULT, SHN_UNDEF, 0, 0);
        assert!(sym.is_undefined());
    }

    #[test]
    fn test_null_symbol_not_undefined() {
        // The null symbol has st_name=0, so is_undefined should be false
        let sym = Elf64Sym::null();
        assert!(!sym.is_undefined());
    }

    #[test]
    fn test_string_table_basic() {
        let mut strtab = ElfStringTable::new();
        assert_eq!(strtab.as_bytes()[0], 0, "first byte must be null");

        let idx1 = strtab.add("main");
        assert_eq!(idx1, 1);

        let idx2 = strtab.add("printf");
        assert_eq!(idx2, 6); // 1 + "main\0" = 6

        let bytes = strtab.as_bytes();
        assert_eq!(bytes[0], 0);
        assert_eq!(&bytes[1..5], b"main");
        assert_eq!(bytes[5], 0);
        assert_eq!(&bytes[6..12], b"printf");
        assert_eq!(bytes[12], 0);
    }

    #[test]
    fn test_string_table_size() {
        let mut strtab = ElfStringTable::new();
        assert_eq!(strtab.size(), 1);
        strtab.add("foo");
        assert_eq!(strtab.size(), 5); // \0 + "foo\0"
    }

    #[test]
    fn test_st_info_encoding() {
        let info = elf64_st_info(STB_GLOBAL, STT_FUNC);
        assert_eq!(info, (1 << 4) | 2);
        assert_eq!(elf64_st_bind(info), STB_GLOBAL);
        assert_eq!(elf64_st_type(info), STT_FUNC);
    }
}

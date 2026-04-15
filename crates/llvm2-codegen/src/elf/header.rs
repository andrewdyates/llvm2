// llvm2-codegen/elf/header.rs - ELF64 file header emission
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! ELF64 file header structure and serialization.
//!
//! The ELF header appears at the very beginning of an ELF file and identifies
//! the file format, target architecture, and location of the section header
//! table. For relocatable object files (.o), the program header table is absent.

use super::constants::*;

/// Target machine architecture for the ELF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfMachine {
    /// ARM 64-bit architecture (AArch64).
    AArch64,
    /// AMD x86-64 architecture.
    X86_64,
}

impl ElfMachine {
    /// Return the ELF e_machine value for this architecture.
    pub fn to_e_machine(self) -> u16 {
        match self {
            ElfMachine::AArch64 => EM_AARCH64,
            ElfMachine::X86_64 => EM_X86_64,
        }
    }
}

/// ELF64 file header (Elf64_Ehdr).
///
/// Reference: System V ABI, Figure 3 "ELF Header"
///
/// ```text
/// Offset  Size  Field
/// 0       16    e_ident (magic, class, data, version, OS/ABI, padding)
/// 16       2    e_type
/// 18       2    e_machine
/// 20       4    e_version
/// 24       8    e_entry
/// 32       8    e_phoff
/// 40       8    e_shoff
/// 48       4    e_flags
/// 52       2    e_ehsize
/// 54       2    e_phentsize
/// 56       2    e_phnum
/// 58       2    e_shentsize
/// 60       2    e_shnum
/// 62       2    e_shstrndx
/// ```
#[derive(Debug, Clone)]
pub struct Elf64Header {
    /// Target machine type.
    pub machine: ElfMachine,
    /// Section header table file offset.
    pub sh_offset: u64,
    /// Number of section header entries.
    pub sh_num: u16,
    /// Section header string table index.
    pub sh_strndx: u16,
    /// Processor-specific flags.
    pub flags: u32,
}

impl Elf64Header {
    /// Create a new ELF64 header for a relocatable object file.
    pub fn new(machine: ElfMachine, sh_offset: u64, sh_num: u16, sh_strndx: u16) -> Self {
        Self {
            machine,
            sh_offset,
            sh_num,
            sh_strndx,
            flags: 0,
        }
    }

    /// Serialize the ELF64 header to bytes (little-endian, 64 bytes).
    pub fn write(&self, buf: &mut Vec<u8>) {
        // e_ident[0..16]
        let mut ident = [0u8; EI_NIDENT];
        ident[0] = ELFMAG0;       // 0x7f
        ident[1] = ELFMAG1;       // 'E'
        ident[2] = ELFMAG2;       // 'L'
        ident[3] = ELFMAG3;       // 'F'
        ident[4] = ELFCLASS64;    // 64-bit
        ident[5] = ELFDATA2LSB;   // Little-endian
        ident[6] = EV_CURRENT;    // ELF version 1
        ident[7] = ELFOSABI_NONE; // UNIX System V
        // ident[8..16] = 0 (padding)
        buf.extend_from_slice(&ident);

        // e_type: ET_REL
        buf.extend_from_slice(&ET_REL.to_le_bytes());
        // e_machine
        buf.extend_from_slice(&self.machine.to_e_machine().to_le_bytes());
        // e_version
        buf.extend_from_slice(&(EV_CURRENT as u32).to_le_bytes());
        // e_entry (0 for relocatable objects)
        buf.extend_from_slice(&0u64.to_le_bytes());
        // e_phoff (0 for relocatable objects — no program header table)
        buf.extend_from_slice(&0u64.to_le_bytes());
        // e_shoff
        buf.extend_from_slice(&self.sh_offset.to_le_bytes());
        // e_flags
        buf.extend_from_slice(&self.flags.to_le_bytes());
        // e_ehsize
        buf.extend_from_slice(&(ELF64_EHDR_SIZE as u16).to_le_bytes());
        // e_phentsize (0 — no program headers)
        buf.extend_from_slice(&0u16.to_le_bytes());
        // e_phnum (0)
        buf.extend_from_slice(&0u16.to_le_bytes());
        // e_shentsize
        buf.extend_from_slice(&(ELF64_SHDR_SIZE as u16).to_le_bytes());
        // e_shnum
        buf.extend_from_slice(&self.sh_num.to_le_bytes());
        // e_shstrndx
        buf.extend_from_slice(&self.sh_strndx.to_le_bytes());
    }

    /// Size of the serialized header in bytes.
    pub fn size() -> usize {
        ELF64_EHDR_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        let header = Elf64Header::new(ElfMachine::AArch64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_header_magic() {
        let header = Elf64Header::new(ElfMachine::AArch64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_header_class_and_data() {
        let header = Elf64Header::new(ElfMachine::X86_64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        assert_eq!(buf[4], ELFCLASS64, "should be 64-bit");
        assert_eq!(buf[5], ELFDATA2LSB, "should be little-endian");
    }

    #[test]
    fn test_header_machine_aarch64() {
        let header = Elf64Header::new(ElfMachine::AArch64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        let machine = u16::from_le_bytes([buf[18], buf[19]]);
        assert_eq!(machine, EM_AARCH64);
    }

    #[test]
    fn test_header_machine_x86_64() {
        let header = Elf64Header::new(ElfMachine::X86_64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        let machine = u16::from_le_bytes([buf[18], buf[19]]);
        assert_eq!(machine, EM_X86_64);
    }

    #[test]
    fn test_header_type_is_rel() {
        let header = Elf64Header::new(ElfMachine::AArch64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        let e_type = u16::from_le_bytes([buf[16], buf[17]]);
        assert_eq!(e_type, ET_REL);
    }

    #[test]
    fn test_header_sh_offset() {
        let header = Elf64Header::new(ElfMachine::AArch64, 0x1234, 8, 6);
        let mut buf = Vec::new();
        header.write(&mut buf);
        let sh_offset = u64::from_le_bytes([
            buf[40], buf[41], buf[42], buf[43],
            buf[44], buf[45], buf[46], buf[47],
        ]);
        assert_eq!(sh_offset, 0x1234);
        let sh_num = u16::from_le_bytes([buf[60], buf[61]]);
        assert_eq!(sh_num, 8);
        let sh_strndx = u16::from_le_bytes([buf[62], buf[63]]);
        assert_eq!(sh_strndx, 6);
    }

    #[test]
    fn test_header_no_program_headers() {
        let header = Elf64Header::new(ElfMachine::X86_64, 0, 0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        // e_phoff = 0
        let phoff = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35],
            buf[36], buf[37], buf[38], buf[39],
        ]);
        assert_eq!(phoff, 0);
        // e_phnum = 0
        let phnum = u16::from_le_bytes([buf[56], buf[57]]);
        assert_eq!(phnum, 0);
    }

    #[test]
    fn test_elf_machine_to_e_machine() {
        assert_eq!(ElfMachine::AArch64.to_e_machine(), 183);
        assert_eq!(ElfMachine::X86_64.to_e_machine(), 62);
    }
}

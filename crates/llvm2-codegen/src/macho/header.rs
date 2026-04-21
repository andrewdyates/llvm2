// llvm2-codegen/macho/header.rs - Mach-O header emission
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Mach-O 64-bit header structure and serialization.

use super::constants::*;

/// 64-bit Mach-O file header.
///
/// Appears at the very beginning of a Mach-O 64-bit object file.
/// Reference: struct mach_header_64 in mach-o/loader.h
#[derive(Debug, Clone)]
pub struct MachHeader {
    /// Magic number: MH_MAGIC_64 = 0xFEEDFACF.
    pub magic: u32,
    /// CPU type (e.g., CPU_TYPE_ARM64).
    pub cputype: u32,
    /// CPU subtype (e.g., CPU_SUBTYPE_ARM64_ALL).
    pub cpusubtype: u32,
    /// File type (e.g., MH_OBJECT for .o files).
    pub filetype: u32,
    /// Number of load commands following the header.
    pub ncmds: u32,
    /// Total size of all load commands in bytes.
    pub sizeofcmds: u32,
    /// Header flags (e.g., MH_SUBSECTIONS_VIA_SYMBOLS).
    pub flags: u32,
    /// Reserved field (must be 0).
    pub reserved: u32,
}

impl MachHeader {
    /// Create a new header for an ARM64 object file.
    pub fn new_arm64_object(ncmds: u32, sizeofcmds: u32) -> Self {
        Self {
            magic: MH_MAGIC_64,
            cputype: CPU_TYPE_ARM64,
            cpusubtype: CPU_SUBTYPE_ARM64_ALL,
            filetype: MH_OBJECT,
            ncmds,
            sizeofcmds,
            flags: MH_SUBSECTIONS_VIA_SYMBOLS,
            reserved: 0,
        }
    }

    /// Create a new header for an x86-64 object file.
    pub fn new_x86_64_object(ncmds: u32, sizeofcmds: u32) -> Self {
        Self {
            magic: MH_MAGIC_64,
            cputype: CPU_TYPE_X86_64,
            cpusubtype: CPU_SUBTYPE_X86_64_ALL,
            filetype: MH_OBJECT,
            ncmds,
            sizeofcmds,
            flags: MH_SUBSECTIONS_VIA_SYMBOLS,
            reserved: 0,
        }
    }

    /// Serialize the header to bytes (little-endian).
    pub fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic.to_le_bytes());
        buf.extend_from_slice(&self.cputype.to_le_bytes());
        buf.extend_from_slice(&self.cpusubtype.to_le_bytes());
        buf.extend_from_slice(&self.filetype.to_le_bytes());
        buf.extend_from_slice(&self.ncmds.to_le_bytes());
        buf.extend_from_slice(&self.sizeofcmds.to_le_bytes());
        buf.extend_from_slice(&self.flags.to_le_bytes());
        buf.extend_from_slice(&self.reserved.to_le_bytes());
    }

    /// Size of the serialized header in bytes.
    pub fn size() -> u32 {
        MACH_HEADER_64_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        let header = MachHeader::new_arm64_object(4, 360);
        let mut buf = Vec::new();
        header.write(&mut buf);
        assert_eq!(buf.len(), 32);
    }

    #[test]
    fn test_header_magic() {
        let header = MachHeader::new_arm64_object(0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        // First 4 bytes should be MH_MAGIC_64 in little-endian
        assert_eq!(&buf[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
    }

    #[test]
    fn test_header_cpu_type() {
        let header = MachHeader::new_arm64_object(0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        // CPU type at offset 4: CPU_TYPE_ARM64 = 0x0100000C
        let cputype = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(cputype, CPU_TYPE_ARM64);
    }

    #[test]
    fn test_x86_64_header_size() {
        let header = MachHeader::new_x86_64_object(4, 360);
        let mut buf = Vec::new();
        header.write(&mut buf);
        assert_eq!(buf.len(), 32);
    }

    #[test]
    fn test_x86_64_header_magic() {
        let header = MachHeader::new_x86_64_object(0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        assert_eq!(&buf[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
    }

    #[test]
    fn test_x86_64_header_cpu_type() {
        let header = MachHeader::new_x86_64_object(0, 0);
        let mut buf = Vec::new();
        header.write(&mut buf);
        // CPU type at offset 4: CPU_TYPE_X86_64 = 0x01000007
        let cputype = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(cputype, CPU_TYPE_X86_64);
        // CPU subtype at offset 8: CPU_SUBTYPE_X86_64_ALL = 3
        let cpusubtype = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        assert_eq!(cpusubtype, CPU_SUBTYPE_X86_64_ALL);
    }
}

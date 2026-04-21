// llvm2-codegen/macho/section.rs - Mach-O section and segment definitions
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Mach-O 64-bit section and segment command structures.

use super::constants::*;

/// 64-bit Mach-O section header.
///
/// Reference: struct section_64 in mach-o/loader.h
#[derive(Debug, Clone)]
pub struct Section64 {
    /// Section name (e.g., "__text"), padded to 16 bytes.
    pub sectname: [u8; 16],
    /// Segment name (e.g., "__TEXT"), padded to 16 bytes.
    pub segname: [u8; 16],
    /// Memory address of this section.
    pub addr: u64,
    /// Size in bytes.
    pub size: u64,
    /// File offset of section content.
    pub offset: u32,
    /// Section alignment as power of 2 (e.g., 2 means 4-byte aligned).
    pub align: u32,
    /// File offset of relocation entries.
    pub reloff: u32,
    /// Number of relocation entries.
    pub nreloc: u32,
    /// Flags (section type | section attributes).
    pub flags: u32,
    /// Reserved (used for indirect symbol table index in some section types).
    pub reserved1: u32,
    /// Reserved (used for stub size in some section types).
    pub reserved2: u32,
    /// Reserved.
    pub reserved3: u32,
}

impl Section64 {
    /// Create a new text section (__text in __TEXT segment).
    pub fn new_text(addr: u64, size: u64, offset: u32, reloff: u32, nreloc: u32) -> Self {
        Self {
            sectname: padded_name(b"__text"),
            segname: padded_name(b"__TEXT"),
            addr,
            size,
            offset,
            align: 2, // 2^2 = 4-byte alignment (ARM64 instruction alignment)
            reloff,
            nreloc,
            flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        }
    }

    /// Create a new data section (__data in __DATA segment).
    pub fn new_data(addr: u64, size: u64, offset: u32, reloff: u32, nreloc: u32) -> Self {
        Self {
            sectname: padded_name(b"__data"),
            segname: padded_name(b"__DATA"),
            addr,
            size,
            offset,
            align: 3, // 2^3 = 8-byte alignment
            reloff,
            nreloc,
            flags: S_REGULAR,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        }
    }

    /// Create a section with custom name, segment, flags, and alignment.
    pub fn new(
        sectname: &[u8],
        segname: &[u8],
        addr: u64,
        size: u64,
        offset: u32,
        align: u32,
        reloff: u32,
        nreloc: u32,
        flags: u32,
    ) -> Self {
        Self {
            sectname: padded_name(sectname),
            segname: padded_name(segname),
            addr,
            size,
            offset,
            align,
            reloff,
            nreloc,
            flags,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        }
    }

    /// Serialize the section header to bytes (little-endian).
    pub fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.sectname);
        buf.extend_from_slice(&self.segname);
        buf.extend_from_slice(&self.addr.to_le_bytes());
        buf.extend_from_slice(&self.size.to_le_bytes());
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.align.to_le_bytes());
        buf.extend_from_slice(&self.reloff.to_le_bytes());
        buf.extend_from_slice(&self.nreloc.to_le_bytes());
        buf.extend_from_slice(&self.flags.to_le_bytes());
        buf.extend_from_slice(&self.reserved1.to_le_bytes());
        buf.extend_from_slice(&self.reserved2.to_le_bytes());
        buf.extend_from_slice(&self.reserved3.to_le_bytes());
    }

    /// Size of the serialized section header in bytes.
    pub fn size() -> u32 {
        SECTION_64_SIZE
    }
}

/// 64-bit Mach-O segment command.
///
/// For MH_OBJECT files, all sections are in one unnamed segment.
/// Reference: struct segment_command_64 in mach-o/loader.h
#[derive(Debug, Clone)]
pub struct SegmentCommand64 {
    /// LC_SEGMENT_64
    pub cmd: u32,
    /// Size of this command including section headers.
    pub cmdsize: u32,
    /// Segment name (empty string for object files).
    pub segname: [u8; 16],
    /// Virtual memory address.
    pub vmaddr: u64,
    /// Virtual memory size.
    pub vmsize: u64,
    /// File offset of segment data.
    pub fileoff: u64,
    /// Size of segment data in the file.
    pub filesize: u64,
    /// Maximum VM protection.
    pub maxprot: i32,
    /// Initial VM protection.
    pub initprot: i32,
    /// Number of sections in this segment.
    pub nsects: u32,
    /// Segment flags.
    pub flags: u32,
}

impl SegmentCommand64 {
    /// Create a segment command for an object file (unnamed segment).
    ///
    /// In MH_OBJECT files, all sections live in one unnamed segment.
    pub fn new_object(
        nsects: u32,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
    ) -> Self {
        Self {
            cmd: LC_SEGMENT_64,
            cmdsize: SEGMENT_COMMAND_64_SIZE + nsects * SECTION_64_SIZE,
            segname: [0u8; 16], // unnamed for object files
            vmaddr: 0,
            vmsize,
            fileoff,
            filesize,
            maxprot: VM_PROT_ALL,
            initprot: VM_PROT_ALL,
            nsects,
            flags: 0,
        }
    }

    /// Serialize the segment command to bytes (little-endian).
    pub fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.cmd.to_le_bytes());
        buf.extend_from_slice(&self.cmdsize.to_le_bytes());
        buf.extend_from_slice(&self.segname);
        buf.extend_from_slice(&self.vmaddr.to_le_bytes());
        buf.extend_from_slice(&self.vmsize.to_le_bytes());
        buf.extend_from_slice(&self.fileoff.to_le_bytes());
        buf.extend_from_slice(&self.filesize.to_le_bytes());
        buf.extend_from_slice(&self.maxprot.to_le_bytes());
        buf.extend_from_slice(&self.initprot.to_le_bytes());
        buf.extend_from_slice(&self.nsects.to_le_bytes());
        buf.extend_from_slice(&self.flags.to_le_bytes());
    }

    /// Size of the serialized segment command in bytes (without sections).
    pub fn base_size() -> u32 {
        SEGMENT_COMMAND_64_SIZE
    }
}

/// Encode a name into a 16-byte zero-padded array.
pub fn padded_name(name: &[u8]) -> [u8; 16] {
    let mut buf = [0u8; 16];
    let len = name.len().min(16);
    buf[..len].copy_from_slice(&name[..len]);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_size() {
        let section = Section64::new_text(0, 16, 0, 0, 0);
        let mut buf = Vec::new();
        section.write(&mut buf);
        assert_eq!(buf.len(), 80);
    }

    #[test]
    fn test_segment_base_size() {
        let segment = SegmentCommand64::new_object(0, 0, 0, 0);
        let mut buf = Vec::new();
        segment.write(&mut buf);
        assert_eq!(buf.len(), 72);
    }

    #[test]
    fn test_padded_name() {
        let name = padded_name(b"__text");
        assert_eq!(&name[..6], b"__text");
        assert_eq!(&name[6..], &[0u8; 10]);
    }

    #[test]
    fn test_text_section_flags() {
        let section = Section64::new_text(0, 4, 0, 0, 0);
        assert_eq!(
            section.flags,
            S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS
        );
    }
}

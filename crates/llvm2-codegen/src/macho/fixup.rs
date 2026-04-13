// llvm2-codegen/macho/fixup.rs - Fixup layer for late relocation encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// The fixup layer sits between instruction encoding and relocation emission.
// During instruction encoding, fixups are recorded with placeholder values.
// After final layout (when section offsets are known), fixups are resolved
// into relocations and the instruction bytes are patched.

//! Fixup layer for deferred relocation resolution.
//!
//! When instructions are encoded, branch targets and address references may not
//! have their final values yet (e.g., forward references to labels, cross-section
//! references). The fixup layer records these unresolved references as [`Fixup`]
//! entries. After layout assigns final addresses, fixups are resolved: the
//! instruction bytes are patched with computed values, and [`Relocation`] entries
//! are generated for the linker.

use super::reloc::{AArch64RelocKind, Relocation};

/// The target of a fixup — what the fixup points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixupTarget {
    /// A named symbol (index into the symbol table).
    /// This generates an external relocation (r_extern=1).
    Symbol(u32),

    /// A section-relative offset. The `u32` is the section ordinal (1-based).
    /// Used for local references within or between sections.
    /// This generates a local relocation (r_extern=0).
    Section(u32),

    /// An expression: symbol + section offset.
    /// Used when we know both the symbol and want to add a section-relative
    /// adjustment (e.g., for .got stubs).
    SymbolPlusOffset {
        symbol_index: u32,
        section_offset: u64,
    },
}

/// A pending fixup that will be resolved after layout.
///
/// Fixups are created during instruction encoding when the final value is not
/// yet known. Each fixup records:
/// - Where in the section to apply the fix (`offset`)
/// - What kind of relocation it needs (`kind`)
/// - What it points at (`target`)
/// - An optional constant addend (`addend`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixup {
    /// Byte offset within the containing section where the fixup applies.
    /// This is the same as `relocation_info.r_address`.
    pub offset: u32,

    /// The ARM64 relocation kind. Determines how the value is encoded into
    /// the instruction and what relocation type is emitted.
    pub kind: AArch64RelocKind,

    /// What this fixup points at — a symbol, section, or expression.
    pub target: FixupTarget,

    /// Constant addend. For most instruction-embedded relocations this is 0.
    /// Non-zero addends on Branch26/Page21/Pageoff12 require an
    /// `ARM64_RELOC_ADDEND` relocation pair.
    pub addend: i64,
}

impl Fixup {
    /// Create a fixup for a branch instruction (B/BL) targeting a symbol.
    pub fn branch(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            kind: AArch64RelocKind::Branch26,
            target: FixupTarget::Symbol(symbol_index),
            addend: 0,
        }
    }

    /// Create a fixup for an ADRP instruction targeting a symbol's page.
    pub fn adrp(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            kind: AArch64RelocKind::Page21,
            target: FixupTarget::Symbol(symbol_index),
            addend: 0,
        }
    }

    /// Create a fixup for an ADD/LDR page offset targeting a symbol.
    pub fn pageoff(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            kind: AArch64RelocKind::Pageoff12,
            target: FixupTarget::Symbol(symbol_index),
            addend: 0,
        }
    }

    /// Create a fixup for an ADRP to a GOT entry.
    pub fn got_adrp(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            kind: AArch64RelocKind::GotLoadPage21,
            target: FixupTarget::Symbol(symbol_index),
            addend: 0,
        }
    }

    /// Create a fixup for an LDR GOT page offset.
    pub fn got_ldr(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            kind: AArch64RelocKind::GotLoadPageoff12,
            target: FixupTarget::Symbol(symbol_index),
            addend: 0,
        }
    }

    /// Create a fixup for a 64-bit absolute pointer.
    pub fn pointer(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            kind: AArch64RelocKind::Unsigned,
            target: FixupTarget::Symbol(symbol_index),
            addend: 0,
        }
    }

    /// Create a fixup with an addend.
    pub fn with_addend(mut self, addend: i64) -> Self {
        self.addend = addend;
        self
    }

    /// Returns true if this fixup has a non-zero addend that requires an
    /// `ARM64_RELOC_ADDEND` relocation pair.
    pub fn needs_addend_reloc(&self) -> bool {
        self.addend != 0
            && matches!(
                self.kind,
                AArch64RelocKind::Branch26
                    | AArch64RelocKind::Page21
                    | AArch64RelocKind::Pageoff12
            )
    }
}

/// A collection of fixups for a single section.
///
/// During instruction encoding, fixups are accumulated here. After layout,
/// [`resolve_fixups`] converts them to relocations and patches the instruction
/// bytes.
#[derive(Debug, Clone, Default)]
pub struct FixupList {
    fixups: Vec<Fixup>,
}

impl FixupList {
    /// Create an empty fixup list.
    pub fn new() -> Self {
        Self {
            fixups: Vec::new(),
        }
    }

    /// Add a fixup to the list.
    pub fn push(&mut self, fixup: Fixup) {
        self.fixups.push(fixup);
    }

    /// Number of fixups in the list.
    pub fn len(&self) -> usize {
        self.fixups.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.fixups.is_empty()
    }

    /// Iterate over fixups.
    pub fn iter(&self) -> impl Iterator<Item = &Fixup> {
        self.fixups.iter()
    }

    /// Get a fixup by index.
    pub fn get(&self, index: usize) -> Option<&Fixup> {
        self.fixups.get(index)
    }

    /// Resolve all fixups into relocations.
    ///
    /// This converts each fixup into one or more `Relocation` entries suitable
    /// for writing into the Mach-O relocation table. Fixups with non-zero
    /// addends on Branch26/Page21/Pageoff12 produce an additional
    /// `ARM64_RELOC_ADDEND` relocation.
    ///
    /// Note: this does NOT patch the instruction bytes. The caller is responsible
    /// for applying fixup values to the section data based on final layout.
    /// The relocations tell the linker what adjustments are needed at link time.
    pub fn resolve_to_relocations(&self) -> Vec<Relocation> {
        let mut relocs = Vec::with_capacity(self.fixups.len() * 2);

        for fixup in &self.fixups {
            let (symbol_index, is_extern) = match &fixup.target {
                FixupTarget::Symbol(idx) => (*idx, true),
                FixupTarget::Section(ordinal) => (*ordinal, false),
                FixupTarget::SymbolPlusOffset {
                    symbol_index, ..
                } => (*symbol_index, true),
            };

            // Emit addend relocation first if needed
            // Per Mach-O ABI: ARM64_RELOC_ADDEND must precede the main relocation
            if fixup.needs_addend_reloc() {
                let addend_i32 = fixup.addend as i32;
                relocs.push(Relocation {
                    offset: fixup.offset,
                    symbol_index: (addend_i32 as u32) & 0x00FF_FFFF,
                    kind: AArch64RelocKind::Addend,
                    pc_relative: false,
                    length: 2,
                    is_extern: false,
                });
            }

            relocs.push(Relocation {
                offset: fixup.offset,
                symbol_index,
                kind: fixup.kind,
                pc_relative: fixup.kind.is_pc_relative(),
                length: fixup.kind.default_log2_size(),
                is_extern,
            });
        }

        relocs
    }
}

/// Apply a Branch26 fixup value to instruction bytes.
///
/// The Branch26 value is a signed 26-bit word offset (byte offset >> 2).
/// It occupies bits [25:0] of the B/BL instruction encoding.
///
/// # Arguments
/// - `insn_bytes`: Mutable slice of 4 instruction bytes (little-endian).
/// - `byte_offset`: Signed byte offset from the instruction to the target.
///
/// # Panics
/// Panics if the offset is not 4-byte aligned or exceeds +/-128 MB.
pub fn apply_branch26(insn_bytes: &mut [u8; 4], byte_offset: i64) {
    assert!(
        byte_offset & 3 == 0,
        "Branch26 offset must be 4-byte aligned, got {}",
        byte_offset
    );

    let word_offset = byte_offset >> 2;
    assert!(
        word_offset >= -(1 << 25) && word_offset < (1 << 25),
        "Branch26 offset out of range: {} words ({} bytes)",
        word_offset,
        byte_offset
    );

    let imm26 = (word_offset as u32) & 0x03FF_FFFF;
    let insn = u32::from_le_bytes(*insn_bytes);
    let patched = (insn & 0xFC00_0000) | imm26;
    *insn_bytes = patched.to_le_bytes();
}

/// Apply a Page21 fixup value to ADRP instruction bytes.
///
/// The ADRP instruction encodes a 21-bit signed page offset as:
/// - `immhi` = bits [23:5] of the instruction (19 bits)
/// - `immlo` = bits [30:29] of the instruction (2 bits)
/// - Full value = `(immhi << 2) | immlo`, sign-extended from 21 bits
///
/// # Arguments
/// - `insn_bytes`: Mutable slice of 4 instruction bytes (little-endian).
/// - `page_offset`: Signed page offset (number of 4KB pages).
///
/// # Panics
/// Panics if the page offset exceeds +/-4 GB (21-bit signed range).
pub fn apply_page21(insn_bytes: &mut [u8; 4], page_offset: i64) {
    assert!(
        page_offset >= -(1 << 20) && page_offset < (1 << 20),
        "Page21 offset out of range: {} pages",
        page_offset
    );

    let imm21 = (page_offset as u32) & 0x001F_FFFF;
    let immlo = imm21 & 0x3;
    let immhi = (imm21 >> 2) & 0x7_FFFF;

    let insn = u32::from_le_bytes(*insn_bytes);
    let patched = (insn & 0x9F00_001F) | (immlo << 29) | (immhi << 5);
    *insn_bytes = patched.to_le_bytes();
}

/// Apply a Pageoff12 fixup value to ADD/LDR instruction bytes.
///
/// The 12-bit page offset is stored in bits [21:10] of the instruction.
/// For LDR instructions, the offset may be scaled by the access size.
///
/// # Arguments
/// - `insn_bytes`: Mutable slice of 4 instruction bytes (little-endian).
/// - `page_offset`: 12-bit unsigned offset within a 4KB page.
/// - `shift`: Scale factor (log2 of access size, 0 for ADD, 2 for LDR W, 3 for LDR X).
///
/// # Panics
/// Panics if the offset exceeds 12 bits or is not aligned to the shift.
pub fn apply_pageoff12(insn_bytes: &mut [u8; 4], page_offset: u32, shift: u8) {
    assert!(
        page_offset < 4096,
        "Pageoff12 value must be < 4096, got {}",
        page_offset
    );

    let scaled_offset = if shift > 0 {
        assert!(
            page_offset & ((1 << shift) - 1) == 0,
            "Pageoff12 value {} not aligned to {} bytes",
            page_offset,
            1 << shift
        );
        page_offset >> shift
    } else {
        page_offset
    };

    assert!(
        scaled_offset < (1 << 12),
        "Scaled pageoff12 value {} exceeds 12-bit field",
        scaled_offset
    );

    let insn = u32::from_le_bytes(*insn_bytes);
    let patched = (insn & 0xFFC0_03FF) | ((scaled_offset & 0xFFF) << 10);
    *insn_bytes = patched.to_le_bytes();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixup_branch() {
        let fixup = Fixup::branch(0x10, 5);
        assert_eq!(fixup.offset, 0x10);
        assert_eq!(fixup.kind, AArch64RelocKind::Branch26);
        assert_eq!(fixup.target, FixupTarget::Symbol(5));
        assert_eq!(fixup.addend, 0);
        assert!(!fixup.needs_addend_reloc());
    }

    #[test]
    fn test_fixup_with_addend() {
        let fixup = Fixup::branch(0x10, 5).with_addend(0x100);
        assert_eq!(fixup.addend, 0x100);
        assert!(fixup.needs_addend_reloc());
    }

    #[test]
    fn test_fixup_adrp_pageoff_pair() {
        let adrp = Fixup::adrp(0x00, 3);
        let add = Fixup::pageoff(0x04, 3);

        assert_eq!(adrp.kind, AArch64RelocKind::Page21);
        assert_eq!(add.kind, AArch64RelocKind::Pageoff12);
    }

    #[test]
    fn test_fixup_got_pair() {
        let got_adrp = Fixup::got_adrp(0x00, 7);
        let got_ldr = Fixup::got_ldr(0x04, 7);

        assert_eq!(got_adrp.kind, AArch64RelocKind::GotLoadPage21);
        assert_eq!(got_ldr.kind, AArch64RelocKind::GotLoadPageoff12);
    }

    #[test]
    fn test_fixup_list_resolve() {
        let mut list = FixupList::new();
        list.push(Fixup::branch(0x00, 1));
        list.push(Fixup::adrp(0x04, 2));
        list.push(Fixup::pageoff(0x08, 2));

        let relocs = list.resolve_to_relocations();
        assert_eq!(relocs.len(), 3);

        assert_eq!(relocs[0].kind, AArch64RelocKind::Branch26);
        assert_eq!(relocs[0].symbol_index, 1);
        assert!(relocs[0].is_extern);

        assert_eq!(relocs[1].kind, AArch64RelocKind::Page21);
        assert_eq!(relocs[1].symbol_index, 2);

        assert_eq!(relocs[2].kind, AArch64RelocKind::Pageoff12);
        assert_eq!(relocs[2].symbol_index, 2);
    }

    #[test]
    fn test_fixup_list_resolve_with_addend() {
        let mut list = FixupList::new();
        list.push(Fixup::branch(0x00, 1).with_addend(4));

        let relocs = list.resolve_to_relocations();
        assert_eq!(relocs.len(), 2); // addend + branch26

        assert_eq!(relocs[0].kind, AArch64RelocKind::Addend);
        assert_eq!(relocs[0].symbol_index, 4); // the addend value
        assert!(!relocs[0].is_extern);

        assert_eq!(relocs[1].kind, AArch64RelocKind::Branch26);
        assert_eq!(relocs[1].symbol_index, 1);
        assert!(relocs[1].is_extern);
    }

    #[test]
    fn test_fixup_list_resolve_section_relative() {
        let mut list = FixupList::new();
        list.push(Fixup {
            offset: 0x10,
            kind: AArch64RelocKind::Unsigned,
            target: FixupTarget::Section(2), // section ordinal
            addend: 0,
        });

        let relocs = list.resolve_to_relocations();
        assert_eq!(relocs.len(), 1);
        assert!(!relocs[0].is_extern);
        assert_eq!(relocs[0].symbol_index, 2);
    }

    #[test]
    fn test_apply_branch26_forward() {
        // BL instruction: opcode = 0x94000000
        let mut insn = 0x9400_0000_u32.to_le_bytes();
        // Branch forward 16 bytes = 4 words
        apply_branch26(&mut insn, 16);
        let result = u32::from_le_bytes(insn);
        assert_eq!(result & 0x03FF_FFFF, 4); // imm26 = 4
        assert_eq!(result & 0xFC00_0000, 0x9400_0000); // opcode preserved
    }

    #[test]
    fn test_apply_branch26_backward() {
        // B instruction: opcode = 0x14000000
        let mut insn = 0x1400_0000_u32.to_le_bytes();
        // Branch backward 8 bytes = -2 words
        apply_branch26(&mut insn, -8);
        let result = u32::from_le_bytes(insn);
        // -2 in 26-bit two's complement = 0x03FF_FFFE
        assert_eq!(result & 0x03FF_FFFF, 0x03FF_FFFE);
        assert_eq!(result & 0xFC00_0000, 0x1400_0000); // opcode preserved
    }

    #[test]
    fn test_apply_page21() {
        // ADRP x0, target_page: 0x90000000
        let mut insn = 0x9000_0000_u32.to_le_bytes();
        // Page offset of 1
        apply_page21(&mut insn, 1);
        let result = u32::from_le_bytes(insn);

        // immhi = (1 >> 2) & 0x7FFFF = 0, immlo = 1 & 3 = 1
        // result should have immlo=1 at bits [30:29] and immhi=0 at bits [23:5]
        let immlo = (result >> 29) & 3;
        let immhi = (result >> 5) & 0x7_FFFF;
        assert_eq!(immlo, 1);
        assert_eq!(immhi, 0);
    }

    #[test]
    fn test_apply_page21_large() {
        let mut insn = 0x9000_0000_u32.to_le_bytes();
        // Page offset = 5 = 0b101 → immlo=01, immhi=1
        apply_page21(&mut insn, 5);
        let result = u32::from_le_bytes(insn);

        let immlo = (result >> 29) & 3;
        let immhi = (result >> 5) & 0x7_FFFF;
        assert_eq!((immhi << 2) | immlo, 5);
    }

    #[test]
    fn test_apply_pageoff12_add() {
        // ADD x0, x0, #imm12: 0x91000000
        let mut insn = 0x9100_0000_u32.to_le_bytes();
        // Page offset = 0x10
        apply_pageoff12(&mut insn, 0x10, 0);
        let result = u32::from_le_bytes(insn);

        let imm12 = (result >> 10) & 0xFFF;
        assert_eq!(imm12, 0x10);
        assert_eq!(result & 0xFFC0_03FF, 0x9100_0000); // rest preserved
    }

    #[test]
    fn test_apply_pageoff12_ldr_x() {
        // LDR x0, [x0, #imm12]: 0xF9400000
        let mut insn = 0xF940_0000_u32.to_le_bytes();
        // Page offset = 0x10 (aligned to 8 bytes), shift=3
        apply_pageoff12(&mut insn, 0x10, 3);
        let result = u32::from_le_bytes(insn);

        let imm12 = (result >> 10) & 0xFFF;
        assert_eq!(imm12, 0x10 >> 3); // scaled by 8
    }

    #[test]
    #[should_panic(expected = "4-byte aligned")]
    fn test_apply_branch26_unaligned() {
        let mut insn = 0x9400_0000_u32.to_le_bytes();
        apply_branch26(&mut insn, 6); // not aligned
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_apply_branch26_overflow() {
        let mut insn = 0x9400_0000_u32.to_le_bytes();
        // 256 MB = 67108864 words, exceeds 26-bit signed range
        apply_branch26(&mut insn, 256 * 1024 * 1024);
    }

    #[test]
    #[should_panic(expected = "not aligned")]
    fn test_apply_pageoff12_unaligned() {
        let mut insn = 0xF940_0000_u32.to_le_bytes();
        apply_pageoff12(&mut insn, 0x11, 3); // 0x11 not aligned to 8
    }

    #[test]
    fn test_fixup_list_empty() {
        let list = FixupList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.resolve_to_relocations().len(), 0);
    }
}

// llvm2-codegen - DWARF CFI (Call Frame Information) for AArch64 macOS
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: DWARF 5 spec, Section 6.4 (Call Frame Information)
// Reference: ~/llvm-project-ref/llvm/lib/MC/MCDwarf.cpp
//            (CIE/FDE emission)
// Reference: Apple __eh_frame format (based on .eh_frame, LSB extensions)

//! DWARF Call Frame Information (CFI) emission for AArch64 Darwin.
//!
//! When a function's frame layout cannot be described by Darwin compact
//! unwind encoding (variable-size frames, unusual layouts), we fall back
//! to DWARF CFI. The compact unwind entry uses `UNWIND_ARM64_MODE_DWARF`
//! and the unwinder looks up the matching FDE in `__TEXT,__eh_frame`.
//!
//! # Section layout
//!
//! The `__eh_frame` section contains:
//! 1. One CIE (Common Information Entry) — shared by all FDEs
//! 2. One FDE (Frame Description Entry) per function needing DWARF unwind
//!
//! ```text
//! ┌──────────────────────┐
//! │       CIE             │  Common for all AArch64 Darwin functions
//! ├──────────────────────┤
//! │       FDE #0          │  For function 0
//! ├──────────────────────┤
//! │       FDE #1          │  For function 1
//! ├──────────────────────┤
//! │       ...             │
//! └──────────────────────┘
//! ```
//!
//! # AArch64 Darwin CIE parameters
//!
//! - Version: 1 (eh_frame uses version 1)
//! - Augmentation: "zR" (pointer encoding follows)
//! - Code alignment factor: 4 (AArch64 instructions are 4 bytes)
//! - Data alignment factor: -8 (stack grows down, 8-byte slots)
//! - Return address register: 30 (X30/LR)
//! - Pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4

use crate::frame::FrameLayout;
use llvm2_ir::regs::PReg;

// ---------------------------------------------------------------------------
// DWARF CFA opcodes (subset needed for AArch64 frames)
// ---------------------------------------------------------------------------

/// DW_CFA_def_cfa: define CFA as register + offset.
const DW_CFA_DEF_CFA: u8 = 0x0C;
/// DW_CFA_def_cfa_offset: change CFA offset only.
const DW_CFA_DEF_CFA_OFFSET: u8 = 0x0E;
/// DW_CFA_offset: register saved at CFA-relative offset (high 2 bits = 0b10).
const DW_CFA_OFFSET: u8 = 0x80;
/// DW_CFA_advance_loc: advance location by delta * code_alignment (high 2 bits = 0b01).
const DW_CFA_ADVANCE_LOC: u8 = 0x40;
/// DW_CFA_advance_loc1: advance location by 1-byte delta * code_alignment.
const DW_CFA_ADVANCE_LOC1: u8 = 0x02;
/// DW_CFA_advance_loc2: advance location by 2-byte delta * code_alignment.
const DW_CFA_ADVANCE_LOC2: u8 = 0x03;
/// DW_CFA_nop: padding.
const DW_CFA_NOP: u8 = 0x00;

/// DW_EH_PE_pcrel: pointer is PC-relative.
const DW_EH_PE_PCREL: u8 = 0x10;
/// DW_EH_PE_sdata4: pointer is signed 4-byte value.
const DW_EH_PE_SDATA4: u8 = 0x0B;

/// AArch64 code alignment factor.
const CODE_ALIGN_FACTOR: u64 = 4;
/// AArch64 data alignment factor (stack grows down, 8-byte slots).
const DATA_ALIGN_FACTOR: i64 = -8;
/// AArch64 return address DWARF register number (X30).
const RA_REGISTER: u64 = 30;
/// AArch64 frame pointer DWARF register number (X29).
const FP_REGISTER: u64 = 29;
/// AArch64 stack pointer DWARF register number (X31/SP).
const SP_REGISTER: u64 = 31;

// ---------------------------------------------------------------------------
// DwarfCie — Common Information Entry
// ---------------------------------------------------------------------------

/// A DWARF Common Information Entry (CIE) for AArch64 Darwin.
///
/// The CIE contains parameters shared by all FDEs: code/data alignment,
/// return address register, and initial instructions. For eh_frame, the
/// CIE also includes augmentation data with pointer encoding.
#[derive(Debug, Clone)]
pub struct DwarfCie {
    /// eh_frame version (always 1 for .eh_frame).
    pub version: u8,
    /// Augmentation string (e.g., "zR" for pointer encoding).
    pub augmentation: Vec<u8>,
    /// Code alignment factor (4 for AArch64).
    pub code_alignment_factor: u64,
    /// Data alignment factor (-8 for AArch64).
    pub data_alignment_factor: i64,
    /// Return address register (30 = X30 for AArch64).
    pub return_address_register: u64,
    /// Pointer encoding (DW_EH_PE_pcrel | DW_EH_PE_sdata4).
    pub fde_pointer_encoding: u8,
    /// Initial CFI instructions (define initial CFA).
    pub initial_instructions: Vec<u8>,
}

impl DwarfCie {
    /// Create the standard AArch64 Darwin CIE.
    ///
    /// Initial instructions define CFA = SP + 0 (before prologue).
    pub fn aarch64_darwin() -> Self {
        let mut initial_instructions = Vec::new();
        // DW_CFA_def_cfa SP, 0 — CFA starts at SP with zero offset
        initial_instructions.push(DW_CFA_DEF_CFA);
        encode_uleb128(SP_REGISTER, &mut initial_instructions);
        encode_uleb128(0, &mut initial_instructions);

        Self {
            version: 1,
            augmentation: b"zR\0".to_vec(),
            code_alignment_factor: CODE_ALIGN_FACTOR,
            data_alignment_factor: DATA_ALIGN_FACTOR,
            return_address_register: RA_REGISTER,
            fde_pointer_encoding: DW_EH_PE_PCREL | DW_EH_PE_SDATA4,
            initial_instructions,
        }
    }

    /// Serialize the CIE to bytes.
    ///
    /// Layout:
    /// - u32: length (excluding this field)
    /// - u32: CIE id (0 for eh_frame)
    /// - u8: version
    /// - augmentation string (null-terminated)
    /// - ULEB128: code alignment factor
    /// - SLEB128: data alignment factor
    /// - ULEB128: return address register
    /// - ULEB128: augmentation data length
    /// - augmentation data (pointer encoding byte)
    /// - initial instructions
    /// - padding to 8-byte boundary (pointer size)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();

        // CIE id = 0 (distinguishes CIE from FDE in eh_frame)
        body.extend_from_slice(&0u32.to_le_bytes());

        // Version
        body.push(self.version);

        // Augmentation string (null-terminated)
        body.extend_from_slice(&self.augmentation);

        // Code alignment factor (ULEB128)
        encode_uleb128(self.code_alignment_factor, &mut body);

        // Data alignment factor (SLEB128)
        encode_sleb128(self.data_alignment_factor, &mut body);

        // Return address register (ULEB128)
        encode_uleb128(self.return_address_register, &mut body);

        // Augmentation data length (ULEB128) — 1 byte for pointer encoding
        encode_uleb128(1, &mut body);

        // Pointer encoding
        body.push(self.fde_pointer_encoding);

        // Initial instructions
        body.extend_from_slice(&self.initial_instructions);

        // Pad to pointer-size (8-byte) alignment with DW_CFA_nop
        while (body.len() + 4) % 8 != 0 {
            body.push(DW_CFA_NOP);
        }

        // Prepend length
        let mut result = Vec::with_capacity(4 + body.len());
        result.extend_from_slice(&(body.len() as u32).to_le_bytes());
        result.extend(body);

        result
    }
}

// ---------------------------------------------------------------------------
// DwarfFde — Frame Description Entry
// ---------------------------------------------------------------------------

/// A DWARF Frame Description Entry (FDE) for one function.
///
/// Contains the PC range and CFI instructions describing how to unwind
/// the function's frame at each point in the code.
#[derive(Debug, Clone)]
pub struct DwarfFde {
    /// Function start address (will be relocated, stored as 0 initially).
    pub function_offset: u64,
    /// Function code length in bytes.
    pub function_length: u32,
    /// Symbol index for the function (for relocations).
    pub symbol_index: u32,
    /// CFI instructions describing the frame state changes.
    pub instructions: Vec<u8>,
}

impl DwarfFde {
    /// Create an FDE from a FrameLayout.
    ///
    /// Generates CFI instructions that describe:
    /// 1. After STP X29, X30, [SP, #-N]! — CFA moves, FP/LR saved
    /// 2. After MOV X29, SP — CFA defined via FP
    /// 3. After each callee-saved STP — register save locations
    /// 4. After SUB SP, SP, #adj — stack frame fully set up
    pub fn from_layout(
        layout: &FrameLayout,
        function_offset: u64,
        function_length: u32,
        symbol_index: u32,
    ) -> Self {
        let mut instructions = Vec::new();

        // After STP X29, X30, [SP, #-callee_saved_area_size]!
        // CFA = SP + callee_saved_area_size
        // (STP pre-index is 1 instruction = 4 bytes)
        encode_advance_loc(1, &mut instructions); // advance 1 instruction (4 bytes)
        instructions.push(DW_CFA_DEF_CFA_OFFSET);
        encode_uleb128(layout.callee_saved_area_size as u64, &mut instructions);

        // X29 (FP) saved at CFA - 16
        let fp_offset = 16u64 / ((-DATA_ALIGN_FACTOR) as u64); // factored offset
        instructions.push(DW_CFA_OFFSET | (FP_REGISTER as u8 & 0x3F));
        encode_uleb128(fp_offset, &mut instructions);

        // X30 (LR) saved at CFA - 8
        let lr_offset = 8u64 / ((-DATA_ALIGN_FACTOR) as u64);
        instructions.push(DW_CFA_OFFSET | (RA_REGISTER as u8 & 0x3F));
        encode_uleb128(lr_offset, &mut instructions);

        // After MOV X29, SP (1 instruction)
        // CFA = FP + callee_saved_area_size (use FP as CFA base now)
        encode_advance_loc(1, &mut instructions);
        instructions.push(DW_CFA_DEF_CFA);
        encode_uleb128(FP_REGISTER, &mut instructions);
        encode_uleb128(layout.callee_saved_area_size as u64, &mut instructions);

        // Callee-saved register pairs (skip pair[0] = FP/LR, already handled)
        for (i, pair) in layout.callee_saved_pairs.iter().enumerate().skip(1) {
            // Each STP is 1 instruction
            encode_advance_loc(1, &mut instructions);

            // Register 1: saved at CFA - offset
            let reg1_dwarf = preg_to_dwarf(pair.reg1, pair.is_fpr);
            let offset1 = (layout.callee_saved_area_size as i64 - ((i as i64) * 16))
                / (-DATA_ALIGN_FACTOR);
            instructions.push(DW_CFA_OFFSET | (reg1_dwarf as u8 & 0x3F));
            encode_uleb128(offset1 as u64, &mut instructions);

            // Register 2: saved at CFA - offset - 8
            let reg2_dwarf = preg_to_dwarf(pair.reg2, pair.is_fpr);
            let offset2 = (layout.callee_saved_area_size as i64 - ((i as i64) * 16) - 8)
                / (-DATA_ALIGN_FACTOR);
            instructions.push(DW_CFA_OFFSET | (reg2_dwarf as u8 & 0x3F));
            encode_uleb128(offset2 as u64, &mut instructions);
        }

        Self {
            function_offset,
            function_length,
            symbol_index,
            instructions,
        }
    }

    /// Serialize the FDE to bytes.
    ///
    /// Layout:
    /// - u32: length (excluding this field)
    /// - u32: CIE pointer (offset from this field to the CIE)
    /// - i32: function start (PC-relative, relocated)
    /// - u32: function length
    /// - ULEB128: augmentation data length (0)
    /// - CFI instructions
    /// - padding to 8-byte boundary
    pub fn to_bytes(&self, cie_offset: u32) -> Vec<u8> {
        let mut body = Vec::new();

        // CIE pointer — offset from this field to the start of the CIE.
        // In eh_frame: this is the byte offset from the CIE pointer field itself
        // back to the CIE start.
        body.extend_from_slice(&cie_offset.to_le_bytes());

        // PC begin (i32, will be relocated to function start)
        body.extend_from_slice(&(self.function_offset as i32).to_le_bytes());

        // PC range (u32, function length)
        body.extend_from_slice(&self.function_length.to_le_bytes());

        // Augmentation data length (ULEB128, 0 = no augmentation data)
        encode_uleb128(0, &mut body);

        // CFI instructions
        body.extend_from_slice(&self.instructions);

        // Pad to 8-byte alignment with DW_CFA_nop
        while (body.len() + 4) % 8 != 0 {
            body.push(DW_CFA_NOP);
        }

        // Prepend length
        let mut result = Vec::with_capacity(4 + body.len());
        result.extend_from_slice(&(body.len() as u32).to_le_bytes());
        result.extend(body);

        result
    }
}

// ---------------------------------------------------------------------------
// DwarfCfiSection — collects CIE + FDEs into __eh_frame data
// ---------------------------------------------------------------------------

/// Collects DWARF CIE and FDEs for emission as the `__TEXT,__eh_frame` section.
///
/// Only used when at least one function requires DWARF CFI fallback
/// (its compact unwind encoding is `UNWIND_ARM64_MODE_DWARF`).
#[derive(Debug, Clone)]
pub struct DwarfCfiSection {
    /// The CIE (one per section).
    cie: DwarfCie,
    /// FDEs for functions needing DWARF unwind.
    fdes: Vec<DwarfFde>,
}

impl DwarfCfiSection {
    /// Create a new DWARF CFI section with the standard AArch64 Darwin CIE.
    pub fn new() -> Self {
        Self {
            cie: DwarfCie::aarch64_darwin(),
            fdes: Vec::new(),
        }
    }

    /// Add an FDE for a function.
    pub fn add_fde(&mut self, fde: DwarfFde) {
        self.fdes.push(fde);
    }

    /// Returns the number of FDEs in this section.
    pub fn fde_count(&self) -> usize {
        self.fdes.len()
    }

    /// Returns true if no FDEs have been added.
    pub fn is_empty(&self) -> bool {
        self.fdes.is_empty()
    }

    /// Serialize the entire __eh_frame section to bytes.
    ///
    /// Layout: CIE followed by all FDEs, terminated by a zero-length entry.
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.fdes.is_empty() {
            return Vec::new();
        }

        let mut data = Vec::new();

        // Emit CIE
        let cie_bytes = self.cie.to_bytes();
        let cie_start = 0usize;
        data.extend_from_slice(&cie_bytes);

        // Emit FDEs
        for fde in &self.fdes {
            let fde_cie_ptr_offset = data.len() as u32;
            // CIE pointer = offset from CIE pointer field to CIE start
            // The CIE pointer field is at (fde_start + 4), and CIE starts at cie_start.
            // cie_offset = (fde_start + 4) - cie_start = fde_cie_ptr_offset + 4 - cie_start
            let cie_offset = (fde_cie_ptr_offset + 4) - cie_start as u32;
            let fde_bytes = fde.to_bytes(cie_offset);
            data.extend_from_slice(&fde_bytes);
        }

        // Terminator: zero-length entry (just 4 zero bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        data
    }

    /// Total size of the section data in bytes.
    pub fn data_size(&self) -> usize {
        self.to_bytes().len()
    }
}

impl Default for DwarfCfiSection {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LEB128 encoding helpers
// ---------------------------------------------------------------------------

/// Encode a value as ULEB128 (unsigned LEB128).
fn encode_uleb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // more bytes follow
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode a value as SLEB128 (signed LEB128).
fn encode_sleb128(mut value: i64, out: &mut Vec<u8>) {
    let mut more = true;
    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        // If the sign bit of the current byte matches the remaining value,
        // we're done.
        if (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        out.push(byte);
    }
}

/// Encode a DW_CFA_advance_loc for `n` instructions.
///
/// Uses the smallest encoding that fits:
/// - DW_CFA_advance_loc (6-bit delta, 0-63 instructions)
/// - DW_CFA_advance_loc1 (8-bit delta)
/// - DW_CFA_advance_loc2 (16-bit delta)
fn encode_advance_loc(n_instructions: u32, out: &mut Vec<u8>) {
    if n_instructions <= 63 {
        // DW_CFA_advance_loc: high 2 bits = 0b01, low 6 bits = delta
        out.push(DW_CFA_ADVANCE_LOC | (n_instructions as u8));
    } else if n_instructions <= 255 {
        out.push(DW_CFA_ADVANCE_LOC1);
        out.push(n_instructions as u8);
    } else {
        out.push(DW_CFA_ADVANCE_LOC2);
        out.extend_from_slice(&(n_instructions as u16).to_le_bytes());
    }
}

/// Map a physical register to its DWARF register number.
///
/// AArch64 DWARF register numbering:
/// - X0-X30: DWARF 0-30
/// - SP: DWARF 31
/// - V0-V31 (SIMD/FP): DWARF 64-95
fn preg_to_dwarf(preg: PReg, is_fpr: bool) -> u64 {
    let enc = preg.encoding();
    if is_fpr {
        // V-registers: encoding 64-95 in our scheme, DWARF 64-95
        // Our unified encoding: V0=64, V1=65, ..., V31=95
        // DWARF: D0=64, D1=65, ..., D31=95
        64 + (enc as u64 - 64)
    } else {
        // X-registers: encoding 0-31, DWARF 0-31
        enc as u64
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{CalleeSavedPair, FrameLayout};
    use llvm2_ir::regs::{X19, X20, X21, X22, X29, X30, V8, V9};

    fn make_simple_layout() -> FrameLayout {
        FrameLayout {
            callee_saved_pairs: vec![CalleeSavedPair {
                reg1: X29,
                reg2: X30,
                fp_offset: 0,
                is_fpr: false,
            }],
            callee_saved_area_size: 16,
            spill_area_size: 0,
            local_area_size: 0,
            outgoing_arg_area_size: 0,
            total_frame_size: 16,
            uses_frame_pointer: true,
            is_leaf: true,
            uses_red_zone: false,
            fp_to_spill_offset: -16,
            has_dynamic_alloc: false,
        }
    }

    fn make_layout_with_callee_saves() -> FrameLayout {
        FrameLayout {
            callee_saved_pairs: vec![
                CalleeSavedPair {
                    reg1: X29,
                    reg2: X30,
                    fp_offset: 0,
                    is_fpr: false,
                },
                CalleeSavedPair {
                    reg1: X19,
                    reg2: X20,
                    fp_offset: -16,
                    is_fpr: false,
                },
                CalleeSavedPair {
                    reg1: X21,
                    reg2: X22,
                    fp_offset: -32,
                    is_fpr: false,
                },
            ],
            callee_saved_area_size: 48,
            spill_area_size: 16,
            local_area_size: 0,
            outgoing_arg_area_size: 0,
            total_frame_size: 64,
            uses_frame_pointer: true,
            is_leaf: false,
            uses_red_zone: false,
            fp_to_spill_offset: -48,
            has_dynamic_alloc: false,
        }
    }

    fn make_layout_with_fpr_saves() -> FrameLayout {
        FrameLayout {
            callee_saved_pairs: vec![
                CalleeSavedPair {
                    reg1: X29,
                    reg2: X30,
                    fp_offset: 0,
                    is_fpr: false,
                },
                CalleeSavedPair {
                    reg1: V8,
                    reg2: V9,
                    fp_offset: -16,
                    is_fpr: true,
                },
            ],
            callee_saved_area_size: 32,
            spill_area_size: 0,
            local_area_size: 0,
            outgoing_arg_area_size: 0,
            total_frame_size: 32,
            uses_frame_pointer: true,
            is_leaf: true,
            uses_red_zone: false,
            fp_to_spill_offset: -32,
            has_dynamic_alloc: false,
        }
    }

    // --- CIE tests ---

    #[test]
    fn test_cie_creation() {
        let cie = DwarfCie::aarch64_darwin();
        assert_eq!(cie.version, 1);
        assert_eq!(cie.code_alignment_factor, 4);
        assert_eq!(cie.data_alignment_factor, -8);
        assert_eq!(cie.return_address_register, 30);
        assert_eq!(cie.fde_pointer_encoding, DW_EH_PE_PCREL | DW_EH_PE_SDATA4);
    }

    #[test]
    fn test_cie_serialization_not_empty() {
        let cie = DwarfCie::aarch64_darwin();
        let bytes = cie.to_bytes();
        assert!(!bytes.is_empty());
        // Minimum: 4 (length) + 4 (CIE id) + 1 (version) + 3 (augmentation) + ...
        assert!(bytes.len() >= 12);
    }

    #[test]
    fn test_cie_serialization_alignment() {
        let cie = DwarfCie::aarch64_darwin();
        let bytes = cie.to_bytes();
        // Total length must be 8-byte aligned (4 bytes for length field + body)
        assert_eq!(bytes.len() % 8, 0, "CIE must be 8-byte aligned, got {}", bytes.len());
    }

    #[test]
    fn test_cie_length_field() {
        let cie = DwarfCie::aarch64_darwin();
        let bytes = cie.to_bytes();
        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_cie_id_is_zero() {
        let cie = DwarfCie::aarch64_darwin();
        let bytes = cie.to_bytes();
        let cie_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(cie_id, 0, "eh_frame CIE ID must be 0");
    }

    #[test]
    fn test_cie_version() {
        let cie = DwarfCie::aarch64_darwin();
        let bytes = cie.to_bytes();
        assert_eq!(bytes[8], 1, "eh_frame version must be 1");
    }

    // --- FDE tests ---

    #[test]
    fn test_fde_from_simple_layout() {
        let layout = make_simple_layout();
        let fde = DwarfFde::from_layout(&layout, 0, 64, 0);

        assert_eq!(fde.function_length, 64);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_fde_from_callee_save_layout() {
        let layout = make_layout_with_callee_saves();
        let fde = DwarfFde::from_layout(&layout, 0, 128, 0);

        assert_eq!(fde.function_length, 128);
        // Should have instructions for: STP FP/LR, MOV FP, STP X19/X20, STP X21/X22
        assert!(fde.instructions.len() > 10, "Expected substantial CFI instructions");
    }

    #[test]
    fn test_fde_from_fpr_layout() {
        let layout = make_layout_with_fpr_saves();
        let fde = DwarfFde::from_layout(&layout, 0, 96, 0);

        assert_eq!(fde.function_length, 96);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_fde_serialization_alignment() {
        let layout = make_simple_layout();
        let fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        let bytes = fde.to_bytes(24); // arbitrary CIE offset

        // Total must be 8-byte aligned
        assert_eq!(bytes.len() % 8, 0, "FDE must be 8-byte aligned, got {}", bytes.len());
    }

    #[test]
    fn test_fde_length_field() {
        let layout = make_simple_layout();
        let fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        let bytes = fde.to_bytes(24);

        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_fde_cie_pointer() {
        let layout = make_simple_layout();
        let fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        let cie_offset = 24u32;
        let bytes = fde.to_bytes(cie_offset);

        let stored_cie_ptr = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(stored_cie_ptr, cie_offset, "CIE pointer should match");
    }

    // --- Section tests ---

    #[test]
    fn test_empty_section() {
        let section = DwarfCfiSection::new();
        assert!(section.is_empty());
        assert_eq!(section.fde_count(), 0);
        assert_eq!(section.to_bytes().len(), 0);
    }

    #[test]
    fn test_section_with_one_fde() {
        let mut section = DwarfCfiSection::new();
        let layout = make_simple_layout();
        let fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        section.add_fde(fde);

        assert_eq!(section.fde_count(), 1);
        assert!(!section.is_empty());

        let bytes = section.to_bytes();
        assert!(!bytes.is_empty());

        // Should contain CIE + FDE + terminator
        // Terminator is 4 zero bytes
        let last_4 = &bytes[bytes.len() - 4..];
        assert_eq!(last_4, &[0, 0, 0, 0], "Section must end with zero terminator");
    }

    #[test]
    fn test_section_with_multiple_fdes() {
        let mut section = DwarfCfiSection::new();

        let layout1 = make_simple_layout();
        section.add_fde(DwarfFde::from_layout(&layout1, 0, 64, 0));

        let layout2 = make_layout_with_callee_saves();
        section.add_fde(DwarfFde::from_layout(&layout2, 64, 128, 1));

        assert_eq!(section.fde_count(), 2);

        let bytes = section.to_bytes();
        assert!(!bytes.is_empty());

        // Check terminator
        let last_4 = &bytes[bytes.len() - 4..];
        assert_eq!(last_4, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_section_data_size_matches_to_bytes() {
        let mut section = DwarfCfiSection::new();
        let layout = make_simple_layout();
        section.add_fde(DwarfFde::from_layout(&layout, 0, 64, 0));

        assert_eq!(section.data_size(), section.to_bytes().len());
    }

    #[test]
    fn test_default_section() {
        let section = DwarfCfiSection::default();
        assert!(section.is_empty());
    }

    // --- LEB128 encoding tests ---

    #[test]
    fn test_uleb128_single_byte() {
        let mut buf = Vec::new();
        encode_uleb128(0, &mut buf);
        assert_eq!(buf, &[0]);

        buf.clear();
        encode_uleb128(1, &mut buf);
        assert_eq!(buf, &[1]);

        buf.clear();
        encode_uleb128(127, &mut buf);
        assert_eq!(buf, &[127]);
    }

    #[test]
    fn test_uleb128_multi_byte() {
        let mut buf = Vec::new();
        encode_uleb128(128, &mut buf);
        assert_eq!(buf, &[0x80, 0x01]);

        buf.clear();
        encode_uleb128(624485, &mut buf);
        assert_eq!(buf, &[0xE5, 0x8E, 0x26]);
    }

    #[test]
    fn test_sleb128_positive() {
        let mut buf = Vec::new();
        encode_sleb128(0, &mut buf);
        assert_eq!(buf, &[0]);

        buf.clear();
        encode_sleb128(63, &mut buf);
        assert_eq!(buf, &[63]);

        buf.clear();
        encode_sleb128(64, &mut buf);
        assert_eq!(buf, &[0xC0, 0x00]);
    }

    #[test]
    fn test_sleb128_negative() {
        let mut buf = Vec::new();
        encode_sleb128(-1, &mut buf);
        assert_eq!(buf, &[0x7F]);

        buf.clear();
        encode_sleb128(-8, &mut buf);
        assert_eq!(buf, &[0x78]);

        buf.clear();
        encode_sleb128(-64, &mut buf);
        assert_eq!(buf, &[0x40]);

        buf.clear();
        encode_sleb128(-65, &mut buf);
        assert_eq!(buf, &[0xBF, 0x7F]);
    }

    // --- Register mapping tests ---

    #[test]
    fn test_preg_to_dwarf_gpr() {
        assert_eq!(preg_to_dwarf(X29, false), 29);
        assert_eq!(preg_to_dwarf(X30, false), 30);
        assert_eq!(preg_to_dwarf(X19, false), 19);
    }

    #[test]
    fn test_preg_to_dwarf_fpr() {
        assert_eq!(preg_to_dwarf(V8, true), 72);
        assert_eq!(preg_to_dwarf(V9, true), 73);
    }

    // --- Advance loc encoding tests ---

    #[test]
    fn test_advance_loc_small() {
        let mut buf = Vec::new();
        encode_advance_loc(1, &mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC | 1);
    }

    #[test]
    fn test_advance_loc_max_inline() {
        let mut buf = Vec::new();
        encode_advance_loc(63, &mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC | 63);
    }

    #[test]
    fn test_advance_loc_1byte() {
        let mut buf = Vec::new();
        encode_advance_loc(100, &mut buf);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC1);
        assert_eq!(buf[1], 100);
    }

    #[test]
    fn test_advance_loc_2byte() {
        let mut buf = Vec::new();
        encode_advance_loc(300, &mut buf);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC2);
        assert_eq!(u16::from_le_bytes([buf[1], buf[2]]), 300);
    }
}

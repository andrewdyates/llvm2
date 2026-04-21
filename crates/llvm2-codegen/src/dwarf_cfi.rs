// llvm2-codegen - DWARF CFI (Call Frame Information) for AArch64 and x86-64 macOS
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: DWARF 5 spec, Section 6.4 (Call Frame Information)
// Reference: ~/llvm-project-ref/llvm/lib/MC/MCDwarf.cpp
//            (CIE/FDE emission)
// Reference: Apple __eh_frame format (based on .eh_frame, LSB extensions)
// Reference: System V AMD64 ABI, Section 3.7 (Register Mapping)

//! DWARF Call Frame Information (CFI) emission for AArch64 and x86-64 Darwin.
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
//!
//! # x86-64 Darwin CIE parameters
//!
//! - Version: 1 (eh_frame uses version 1)
//! - Augmentation: "zR" (pointer encoding follows)
//! - Code alignment factor: 1 (variable-length x86-64 instructions)
//! - Data alignment factor: -8 (stack grows down, 8-byte slots)
//! - Return address register: 16 (RIP)
//! - Pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
//!
//! # x86-64 DWARF register numbers (AMD64 ABI)
//!
//! | Register | DWARF # | Register | DWARF # |
//! |----------|---------|----------|---------|
//! | RAX      | 0       | R8       | 8       |
//! | RDX      | 1       | R9       | 9       |
//! | RCX      | 2       | R10      | 10      |
//! | RBX      | 3       | R11      | 11      |
//! | RSI      | 4       | R12      | 12      |
//! | RDI      | 5       | R13      | 13      |
//! | RBP      | 6       | R14      | 14      |
//! | RSP      | 7       | R15      | 15      |
//! | RIP      | 16      | XMM0-15  | 17-32   |

use crate::frame::FrameLayout;
use llvm2_ir::regs::PReg;
use llvm2_ir::x86_64_regs::X86PReg;

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
// x86-64 DWARF constants (AMD64 ABI)
// ---------------------------------------------------------------------------

/// x86-64 code alignment factor (variable-length instructions, 1 byte granularity).
const X86_64_CODE_ALIGN_FACTOR: u64 = 1;
/// x86-64 data alignment factor (stack grows down, 8-byte slots).
const X86_64_DATA_ALIGN_FACTOR: i64 = -8;
/// x86-64 return address DWARF register number (RIP).
const X86_64_RA_REGISTER: u64 = 16;
/// x86-64 stack pointer DWARF register number (RSP).
const X86_64_SP_REGISTER: u64 = 7;
/// x86-64 frame pointer DWARF register number (RBP).
const X86_64_FP_REGISTER: u64 = 6;

/// x86-64 DWARF register numbers (AMD64 ABI, DWARF 5 mapping).
///
/// Reference: System V AMD64 ABI, Figure 3.36 "DWARF Register Number Mapping"
pub const X86_64_DWARF_RAX: u64 = 0;
pub const X86_64_DWARF_RDX: u64 = 1;
pub const X86_64_DWARF_RCX: u64 = 2;
pub const X86_64_DWARF_RBX: u64 = 3;
pub const X86_64_DWARF_RSI: u64 = 4;
pub const X86_64_DWARF_RDI: u64 = 5;
pub const X86_64_DWARF_RBP: u64 = 6;
pub const X86_64_DWARF_RSP: u64 = 7;
pub const X86_64_DWARF_R8: u64 = 8;
pub const X86_64_DWARF_R9: u64 = 9;
pub const X86_64_DWARF_R10: u64 = 10;
pub const X86_64_DWARF_R11: u64 = 11;
pub const X86_64_DWARF_R12: u64 = 12;
pub const X86_64_DWARF_R13: u64 = 13;
pub const X86_64_DWARF_R14: u64 = 14;
pub const X86_64_DWARF_R15: u64 = 15;
pub const X86_64_DWARF_RIP: u64 = 16;
pub const X86_64_DWARF_XMM0: u64 = 17;
pub const X86_64_DWARF_XMM15: u64 = 32;

// ---------------------------------------------------------------------------
// DwarfCie — Common Information Entry
// ---------------------------------------------------------------------------

/// A DWARF Common Information Entry (CIE) for AArch64 Darwin.
///
/// The CIE contains parameters shared by all FDEs: code/data alignment,
/// return address register, and initial instructions. For eh_frame, the
/// CIE also includes augmentation data with pointer encoding.
///
/// ## Augmentation strings
///
/// - `"zR"` — Basic pointer encoding (no personality, no LSDA).
/// - `"zPLR"` — Personality function + LSDA pointers + pointer encoding.
///   Used for functions that participate in C++/Rust exception handling.
///   The "P" augmentation encodes the personality function pointer in the
///   CIE, and "L" indicates that each FDE carries an LSDA pointer.
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
    /// Personality function encoding (only used with "zPLR" augmentation).
    /// DW_EH_PE_pcrel | DW_EH_PE_sdata4 on Apple AArch64.
    pub personality_encoding: Option<u8>,
    /// Personality function pointer (placeholder, relocated by linker).
    /// Only used with "zPLR" augmentation.
    pub personality_pointer: Option<u32>,
    /// LSDA pointer encoding (only used with "zPLR" augmentation).
    /// DW_EH_PE_pcrel | DW_EH_PE_sdata4 on Apple AArch64.
    pub lsda_encoding: Option<u8>,
    /// Initial CFI instructions (define initial CFA).
    pub initial_instructions: Vec<u8>,
}

impl DwarfCie {
    /// Create the standard AArch64 Darwin CIE (no personality/LSDA).
    ///
    /// Augmentation: "zR" — only FDE pointer encoding.
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
            personality_encoding: None,
            personality_pointer: None,
            lsda_encoding: None,
            initial_instructions,
        }
    }

    /// Create an AArch64 Darwin CIE with personality and LSDA support.
    ///
    /// Augmentation: "zPLR" — personality pointer, LSDA pointer encoding,
    /// and FDE pointer encoding. Used for functions that participate in
    /// C++ exception handling or Rust panic unwinding.
    ///
    /// The personality pointer is a placeholder (0) that gets a relocation
    /// pointing to the personality function symbol (e.g., `__gxx_personality_v0`).
    ///
    /// Reference: LSB Core spec, Section 10.6.1 (Augmentation String Format)
    /// Reference: LLVM MCDwarf.cpp, EmitPersonality()
    pub fn aarch64_darwin_with_eh() -> Self {
        let mut initial_instructions = Vec::new();
        initial_instructions.push(DW_CFA_DEF_CFA);
        encode_uleb128(SP_REGISTER, &mut initial_instructions);
        encode_uleb128(0, &mut initial_instructions);

        let personality_enc = DW_EH_PE_PCREL | DW_EH_PE_SDATA4;
        let lsda_enc = DW_EH_PE_PCREL | DW_EH_PE_SDATA4;

        Self {
            version: 1,
            augmentation: b"zPLR\0".to_vec(),
            code_alignment_factor: CODE_ALIGN_FACTOR,
            data_alignment_factor: DATA_ALIGN_FACTOR,
            return_address_register: RA_REGISTER,
            fde_pointer_encoding: DW_EH_PE_PCREL | DW_EH_PE_SDATA4,
            personality_encoding: Some(personality_enc),
            personality_pointer: Some(0), // placeholder, relocated
            lsda_encoding: Some(lsda_enc),
            initial_instructions,
        }
    }

    /// Create the standard x86-64 Darwin CIE (no personality/LSDA).
    ///
    /// Augmentation: "zR" -- only FDE pointer encoding.
    /// Initial instructions: CFA = RSP + 8 (return address is pushed at entry).
    /// Return address register: RIP (DWARF 16).
    ///
    /// Key differences from AArch64:
    /// - code_alignment_factor = 1 (variable-length instructions)
    /// - CFA starts at RSP + 8 (not RSP + 0) because CALL pushes return address
    /// - Return address register = 16 (RIP) instead of 30 (X30/LR)
    ///
    /// Reference: System V AMD64 ABI, Section 3.7
    pub fn x86_64_darwin() -> Self {
        let mut initial_instructions = Vec::new();
        // DW_CFA_def_cfa RSP(7), 8 -- CFA = RSP + 8 at function entry
        // (the CALL instruction pushed the 8-byte return address onto the stack)
        initial_instructions.push(DW_CFA_DEF_CFA);
        encode_uleb128(X86_64_SP_REGISTER, &mut initial_instructions);
        encode_uleb128(8, &mut initial_instructions);

        // DW_CFA_offset RIP(16), 1 -- return address at CFA-8 (factored: 8/8=1)
        initial_instructions.push(DW_CFA_OFFSET | (X86_64_RA_REGISTER as u8 & 0x3F));
        encode_uleb128(1, &mut initial_instructions);

        Self {
            version: 1,
            augmentation: b"zR\0".to_vec(),
            code_alignment_factor: X86_64_CODE_ALIGN_FACTOR,
            data_alignment_factor: X86_64_DATA_ALIGN_FACTOR,
            return_address_register: X86_64_RA_REGISTER,
            fde_pointer_encoding: DW_EH_PE_PCREL | DW_EH_PE_SDATA4,
            personality_encoding: None,
            personality_pointer: None,
            lsda_encoding: None,
            initial_instructions,
        }
    }

    /// Create an x86-64 Darwin CIE with personality and LSDA support.
    ///
    /// Augmentation: "zPLR" -- personality pointer, LSDA pointer encoding,
    /// and FDE pointer encoding. Used for functions that participate in
    /// C++ exception handling or Rust panic unwinding.
    pub fn x86_64_darwin_with_eh() -> Self {
        let mut initial_instructions = Vec::new();
        initial_instructions.push(DW_CFA_DEF_CFA);
        encode_uleb128(X86_64_SP_REGISTER, &mut initial_instructions);
        encode_uleb128(8, &mut initial_instructions);

        initial_instructions.push(DW_CFA_OFFSET | (X86_64_RA_REGISTER as u8 & 0x3F));
        encode_uleb128(1, &mut initial_instructions);

        let personality_enc = DW_EH_PE_PCREL | DW_EH_PE_SDATA4;
        let lsda_enc = DW_EH_PE_PCREL | DW_EH_PE_SDATA4;

        Self {
            version: 1,
            augmentation: b"zPLR\0".to_vec(),
            code_alignment_factor: X86_64_CODE_ALIGN_FACTOR,
            data_alignment_factor: X86_64_DATA_ALIGN_FACTOR,
            return_address_register: X86_64_RA_REGISTER,
            fde_pointer_encoding: DW_EH_PE_PCREL | DW_EH_PE_SDATA4,
            personality_encoding: Some(personality_enc),
            personality_pointer: Some(0), // placeholder, relocated
            lsda_encoding: Some(lsda_enc),
            initial_instructions,
        }
    }

    /// Returns true if this CIE has personality/LSDA augmentation ("zPLR").
    pub fn has_personality(&self) -> bool {
        self.personality_encoding.is_some()
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
    /// - augmentation data (varies by augmentation string):
    ///   "zR":   [FDE pointer encoding]
    ///   "zPLR": [personality encoding, personality pointer(4), LSDA encoding, FDE pointer encoding]
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

        // Augmentation data: build it first, then emit length + data.
        let mut aug_data = Vec::new();

        if let Some(personality_enc) = self.personality_encoding {
            // "P" augmentation: personality encoding byte + personality pointer.
            aug_data.push(personality_enc);
            let ptr = self.personality_pointer.unwrap_or(0);
            aug_data.extend_from_slice(&ptr.to_le_bytes());
        }

        if let Some(lsda_enc) = self.lsda_encoding {
            // "L" augmentation: LSDA pointer encoding byte.
            aug_data.push(lsda_enc);
        }

        // "R" augmentation: FDE pointer encoding byte.
        aug_data.push(self.fde_pointer_encoding);

        // Augmentation data length (ULEB128)
        encode_uleb128(aug_data.len() as u64, &mut body);

        // Augmentation data
        body.extend_from_slice(&aug_data);

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
///
/// When the parent CIE uses "zPLR" augmentation, the FDE carries an LSDA
/// pointer in its augmentation data (an i32 PC-relative offset to the
/// function's LSDA in `__TEXT,__gcc_except_table`).
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
    /// LSDA pointer (i32, PC-relative, for "zPLR" CIE augmentation).
    /// `None` if the CIE uses "zR" augmentation or the function has no LSDA.
    /// The value is a placeholder (0) that gets a relocation to the LSDA.
    pub lsda_pointer: Option<i32>,
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
            lsda_pointer: None,
        }
    }

    /// Serialize the FDE to bytes.
    ///
    /// Layout:
    /// - u32: length (excluding this field)
    /// - u32: CIE pointer (offset from this field to the CIE)
    /// - i32: function start (PC-relative, relocated)
    /// - u32: function length
    /// - ULEB128: augmentation data length
    /// - augmentation data (LSDA pointer if "zPLR" CIE)
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

        // Augmentation data: LSDA pointer for "zPLR" CIE, empty for "zR".
        if let Some(lsda_ptr) = self.lsda_pointer {
            // LSDA pointer is 4 bytes (sdata4, PC-relative).
            encode_uleb128(4, &mut body);
            body.extend_from_slice(&lsda_ptr.to_le_bytes());
        } else {
            // No augmentation data.
            encode_uleb128(0, &mut body);
        }

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

    /// Create a new DWARF CFI section with the standard x86-64 Darwin CIE.
    pub fn new_x86_64() -> Self {
        Self {
            cie: DwarfCie::x86_64_darwin(),
            fdes: Vec::new(),
        }
    }

    /// Create a new DWARF CFI section with x86-64 Darwin CIE + personality/LSDA.
    pub fn new_x86_64_with_eh() -> Self {
        Self {
            cie: DwarfCie::x86_64_darwin_with_eh(),
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

/// Encode a DW_CFA_advance_loc for `n_bytes` of code (byte granularity).
///
/// For x86-64 where code_alignment_factor = 1, the delta is in bytes.
/// Uses the smallest encoding that fits:
/// - DW_CFA_advance_loc (6-bit delta, 0-63 bytes)
/// - DW_CFA_advance_loc1 (8-bit delta, 0-255 bytes)
/// - DW_CFA_advance_loc2 (16-bit delta, 0-65535 bytes)
fn encode_advance_loc_bytes(n_bytes: u32, out: &mut Vec<u8>) {
    if n_bytes == 0 {
        return;
    }
    if n_bytes <= 63 {
        out.push(DW_CFA_ADVANCE_LOC | (n_bytes as u8));
    } else if n_bytes <= 255 {
        out.push(DW_CFA_ADVANCE_LOC1);
        out.push(n_bytes as u8);
    } else {
        out.push(DW_CFA_ADVANCE_LOC2);
        out.extend_from_slice(&(n_bytes as u16).to_le_bytes());
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

/// Map an x86-64 physical register (X86PReg) to its DWARF register number.
///
/// CRITICAL: X86PReg hardware encoding differs from DWARF numbering!
///
/// | X86PReg  | hw_enc | DWARF |   | X86PReg  | hw_enc | DWARF |
/// |----------|--------|-------|---|----------|--------|-------|
/// | RAX      | 0      | 0     |   | R8       | 8      | 8     |
/// | RCX      | 1      | 2     |   | R9       | 9      | 9     |
/// | RDX      | 2      | 1     |   | R10      | 10     | 10    |
/// | RBX      | 3      | 3     |   | R11      | 11     | 11    |
/// | RSP      | 4      | 7     |   | R12      | 12     | 12    |
/// | RBP      | 5      | 6     |   | R13      | 13     | 13    |
/// | RSI      | 6      | 4     |   | R14      | 14     | 14    |
/// | RDI      | 7      | 5     |   | R15      | 15     | 15    |
/// | XMM0-15  | 64-79  | 17-32 |
///
/// Reference: System V AMD64 ABI, Figure 3.36
pub fn x86_64_preg_to_dwarf(preg: X86PReg) -> u64 {
    let enc = preg.encoding();
    match enc {
        // GPR64 (0-15): non-trivial mapping for first 8 registers
        0 => X86_64_DWARF_RAX,   // RAX: hw 0 -> DWARF 0
        1 => X86_64_DWARF_RCX,   // RCX: hw 1 -> DWARF 2
        2 => X86_64_DWARF_RDX,   // RDX: hw 2 -> DWARF 1
        3 => X86_64_DWARF_RBX,   // RBX: hw 3 -> DWARF 3
        4 => X86_64_DWARF_RSP,   // RSP: hw 4 -> DWARF 7
        5 => X86_64_DWARF_RBP,   // RBP: hw 5 -> DWARF 6
        6 => X86_64_DWARF_RSI,   // RSI: hw 6 -> DWARF 4
        7 => X86_64_DWARF_RDI,   // RDI: hw 7 -> DWARF 5
        8..=15 => enc as u64,     // R8-R15: hw == DWARF (8-15)
        // XMM registers (64-79): DWARF 17-32
        64..=79 => X86_64_DWARF_XMM0 + (enc as u64 - 64),
        _ => panic!("Unknown x86-64 register encoding: {}", enc),
    }
}

/// Generate x86-64 FDE CFI instructions from prologue parameters.
///
/// Describes the standard System V AMD64 ABI frame:
/// ```text
///   [entry]     CFA = RSP + 8       (return address on stack)
///   PUSH RBP    CFA = RSP + 16      (RBP saved at CFA-16)
///   MOV RBP,RSP CFA = RBP + 16      (switch to frame pointer based CFA)
///   PUSH reg    CFA = RBP + 16      (callee-saved at known offsets from RBP)
///   ...
///   SUB RSP, N  CFA = RBP + 16      (CFA still FP-based, unaffected)
/// ```
///
/// # Arguments
/// - `callee_saved`: callee-saved registers pushed after RBP (in push order)
/// - `frame_size`: bytes allocated via SUB RSP (for locals/spills, may be 0)
/// - `function_offset`: function start address (for relocations)
/// - `function_length`: total code size in bytes
/// - `symbol_index`: symbol table index for relocations
///
/// # Instruction size estimates (conservative, used for advance_loc)
/// - PUSH reg: 1 byte (no REX) or 2 bytes (REX prefix for R8-R15)
/// - MOV RBP, RSP: 3 bytes (REX.W + 0x89 + ModRM)
/// - SUB RSP, imm: 4 bytes (REX.W + 0x83 + ModRM + imm8) or 7 bytes (imm32)
pub fn x86_64_fde_from_prologue(
    callee_saved: &[X86PReg],
    frame_size: u32,
    function_offset: u64,
    function_length: u32,
    symbol_index: u32,
) -> DwarfFde {
    let mut instructions = Vec::new();

    // After PUSH RBP: CFA = RSP + 16 (was RSP+8, pushed 8 bytes)
    // PUSH RBP is 1 byte (opcode 0x55)
    encode_advance_loc_bytes(1, &mut instructions);
    instructions.push(DW_CFA_DEF_CFA_OFFSET);
    encode_uleb128(16, &mut instructions);

    // RBP saved at CFA - 16 (factored: 16/8 = 2)
    instructions.push(DW_CFA_OFFSET | (X86_64_FP_REGISTER as u8 & 0x3F));
    encode_uleb128(2, &mut instructions);

    // After MOV RBP, RSP: CFA = RBP + 16 (switch to FP-based CFA)
    // MOV RBP, RSP is 3 bytes (REX.W 0x48, MOV 0x89, ModRM 0xE5)
    encode_advance_loc_bytes(3, &mut instructions);
    instructions.push(DW_CFA_DEF_CFA);
    encode_uleb128(X86_64_FP_REGISTER, &mut instructions);
    encode_uleb128(16, &mut instructions);

    // After each PUSH of callee-saved register:
    // Register is saved at RBP - 8*(i+1) where i is 0-based push index
    // In DWARF terms: saved at CFA - 16 - 8*(i+1) = CFA - (24 + 8*i)
    // Factored offset: (24 + 8*i) / 8 = 3 + i
    for (i, &reg) in callee_saved.iter().enumerate() {
        // PUSH is 1 byte for RAX-RDI, 2 bytes for R8-R15 (REX prefix)
        let enc = reg.encoding();
        let push_size: u32 = if enc >= 8 && enc <= 15 { 2 } else { 1 };
        encode_advance_loc_bytes(push_size, &mut instructions);

        let dwarf_reg = x86_64_preg_to_dwarf(reg);
        let factored_offset = (3 + i) as u64;
        instructions.push(DW_CFA_OFFSET | (dwarf_reg as u8 & 0x3F));
        encode_uleb128(factored_offset, &mut instructions);
    }

    // After SUB RSP, frame_size: no CFA change needed (CFA is RBP-based)
    // We still advance the location counter for completeness if frame_size > 0
    if frame_size > 0 {
        // SUB RSP, imm8 = 4 bytes; SUB RSP, imm32 = 7 bytes
        let sub_size: u32 = if frame_size <= 127 { 4 } else { 7 };
        encode_advance_loc_bytes(sub_size, &mut instructions);
    }

    DwarfFde {
        function_offset,
        function_length,
        symbol_index,
        instructions,
        lsda_pointer: None,
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

    // --- zPLR augmentation CIE tests ---

    #[test]
    fn test_cie_with_eh_creation() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        assert_eq!(cie.version, 1);
        assert_eq!(cie.augmentation, b"zPLR\0");
        assert!(cie.has_personality());
        assert!(cie.personality_encoding.is_some());
        assert!(cie.lsda_encoding.is_some());
    }

    #[test]
    fn test_cie_without_eh_has_no_personality() {
        let cie = DwarfCie::aarch64_darwin();
        assert!(!cie.has_personality());
        assert!(cie.personality_encoding.is_none());
        assert!(cie.lsda_encoding.is_none());
    }

    #[test]
    fn test_cie_with_eh_serialization_not_empty() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let bytes = cie.to_bytes();
        assert!(!bytes.is_empty());
        // zPLR CIE is larger than zR because of personality pointer
        let basic_cie = DwarfCie::aarch64_darwin();
        let basic_bytes = basic_cie.to_bytes();
        assert!(bytes.len() > basic_bytes.len(),
            "zPLR CIE ({}) should be larger than zR CIE ({})",
            bytes.len(), basic_bytes.len());
    }

    #[test]
    fn test_cie_with_eh_alignment() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let bytes = cie.to_bytes();
        assert_eq!(bytes.len() % 8, 0, "CIE must be 8-byte aligned");
    }

    #[test]
    fn test_cie_with_eh_id_is_zero() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let bytes = cie.to_bytes();
        let cie_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(cie_id, 0, "eh_frame CIE ID must be 0");
    }

    #[test]
    fn test_cie_with_eh_version() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let bytes = cie.to_bytes();
        assert_eq!(bytes[8], 1, "eh_frame version must be 1");
    }

    #[test]
    fn test_cie_with_eh_augmentation_string() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let bytes = cie.to_bytes();
        // After length(4) + CIE id(4) + version(1) = byte 9, augmentation starts
        assert_eq!(&bytes[9..14], b"zPLR\0");
    }

    #[test]
    fn test_cie_with_eh_length_field() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let bytes = cie.to_bytes();
        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(length as usize, bytes.len() - 4);
    }

    // --- FDE with LSDA pointer tests ---

    #[test]
    fn test_fde_with_lsda_pointer() {
        let layout = make_simple_layout();
        let mut fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        fde.lsda_pointer = Some(0); // placeholder

        let bytes = fde.to_bytes(24);
        assert!(!bytes.is_empty());
        assert_eq!(bytes.len() % 8, 0, "FDE must be 8-byte aligned");
    }

    #[test]
    fn test_fde_with_lsda_larger_than_without() {
        let layout = make_simple_layout();

        let fde_no_lsda = DwarfFde::from_layout(&layout, 0, 64, 0);
        let bytes_no_lsda = fde_no_lsda.to_bytes(24);

        let mut fde_lsda = DwarfFde::from_layout(&layout, 0, 64, 0);
        fde_lsda.lsda_pointer = Some(0);
        let bytes_lsda = fde_lsda.to_bytes(24);

        // FDE with LSDA has 4 extra bytes for the pointer + augmentation length change
        assert!(bytes_lsda.len() >= bytes_no_lsda.len(),
            "FDE with LSDA ({}) should be >= FDE without ({})",
            bytes_lsda.len(), bytes_no_lsda.len());
    }

    #[test]
    fn test_fde_without_lsda_aug_data_zero() {
        let layout = make_simple_layout();
        let fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        let bytes = fde.to_bytes(24);

        // After length(4) + CIE ptr(4) + PC begin(4) + PC range(4) = byte 16,
        // augmentation data length should be ULEB128(0) = 0x00
        // But we need to account for the length prefix: offset is 4+4+4+4 = 16 in body
        // The body starts at byte 4 (after the length field)
        // Body: cie_ptr(4) + pc_begin(4) + pc_range(4) = 12, then aug data len
        assert_eq!(bytes[4 + 12], 0x00, "Augmentation data length should be 0");
    }

    #[test]
    fn test_fde_with_lsda_aug_data_four() {
        let layout = make_simple_layout();
        let mut fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        fde.lsda_pointer = Some(0x42);
        let bytes = fde.to_bytes(24);

        // Augmentation data length should be 4 (i32 LSDA pointer)
        assert_eq!(bytes[4 + 12], 0x04, "Augmentation data length should be 4");
        // LSDA pointer value
        let lsda_val = i32::from_le_bytes(bytes[4+13..4+17].try_into().unwrap());
        assert_eq!(lsda_val, 0x42);
    }

    // --- Section with EH CIE ---

    #[test]
    fn test_section_with_eh_cie() {
        let cie = DwarfCie::aarch64_darwin_with_eh();
        let mut section = DwarfCfiSection {
            cie,
            fdes: Vec::new(),
        };
        assert!(section.is_empty());

        let layout = make_simple_layout();
        let mut fde = DwarfFde::from_layout(&layout, 0, 64, 0);
        fde.lsda_pointer = Some(0);
        section.add_fde(fde);

        assert_eq!(section.fde_count(), 1);
        let bytes = section.to_bytes();
        assert!(!bytes.is_empty());

        // Verify terminator
        let last_4 = &bytes[bytes.len() - 4..];
        assert_eq!(last_4, &[0, 0, 0, 0]);
    }

    // =========================================================================
    // x86-64 DWARF CFI tests
    // =========================================================================

    // --- x86-64 CIE tests ---

    #[test]
    fn test_x86_64_cie_creation() {
        let cie = DwarfCie::x86_64_darwin();
        assert_eq!(cie.version, 1);
        assert_eq!(cie.code_alignment_factor, 1);
        assert_eq!(cie.data_alignment_factor, -8);
        assert_eq!(cie.return_address_register, 16); // RIP
        assert_eq!(cie.fde_pointer_encoding, DW_EH_PE_PCREL | DW_EH_PE_SDATA4);
        assert!(!cie.has_personality());
    }

    #[test]
    fn test_x86_64_cie_serialization_not_empty() {
        let cie = DwarfCie::x86_64_darwin();
        let bytes = cie.to_bytes();
        assert!(!bytes.is_empty());
        assert!(bytes.len() >= 12);
    }

    #[test]
    fn test_x86_64_cie_alignment() {
        let cie = DwarfCie::x86_64_darwin();
        let bytes = cie.to_bytes();
        assert_eq!(bytes.len() % 8, 0, "CIE must be 8-byte aligned, got {}", bytes.len());
    }

    #[test]
    fn test_x86_64_cie_length_field() {
        let cie = DwarfCie::x86_64_darwin();
        let bytes = cie.to_bytes();
        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_x86_64_cie_id_is_zero() {
        let cie = DwarfCie::x86_64_darwin();
        let bytes = cie.to_bytes();
        let cie_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(cie_id, 0, "eh_frame CIE ID must be 0");
    }

    #[test]
    fn test_x86_64_cie_version() {
        let cie = DwarfCie::x86_64_darwin();
        let bytes = cie.to_bytes();
        assert_eq!(bytes[8], 1, "eh_frame version must be 1");
    }

    #[test]
    fn test_x86_64_cie_augmentation_string() {
        let cie = DwarfCie::x86_64_darwin();
        let bytes = cie.to_bytes();
        // After length(4) + CIE id(4) + version(1) = byte 9
        assert_eq!(&bytes[9..12], b"zR\0");
    }

    #[test]
    fn test_x86_64_cie_initial_instructions_cfa_rsp_plus_8() {
        let cie = DwarfCie::x86_64_darwin();
        // Initial instructions should start with DW_CFA_def_cfa RSP(7), 8
        assert!(cie.initial_instructions.len() >= 3);
        assert_eq!(cie.initial_instructions[0], DW_CFA_DEF_CFA);
        assert_eq!(cie.initial_instructions[1], 7); // RSP = ULEB128(7)
        assert_eq!(cie.initial_instructions[2], 8); // offset = ULEB128(8)
    }

    #[test]
    fn test_x86_64_cie_initial_instructions_ra_offset() {
        let cie = DwarfCie::x86_64_darwin();
        // After DW_CFA_def_cfa(1) + RSP(1) + 8(1) = 3 bytes,
        // should have DW_CFA_offset RIP(16), 1
        assert!(cie.initial_instructions.len() >= 5);
        // DW_CFA_offset with register 16: 0x80 | (16 & 0x3F) = 0x80 | 0x10 = 0x90
        assert_eq!(cie.initial_instructions[3], 0x90);
        assert_eq!(cie.initial_instructions[4], 1); // factored offset: 8/8 = 1
    }

    // --- x86-64 zPLR CIE tests ---

    #[test]
    fn test_x86_64_cie_with_eh_creation() {
        let cie = DwarfCie::x86_64_darwin_with_eh();
        assert_eq!(cie.version, 1);
        assert_eq!(cie.augmentation, b"zPLR\0");
        assert_eq!(cie.code_alignment_factor, 1);
        assert_eq!(cie.return_address_register, 16);
        assert!(cie.has_personality());
    }

    #[test]
    fn test_x86_64_cie_with_eh_alignment() {
        let cie = DwarfCie::x86_64_darwin_with_eh();
        let bytes = cie.to_bytes();
        assert_eq!(bytes.len() % 8, 0, "CIE must be 8-byte aligned");
    }

    #[test]
    fn test_x86_64_cie_with_eh_length_field() {
        let cie = DwarfCie::x86_64_darwin_with_eh();
        let bytes = cie.to_bytes();
        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_x86_64_cie_with_eh_larger_than_basic() {
        let eh_cie = DwarfCie::x86_64_darwin_with_eh();
        let basic_cie = DwarfCie::x86_64_darwin();
        assert!(eh_cie.to_bytes().len() > basic_cie.to_bytes().len(),
            "zPLR CIE should be larger than zR CIE");
    }

    // --- x86-64 register mapping tests ---

    #[test]
    fn test_x86_64_preg_to_dwarf_trivial_mappings() {
        use llvm2_ir::x86_64_regs::*;
        // RAX: hw 0 -> DWARF 0
        assert_eq!(x86_64_preg_to_dwarf(RAX), 0);
        // RBX: hw 3 -> DWARF 3
        assert_eq!(x86_64_preg_to_dwarf(RBX), 3);
    }

    #[test]
    fn test_x86_64_preg_to_dwarf_swapped_mappings() {
        use llvm2_ir::x86_64_regs::*;
        // RCX: hw 1 -> DWARF 2 (swapped with RDX!)
        assert_eq!(x86_64_preg_to_dwarf(RCX), 2);
        // RDX: hw 2 -> DWARF 1
        assert_eq!(x86_64_preg_to_dwarf(RDX), 1);
        // RSP: hw 4 -> DWARF 7
        assert_eq!(x86_64_preg_to_dwarf(RSP), 7);
        // RBP: hw 5 -> DWARF 6
        assert_eq!(x86_64_preg_to_dwarf(RBP), 6);
        // RSI: hw 6 -> DWARF 4
        assert_eq!(x86_64_preg_to_dwarf(RSI), 4);
        // RDI: hw 7 -> DWARF 5
        assert_eq!(x86_64_preg_to_dwarf(RDI), 5);
    }

    #[test]
    fn test_x86_64_preg_to_dwarf_extended_regs() {
        use llvm2_ir::x86_64_regs::*;
        // R8-R15: hw == DWARF
        assert_eq!(x86_64_preg_to_dwarf(R8), 8);
        assert_eq!(x86_64_preg_to_dwarf(R9), 9);
        assert_eq!(x86_64_preg_to_dwarf(R12), 12);
        assert_eq!(x86_64_preg_to_dwarf(R15), 15);
    }

    #[test]
    fn test_x86_64_preg_to_dwarf_xmm() {
        use llvm2_ir::x86_64_regs::*;
        // XMM0: hw 64 -> DWARF 17
        assert_eq!(x86_64_preg_to_dwarf(XMM0), 17);
        // XMM15: hw 79 -> DWARF 32
        assert_eq!(x86_64_preg_to_dwarf(XMM15), 32);
        // XMM7: hw 71 -> DWARF 24
        assert_eq!(x86_64_preg_to_dwarf(XMM7), 24);
    }

    // --- x86-64 FDE tests ---

    #[test]
    fn test_x86_64_fde_minimal_frame() {
        // Simplest case: no callee-saved, no locals
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 32, 0);
        assert_eq!(fde.function_length, 32);
        assert!(!fde.instructions.is_empty());
        // Should have: advance_loc(PUSH RBP) + def_cfa_offset(16) + offset(RBP) +
        //             advance_loc(MOV) + def_cfa(RBP,16) = at least ~10 bytes
        assert!(fde.instructions.len() >= 8,
            "Minimal x86-64 FDE should have at least 8 instruction bytes, got {}",
            fde.instructions.len());
    }

    #[test]
    fn test_x86_64_fde_with_callee_saved() {
        use llvm2_ir::x86_64_regs::*;
        let callee_saved = vec![RBX, R12, R13];
        let fde = x86_64_fde_from_prologue(&callee_saved, 64, 0, 256, 1);
        assert_eq!(fde.function_length, 256);
        assert_eq!(fde.symbol_index, 1);
        // More instructions than minimal due to callee-saved + SUB RSP
        let minimal = x86_64_fde_from_prologue(&[], 0, 0, 32, 0);
        assert!(fde.instructions.len() > minimal.instructions.len(),
            "FDE with callee-saved ({}) should be larger than minimal ({})",
            fde.instructions.len(), minimal.instructions.len());
    }

    #[test]
    fn test_x86_64_fde_with_large_frame() {
        use llvm2_ir::x86_64_regs::*;
        let callee_saved = vec![RBX, R12, R13, R14, R15];
        let fde = x86_64_fde_from_prologue(&callee_saved, 4096, 0, 512, 2);
        assert_eq!(fde.function_length, 512);
        assert!(!fde.instructions.is_empty());
    }

    #[test]
    fn test_x86_64_fde_serialization_alignment() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        let bytes = fde.to_bytes(24);
        assert_eq!(bytes.len() % 8, 0, "FDE must be 8-byte aligned, got {}", bytes.len());
    }

    #[test]
    fn test_x86_64_fde_length_field() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        let bytes = fde.to_bytes(24);
        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_x86_64_fde_instructions_start_with_advance_push_rbp() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        // First instruction: DW_CFA_advance_loc 1 (PUSH RBP = 1 byte)
        assert_eq!(fde.instructions[0], DW_CFA_ADVANCE_LOC | 1);
    }

    #[test]
    fn test_x86_64_fde_cfa_offset_16_after_push_rbp() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        // After advance_loc(1): DW_CFA_def_cfa_offset 16
        assert_eq!(fde.instructions[1], DW_CFA_DEF_CFA_OFFSET);
        assert_eq!(fde.instructions[2], 16); // ULEB128(16)
    }

    #[test]
    fn test_x86_64_fde_rbp_saved_at_cfa_minus_16() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        // After def_cfa_offset: DW_CFA_offset RBP(6), 2 (factored: 16/8=2)
        // DW_CFA_offset | (6 & 0x3F) = 0x80 | 0x06 = 0x86
        assert_eq!(fde.instructions[3], 0x86);
        assert_eq!(fde.instructions[4], 2); // factored offset
    }

    #[test]
    fn test_x86_64_fde_advance_3_for_mov_rbp_rsp() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        // After RBP offset: advance_loc 3 (MOV RBP,RSP = 3 bytes)
        assert_eq!(fde.instructions[5], DW_CFA_ADVANCE_LOC | 3);
    }

    #[test]
    fn test_x86_64_fde_cfa_rbp_plus_16_after_mov() {
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        // After advance_loc(3): DW_CFA_def_cfa RBP(6), 16
        assert_eq!(fde.instructions[6], DW_CFA_DEF_CFA);
        assert_eq!(fde.instructions[7], 6);  // RBP DWARF number
        assert_eq!(fde.instructions[8], 16); // offset
    }

    #[test]
    fn test_x86_64_fde_callee_saved_rbx_at_correct_offset() {
        use llvm2_ir::x86_64_regs::*;
        let fde = x86_64_fde_from_prologue(&[RBX], 0, 0, 64, 0);
        // After the base prologue (9 bytes):
        // advance_loc(1) for PUSH RBX (RBX hw=3, 1 byte push)
        assert_eq!(fde.instructions[9], DW_CFA_ADVANCE_LOC | 1);
        // DW_CFA_offset RBX(DWARF 3), factored_offset=3
        // 0x80 | (3 & 0x3F) = 0x83
        assert_eq!(fde.instructions[10], 0x83);
        assert_eq!(fde.instructions[11], 3); // factored: (24)/8 = 3
    }

    #[test]
    fn test_x86_64_fde_callee_saved_r12_uses_2byte_push() {
        use llvm2_ir::x86_64_regs::*;
        let fde = x86_64_fde_from_prologue(&[R12], 0, 0, 64, 0);
        // R12 has hw encoding 12 (>= 8), so PUSH needs REX prefix = 2 bytes
        assert_eq!(fde.instructions[9], DW_CFA_ADVANCE_LOC | 2);
        // DW_CFA_offset R12(DWARF 12), factored_offset=3
        // 0x80 | (12 & 0x3F) = 0x8C
        assert_eq!(fde.instructions[10], 0x8C);
        assert_eq!(fde.instructions[11], 3);
    }

    // --- x86-64 section tests ---

    #[test]
    fn test_x86_64_section_empty() {
        let section = DwarfCfiSection::new_x86_64();
        assert!(section.is_empty());
        assert_eq!(section.fde_count(), 0);
        assert_eq!(section.to_bytes().len(), 0);
    }

    #[test]
    fn test_x86_64_section_with_one_fde() {
        let mut section = DwarfCfiSection::new_x86_64();
        let fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        section.add_fde(fde);

        assert_eq!(section.fde_count(), 1);
        let bytes = section.to_bytes();
        assert!(!bytes.is_empty());

        // Terminator
        let last_4 = &bytes[bytes.len() - 4..];
        assert_eq!(last_4, &[0, 0, 0, 0], "Section must end with zero terminator");
    }

    #[test]
    fn test_x86_64_section_with_multiple_fdes() {
        use llvm2_ir::x86_64_regs::*;
        let mut section = DwarfCfiSection::new_x86_64();

        section.add_fde(x86_64_fde_from_prologue(&[], 0, 0, 64, 0));
        section.add_fde(x86_64_fde_from_prologue(&[RBX, R12], 128, 64, 128, 1));
        section.add_fde(x86_64_fde_from_prologue(&[RBX, R12, R13, R14, R15], 256, 192, 512, 2));

        assert_eq!(section.fde_count(), 3);
        let bytes = section.to_bytes();
        assert!(!bytes.is_empty());

        // Terminator
        let last_4 = &bytes[bytes.len() - 4..];
        assert_eq!(last_4, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_x86_64_section_with_eh_cie() {
        let mut section = DwarfCfiSection::new_x86_64_with_eh();
        assert!(section.is_empty());

        let mut fde = x86_64_fde_from_prologue(&[], 0, 0, 64, 0);
        fde.lsda_pointer = Some(0);
        section.add_fde(fde);

        assert_eq!(section.fde_count(), 1);
        let bytes = section.to_bytes();
        assert!(!bytes.is_empty());

        let last_4 = &bytes[bytes.len() - 4..];
        assert_eq!(last_4, &[0, 0, 0, 0]);
    }

    // --- encode_advance_loc_bytes tests ---

    #[test]
    fn test_advance_loc_bytes_zero() {
        let mut buf = Vec::new();
        encode_advance_loc_bytes(0, &mut buf);
        assert!(buf.is_empty(), "advance_loc_bytes(0) should emit nothing");
    }

    #[test]
    fn test_advance_loc_bytes_small() {
        let mut buf = Vec::new();
        encode_advance_loc_bytes(1, &mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC | 1);
    }

    #[test]
    fn test_advance_loc_bytes_max_inline() {
        let mut buf = Vec::new();
        encode_advance_loc_bytes(63, &mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC | 63);
    }

    #[test]
    fn test_advance_loc_bytes_1byte_encoding() {
        let mut buf = Vec::new();
        encode_advance_loc_bytes(100, &mut buf);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC1);
        assert_eq!(buf[1], 100);
    }

    #[test]
    fn test_advance_loc_bytes_2byte_encoding() {
        let mut buf = Vec::new();
        encode_advance_loc_bytes(300, &mut buf);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], DW_CFA_ADVANCE_LOC2);
        assert_eq!(u16::from_le_bytes([buf[1], buf[2]]), 300);
    }

    // --- Cross-architecture comparison tests ---

    #[test]
    fn test_x86_64_cie_differs_from_aarch64() {
        let x86_cie = DwarfCie::x86_64_darwin();
        let arm_cie = DwarfCie::aarch64_darwin();

        // Code alignment factor differs
        assert_ne!(x86_cie.code_alignment_factor, arm_cie.code_alignment_factor);
        assert_eq!(x86_cie.code_alignment_factor, 1);
        assert_eq!(arm_cie.code_alignment_factor, 4);

        // Return address register differs
        assert_ne!(x86_cie.return_address_register, arm_cie.return_address_register);
        assert_eq!(x86_cie.return_address_register, 16); // RIP
        assert_eq!(arm_cie.return_address_register, 30); // X30

        // Data alignment factor is the same
        assert_eq!(x86_cie.data_alignment_factor, arm_cie.data_alignment_factor);

        // Initial instructions differ (x86-64 has CFA=RSP+8, AArch64 has CFA=SP+0)
        assert_ne!(x86_cie.initial_instructions, arm_cie.initial_instructions);
    }

    #[test]
    fn test_x86_64_cie_serialized_bytes_differ_from_aarch64() {
        let x86_bytes = DwarfCie::x86_64_darwin().to_bytes();
        let arm_bytes = DwarfCie::aarch64_darwin().to_bytes();
        assert_ne!(x86_bytes, arm_bytes);
    }
}

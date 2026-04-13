// llvm2-codegen - Frame lowering for AArch64 macOS
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: ~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64FrameLowering.cpp
// Reference: ~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64PrologueEpilogue.cpp
// Reference: ~/llvm-project-ref/llvm/lib/Target/AArch64/MCTargetDesc/AArch64AsmBackend.cpp
//            (generateCompactUnwindEncoding, line 576)

//! AArch64 frame lowering for Apple/Darwin targets.
//!
//! Implements prologue/epilogue generation, frame index elimination,
//! and Darwin compact unwind encoding for the AArch64 macOS ABI.
//!
//! # Frame layout (high address to low)
//!
//! ```text
//! ┌──────────────────────┐  ← caller SP
//! │  incoming arguments   │
//! ├──────────────────────┤
//! │  X29 (FP) / X30 (LR) │  ← FP points here (X29 saved value)
//! ├──────────────────────┤
//! │  callee-saved GPR     │  pairs: X19/X20, X21/X22, ...
//! │  callee-saved FPR     │  pairs: D8/D9, D10/D11, ...
//! ├──────────────────────┤
//! │  spill slots          │  from register allocator
//! │  local variables      │  alloca / aggregates
//! ├──────────────────────┤
//! │  outgoing arg area    │  for stack-passed call arguments
//! └──────────────────────┘  ← SP (16-byte aligned)
//! ```
//!
//! # Key invariants
//!
//! - Apple AArch64 **requires** a valid frame pointer (X29).
//! - X29/X30 are always saved as the first pair.
//! - Stack pointer must be 16-byte aligned at all times.
//! - Callee-saved registers are saved in pairs (STP/LDP) for compact
//!   unwind compatibility.
//! - Red zone (128 bytes below SP) is disabled by default.

use llvm2_ir::function::MachFunction;
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{PReg, SP, SpecialReg, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, V8, V9, V10, V11, V12, V13, V14, V15};

// ---------------------------------------------------------------------------
// Constants — Darwin compact unwind encoding (ARM64)
// ---------------------------------------------------------------------------
// Reference: AArch64AsmBackend.cpp line 517

/// Standard frame-pointer-based unwind mode.
pub const UNWIND_ARM64_MODE_FRAME: u32 = 0x04000000;
/// Frameless leaf function unwind mode.
pub const UNWIND_ARM64_MODE_FRAMELESS: u32 = 0x02000000;
/// Fallback to full DWARF FDE.
pub const UNWIND_ARM64_MODE_DWARF: u32 = 0x03000000;

/// Compact unwind register-pair flags for GPRs.
pub const UNWIND_ARM64_FRAME_X19_X20_PAIR: u32 = 0x00000001;
pub const UNWIND_ARM64_FRAME_X21_X22_PAIR: u32 = 0x00000002;
pub const UNWIND_ARM64_FRAME_X23_X24_PAIR: u32 = 0x00000004;
pub const UNWIND_ARM64_FRAME_X25_X26_PAIR: u32 = 0x00000008;
pub const UNWIND_ARM64_FRAME_X27_X28_PAIR: u32 = 0x00000010;

/// Compact unwind register-pair flags for FPRs (D-regs = lower 64 bits of V-regs).
pub const UNWIND_ARM64_FRAME_D8_D9_PAIR: u32 = 0x00000100;
pub const UNWIND_ARM64_FRAME_D10_D11_PAIR: u32 = 0x00000200;
pub const UNWIND_ARM64_FRAME_D12_D13_PAIR: u32 = 0x00000400;
pub const UNWIND_ARM64_FRAME_D14_D15_PAIR: u32 = 0x00000800;

/// Maximum leaf-function frame size eligible for red zone optimization.
/// Apple AArch64 red zone is 128 bytes.
pub const RED_ZONE_SIZE: u32 = 128;

/// AArch64 stack alignment requirement (Apple/Darwin).
pub const STACK_ALIGNMENT: u32 = 16;

// ---------------------------------------------------------------------------
// CalleeSavedPair — a pair of registers saved/restored together
// ---------------------------------------------------------------------------

/// A pair of callee-saved registers stored together via STP/LDP.
///
/// AArch64 convention saves registers in pairs to maintain 16-byte
/// alignment and enable compact unwind encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalleeSavedPair {
    /// First register in the pair (lower-numbered).
    pub reg1: PReg,
    /// Second register in the pair (higher-numbered).
    pub reg2: PReg,
    /// Offset from FP where this pair is stored (negative, growing down).
    pub fp_offset: i32,
    /// Whether this is an FPR pair (D-registers) vs GPR pair (X-registers).
    pub is_fpr: bool,
}

// ---------------------------------------------------------------------------
// FrameLayout — complete stack frame description
// ---------------------------------------------------------------------------

/// Complete description of a function's stack frame layout.
///
/// Computed after register allocation, before prologue/epilogue insertion.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    /// Callee-saved register pairs to save/restore (in save order).
    /// The first pair is always X29/X30 (FP/LR) when `uses_frame_pointer` is true.
    pub callee_saved_pairs: Vec<CalleeSavedPair>,

    /// Total size of the callee-saved register area (bytes).
    pub callee_saved_area_size: u32,

    /// Size of the spill slot area (from regalloc), in bytes.
    pub spill_area_size: u32,

    /// Size of local variable area (alloca, aggregates), in bytes.
    pub local_area_size: u32,

    /// Size of outgoing argument area (max across all calls), in bytes.
    pub outgoing_arg_area_size: u32,

    /// Total frame size (callee-saved + spills + locals + outgoing args),
    /// rounded up to 16-byte alignment.
    pub total_frame_size: u32,

    /// Whether this function uses a frame pointer (always true on Apple AArch64).
    pub uses_frame_pointer: bool,

    /// Whether this function is a leaf (no calls).
    pub is_leaf: bool,

    /// Whether the red zone optimization is applied (leaf + small frame).
    pub uses_red_zone: bool,

    /// Offset from FP to the start of the spill/local area.
    /// On AArch64: FP points at the saved FP/LR pair, so spills start
    /// at FP - callee_saved_area_size.
    pub fp_to_spill_offset: i32,
}

impl FrameLayout {
    /// Size of the SP adjustment needed after callee-save pushes.
    /// This is locals + spills + outgoing args.
    pub fn sp_adjustment(&self) -> u32 {
        self.total_frame_size - self.callee_saved_area_size
    }
}

// ---------------------------------------------------------------------------
// Frame layout computation
// ---------------------------------------------------------------------------

/// Determine which callee-saved GPR pairs are needed.
///
/// Scans all instructions for physical register uses/defs in X19-X28.
/// Returns a bitmask where bit N means X(19+N) is used.
fn scan_callee_saved_gprs(func: &MachFunction) -> u16 {
    let mut used = 0u16;
    for inst in &func.insts {
        for op in &inst.operands {
            if let MachOperand::PReg(preg) = op {
                let r = preg.encoding();
                // X19=19 through X28=28
                if (19..=28).contains(&r) {
                    used |= 1 << (r - 19);
                }
            }
        }
        // Check implicit defs/uses.
        for preg in inst.implicit_defs.iter().chain(inst.implicit_uses.iter()) {
            let r = preg.encoding();
            if (19..=28).contains(&r) {
                used |= 1 << (r - 19);
            }
        }
    }
    used
}

/// Determine which callee-saved FPR pairs are needed.
///
/// Scans all instructions for physical register uses/defs in V8-V15.
/// Returns a bitmask where bit N means V(8+N) is used.
fn scan_callee_saved_fprs(func: &MachFunction) -> u8 {
    let mut used = 0u8;
    for inst in &func.insts {
        for op in &inst.operands {
            if let MachOperand::PReg(preg) = op {
                let r = preg.encoding();
                // V8=72 through V15=79 (unified PReg encoding)
                if (72..=79).contains(&r) {
                    used |= 1 << (r - 72);
                }
            }
        }
        for preg in inst.implicit_defs.iter().chain(inst.implicit_uses.iter()) {
            let r = preg.encoding();
            if (72..=79).contains(&r) {
                used |= 1 << (r - 72);
            }
        }
    }
    used
}

/// Returns true if the function contains any call instructions.
fn has_calls(func: &MachFunction) -> bool {
    func.insts.iter().any(|inst| inst.is_call())
}

/// Compute the total size of all stack slots (spills + locals), respecting alignment.
fn compute_stack_slot_area(func: &MachFunction) -> u32 {
    let mut offset: u32 = 0;
    for slot in &func.stack_slots {
        // Align up to the slot's required alignment.
        offset = align_up(offset, slot.align);
        offset += slot.size;
    }
    offset
}

/// Compute the complete frame layout for a function.
///
/// This is called after register allocation, when physical register
/// assignments are known and spill slots have been allocated.
///
/// # Arguments
/// * `func` — The machine function (post-regalloc).
/// * `outgoing_arg_size` — Maximum outgoing argument area across all call sites.
/// * `enable_red_zone` — Whether to consider red zone optimization for leaf functions.
pub fn compute_frame_layout(
    func: &MachFunction,
    outgoing_arg_size: u32,
    enable_red_zone: bool,
) -> FrameLayout {
    let is_leaf = !has_calls(func);

    // On Apple AArch64, frame pointer is ALWAYS required.
    let uses_frame_pointer = true;

    // Scan for callee-saved register usage.
    let gpr_used = scan_callee_saved_gprs(func);
    let fpr_used = scan_callee_saved_fprs(func);

    // Build callee-saved pairs. FP/LR is always first.
    let mut pairs = Vec::new();
    let mut csa_offset: i32 = -16; // FP/LR pair is at [FP, #0] but stored with offset -16 from old SP

    // Always save FP/LR pair.
    pairs.push(CalleeSavedPair {
        reg1: X29,
        reg2: X30,
        fp_offset: 0, // FP points exactly at saved FP/LR
        is_fpr: false,
    });

    // GPR pairs: X19/X20, X21/X22, X23/X24, X25/X26, X27/X28
    let gpr_pair_regs: [(PReg, PReg, u16); 5] = [
        (X19, X20, 0b0000_0000_0011), // bits 0,1
        (X21, X22, 0b0000_0000_1100), // bits 2,3
        (X23, X24, 0b0000_0011_0000), // bits 4,5
        (X25, X26, 0b0000_1100_0000), // bits 6,7
        (X27, X28, 0b0011_0000_0000), // bits 8,9
    ];

    for (reg1, reg2, mask) in &gpr_pair_regs {
        if gpr_used & mask != 0 {
            csa_offset -= 16;
            pairs.push(CalleeSavedPair {
                reg1: *reg1,
                reg2: *reg2,
                fp_offset: csa_offset,
                is_fpr: false,
            });
        }
    }

    // FPR pairs: D8/D9, D10/D11, D12/D13, D14/D15
    // (V8-V15 in our encoding, but we save the lower 64 bits = D-regs)
    let fpr_pair_regs: [(PReg, PReg, u8); 4] = [
        (V8, V9, 0b0000_0011),   // bits 0,1
        (V10, V11, 0b0000_1100), // bits 2,3
        (V12, V13, 0b0011_0000), // bits 4,5
        (V14, V15, 0b1100_0000), // bits 6,7
    ];

    for (reg1, reg2, mask) in &fpr_pair_regs {
        if fpr_used & mask != 0 {
            csa_offset -= 16;
            pairs.push(CalleeSavedPair {
                reg1: *reg1,
                reg2: *reg2,
                fp_offset: csa_offset,
                is_fpr: true,
            });
        }
    }

    // Callee-saved area = 16 bytes per pair (always 16-byte aligned by construction).
    let callee_saved_area_size = (pairs.len() as u32) * 16;

    // Stack slot area (spills + locals).
    let stack_slot_area = compute_stack_slot_area(func);

    // Outgoing argument area (only non-leaf functions need this).
    let outgoing_arg_area_size = if is_leaf { 0 } else { align_up(outgoing_arg_size, STACK_ALIGNMENT) };

    // Total frame = callee-saved + stack slots + outgoing args, aligned to 16.
    let raw_total = callee_saved_area_size + stack_slot_area + outgoing_arg_area_size;
    let total_frame_size = align_up(raw_total, STACK_ALIGNMENT);

    // Red zone: leaf function with no stack slots and total frame <= 128 bytes.
    let uses_red_zone = enable_red_zone
        && is_leaf
        && stack_slot_area == 0
        && outgoing_arg_area_size == 0
        && total_frame_size <= RED_ZONE_SIZE;

    let fp_to_spill_offset = -(callee_saved_area_size as i32);

    FrameLayout {
        callee_saved_pairs: pairs,
        callee_saved_area_size,
        spill_area_size: stack_slot_area,
        local_area_size: 0, // Currently combined with spill_area_size
        outgoing_arg_area_size,
        total_frame_size,
        uses_frame_pointer,
        is_leaf,
        uses_red_zone,
        fp_to_spill_offset,
    }
}

// ---------------------------------------------------------------------------
// Prologue emission
// ---------------------------------------------------------------------------

/// Generate prologue instructions for the function.
///
/// # Prologue sequence (standard frame-pointer frame)
///
/// ```asm
/// ; Save FP/LR (pre-index decrement)
/// stp  x29, x30, [sp, #-CSA_SIZE]!    ; allocate callee-saved area
/// mov  x29, sp                         ; establish frame pointer
///
/// ; Save callee-saved GPR/FPR pairs (positive offsets from SP)
/// stp  x19, x20, [sp, #16]
/// stp  x21, x22, [sp, #32]
/// ...
/// stp  d8,  d9,  [sp, #N]
/// ...
///
/// ; Allocate local + spill + outgoing arg space
/// sub  sp, sp, #(total_frame - callee_saved_area)
/// ```
///
/// Returns the generated instructions in order.
pub fn emit_prologue(layout: &FrameLayout) -> Vec<MachInst> {
    let mut insts = Vec::new();

    if layout.uses_red_zone && layout.sp_adjustment() == 0 && layout.callee_saved_pairs.len() <= 1 {
        // Red zone: minimal prologue for trivial leaf functions.
        // Still save FP/LR if frame pointer is used (Apple requires it).
        if layout.uses_frame_pointer && !layout.callee_saved_pairs.is_empty() {
            // STP X29, X30, [SP, #-16]!
            insts.push(make_stp_pre_index(X29, X30, -(16 as i64)));
            // MOV X29, SP
            insts.push(make_mov_sp_to_fp());
        }
        return insts;
    }

    if layout.callee_saved_pairs.is_empty() {
        // No callee-saves — just allocate the frame.
        let adj = layout.sp_adjustment();
        if adj > 0 {
            insts.push(make_sub_sp_imm(adj as i64));
        }
        return insts;
    }

    // Save FP/LR with pre-index to allocate the callee-saved area.
    // STP X29, X30, [SP, #-callee_saved_area_size]!
    let csa = layout.callee_saved_area_size as i64;
    insts.push(make_stp_pre_index(X29, X30, -csa));

    // Establish frame pointer: MOV X29, SP
    if layout.uses_frame_pointer {
        insts.push(make_mov_sp_to_fp());
    }

    // Save remaining callee-saved pairs at positive offsets from current SP.
    // Pairs[0] is FP/LR (already saved), so start at index 1.
    for (i, pair) in layout.callee_saved_pairs.iter().enumerate().skip(1) {
        let offset = (i as i64) * 16;
        insts.push(make_stp_offset(pair.reg1, pair.reg2, offset));
    }

    // Allocate locals + spills + outgoing args.
    let sp_adj = layout.sp_adjustment();
    if sp_adj > 0 {
        insts.push(make_sub_sp_imm(sp_adj as i64));
    }

    insts
}

// ---------------------------------------------------------------------------
// Epilogue emission
// ---------------------------------------------------------------------------

/// Generate epilogue instructions for the function.
///
/// # Epilogue sequence
///
/// ```asm
/// ; Deallocate local + spill + outgoing arg space
/// add  sp, sp, #(total_frame - callee_saved_area)
///
/// ; Restore callee-saved pairs (reverse order of save, positive offsets)
/// ldp  d8,  d9,  [sp, #N]
/// ...
/// ldp  x21, x22, [sp, #32]
/// ldp  x19, x20, [sp, #16]
///
/// ; Restore FP/LR (post-index increment)
/// ldp  x29, x30, [sp], #CSA_SIZE
///
/// ret
/// ```
///
/// Returns the generated instructions in order.
pub fn emit_epilogue(layout: &FrameLayout) -> Vec<MachInst> {
    let mut insts = Vec::new();

    if layout.uses_red_zone && layout.sp_adjustment() == 0 && layout.callee_saved_pairs.len() <= 1 {
        // Red zone: minimal epilogue.
        if layout.uses_frame_pointer && !layout.callee_saved_pairs.is_empty() {
            // LDP X29, X30, [SP], #16
            insts.push(make_ldp_post_index(X29, X30, 16));
        }
        insts.push(make_ret());
        return insts;
    }

    // Deallocate locals + spills + outgoing args.
    let sp_adj = layout.sp_adjustment();
    if sp_adj > 0 {
        insts.push(make_add_sp_imm(sp_adj as i64));
    }

    // Restore callee-saved pairs in reverse order (skip pair[0] = FP/LR).
    let num_pairs = layout.callee_saved_pairs.len();
    for i in (1..num_pairs).rev() {
        let pair = &layout.callee_saved_pairs[i];
        let offset = (i as i64) * 16;
        insts.push(make_ldp_offset(pair.reg1, pair.reg2, offset));
    }

    // Restore FP/LR with post-index to deallocate callee-saved area.
    if !layout.callee_saved_pairs.is_empty() {
        let csa = layout.callee_saved_area_size as i64;
        insts.push(make_ldp_post_index(X29, X30, csa));
    }

    // Return.
    insts.push(make_ret());

    insts
}

// ---------------------------------------------------------------------------
// Frame index elimination
// ---------------------------------------------------------------------------

/// Resolve all FrameIndex operands in the function to concrete
/// SP+offset or FP+offset memory operands.
///
/// After this pass, no FrameIndex operands should remain in the function.
///
/// # Addressing strategy
///
/// When a frame pointer is available (always on Apple AArch64):
/// - Spill slots use FP-relative addressing (stable across SP changes).
/// - Outgoing args use SP-relative addressing.
///
/// Frame index encoding: `FrameIdx(i)` where `i` is the stack slot index.
/// The concrete offset is computed from the frame layout.
pub fn eliminate_frame_indices(func: &mut MachFunction, layout: &FrameLayout) {
    // Precompute stack slot offsets from FP.
    // Stack slots are in the spill/local area, which starts at FP - callee_saved_area_size.
    let slot_offsets = compute_slot_offsets(func, layout);

    // Walk all instructions and rewrite FrameIndex operands.
    for inst in &mut func.insts {
        for operand in &mut inst.operands {
            if let MachOperand::FrameIndex(fi) = operand {
                let slot_idx = fi.0 as usize;
                if slot_idx < slot_offsets.len() {
                    let offset = slot_offsets[slot_idx];
                    if layout.uses_frame_pointer {
                        // FP-relative: MemOp { base: X29, offset }
                        *operand = MachOperand::MemOp {
                            base: X29,
                            offset: offset as i64,
                        };
                    } else {
                        // SP-relative: offset from current SP
                        // SP-relative offset = FP_offset + callee_saved_area + sp_adjustment
                        let sp_offset = offset + (layout.sp_adjustment() as i32);
                        *operand = MachOperand::MemOp {
                            base: SP, // SP = PReg(31)
                            offset: sp_offset as i64,
                        };
                    }
                }
            }
        }
    }
}

/// Compute the FP-relative offset for each stack slot.
///
/// Returns a vector indexed by stack slot index, where each value is the
/// signed offset from FP to the start of that slot.
fn compute_slot_offsets(func: &MachFunction, layout: &FrameLayout) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(func.stack_slots.len());
    // Spill/local area starts at FP - callee_saved_area_size.
    // We lay out slots growing downward from there.
    let mut current_offset = layout.fp_to_spill_offset;

    for slot in &func.stack_slots {
        // Grow downward: subtract size, then align.
        current_offset -= slot.size as i32;
        // Align the offset (make it more negative if needed).
        let align = slot.align as i32;
        if align > 0 {
            // Round down to alignment boundary (for negative offsets).
            current_offset = current_offset & !(align - 1);
        }
        offsets.push(current_offset);
    }

    offsets
}

// ---------------------------------------------------------------------------
// Compact unwind encoding
// ---------------------------------------------------------------------------

/// Darwin compact unwind encoding for an AArch64 function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactUnwindEncoding {
    /// The 32-bit encoding value.
    pub encoding: u32,
}

/// Encode the Darwin compact unwind for a frame layout.
///
/// For standard FP-based frames, this produces UNWIND_ARM64_MODE_FRAME
/// with register-pair flags indicating which callee-saved registers are saved.
///
/// Reference: AArch64AsmBackend.cpp `generateCompactUnwindEncoding()` (line 576)
pub fn encode_compact_unwind(layout: &FrameLayout) -> CompactUnwindEncoding {
    if !layout.uses_frame_pointer {
        // Frameless functions use DWARF fallback for now.
        return CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
    }

    let mut encoding = UNWIND_ARM64_MODE_FRAME;

    // Encode which callee-saved pairs are saved (skip pair[0] = FP/LR, always implicit).
    for pair in layout.callee_saved_pairs.iter().skip(1) {
        if !pair.is_fpr {
            // GPR pair — X19-X28 encoding unchanged (19-28)
            let flag = match (pair.reg1.encoding(), pair.reg2.encoding()) {
                (19, 20) => UNWIND_ARM64_FRAME_X19_X20_PAIR,
                (21, 22) => UNWIND_ARM64_FRAME_X21_X22_PAIR,
                (23, 24) => UNWIND_ARM64_FRAME_X23_X24_PAIR,
                (25, 26) => UNWIND_ARM64_FRAME_X25_X26_PAIR,
                (27, 28) => UNWIND_ARM64_FRAME_X27_X28_PAIR,
                _ => 0, // Unknown pair — should not happen
            };
            encoding |= flag;
        } else {
            // FPR pair (V8-V15 encode as D8-D15 for compact unwind)
            // V8=72, V9=73, ..., V15=79 in unified PReg encoding
            let flag = match (pair.reg1.encoding(), pair.reg2.encoding()) {
                (72, 73) => UNWIND_ARM64_FRAME_D8_D9_PAIR,   // V8/V9
                (74, 75) => UNWIND_ARM64_FRAME_D10_D11_PAIR, // V10/V11
                (76, 77) => UNWIND_ARM64_FRAME_D12_D13_PAIR, // V12/V13
                (78, 79) => UNWIND_ARM64_FRAME_D14_D15_PAIR, // V14/V15
                _ => 0,
            };
            encoding |= flag;
        }
    }

    CompactUnwindEncoding { encoding }
}

// ---------------------------------------------------------------------------
// Instruction builders (private helpers)
// ---------------------------------------------------------------------------

/// STP Rt, Rt2, [SP, #-imm]! (pre-index, allocates stack space)
fn make_stp_pre_index(reg1: PReg, reg2: PReg, offset: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::StpPreIndex,
        vec![
            MachOperand::PReg(reg1),
            MachOperand::PReg(reg2),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(offset),
        ],
    )
}

/// STP Rt, Rt2, [SP, #imm] (signed offset from current SP)
fn make_stp_offset(reg1: PReg, reg2: PReg, offset: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::StpRI,
        vec![
            MachOperand::PReg(reg1),
            MachOperand::PReg(reg2),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(offset),
        ],
    )
}

/// LDP Rt, Rt2, [SP, #imm] (signed offset load pair)
fn make_ldp_offset(reg1: PReg, reg2: PReg, offset: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::LdpRI,
        vec![
            MachOperand::PReg(reg1),
            MachOperand::PReg(reg2),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(offset),
        ],
    )
}

/// LDP Rt, Rt2, [SP], #imm (post-index, deallocates stack space)
fn make_ldp_post_index(reg1: PReg, reg2: PReg, offset: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::LdpPostIndex,
        vec![
            MachOperand::PReg(reg1),
            MachOperand::PReg(reg2),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(offset),
        ],
    )
}

/// MOV X29, SP (establish frame pointer)
///
/// Encoded as ADD X29, SP, #0 because register 31 in ADD context is SP,
/// whereas in ORR (logical) context register 31 is XZR.
fn make_mov_sp_to_fp() -> MachInst {
    MachInst::new(
        AArch64Opcode::AddRI,
        vec![
            MachOperand::PReg(X29),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(0),
        ],
    )
}

/// SUB SP, SP, #imm (allocate stack space)
fn make_sub_sp_imm(imm: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::SubRI,
        vec![
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(imm),
        ],
    )
}

/// ADD SP, SP, #imm (deallocate stack space)
fn make_add_sp_imm(imm: i64) -> MachInst {
    MachInst::new(
        AArch64Opcode::AddRI,
        vec![
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Imm(imm),
        ],
    )
}

/// RET (return via X30)
fn make_ret() -> MachInst {
    MachInst::new(AArch64Opcode::Ret, vec![])
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Align `value` up to the next multiple of `align`.
///
/// `align` must be a power of two.
#[inline]
fn align_up(value: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// Prologue/epilogue insertion into a MachFunction
// ---------------------------------------------------------------------------

/// Insert prologue instructions at the beginning of the entry block,
/// and epilogue instructions before every return instruction.
///
/// This is the main entry point for frame lowering after layout computation.
pub fn insert_prologue_epilogue(func: &mut MachFunction, layout: &FrameLayout) {
    let prologue = emit_prologue(layout);
    let epilogue_before_ret = emit_epilogue(layout);
    // The epilogue includes RET, so we strip the existing RET instructions
    // from blocks and let the epilogue provide them.

    // Insert prologue at the start of the entry block.
    let entry = func.entry;
    let entry_block = &func.blocks[entry.0 as usize];
    let old_entry_insts = entry_block.insts.clone();

    let mut new_entry_insts = Vec::new();
    for prologue_inst in prologue {
        let id = func.push_inst(prologue_inst);
        new_entry_insts.push(id);
    }
    new_entry_insts.extend(old_entry_insts);
    func.blocks[entry.0 as usize].insts = new_entry_insts;

    // For each block, find return instructions and insert epilogue before them.
    let num_blocks = func.blocks.len();
    for block_idx in 0..num_blocks {
        let block_insts = func.blocks[block_idx].insts.clone();
        let mut new_insts = Vec::new();
        for &inst_id in &block_insts {
            let inst = func.inst(inst_id);
            if inst.is_return() {
                // Insert epilogue (which includes its own RET).
                // Drop the original return instruction.
                for epi_inst in &epilogue_before_ret {
                    let id = func.push_inst(epi_inst.clone());
                    new_insts.push(id);
                }
            } else {
                new_insts.push(inst_id);
            }
        }
        func.blocks[block_idx].insts = new_insts;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::function::{MachFunction, Signature, StackSlot};
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    use llvm2_ir::operand::MachOperand;
    use llvm2_ir::regs::PReg;
    use llvm2_ir::types::{BlockId, FrameIdx};

    /// Helper: create a minimal function with given instructions in entry block.
    fn make_func(name: &str, insts: Vec<MachInst>) -> MachFunction {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new(name.to_string(), sig);
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(BlockId(0), id);
        }
        func
    }

    /// Helper: create a function that uses specific callee-saved GPRs.
    fn make_func_with_callee_saved_gprs(regs: &[PReg]) -> MachFunction {
        let mut insts = vec![];
        for &r in regs {
            // Use the register in a simple MOV.
            insts.push(MachInst::new(
                AArch64Opcode::MovR,
                vec![MachOperand::PReg(r), MachOperand::PReg(r)],
            ));
        }
        insts.push(MachInst::new(AArch64Opcode::Ret, vec![]));
        make_func("test_cs_gprs", insts)
    }

    /// Helper: create a function that uses specific callee-saved FPRs.
    fn make_func_with_callee_saved_fprs(regs: &[PReg]) -> MachFunction {
        let mut insts = vec![];
        for &r in regs {
            insts.push(MachInst::new(
                AArch64Opcode::MovR,
                vec![MachOperand::PReg(r), MachOperand::PReg(r)],
            ));
        }
        insts.push(MachInst::new(AArch64Opcode::Ret, vec![]));
        make_func("test_cs_fprs", insts)
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(32, 16), 32);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
    }

    #[test]
    fn test_layout_empty_leaf() {
        // Empty leaf function: no callee-saves, no spills, no calls.
        let func = make_func("empty_leaf", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, false);

        assert!(layout.is_leaf);
        assert!(layout.uses_frame_pointer);
        // FP/LR pair always saved on Apple.
        assert_eq!(layout.callee_saved_pairs.len(), 1);
        assert_eq!(layout.callee_saved_area_size, 16);
        assert_eq!(layout.spill_area_size, 0);
        assert_eq!(layout.outgoing_arg_area_size, 0);
        assert_eq!(layout.total_frame_size, 16);
    }

    #[test]
    fn test_layout_with_callee_saved_gprs() {
        // Function uses X19 and X20.
        let func = make_func_with_callee_saved_gprs(&[X19, X20]);
        let layout = compute_frame_layout(&func, 0, false);

        // FP/LR + X19/X20 pair = 2 pairs.
        assert_eq!(layout.callee_saved_pairs.len(), 2);
        assert_eq!(layout.callee_saved_area_size, 32);
        assert_eq!(layout.total_frame_size, 32);
    }

    #[test]
    fn test_layout_with_all_callee_saved() {
        // Function uses all callee-saved GPRs (X19-X28) and all FPRs (V8-V15).
        let mut regs: Vec<PReg> = (19..=28).map(|r| PReg::new(r)).collect();
        let fprs: Vec<PReg> = (72..=79).map(|r| PReg::new(r)).collect(); // V8-V15
        regs.extend(fprs);
        let func = make_func_with_callee_saved_gprs(&regs);
        let layout = compute_frame_layout(&func, 0, false);

        // FP/LR + 5 GPR pairs + 4 FPR pairs = 10 pairs.
        assert_eq!(layout.callee_saved_pairs.len(), 10);
        assert_eq!(layout.callee_saved_area_size, 160);
        assert_eq!(layout.total_frame_size, 160);
    }

    #[test]
    fn test_layout_with_spills() {
        // Function with 3 spill slots.
        let mut func = make_func("spills", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));
        func.alloc_stack_slot(StackSlot::new(8, 8));
        func.alloc_stack_slot(StackSlot::new(4, 4));

        let layout = compute_frame_layout(&func, 0, false);

        // Stack slots: 8 + 8 + 4 = 20 bytes.
        assert_eq!(layout.spill_area_size, 20);
        // Total: 16 (FP/LR) + 20 (spills) = 36, aligned to 48.
        assert_eq!(layout.total_frame_size, 48);
    }

    #[test]
    fn test_layout_with_outgoing_args() {
        // Non-leaf function with outgoing args.
        let func = make_func("with_call", vec![
            MachInst::new(AArch64Opcode::Bl, vec![MachOperand::Imm(0)]),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 24, false);

        assert!(!layout.is_leaf);
        // Outgoing args: 24 aligned to 32.
        assert_eq!(layout.outgoing_arg_area_size, 32);
        // Total: 16 (FP/LR) + 32 (args) = 48.
        assert_eq!(layout.total_frame_size, 48);
    }

    #[test]
    fn test_layout_alignment_enforcement() {
        // Ensure total frame size is always 16-byte aligned.
        let mut func = make_func("align_test", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        // One 1-byte slot: forces alignment padding.
        func.alloc_stack_slot(StackSlot::new(1, 1));

        let layout = compute_frame_layout(&func, 0, false);

        assert_eq!(layout.total_frame_size % 16, 0);
        // 16 (FP/LR) + 1 (slot) = 17, aligned to 32.
        assert_eq!(layout.total_frame_size, 32);
    }

    #[test]
    fn test_red_zone_eligible() {
        // Leaf function with no spills and small frame.
        let func = make_func("leaf", vec![
            MachInst::new(AArch64Opcode::Nop, vec![]),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, true);

        assert!(layout.is_leaf);
        assert!(layout.uses_red_zone);
    }

    #[test]
    fn test_red_zone_disabled_for_non_leaf() {
        let func = make_func("non_leaf", vec![
            MachInst::new(AArch64Opcode::Bl, vec![MachOperand::Imm(0)]),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, true);

        assert!(!layout.is_leaf);
        assert!(!layout.uses_red_zone);
    }

    #[test]
    fn test_red_zone_disabled_with_spills() {
        let mut func = make_func("leaf_spills", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));
        let layout = compute_frame_layout(&func, 0, true);

        assert!(layout.is_leaf);
        assert!(!layout.uses_red_zone); // Has stack slots.
    }

    #[test]
    fn test_prologue_simple() {
        // Simple function: just FP/LR, no other callee-saves, no spills.
        let func = make_func("simple", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, false);
        let prologue = emit_prologue(&layout);

        // Expect: STP X29, X30, [SP, #-16]!; ADD X29, SP, #0 (MOV X29, SP)
        assert_eq!(prologue.len(), 2);
        assert_eq!(prologue[0].opcode, AArch64Opcode::StpPreIndex);
        assert_eq!(prologue[1].opcode, AArch64Opcode::AddRI);
    }

    #[test]
    fn test_prologue_with_callee_saves_and_spills() {
        let mut func = make_func_with_callee_saved_gprs(&[X19, X20, X21, X22]);
        func.alloc_stack_slot(StackSlot::new(16, 8));

        let layout = compute_frame_layout(&func, 0, false);
        let prologue = emit_prologue(&layout);

        // STP X29, X30, [SP, #-48]!  (3 pairs * 16 = 48, pre-index)
        // ADD X29, SP, #0            (MOV X29, SP)
        // STP X19, X20, [SP, #16]    (signed offset)
        // STP X21, X22, [SP, #32]    (signed offset)
        // SUB SP, SP, #16            (spill area, aligned)
        assert_eq!(prologue.len(), 5);
        assert_eq!(prologue[0].opcode, AArch64Opcode::StpPreIndex);
        assert_eq!(prologue[1].opcode, AArch64Opcode::AddRI);
        assert_eq!(prologue[2].opcode, AArch64Opcode::StpRI);
        assert_eq!(prologue[3].opcode, AArch64Opcode::StpRI);
        assert_eq!(prologue[4].opcode, AArch64Opcode::SubRI);
    }

    #[test]
    fn test_epilogue_simple() {
        let func = make_func("simple", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, false);
        let epilogue = emit_epilogue(&layout);

        // LDP X29, X30, [SP], #16 (post-index); RET
        assert_eq!(epilogue.len(), 2);
        assert_eq!(epilogue[0].opcode, AArch64Opcode::LdpPostIndex);
        assert_eq!(epilogue[1].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn test_epilogue_with_callee_saves_and_spills() {
        let mut func = make_func_with_callee_saved_gprs(&[X19, X20, X21, X22]);
        func.alloc_stack_slot(StackSlot::new(16, 8));

        let layout = compute_frame_layout(&func, 0, false);
        let epilogue = emit_epilogue(&layout);

        // ADD SP, SP, #16
        // LDP X21, X22, [SP, #32]   (signed offset)
        // LDP X19, X20, [SP, #16]   (signed offset)
        // LDP X29, X30, [SP], #48   (post-index)
        // RET
        assert_eq!(epilogue.len(), 5);
        assert_eq!(epilogue[0].opcode, AArch64Opcode::AddRI);
        assert_eq!(epilogue[1].opcode, AArch64Opcode::LdpRI);
        assert_eq!(epilogue[2].opcode, AArch64Opcode::LdpRI);
        assert_eq!(epilogue[3].opcode, AArch64Opcode::LdpPostIndex);
        assert_eq!(epilogue[4].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn test_prologue_epilogue_symmetry() {
        // Verify prologue saves and epilogue restores match.
        let func = make_func_with_callee_saved_gprs(&[X19, X20, X25, X26]);
        let layout = compute_frame_layout(&func, 0, false);

        let prologue = emit_prologue(&layout);
        let epilogue = emit_epilogue(&layout);

        // Count STP in prologue vs LDP in epilogue.
        // STP: StpPreIndex (FP/LR) + StpRI (X19/X20) + StpRI (X25/X26) = 1 pre-index + 2 offset = 3 total
        // LDP: LdpRI (X25/X26) + LdpRI (X19/X20) + LdpPostIndex (FP/LR) = 2 offset + 1 post-index = 3 total
        let stp_count = prologue.iter().filter(|i| i.opcode == AArch64Opcode::StpRI || i.opcode == AArch64Opcode::StpPreIndex).count();
        let ldp_count = epilogue.iter().filter(|i| i.opcode == AArch64Opcode::LdpRI || i.opcode == AArch64Opcode::LdpPostIndex).count();

        assert_eq!(stp_count, 3);
        assert_eq!(ldp_count, 3);
    }

    #[test]
    fn test_frame_index_elimination() {
        // Create a function with a FrameIndex operand and eliminate it.
        let mut func = make_func("fi_test", vec![
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![
                    MachOperand::PReg(PReg::new(0)), // X0
                    MachOperand::FrameIndex(FrameIdx(0)),
                ],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));

        let layout = compute_frame_layout(&func, 0, false);
        eliminate_frame_indices(&mut func, &layout);

        // The FrameIndex should be replaced with a MemOp.
        let inst = &func.insts[0];
        match &inst.operands[1] {
            MachOperand::MemOp { base, offset } => {
                assert_eq!(*base, X29); // FP-relative
                // Offset should be negative (below FP).
                assert!(*offset < 0, "FP offset should be negative, got {}", offset);
            }
            other => panic!("Expected MemOp, got {:?}", other),
        }
    }

    #[test]
    fn test_frame_index_multiple_slots() {
        // Multiple stack slots with different alignments.
        let mut func = make_func("fi_multi", vec![
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(0)), MachOperand::FrameIndex(FrameIdx(0))],
            ),
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(1)), MachOperand::FrameIndex(FrameIdx(1))],
            ),
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(2)), MachOperand::FrameIndex(FrameIdx(2))],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));  // Slot 0
        func.alloc_stack_slot(StackSlot::new(4, 4));  // Slot 1
        func.alloc_stack_slot(StackSlot::new(1, 1));  // Slot 2

        let layout = compute_frame_layout(&func, 0, false);
        eliminate_frame_indices(&mut func, &layout);

        // All FrameIndex operands should be replaced.
        for i in 0..3 {
            match &func.insts[i].operands[1] {
                MachOperand::MemOp { base, .. } => {
                    assert_eq!(*base, X29);
                }
                other => panic!("Slot {} not eliminated: {:?}", i, other),
            }
        }

        // Verify offsets are distinct and decreasing.
        let offsets: Vec<i64> = (0..3)
            .map(|i| match &func.insts[i].operands[1] {
                MachOperand::MemOp { offset, .. } => *offset,
                _ => unreachable!(),
            })
            .collect();

        // Each subsequent slot should be at a lower (more negative) offset.
        assert!(offsets[0] > offsets[1], "slot 0 offset {} > slot 1 offset {}", offsets[0], offsets[1]);
        assert!(offsets[1] > offsets[2], "slot 1 offset {} > slot 2 offset {}", offsets[1], offsets[2]);
    }

    #[test]
    fn test_compact_unwind_fp_only() {
        // Function with only FP/LR saved.
        let func = make_func("fp_only", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, false);
        let cu = encode_compact_unwind(&layout);

        // Should be FRAME mode with no register-pair flags.
        assert_eq!(cu.encoding, UNWIND_ARM64_MODE_FRAME);
    }

    #[test]
    fn test_compact_unwind_with_gpr_pairs() {
        let func = make_func_with_callee_saved_gprs(&[X19, X20, X21, X22]);
        let layout = compute_frame_layout(&func, 0, false);
        let cu = encode_compact_unwind(&layout);

        let expected = UNWIND_ARM64_MODE_FRAME
            | UNWIND_ARM64_FRAME_X19_X20_PAIR
            | UNWIND_ARM64_FRAME_X21_X22_PAIR;
        assert_eq!(cu.encoding, expected);
    }

    #[test]
    fn test_compact_unwind_with_fpr_pairs() {
        let func = make_func_with_callee_saved_fprs(&[V8, V9, V10, V11]);
        let layout = compute_frame_layout(&func, 0, false);
        let cu = encode_compact_unwind(&layout);

        let expected = UNWIND_ARM64_MODE_FRAME
            | UNWIND_ARM64_FRAME_D8_D9_PAIR
            | UNWIND_ARM64_FRAME_D10_D11_PAIR;
        assert_eq!(cu.encoding, expected);
    }

    #[test]
    fn test_compact_unwind_all_regs() {
        // All callee-saved registers.
        let mut regs: Vec<PReg> = (19..=28).map(|r| PReg::new(r)).collect();
        let fprs: Vec<PReg> = (72..=79).map(|r| PReg::new(r)).collect(); // V8-V15
        regs.extend(fprs);
        let func = make_func_with_callee_saved_gprs(&regs);
        let layout = compute_frame_layout(&func, 0, false);
        let cu = encode_compact_unwind(&layout);

        let expected = UNWIND_ARM64_MODE_FRAME
            | UNWIND_ARM64_FRAME_X19_X20_PAIR
            | UNWIND_ARM64_FRAME_X21_X22_PAIR
            | UNWIND_ARM64_FRAME_X23_X24_PAIR
            | UNWIND_ARM64_FRAME_X25_X26_PAIR
            | UNWIND_ARM64_FRAME_X27_X28_PAIR
            | UNWIND_ARM64_FRAME_D8_D9_PAIR
            | UNWIND_ARM64_FRAME_D10_D11_PAIR
            | UNWIND_ARM64_FRAME_D12_D13_PAIR
            | UNWIND_ARM64_FRAME_D14_D15_PAIR;
        assert_eq!(cu.encoding, expected);
    }

    #[test]
    fn test_compact_unwind_single_gpr() {
        // Only X19 used — still saves X19/X20 as a pair.
        let func = make_func_with_callee_saved_gprs(&[X19]);
        let layout = compute_frame_layout(&func, 0, false);
        let cu = encode_compact_unwind(&layout);

        let expected = UNWIND_ARM64_MODE_FRAME | UNWIND_ARM64_FRAME_X19_X20_PAIR;
        assert_eq!(cu.encoding, expected);
    }

    #[test]
    fn test_insert_prologue_epilogue() {
        // Create a simple function and insert prologue/epilogue.
        let func_insts = vec![
            MachInst::new(AArch64Opcode::Nop, vec![]),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ];
        let mut func = make_func("insert_test", func_insts);
        let layout = compute_frame_layout(&func, 0, false);

        insert_prologue_epilogue(&mut func, &layout);

        // Entry block should start with prologue instructions.
        let entry_insts = &func.blocks[0].insts;
        assert!(entry_insts.len() >= 3, "Expected at least prologue + NOP + epilogue");

        // First instruction should be STP pre-index (prologue start).
        assert_eq!(func.inst(entry_insts[0]).opcode, AArch64Opcode::StpPreIndex);

        // Last instruction should be RET (from epilogue).
        let last_id = *entry_insts.last().unwrap();
        assert_eq!(func.inst(last_id).opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn test_sp_adjustment() {
        let mut func = make_func("sp_adj", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(32, 8));

        let layout = compute_frame_layout(&func, 0, false);

        // sp_adjustment = total - callee_saved_area
        assert_eq!(layout.sp_adjustment(), layout.total_frame_size - layout.callee_saved_area_size);
        assert!(layout.sp_adjustment() >= 32); // At least covers the 32-byte slot.
    }

    #[test]
    fn test_callee_saved_pair_offsets() {
        // Verify callee-saved pair FP offsets are correct and non-overlapping.
        let func = make_func_with_callee_saved_gprs(&[X19, X20, X23, X24]);
        let layout = compute_frame_layout(&func, 0, false);

        // Pair 0: FP/LR at offset 0
        assert_eq!(layout.callee_saved_pairs[0].fp_offset, 0);
        assert_eq!(layout.callee_saved_pairs[0].reg1, X29);

        // Subsequent pairs have decreasing (more negative) offsets.
        for i in 1..layout.callee_saved_pairs.len() {
            assert!(layout.callee_saved_pairs[i].fp_offset < layout.callee_saved_pairs[i - 1].fp_offset);
        }
    }

    #[test]
    fn test_scan_callee_saved_gprs() {
        let func = make_func_with_callee_saved_gprs(&[X19, X22, X28]);
        let mask = scan_callee_saved_gprs(&func);

        assert!(mask & (1 << 0) != 0); // X19
        assert!(mask & (1 << 1) == 0); // X20 not used
        assert!(mask & (1 << 3) != 0); // X22
        assert!(mask & (1 << 9) != 0); // X28
    }

    #[test]
    fn test_scan_callee_saved_fprs() {
        let func = make_func_with_callee_saved_fprs(&[V8, V11, V15]);
        let mask = scan_callee_saved_fprs(&func);

        assert!(mask & (1 << 0) != 0); // V8
        assert!(mask & (1 << 3) != 0); // V11
        assert!(mask & (1 << 7) != 0); // V15
        assert!(mask & (1 << 1) == 0); // V9 not used
    }
}

// llvm2-codegen - Frame lowering for AArch64 macOS
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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

    /// Whether this function uses dynamic stack allocation (alloca).
    ///
    /// Dynamic allocation means the stack pointer moves by an amount unknown
    /// at compile time, so compact unwind cannot describe the frame. When true,
    /// `encode_compact_unwind` returns `UNWIND_ARM64_MODE_DWARF` to request
    /// DWARF CFI fallback.
    pub has_dynamic_alloc: bool,
}

impl FrameLayout {
    /// Size of the SP adjustment needed after callee-save pushes.
    /// This is locals + spills + outgoing args.
    #[inline]
    pub fn sp_adjustment(&self) -> u32 {
        self.total_frame_size - self.callee_saved_area_size
    }
}

// ---------------------------------------------------------------------------
// Frame layout computation
// ---------------------------------------------------------------------------

/// Results of a single-pass scan over all instructions.
///
/// Replaces the previous three separate passes (`scan_callee_saved_gprs`,
/// `scan_callee_saved_fprs`, `has_calls`) with one pass for better cache
/// utilization on large functions.
struct ScanResult {
    /// Bitmask of callee-saved GPRs used (bit N = X(19+N)).
    gpr_used: u16,
    /// Bitmask of callee-saved FPRs used (bit N = V(8+N)).
    fpr_used: u8,
    /// Whether the function contains any call instructions.
    has_calls: bool,
}

/// Scan all instructions in a single pass to determine:
/// - Which callee-saved GPRs (X19-X28) are used
/// - Which callee-saved FPRs (V8-V15) are used
/// - Whether the function contains any call instructions
///
/// This merged scan replaces three separate iteration passes for better
/// cache locality and reduced overhead on functions with many instructions.
#[inline]
fn scan_function(func: &MachFunction) -> ScanResult {
    let mut gpr_used = 0u16;
    let mut fpr_used = 0u8;
    let mut has_calls = false;

    for inst in &func.insts {
        // Check for call instructions.
        if !has_calls && inst.is_call() {
            has_calls = true;
        }

        // Scan explicit operands for callee-saved register uses.
        for op in &inst.operands {
            if let MachOperand::PReg(preg) = op {
                let r = preg.encoding();
                // X19=19 through X28=28
                if (19..=28).contains(&r) {
                    gpr_used |= 1 << (r - 19);
                }
                // V8=72 through V15=79 (unified PReg encoding)
                else if (72..=79).contains(&r) {
                    fpr_used |= 1 << (r - 72);
                }
            }
        }

        // Check implicit defs/uses.
        for preg in inst.implicit_defs.iter().chain(inst.implicit_uses.iter()) {
            let r = preg.encoding();
            if (19..=28).contains(&r) {
                gpr_used |= 1 << (r - 19);
            } else if (72..=79).contains(&r) {
                fpr_used |= 1 << (r - 72);
            }
        }
    }

    ScanResult { gpr_used, fpr_used, has_calls }
}

/// Compute the total size of all stack slots (spills + locals), respecting alignment.
#[inline]
fn compute_stack_slot_area(func: &MachFunction) -> u32 {
    let mut offset: u32 = 0;
    for slot in &func.stack_slots {
        // Align up to the slot's required alignment.
        offset = align_up(offset, slot.align);
        offset += slot.size;
    }
    offset
}

/// Compute the maximum outgoing stack-argument area needed by any call site.
///
/// ISel emits outgoing stack arguments as `STR{,B,H}` / `STP` with a base of
/// `PReg(SP)` (or `Special(SP)`) and a non-negative immediate offset. This
/// helper scans every instruction and returns the high-water mark of
/// `offset + access_size` across all such stores.
///
/// This runs BEFORE [`eliminate_frame_indices`], so spill stores are still
/// encoded as `FrameIndex`/`StackSlot` operands and do not use `PReg(SP)`
/// bases directly — meaning this scan only matches genuine outgoing-arg
/// stores emitted by ISel, never spill stores.
///
/// The result is rounded up to 16 bytes (AArch64 stack alignment requirement)
/// so the caller can safely subtract it from SP on entry without misaligning
/// the stack.
pub fn compute_max_outgoing_arg_size(func: &MachFunction) -> u32 {
    use AArch64Opcode::*;

    let mut max_end: i64 = 0;
    for inst in &func.insts {
        // Access size in bytes keyed off opcode. StrRI covers both 32- and
        // 64-bit variants depending on source register class; we conservatively
        // assume 8 bytes (worst case). StpRI pair is 16 bytes.
        let (base_idx, offset_idx, access_size) = match inst.opcode {
            StrRI => (1, 2, 8),
            StrbRI => (1, 2, 1),
            StrhRI => (1, 2, 2),
            // StpRI layout: [Rt, Rt2, base, Imm(offset)] — 16 bytes total.
            StpRI => (2, 3, 16),
            _ => continue,
        };

        if inst.operands.len() <= offset_idx {
            continue;
        }

        // Base must be SP (either PReg(SP) or Special(SP)).
        let base_is_sp = match &inst.operands[base_idx] {
            MachOperand::PReg(p) if *p == SP => true,
            MachOperand::Special(SpecialReg::SP) => true,
            _ => false,
        };
        if !base_is_sp {
            continue;
        }

        // Offset must be a non-negative literal immediate (not an IncomingArg
        // marker or FrameIndex; those never reach this point on SP bases).
        if let MachOperand::Imm(off) = &inst.operands[offset_idx] {
            if *off >= 0 {
                let end = *off + access_size as i64;
                if end > max_end {
                    max_end = end;
                }
            }
        }
    }

    // Round up to 16-byte alignment for AArch64 SP requirements.
    align_up(max_end as u32, 16)
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
    compute_frame_layout_inner(func, outgoing_arg_size, enable_red_zone, false)
}

/// Compute the complete frame layout for a function with dynamic allocation.
///
/// Like [`compute_frame_layout`] but explicitly marks the frame as having
/// dynamic stack allocation (alloca), which forces DWARF CFI fallback for
/// unwind encoding since compact unwind cannot describe variable-size frames.
pub fn compute_frame_layout_dynamic(
    func: &MachFunction,
    outgoing_arg_size: u32,
    enable_red_zone: bool,
) -> FrameLayout {
    compute_frame_layout_inner(func, outgoing_arg_size, enable_red_zone, true)
}

fn compute_frame_layout_inner(
    func: &MachFunction,
    outgoing_arg_size: u32,
    enable_red_zone: bool,
    has_dynamic_alloc: bool,
) -> FrameLayout {
    // Single-pass scan replaces three separate iterations.
    let scan = scan_function(func);
    let is_leaf = !scan.has_calls;

    // On Apple AArch64, frame pointer is ALWAYS required.
    let uses_frame_pointer = true;

    // Callee-saved register usage from merged scan.
    let gpr_used = scan.gpr_used;
    let fpr_used = scan.fpr_used;

    // Build callee-saved pairs. FP/LR is always first.
    // Pre-allocate for max 10 pairs (1 FP/LR + 5 GPR + 4 FPR).
    let mut pairs = Vec::with_capacity(10);
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
        has_dynamic_alloc,
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
    // Pre-allocate: worst case is 1 (STP pre-index) + 1 (MOV FP) + N-1 (STP pairs) + 1 (SUB SP)
    let capacity = layout.callee_saved_pairs.len() + 2;
    let mut insts = Vec::with_capacity(capacity);

    if layout.uses_red_zone && layout.sp_adjustment() == 0 && layout.callee_saved_pairs.len() <= 1 {
        // Red zone: minimal prologue for trivial leaf functions.
        // Still save FP/LR if frame pointer is used (Apple requires it).
        if layout.uses_frame_pointer && !layout.callee_saved_pairs.is_empty() {
            // STP X29, X30, [SP, #-16]!
            insts.push(make_stp_pre_index(X29, X30, -16_i64));
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
    // Pre-allocate: worst case is 1 (ADD SP) + N-1 (LDP pairs) + 1 (LDP post-index) + 1 (RET)
    let capacity = layout.callee_saved_pairs.len() + 2;
    let mut insts = Vec::with_capacity(capacity);

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
    // Precompute stack slot offsets from FP (empty if no stack slots).
    // Stack slots are in the spill/local area, which starts at FP - callee_saved_area_size.
    let slot_offsets = compute_slot_offsets(func, layout);

    // Pre-compute the SP adjustment once (used only in the SP-relative path).
    let sp_adj = layout.sp_adjustment() as i32;
    let uses_fp = layout.uses_frame_pointer;

    // IncomingArg offsets resolve to `[FP, #callee_saved_area_size + offset]`.
    // The callee's FP points at the saved FP/LR pair; callee-saves occupy
    // positive offsets [0..CSA_size); incoming stack args sit directly above
    // the callee-saved area at the caller's SP = FP + CSA_size.
    let csa_size = layout.callee_saved_area_size as i64;

    // Walk all instructions and rewrite frame-related operands.
    for inst in &mut func.insts {
        for operand in &mut inst.operands {
            match operand {
                MachOperand::FrameIndex(fi) => {
                    let slot_idx = fi.0 as usize;
                    if slot_idx < slot_offsets.len() {
                        let offset = slot_offsets[slot_idx];
                        if uses_fp {
                            // FP-relative: MemOp { base: X29, offset }
                            *operand = MachOperand::MemOp {
                                base: X29,
                                offset: offset as i64,
                            };
                        } else {
                            // SP-relative: offset from current SP
                            // SP-relative offset = FP_offset + callee_saved_area + sp_adjustment
                            let sp_offset = offset + sp_adj;
                            *operand = MachOperand::MemOp {
                                base: SP, // SP = PReg(31)
                                offset: sp_offset as i64,
                            };
                        }
                    }
                }
                MachOperand::IncomingArg(arg_offset) => {
                    let fp_offset = csa_size + *arg_offset;
                    *operand = MachOperand::Imm(fp_offset);
                }
                _ => {}
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
            current_offset &= !(align - 1);
        }
        offsets.push(current_offset);
    }

    offsets
}

// ---------------------------------------------------------------------------
// Frame index elimination — enhanced pass
// ---------------------------------------------------------------------------

/// AArch64 LDR/STR unsigned immediate offset upper bound (conservative).
/// The real limit depends on access size (4095 * scale), but we use the
/// unscaled upper bound for simplicity.
const AARCH64_MAX_IMM_OFFSET: i64 = 4095;

/// AArch64 LDR/STR signed immediate offset lower bound (LDUR/STUR range).
const AARCH64_MIN_IMM_OFFSET: i64 = -256;

/// AArch64 scratch register for offset materialization.
/// X16 (IP0) is reserved by the ABI as an intra-procedure-call scratch register
/// and is safe to clobber between instructions.
use llvm2_ir::regs::X16;

/// Check whether an offset exceeds the AArch64 immediate encoding range
/// for load/store instructions.
///
/// Conservative range: -256 <= offset <= 4095.
/// Offsets outside this range require materialization in a scratch register.
#[inline]
pub fn is_large_offset(offset: i64) -> bool {
    !(AARCH64_MIN_IMM_OFFSET..=AARCH64_MAX_IMM_OFFSET).contains(&offset)
}

/// Statistics from a frame index elimination pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EliminationStats {
    /// Number of FrameIndex/StackSlot operands replaced.
    pub eliminated_count: u32,
    /// Number of large offsets that required scratch register materialization.
    pub large_offset_count: u32,
}

/// Enhanced frame index elimination pass.
///
/// Resolves all `FrameIndex` and `StackSlot` operands in a function to
/// concrete memory operands (FP+offset or SP+offset), handling large
/// offsets that exceed AArch64 immediate encoding range.
///
/// # Large offset handling
///
/// When an offset exceeds the encodable immediate range (-256..4095),
/// the eliminator inserts instructions to materialize the offset in X16 (IP0):
///
/// ```asm
/// movz  x16, #offset_lo             ; lower 16 bits
/// movk  x16, #offset_hi, lsl #16    ; upper 16 bits (if needed)
/// add   x16, fp, x16                ; compute absolute address
/// ; then use [x16, #0] as the memory operand
/// ```
pub struct FrameIndexEliminator<'a> {
    /// The computed frame layout.
    layout: &'a FrameLayout,
    /// Precomputed FP-relative offset for each stack slot (indexed by slot number).
    slot_offsets: Vec<i32>,
}

impl<'a> FrameIndexEliminator<'a> {
    /// Create a new eliminator for the given layout and function.
    pub fn new(layout: &'a FrameLayout, func: &MachFunction) -> Self {
        let slot_offsets = compute_slot_offsets(func, layout);
        Self { layout, slot_offsets }
    }

    /// Resolve a stack slot index to (base_register, offset).
    ///
    /// Spill/local slots use FP-relative addressing when the frame pointer
    /// is available (always on Apple AArch64). The returned offset is signed.
    pub fn resolve_slot_operand(&self, slot_idx: usize) -> (PReg, i64) {
        if slot_idx < self.slot_offsets.len() {
            let fp_offset = self.slot_offsets[slot_idx] as i64;
            if self.layout.uses_frame_pointer {
                (X29, fp_offset)
            } else {
                // SP-relative: FP offset + callee_saved_area + sp_adjustment
                let sp_offset = fp_offset + self.layout.sp_adjustment() as i64;
                (SP, sp_offset)
            }
        } else {
            // Out-of-range slot index — treat as outgoing arg area (SP-relative).
            // This shouldn't happen in well-formed IR but we handle it defensively.
            (SP, 0)
        }
    }

    /// Run the frame index elimination pass over the function.
    ///
    /// Replaces all `FrameIndex` and `StackSlot` operands with concrete
    /// `MemOp` operands. For large offsets, inserts scratch register
    /// materialization instructions.
    ///
    /// Returns statistics about the elimination.
    pub fn run(&self, func: &mut MachFunction) -> EliminationStats {
        let mut stats = EliminationStats::default();

        // Early exit: if no stack slots, no frame indices can exist.
        if func.stack_slots.is_empty() && !self.has_frame_operands(func) {
            return stats;
        }

        // Process each block. We must rebuild block instruction lists when
        // large offsets require inserting materialization instructions.
        let num_blocks = func.blocks.len();
        for block_idx in 0..num_blocks {
            let block_insts = std::mem::take(&mut func.blocks[block_idx].insts);
            let mut new_insts = Vec::with_capacity(block_insts.len());
            let mut block_modified = false;

            for &inst_id in &block_insts {
                // Check if this instruction has any frame-related operands.
                let mut has_frame_op = false;
                let mut needs_large_offset = false;

                for operand in &func.insts[inst_id.0 as usize].operands {
                    match operand {
                        MachOperand::FrameIndex(fi) => {
                            has_frame_op = true;
                            let slot_idx = fi.0 as usize;
                            let (_, offset) = self.resolve_slot_operand(slot_idx);
                            if is_large_offset(offset) {
                                needs_large_offset = true;
                            }
                        }
                        MachOperand::StackSlot(ss) => {
                            has_frame_op = true;
                            let slot_idx = ss.0 as usize;
                            let (_, offset) = self.resolve_slot_operand(slot_idx);
                            if is_large_offset(offset) {
                                needs_large_offset = true;
                            }
                        }
                        _ => {}
                    }
                }

                if !has_frame_op {
                    new_insts.push(inst_id);
                    continue;
                }

                if needs_large_offset {
                    // Insert materialization instructions before the original.
                    // We need to find the frame operand, compute its offset,
                    // materialize it in X16, then rewrite the operand.
                    let mat_insts = self.materialize_and_rewrite(func, inst_id, &mut stats);
                    for mat_id in mat_insts {
                        new_insts.push(mat_id);
                    }
                    block_modified = true;
                } else {
                    // Small offset — just rewrite operands in place.
                    self.rewrite_operands_small(func, inst_id, &mut stats);
                    new_insts.push(inst_id);
                }
            }

            if block_modified || new_insts.len() != block_insts.len() {
                func.blocks[block_idx].insts = new_insts;
            } else {
                // Restore the original inst list (operands were modified in place).
                func.blocks[block_idx].insts = new_insts;
            }
        }

        stats
    }

    /// Check if any instruction has FrameIndex or StackSlot operands.
    fn has_frame_operands(&self, func: &MachFunction) -> bool {
        func.insts.iter().any(|inst| {
            inst.operands.iter().any(|op| {
                matches!(op, MachOperand::FrameIndex(_) | MachOperand::StackSlot(_))
            })
        })
    }

    /// Rewrite small-offset frame operands in place (no new instructions needed).
    fn rewrite_operands_small(
        &self,
        func: &mut MachFunction,
        inst_id: llvm2_ir::types::InstId,
        stats: &mut EliminationStats,
    ) {
        let inst = &mut func.insts[inst_id.0 as usize];
        for operand in &mut inst.operands {
            match operand {
                MachOperand::FrameIndex(fi) => {
                    let slot_idx = fi.0 as usize;
                    let (base, offset) = self.resolve_slot_operand(slot_idx);
                    *operand = MachOperand::MemOp { base, offset };
                    stats.eliminated_count += 1;
                }
                MachOperand::StackSlot(ss) => {
                    let slot_idx = ss.0 as usize;
                    let (base, offset) = self.resolve_slot_operand(slot_idx);
                    *operand = MachOperand::MemOp { base, offset };
                    stats.eliminated_count += 1;
                }
                _ => {}
            }
        }
    }

    /// Handle a large-offset frame index: materialize offset in X16, then
    /// rewrite the operand. Returns the list of InstIds that replace the
    /// original instruction (materialization + original with rewritten operand).
    fn materialize_and_rewrite(
        &self,
        func: &mut MachFunction,
        inst_id: llvm2_ir::types::InstId,
        stats: &mut EliminationStats,
    ) -> Vec<llvm2_ir::types::InstId> {
        let mut result = Vec::with_capacity(4);

        // Find the first frame operand to determine offset and base.
        // (In practice there's usually only one frame operand per instruction.)
        let (base_reg, offset) = {
            let inst = &func.insts[inst_id.0 as usize];
            let mut found = None;
            for operand in &inst.operands {
                match operand {
                    MachOperand::FrameIndex(fi) => {
                        found = Some(self.resolve_slot_operand(fi.0 as usize));
                        break;
                    }
                    MachOperand::StackSlot(ss) => {
                        found = Some(self.resolve_slot_operand(ss.0 as usize));
                        break;
                    }
                    _ => {}
                }
            }
            found.unwrap_or((SP, 0))
        };

        // Materialize the offset in X16.
        // For negative offsets, we use MOVN + optional MOVK.
        // For positive offsets, we use MOVZ + optional MOVK.
        let abs_offset = if offset < 0 { (-offset) as u64 } else { offset as u64 };

        if offset >= 0 {
            let lo16 = (abs_offset & 0xFFFF) as i64;
            // MOVZ X16, #lo16
            let movz = MachInst::new(
                AArch64Opcode::Movz,
                vec![MachOperand::PReg(X16), MachOperand::Imm(lo16)],
            );
            let movz_id = func.push_inst(movz);
            result.push(movz_id);

            if abs_offset > 0xFFFF {
                let hi16 = ((abs_offset >> 16) & 0xFFFF) as i64;
                // MOVK X16, #hi16, LSL #16
                // We encode the shift amount as a second immediate.
                let movk = MachInst::new(
                    AArch64Opcode::Movk,
                    vec![
                        MachOperand::PReg(X16),
                        MachOperand::Imm(hi16),
                        MachOperand::Imm(16), // shift amount
                    ],
                );
                let movk_id = func.push_inst(movk);
                result.push(movk_id);
            }
        } else {
            // Negative offset: MOVN X16, #(~lo16) then adjust.
            // Use MOVZ with the absolute value and then SUB instead of MOVN
            // for simplicity and clarity.
            let lo16 = (abs_offset & 0xFFFF) as i64;
            let movz = MachInst::new(
                AArch64Opcode::Movz,
                vec![MachOperand::PReg(X16), MachOperand::Imm(lo16)],
            );
            let movz_id = func.push_inst(movz);
            result.push(movz_id);

            if abs_offset > 0xFFFF {
                let hi16 = ((abs_offset >> 16) & 0xFFFF) as i64;
                let movk = MachInst::new(
                    AArch64Opcode::Movk,
                    vec![
                        MachOperand::PReg(X16),
                        MachOperand::Imm(hi16),
                        MachOperand::Imm(16),
                    ],
                );
                let movk_id = func.push_inst(movk);
                result.push(movk_id);
            }

            // SUB X16, base_reg, X16  (base - abs_offset = base + offset)
            let sub = MachInst::new(
                AArch64Opcode::SubRR,
                vec![
                    MachOperand::PReg(X16),
                    if base_reg == SP {
                        MachOperand::Special(SpecialReg::SP)
                    } else {
                        MachOperand::PReg(base_reg)
                    },
                    MachOperand::PReg(X16),
                ],
            );
            let sub_id = func.push_inst(sub);
            result.push(sub_id);

            // Rewrite operands to use [X16, #0].
            let inst = &mut func.insts[inst_id.0 as usize];
            for operand in &mut inst.operands {
                match operand {
                    MachOperand::FrameIndex(_) | MachOperand::StackSlot(_) => {
                        *operand = MachOperand::MemOp { base: X16, offset: 0 };
                        stats.eliminated_count += 1;
                        stats.large_offset_count += 1;
                    }
                    _ => {}
                }
            }
            result.push(inst_id);
            return result;
        }

        // Positive offset: ADD X16, base_reg, X16
        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::PReg(X16),
                if base_reg == SP {
                    MachOperand::Special(SpecialReg::SP)
                } else {
                    MachOperand::PReg(base_reg)
                },
                MachOperand::PReg(X16),
            ],
        );
        let add_id = func.push_inst(add);
        result.push(add_id);

        // Rewrite frame operands to [X16, #0].
        let inst = &mut func.insts[inst_id.0 as usize];
        for operand in &mut inst.operands {
            match operand {
                MachOperand::FrameIndex(_) | MachOperand::StackSlot(_) => {
                    *operand = MachOperand::MemOp { base: X16, offset: 0 };
                    stats.eliminated_count += 1;
                    stats.large_offset_count += 1;
                }
                _ => {}
            }
        }
        result.push(inst_id);

        result
    }
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

impl CompactUnwindEncoding {
    /// Returns true if this encoding requires DWARF CFI fallback.
    ///
    /// When `UNWIND_ARM64_MODE_DWARF` is set, the linker expects a
    /// corresponding FDE in `__eh_frame` to describe how to unwind this
    /// function. The compact unwind entry still exists but its encoding
    /// tells the unwinder to look up the DWARF info instead.
    pub fn needs_dwarf_fallback(&self) -> bool {
        (self.encoding & 0x0F00_0000) == UNWIND_ARM64_MODE_DWARF
    }

    /// Returns the mode bits (top nibble of the encoding).
    pub fn mode(&self) -> u32 {
        self.encoding & 0x0F00_0000
    }

    /// Returns the register-pair flags (low 12 bits for FRAME mode).
    pub fn register_pair_flags(&self) -> u32 {
        self.encoding & 0x0000_0FFF
    }
}

/// Encode the Darwin compact unwind for a frame layout.
///
/// For standard FP-based frames, this produces `UNWIND_ARM64_MODE_FRAME`
/// with register-pair flags indicating which callee-saved registers are saved.
///
/// Falls back to `UNWIND_ARM64_MODE_DWARF` when the frame cannot be
/// described by compact unwind:
/// - Frameless functions (no FP — not possible on Apple AArch64 but handled)
/// - Variable-size frames (dynamic alloca — SP moves by runtime amount)
///
/// Reference: AArch64AsmBackend.cpp `generateCompactUnwindEncoding()` (line 576)
pub fn encode_compact_unwind(layout: &FrameLayout) -> CompactUnwindEncoding {
    if !layout.uses_frame_pointer {
        // Frameless functions use DWARF fallback for now.
        return CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
    }

    if layout.has_dynamic_alloc {
        // Variable-size frames cannot be described by compact unwind.
        // The unwinder needs full DWARF CFI to handle the dynamic SP offsets.
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
                (r1, r2) => {
                    // Unrecognized GPR callee-saved pair — compact unwind cannot
                    // encode it. Fall back to DWARF mode so the unwinder uses
                    // full CFI instead of producing wrong unwind info.
                    eprintln!(
                        "WARNING: unrecognized callee-saved GPR pair ({}, {}) in \
                         compact unwind encoding, falling back to DWARF mode",
                        r1, r2
                    );
                    return CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
                }
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
                (r1, r2) => {
                    // Unrecognized FPR callee-saved pair — compact unwind cannot
                    // encode it. Fall back to DWARF mode so the unwinder uses
                    // full CFI instead of producing wrong unwind info.
                    eprintln!(
                        "WARNING: unrecognized callee-saved FPR pair ({}, {}) in \
                         compact unwind encoding, falling back to DWARF mode",
                        r1, r2
                    );
                    return CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
                }
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
///
/// # Performance notes
///
/// - Entry block insts are moved (via `mem::take`) rather than cloned.
/// - Epilogue instructions are generated fresh per return site rather than
///   cloning a template, avoiding heap allocation for operand Vecs.
/// - Block inst lists are only rebuilt for blocks that contain returns,
///   skipping non-terminating blocks entirely.
pub fn insert_prologue_epilogue(func: &mut MachFunction, layout: &FrameLayout) {
    let prologue = emit_prologue(layout);

    // Insert prologue at the start of the entry block.
    // Use mem::take to move the old insts without cloning.
    let entry = func.entry;
    let old_entry_insts = std::mem::take(&mut func.blocks[entry.0 as usize].insts);

    let mut new_entry_insts = Vec::with_capacity(prologue.len() + old_entry_insts.len());
    for prologue_inst in prologue {
        let id = func.push_inst(prologue_inst);
        new_entry_insts.push(id);
    }
    new_entry_insts.extend(old_entry_insts);
    func.blocks[entry.0 as usize].insts = new_entry_insts;

    // For each block, find return instructions and insert epilogue before them.
    // First pass: identify which blocks contain returns to avoid unnecessary work.
    let num_blocks = func.blocks.len();
    for block_idx in 0..num_blocks {
        // Check if this block has any return instructions.
        let has_return = func.blocks[block_idx].insts.iter().any(|&inst_id| {
            func.insts[inst_id.0 as usize].is_return()
        });

        if !has_return {
            continue;
        }

        // Move the old insts out to avoid borrow conflict.
        let block_insts = std::mem::take(&mut func.blocks[block_idx].insts);
        let mut new_insts = Vec::with_capacity(block_insts.len() + 8);

        for &inst_id in &block_insts {
            if func.insts[inst_id.0 as usize].is_return() {
                // Generate epilogue instructions fresh (avoids cloning a template).
                // The epilogue includes its own RET, so we drop the original.
                let epilogue = emit_epilogue(layout);
                for epi_inst in epilogue {
                    let id = func.push_inst(epi_inst);
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
        let mut regs: Vec<PReg> = (19..=28).map(PReg::new).collect();
        let fprs: Vec<PReg> = (72..=79).map(PReg::new).collect(); // V8-V15
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
        let mut regs: Vec<PReg> = (19..=28).map(PReg::new).collect();
        let fprs: Vec<PReg> = (72..=79).map(PReg::new).collect(); // V8-V15
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
        let scan = scan_function(&func);

        assert!(scan.gpr_used & (1 << 0) != 0); // X19
        assert!(scan.gpr_used & (1 << 1) == 0); // X20 not used
        assert!(scan.gpr_used & (1 << 3) != 0); // X22
        assert!(scan.gpr_used & (1 << 9) != 0); // X28
    }

    #[test]
    fn test_scan_callee_saved_fprs() {
        let func = make_func_with_callee_saved_fprs(&[V8, V11, V15]);
        let scan = scan_function(&func);

        assert!(scan.fpr_used & (1 << 0) != 0); // V8
        assert!(scan.fpr_used & (1 << 3) != 0); // V11
        assert!(scan.fpr_used & (1 << 7) != 0); // V15
        assert!(scan.fpr_used & (1 << 1) == 0); // V9 not used
    }

    // --- has_dynamic_alloc field tests ---

    #[test]
    fn test_layout_default_no_dynamic_alloc() {
        let func = make_func("no_alloca", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout(&func, 0, false);
        assert!(!layout.has_dynamic_alloc);
    }

    #[test]
    fn test_layout_dynamic_alloc_flag() {
        let func = make_func("with_alloca", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout_dynamic(&func, 0, false);
        assert!(layout.has_dynamic_alloc);
    }

    #[test]
    fn test_compact_unwind_dynamic_alloc_dwarf_fallback() {
        let func = make_func("alloca_func", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        let layout = compute_frame_layout_dynamic(&func, 0, false);
        let cu = encode_compact_unwind(&layout);

        assert_eq!(cu.encoding, UNWIND_ARM64_MODE_DWARF);
        assert!(cu.needs_dwarf_fallback());
    }

    // --- CompactUnwindEncoding method tests ---

    #[test]
    fn test_encoding_needs_dwarf_fallback() {
        let frame = CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_FRAME };
        assert!(!frame.needs_dwarf_fallback());

        let dwarf = CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
        assert!(dwarf.needs_dwarf_fallback());

        let frameless = CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_FRAMELESS };
        assert!(!frameless.needs_dwarf_fallback());
    }

    #[test]
    fn test_encoding_mode() {
        let frame = CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_FRAME | UNWIND_ARM64_FRAME_X19_X20_PAIR };
        assert_eq!(frame.mode(), UNWIND_ARM64_MODE_FRAME);

        let dwarf = CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
        assert_eq!(dwarf.mode(), UNWIND_ARM64_MODE_DWARF);
    }

    #[test]
    fn test_encoding_register_pair_flags() {
        let encoding = CompactUnwindEncoding {
            encoding: UNWIND_ARM64_MODE_FRAME
                | UNWIND_ARM64_FRAME_X19_X20_PAIR
                | UNWIND_ARM64_FRAME_D8_D9_PAIR,
        };
        let flags = encoding.register_pair_flags();
        assert_ne!(flags & UNWIND_ARM64_FRAME_X19_X20_PAIR, 0);
        assert_ne!(flags & UNWIND_ARM64_FRAME_D8_D9_PAIR, 0);
        assert_eq!(flags & UNWIND_ARM64_FRAME_X21_X22_PAIR, 0);
    }

    #[test]
    fn test_encoding_zero_register_pair_flags_for_dwarf() {
        let encoding = CompactUnwindEncoding { encoding: UNWIND_ARM64_MODE_DWARF };
        assert_eq!(encoding.register_pair_flags(), 0);
    }

    // --- Unrecognized register pair fallback tests (#97) ---

    #[test]
    fn test_compact_unwind_unrecognized_gpr_pair_falls_back_to_dwarf() {
        // Create a frame layout with a GPR pair that isn't a standard AArch64
        // callee-saved pair. This tests that encode_compact_unwind falls back
        // to DWARF mode instead of silently dropping the pair.
        let layout = FrameLayout {
            callee_saved_pairs: vec![
                CalleeSavedPair { reg1: X29, reg2: X30, fp_offset: 0, is_fpr: false },
                // X0/X1 is not a valid callee-saved pair
                CalleeSavedPair {
                    reg1: PReg::new(0),  // X0
                    reg2: PReg::new(1),  // X1
                    fp_offset: -16,
                    is_fpr: false,
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
        };

        let cu = encode_compact_unwind(&layout);
        assert_eq!(cu.encoding, UNWIND_ARM64_MODE_DWARF,
            "Unrecognized GPR pair must trigger DWARF fallback, not be silently dropped");
        assert!(cu.needs_dwarf_fallback());
    }

    #[test]
    fn test_compact_unwind_unrecognized_fpr_pair_falls_back_to_dwarf() {
        // Create a frame layout with an FPR pair that isn't a standard AArch64
        // callee-saved pair. Tests fallback for unrecognized FPR pairs.
        let layout = FrameLayout {
            callee_saved_pairs: vec![
                CalleeSavedPair { reg1: X29, reg2: X30, fp_offset: 0, is_fpr: false },
                // V0/V1 (encoding 64,65) is not a callee-saved FPR pair
                CalleeSavedPair {
                    reg1: PReg::new(64), // V0
                    reg2: PReg::new(65), // V1
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
        };

        let cu = encode_compact_unwind(&layout);
        assert_eq!(cu.encoding, UNWIND_ARM64_MODE_DWARF,
            "Unrecognized FPR pair must trigger DWARF fallback, not be silently dropped");
        assert!(cu.needs_dwarf_fallback());
    }

    #[test]
    fn test_compact_unwind_valid_pair_after_unrecognized_not_reached() {
        // If the first non-FP/LR pair is unrecognized, we should fall back to
        // DWARF immediately. The valid X19/X20 pair after it should not be
        // reached — verifying early return behavior.
        let layout = FrameLayout {
            callee_saved_pairs: vec![
                CalleeSavedPair { reg1: X29, reg2: X30, fp_offset: 0, is_fpr: false },
                // Bogus GPR pair
                CalleeSavedPair {
                    reg1: PReg::new(5),  // X5
                    reg2: PReg::new(6),  // X6
                    fp_offset: -16,
                    is_fpr: false,
                },
                // Valid pair that should never be reached
                CalleeSavedPair { reg1: X19, reg2: X20, fp_offset: -32, is_fpr: false },
            ],
            callee_saved_area_size: 48,
            spill_area_size: 0,
            local_area_size: 0,
            outgoing_arg_area_size: 0,
            total_frame_size: 48,
            uses_frame_pointer: true,
            is_leaf: true,
            uses_red_zone: false,
            fp_to_spill_offset: -48,
            has_dynamic_alloc: false,
        };

        let cu = encode_compact_unwind(&layout);
        assert_eq!(cu.encoding, UNWIND_ARM64_MODE_DWARF,
            "Early DWARF fallback on unrecognized pair");
    }

    // =======================================================================
    // FrameIndexEliminator tests
    // =======================================================================

    #[test]
    fn test_fie_simple_frame() {
        // Simple function with a few stack slots and FrameIndex operands.
        let mut func = make_func("fie_simple", vec![
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![
                    MachOperand::PReg(PReg::new(0)),
                    MachOperand::FrameIndex(FrameIdx(0)),
                ],
            ),
            MachInst::new(
                AArch64Opcode::StrRI,
                vec![
                    MachOperand::PReg(PReg::new(1)),
                    MachOperand::FrameIndex(FrameIdx(1)),
                ],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));
        func.alloc_stack_slot(StackSlot::new(8, 8));

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);
        let stats = fie.run(&mut func);

        // Both FrameIndex operands should be eliminated.
        assert_eq!(stats.eliminated_count, 2);
        assert_eq!(stats.large_offset_count, 0);

        // Verify operands are now MemOp with FP (X29) as base.
        match &func.insts[0].operands[1] {
            MachOperand::MemOp { base, offset } => {
                assert_eq!(*base, X29);
                assert!(*offset < 0, "FP offset should be negative, got {}", offset);
            }
            other => panic!("Expected MemOp, got {:?}", other),
        }
        match &func.insts[1].operands[1] {
            MachOperand::MemOp { base, offset } => {
                assert_eq!(*base, X29);
                assert!(*offset < 0, "FP offset should be negative, got {}", offset);
            }
            other => panic!("Expected MemOp, got {:?}", other),
        }

        // Offsets should be distinct (different slots).
        let off0 = match &func.insts[0].operands[1] {
            MachOperand::MemOp { offset, .. } => *offset,
            _ => unreachable!(),
        };
        let off1 = match &func.insts[1].operands[1] {
            MachOperand::MemOp { offset, .. } => *offset,
            _ => unreachable!(),
        };
        assert_ne!(off0, off1, "Different slots must have different offsets");
    }

    #[test]
    fn test_fie_large_frame() {
        // Function with many stack slots so that offsets exceed 4096.
        // Create a function with a very large stack frame.
        let mut func = make_func("fie_large", vec![
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![
                    MachOperand::PReg(PReg::new(0)),
                    MachOperand::FrameIndex(FrameIdx(0)),
                ],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        // Allocate one very large slot to push offset beyond 4096.
        func.alloc_stack_slot(StackSlot::new(8192, 16));

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);
        let stats = fie.run(&mut func);

        assert_eq!(stats.eliminated_count, 1);
        assert_eq!(stats.large_offset_count, 1);

        // The block should now have more instructions (materialization + original + ret).
        let block_insts = &func.blocks[0].insts;
        assert!(
            block_insts.len() > 2,
            "Expected materialization instructions, got {} insts",
            block_insts.len()
        );

        // First instruction(s) should be MOVZ/ADD for offset materialization.
        let first_inst = func.inst(block_insts[0]);
        assert_eq!(
            first_inst.opcode,
            AArch64Opcode::Movz,
            "Expected MOVZ for large offset materialization"
        );

        // The rewritten instruction should use X16 as base with offset 0.
        // Find the LdrRI instruction in the block.
        let ldr_inst_id = block_insts.iter().find(|&&id| {
            func.inst(id).opcode == AArch64Opcode::LdrRI
        });
        assert!(ldr_inst_id.is_some(), "LdrRI should still be in the block");
        let ldr_inst = func.inst(*ldr_inst_id.unwrap());
        match &ldr_inst.operands[1] {
            MachOperand::MemOp { base, offset } => {
                assert_eq!(*base, X16, "Large offset should use X16 scratch register");
                assert_eq!(*offset, 0, "After materialization, offset should be 0");
            }
            other => panic!("Expected MemOp with X16, got {:?}", other),
        }
    }

    #[test]
    fn test_fie_mixed_slots() {
        // Function with slots of different sizes and alignments.
        let mut func = make_func("fie_mixed", vec![
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
        func.alloc_stack_slot(StackSlot::new(16, 16)); // 16-byte aligned local
        func.alloc_stack_slot(StackSlot::new(8, 8));   // 8-byte spill
        func.alloc_stack_slot(StackSlot::new(1, 1));   // 1-byte local

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);
        let stats = fie.run(&mut func);

        assert_eq!(stats.eliminated_count, 3);

        // All operands should be MemOp now.
        let offsets: Vec<i64> = (0..3)
            .map(|i| match &func.insts[i].operands[1] {
                MachOperand::MemOp { base, offset } => {
                    assert_eq!(*base, X29);
                    *offset
                }
                other => panic!("Slot {} not eliminated: {:?}", i, other),
            })
            .collect();

        // Each subsequent slot should be at a lower (more negative) offset.
        assert!(offsets[0] > offsets[1], "slot 0 > slot 1: {} > {}", offsets[0], offsets[1]);
        assert!(offsets[1] > offsets[2], "slot 1 > slot 2: {} > {}", offsets[1], offsets[2]);
    }

    #[test]
    fn test_fie_outgoing_arg_area() {
        // Non-leaf function with outgoing arg area and stack slots.
        let mut func = make_func("fie_args", vec![
            MachInst::new(AArch64Opcode::Bl, vec![MachOperand::Imm(0)]),
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(0)), MachOperand::FrameIndex(FrameIdx(0))],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));

        let layout = compute_frame_layout(&func, 32, false);
        assert!(!layout.is_leaf);
        assert_eq!(layout.outgoing_arg_area_size, 32);

        let fie = FrameIndexEliminator::new(&layout, &func);
        let stats = fie.run(&mut func);

        assert_eq!(stats.eliminated_count, 1);
        // Spill slot should use FP-relative addressing.
        match &func.insts[1].operands[1] {
            MachOperand::MemOp { base, .. } => {
                assert_eq!(*base, X29, "Spill slot should use FP-relative addressing");
            }
            other => panic!("Expected MemOp, got {:?}", other),
        }
    }

    #[test]
    fn test_fie_fp_vs_sp_relative() {
        // Test that when uses_frame_pointer is true, we get FP-relative addressing.
        let mut func = make_func("fie_fp", vec![
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(0)), MachOperand::FrameIndex(FrameIdx(0))],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));

        // Standard layout (uses_frame_pointer = true on Apple AArch64).
        let layout = compute_frame_layout(&func, 0, false);
        assert!(layout.uses_frame_pointer);

        let fie = FrameIndexEliminator::new(&layout, &func);
        let (base, _offset) = fie.resolve_slot_operand(0);
        assert_eq!(base, X29, "With FP, should use FP-relative");

        // Test SP-relative by creating a layout with uses_frame_pointer=false.
        let sp_layout = FrameLayout {
            uses_frame_pointer: false,
            ..layout.clone()
        };
        let fie_sp = FrameIndexEliminator::new(&sp_layout, &func);
        let (base_sp, _offset_sp) = fie_sp.resolve_slot_operand(0);
        assert_eq!(base_sp, SP, "Without FP, should use SP-relative");
    }

    #[test]
    fn test_fie_stats_tracking() {
        // Verify stats are correctly tracked.
        let mut func = make_func("fie_stats", vec![
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(0)), MachOperand::FrameIndex(FrameIdx(0))],
            ),
            MachInst::new(
                AArch64Opcode::StrRI,
                vec![MachOperand::PReg(PReg::new(1)), MachOperand::FrameIndex(FrameIdx(0))],
            ),
            MachInst::new(
                AArch64Opcode::LdrRI,
                vec![MachOperand::PReg(PReg::new(2)), MachOperand::FrameIndex(FrameIdx(1))],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));
        func.alloc_stack_slot(StackSlot::new(4, 4));

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);
        let stats = fie.run(&mut func);

        assert_eq!(stats.eliminated_count, 3, "Should eliminate 3 frame indices");
        assert_eq!(stats.large_offset_count, 0, "No large offsets expected");
    }

    #[test]
    fn test_fie_no_frame_indices() {
        // Function with no frame index operands — should be a no-op.
        let mut func = make_func("fie_noop", vec![
            MachInst::new(
                AArch64Opcode::AddRR,
                vec![
                    MachOperand::PReg(PReg::new(0)),
                    MachOperand::PReg(PReg::new(1)),
                    MachOperand::PReg(PReg::new(2)),
                ],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);
        let stats = fie.run(&mut func);

        assert_eq!(stats.eliminated_count, 0);
        assert_eq!(stats.large_offset_count, 0);
        assert_eq!(func.blocks[0].insts.len(), 2, "Block should be unchanged");
    }

    #[test]
    fn test_is_large_offset() {
        // Boundary cases for AArch64 immediate range.
        assert!(!is_large_offset(0));
        assert!(!is_large_offset(100));
        assert!(!is_large_offset(4095));
        assert!(is_large_offset(4096));
        assert!(is_large_offset(10000));
        assert!(!is_large_offset(-1));
        assert!(!is_large_offset(-256));
        assert!(is_large_offset(-257));
        assert!(is_large_offset(-1000));
        assert!(is_large_offset(i64::MAX));
        assert!(is_large_offset(i64::MIN));
    }

    #[test]
    fn test_fie_resolve_slot_operand() {
        // Directly test resolve_slot_operand for correctness.
        let mut func = make_func("fie_resolve", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        func.alloc_stack_slot(StackSlot::new(8, 8));
        func.alloc_stack_slot(StackSlot::new(16, 16));

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);

        let (base0, off0) = fie.resolve_slot_operand(0);
        let (base1, off1) = fie.resolve_slot_operand(1);

        assert_eq!(base0, X29);
        assert_eq!(base1, X29);
        // Both offsets should be negative (below FP).
        assert!(off0 < 0, "slot 0 offset should be negative: {}", off0);
        assert!(off1 < 0, "slot 1 offset should be negative: {}", off1);
        // Slot 1 should be at a lower offset than slot 0.
        assert!(off0 > off1, "slot 0 ({}) should be above slot 1 ({})", off0, off1);
    }

    #[test]
    fn test_fie_out_of_range_slot() {
        // Test defensive handling of out-of-range slot index.
        let func = make_func("fie_oob", vec![
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);

        let layout = compute_frame_layout(&func, 0, false);
        let fie = FrameIndexEliminator::new(&layout, &func);

        // No slots allocated, so index 0 is out of range.
        let (base, offset) = fie.resolve_slot_operand(0);
        assert_eq!(base, SP, "Out-of-range slot should default to SP");
        assert_eq!(offset, 0, "Out-of-range slot should default to offset 0");
    }

    #[test]
    fn test_fie_elimination_stats_default() {
        let stats = EliminationStats::default();
        assert_eq!(stats.eliminated_count, 0);
        assert_eq!(stats.large_offset_count, 0);
    }
}

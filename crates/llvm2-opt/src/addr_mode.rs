// llvm2-opt - AArch64 Address Mode Formation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! AArch64 address mode formation pass.
//!
//! A late machine-level optimization that folds ADD instructions into
//! the addressing mode of subsequent LDR/STR instructions, exploiting
//! AArch64's rich addressing modes.
//!
//! # Patterns
//!
//! ## Base + Immediate (form_base_plus_imm)
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | `ADD Xd, Xn, #imm` + `LDR Xt, [Xd, #0]` | `LDR Xt, [Xn, #imm]` |
//! | `ADD Xd, Xn, #imm` + `STR Xt, [Xd, #0]` | `STR Xt, [Xn, #imm]` |
//! | `ADD Xd, Xn, #imm1` + `LDR Xt, [Xd, #imm2]` | `LDR Xt, [Xn, #(imm1+imm2)]` |
//! | `ADD Xd, Xn, #imm1` + `STR Xt, [Xd, #imm2]` | `STR Xt, [Xn, #(imm1+imm2)]` |
//!
//! ## Base + Register (form_base_plus_reg)
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | `ADD Xd, Xn, Xm` + `LDR Xt, [Xd, #0]` | `LDR Xt, [Xn, Xm]` (LdrRO) |
//! | `ADD Xd, Xn, Xm` + `STR Xt, [Xd, #0]` | `STR Xt, [Xn, Xm]` (StrRO) |
//!
//! ## Pre-Index / Post-Index (future)
//!
//! Pre-index (`LDR Xt, [Xn, #imm]!`) and post-index (`LDR Xt, [Xn], #imm`)
//! patterns are detected but not yet transformed because the IR does not
//! have dedicated pre/post-index opcodes for single loads/stores. See
//! `form_pre_index()` and `form_post_index()` for pattern detection logic.
//!
//! # Safety Constraints
//!
//! - The ADD result must have exactly one use (the LDR/STR base operand).
//!   If the ADD result is used elsewhere, folding would break those uses.
//! - For base+imm: the combined offset must fit in the AArch64 unsigned
//!   immediate range, validated per access size via `is_encodable_offset()`.
//! - For base+reg: the LDR/STR offset must be exactly 0 (folding a non-zero
//!   offset into a register-offset form is not valid).
//! - Proof annotations on the LDR/STR are preserved (unchanged).
//! - The ADD instruction is deleted after folding.
//!
//! # Offset Encoding
//!
//! AArch64 LDR/STR unsigned immediate offset is 12 bits, scaled by the
//! access size:
//!
//! | Access Size | Scale | Max Offset | Range |
//! |-------------|-------|------------|-------|
//! | 1 byte      | 1     | 4095       | 0..=4095 |
//! | 2 bytes     | 2     | 8190       | 0..=8190 (2-aligned) |
//! | 4 bytes     | 4     | 16380      | 0..=16380 (4-aligned) |
//! | 8 bytes     | 8     | 32760      | 0..=32760 (8-aligned) |

use std::collections::{HashMap, HashSet};

use llvm2_ir::{AArch64Opcode, InstId, MachFunction, MachOperand};

use crate::effects::inst_produces_value;
use crate::pass_manager::MachinePass;

/// Maximum unsigned immediate byte offset for 64-bit LDR/STR.
/// AArch64: 12-bit immediate scaled by 8 -> 4095 * 8 = 32760.
const MAX_UNSIGNED_OFFSET: i64 = 32760;

// ---------------------------------------------------------------------------
// Offset encoding helpers
// ---------------------------------------------------------------------------

/// Check if an offset is encodable as an AArch64 scaled unsigned 12-bit
/// immediate for a load/store of the given access size.
///
/// AArch64 LDR/STR unsigned immediate: `imm12 * access_size`, where
/// `imm12` is a 12-bit unsigned value (0..4095).
///
/// Requirements:
/// - `offset >= 0`
/// - `offset` is aligned to `access_size`
/// - `offset / access_size <= 4095`
///
/// `access_size` must be 1, 2, 4, or 8. Other values return `false`.
pub fn is_encodable_offset(offset: i64, access_size: u8) -> bool {
    if offset < 0 {
        return false;
    }
    match access_size {
        1 | 2 | 4 | 8 => {
            let scale = access_size as i64;
            offset % scale == 0 && offset / scale <= 4095
        }
        _ => false,
    }
}

/// Check if an offset is encodable as an AArch64 signed 9-bit immediate
/// for pre-index or post-index addressing.
///
/// Pre/post-index immediates are unscaled, range: -256..=255.
pub fn is_encodable_pre_post_offset(offset: i64) -> bool {
    (-256..=255).contains(&offset)
}

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------

/// AArch64 address mode formation pass.
pub struct AddrModeFormation;

impl MachinePass for AddrModeFormation {
    fn name(&self) -> &str {
        "addr-mode"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;

        // Step 1: Count uses of each VReg as a source operand.
        let use_counts = count_vreg_uses(func);

        // Step 2: Build maps from VReg ID -> InstId for ADD definitions.
        let add_ri_defs = collect_add_ri_defs(func);
        let add_rr_defs = collect_add_rr_defs(func);

        // Step 3: Scan for LDR/STR instructions and try to fold.
        let mut to_delete: HashSet<InstId> = HashSet::new();

        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            for &inst_id in block.insts.clone().iter() {
                let inst = func.inst(inst_id);

                // Only handle LdrRI and StrRI.
                let (base_idx, offset_idx) = match inst.opcode {
                    // LdrRI: [dst, base, offset] — base is operand[1]
                    AArch64Opcode::LdrRI => (1, 2),
                    // StrRI: [src, base, offset] — base is operand[1]
                    AArch64Opcode::StrRI => (1, 2),
                    _ => continue,
                };

                if inst.operands.len() <= offset_idx {
                    continue;
                }

                // Get the base VReg.
                let base_vreg = match inst.operands[base_idx].as_vreg() {
                    Some(v) => v,
                    None => continue,
                };

                // Get the current offset immediate.
                let mem_offset = match inst.operands[offset_idx].as_imm() {
                    Some(v) => v,
                    None => continue,
                };

                // Check the base vreg has exactly 1 use.
                let count = use_counts.get(&base_vreg.id).copied().unwrap_or(0);
                if count != 1 {
                    continue;
                }

                // --- Try form_base_plus_imm: fold AddRI into LDR/STR offset ---
                if let Some(&add_inst_id) = add_ri_defs.get(&base_vreg.id)
                    && !to_delete.contains(&add_inst_id)
                        && form_base_plus_imm(func, inst_id, add_inst_id, base_idx, offset_idx, mem_offset) {
                            to_delete.insert(add_inst_id);
                            changed = true;
                            continue;
                        }

                // --- Try form_base_plus_reg: fold AddRR into LdrRO/StrRO ---
                if let Some(&add_inst_id) = add_rr_defs.get(&base_vreg.id)
                    && !to_delete.contains(&add_inst_id)
                        && form_base_plus_reg(func, inst_id, add_inst_id, base_idx, offset_idx, mem_offset) {
                            to_delete.insert(add_inst_id);
                            changed = true;
                            continue;
                        }
            }
        }

        // Step 4: Delete folded ADD instructions from blocks.
        if !to_delete.is_empty() {
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !to_delete.contains(id));
            }
        }

        changed
    }
}

// ---------------------------------------------------------------------------
// form_base_plus_imm: AddRI + LDR/STR -> LDR/STR with combined offset
// ---------------------------------------------------------------------------

/// Attempt to fold an AddRI instruction into a LDR/STR's addressing mode.
///
/// Returns `true` if the fold was performed. The ADD instruction should
/// then be deleted by the caller.
fn form_base_plus_imm(
    func: &mut MachFunction,
    load_store_id: InstId,
    add_inst_id: InstId,
    base_idx: usize,
    offset_idx: usize,
    mem_offset: i64,
) -> bool {
    let add_inst = func.inst(add_inst_id);
    if add_inst.operands.len() < 3 {
        return false;
    }

    let add_base = add_inst.operands[1].clone();
    let add_offset = match add_inst.operands[2].as_imm() {
        Some(v) => v,
        None => return false,
    };

    // Compute combined offset.
    let combined_offset = add_offset + mem_offset;

    // Validate range: must be non-negative and fit in unsigned
    // 12-bit scaled immediate. Use conservative 64-bit bound.
    if !(0..=MAX_UNSIGNED_OFFSET).contains(&combined_offset) {
        return false;
    }

    // Rewrite the LDR/STR: replace base with the ADD's source,
    // replace offset with combined offset.
    let load_store = func.inst_mut(load_store_id);
    load_store.operands[base_idx] = add_base;
    load_store.operands[offset_idx] = MachOperand::Imm(combined_offset);

    true
}

// ---------------------------------------------------------------------------
// form_base_plus_reg: AddRR + LDR/STR -> LdrRO/StrRO
// ---------------------------------------------------------------------------

/// Attempt to fold an AddRR instruction into a register-offset addressing
/// mode (LdrRO/StrRO).
///
/// Pattern: `ADD Xd, Xn, Xm` + `LDR Xt, [Xd, #0]` -> `LDR Xt, [Xn, Xm]`
///
/// Constraints:
/// - The LDR/STR must have offset == 0 (register-offset mode has no
///   additional immediate offset).
/// - The ADD must have exactly 3 operands: [dst, src1, src2].
///
/// Returns `true` if the fold was performed. The ADD instruction should
/// then be deleted by the caller.
fn form_base_plus_reg(
    func: &mut MachFunction,
    load_store_id: InstId,
    add_inst_id: InstId,
    base_idx: usize,
    _offset_idx: usize,
    mem_offset: i64,
) -> bool {
    // Register-offset addressing does not support an additional immediate,
    // so the existing LDR/STR offset must be exactly 0.
    if mem_offset != 0 {
        return false;
    }

    let add_inst = func.inst(add_inst_id);
    if add_inst.operands.len() < 3 {
        return false;
    }

    // AddRR: [dst, src1, src2] — both src1 and src2 must be VRegs.
    let add_src1 = add_inst.operands[1].clone();
    let add_src2 = add_inst.operands[2].clone();

    if !add_src1.is_vreg() || !add_src2.is_vreg() {
        return false;
    }

    // Determine the new opcode: LdrRI -> LdrRO, StrRI -> StrRO.
    let load_store = func.inst(load_store_id);
    let new_opcode = match load_store.opcode {
        AArch64Opcode::LdrRI => AArch64Opcode::LdrRO,
        AArch64Opcode::StrRI => AArch64Opcode::StrRO,
        _ => return false,
    };

    // Rewrite: change opcode and replace base+offset with src1+src2.
    // LdrRO operands: [dst, base, index]
    // StrRO operands: [src, base, index]
    let load_store = func.inst_mut(load_store_id);
    load_store.opcode = new_opcode;
    load_store.operands[base_idx] = add_src1;
    // offset_idx holds the offset operand; replace with the index register.
    load_store.operands[base_idx + 1] = add_src2;

    // Update flags to match the new opcode.
    load_store.flags = new_opcode.default_flags();

    true
}

// ---------------------------------------------------------------------------
// form_pre_index / form_post_index (detection only — no transform yet)
// ---------------------------------------------------------------------------

/// Detect a pre-index addressing pattern.
///
/// Pattern: `ADD Rbase, Rbase, #N` followed by `LDR Rt, [Rbase, #0]`
/// where Rbase is updated in place (dst == src) and N is in -256..255.
///
/// This would fold to `LDR Rt, [Rbase, #N]!` (pre-index with writeback).
///
/// Currently returns `false` unconditionally because the IR does not have
/// dedicated pre/post-index opcodes for single LDR/STR instructions.
/// The StpPreIndex and LdpPostIndex opcodes exist only for pair operations.
///
/// TODO(#407): Add LdrPreIndex/StrPreIndex/LdrPostIndex/StrPostIndex opcodes
/// to AArch64Opcode, then implement the actual transformation here.
#[allow(dead_code)]
fn form_pre_index(
    func: &MachFunction,
    _load_store_id: InstId,
    add_inst_id: InstId,
    _mem_offset: i64,
) -> bool {
    // Detect: ADD dst == src (in-place update).
    let add_inst = func.inst(add_inst_id);
    if add_inst.opcode != AArch64Opcode::AddRI || add_inst.operands.len() < 3 {
        return false;
    }

    let dst = add_inst.operands[0].as_vreg();
    let src = add_inst.operands[1].as_vreg();
    let offset = add_inst.operands[2].as_imm();

    if let (Some(d), Some(s), Some(off)) = (dst, src, offset)
        && d.id == s.id && is_encodable_pre_post_offset(off) {
            // Pattern detected but cannot transform yet.
            // TODO: emit LdrPreIndex/StrPreIndex when opcodes exist.
            let _ = (d, off);
            return false;
        }

    false
}

/// Detect a post-index addressing pattern.
///
/// Pattern: `LDR Rt, [Rbase, #0]` followed by `ADD Rbase, Rbase, #N`
/// where N is in -256..255 and Rbase is updated after the access.
///
/// This would fold to `LDR Rt, [Rbase], #N` (post-index with writeback).
///
/// Currently returns `false` unconditionally because the IR does not have
/// dedicated post-index opcodes for single LDR/STR instructions.
///
/// TODO(#407): Add LdrPostIndex/StrPostIndex opcodes to AArch64Opcode,
/// then implement the actual transformation here.
#[allow(dead_code)]
fn form_post_index(
    func: &MachFunction,
    _load_store_id: InstId,
    add_inst_id: InstId,
    _mem_offset: i64,
) -> bool {
    // For post-index, the ADD comes AFTER the LDR/STR. The caller would
    // need to scan in a different order (load first, then find subsequent
    // ADD to same base). This function detects the ADD side of the pattern.
    let add_inst = func.inst(add_inst_id);
    if add_inst.opcode != AArch64Opcode::AddRI || add_inst.operands.len() < 3 {
        return false;
    }

    let dst = add_inst.operands[0].as_vreg();
    let src = add_inst.operands[1].as_vreg();
    let offset = add_inst.operands[2].as_imm();

    if let (Some(d), Some(s), Some(off)) = (dst, src, offset)
        && d.id == s.id && is_encodable_pre_post_offset(off) {
            // Pattern detected but cannot transform yet.
            // TODO: emit LdrPostIndex/StrPostIndex when opcodes exist.
            let _ = (d, off);
            return false;
        }

    false
}

// ---------------------------------------------------------------------------
// Analysis helpers
// ---------------------------------------------------------------------------

/// Count how many times each VReg ID appears as a source (use) operand
/// across the entire function.
///
/// Convention: for value-producing instructions, operand[0] is the def;
/// operands[1..] are uses. For non-value-producing instructions (stores,
/// compares, branches), all operands are uses.
fn count_vreg_uses(func: &MachFunction) -> HashMap<u32, u32> {
    let mut counts: HashMap<u32, u32> = HashMap::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            let use_start = if inst_produces_value(inst) { 1 } else { 0 };

            for operand in &inst.operands[use_start..] {
                if let MachOperand::VReg(vreg) = operand {
                    *counts.entry(vreg.id).or_insert(0) += 1;
                }
            }
        }
    }

    counts
}

/// Collect a map from VReg ID (def) -> InstId for all AddRI instructions.
///
/// Only records AddRI instructions since those are the ones we can fold
/// into base+immediate addressing modes.
fn collect_add_ri_defs(func: &MachFunction) -> HashMap<u32, InstId> {
    let mut defs: HashMap<u32, InstId> = HashMap::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.opcode == AArch64Opcode::AddRI
                && let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                    defs.insert(vreg.id, inst_id);
                }
        }
    }

    defs
}

/// Collect a map from VReg ID (def) -> InstId for all AddRR instructions.
///
/// Only records AddRR instructions since those are the ones we can fold
/// into base+register addressing modes (LdrRO/StrRO).
fn collect_add_rr_defs(func: &MachFunction) -> HashMap<u32, InstId> {
    let mut defs: HashMap<u32, InstId> = HashMap::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.opcode == AArch64Opcode::AddRR
                && let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                    defs.insert(vreg.id, inst_id);
                }
        }
    }

    defs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, ProofAnnotation,
        RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_addr_mode".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    // ================================================================
    // is_encodable_offset tests
    // ================================================================

    #[test]
    fn test_encodable_offset_byte() {
        // Byte: scale=1, max=4095
        assert!(is_encodable_offset(0, 1));
        assert!(is_encodable_offset(1, 1));
        assert!(is_encodable_offset(4095, 1));
        assert!(!is_encodable_offset(4096, 1));
        assert!(!is_encodable_offset(-1, 1));
    }

    #[test]
    fn test_encodable_offset_half() {
        // Half: scale=2, max=8190, must be 2-aligned
        assert!(is_encodable_offset(0, 2));
        assert!(is_encodable_offset(2, 2));
        assert!(is_encodable_offset(8190, 2));
        assert!(!is_encodable_offset(8192, 2));
        assert!(!is_encodable_offset(1, 2));   // misaligned
        assert!(!is_encodable_offset(3, 2));   // misaligned
        assert!(!is_encodable_offset(-2, 2));
    }

    #[test]
    fn test_encodable_offset_word() {
        // Word: scale=4, max=16380, must be 4-aligned
        assert!(is_encodable_offset(0, 4));
        assert!(is_encodable_offset(4, 4));
        assert!(is_encodable_offset(16380, 4));
        assert!(!is_encodable_offset(16384, 4));
        assert!(!is_encodable_offset(1, 4));   // misaligned
        assert!(!is_encodable_offset(2, 4));   // misaligned
        assert!(!is_encodable_offset(-4, 4));
    }

    #[test]
    fn test_encodable_offset_double() {
        // Double: scale=8, max=32760, must be 8-aligned
        assert!(is_encodable_offset(0, 8));
        assert!(is_encodable_offset(8, 8));
        assert!(is_encodable_offset(32760, 8));
        assert!(!is_encodable_offset(32768, 8));
        assert!(!is_encodable_offset(1, 8));   // misaligned
        assert!(!is_encodable_offset(4, 8));   // misaligned
        assert!(!is_encodable_offset(-8, 8));
    }

    #[test]
    fn test_encodable_offset_invalid_size() {
        assert!(!is_encodable_offset(0, 0));
        assert!(!is_encodable_offset(0, 3));
        assert!(!is_encodable_offset(0, 16));
    }

    #[test]
    fn test_encodable_pre_post_offset() {
        assert!(is_encodable_pre_post_offset(0));
        assert!(is_encodable_pre_post_offset(255));
        assert!(is_encodable_pre_post_offset(-256));
        assert!(!is_encodable_pre_post_offset(256));
        assert!(!is_encodable_pre_post_offset(-257));
        assert!(is_encodable_pre_post_offset(1));
        assert!(is_encodable_pre_post_offset(-1));
    }

    // ================================================================
    // form_base_plus_imm tests (existing tests, preserved)
    // ================================================================

    #[test]
    fn test_fold_add_imm_into_ldr() {
        // ADD v1, v0, #16
        // LDR v2, [v1, #0]
        // RET
        // -> LDR v2, [v0, #16], ADD deleted
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // ADD should be deleted, leaving LDR + RET
        assert_eq!(block.insts.len(), 2);

        // LDR should now use v0 as base with offset 16
        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.opcode, AArch64Opcode::LdrRI);
        assert_eq!(ldr_inst.operands[0], vreg(2)); // dst unchanged
        assert_eq!(ldr_inst.operands[1], vreg(0)); // base = ADD's source
        assert_eq!(ldr_inst.operands[2], imm(16)); // offset = ADD's imm
    }

    #[test]
    fn test_fold_add_imm_into_str() {
        // ADD v1, v0, #8
        // STR v2, [v1, #0]
        // RET
        // -> STR v2, [v0, #8], ADD deleted
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // STR + RET

        let str_result = func.inst(InstId(1));
        assert_eq!(str_result.opcode, AArch64Opcode::StrRI);
        assert_eq!(str_result.operands[0], vreg(2)); // src unchanged
        assert_eq!(str_result.operands[1], vreg(0)); // base = ADD's source
        assert_eq!(str_result.operands[2], imm(8));  // offset = ADD's imm
    }

    #[test]
    fn test_fold_combined_offsets() {
        // ADD v1, v0, #8
        // LDR v2, [v1, #16]
        // RET
        // -> LDR v2, [v0, #24]
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(16)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(24)); // 8 + 16
    }

    // ---- Safety: no fold when multiple uses ----

    #[test]
    fn test_no_fold_multiple_uses() {
        // ADD v1, v0, #16
        // LDR v2, [v1, #0]
        // ADD v3, v1, #4    <- second use of v1
        // RET
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(1), imm(4)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, add2, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        // All instructions preserved
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4);
    }

    // ---- Safety: offset range validation ----

    #[test]
    fn test_no_fold_offset_out_of_range() {
        // ADD v1, v0, #32000
        // LDR v2, [v1, #1000]
        // RET
        // Combined = 33000, exceeds 32760
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(32000)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(1000)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // unchanged
    }

    #[test]
    fn test_no_fold_negative_offset() {
        // ADD v1, v0, #-8
        // LDR v2, [v1, #0]
        // RET
        // Combined offset = -8, negative -> reject
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(-8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
    }

    // ---- Non-ADD definitions ----

    #[test]
    fn test_no_fold_non_add_def() {
        // MOV v1, v0
        // LDR v2, [v1, #0]
        // RET
        // v1 defined by MOV, not ADD -> no fold
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(1), vreg(0)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
    }

    // ---- Idempotency ----

    #[test]
    fn test_idempotent() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));  // First run: folds
        assert!(!pass.run(&mut func)); // Second run: nothing to do
    }

    // ---- Proof annotation preservation ----

    #[test]
    fn test_preserves_ldr_proof() {
        // ADD v1, v0, #16
        // LDR v2, [v1, #0] [InBounds]
        // RET
        // -> LDR v2, [v0, #16] [InBounds] — proof preserved
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.proof, Some(ProofAnnotation::InBounds));
    }

    // ---- Edge cases ----

    #[test]
    fn test_max_valid_offset() {
        // ADD v1, v0, #32760 (max valid)
        // LDR v2, [v1, #0]
        // RET
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(32760)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[2], imm(32760));
    }

    #[test]
    fn test_just_over_max_offset() {
        // ADD v1, v0, #32761 (one over max)
        // LDR v2, [v1, #0]
        // RET
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(32761)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_zero_offset_add() {
        // ADD v1, v0, #0 + LDR v2, [v1, #8] -> LDR v2, [v0, #8]
        // (Peephole would normally remove this ADD, but if it reaches
        // addr-mode first, we can still fold it.)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(0)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(8));
    }

    #[test]
    fn test_multiple_folds_in_one_block() {
        // ADD v1, v0, #8
        // LDR v2, [v1, #0]   <- fold this
        // ADD v4, v3, #16
        // STR v5, [v4, #0]   <- fold this
        // RET
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(4), vreg(3), imm(16)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(5), vreg(4), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, ldr, add2, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // Both ADDs deleted: LDR + STR + RET = 3
        assert_eq!(block.insts.len(), 3);

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(8));

        let str_inst = func.inst(InstId(3));
        assert_eq!(str_inst.operands[1], vreg(3));
        assert_eq!(str_inst.operands[2], imm(16));
    }

    #[test]
    fn test_no_change_empty_function() {
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_str_base_is_use_not_def() {
        // Ensure that for STR [src, base, offset], the base (operand[1])
        // is correctly identified as a use, not a def.
        // ADD v1, v0, #8    (v1 is used once by STR as base)
        // STR v2, [v1, #0]  (v1 is base at index 1, v2 is src at index 0)
        // RET
        //
        // v1 use count: used by STR operand[1] = 1 use
        // v2 use count: used by STR operand[0] = 1 use (STR has no def, all are uses)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        // ADD deleted, STR rewritten
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);

        let str_result = func.inst(InstId(1));
        assert_eq!(str_result.operands[1], vreg(0));
        assert_eq!(str_result.operands[2], imm(8));
    }

    // ================================================================
    // form_base_plus_reg tests
    // ================================================================

    #[test]
    fn test_fold_add_rr_into_ldr_ro() {
        // ADD v2, v0, v1     (v2 = v0 + v1)
        // LDR v3, [v2, #0]   (load from [v2])
        // RET
        // -> LDR v3, [v0, v1] (LdrRO), ADD deleted
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // ADD deleted: LDR + RET = 2
        assert_eq!(block.insts.len(), 2);

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.opcode, AArch64Opcode::LdrRO);
        assert_eq!(ldr_inst.operands[0], vreg(3)); // dst unchanged
        assert_eq!(ldr_inst.operands[1], vreg(0)); // base = ADD src1
        assert_eq!(ldr_inst.operands[2], vreg(1)); // index = ADD src2
    }

    #[test]
    fn test_fold_add_rr_into_str_ro() {
        // ADD v2, v0, v1
        // STR v3, [v2, #0]
        // RET
        // -> STR v3, [v0, v1] (StrRO), ADD deleted
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(3), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);

        let str_result = func.inst(InstId(1));
        assert_eq!(str_result.opcode, AArch64Opcode::StrRO);
        assert_eq!(str_result.operands[0], vreg(3)); // src unchanged
        assert_eq!(str_result.operands[1], vreg(0)); // base = ADD src1
        assert_eq!(str_result.operands[2], vreg(1)); // index = ADD src2
    }

    #[test]
    fn test_no_fold_add_rr_nonzero_offset() {
        // ADD v2, v0, v1
        // LDR v3, [v2, #8]   <- offset != 0, can't fold to reg-offset
        // RET
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(2), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // unchanged
    }

    #[test]
    fn test_no_fold_add_rr_multiple_uses() {
        // ADD v2, v0, v1
        // LDR v3, [v2, #0]
        // ADD v4, v2, v5    <- second use of v2
        // RET
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(2), imm(0)]);
        let add2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(4), vreg(2), vreg(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, add2, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4);
    }

    #[test]
    fn test_fold_add_rr_preserves_proof() {
        // ADD v2, v0, v1
        // LDR v3, [v2, #0] [InBounds]
        // RET
        // -> LDR v3, [v0, v1] [InBounds] — proof preserved
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(2), imm(0)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.opcode, AArch64Opcode::LdrRO);
        assert_eq!(ldr_inst.proof, Some(ProofAnnotation::InBounds));
    }

    #[test]
    fn test_fold_add_rr_idempotent() {
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));  // First: folds
        assert!(!pass.run(&mut func)); // Second: nothing to do
    }

    #[test]
    fn test_mixed_imm_and_reg_folds() {
        // ADD v1, v0, #8
        // LDR v2, [v1, #0]   <- fold via AddRI (base+imm)
        // ADD v4, v3, v5
        // STR v6, [v4, #0]   <- fold via AddRR (base+reg)
        // RET
        let add_ri = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let add_rr = MachInst::new(AArch64Opcode::AddRR, vec![vreg(4), vreg(3), vreg(5)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(6), vreg(4), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add_ri, ldr, add_rr, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // Both ADDs deleted: LDR + STR + RET = 3
        assert_eq!(block.insts.len(), 3);

        // LDR got base+imm fold
        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.opcode, AArch64Opcode::LdrRI);
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(8));

        // STR got base+reg fold
        let str_result = func.inst(InstId(3));
        assert_eq!(str_result.opcode, AArch64Opcode::StrRO);
        assert_eq!(str_result.operands[1], vreg(3));
        assert_eq!(str_result.operands[2], vreg(5));
    }

    // ================================================================
    // form_pre_index / form_post_index tests (detection only)
    // ================================================================

    #[test]
    fn test_pre_index_detection_returns_false() {
        // ADD v0, v0, #16  (in-place update, pre-index candidate)
        // LDR v1, [v0, #0]
        // RET
        // Currently returns false since pre-index opcodes don't exist.
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add, ldr, ret]);

        // Directly test the detection function.
        assert!(!form_pre_index(&func, InstId(1), InstId(0), 0));
    }

    #[test]
    fn test_post_index_detection_returns_false() {
        // LDR v1, [v0, #0]
        // ADD v0, v0, #16  (post-index candidate)
        // RET
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(0), imm(16)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![ldr, add, ret]);

        // Directly test the detection function.
        assert!(!form_post_index(&func, InstId(0), InstId(1), 0));
    }

    #[test]
    fn test_pre_index_not_in_range() {
        // ADD v0, v0, #300 (> 255, not encodable in 9-bit signed)
        // LDR v1, [v0, #0]
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(0), imm(300)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let func = make_func_with_insts(vec![add, ldr]);

        assert!(!form_pre_index(&func, InstId(1), InstId(0), 0));
    }

    #[test]
    fn test_pre_index_dst_ne_src() {
        // ADD v1, v0, #16 (dst != src, not an in-place update)
        // LDR v2, [v1, #0]
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let func = make_func_with_insts(vec![add, ldr]);

        assert!(!form_pre_index(&func, InstId(1), InstId(0), 0));
    }
}

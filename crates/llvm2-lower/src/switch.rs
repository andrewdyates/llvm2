// llvm2-lower/switch.rs - Switch lowering strategies
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Switch statement lowering with strategy selection.
//!
//! Three strategies for lowering `Switch` opcodes to AArch64 machine code:
//!
//! 1. **Linear scan** (N <= 3): Sequential CMP+B.EQ chain. O(n) but
//!    the constant factor is low enough that it beats BST overhead.
//!
//! 2. **Binary search tree** (N > 3, sparse): Balanced BST of compare-
//!    and-branch nodes, giving O(log n) worst-case dispatch. Each internal
//!    node compares the selector against a pivot, branching to the target
//!    if equal, or to left/right subtrees otherwise.
//!
//! 3. **Jump table** (N >= 4, density > 0.4): O(1) table lookup via
//!    bounds check + indexed indirect branch. Emits SUB (normalize),
//!    CMP+B.HI (range check), ADR+LDRSW+ADD+BR (table dispatch).
//!
//! Reference: LLVM `SwitchLoweringUtils.cpp`, `SwitchLowering.cpp`

use std::collections::HashMap;

use crate::instructions::Block;
use crate::isel::{
    AArch64CC, AArch64Opcode, ISelFunction, ISelInst, ISelOperand,
};
use llvm2_ir::regs::{RegClass, VReg};

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

/// Switch lowering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchStrategy {
    /// Sequential CMP+B.EQ for N <= 3 cases.
    LinearScan,
    /// Balanced binary search tree for sparse switches (N > 3, density <= 0.4).
    BinarySearch,
    /// Indexed jump table for dense switches (N >= 4, density > 0.4).
    JumpTable,
}

/// Density threshold: jump table is used when `num_cases / range > DENSITY_THRESHOLD`.
const DENSITY_THRESHOLD: f64 = 0.4;

/// Minimum case count for jump table or BST. Below this, linear scan wins.
const LINEAR_SCAN_MAX: usize = 3;

/// Choose the optimal lowering strategy for a switch statement.
///
/// - N <= 3: `LinearScan` (sequential compare is cheaper than BST overhead)
/// - N >= 4 and density > 0.4: `JumpTable` (O(1) dispatch)
/// - N > 3 and sparse: `BinarySearch` (O(log n) dispatch)
pub fn choose_strategy(cases: &[(i64, Block)]) -> SwitchStrategy {
    if cases.len() <= LINEAR_SCAN_MAX {
        return SwitchStrategy::LinearScan;
    }
    // Compute density = num_cases / (max - min + 1)
    let min_val = cases.iter().map(|(v, _)| *v).min().unwrap();
    let max_val = cases.iter().map(|(v, _)| *v).max().unwrap();
    let range = (max_val - min_val + 1) as f64;
    let density = cases.len() as f64 / range;
    if density > DENSITY_THRESHOLD {
        SwitchStrategy::JumpTable
    } else {
        SwitchStrategy::BinarySearch
    }
}

// ---------------------------------------------------------------------------
// Block and VReg allocation helpers
// ---------------------------------------------------------------------------

/// Allocate a fresh `Block` ID, inserting it into the function.
fn alloc_block(func: &mut ISelFunction, next_block_id: &mut u32) -> Block {
    let block = Block(*next_block_id);
    *next_block_id += 1;
    func.ensure_block(block);
    block
}

/// Allocate a fresh `VReg` for constant materialization.
fn alloc_vreg(func: &mut ISelFunction, class: RegClass) -> VReg {
    let id = func.next_vreg;
    func.next_vreg += 1;
    VReg { id, class }
}

// ---------------------------------------------------------------------------
// CMP emission helpers (shared by linear scan and BST)
// ---------------------------------------------------------------------------

/// Emit a CMP of `selector` against `case_val`.
///
/// Values in `0..=0xFFF` use `CmpRI` (12-bit unsigned immediate).
/// Others are materialized into a register via `Movz`, then `CmpRR`.
fn emit_cmp(
    func: &mut ISelFunction,
    block: Block,
    selector: &ISelOperand,
    case_val: i64,
    is_32: bool,
) {
    let fits_imm12 = case_val >= 0 && case_val <= 0xFFF;
    if fits_imm12 {
        func.push_inst(
            block,
            ISelInst::new(
                AArch64Opcode::CmpRI,
                vec![selector.clone(), ISelOperand::Imm(case_val)],
            ),
        );
    } else {
        // Materialize into register, then CmpRR.
        let class = if is_32 { RegClass::Gpr32 } else { RegClass::Gpr64 };
        let tmp = alloc_vreg(func, class);
        func.push_inst(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(tmp), ISelOperand::Imm(case_val)],
            ),
        );
        func.push_inst(
            block,
            ISelInst::new(
                AArch64Opcode::CmpRR,
                vec![selector.clone(), ISelOperand::VReg(tmp)],
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Linear scan emission
// ---------------------------------------------------------------------------

/// Emit a linear scan (sequential CMP+B.EQ chain) for a small switch.
///
/// For each case: `CMP selector, #val; B.EQ target`
/// After all cases: `B default`
pub fn emit_linear_scan(
    func: &mut ISelFunction,
    block: Block,
    selector: &ISelOperand,
    is_32: bool,
    cases: &[(i64, Block)],
    default: Block,
) {
    for &(case_val, target) in cases {
        emit_cmp(func, block, selector, case_val, is_32);

        // B.EQ target
        func.push_inst(
            block,
            ISelInst::new(
                AArch64Opcode::BCond,
                vec![
                    ISelOperand::CondCode(AArch64CC::EQ),
                    ISelOperand::Block(target),
                ],
            ),
        );

        // Record successor
        let entry = func.blocks.entry(block).or_default();
        if !entry.successors.contains(&target) {
            entry.successors.push(target);
        }
    }

    // Unconditional branch to default
    func.push_inst(
        block,
        ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(default)]),
    );
    let entry = func.blocks.entry(block).or_default();
    if !entry.successors.contains(&default) {
        entry.successors.push(default);
    }
}

// ---------------------------------------------------------------------------
// Binary search tree emission
// ---------------------------------------------------------------------------

/// Emit a binary search tree switch lowering.
///
/// Creates a balanced BST of compare-and-branch blocks:
/// - Each node compares the selector against the median case value.
/// - Equal: branch to the case's target block.
/// - Less than: branch to the left subtree's block.
/// - Greater/equal (after EQ check): branch to the right subtree's block.
/// - Leaf groups (1-3 cases) use linear scan within a single block.
///
/// The entry block (`entry_block`) receives the root of the BST.
pub fn emit_binary_search(
    func: &mut ISelFunction,
    next_block_id: &mut u32,
    selector: &ISelOperand,
    is_32: bool,
    cases: &[(i64, Block)],
    default: Block,
    entry_block: Block,
) {
    // Sort cases by value for balanced partitioning.
    let mut sorted_cases: Vec<(i64, Block)> = cases.to_vec();
    sorted_cases.sort_by_key(|(v, _)| *v);

    emit_bst_node(
        func,
        next_block_id,
        selector,
        is_32,
        &sorted_cases,
        default,
        entry_block,
    );
}

/// Recursive BST node emission.
///
/// `cases` must be sorted by value. Emits instructions into `block`:
/// - For 1-3 cases: linear scan (base case).
/// - For 4+ cases: pick median, emit CMP+B.EQ+B.LT, recurse into
///   left/right subtree blocks.
fn emit_bst_node(
    func: &mut ISelFunction,
    next_block_id: &mut u32,
    selector: &ISelOperand,
    is_32: bool,
    cases: &[(i64, Block)],
    default: Block,
    block: Block,
) {
    // Base case: small enough for linear scan.
    if cases.len() <= LINEAR_SCAN_MAX {
        emit_linear_scan(func, block, selector, is_32, cases, default);
        return;
    }

    // Pick the median as pivot.
    let mid = cases.len() / 2;
    let (pivot_val, pivot_target) = cases[mid];

    // Partition:
    // left = cases[..mid]     (values < pivot_val)
    // right = cases[mid+1..]  (values > pivot_val)
    let left_cases = &cases[..mid];
    let right_cases = &cases[mid + 1..];

    // Allocate blocks for left and right subtrees.
    let left_block = alloc_block(func, next_block_id);
    let right_block = alloc_block(func, next_block_id);

    // Emit CMP selector, pivot_val
    emit_cmp(func, block, selector, pivot_val, is_32);

    // B.EQ pivot_target (exact match)
    func.push_inst(
        block,
        ISelInst::new(
            AArch64Opcode::BCond,
            vec![
                ISelOperand::CondCode(AArch64CC::EQ),
                ISelOperand::Block(pivot_target),
            ],
        ),
    );

    // B.LT left_block (selector < pivot, search left subtree)
    // Use signed comparison: B.LT for signed less-than.
    func.push_inst(
        block,
        ISelInst::new(
            AArch64Opcode::BCond,
            vec![
                ISelOperand::CondCode(AArch64CC::LT),
                ISelOperand::Block(left_block),
            ],
        ),
    );

    // Fall through to right_block (selector > pivot)
    func.push_inst(
        block,
        ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(right_block)]),
    );

    // Record successors for the current block.
    {
        let entry = func.blocks.entry(block).or_default();
        if !entry.successors.contains(&pivot_target) {
            entry.successors.push(pivot_target);
        }
        if !entry.successors.contains(&left_block) {
            entry.successors.push(left_block);
        }
        if !entry.successors.contains(&right_block) {
            entry.successors.push(right_block);
        }
    }

    // Recurse: emit left subtree into left_block.
    // If left_cases is empty, left subtree just jumps to default.
    if left_cases.is_empty() {
        func.push_inst(
            left_block,
            ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(default)]),
        );
        let entry = func.blocks.entry(left_block).or_default();
        if !entry.successors.contains(&default) {
            entry.successors.push(default);
        }
    } else {
        emit_bst_node(
            func,
            next_block_id,
            selector,
            is_32,
            left_cases,
            default,
            left_block,
        );
    }

    // Recurse: emit right subtree into right_block.
    if right_cases.is_empty() {
        func.push_inst(
            right_block,
            ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(default)]),
        );
        let entry = func.blocks.entry(right_block).or_default();
        if !entry.successors.contains(&default) {
            entry.successors.push(default);
        }
    } else {
        emit_bst_node(
            func,
            next_block_id,
            selector,
            is_32,
            right_cases,
            default,
            right_block,
        );
    }
}

// ---------------------------------------------------------------------------
// Jump table emission
// ---------------------------------------------------------------------------

/// Emit a jump table switch lowering.
///
/// Produces the AArch64 indirect-branch sequence:
/// ```asm
///   SUB  Xindex, Xselector, #min_val    ; normalize to 0-based index
///   CMP  Xindex, #range                  ; range check
///   B.HI default_block                   ; out of range -> default
///   ADR  Xbase, jump_table               ; PC-relative address of table
///   LDRSW Xoffset, [Xbase, Xindex, LSL #2] ; load 32-bit signed offset
///   ADD  Xtarget, Xbase, Xoffset        ; compute target address
///   BR   Xtarget                          ; indirect branch
/// ```
///
/// The jump table is a dense array of targets indexed by `selector - min_val`.
/// Holes (case values without explicit targets) map to the default block.
///
/// `next_block_id` is accepted for API consistency with `emit_binary_search`
/// but is not used (jump tables don't require intermediate blocks).
#[allow(unused_variables)]
pub fn emit_jump_table(
    func: &mut ISelFunction,
    next_block_id: &mut u32,
    selector: &ISelOperand,
    is_32: bool,
    cases: &[(i64, Block)],
    default: Block,
    entry_block: Block,
) {
    assert!(!cases.is_empty(), "Jump table requires at least one case");

    let min_val = cases.iter().map(|(v, _)| *v).min().unwrap();
    let max_val = cases.iter().map(|(v, _)| *v).max().unwrap();
    let range = max_val - min_val;

    // Build the dense targets vector: for each index 0..=range,
    // map to the case target if one exists, otherwise to the default block.
    let case_map: HashMap<i64, Block> = cases.iter().cloned().collect();
    let mut targets = Vec::with_capacity((range + 1) as usize);
    for i in 0..=range {
        let val = min_val + i;
        targets.push(*case_map.get(&val).unwrap_or(&default));
    }

    // All vregs are 64-bit for address computation.
    let index_vreg = alloc_vreg(func, RegClass::Gpr64);
    let base_vreg = alloc_vreg(func, RegClass::Gpr64);
    let offset_vreg = alloc_vreg(func, RegClass::Gpr64);
    let target_vreg = alloc_vreg(func, RegClass::Gpr64);

    // 1. SUB index_vreg, selector, #min_val (normalize to 0-based)
    if min_val == 0 {
        // No subtraction needed; just move the selector.
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::MovR,
                vec![ISelOperand::VReg(index_vreg), selector.clone()],
            ),
        );
    } else if min_val > 0 && min_val <= 0xFFF {
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::SubRI,
                vec![
                    ISelOperand::VReg(index_vreg),
                    selector.clone(),
                    ISelOperand::Imm(min_val),
                ],
            ),
        );
    } else {
        // min_val doesn't fit in imm12; materialize then SubRR.
        let tmp_vreg = alloc_vreg(func, RegClass::Gpr64);
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(tmp_vreg), ISelOperand::Imm(min_val)],
            ),
        );
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(index_vreg),
                    selector.clone(),
                    ISelOperand::VReg(tmp_vreg),
                ],
            ),
        );
    }

    // 2. CMP index_vreg, #range then B.HI default (out-of-range check)
    if range >= 0 && range <= 0xFFF {
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::CmpRI,
                vec![ISelOperand::VReg(index_vreg), ISelOperand::Imm(range)],
            ),
        );
    } else {
        let range_vreg = alloc_vreg(func, RegClass::Gpr64);
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(range_vreg), ISelOperand::Imm(range)],
            ),
        );
        func.push_inst(
            entry_block,
            ISelInst::new(
                AArch64Opcode::CmpRR,
                vec![ISelOperand::VReg(index_vreg), ISelOperand::VReg(range_vreg)],
            ),
        );
    }

    func.push_inst(
        entry_block,
        ISelInst::new(
            AArch64Opcode::BCond,
            vec![
                ISelOperand::CondCode(AArch64CC::HI),
                ISelOperand::Block(default),
            ],
        ),
    );

    // 3. ADR base_vreg, jump_table
    //
    // Register the table data on the function's side-table and reference
    // it by index. The codegen pipeline will patch this Adr's placeholder
    // immediate with the byte offset from the Adr to the appended table
    // once block layout is finalized.
    let jt_idx = func.add_jump_table(min_val, targets.clone());
    func.push_inst(
        entry_block,
        ISelInst::new(
            AArch64Opcode::Adr,
            vec![
                ISelOperand::VReg(base_vreg),
                ISelOperand::JumpTableIndex(jt_idx),
            ],
        ),
    );

    // 4. LDRSW offset_vreg, [base_vreg, index_vreg, LSL #2]
    func.push_inst(
        entry_block,
        ISelInst::new(
            AArch64Opcode::LdrswRO,
            vec![
                ISelOperand::VReg(offset_vreg),
                ISelOperand::VReg(base_vreg),
                ISelOperand::VReg(index_vreg),
            ],
        ),
    );

    // 5. ADD target_vreg, base_vreg, offset_vreg
    func.push_inst(
        entry_block,
        ISelInst::new(
            AArch64Opcode::AddRR,
            vec![
                ISelOperand::VReg(target_vreg),
                ISelOperand::VReg(base_vreg),
                ISelOperand::VReg(offset_vreg),
            ],
        ),
    );

    // 6. BR target_vreg (indirect branch)
    func.push_inst(
        entry_block,
        ISelInst::new(
            AArch64Opcode::Br,
            vec![ISelOperand::VReg(target_vreg)],
        ),
    );

    // Record all unique successors (case targets + default).
    let block_entry = func.blocks.entry(entry_block).or_default();
    for target in &targets {
        if !block_entry.successors.contains(target) {
            block_entry.successors.push(*target);
        }
    }
    if !block_entry.successors.contains(&default) {
        block_entry.successors.push(default);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Signature;

    fn make_test_func() -> (ISelFunction, u32) {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let func = ISelFunction::new("test_switch".to_string(), sig);
        // Reserve block IDs 0-20 for test targets; BST intermediate blocks start at 100.
        (func, 100)
    }

    // -------------------------------------------------------------------
    // Strategy selection tests
    // -------------------------------------------------------------------

    #[test]
    fn strategy_empty() {
        assert_eq!(choose_strategy(&[]), SwitchStrategy::LinearScan);
    }

    #[test]
    fn strategy_one_case() {
        assert_eq!(
            choose_strategy(&[(42, Block(1))]),
            SwitchStrategy::LinearScan
        );
    }

    #[test]
    fn strategy_three_cases() {
        assert_eq!(
            choose_strategy(&[(0, Block(1)), (5, Block(2)), (10, Block(3))]),
            SwitchStrategy::LinearScan
        );
    }

    #[test]
    fn strategy_four_dense_cases() {
        // 4 cases, range = 4, density = 4/4 = 1.0 > 0.4 -> JumpTable
        assert_eq!(
            choose_strategy(&[
                (0, Block(1)),
                (1, Block(2)),
                (2, Block(3)),
                (3, Block(4)),
            ]),
            SwitchStrategy::JumpTable
        );
    }

    #[test]
    fn strategy_four_sparse_cases() {
        // 4 cases, range = 100, density = 4/100 = 0.04 -> BinarySearch
        assert_eq!(
            choose_strategy(&[
                (0, Block(1)),
                (33, Block(2)),
                (66, Block(3)),
                (99, Block(4)),
            ]),
            SwitchStrategy::BinarySearch
        );
    }

    #[test]
    fn strategy_density_boundary() {
        // 4 cases, range = 10, density = 4/10 = 0.4 -> NOT > 0.4 -> BinarySearch
        assert_eq!(
            choose_strategy(&[
                (0, Block(1)),
                (3, Block(2)),
                (6, Block(3)),
                (9, Block(4)),
            ]),
            SwitchStrategy::BinarySearch
        );
    }

    #[test]
    fn strategy_density_just_above() {
        // 5 cases, range = 10, density = 5/10 = 0.5 > 0.4 -> JumpTable
        assert_eq!(
            choose_strategy(&[
                (0, Block(1)),
                (2, Block(2)),
                (4, Block(3)),
                (6, Block(4)),
                (9, Block(5)),
            ]),
            SwitchStrategy::JumpTable
        );
    }

    // -------------------------------------------------------------------
    // Linear scan emission tests
    // -------------------------------------------------------------------

    #[test]
    fn linear_scan_two_cases() {
        let (mut func, _) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        func.ensure_block(Block(1));
        func.ensure_block(Block(2));
        func.ensure_block(Block(3));

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr32));
        emit_linear_scan(
            &mut func,
            entry,
            &selector,
            true,
            &[(10, Block(1)), (20, Block(2))],
            Block(3),
        );

        let insts = &func.blocks[&entry].insts;
        // 2 cases * (CMP + B.EQ) + 1 B = 5 instructions
        assert_eq!(insts.len(), 5, "2 cases: 2*(CMP+B.EQ) + B default");

        // Last instruction should be B to default
        let last = insts.last().unwrap();
        assert_eq!(last.opcode, AArch64Opcode::B);
        assert_eq!(last.operands[0], ISelOperand::Block(Block(3)));
    }

    // -------------------------------------------------------------------
    // Binary search tree emission tests
    // -------------------------------------------------------------------

    #[test]
    fn bst_four_cases() {
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (10, Block(1)),
            (30, Block(2)),
            (50, Block(3)),
            (70, Block(4)),
        ];

        emit_binary_search(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        // Root node (entry block) should have:
        // CMP selector, #50 (pivot = cases[2])
        // B.EQ Block(3)
        // B.LT left_block
        // B right_block
        let root_insts = &func.blocks[&entry].insts;
        assert!(root_insts.len() >= 3, "Root node should have CMP+B.EQ+B.LT+B");

        // Check that the root compares against pivot value 50 (median of sorted [10,30,50,70])
        let cmp_inst = &root_insts[0];
        assert_eq!(cmp_inst.opcode, AArch64Opcode::CmpRI);
        assert_eq!(cmp_inst.operands[1], ISelOperand::Imm(50));

        // B.EQ should target Block(3) (the case for value 50)
        let beq = &root_insts[1];
        assert_eq!(beq.opcode, AArch64Opcode::BCond);
        assert_eq!(beq.operands[0], ISelOperand::CondCode(AArch64CC::EQ));
        assert_eq!(beq.operands[1], ISelOperand::Block(Block(3)));

        // B.LT to left subtree
        let blt = &root_insts[2];
        assert_eq!(blt.opcode, AArch64Opcode::BCond);
        assert_eq!(blt.operands[0], ISelOperand::CondCode(AArch64CC::LT));

        // Unconditional B to right subtree
        let b_right = &root_insts[3];
        assert_eq!(b_right.opcode, AArch64Opcode::B);

        // Verify intermediate blocks were created
        let total_blocks = func.blocks.len();
        assert!(
            total_blocks > 6,
            "BST should create intermediate blocks: got {}",
            total_blocks
        );
    }

    #[test]
    fn bst_eight_cases_depth() {
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=9 {
            func.ensure_block(Block(i));
        }
        let default = Block(9);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        // 8 sparse cases with large gaps -> BinarySearch
        let cases: Vec<(i64, Block)> = vec![
            (100, Block(1)),
            (200, Block(2)),
            (300, Block(3)),
            (400, Block(4)),
            (500, Block(5)),
            (600, Block(6)),
            (700, Block(7)),
            (800, Block(8)),
        ];

        emit_binary_search(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        // Root pivot should be cases[4] = (500, Block(5))
        let root_insts = &func.blocks[&entry].insts;
        // CMP may be CmpRI (500 fits in imm12)
        let cmp = &root_insts[0];
        assert_eq!(cmp.opcode, AArch64Opcode::CmpRI);
        assert_eq!(cmp.operands[1], ISelOperand::Imm(500));

        // Verify all 8 target blocks are reachable as successors somewhere
        let all_succs: Vec<Block> = func
            .blocks
            .values()
            .flat_map(|b| b.successors.iter().copied())
            .collect();
        for i in 1..=8 {
            assert!(
                all_succs.contains(&Block(i)),
                "Block({}) should be reachable",
                i
            );
        }
        assert!(
            all_succs.contains(&default),
            "Default block should be reachable"
        );
    }

    #[test]
    fn bst_large_values_use_reg() {
        // Case values > 0xFFF should use Movz + CmpRR
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (0x1000, Block(1)),
            (0x2000, Block(2)),
            (0x3000, Block(3)),
            (0x4000, Block(4)),
        ];

        emit_binary_search(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        // Root pivot = cases[2] = 0x3000 > 0xFFF, so should use Movz+CmpRR
        let root_insts = &func.blocks[&entry].insts;
        assert_eq!(root_insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(root_insts[1].opcode, AArch64Opcode::CmpRR);
    }

    #[test]
    fn bst_successors_correct() {
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr32));
        let cases = vec![
            (1, Block(1)),
            (2, Block(2)),
            (3, Block(3)),
            (4, Block(4)),
        ];

        emit_binary_search(
            &mut func,
            &mut next_block,
            &selector,
            true,
            &cases,
            default,
            entry,
        );

        // Every block should have at least one successor
        for (block_id, block) in &func.blocks {
            if block.insts.is_empty() {
                continue; // Skip pre-existing empty target blocks
            }
            assert!(
                !block.successors.is_empty(),
                "Block {:?} has instructions but no successors",
                block_id
            );
        }
    }

    #[test]
    fn bst_single_case_per_subtree() {
        // 4 cases -> BST splits into [1 case] + pivot + [2 cases]
        // The 1-case subtree should use linear scan
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (10, Block(1)),
            (20, Block(2)),
            (30, Block(3)),
            (40, Block(4)),
        ];

        emit_binary_search(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        // Sorted: [10,20,30,40], pivot=cases[2]=(30, Block(3))
        // Left=[10,20], right=[40]
        // Left subtree (2 cases) -> linear scan: 2*(CMP+B.EQ) + B = 5 insts
        // Right subtree (1 case) -> linear scan: 1*(CMP+B.EQ) + B = 3 insts
        let left_block = Block(100); // first allocated
        let right_block = Block(101); // second allocated

        let left_insts = &func.blocks[&left_block].insts;
        assert_eq!(
            left_insts.len(),
            5,
            "Left subtree (2 cases) should have 5 instructions: got {}",
            left_insts.len()
        );

        let right_insts = &func.blocks[&right_block].insts;
        assert_eq!(
            right_insts.len(),
            3,
            "Right subtree (1 case) should have 3 instructions: got {}",
            right_insts.len()
        );
    }

    // -------------------------------------------------------------------
    // Jump table emission tests
    // -------------------------------------------------------------------

    #[test]
    fn jump_table_four_consecutive_cases() {
        // Cases 0,1,2,3 -> min=0, range=3, density=1.0
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (0, Block(1)),
            (1, Block(2)),
            (2, Block(3)),
            (3, Block(4)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // min_val=0: MovR (copy selector), CmpRI #3, B.HI, ADR, LDRSW, ADD, BR = 7 insts
        assert_eq!(insts.len(), 7, "4 consecutive cases from 0: 7 instructions");

        // First inst: MovR (min_val == 0 path)
        assert_eq!(insts[0].opcode, AArch64Opcode::MovR);

        // CmpRI #3 (range = 3)
        assert_eq!(insts[1].opcode, AArch64Opcode::CmpRI);
        assert_eq!(insts[1].operands[1], ISelOperand::Imm(3));

        // B.HI default
        assert_eq!(insts[2].opcode, AArch64Opcode::BCond);
        assert_eq!(insts[2].operands[0], ISelOperand::CondCode(AArch64CC::HI));
        assert_eq!(insts[2].operands[1], ISelOperand::Block(default));

        // ADR with JumpTableIndex — table data is now registered on the
        // function side-table (`func.jump_tables`) and referenced by index.
        assert_eq!(insts[3].opcode, AArch64Opcode::Adr);
        let jt_idx = if let ISelOperand::JumpTableIndex(idx) = &insts[3].operands[1] {
            *idx
        } else {
            panic!("ADR operand[1] should be JumpTableIndex, got {:?}", insts[3].operands[1]);
        };
        let jt = &func.jump_tables[jt_idx as usize];
        assert_eq!(jt.min_val, 0);
        assert_eq!(jt.targets.len(), 4);
        assert_eq!(jt.targets[0], Block(1));
        assert_eq!(jt.targets[1], Block(2));
        assert_eq!(jt.targets[2], Block(3));
        assert_eq!(jt.targets[3], Block(4));

        // LDRSW
        assert_eq!(insts[4].opcode, AArch64Opcode::LdrswRO);

        // ADD
        assert_eq!(insts[5].opcode, AArch64Opcode::AddRR);

        // BR
        assert_eq!(insts[6].opcode, AArch64Opcode::Br);
    }

    #[test]
    fn jump_table_with_holes() {
        // Cases 0,1,3,4,5 -> hole at 2, range=5, density=5/6=0.83
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=6 {
            func.ensure_block(Block(i));
        }
        let default = Block(6);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (0, Block(1)),
            (1, Block(2)),
            (3, Block(3)),
            (4, Block(4)),
            (5, Block(5)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // Check the jump table has 6 entries with hole at index 2 -> default
        let adr_inst = insts.iter().find(|i| i.opcode == AArch64Opcode::Adr).unwrap();
        let jt_idx = if let ISelOperand::JumpTableIndex(idx) = &adr_inst.operands[1] {
            *idx
        } else {
            panic!("ADR operand[1] should be JumpTableIndex, got {:?}", adr_inst.operands[1]);
        };
        let jt = &func.jump_tables[jt_idx as usize];
        assert_eq!(jt.min_val, 0);
        assert_eq!(jt.targets.len(), 6, "Range 0..5 = 6 entries");
        assert_eq!(jt.targets[0], Block(1));
        assert_eq!(jt.targets[1], Block(2));
        assert_eq!(jt.targets[2], default, "Hole at index 2 should map to default");
        assert_eq!(jt.targets[3], Block(3));
        assert_eq!(jt.targets[4], Block(4));
        assert_eq!(jt.targets[5], Block(5));
    }

    #[test]
    fn jump_table_negative_min_val() {
        // Cases -3,-2,-1,0,1 -> min=-3, range=4, density=5/5=1.0
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=6 {
            func.ensure_block(Block(i));
        }
        let default = Block(6);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (-3, Block(1)),
            (-2, Block(2)),
            (-1, Block(3)),
            (0, Block(4)),
            (1, Block(5)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // min_val=-3 doesn't fit imm12 (negative), so should use Movz+SubRR
        assert_eq!(insts[0].opcode, AArch64Opcode::Movz, "Negative min_val uses Movz");
        assert_eq!(insts[0].operands[1], ISelOperand::Imm(-3));
        assert_eq!(insts[1].opcode, AArch64Opcode::SubRR, "Negative min_val uses SubRR");

        // Jump table should have 5 entries
        let adr_inst = insts.iter().find(|i| i.opcode == AArch64Opcode::Adr).unwrap();
        let jt_idx = if let ISelOperand::JumpTableIndex(idx) = &adr_inst.operands[1] {
            *idx
        } else {
            panic!("ADR operand[1] should be JumpTableIndex, got {:?}", adr_inst.operands[1]);
        };
        let jt = &func.jump_tables[jt_idx as usize];
        assert_eq!(jt.min_val, -3);
        assert_eq!(jt.targets.len(), 5);
    }

    #[test]
    fn jump_table_min_val_zero_no_sub() {
        // min_val=0 should use MovR, not SUB
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (0, Block(1)),
            (1, Block(2)),
            (2, Block(3)),
            (3, Block(4)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // Should NOT have SubRI or SubRR
        let has_sub = insts.iter().any(|i| {
            i.opcode == AArch64Opcode::SubRI || i.opcode == AArch64Opcode::SubRR
        });
        assert!(!has_sub, "min_val=0 should not emit SUB");

        // Should start with MovR
        assert_eq!(insts[0].opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn jump_table_large_min_val() {
        // min_val=0x2000 > 0xFFF -> Movz + SubRR
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (0x2000, Block(1)),
            (0x2001, Block(2)),
            (0x2002, Block(3)),
            (0x2003, Block(4)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // min_val=0x2000 > 0xFFF: Movz to materialize, then SubRR
        assert_eq!(insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(insts[0].operands[1], ISelOperand::Imm(0x2000));
        assert_eq!(insts[1].opcode, AArch64Opcode::SubRR);
    }

    #[test]
    fn jump_table_small_positive_min_val() {
        // min_val=5, fits in imm12 -> SubRI
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (5, Block(1)),
            (6, Block(2)),
            (7, Block(3)),
            (8, Block(4)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // min_val=5 fits in imm12: SubRI directly
        assert_eq!(insts[0].opcode, AArch64Opcode::SubRI);
        assert_eq!(insts[0].operands[2], ISelOperand::Imm(5));
    }

    #[test]
    fn jump_table_successors_correct() {
        // Verify all target blocks are recorded as successors
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let cases = vec![
            (0, Block(1)),
            (1, Block(2)),
            (2, Block(3)),
            (3, Block(4)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let succs = &func.blocks[&entry].successors;
        for i in 1..=4 {
            assert!(
                succs.contains(&Block(i)),
                "Block({}) should be a successor",
                i
            );
        }
        assert!(
            succs.contains(&default),
            "Default block should be a successor"
        );
    }

    #[test]
    fn jump_table_large_range_uses_reg_cmp() {
        // range > 0xFFF -> CmpRR instead of CmpRI
        let (mut func, mut next_block) = make_test_func();
        let entry = Block(0);
        func.ensure_block(entry);
        for i in 1..=5 {
            func.ensure_block(Block(i));
        }
        let default = Block(5);

        let selector = ISelOperand::VReg(VReg::new(0, RegClass::Gpr64));
        // range = 0x5000 - 0 = 0x5000 > 0xFFF
        let cases = vec![
            (0, Block(1)),
            (0x1000, Block(2)),
            (0x3000, Block(3)),
            (0x5000, Block(4)),
        ];

        emit_jump_table(
            &mut func,
            &mut next_block,
            &selector,
            false,
            &cases,
            default,
            entry,
        );

        let insts = &func.blocks[&entry].insts;

        // range=0x5000 > 0xFFF: Movz to materialize range, then CmpRR
        // After MovR (min=0), next should be Movz for range, then CmpRR
        let has_cmp_rr = insts.iter().any(|i| i.opcode == AArch64Opcode::CmpRR);
        assert!(has_cmp_rr, "Range > 0xFFF should use CmpRR for range check");

        // Should NOT have CmpRI for the range check (may have it elsewhere)
        // The only CmpRI would be if range fit, but here it doesn't
        let cmp_ri_count = insts.iter().filter(|i| i.opcode == AArch64Opcode::CmpRI).count();
        assert_eq!(cmp_ri_count, 0, "Range > 0xFFF should not use CmpRI");
    }
}

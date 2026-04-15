// llvm2-opt - Instruction scheduling
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Pre-register-allocation instruction scheduling for AArch64.
//!
//! Reorders instructions within a basic block to maximize instruction-level
//! parallelism (ILP) and minimize pipeline stalls on wide-dispatch cores
//! (Apple M-series: 8-wide decode, 6 ALU + 2 LD/ST + 4 FP/NEON units).
//!
//! # Algorithm
//!
//! List scheduling with a critical-path priority heuristic:
//!
//! 1. **Build DAG**: Construct a dependency graph from data dependencies (RAW),
//!    memory ordering (conservative store-load/store-store/load-store chains),
//!    and control dependencies (terminators depend on all prior instructions).
//!
//! 2. **Compute priorities**: For each node, compute the longest path to any
//!    exit node (critical-path length). Higher priority = longer remaining
//!    critical path = should be scheduled earlier.
//!
//! 3. **Schedule**: Maintain a ready set. At each cycle, pick the highest-
//!    priority ready node, schedule it, and update dependents.
//!
//! # Latency Model
//!
//! Approximate Apple M-series (Firestorm) latencies:
//!
//! | Category | Latency | Port |
//! |----------|---------|------|
//! | ALU (add, sub, logic, shift, move, cmp, csel) | 1 cycle | IntAlu |
//! | MUL (mul, msub, smull, umull) | 3 cycles | IntMul |
//! | DIV (sdiv, udiv) | 10 cycles | IntDiv |
//! | Load (ldr, ldp, ldrb, etc.) | 4 cycles | LoadStore |
//! | Store (str, stp, strb, etc.) | 1 cycle | LoadStore |
//! | Branch/Ret | 1 cycle | Branch |
//! | FP arith (fadd, fsub, fmul, fdiv, fcvt) | 3 cycles | FpAlu |
//!
//! Reference: Dougall Johnson, "Apple M1 Firestorm Microarchitecture"

use std::collections::{HashMap, HashSet};

use llvm2_ir::{AArch64Opcode, BlockId, InstFlags, InstId, MachFunction, MachOperand};

use crate::effects::inst_produces_value;
use crate::pass_manager::MachinePass;

// ---------------------------------------------------------------------------
// Execution port model
// ---------------------------------------------------------------------------

/// Execution port classification for Apple M-series cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPort {
    /// Integer ALU (6 units on M1 Firestorm).
    IntAlu,
    /// Integer multiply (2 complex-integer units).
    IntMul,
    /// Integer divide (1 unit, fully pipelined for throughput but high latency).
    IntDiv,
    /// Load/store unit (2 units).
    LoadStore,
    /// Branch unit (1 unit).
    Branch,
    /// Floating-point / NEON ALU (4 units: 2 FADD + 2 FMUL).
    FpAlu,
}

// ---------------------------------------------------------------------------
// Latency model
// ---------------------------------------------------------------------------

/// Returns `(latency_cycles, execution_port)` for an AArch64 opcode.
///
/// Latencies are approximate for Apple M1 Firestorm core.
pub fn opcode_latency(opcode: AArch64Opcode) -> (u32, ExecutionPort) {
    use AArch64Opcode::*;
    match opcode {
        // Integer ALU: 1 cycle
        AddRR | AddRI | SubRR | SubRI | Neg => (1, ExecutionPort::IntAlu),
        AndRR | AndRI | OrrRR | OrrRI | EorRR | EorRI | OrnRR | BicRR => {
            (1, ExecutionPort::IntAlu)
        }
        LslRR | LsrRR | AsrRR | LslRI | LsrRI | AsrRI => (1, ExecutionPort::IntAlu),
        CmpRR | CmpRI | CMPWrr | CMPXrr | CMPWri | CMPXri | Tst => {
            (1, ExecutionPort::IntAlu)
        }
        Csel | CSet | Csinc | Csinv | Csneg => (1, ExecutionPort::IntAlu),
        MovR | MovI | Movz | Movn | Movk | MOVWrr | MOVXrr | MOVZWi | MOVZXi => {
            (1, ExecutionPort::IntAlu)
        }
        Sxtw | Uxtw | Sxtb | Sxth | Ubfm | Sbfm | Bfm => (1, ExecutionPort::IntAlu),
        Adrp | AddPCRel => (1, ExecutionPort::IntAlu),
        AddsRR | AddsRI | SubsRR | SubsRI => (1, ExecutionPort::IntAlu),

        // Integer multiply: 3 cycles
        MulRR | Msub | Smull | Umull => (3, ExecutionPort::IntMul),

        // Integer divide: 10 cycles
        SDiv | UDiv => (10, ExecutionPort::IntDiv),

        // Loads: 4 cycles (L1 hit)
        LdrRI | LdrbRI | LdrhRI | LdrsbRI | LdrshRI | LdrRO | LdrLiteral | LdpRI
        | LdpPostIndex | LdrGot | LdrTlvp => (4, ExecutionPort::LoadStore),

        // Stores: 1 cycle (non-blocking dispatch)
        StrRI | StrbRI | StrhRI | StrRO | StpRI | StpPreIndex | STRWui | STRXui
        | STRSui | STRDui => (1, ExecutionPort::LoadStore),

        // Stack allocation pseudo
        StackAlloc => (1, ExecutionPort::IntAlu),

        // Branches
        B | BCond | Bcc | Cbz | Cbnz | Tbz | Tbnz | Br => (1, ExecutionPort::Branch),

        // Calls / return
        Bl | Blr | BL | BLR | Ret => (1, ExecutionPort::Branch),

        // Floating-point arithmetic: 3 cycles
        FaddRR | FsubRR | FmulRR | FdivRR | FnegRR | Fcmp => (3, ExecutionPort::FpAlu),
        FcvtzsRR | FcvtzuRR | ScvtfRR | UcvtfRR => (3, ExecutionPort::FpAlu),
        FcvtSD | FcvtDS => (3, ExecutionPort::FpAlu),
        FmovGprFpr | FmovFprGpr | FmovImm => (1, ExecutionPort::FpAlu),

        // Trap pseudo-instructions: treated as branches
        TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => {
            (1, ExecutionPort::Branch)
        }

        // Reference counting: memory-like
        Retain | Release => (1, ExecutionPort::LoadStore),

        // Pseudo-instructions
        Phi | Copy | Nop => (1, ExecutionPort::IntAlu),
    }
}

// ---------------------------------------------------------------------------
// Schedule node and DAG
// ---------------------------------------------------------------------------

/// A node in the scheduling dependency graph.
#[derive(Debug, Clone)]
pub struct ScheduleNode {
    /// The instruction this node represents.
    pub inst_id: InstId,
    /// Execution latency in cycles.
    pub latency: u32,
    /// Which execution port this instruction uses.
    pub port: ExecutionPort,
    /// Indices of nodes this node depends on (predecessors).
    pub deps: Vec<usize>,
    /// Indices of nodes that depend on this node (successors).
    pub rev_deps: Vec<usize>,
    /// Earliest cycle this node can start (computed during scheduling).
    pub earliest_start: u32,
    /// Priority: longest path from this node to any exit (critical path).
    pub priority: u32,
    /// Whether this node has been scheduled.
    pub scheduled: bool,
}

/// Dependency graph for instruction scheduling within a basic block.
#[derive(Debug, Clone)]
pub struct ScheduleDAG {
    /// Nodes indexed by position in the original block instruction list.
    pub nodes: Vec<ScheduleNode>,
}

impl ScheduleDAG {
    /// Compute critical-path priorities (longest path to exit) for all nodes.
    ///
    /// Uses reverse topological traversal: start from nodes with no successors
    /// and propagate backward.
    fn compute_priorities(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // Initialize: nodes with no successors have priority = own latency.
        for i in 0..n {
            if self.nodes[i].rev_deps.is_empty() {
                self.nodes[i].priority = self.nodes[i].latency;
            }
        }

        // Iterate until stable (simple relaxation; DAG guarantees convergence).
        let mut changed = true;
        while changed {
            changed = false;
            for i in (0..n).rev() {
                let current_priority = self.nodes[i].priority;
                let latency = self.nodes[i].latency;
                let rev_deps = self.nodes[i].rev_deps.clone();
                for &succ in &rev_deps {
                    let new_priority = latency + self.nodes[succ].priority;
                    if new_priority > current_priority {
                        self.nodes[i].priority = new_priority;
                        changed = true;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DAG construction
// ---------------------------------------------------------------------------

/// Build a scheduling dependency graph for one basic block.
///
/// Dependency types:
/// - **Data (RAW)**: instruction B uses a VReg defined by instruction A.
/// - **Memory ordering**: conservative ordering between memory operations
///   (store-store, store-load, load-store are all ordered; load-load can reorder).
/// - **Control**: terminators depend on all prior non-terminator instructions.
/// - **Side-effect ordering**: instructions with `HAS_SIDE_EFFECTS` are ordered
///   relative to each other.
pub fn build_dag(func: &MachFunction, block_id: BlockId) -> ScheduleDAG {
    let block = func.block(block_id);
    let inst_ids = &block.insts;
    let n = inst_ids.len();

    // Create nodes.
    let mut nodes: Vec<ScheduleNode> = Vec::with_capacity(n);
    for &inst_id in inst_ids {
        let inst = func.inst(inst_id);
        let (latency, port) = opcode_latency(inst.opcode);
        nodes.push(ScheduleNode {
            inst_id,
            latency,
            port,
            deps: Vec::new(),
            rev_deps: Vec::new(),
            earliest_start: 0,
            priority: 0,
            scheduled: false,
        });
    }

    // Helper: add edge from `from` to `to` (from must execute before to).
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    let add_edge = |from: usize, to: usize, edges: &mut HashSet<(usize, usize)>| {
        if from != to && edges.insert((from, to)) {
            // edges set prevents duplicates; actual deps/rev_deps updated below
        }
    };

    // Build VReg def map: vreg_id -> node index that defines it.
    let mut def_map: HashMap<u32, usize> = HashMap::new();
    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        let inst = func.inst(inst_id);
        if inst_produces_value(inst) {
            if let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                def_map.insert(vreg.id, idx);
            }
        }
    }

    // 1. Data dependencies (RAW): for each use operand, add edge from def.
    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        let inst = func.inst(inst_id);
        let use_start = if inst_produces_value(inst) { 1 } else { 0 };
        for operand in &inst.operands[use_start..] {
            if let MachOperand::VReg(vreg) = operand {
                if let Some(&def_idx) = def_map.get(&vreg.id) {
                    add_edge(def_idx, idx, &mut edges);
                }
            }
        }
    }

    // 2. Memory dependencies: order memory operations conservatively.
    //    - Load-Load: no dependency (can reorder freely).
    //    - Store-Store: WAW dependency (must maintain order).
    //    - Store-Load: RAW dependency (load after store).
    //    - Load-Store: WAR dependency (store after load).
    let mut last_store: Option<usize> = None;
    let mut last_loads: Vec<usize> = Vec::new();

    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        let inst = func.inst(inst_id);
        let flags = inst.flags;

        let is_load = flags.contains(InstFlags::READS_MEMORY)
            && !flags.contains(InstFlags::WRITES_MEMORY);
        let is_store = flags.contains(InstFlags::WRITES_MEMORY);
        let is_call = flags.contains(InstFlags::IS_CALL);

        if is_call {
            // Calls are full barriers: depend on all prior memory ops.
            if let Some(s) = last_store {
                add_edge(s, idx, &mut edges);
            }
            for &l in &last_loads {
                add_edge(l, idx, &mut edges);
            }
            last_store = Some(idx);
            last_loads.clear();
        } else if is_store {
            // Store depends on prior store (WAW) and prior loads (WAR).
            if let Some(s) = last_store {
                add_edge(s, idx, &mut edges);
            }
            for &l in &last_loads {
                add_edge(l, idx, &mut edges);
            }
            last_store = Some(idx);
            last_loads.clear();
        } else if is_load {
            // Load depends on prior store (RAW), but NOT on prior loads.
            if let Some(s) = last_store {
                add_edge(s, idx, &mut edges);
            }
            last_loads.push(idx);
        }
    }

    // 3. Side-effect ordering: instructions with HAS_SIDE_EFFECTS that are not
    //    already covered by memory deps are ordered relative to each other.
    let mut last_side_effect: Option<usize> = None;
    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        let inst = func.inst(inst_id);
        if inst.flags.contains(InstFlags::HAS_SIDE_EFFECTS) {
            if let Some(prev) = last_side_effect {
                add_edge(prev, idx, &mut edges);
            }
            last_side_effect = Some(idx);
        }
    }

    // 4. Control dependencies: terminators depend on all prior instructions.
    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        let inst = func.inst(inst_id);
        if inst.flags.contains(InstFlags::IS_TERMINATOR) {
            for prior in 0..idx {
                add_edge(prior, idx, &mut edges);
            }
        }
    }

    // Populate deps/rev_deps from the edge set.
    for &(from, to) in &edges {
        nodes[to].deps.push(from);
        nodes[from].rev_deps.push(to);
    }

    let mut dag = ScheduleDAG { nodes };
    dag.compute_priorities();
    dag
}

// ---------------------------------------------------------------------------
// List scheduling
// ---------------------------------------------------------------------------

/// List scheduling: produce a new instruction order that minimizes stalls.
///
/// Uses a priority queue (sorted ready set) with critical-path heuristic.
/// At each step, picks the ready node with the highest priority (longest
/// remaining critical path).
pub fn schedule_list(dag: &mut ScheduleDAG) -> Vec<InstId> {
    let n = dag.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut scheduled_order: Vec<InstId> = Vec::with_capacity(n);
    let mut remaining_deps: Vec<usize> = dag.nodes.iter().map(|node| node.deps.len()).collect();
    let mut cycle: u32 = 0;

    while scheduled_order.len() < n {
        // Collect ready nodes: all deps satisfied and earliest_start <= cycle.
        let mut ready: Vec<usize> = Vec::new();
        for i in 0..n {
            if !dag.nodes[i].scheduled && remaining_deps[i] == 0 && dag.nodes[i].earliest_start <= cycle {
                ready.push(i);
            }
        }

        if ready.is_empty() {
            // No node ready at this cycle — advance to the earliest available.
            let mut min_start = u32::MAX;
            for i in 0..n {
                if !dag.nodes[i].scheduled && remaining_deps[i] == 0 {
                    min_start = min_start.min(dag.nodes[i].earliest_start);
                }
            }
            if min_start == u32::MAX {
                // All remaining nodes have unsatisfied deps — advance cycle.
                // This shouldn't happen in a well-formed DAG.
                cycle += 1;
                continue;
            }
            cycle = min_start;
            continue;
        }

        // Pick the highest-priority ready node.
        ready.sort_by(|&a, &b| {
            dag.nodes[b]
                .priority
                .cmp(&dag.nodes[a].priority)
                .then_with(|| a.cmp(&b)) // tie-break: original order
        });

        let best = ready[0];
        dag.nodes[best].scheduled = true;
        dag.nodes[best].earliest_start = cycle;
        scheduled_order.push(dag.nodes[best].inst_id);

        // Update dependents: their earliest_start is at least (cycle + latency).
        let finish = cycle + dag.nodes[best].latency;
        let rev_deps = dag.nodes[best].rev_deps.clone();
        for &succ in &rev_deps {
            remaining_deps[succ] -= 1;
            if dag.nodes[succ].earliest_start < finish {
                dag.nodes[succ].earliest_start = finish;
            }
        }

        cycle += 1;
    }

    scheduled_order
}

// ---------------------------------------------------------------------------
// Block and function scheduling
// ---------------------------------------------------------------------------

/// Schedule one basic block: build DAG, run list scheduling, reorder instructions.
///
/// Returns true if the instruction order changed.
pub fn schedule_block(func: &mut MachFunction, block_id: BlockId) -> bool {
    let block = func.block(block_id);
    if block.insts.len() <= 1 {
        return false;
    }

    let original_order: Vec<InstId> = block.insts.clone();
    let mut dag = build_dag(func, block_id);
    let new_order = schedule_list(&mut dag);

    if new_order == original_order {
        return false;
    }

    let block_mut = func.block_mut(block_id);
    block_mut.insts = new_order;
    true
}

/// Schedule all basic blocks in a function.
///
/// Returns true if any block was reordered.
pub fn schedule_function(func: &mut MachFunction) -> bool {
    let mut changed = false;
    let block_ids: Vec<BlockId> = func.block_order.clone();
    for block_id in block_ids {
        if schedule_block(func, block_id) {
            changed = true;
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// MachinePass implementation
// ---------------------------------------------------------------------------

/// Instruction scheduling pass for AArch64.
///
/// Reorders instructions within each basic block to maximize ILP and
/// minimize pipeline stalls. Runs as a pre-register-allocation pass.
pub struct InstructionScheduler;

impl MachinePass for InstructionScheduler {
    fn name(&self) -> &str {
        "instruction-scheduler"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        schedule_function(func)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_sched".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    // ---- Latency model tests ----

    #[test]
    fn test_latency_alu() {
        let (lat, port) = opcode_latency(AArch64Opcode::AddRR);
        assert_eq!(lat, 1);
        assert_eq!(port, ExecutionPort::IntAlu);
    }

    #[test]
    fn test_latency_mul() {
        let (lat, port) = opcode_latency(AArch64Opcode::MulRR);
        assert_eq!(lat, 3);
        assert_eq!(port, ExecutionPort::IntMul);
    }

    #[test]
    fn test_latency_div() {
        let (lat, port) = opcode_latency(AArch64Opcode::SDiv);
        assert_eq!(lat, 10);
        assert_eq!(port, ExecutionPort::IntDiv);
    }

    #[test]
    fn test_latency_load() {
        let (lat, port) = opcode_latency(AArch64Opcode::LdrRI);
        assert_eq!(lat, 4);
        assert_eq!(port, ExecutionPort::LoadStore);
    }

    #[test]
    fn test_latency_store() {
        let (lat, port) = opcode_latency(AArch64Opcode::StrRI);
        assert_eq!(lat, 1);
        assert_eq!(port, ExecutionPort::LoadStore);
    }

    #[test]
    fn test_latency_branch() {
        let (lat, port) = opcode_latency(AArch64Opcode::B);
        assert_eq!(lat, 1);
        assert_eq!(port, ExecutionPort::Branch);
    }

    #[test]
    fn test_latency_fp() {
        let (lat, port) = opcode_latency(AArch64Opcode::FaddRR);
        assert_eq!(lat, 3);
        assert_eq!(port, ExecutionPort::FpAlu);
    }

    // ---- Empty and trivial block tests ----

    #[test]
    fn test_empty_block() {
        let mut func = MachFunction::new(
            "empty".to_string(),
            Signature::new(vec![], vec![]),
        );
        let mut sched = InstructionScheduler;
        assert!(!sched.run(&mut func));
    }

    #[test]
    fn test_single_instruction() {
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ret]);

        let mut sched = InstructionScheduler;
        assert!(!sched.run(&mut func));
    }

    // ---- Data dependency tests ----

    #[test]
    fn test_data_dependency_respected() {
        // v1 = add v0, #1    (inst0)
        // v2 = add v1, #2    (inst1, depends on inst0)
        // ret                 (inst2)
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, add2, ret]);

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        // add1 must come before add2 (data dependency v1).
        let pos_add1 = order.iter().position(|&id| id == InstId(0)).unwrap();
        let pos_add2 = order.iter().position(|&id| id == InstId(1)).unwrap();
        assert!(
            pos_add1 < pos_add2,
            "add1 (def v1) must precede add2 (use v1)"
        );

        // ret must be last.
        assert_eq!(
            *order.last().unwrap(),
            InstId(2),
            "ret must be last"
        );
    }

    // ---- Independent instructions reordering test ----

    #[test]
    fn test_independent_instructions_reordered() {
        // Original order:
        //   v1 = add v0, #1      (1 cycle, low priority)
        //   v2 = mul v3, v4      (3 cycles, high priority — longer critical path)
        //   v5 = add v2, #1      (uses v2)
        //   ret
        //
        // The scheduler should prefer to schedule mul first because it has
        // higher latency and its dependent (add v2, #1) is on the critical path.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(3), vreg(4)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(5), vreg(2), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, mul, add2, ret]);

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        // mul (InstId(1)) should be scheduled before or at same position as
        // add1 (InstId(0)) because it has higher priority (3+1 > 1).
        let pos_mul = order.iter().position(|&id| id == InstId(1)).unwrap();
        let pos_add1 = order.iter().position(|&id| id == InstId(0)).unwrap();
        assert!(
            pos_mul <= pos_add1,
            "mul should be scheduled before add1 (critical path), got mul@{} add1@{}",
            pos_mul,
            pos_add1,
        );

        // add2 must come after mul (data dep on v2).
        let pos_add2 = order.iter().position(|&id| id == InstId(2)).unwrap();
        assert!(pos_mul < pos_add2, "add2 depends on mul");

        // ret must be last.
        assert_eq!(*order.last().unwrap(), InstId(3));
    }

    // ---- Memory dependency tests ----

    #[test]
    fn test_memory_dependency_prevents_reorder() {
        // str v0, [sp, #0]    (store, inst0)
        // v1 = ldr [v2, #0]   (load, inst1 — must come after store)
        // ret                  (inst2)
        let store = MachInst::new(AArch64Opcode::StrRI, vec![vreg(0), vreg(10), imm(0)]);
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![store, load, ret]);

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        // Store must come before load (conservative memory ordering).
        let pos_store = order.iter().position(|&id| id == InstId(0)).unwrap();
        let pos_load = order.iter().position(|&id| id == InstId(1)).unwrap();
        assert!(
            pos_store < pos_load,
            "store must precede load (memory dependency)"
        );
    }

    #[test]
    fn test_store_store_ordering_preserved() {
        // str v0, [v1, #0]    (inst0)
        // str v2, [v3, #8]    (inst1, must come after inst0)
        // ret                  (inst2)
        let store1 = MachInst::new(AArch64Opcode::StrRI, vec![vreg(0), vreg(1), imm(0)]);
        let store2 = MachInst::new(AArch64Opcode::StrRI, vec![vreg(2), vreg(3), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![store1, store2, ret]);

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        let pos_s1 = order.iter().position(|&id| id == InstId(0)).unwrap();
        let pos_s2 = order.iter().position(|&id| id == InstId(1)).unwrap();
        assert!(pos_s1 < pos_s2, "store-store ordering must be preserved");
    }

    #[test]
    fn test_load_load_can_reorder() {
        // Two independent loads with no intervening store can be reordered.
        // v1 = ldr [v0, #0]    (inst0, 4 cycles)
        // v3 = ldr [v2, #8]    (inst1, 4 cycles)
        // v4 = add v1, v3      (inst2, uses both)
        // ret                   (inst3)
        let load1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let load2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(2), imm(8)]);
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(4), vreg(1), vreg(3)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![load1, load2, add, ret]);

        // Build DAG and verify loads have no dependency on each other.
        let dag = build_dag(&func, func.entry);

        // Node 0 (load1) and node 1 (load2) should NOT have edges between them.
        assert!(
            !dag.nodes[0].rev_deps.contains(&1),
            "load1 should not have dep edge to load2"
        );
        assert!(
            !dag.nodes[1].deps.contains(&0),
            "load2 should not depend on load1"
        );
    }

    // ---- Terminator tests ----

    #[test]
    fn test_terminator_stays_last() {
        // v0 = mov #42         (inst0)
        // v1 = mul v2, v3      (inst1, 3 cycles)
        // b.cond <target>      (inst2, terminator — must be last)
        let mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(42)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(1), vreg(2), vreg(3)]);
        let branch = MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(llvm2_ir::BlockId(1))],
        );
        let mut func = make_func_with_insts(vec![mov, mul, branch]);
        // Need a second block for the branch target.
        let _bb1 = func.create_block();

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        // Branch must be last.
        assert_eq!(
            *order.last().unwrap(),
            InstId(2),
            "branch terminator must remain last"
        );
    }

    // ---- Latency hiding test ----

    #[test]
    fn test_latency_model_produces_better_schedule() {
        // Input order (suboptimal):
        //   v1 = ldr [v0, #0]     (inst0, 4 cycles)
        //   v2 = add v1, #1       (inst1, depends on inst0)
        //   v3 = mov #99          (inst2, independent)
        //   ret                    (inst3)
        //
        // Optimal schedule: ldr, mov, add, ret
        // The mov can execute during the load's latency.
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(3), imm(99)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![load, add, mov, ret]);

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        // Load should be first (highest critical path: 4 + 1 = 5).
        assert_eq!(order[0], InstId(0), "load should be first");

        // Mov (InstId(2)) should be scheduled before add (InstId(1))
        // because add can't start until cycle 4 (waiting on load).
        let pos_mov = order.iter().position(|&id| id == InstId(2)).unwrap();
        let pos_add = order.iter().position(|&id| id == InstId(1)).unwrap();
        assert!(
            pos_mov < pos_add,
            "mov should be scheduled during load latency, before add"
        );

        // Ret must be last.
        assert_eq!(*order.last().unwrap(), InstId(3));
    }

    // ---- DAG construction test ----

    #[test]
    fn test_build_dag_data_deps() {
        // v1 = add v0, #1     (node 0, def v1)
        // v2 = sub v1, #2     (node 1, uses v1 -> dep on node 0)
        // ret                  (node 2)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(2), vreg(1), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add, sub, ret]);

        let dag = build_dag(&func, func.entry);

        assert_eq!(dag.nodes.len(), 3);
        // Node 1 depends on node 0 (data: v1).
        assert!(dag.nodes[1].deps.contains(&0));
        // Node 0 has node 1 as reverse dep.
        assert!(dag.nodes[0].rev_deps.contains(&1));
    }

    #[test]
    fn test_call_is_memory_barrier() {
        // v0 = ldr [v1, #0]   (inst0, load)
        // bl <func>            (inst1, call — barrier)
        // v2 = ldr [v3, #0]   (inst2, load after call)
        // ret                  (inst3)
        let load1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(0), vreg(1), imm(0)]);
        let call = MachInst::new(AArch64Opcode::Bl, vec![MachOperand::Imm(0)]);
        let load2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(3), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![load1, call, load2, ret]);

        let dag = build_dag(&func, func.entry);

        // load1 (node 0) -> call (node 1): load before call.
        assert!(dag.nodes[1].deps.contains(&0), "call depends on prior load");
        // call (node 1) -> load2 (node 2): load after call.
        assert!(
            dag.nodes[2].deps.contains(&1),
            "post-call load depends on call"
        );
    }

    // ---- Priority computation test ----

    #[test]
    fn test_priority_critical_path() {
        // v1 = mul v0, v0      (node 0, lat=3)
        // v2 = add v1, #1      (node 1, lat=1, dep on node 0)
        // ret                   (node 2, lat=1, dep on all)
        //
        // Critical path: mul(3) -> add(1) -> ret(1) = 5
        // Node 0 priority = 3 + 1 + 1 = 5
        // Node 1 priority = 1 + 1 = 2
        // Node 2 priority = 1
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(1), vreg(0), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![mul, add, ret]);

        let dag = build_dag(&func, func.entry);

        assert_eq!(dag.nodes[0].priority, 5, "mul has critical path 3+1+1=5");
        assert_eq!(dag.nodes[1].priority, 2, "add has critical path 1+1=2");
        assert_eq!(dag.nodes[2].priority, 1, "ret has priority 1");
    }

    // ---- Idempotency test ----

    #[test]
    fn test_scheduler_idempotent() {
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(0), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, add2, ret]);

        let mut sched = InstructionScheduler;
        sched.run(&mut func);

        let order1: Vec<InstId> = func.block(func.entry).insts.clone();

        // Second run should produce same order.
        let changed = sched.run(&mut func);
        let order2: Vec<InstId> = func.block(func.entry).insts.clone();

        assert_eq!(order1, order2, "scheduler should be idempotent");
        assert!(!changed, "second run should report no change");
    }
}

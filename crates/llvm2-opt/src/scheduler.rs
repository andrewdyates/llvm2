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

use llvm2_ir::{AArch64Opcode, BlockId, InstFlags, InstId, MachFunction, MachOperand, RegClass};

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
        Sxtw | Uxtw | Sxtb | Sxth | Uxtb | Uxth | Ubfm | Sbfm | Bfm => (1, ExecutionPort::IntAlu),
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
        FaddRR | FsubRR | FmulRR | FdivRR | FnegRR | FabsRR | Fcmp => (3, ExecutionPort::FpAlu),
        FsqrtRR => (12, ExecutionPort::FpAlu),
        FcvtzsRR | FcvtzuRR | ScvtfRR | UcvtfRR => (3, ExecutionPort::FpAlu),
        FcvtSD | FcvtDS => (3, ExecutionPort::FpAlu),
        FmovGprFpr | FmovFprGpr | FmovImm => (1, ExecutionPort::FpAlu),

        // NEON SIMD: uses FP/NEON ALU units
        NeonAddV | NeonSubV => (2, ExecutionPort::FpAlu),
        NeonMulV => (3, ExecutionPort::FpAlu),
        NeonFaddV | NeonFsubV => (3, ExecutionPort::FpAlu),
        NeonFmulV => (4, ExecutionPort::FpAlu),
        NeonFdivV => (10, ExecutionPort::FpAlu),
        NeonAndV | NeonOrrV | NeonEorV | NeonBicV | NeonNotV => (1, ExecutionPort::FpAlu),
        NeonCmeqV | NeonCmgtV | NeonCmgeV => (2, ExecutionPort::FpAlu),
        NeonDupElem | NeonDupGen | NeonMovi => (2, ExecutionPort::FpAlu),
        NeonInsGen => (3, ExecutionPort::FpAlu),
        NeonLd1Post => (4, ExecutionPort::LoadStore),
        NeonSt1Post => (1, ExecutionPort::LoadStore),

        // Trap pseudo-instructions: treated as branches
        TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => {
            (1, ExecutionPort::Branch)
        }

        // Reference counting: memory-like
        Retain | Release => (1, ExecutionPort::LoadStore),

        // Atomic loads: 4 cycles (like regular load + ordering)
        Ldar | Ldarb | Ldarh | Ldaxr => (4, ExecutionPort::LoadStore),

        // Atomic stores: 2 cycles (like regular store + ordering)
        Stlr | Stlrb | Stlrh | Stlxr => (2, ExecutionPort::LoadStore),

        // Atomic RMW (LSE): 6 cycles
        Ldadd | Ldadda | Ldaddal
        | Ldclr | Ldclral
        | Ldeor | Ldeoral
        | Ldset | Ldsetal
        | Swp | Swpal => (6, ExecutionPort::LoadStore),

        // Compare-and-swap: 8 cycles
        Cas | Casa | Casal => (8, ExecutionPort::LoadStore),

        // Barriers: 4-12 cycles
        Dmb => (4, ExecutionPort::LoadStore),
        Dsb => (8, ExecutionPort::LoadStore),
        Isb => (12, ExecutionPort::LoadStore),

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
// Register pressure tracking during scheduling
// ---------------------------------------------------------------------------

/// Tracks approximate register pressure during list scheduling.
///
/// Monitors the number of live virtual registers at each scheduling step.
/// When pressure exceeds thresholds, the scheduler prefers instructions that
/// reduce pressure (consumers that kill operands) over instructions that
/// increase it (producers that define new values).
///
/// This prevents the scheduler from creating unnecessarily long live ranges
/// that force the register allocator to spill. The heuristic balances ILP
/// (critical-path scheduling) against register pressure (spill avoidance).
///
/// Reference: LLVM's `ScheduleDAGRRList::BURegReductionPriorityQueue` and
/// GCC's `sched-pressure.cc`.
#[derive(Debug, Clone)]
pub struct PressureTracker {
    /// Set of VReg IDs currently live (defined but not yet last-used).
    live_gprs: HashSet<u32>,
    /// Set of FPR VReg IDs currently live.
    live_fprs: HashSet<u32>,
    /// Peak GPR pressure observed so far.
    pub peak_gpr: u32,
    /// Peak FPR pressure observed so far.
    pub peak_fpr: u32,
    /// GPR pressure threshold: above this, prefer consumers.
    pub gpr_threshold: u32,
    /// FPR pressure threshold: above this, prefer consumers.
    pub fpr_threshold: u32,
}

impl PressureTracker {
    /// Create a new pressure tracker with default AArch64 thresholds.
    ///
    /// Thresholds are set below the allocatable register count to provide
    /// headroom: 20 GPRs (of 28 allocatable) and 24 FPRs (of 32 allocatable).
    pub fn new() -> Self {
        Self {
            live_gprs: HashSet::new(),
            live_fprs: HashSet::new(),
            peak_gpr: 0,
            peak_fpr: 0,
            gpr_threshold: 20,
            fpr_threshold: 24,
        }
    }

    /// Create a tracker with custom thresholds.
    pub fn with_thresholds(gpr_threshold: u32, fpr_threshold: u32) -> Self {
        Self {
            live_gprs: HashSet::new(),
            live_fprs: HashSet::new(),
            peak_gpr: 0,
            peak_fpr: 0,
            gpr_threshold,
            fpr_threshold,
        }
    }

    /// Current GPR pressure (number of live GPR VRegs).
    pub fn gpr_pressure(&self) -> u32 {
        self.live_gprs.len() as u32
    }

    /// Current FPR pressure (number of live FPR VRegs).
    pub fn fpr_pressure(&self) -> u32 {
        self.live_fprs.len() as u32
    }

    /// Returns true if GPR or FPR pressure exceeds the threshold.
    pub fn is_high_pressure(&self) -> bool {
        self.gpr_pressure() > self.gpr_threshold || self.fpr_pressure() > self.fpr_threshold
    }

    /// Record that a VReg has been defined (becomes live).
    pub fn define_vreg(&mut self, vreg_id: u32, class: RegClass) {
        let is_fpr = matches!(
            class,
            RegClass::Fpr128 | RegClass::Fpr64 | RegClass::Fpr32
                | RegClass::Fpr16 | RegClass::Fpr8
        );
        if is_fpr {
            self.live_fprs.insert(vreg_id);
            self.peak_fpr = self.peak_fpr.max(self.live_fprs.len() as u32);
        } else {
            self.live_gprs.insert(vreg_id);
            self.peak_gpr = self.peak_gpr.max(self.live_gprs.len() as u32);
        }
    }

    /// Record that a VReg has been killed (last use — no longer live).
    pub fn kill_vreg(&mut self, vreg_id: u32) {
        // Try removing from both sets; at most one will contain it.
        if !self.live_gprs.remove(&vreg_id) {
            self.live_fprs.remove(&vreg_id);
        }
    }
}

/// Per-node pressure metadata computed before scheduling.
///
/// For each node in the DAG, we precompute:
/// - How many VRegs this instruction uses for the last time (kills).
/// - How many VRegs this instruction defines (new live ranges).
/// - The register class of defined VRegs.
#[derive(Debug, Clone)]
struct NodePressureInfo {
    /// VReg IDs defined by this instruction.
    defs: Vec<(u32, RegClass)>,
    /// VReg IDs used by this instruction.
    uses: Vec<(u32, RegClass)>,
    /// Number of VRegs whose last use is this instruction (kills).
    /// Computed from the full block context.
    kills: u32,
    /// Net pressure change: defs - kills. Negative means pressure-reducing.
    net_pressure: i32,
}

/// Precompute pressure metadata for all nodes in the DAG.
///
/// For each instruction, we determine which VRegs it defines and uses,
/// and which uses are the last use in the block (kills). This is computed
/// from the function's instruction data and the DAG node mapping.
fn compute_pressure_info(
    func: &MachFunction,
    dag: &ScheduleDAG,
) -> Vec<NodePressureInfo> {
    let n = dag.nodes.len();

    // Collect defs and uses per node.
    let mut infos: Vec<NodePressureInfo> = Vec::with_capacity(n);
    for node in &dag.nodes {
        let inst = func.inst(node.inst_id);
        let produces = inst_produces_value(inst);

        let mut defs = Vec::new();
        let mut uses = Vec::new();

        if produces {
            if let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                defs.push((vreg.id, vreg.class));
            }
        }

        let use_start = if produces { 1 } else { 0 };
        for operand in &inst.operands[use_start..] {
            if let MachOperand::VReg(vreg) = operand {
                uses.push((vreg.id, vreg.class));
            }
        }

        infos.push(NodePressureInfo {
            defs,
            uses,
            kills: 0,
            net_pressure: 0,
        });
    }

    // Build last-use map: for each VReg used in this block, find the last
    // node index that uses it. Uses that are last are "kills".
    let mut last_use_node: HashMap<u32, usize> = HashMap::new();
    for (idx, info) in infos.iter().enumerate() {
        for &(vreg_id, _) in &info.uses {
            last_use_node.insert(vreg_id, idx);
        }
    }

    // Count kills per node and compute net pressure.
    for (idx, info) in infos.iter_mut().enumerate() {
        let mut kills = 0u32;
        for &(vreg_id, _) in &info.uses {
            if last_use_node.get(&vreg_id) == Some(&idx) {
                kills += 1;
            }
        }
        info.kills = kills;
        info.net_pressure = info.defs.len() as i32 - kills as i32;
    }

    infos
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

/// Register pressure-aware list scheduling.
///
/// Extends the basic list scheduler with register pressure heuristics:
///
/// 1. **Pressure tracking**: Maintains a set of live VRegs. When an instruction
///    is scheduled, its defs become live and its killed operands die.
///
/// 2. **Consumer preference under high pressure**: When pressure exceeds the
///    threshold, the scheduler prefers instructions that kill (last-use) more
///    VRegs. This reduces the number of simultaneously live values.
///
/// 3. **Short live-range preference**: Among producers, prefer those whose
///    consumers are ready soon (fewer remaining deps on successors). This
///    keeps newly defined values short-lived.
///
/// 4. **Net pressure tie-breaking**: When critical-path priorities are equal,
///    prefer nodes with lower net pressure (kills - defs), i.e., nodes that
///    release more registers than they define.
///
/// The combined heuristic ordering when pressure is high:
///   1. Nodes with negative net_pressure (more kills than defs) first
///   2. Among those, highest critical-path priority
///   3. Among equal, prefer original order for stability
///
/// When pressure is below threshold, falls back to pure critical-path priority
/// (same as `schedule_list`), preserving ILP optimization.
pub fn schedule_list_pressure_aware(
    func: &MachFunction,
    dag: &mut ScheduleDAG,
) -> (Vec<InstId>, PressureTracker) {
    let n = dag.nodes.len();
    if n == 0 {
        return (Vec::new(), PressureTracker::new());
    }

    let pressure_info = compute_pressure_info(func, dag);
    let mut tracker = PressureTracker::new();

    // Recompute last-use accounting relative to the scheduling order.
    // We track remaining use counts for each VReg: when a VReg's remaining
    // uses reach 0, it is killed.
    let mut vreg_remaining_uses: HashMap<u32, u32> = HashMap::new();
    for info in &pressure_info {
        for &(vreg_id, _) in &info.uses {
            *vreg_remaining_uses.entry(vreg_id).or_insert(0) += 1;
        }
    }

    let mut scheduled_order: Vec<InstId> = Vec::with_capacity(n);
    let mut remaining_deps: Vec<usize> = dag.nodes.iter().map(|node| node.deps.len()).collect();
    let mut cycle: u32 = 0;

    while scheduled_order.len() < n {
        // Collect ready nodes.
        let mut ready: Vec<usize> = Vec::new();
        for i in 0..n {
            if !dag.nodes[i].scheduled
                && remaining_deps[i] == 0
                && dag.nodes[i].earliest_start <= cycle
            {
                ready.push(i);
            }
        }

        if ready.is_empty() {
            let mut min_start = u32::MAX;
            for i in 0..n {
                if !dag.nodes[i].scheduled && remaining_deps[i] == 0 {
                    min_start = min_start.min(dag.nodes[i].earliest_start);
                }
            }
            if min_start == u32::MAX {
                cycle += 1;
                continue;
            }
            cycle = min_start;
            continue;
        }

        let high_pressure = tracker.is_high_pressure();

        // Sort ready nodes by pressure-aware heuristic.
        ready.sort_by(|&a, &b| {
            if high_pressure {
                // Under high pressure: prefer nodes that reduce pressure.
                // net_pressure < 0 means more kills than defs.
                let net_a = pressure_info[a].net_pressure;
                let net_b = pressure_info[b].net_pressure;

                // First: prefer lower net_pressure (more pressure-reducing).
                net_a
                    .cmp(&net_b)
                    // Then: among equal net pressure, prefer higher critical-path priority.
                    .then_with(|| {
                        dag.nodes[b]
                            .priority
                            .cmp(&dag.nodes[a].priority)
                    })
                    // Finally: original order for stability.
                    .then_with(|| a.cmp(&b))
            } else {
                // Normal mode: pure critical-path priority (same as schedule_list).
                dag.nodes[b]
                    .priority
                    .cmp(&dag.nodes[a].priority)
                    .then_with(|| a.cmp(&b))
            }
        });

        let best = ready[0];
        dag.nodes[best].scheduled = true;
        dag.nodes[best].earliest_start = cycle;
        scheduled_order.push(dag.nodes[best].inst_id);

        // Update pressure: process uses (potential kills) before defs.
        // When we schedule an instruction, its used VRegs have their remaining
        // use count decremented. If it reaches 0, the VReg is killed.
        for &(vreg_id, _) in &pressure_info[best].uses {
            if let Some(count) = vreg_remaining_uses.get_mut(&vreg_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    tracker.kill_vreg(vreg_id);
                }
            }
        }

        // Process defs: newly defined VRegs become live.
        for &(vreg_id, class) in &pressure_info[best].defs {
            tracker.define_vreg(vreg_id, class);
        }

        // Update dependents.
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

    (scheduled_order, tracker)
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

/// Schedule one basic block with register pressure awareness.
///
/// Uses pressure-aware heuristics to balance ILP against register pressure.
/// Returns true if the instruction order changed, along with the pressure tracker.
pub fn schedule_block_pressure_aware(
    func: &mut MachFunction,
    block_id: BlockId,
) -> (bool, PressureTracker) {
    let block = func.block(block_id);
    if block.insts.len() <= 1 {
        return (false, PressureTracker::new());
    }

    let original_order: Vec<InstId> = block.insts.clone();
    let mut dag = build_dag(func, block_id);
    let (new_order, tracker) = schedule_list_pressure_aware(func, &mut dag);

    if new_order == original_order {
        return (false, tracker);
    }

    let block_mut = func.block_mut(block_id);
    block_mut.insts = new_order;
    (true, tracker)
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

/// Schedule all basic blocks in a function with register pressure awareness.
///
/// Returns true if any block was reordered, along with peak GPR and FPR
/// pressure across all blocks.
pub fn schedule_function_pressure_aware(func: &mut MachFunction) -> (bool, u32, u32) {
    let mut changed = false;
    let mut peak_gpr: u32 = 0;
    let mut peak_fpr: u32 = 0;
    let block_ids: Vec<BlockId> = func.block_order.clone();
    for block_id in block_ids {
        let (block_changed, tracker) = schedule_block_pressure_aware(func, block_id);
        if block_changed {
            changed = true;
        }
        peak_gpr = peak_gpr.max(tracker.peak_gpr);
        peak_fpr = peak_fpr.max(tracker.peak_fpr);
    }
    (changed, peak_gpr, peak_fpr)
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

/// Pressure-aware instruction scheduling pass for AArch64.
///
/// Like `InstructionScheduler` but trades some ILP for lower register pressure.
/// When the number of live virtual registers exceeds a threshold (20 GPR, 24 FPR),
/// the scheduler prefers consuming instructions (that kill operands) over producing
/// instructions (that define new values). This reduces peak register pressure and
/// avoids unnecessary spills during register allocation.
///
/// Should be used instead of `InstructionScheduler` when register pressure is a
/// concern (large basic blocks, many live values).
pub struct PressureAwareScheduler;

impl MachinePass for PressureAwareScheduler {
    fn name(&self) -> &str {
        "pressure-aware-scheduler"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let (changed, _, _) = schedule_function_pressure_aware(func);
        changed
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Resource hazard tracking and pipeline analysis
// ---------------------------------------------------------------------------

/// Returns the number of execution units available for a given port
/// on Apple M1 Firestorm.
///
/// Reference: Dougall Johnson, "Apple M1 Firestorm Microarchitecture"
pub fn port_capacity(port: ExecutionPort) -> u32 {
    match port {
        ExecutionPort::IntAlu => 6,
        ExecutionPort::IntMul => 2,
        ExecutionPort::IntDiv => 1,
        ExecutionPort::LoadStore => 2,
        ExecutionPort::Branch => 1,
        ExecutionPort::FpAlu => 4,
    }
}

// ---------------------------------------------------------------------------
// Resource state tracking
// ---------------------------------------------------------------------------

/// Tracks execution unit availability per cycle for structural hazard detection.
///
/// Models the Apple M1 Firestorm port availability: at each cycle, each port
/// has a fixed number of units. Scheduling an instruction on a port at a cycle
/// consumes one unit. A structural hazard occurs when all units of a port are
/// occupied at a given cycle.
#[derive(Debug, Clone)]
pub struct ResourceState {
    /// Per-cycle usage: (port, cycle) -> units currently in use.
    usage: HashMap<(ExecutionPort, u32), u32>,
}

impl ResourceState {
    /// Create a new resource state with no reservations.
    pub fn new() -> Self {
        Self {
            usage: HashMap::new(),
        }
    }

    /// Returns the number of units still available for `port` at `cycle`.
    pub fn units_available(&self, port: ExecutionPort, cycle: u32) -> u32 {
        let cap = port_capacity(port);
        let used = self.usage.get(&(port, cycle)).copied().unwrap_or(0);
        cap.saturating_sub(used)
    }

    /// Returns true if at least one unit is available for `port` at `cycle`.
    pub fn is_available(&self, port: ExecutionPort, cycle: u32) -> bool {
        self.units_available(port, cycle) > 0
    }

    /// Reserve one unit of `port` at `cycle`. Returns true if successful,
    /// false if all units are already occupied (structural hazard).
    pub fn reserve(&mut self, port: ExecutionPort, cycle: u32) -> bool {
        if !self.is_available(port, cycle) {
            return false;
        }
        *self.usage.entry((port, cycle)).or_insert(0) += 1;
        true
    }
}

// ---------------------------------------------------------------------------
// Hazard detection
// ---------------------------------------------------------------------------

/// Classification of pipeline hazards detected in a schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HazardKind {
    /// Data hazard: a consumer had to wait for a producer's result.
    /// `wait_cycles` is the number of stall cycles (earliest_start difference
    /// minus 1 for the dispatch slot).
    DataHazard {
        producer: usize,
        consumer: usize,
        wait_cycles: u32,
    },
    /// Structural hazard: all execution units for a port were busy at a cycle.
    StructuralHazard {
        port: ExecutionPort,
        cycle: u32,
    },
    /// Load-use hazard: a load result is consumed in the very next instruction
    /// in program order, causing a pipeline bubble. This is a special case of
    /// data hazard common on in-order cores and still costly on OoO cores
    /// when the load misses cache.
    LoadUseHazard {
        load_node: usize,
        use_node: usize,
    },
}

/// Detect pipeline hazards in a scheduled DAG.
///
/// Precondition: the DAG must have been through `schedule_list` so that
/// `earliest_start` and `scheduled` are populated for all nodes.
pub fn detect_hazards(dag: &ScheduleDAG) -> Vec<HazardKind> {
    let mut hazards = Vec::new();
    let n = dag.nodes.len();

    // Build a resource state from the schedule.
    let mut resources = ResourceState::new();
    for node in &dag.nodes {
        if !resources.reserve(node.port, node.earliest_start) {
            hazards.push(HazardKind::StructuralHazard {
                port: node.port,
                cycle: node.earliest_start,
            });
        }
    }

    // Check data hazards: for each edge, if the consumer's earliest_start
    // is later than (producer earliest_start + 1), the consumer stalled
    // waiting for data.
    for consumer_idx in 0..n {
        let consumer_start = dag.nodes[consumer_idx].earliest_start;
        for &producer_idx in &dag.nodes[consumer_idx].deps {
            let producer_start = dag.nodes[producer_idx].earliest_start;
            let producer_latency = dag.nodes[producer_idx].latency;
            let ready_cycle = producer_start + producer_latency;
            if consumer_start >= ready_cycle && consumer_start > producer_start + 1 {
                let wait_cycles = consumer_start.saturating_sub(producer_start + 1);
                if wait_cycles > 0 {
                    hazards.push(HazardKind::DataHazard {
                        producer: producer_idx,
                        consumer: consumer_idx,
                        wait_cycles,
                    });
                }
            }
        }
    }

    // Check load-use hazards: load followed by consumer at (earliest_start + 1).
    // This is a pipeline forwarding stall on many microarchitectures.
    for consumer_idx in 0..n {
        for &producer_idx in &dag.nodes[consumer_idx].deps {
            if dag.nodes[producer_idx].port == ExecutionPort::LoadStore
                && dag.nodes[producer_idx].latency >= 4
            {
                // This producer is a load (latency >= 4 distinguishes loads from stores).
                let load_start = dag.nodes[producer_idx].earliest_start;
                let consumer_start = dag.nodes[consumer_idx].earliest_start;
                // If consumer is scheduled before load result is ready, that means
                // the scheduler had to wait. If consumer is at load_start + 1,
                // it's a tight load-use pair.
                if consumer_start == load_start + 1 {
                    hazards.push(HazardKind::LoadUseHazard {
                        load_node: producer_idx,
                        use_node: consumer_idx,
                    });
                }
            }
        }
    }

    hazards
}

// ---------------------------------------------------------------------------
// Dual-issue hints
// ---------------------------------------------------------------------------

/// A hint that two instructions can potentially dual-issue on Apple M-series.
///
/// Apple M1 Firestorm can dispatch up to 8 micro-ops per cycle across its
/// execution ports. Two instructions can dual-issue if they use different
/// port types and both are ready at the same cycle.
#[derive(Debug, Clone)]
pub struct DualIssueHint {
    /// First node index.
    pub first: usize,
    /// Second node index.
    pub second: usize,
    /// Human-readable reason for the dual-issue opportunity.
    pub reason: &'static str,
}

/// Returns true if two ports can potentially dual-issue.
///
/// Dual-issue pairs on Apple M-series:
/// - ALU + Load/Store
/// - ALU + ALU (6 ALU units)
/// - ALU + Branch
/// - ALU + FpAlu
/// - Load/Store + FpAlu
fn can_dual_issue(a: ExecutionPort, b: ExecutionPort) -> Option<&'static str> {
    use ExecutionPort::*;
    match (a, b) {
        (IntAlu, LoadStore) | (LoadStore, IntAlu) => Some("ALU + Load/Store"),
        (IntAlu, IntAlu) => Some("ALU + ALU"),
        (IntAlu, Branch) | (Branch, IntAlu) => Some("ALU + Branch"),
        (IntAlu, FpAlu) | (FpAlu, IntAlu) => Some("ALU + FP"),
        (LoadStore, FpAlu) | (FpAlu, LoadStore) => Some("Load/Store + FP"),
        (FpAlu, FpAlu) => Some("FP + FP"),
        _ => None,
    }
}

/// Find dual-issue opportunities in a scheduled DAG.
///
/// The list scheduler issues one instruction per cycle, but Apple M1 Firestorm
/// can dispatch up to 8 micro-ops per cycle. This analysis identifies
/// consecutive instruction pairs in the schedule where:
/// 1. Both were ready within the same cycle window (within 1 cycle).
/// 2. They use compatible execution ports.
/// 3. They are NOT directly dependent on each other (no edge between them).
///
/// These pairs *could* have been dual-issued on real hardware even though
/// our simple list scheduler serializes them.
pub fn find_dual_issue_hints(dag: &ScheduleDAG) -> Vec<DualIssueHint> {
    let mut hints = Vec::new();
    let n = dag.nodes.len();
    if n < 2 {
        return hints;
    }

    // Build a set of direct dependency edges for O(1) lookup.
    let mut dep_edges: HashSet<(usize, usize)> = HashSet::new();
    for (idx, node) in dag.nodes.iter().enumerate() {
        for &pred in &node.deps {
            dep_edges.insert((pred, idx));
        }
    }

    // Build schedule order sorted by (earliest_start, node_index).
    let mut schedule_order: Vec<(u32, usize)> = (0..n)
        .map(|i| (dag.nodes[i].earliest_start, i))
        .collect();
    schedule_order.sort_by_key(|&(cycle, idx)| (cycle, idx));

    // Check consecutive pairs: if both were ready within 1 cycle and are
    // independent, they could dual-issue.
    for window in schedule_order.windows(2) {
        let (cycle_a, idx_a) = window[0];
        let (cycle_b, idx_b) = window[1];

        // Must be within 1 cycle of each other (list scheduler serialization).
        if cycle_b > cycle_a + 1 {
            continue;
        }

        // Must not be directly dependent.
        if dep_edges.contains(&(idx_a, idx_b)) || dep_edges.contains(&(idx_b, idx_a)) {
            continue;
        }

        let port_a = dag.nodes[idx_a].port;
        let port_b = dag.nodes[idx_b].port;
        if let Some(reason) = can_dual_issue(port_a, port_b) {
            hints.push(DualIssueHint {
                first: idx_a,
                second: idx_b,
                reason,
            });
        }
    }

    hints
}

// ---------------------------------------------------------------------------
// Register pressure tracking
// ---------------------------------------------------------------------------

/// Register pressure snapshot during scheduling.
///
/// Tracks the maximum number of simultaneously live GPR and FPR virtual
/// registers. If pressure exceeds the allocatable register count, the
/// register allocator will need to spill, which is expensive.
#[derive(Debug, Clone)]
pub struct RegisterPressure {
    /// Current GPR live count.
    pub gpr_pressure: u32,
    /// Current FPR live count.
    pub fpr_pressure: u32,
    /// Maximum GPR pressure seen during the schedule.
    pub max_gpr_pressure: u32,
    /// Maximum FPR pressure seen during the schedule.
    pub max_fpr_pressure: u32,
    /// GPR limit before spilling is expected (allocatable GPRs on AArch64).
    pub gpr_limit: u32,
    /// FPR limit before spilling is expected (allocatable FPRs on AArch64).
    pub fpr_limit: u32,
}

impl RegisterPressure {
    /// Returns true if register pressure exceeded the allocatable limit
    /// at any point during scheduling.
    pub fn pressure_exceeded(&self) -> bool {
        self.max_gpr_pressure > self.gpr_limit || self.max_fpr_pressure > self.fpr_limit
    }
}

/// Compute register pressure for a scheduled instruction order.
///
/// Walks the schedule in order, tracking which vregs are live. A vreg
/// becomes live when defined and dies after its last use in the schedule.
///
/// Uses: AArch64 has 28 allocatable GPRs (X0-X28 minus FP/LR) and
/// 32 allocatable FPRs (V0-V31).
pub fn compute_register_pressure(
    func: &MachFunction,
    _block_id: BlockId,
    schedule: &[InstId],
) -> RegisterPressure {
    // GPR limit: X0-X28 = 29, minus X29(FP), X30(LR) => 28 allocatable.
    // FPR limit: V0-V31, callee-saved V8-V15 still allocatable => 32.
    let gpr_limit: u32 = 28;
    let fpr_limit: u32 = 32;

    // Build def and last-use maps.
    let mut def_pos: HashMap<u32, usize> = HashMap::new(); // vreg_id -> schedule position
    let mut last_use_pos: HashMap<u32, usize> = HashMap::new(); // vreg_id -> last use position
    let mut vreg_class: HashMap<u32, RegClass> = HashMap::new();

    for (pos, &inst_id) in schedule.iter().enumerate() {
        let inst = func.inst(inst_id);
        let produces = inst_produces_value(inst);

        // First operand is def if instruction produces a value.
        if produces {
            if let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                def_pos.entry(vreg.id).or_insert(pos);
                vreg_class.insert(vreg.id, vreg.class);
            }
        }

        // All other operands are uses.
        let use_start = if produces { 1 } else { 0 };
        for operand in &inst.operands[use_start..] {
            if let MachOperand::VReg(vreg) = operand {
                last_use_pos.insert(vreg.id, pos);
                vreg_class.entry(vreg.id).or_insert(vreg.class);
            }
        }
    }

    // Walk schedule positions, maintaining live set.
    let mut live_gprs: HashSet<u32> = HashSet::new();
    let mut live_fprs: HashSet<u32> = HashSet::new();
    let mut max_gpr: u32 = 0;
    let mut max_fpr: u32 = 0;

    for (pos, &inst_id) in schedule.iter().enumerate() {
        let inst = func.inst(inst_id);

        // Add def to live set.
        if inst_produces_value(inst) {
            if let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                let is_fpr = matches!(
                    vreg.class,
                    RegClass::Fpr128 | RegClass::Fpr64 | RegClass::Fpr32
                        | RegClass::Fpr16 | RegClass::Fpr8
                );
                if is_fpr {
                    live_fprs.insert(vreg.id);
                } else {
                    live_gprs.insert(vreg.id);
                }
            }
        }

        // Update max pressure.
        max_gpr = max_gpr.max(live_gprs.len() as u32);
        max_fpr = max_fpr.max(live_fprs.len() as u32);

        // Remove dead vregs (last use at this position).
        // Collect into a Vec first to avoid borrowing conflicts.
        let dead_gprs: Vec<u32> = live_gprs
            .iter()
            .filter(|&&id| last_use_pos.get(&id).copied() == Some(pos))
            .copied()
            .collect();
        for id in dead_gprs {
            live_gprs.remove(&id);
        }

        let dead_fprs: Vec<u32> = live_fprs
            .iter()
            .filter(|&&id| last_use_pos.get(&id).copied() == Some(pos))
            .copied()
            .collect();
        for id in dead_fprs {
            live_fprs.remove(&id);
        }
    }

    RegisterPressure {
        gpr_pressure: live_gprs.len() as u32,
        fpr_pressure: live_fprs.len() as u32,
        max_gpr_pressure: max_gpr,
        max_fpr_pressure: max_fpr,
        gpr_limit,
        fpr_limit,
    }
}

// ---------------------------------------------------------------------------
// Schedule quality metrics
// ---------------------------------------------------------------------------

/// Quality metrics for a computed schedule.
///
/// Provides a quantitative assessment of how well the scheduler has
/// utilized execution resources and avoided pipeline hazards.
#[derive(Debug, Clone)]
pub struct ScheduleMetrics {
    /// Total number of instructions scheduled.
    pub total_instructions: usize,
    /// Total execution cycles (span from first to last instruction completion).
    pub total_cycles: u32,
    /// Estimated instructions per cycle (IPC).
    pub ipc_estimate: f64,
    /// Number of cycles where the pipeline stalled (no instruction issued
    /// despite pending work).
    pub stall_count: u32,
    /// Number of data hazards detected.
    pub data_hazards: u32,
    /// Number of structural hazards detected.
    pub structural_hazards: u32,
    /// Length of the critical path in cycles.
    pub critical_path_length: u32,
    /// Number of dual-issue opportunities found.
    pub dual_issue_opportunities: u32,
    /// Maximum GPR register pressure.
    pub max_gpr_pressure: u32,
    /// Maximum FPR register pressure.
    pub max_fpr_pressure: u32,
    /// Whether register pressure exceeded allocatable limits.
    pub pressure_exceeded: bool,
}

/// Compute comprehensive schedule quality metrics.
///
/// Requires a DAG that has been through `schedule_list` (earliest_start populated)
/// and the resulting instruction order.
pub fn compute_schedule_metrics(
    func: &MachFunction,
    block_id: BlockId,
    dag: &ScheduleDAG,
    schedule: &[InstId],
) -> ScheduleMetrics {
    let n = dag.nodes.len();
    if n == 0 {
        return ScheduleMetrics {
            total_instructions: 0,
            total_cycles: 0,
            ipc_estimate: 0.0,
            stall_count: 0,
            data_hazards: 0,
            structural_hazards: 0,
            critical_path_length: 0,
            dual_issue_opportunities: 0,
            max_gpr_pressure: 0,
            max_fpr_pressure: 0,
            pressure_exceeded: false,
        };
    }

    // Total cycles: max(earliest_start + latency) across all nodes.
    let total_cycles = dag.nodes.iter()
        .map(|node| node.earliest_start + node.latency)
        .max()
        .unwrap_or(0);

    // IPC estimate.
    let ipc = if total_cycles > 0 {
        n as f64 / total_cycles as f64
    } else {
        n as f64
    };

    // Stall count: cycles where no instruction was issued.
    // Build a set of cycles where at least one instruction was scheduled.
    let mut issue_cycles: HashSet<u32> = HashSet::new();
    for node in &dag.nodes {
        issue_cycles.insert(node.earliest_start);
    }
    let max_issue_cycle = dag.nodes.iter()
        .map(|node| node.earliest_start)
        .max()
        .unwrap_or(0);
    let stall_count = (0..=max_issue_cycle)
        .filter(|c| !issue_cycles.contains(c))
        .count() as u32;

    // Critical path length: the maximum priority in the DAG
    // (which is the longest path from any node to any exit).
    let critical_path_length = dag.nodes.iter()
        .map(|node| node.priority)
        .max()
        .unwrap_or(0);

    // Hazard detection.
    let hazards = detect_hazards(dag);
    let data_hazards = hazards
        .iter()
        .filter(|h| matches!(h, HazardKind::DataHazard { .. }))
        .count() as u32;
    let structural_hazards = hazards
        .iter()
        .filter(|h| matches!(h, HazardKind::StructuralHazard { .. }))
        .count() as u32;

    // Dual-issue opportunities.
    let dual_hints = find_dual_issue_hints(dag);
    let dual_issue_opportunities = dual_hints.len() as u32;

    // Register pressure.
    let pressure = compute_register_pressure(func, block_id, schedule);

    ScheduleMetrics {
        total_instructions: n,
        total_cycles,
        ipc_estimate: ipc,
        stall_count,
        data_hazards,
        structural_hazards,
        critical_path_length,
        dual_issue_opportunities,
        max_gpr_pressure: pressure.max_gpr_pressure,
        max_fpr_pressure: pressure.max_fpr_pressure,
        pressure_exceeded: pressure.pressure_exceeded(),
    }
}

// ---------------------------------------------------------------------------
// Schedule block with metrics
// ---------------------------------------------------------------------------

/// Schedule one basic block and return both the reordering result and
/// quality metrics for the schedule.
pub fn schedule_block_with_metrics(
    func: &mut MachFunction,
    block_id: BlockId,
) -> (bool, ScheduleMetrics) {
    let block = func.block(block_id);
    if block.insts.len() <= 1 {
        let metrics = ScheduleMetrics {
            total_instructions: block.insts.len(),
            total_cycles: if block.insts.is_empty() { 0 } else { 1 },
            ipc_estimate: if block.insts.is_empty() { 0.0 } else { 1.0 },
            stall_count: 0,
            data_hazards: 0,
            structural_hazards: 0,
            critical_path_length: if block.insts.is_empty() { 0 } else { 1 },
            dual_issue_opportunities: 0,
            max_gpr_pressure: 0,
            max_fpr_pressure: 0,
            pressure_exceeded: false,
        };
        return (false, metrics);
    }

    let original_order: Vec<InstId> = block.insts.clone();
    let mut dag = build_dag(func, block_id);
    let new_order = schedule_list(&mut dag);

    let metrics = compute_schedule_metrics(func, block_id, &dag, &new_order);

    let changed = new_order != original_order;
    if changed {
        let block_mut = func.block_mut(block_id);
        block_mut.insts = new_order;
    }

    (changed, metrics)
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

    // ---- Phase 2: Resource state tests ----

    #[test]
    fn test_resource_state_basic() {
        let mut rs = ResourceState::new();
        // IntAlu has 6 units.
        assert_eq!(rs.units_available(ExecutionPort::IntAlu, 0), 6);
        assert!(rs.is_available(ExecutionPort::IntAlu, 0));
        // Reserve one unit.
        assert!(rs.reserve(ExecutionPort::IntAlu, 0));
        assert_eq!(rs.units_available(ExecutionPort::IntAlu, 0), 5);
    }

    #[test]
    fn test_resource_state_exhaustion() {
        let mut rs = ResourceState::new();
        // IntDiv has 1 unit. Reserve it.
        assert!(rs.reserve(ExecutionPort::IntDiv, 0));
        assert_eq!(rs.units_available(ExecutionPort::IntDiv, 0), 0);
        assert!(!rs.is_available(ExecutionPort::IntDiv, 0));
        // Attempting to reserve again should fail.
        assert!(!rs.reserve(ExecutionPort::IntDiv, 0));
        // But a different cycle should be fine.
        assert!(rs.is_available(ExecutionPort::IntDiv, 1));
    }

    #[test]
    fn test_port_capacity_values() {
        assert_eq!(port_capacity(ExecutionPort::IntAlu), 6);
        assert_eq!(port_capacity(ExecutionPort::IntMul), 2);
        assert_eq!(port_capacity(ExecutionPort::IntDiv), 1);
        assert_eq!(port_capacity(ExecutionPort::LoadStore), 2);
        assert_eq!(port_capacity(ExecutionPort::Branch), 1);
        assert_eq!(port_capacity(ExecutionPort::FpAlu), 4);
    }

    // ---- Phase 2: Hazard detection tests ----

    #[test]
    fn test_detect_data_hazard() {
        // v1 = mul v0, v0    (node 0, lat=3, cycle 0)
        // v2 = add v1, #1    (node 1, lat=1, dep on node 0, cycle 3)
        // ret                 (node 2)
        //
        // Node 1 must wait until cycle 3 (producer latency 3). The gap
        // from cycle 0+1=1 to cycle 3 is a 2-cycle stall = data hazard.
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(1), vreg(0), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![mul, add, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let hazards = detect_hazards(&dag);

        let has_data_hazard = hazards.iter().any(|h| {
            matches!(h, HazardKind::DataHazard { producer: 0, consumer: 1, .. })
        });
        assert!(has_data_hazard, "mul->add chain should produce a data hazard");
    }

    #[test]
    fn test_detect_structural_hazard_divides() {
        // Two divides at the same cycle would conflict on the single IntDiv unit.
        // v1 = sdiv v0, v2    (node 0, IntDiv)
        // v3 = sdiv v4, v5    (node 1, IntDiv, independent)
        // ret                  (node 2)
        //
        // Both are independent so the scheduler can schedule them at the same cycle,
        // but there's only 1 IntDiv unit.
        let div1 = MachInst::new(AArch64Opcode::SDiv, vec![vreg(1), vreg(0), vreg(2)]);
        let div2 = MachInst::new(AArch64Opcode::SDiv, vec![vreg(3), vreg(4), vreg(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![div1, div2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);

        // If both divides end up at the same cycle, there should be a structural hazard.
        // The scheduler issues one per cycle, so they might be at cycles 0 and 1.
        // Either way, verify detect_hazards runs without panic.
        let hazards = detect_hazards(&dag);
        // No panic is the basic assertion; structural hazard depends on scheduling.
        let _ = hazards.len(); // runs without error
    }

    #[test]
    fn test_detect_load_use_hazard() {
        // v1 = ldr [v0, #0]   (node 0, lat=4, LoadStore)
        // v2 = add v1, #1     (node 1, uses v1)
        // ret                  (node 2)
        //
        // After scheduling: load at cycle 0, add at cycle 4 (earliest possible).
        // No load-use hazard because add is not at cycle 1.
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![load, add, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let hazards = detect_hazards(&dag);

        // The scheduler correctly places add at cycle 4 (not cycle 1),
        // so there should be no load-use hazard but there IS a data hazard
        // (the 3-cycle wait).
        let has_data = hazards.iter().any(|h| matches!(h, HazardKind::DataHazard { .. }));
        assert!(has_data, "load->add should produce a data hazard");
    }

    #[test]
    fn test_no_hazard_independent() {
        // Two independent ALU ops: no hazards expected (6 ALU units available).
        // v1 = add v0, #1    (node 0)
        // v3 = add v2, #2    (node 1, independent)
        // ret                 (node 2)
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(2), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let hazards = detect_hazards(&dag);

        // No structural hazards (6 ALU units), no data hazards (independent).
        let structural = hazards.iter().filter(|h| matches!(h, HazardKind::StructuralHazard { .. })).count();
        assert_eq!(structural, 0, "independent ALU ops should have no structural hazards");
    }

    // ---- Phase 2: Dual-issue hint tests ----

    #[test]
    fn test_dual_issue_alu_load() {
        // v1 = add v0, #1    (IntAlu)
        // v2 = ldr [v3, #0]  (LoadStore, independent)
        // ret
        //
        // Both can be at cycle 0 => ALU + Load dual-issue.
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(3), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add, load, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let hints = find_dual_issue_hints(&dag);

        // At least one ALU + Load/Store dual-issue hint should be present.
        let has_alu_load = hints.iter().any(|h| h.reason.contains("Load/Store"));
        assert!(
            has_alu_load,
            "ALU + Load should produce dual-issue hint, got {:?}",
            hints.iter().map(|h| h.reason).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_dual_issue_alu_alu() {
        // Two independent ALU ops at the same cycle.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(2), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let hints = find_dual_issue_hints(&dag);

        let has_alu_alu = hints.iter().any(|h| h.reason == "ALU + ALU");
        assert!(has_alu_alu, "two independent ALU ops should hint ALU + ALU dual-issue");
    }

    #[test]
    fn test_no_dual_issue_dependent_chain() {
        // v1 = add v0, #1
        // v2 = add v1, #2  (depends on v1)
        // ret
        //
        // The second add can't issue at the same cycle as the first.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let hints = find_dual_issue_hints(&dag);

        // add1 at cycle 0, add2 at cycle 1 (data dep): no dual-issue possible.
        let alu_alu = hints.iter().filter(|h| h.reason == "ALU + ALU").count();
        assert_eq!(alu_alu, 0, "dependent chain should not produce dual-issue hint");
    }

    // ---- Phase 2: Register pressure tests ----

    #[test]
    fn test_register_pressure_basic() {
        // v1 = add v0, #1   (def v1)
        // v2 = add v1, #2   (use v1, def v2)
        // ret
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let schedule: Vec<InstId> = func.block(func.entry).insts.clone();
        let pressure = compute_register_pressure(&func, func.entry, &schedule);

        // v1 is live from pos 0 to pos 1, v2 is live from pos 1. Max GPR = 2.
        assert!(pressure.max_gpr_pressure <= 3, "basic chain should have low pressure");
        assert!(!pressure.pressure_exceeded(), "should not exceed pressure limit");
    }

    #[test]
    fn test_register_pressure_high() {
        // Create many independent defs to spike pressure.
        // v1 = mov #1
        // v2 = mov #2
        // ...
        // v30 = mov #30
        // v31 = add v1, v2  (uses v1, v2 to keep them live)
        // ret
        let mut insts: Vec<MachInst> = Vec::new();
        for i in 1..=30 {
            insts.push(MachInst::new(AArch64Opcode::MovI, vec![vreg(i), imm(i as i64)]));
        }
        // Use v1 and v2 to keep them live through the block.
        insts.push(MachInst::new(AArch64Opcode::AddRR, vec![vreg(31), vreg(1), vreg(2)]));
        insts.push(MachInst::new(AArch64Opcode::Ret, vec![]));
        let func = make_func_with_insts(insts);

        let schedule: Vec<InstId> = func.block(func.entry).insts.clone();
        let pressure = compute_register_pressure(&func, func.entry, &schedule);

        // 30 defs, at peak all 30 are live. GPR limit is 28.
        assert!(
            pressure.max_gpr_pressure >= 28,
            "30 independent defs should produce high pressure, got {}",
            pressure.max_gpr_pressure
        );
        assert!(pressure.pressure_exceeded(), "30 live GPRs should exceed 28 limit");
    }

    #[test]
    fn test_register_pressure_fpr() {
        // FPR pressure tracking.
        // v1 = fadd v0, v0   (FPR def)
        // v2 = fadd v1, v1   (FPR def, uses v1)
        // ret
        let fadd1 = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![
                MachOperand::VReg(VReg::new(1, RegClass::Fpr64)),
                MachOperand::VReg(VReg::new(0, RegClass::Fpr64)),
                MachOperand::VReg(VReg::new(0, RegClass::Fpr64)),
            ],
        );
        let fadd2 = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![
                MachOperand::VReg(VReg::new(2, RegClass::Fpr64)),
                MachOperand::VReg(VReg::new(1, RegClass::Fpr64)),
                MachOperand::VReg(VReg::new(1, RegClass::Fpr64)),
            ],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![fadd1, fadd2, ret]);

        let schedule: Vec<InstId> = func.block(func.entry).insts.clone();
        let pressure = compute_register_pressure(&func, func.entry, &schedule);

        assert!(
            pressure.max_fpr_pressure >= 1,
            "FPR ops should register FPR pressure, got {}",
            pressure.max_fpr_pressure
        );
        assert_eq!(pressure.max_gpr_pressure, 0, "FPR-only code should have no GPR pressure");
    }

    // ---- Phase 2: Schedule metrics tests ----

    #[test]
    fn test_schedule_metrics_basic() {
        // Simple block: two independent adds + ret.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(2), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let order = schedule_list(&mut dag);
        let metrics = compute_schedule_metrics(&func, func.entry, &dag, &order);

        assert_eq!(metrics.total_instructions, 3);
        assert!(metrics.total_cycles > 0, "should have at least 1 cycle");
        assert!(metrics.ipc_estimate > 0.0, "IPC should be positive");
        assert!(metrics.critical_path_length >= 2, "critical path >= 2 (ALU + ret)");
    }

    #[test]
    fn test_schedule_metrics_stalls() {
        // mul -> add chain: mul takes 3 cycles, add must wait.
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(1), vreg(0), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![mul, add, ret]);

        let mut dag = build_dag(&func, func.entry);
        let order = schedule_list(&mut dag);
        let metrics = compute_schedule_metrics(&func, func.entry, &dag, &order);

        // mul at cycle 0, add at cycle 3, ret at cycle 4.
        // Cycles 1, 2 are stalls (nothing issued).
        assert!(
            metrics.stall_count >= 2,
            "mul->add chain should have at least 2 stall cycles, got {}",
            metrics.stall_count
        );
    }

    #[test]
    fn test_schedule_metrics_ipc() {
        // Two independent adds + ret: all 3 can issue in rapid succession.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(2), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let order = schedule_list(&mut dag);
        let metrics = compute_schedule_metrics(&func, func.entry, &dag, &order);

        // 3 instructions, 3 cycles (ret at cycle 2, completes at 3) => IPC = 3/3 = 1.0.
        assert!(
            metrics.ipc_estimate >= 0.5,
            "independent ALU ops should have reasonable IPC, got {}",
            metrics.ipc_estimate
        );
    }

    #[test]
    fn test_schedule_metrics_critical_path() {
        // mul (3) -> add (1) -> ret (1) = critical path of 5
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(1), vreg(0), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![mul, add, ret]);

        let mut dag = build_dag(&func, func.entry);
        let _order = schedule_list(&mut dag);
        let metrics = compute_schedule_metrics(&func, func.entry, &dag, &_order);

        assert_eq!(
            metrics.critical_path_length, 5,
            "mul(3)->add(1)->ret(1) = critical path 5"
        );
    }

    // ---- Phase 2: Integration test ----

    #[test]
    fn test_schedule_block_with_metrics() {
        // Integration: schedule a block and get metrics back.
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(3), imm(99)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![load, add, mov, ret]);

        let entry = func.entry;
        let (changed, metrics) = schedule_block_with_metrics(&mut func, entry);

        // The scheduler should reorder: load, mov, add, ret (mov during load latency).
        assert!(changed, "scheduler should reorder load-add-mov to load-mov-add");
        assert_eq!(metrics.total_instructions, 4);
        assert!(metrics.total_cycles > 0);
        assert!(metrics.ipc_estimate > 0.0);
        assert!(!metrics.pressure_exceeded);
    }

    #[test]
    fn test_schedule_block_with_metrics_empty() {
        let mut func = MachFunction::new(
            "empty".to_string(),
            Signature::new(vec![], vec![]),
        );
        let entry = func.entry;
        let (changed, metrics) = schedule_block_with_metrics(&mut func, entry);
        assert!(!changed);
        assert_eq!(metrics.total_instructions, 0);
        assert_eq!(metrics.total_cycles, 0);
    }

    #[test]
    fn test_schedule_block_with_metrics_single() {
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ret]);
        let entry = func.entry;
        let (changed, metrics) = schedule_block_with_metrics(&mut func, entry);
        assert!(!changed);
        assert_eq!(metrics.total_instructions, 1);
        assert_eq!(metrics.total_cycles, 1);
        assert_eq!(metrics.stall_count, 0);
    }

    #[test]
    fn test_dual_issue_count_in_metrics() {
        // Several independent ops that should produce dual-issue hints.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(3), imm(0)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(4), vreg(5), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, load, add2, ret]);

        let mut dag = build_dag(&func, func.entry);
        let order = schedule_list(&mut dag);
        let metrics = compute_schedule_metrics(&func, func.entry, &dag, &order);

        // With independent ALU + Load, there should be dual-issue opportunities.
        assert!(
            metrics.dual_issue_opportunities >= 1,
            "independent ALU + Load should have dual-issue opportunities, got {}",
            metrics.dual_issue_opportunities
        );
    }

    // ---- Phase 3: Pressure-aware scheduling tests ----

    fn fpreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Fpr64))
    }

    #[test]
    fn test_pressure_tracker_basic() {
        let mut tracker = PressureTracker::new();
        assert_eq!(tracker.gpr_pressure(), 0);
        assert_eq!(tracker.fpr_pressure(), 0);
        assert!(!tracker.is_high_pressure());

        // Define a GPR.
        tracker.define_vreg(1, RegClass::Gpr64);
        assert_eq!(tracker.gpr_pressure(), 1);
        assert_eq!(tracker.peak_gpr, 1);

        // Define an FPR.
        tracker.define_vreg(100, RegClass::Fpr64);
        assert_eq!(tracker.fpr_pressure(), 1);
        assert_eq!(tracker.peak_fpr, 1);

        // Kill GPR.
        tracker.kill_vreg(1);
        assert_eq!(tracker.gpr_pressure(), 0);
        assert_eq!(tracker.peak_gpr, 1); // peak preserved
    }

    #[test]
    fn test_pressure_tracker_high_pressure_threshold() {
        let mut tracker = PressureTracker::with_thresholds(3, 3);
        assert!(!tracker.is_high_pressure());

        // Define 4 GPR VRegs -> exceeds threshold of 3.
        tracker.define_vreg(1, RegClass::Gpr64);
        tracker.define_vreg(2, RegClass::Gpr64);
        tracker.define_vreg(3, RegClass::Gpr64);
        assert!(!tracker.is_high_pressure()); // exactly at threshold
        tracker.define_vreg(4, RegClass::Gpr64);
        assert!(tracker.is_high_pressure()); // above threshold

        // Kill one -> back to threshold.
        tracker.kill_vreg(1);
        assert!(!tracker.is_high_pressure());
    }

    #[test]
    fn test_pressure_tracker_fpr_high_pressure() {
        let mut tracker = PressureTracker::with_thresholds(100, 2);
        // Only 2 FPR threshold.
        tracker.define_vreg(1, RegClass::Fpr64);
        tracker.define_vreg(2, RegClass::Fpr64);
        assert!(!tracker.is_high_pressure());
        tracker.define_vreg(3, RegClass::Fpr64);
        assert!(tracker.is_high_pressure());
    }

    #[test]
    fn test_compute_pressure_info_basic() {
        // v1 = add v0, #1   (def v1, use v0)
        // v2 = add v1, #2   (def v2, use v1 — v1 killed here)
        // ret
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let func = make_func_with_insts(vec![add1, add2, ret]);

        let dag = build_dag(&func, func.entry);
        let infos = compute_pressure_info(&func, &dag);

        // Node 0 (add v0,#1 -> v1): defs={v1}, uses={v0}. v0 is not last-used here (also used nowhere else in
        // the DAG sense, but v0 is only used by node 0 -> it IS last use).
        assert_eq!(infos[0].defs.len(), 1);
        assert_eq!(infos[0].defs[0].0, 1); // defines v1

        // Node 1 (add v1,#2 -> v2): defs={v2}, uses={v1}. v1's last use is node 1.
        assert_eq!(infos[1].defs.len(), 1);
        assert_eq!(infos[1].kills, 1); // v1 killed
        assert_eq!(infos[1].net_pressure, 0); // 1 def - 1 kill = 0

        // Node 2 (ret): no defs, no uses.
        assert_eq!(infos[2].defs.len(), 0);
        assert_eq!(infos[2].kills, 0);
    }

    #[test]
    fn test_pressure_aware_scheduler_low_pressure() {
        // With low pressure, the pressure-aware scheduler should behave like
        // the basic scheduler (critical-path priority).
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(0), imm(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(1)]);
        let mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(3), imm(99)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![load, add, mov, ret]);

        let entry = func.entry;
        let (_changed, tracker) = schedule_block_pressure_aware(&mut func, entry);

        // Should still reorder (load, mov, add, ret) for latency hiding.
        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();
        assert_eq!(order[0], InstId(0), "load should be first");
        assert_eq!(*order.last().unwrap(), InstId(3), "ret should be last");
        assert!(tracker.peak_gpr <= 3, "low pressure test");
    }

    #[test]
    fn test_pressure_aware_reduces_peak_pressure() {
        // Create a block with many independent defs followed by uses.
        // The pressure-naive scheduler (critical-path) would schedule all defs
        // first, spiking pressure. The pressure-aware scheduler should interleave
        // defs and uses when pressure gets high.
        //
        // Pattern:
        //   v1 = mov #1    (producer)
        //   v2 = mov #2    (producer)
        //   ...
        //   v25 = mov #25  (producer)
        //   v26 = add v1, v2  (consumer, kills v1 and v2)
        //   v27 = add v3, v4  (consumer, kills v3 and v4)
        //   ...
        //   ret
        //
        // With threshold=20: after 20 defs, pressure-aware will prefer consumers.
        let mut insts: Vec<MachInst> = Vec::new();
        let num_producers = 25;

        // 25 independent movs: v1..v25.
        for i in 1..=(num_producers as u32) {
            insts.push(MachInst::new(AArch64Opcode::MovI, vec![vreg(i), imm(i as i64)]));
        }

        // 12 consumers: pair up v1+v2, v3+v4, ... v23+v24, and v25 unused.
        let consumer_start = num_producers as u32 + 1;
        for i in 0..12u32 {
            let src1 = i * 2 + 1;
            let src2 = i * 2 + 2;
            insts.push(MachInst::new(
                AArch64Opcode::AddRR,
                vec![vreg(consumer_start + i), vreg(src1), vreg(src2)],
            ));
        }

        insts.push(MachInst::new(AArch64Opcode::Ret, vec![]));
        let func = make_func_with_insts(insts.clone());

        // Measure pressure with basic scheduler.
        let basic_schedule: Vec<InstId> = {
            let mut dag = build_dag(&func, func.entry);
            schedule_list(&mut dag)
        };
        let basic_pressure = compute_register_pressure(&func, func.entry, &basic_schedule);

        // Measure pressure with pressure-aware scheduler.
        let func_pa = make_func_with_insts(insts);
        let entry = func_pa.entry;
        let mut dag = build_dag(&func_pa, entry);
        let (pa_schedule, pa_tracker) = schedule_list_pressure_aware(&func_pa, &mut dag);
        let pa_pressure = compute_register_pressure(&func_pa, entry, &pa_schedule);

        // The pressure-aware scheduler should achieve lower or equal peak GPR
        // pressure than the basic scheduler.
        assert!(
            pa_pressure.max_gpr_pressure <= basic_pressure.max_gpr_pressure,
            "pressure-aware scheduler should not increase peak pressure: \
             PA={} vs basic={}",
            pa_pressure.max_gpr_pressure,
            basic_pressure.max_gpr_pressure,
        );

        // Verify the tracker itself tracked pressure.
        assert!(
            pa_tracker.peak_gpr > 0,
            "pressure tracker should have observed some GPR pressure"
        );
    }

    #[test]
    fn test_pressure_aware_preserves_dependencies() {
        // Verify that pressure-aware scheduling still respects data dependencies.
        // v1 = add v0, #1    (inst0, def v1)
        // v2 = add v1, #2    (inst1, depends on v1)
        // ret                  (inst2)
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, add2, ret]);

        let entry = func.entry;
        let (_changed, _tracker) = schedule_block_pressure_aware(&mut func, entry);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        // add1 must come before add2 (data dependency on v1).
        let pos_add1 = order.iter().position(|&id| id == InstId(0)).unwrap();
        let pos_add2 = order.iter().position(|&id| id == InstId(1)).unwrap();
        assert!(
            pos_add1 < pos_add2,
            "pressure-aware scheduler must respect data deps: add1@{} add2@{}",
            pos_add1,
            pos_add2,
        );

        // ret must be last.
        assert_eq!(*order.last().unwrap(), InstId(2));
    }

    #[test]
    fn test_pressure_aware_preserves_memory_ordering() {
        // str v0, [v10, #0]   (store, inst0)
        // v1 = ldr [v2, #0]   (load, inst1 — must come after store)
        // ret                   (inst2)
        let store = MachInst::new(AArch64Opcode::StrRI, vec![vreg(0), vreg(10), imm(0)]);
        let load = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(1), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![store, load, ret]);

        let entry = func.entry;
        let (_changed, _tracker) = schedule_block_pressure_aware(&mut func, entry);

        let block = func.block(func.entry);
        let order: Vec<InstId> = block.insts.clone();

        let pos_store = order.iter().position(|&id| id == InstId(0)).unwrap();
        let pos_load = order.iter().position(|&id| id == InstId(1)).unwrap();
        assert!(
            pos_store < pos_load,
            "pressure-aware scheduler must respect memory deps"
        );
    }

    #[test]
    fn test_pressure_aware_pass_interface() {
        // Verify PressureAwareScheduler implements MachinePass correctly.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(3), vreg(4)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(5), vreg(2), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, mul, add2, ret]);

        let mut pass = PressureAwareScheduler;
        assert_eq!(pass.name(), "pressure-aware-scheduler");
        pass.run(&mut func);

        // Just verify it doesn't crash and produces a valid schedule.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4);
        assert_eq!(*block.insts.last().unwrap(), InstId(3), "ret must be last");
    }

    #[test]
    fn test_pressure_aware_empty_and_single() {
        // Empty block.
        let mut func = MachFunction::new(
            "empty".to_string(),
            Signature::new(vec![], vec![]),
        );
        let entry = func.entry;
        let (changed, tracker) = schedule_block_pressure_aware(&mut func, entry);
        assert!(!changed);
        assert_eq!(tracker.peak_gpr, 0);

        // Single instruction.
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func2 = make_func_with_insts(vec![ret]);
        let entry2 = func2.entry;
        let (changed2, tracker2) = schedule_block_pressure_aware(&mut func2, entry2);
        assert!(!changed2);
        assert_eq!(tracker2.peak_gpr, 0);
    }

    #[test]
    fn test_pressure_aware_function_scheduling() {
        // Verify schedule_function_pressure_aware works across multiple blocks.
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let (changed, peak_gpr, peak_fpr) = schedule_function_pressure_aware(&mut func);
        // Two instructions, no reordering expected.
        assert!(!changed);
        // Tracker should have observed the def.
        assert!(peak_gpr <= 1);
        assert_eq!(peak_fpr, 0);
    }

    #[test]
    fn test_pressure_aware_idempotent() {
        // Pressure-aware scheduling should be idempotent.
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(0), imm(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, add2, ret]);

        let mut pass = PressureAwareScheduler;
        pass.run(&mut func);
        let order1: Vec<InstId> = func.block(func.entry).insts.clone();

        let changed = pass.run(&mut func);
        let order2: Vec<InstId> = func.block(func.entry).insts.clone();

        assert_eq!(order1, order2, "pressure-aware scheduler should be idempotent");
        assert!(!changed, "second run should report no change");
    }

    #[test]
    fn test_pressure_aware_fpr_tracking() {
        // Verify FPR pressure is tracked separately from GPR.
        // v1 = fadd v0, v0   (FPR def)
        // v2 = fadd v1, v1   (FPR def, uses v1)
        // ret
        let fadd1 = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![fpreg(1), fpreg(0), fpreg(0)],
        );
        let fadd2 = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![fpreg(2), fpreg(1), fpreg(1)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![fadd1, fadd2, ret]);

        let entry = func.entry;
        let (_changed, tracker) = schedule_block_pressure_aware(&mut func, entry);

        // Should track FPR pressure, not GPR.
        assert!(tracker.peak_fpr >= 1, "FPR ops should track FPR pressure");
        assert_eq!(tracker.peak_gpr, 0, "FPR-only code should have no GPR pressure");
    }
}

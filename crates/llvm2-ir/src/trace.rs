// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Compilation event log infrastructure for glass-box transparency.
//!
//! Every compilation pass can emit structured [`CompilationEvent`]s to a
//! [`CompilationTrace`] collector. Events record *what* happened and *why*,
//! enabling full instruction-level provenance from tMIR input to binary output.
//!
//! # Opt-in levels
//!
//! | Level | Overhead | Use case |
//! |-------|----------|----------|
//! | [`TraceLevel::None`] | 0% | Production builds |
//! | [`TraceLevel::Summary`] | <1% | CI/CD quality monitoring |
//! | [`TraceLevel::Full`] | 5-15% | Development, debugging |
//! | [`TraceLevel::Debug`] | 20-50% | Compiler development, auditing |
//!
//! # Thread safety
//!
//! [`CompilationTrace`] is `Send + Sync`. Multiple passes can emit events
//! concurrently via `Arc<Mutex<Vec<CompilationEvent>>>` with atomic sequence
//! numbers for ordering.
//!
//! # Serialization
//!
//! All event types derive `serde::Serialize` for JSON output via
//! [`CompilationTrace::to_json`].
//!
//! # Design reference
//!
//! See `designs/2026-04-13-debugging-transparency.md` for the full design.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::provenance::{PassId, TmirInstId};

// ---------------------------------------------------------------------------
// Trace level
// ---------------------------------------------------------------------------

/// Controls how much compilation event data is recorded.
///
/// Ordered from least to most verbose: `None < Summary < Full < Debug`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[derive(Default)]
pub enum TraceLevel {
    /// No logging. Zero overhead. Production default.
    #[default]
    None = 0,
    /// Pass-level applied/rejected counts only.
    Summary = 1,
    /// Every event with full justification.
    Full = 2,
    /// Full events plus intermediate IR snapshots.
    Debug = 3,
}


// ---------------------------------------------------------------------------
// Typed IDs (serializable, self-contained)
// ---------------------------------------------------------------------------

// PassId and TmirInstId are imported from crate::provenance (canonical definitions).

/// Identifies a transformation rule (e.g., a peephole pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct RuleId(pub u32);

// ---------------------------------------------------------------------------
// EventKind
// ---------------------------------------------------------------------------

/// What happened during compilation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum EventKind {
    /// An optimization or transformation rule was applied.
    Applied {
        rule: RuleId,
        /// Instruction IDs before the transformation.
        before: Vec<u32>,
        /// Instruction IDs after the transformation.
        after: Vec<u32>,
    },
    /// An optimization was considered but rejected.
    Rejected {
        rule: RuleId,
        /// Human-readable reason for rejection.
        reason: String,
    },
    /// A tMIR instruction was lowered to one or more machine instructions.
    Lowered {
        tmir_inst: TmirInstId,
        /// Machine instruction IDs produced.
        mach_insts: Vec<u32>,
    },
    /// A virtual register was assigned to a physical register.
    RegAssigned {
        /// Virtual register ID (VRegId.0).
        vreg: u32,
        /// Physical register encoding (PReg raw value).
        preg: u16,
        /// Why this register was chosen.
        reason: String,
    },
    /// A value was spilled to the stack.
    Spilled {
        /// Virtual register ID that was spilled.
        vreg: u32,
        /// Stack slot size in bytes.
        slot_size: u32,
        /// Estimated spill cost.
        cost: f64,
    },
    /// An instruction was encoded to binary.
    Encoded {
        /// Machine instruction ID.
        inst: u32,
        /// Byte offset in the output section.
        offset: u32,
        /// Raw encoded bytes.
        bytes: Vec<u8>,
    },
}

// ---------------------------------------------------------------------------
// Justification
// ---------------------------------------------------------------------------

/// Why a compilation decision was made.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Justification {
    /// Cost model comparison: the transformation improves estimated cost.
    CostModel {
        /// Estimated cost before the transformation.
        before: f64,
        /// Estimated cost after the transformation.
        after: f64,
    },
    /// A structural pattern match triggered the transformation.
    PatternMatch {
        /// Name or description of the matched pattern.
        pattern: String,
    },
    /// An SMT solver (z4) proved semantic equivalence.
    SolverProved {
        /// Hash of the proof obligation for cross-referencing.
        proof_hash: u64,
    },
    /// A legality constraint required this decision.
    Legality {
        /// Description of the constraint (e.g., "ABI requires X0 for return").
        constraint: String,
    },
    /// Profile-guided optimization data influenced this decision.
    ProfileGuided {
        /// Execution count or hotness metric.
        hotness: u64,
    },
}

// ---------------------------------------------------------------------------
// CompilationEvent
// ---------------------------------------------------------------------------

/// A single structured event emitted by a compilation pass.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompilationEvent {
    /// Which pass emitted this event.
    pub pass: PassId,
    /// What happened.
    pub kind: EventKind,
    /// The instruction(s) affected (machine InstId raw values).
    pub subject: Vec<u32>,
    /// Why this decision was made.
    pub justification: Justification,
    /// Monotonic sequence number for global event ordering.
    pub seq: u64,
}

// ---------------------------------------------------------------------------
// CompilationTrace (thread-safe collector)
// ---------------------------------------------------------------------------

/// Thread-safe collector for compilation events.
///
/// Wraps an `Arc<Mutex<Vec<CompilationEvent>>>` so multiple passes can
/// emit events concurrently. Sequence numbers are assigned atomically
/// for consistent global ordering.
pub struct CompilationTrace {
    level: TraceLevel,
    events: Arc<Mutex<Vec<CompilationEvent>>>,
    next_seq: AtomicU64,
}

impl CompilationTrace {
    /// Create a new trace collector at the given verbosity level.
    pub fn new(level: TraceLevel) -> Self {
        Self {
            level,
            events: Arc::new(Mutex::new(Vec::new())),
            next_seq: AtomicU64::new(0),
        }
    }

    /// Emit a compilation event. Only records if the trace level is
    /// [`TraceLevel::Summary`] or higher.
    pub fn emit(
        &self,
        pass: PassId,
        kind: EventKind,
        subject: Vec<u32>,
        justification: Justification,
    ) {
        if self.level < TraceLevel::Summary {
            return;
        }
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let event = CompilationEvent {
            pass,
            kind,
            subject,
            justification,
            seq,
        };
        let mut events = self.events.lock().expect("trace lock poisoned");
        events.push(event);
    }

    /// Returns a snapshot (clone) of all recorded events.
    pub fn events(&self) -> Vec<CompilationEvent> {
        let events = self.events.lock().expect("trace lock poisoned");
        events.clone()
    }

    /// Returns the number of recorded events.
    pub fn len(&self) -> usize {
        let events = self.events.lock().expect("trace lock poisoned");
        events.len()
    }

    /// Returns `true` if no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize all recorded events to pretty-printed JSON.
    pub fn to_json(&self) -> String {
        let events = self.events.lock().expect("trace lock poisoned");
        serde_json::to_string_pretty(&*events).expect("event serialization failed")
    }

    /// Clear all recorded events and reset the sequence counter.
    pub fn clear(&self) {
        let mut events = self.events.lock().expect("trace lock poisoned");
        events.clear();
        self.next_seq.store(0, Ordering::Relaxed);
    }

    /// Returns the current trace level.
    pub fn level(&self) -> TraceLevel {
        self.level
    }
}

// Implement Clone manually since AtomicU64 doesn't derive Clone.
impl Clone for CompilationTrace {
    fn clone(&self) -> Self {
        Self {
            level: self.level,
            events: Arc::clone(&self.events),
            next_seq: AtomicU64::new(self.next_seq.load(Ordering::Relaxed)),
        }
    }
}

impl std::fmt::Debug for CompilationTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.len();
        f.debug_struct("CompilationTrace")
            .field("level", &self.level)
            .field("event_count", &count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TraceLevel ordering --

    #[test]
    fn trace_level_ordering() {
        assert!(TraceLevel::None < TraceLevel::Summary);
        assert!(TraceLevel::Summary < TraceLevel::Full);
        assert!(TraceLevel::Full < TraceLevel::Debug);
        assert!(TraceLevel::None < TraceLevel::Debug);
    }

    #[test]
    fn trace_level_default_is_none() {
        assert_eq!(TraceLevel::default(), TraceLevel::None);
    }

    // -- Emit at Summary level records events --

    #[test]
    fn emit_at_summary_level() {
        let trace = CompilationTrace::new(TraceLevel::Summary);
        assert!(trace.is_empty());

        trace.emit(
            PassId::new("1"),
            EventKind::Applied {
                rule: RuleId(10),
                before: vec![0, 1],
                after: vec![2],
            },
            vec![0, 1, 2],
            Justification::CostModel {
                before: 3.0,
                after: 1.0,
            },
        );

        assert_eq!(trace.len(), 1);
        assert!(!trace.is_empty());

        let events = trace.events();
        assert_eq!(events[0].pass, PassId::new("1"));
        assert_eq!(events[0].seq, 0);
        assert_eq!(events[0].subject, vec![0, 1, 2]);
    }

    // -- Emit at None level does NOT record --

    #[test]
    fn emit_at_none_level() {
        let trace = CompilationTrace::new(TraceLevel::None);

        trace.emit(
            PassId::new("1"),
            EventKind::Applied {
                rule: RuleId(10),
                before: vec![0],
                after: vec![1],
            },
            vec![0],
            Justification::CostModel {
                before: 2.0,
                after: 1.0,
            },
        );

        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
    }

    // -- Emit at Full and Debug levels records events --

    #[test]
    fn emit_at_full_level() {
        let trace = CompilationTrace::new(TraceLevel::Full);
        trace.emit(
            PassId::new("0"),
            EventKind::Rejected {
                rule: RuleId(5),
                reason: "loop trip count unknown".into(),
            },
            vec![3],
            Justification::Legality {
                constraint: "vectorization requires known trip count".into(),
            },
        );
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn emit_at_debug_level() {
        let trace = CompilationTrace::new(TraceLevel::Debug);
        trace.emit(
            PassId::new("0"),
            EventKind::Encoded {
                inst: 7,
                offset: 0x48,
                bytes: vec![0x8b, 0x02, 0x00, 0x20],
            },
            vec![7],
            Justification::Legality {
                constraint: "AArch64 encoding".into(),
            },
        );
        assert_eq!(trace.len(), 1);
    }

    // -- Sequence numbers increment --

    #[test]
    fn sequence_numbers_increment() {
        let trace = CompilationTrace::new(TraceLevel::Full);

        for i in 0..5u32 {
            trace.emit(
                PassId::new(format!("{}", i)),
                EventKind::Applied {
                    rule: RuleId(0),
                    before: vec![i],
                    after: vec![i + 10],
                },
                vec![i],
                Justification::PatternMatch {
                    pattern: format!("rule_{}", i),
                },
            );
        }

        let events = trace.events();
        assert_eq!(events.len(), 5);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq, i as u64);
        }
    }

    // -- JSON serialization --

    #[test]
    fn json_serialization() {
        let trace = CompilationTrace::new(TraceLevel::Full);

        trace.emit(
            PassId::new("1"),
            EventKind::Lowered {
                tmir_inst: TmirInstId(42),
                mach_insts: vec![100, 101],
            },
            vec![100, 101],
            Justification::PatternMatch {
                pattern: "i64_add_rr".into(),
            },
        );

        let json = trace.to_json();
        assert!(json.contains("\"pass\""));
        assert!(json.contains("Lowered"));
        assert!(json.contains("i64_add_rr"));
        assert!(json.contains("42"));

        // Verify it's valid JSON by parsing.
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");
        assert!(parsed.is_array());
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 1);
    }

    // -- Clear resets state --

    #[test]
    fn clear_resets_events_and_seq() {
        let trace = CompilationTrace::new(TraceLevel::Summary);

        trace.emit(
            PassId::new("0"),
            EventKind::Applied {
                rule: RuleId(0),
                before: vec![0],
                after: vec![1],
            },
            vec![0],
            Justification::CostModel {
                before: 2.0,
                after: 1.0,
            },
        );
        assert_eq!(trace.len(), 1);

        trace.clear();
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);

        // Seq resets so next event gets seq=0.
        trace.emit(
            PassId::new("0"),
            EventKind::Applied {
                rule: RuleId(0),
                before: vec![0],
                after: vec![1],
            },
            vec![0],
            Justification::CostModel {
                before: 2.0,
                after: 1.0,
            },
        );
        let events = trace.events();
        assert_eq!(events[0].seq, 0);
    }

    // -- All EventKind variants --

    #[test]
    fn all_event_kinds() {
        let trace = CompilationTrace::new(TraceLevel::Full);

        let kinds = vec![
            EventKind::Applied {
                rule: RuleId(1),
                before: vec![0],
                after: vec![1],
            },
            EventKind::Rejected {
                rule: RuleId(2),
                reason: "not profitable".into(),
            },
            EventKind::Lowered {
                tmir_inst: TmirInstId(10),
                mach_insts: vec![20, 21],
            },
            EventKind::RegAssigned {
                vreg: 5,
                preg: 3,
                reason: "X3 is callee-saved".into(),
            },
            EventKind::Spilled {
                vreg: 7,
                slot_size: 8,
                cost: 2.5,
            },
            EventKind::Encoded {
                inst: 15,
                offset: 0x10,
                bytes: vec![0xAA, 0xBB, 0xCC, 0xDD],
            },
        ];

        for kind in kinds {
            trace.emit(
                PassId::new("0"),
                kind,
                vec![0],
                Justification::Legality {
                    constraint: "test".into(),
                },
            );
        }

        assert_eq!(trace.len(), 6);

        // Verify JSON round-trip for all kinds.
        let json = trace.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");
        assert_eq!(parsed.as_array().unwrap().len(), 6);
    }

    // -- All Justification variants --

    #[test]
    fn all_justifications() {
        let trace = CompilationTrace::new(TraceLevel::Full);

        let justifications = vec![
            Justification::CostModel {
                before: 10.0,
                after: 5.0,
            },
            Justification::PatternMatch {
                pattern: "add_zero_elim".into(),
            },
            Justification::SolverProved {
                proof_hash: 0xDEADBEEF,
            },
            Justification::Legality {
                constraint: "ABI: return in X0".into(),
            },
            Justification::ProfileGuided { hotness: 1_000_000 },
        ];

        for j in justifications {
            trace.emit(
                PassId::new("0"),
                EventKind::Applied {
                    rule: RuleId(0),
                    before: vec![0],
                    after: vec![1],
                },
                vec![0],
                j,
            );
        }

        assert_eq!(trace.len(), 5);
        let json = trace.to_json();
        assert!(json.contains("CostModel"));
        assert!(json.contains("PatternMatch"));
        assert!(json.contains("SolverProved"));
        assert!(json.contains("Legality"));
        assert!(json.contains("ProfileGuided"));
    }

    // -- Thread safety --

    #[test]
    fn thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let trace = Arc::new(CompilationTrace::new(TraceLevel::Full));
        let num_threads = 8;
        let events_per_thread = 100;

        let mut handles = Vec::new();
        for t in 0..num_threads {
            let trace = Arc::clone(&trace);
            handles.push(thread::spawn(move || {
                for i in 0..events_per_thread {
                    trace.emit(
                        PassId::new(format!("{}", t)),
                        EventKind::Applied {
                            rule: RuleId(i as u32),
                            before: vec![i as u32],
                            after: vec![i as u32 + 1],
                        },
                        vec![i as u32],
                        Justification::CostModel {
                            before: 2.0,
                            after: 1.0,
                        },
                    );
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All events recorded.
        assert_eq!(trace.len(), num_threads * events_per_thread);

        // All sequence numbers are unique.
        let events = trace.events();
        let mut seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
        seqs.sort();
        seqs.dedup();
        assert_eq!(seqs.len(), num_threads * events_per_thread);
    }

    // -- Level accessor --

    #[test]
    fn level_accessor() {
        let trace = CompilationTrace::new(TraceLevel::Debug);
        assert_eq!(trace.level(), TraceLevel::Debug);

        let trace = CompilationTrace::new(TraceLevel::None);
        assert_eq!(trace.level(), TraceLevel::None);
    }

    // -- Clone shares event storage --

    #[test]
    fn clone_shares_events() {
        let trace = CompilationTrace::new(TraceLevel::Full);
        let clone = trace.clone();

        trace.emit(
            PassId::new("0"),
            EventKind::Applied {
                rule: RuleId(0),
                before: vec![0],
                after: vec![1],
            },
            vec![0],
            Justification::CostModel {
                before: 2.0,
                after: 1.0,
            },
        );

        // Clone sees the event because they share the Arc.
        assert_eq!(clone.len(), 1);
    }

    // -- Debug impl --

    #[test]
    fn debug_impl() {
        let trace = CompilationTrace::new(TraceLevel::Summary);
        let debug = format!("{:?}", trace);
        assert!(debug.contains("CompilationTrace"));
        assert!(debug.contains("Summary"));
        assert!(debug.contains("0"));
    }
}

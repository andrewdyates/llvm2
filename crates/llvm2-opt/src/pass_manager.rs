// llvm2-opt - Pass manager framework
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Pass manager framework for running optimization passes on machine functions.
//!
//! The `MachinePass` trait defines the interface for all optimization passes.
//! `PassManager` runs a sequence of passes, optionally iterating to fixed point.
//!
//! # Architecture
//!
//! Each pass operates on a `MachFunction` from `llvm2-ir` and returns `true`
//! if it made any modifications. The pass manager can run passes once or
//! iterate the entire sequence until no pass reports changes (fixed point).
//!
//! ```text
//! PassManager { passes: [DCE, ConstFold, CopyProp, Peephole] }
//!     │
//!     ├── run_once(func)   → runs each pass in order, once
//!     └── run_to_fixpoint(func, max_iters) → repeats until stable
//! ```

use llvm2_ir::MachFunction;

/// A single optimization pass that transforms machine-level IR.
///
/// Passes must be idempotent: running a pass twice on unchanged input
/// should not modify anything on the second run.
pub trait MachinePass {
    /// Human-readable name for diagnostics and logging.
    fn name(&self) -> &str;

    /// Run the pass on a machine function.
    ///
    /// Returns `true` if the function was modified, `false` if unchanged.
    /// A pass returning `true` may enable further optimizations in
    /// subsequent passes.
    fn run(&mut self, func: &mut MachFunction) -> bool;
}

/// Statistics collected during pass execution.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Number of times each pass was run.
    pub runs: Vec<(String, u32)>,
    /// Total number of passes that reported changes.
    pub changes: u32,
    /// Number of fixed-point iterations.
    pub iterations: u32,
}

/// Manages and executes a pipeline of optimization passes.
///
/// Passes are run in insertion order. The pass manager supports both
/// single-run and fixed-point iteration modes.
pub struct PassManager {
    passes: Vec<Box<dyn MachinePass>>,
}

impl PassManager {
    /// Create an empty pass manager.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    /// Add a pass to the end of the pipeline.
    pub fn add_pass(&mut self, pass: Box<dyn MachinePass>) {
        self.passes.push(pass);
    }

    /// Add a pass to the end of the pipeline (builder pattern).
    pub fn with_pass(mut self, pass: Box<dyn MachinePass>) -> Self {
        self.passes.push(pass);
        self
    }

    /// Returns the number of registered passes.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }

    /// Run all passes once in order.
    ///
    /// Returns `true` if any pass modified the function.
    pub fn run_once(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;
        for pass in &mut self.passes {
            if pass.run(func) {
                changed = true;
            }
        }
        changed
    }

    /// Run all passes repeatedly until no pass reports changes, or
    /// `max_iterations` is reached.
    ///
    /// Returns statistics about the run.
    pub fn run_to_fixpoint(
        &mut self,
        func: &mut MachFunction,
        max_iterations: u32,
    ) -> PassStats {
        let mut stats = PassStats {
            runs: self.passes.iter().map(|p| (p.name().to_string(), 0)).collect(),
            changes: 0,
            iterations: 0,
        };

        for iteration in 0..max_iterations {
            stats.iterations = iteration + 1;
            let mut any_changed = false;

            for (i, pass) in self.passes.iter_mut().enumerate() {
                stats.runs[i].1 += 1;
                if pass.run(func) {
                    any_changed = true;
                    stats.changes += 1;
                }
            }

            if !any_changed {
                break;
            }
        }

        stats
    }

    /// Run all passes once, collecting per-pass statistics.
    pub fn run_once_with_stats(&mut self, func: &mut MachFunction) -> PassStats {
        let mut stats = PassStats {
            runs: self.passes.iter().map(|p| (p.name().to_string(), 0)).collect(),
            changes: 0,
            iterations: 1,
        };

        for (i, pass) in self.passes.iter_mut().enumerate() {
            stats.runs[i].1 = 1;
            if pass.run(func) {
                stats.changes += 1;
            }
        }

        stats
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{MachFunction, Signature};

    /// A no-op pass for testing.
    struct NoOpPass;

    impl MachinePass for NoOpPass {
        fn name(&self) -> &str {
            "no-op"
        }
        fn run(&mut self, _func: &mut MachFunction) -> bool {
            false
        }
    }

    /// A pass that reports a change exactly N times, then stops.
    struct CountingPass {
        remaining: u32,
    }

    impl MachinePass for CountingPass {
        fn name(&self) -> &str {
            "counting"
        }
        fn run(&mut self, _func: &mut MachFunction) -> bool {
            if self.remaining > 0 {
                self.remaining -= 1;
                true
            } else {
                false
            }
        }
    }

    fn make_empty_func() -> MachFunction {
        MachFunction::new("test".to_string(), Signature::new(vec![], vec![]))
    }

    #[test]
    fn test_empty_pass_manager() {
        let mut pm = PassManager::new();
        let mut func = make_empty_func();
        assert!(!pm.run_once(&mut func));
    }

    #[test]
    fn test_noop_pass() {
        let mut pm = PassManager::new();
        pm.add_pass(Box::new(NoOpPass));
        let mut func = make_empty_func();
        assert!(!pm.run_once(&mut func));
    }

    #[test]
    fn test_fixpoint_convergence() {
        let mut pm = PassManager::new();
        pm.add_pass(Box::new(CountingPass { remaining: 3 }));
        let mut func = make_empty_func();

        let stats = pm.run_to_fixpoint(&mut func, 10);
        // Should run 4 iterations: 3 with changes + 1 that detects fixpoint
        assert_eq!(stats.iterations, 4);
        assert_eq!(stats.changes, 3);
    }

    #[test]
    fn test_fixpoint_max_iterations() {
        let mut pm = PassManager::new();
        pm.add_pass(Box::new(CountingPass { remaining: 100 }));
        let mut func = make_empty_func();

        let stats = pm.run_to_fixpoint(&mut func, 5);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.changes, 5);
    }

    #[test]
    fn test_builder_pattern() {
        let pm = PassManager::new()
            .with_pass(Box::new(NoOpPass))
            .with_pass(Box::new(NoOpPass));
        assert_eq!(pm.num_passes(), 2);
    }
}

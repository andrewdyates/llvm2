// llvm2-fuzz/src/tmir_gen.rs - Random valid tMIR module generator
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Generates structurally valid tMIR modules from a seed. The generated
// programs are intentionally narrow: one function, integer arithmetic only
// (i64), single entry block with a linear chain of binops, a return of the
// final value. This is the smallest useful shape that exercises ISel,
// optimization, register allocation, and encoding.
//
// "Valid" means: the interpreter can execute it, and the compiler pipeline
// (Pipeline::compile_function) should accept it. Anything else is either a
// generator bug or a compiler bug — both are interesting.
//
// Expansion roadmap (not in MVP):
//   - Multi-block CFG with CondBr
//   - Multiple integer widths (i8/i16/i32/i128)
//   - Loads/stores with alloca
//   - Multiple functions with Call
//   - Float ops

use crate::prng::Prng;
use tmir::{BinOp, Module as TmirModule, Ty};
use tmir_build::ModuleBuilder;

/// Configuration for tMIR random generation.
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Number of i64 parameters to the generated function.
    pub num_params: u32,
    /// Number of binop instructions in the entry block.
    pub num_ops: u32,
    /// Include division/remainder operators (which the driver must then
    /// guard against divide-by-zero in the oracle).
    pub allow_div: bool,
    /// Include shift operators.
    pub allow_shift: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            num_params: 4,
            num_ops: 8,
            // Divisions and remainder are allowed — we handle zero-input
            // tests explicitly in the driver so the interpreter's
            // DivisionByZero error is caught rather than treated as a
            // miscompile.
            allow_div: true,
            allow_shift: true,
        }
    }
}

/// Operators we know the interpreter and the compiler both support.
fn op_pool(cfg: &GenConfig) -> &'static [BinOp] {
    // Narrow static tables depending on config flags. Using match on
    // (allow_div, allow_shift) keeps the returned slice &'static.
    match (cfg.allow_div, cfg.allow_shift) {
        (true, true) => &[
            BinOp::Add, BinOp::Sub, BinOp::Mul,
            BinOp::SDiv, BinOp::UDiv, BinOp::SRem, BinOp::URem,
            BinOp::And, BinOp::Or, BinOp::Xor,
            BinOp::Shl, BinOp::LShr, BinOp::AShr,
        ],
        (true, false) => &[
            BinOp::Add, BinOp::Sub, BinOp::Mul,
            BinOp::SDiv, BinOp::UDiv, BinOp::SRem, BinOp::URem,
            BinOp::And, BinOp::Or, BinOp::Xor,
        ],
        (false, true) => &[
            BinOp::Add, BinOp::Sub, BinOp::Mul,
            BinOp::And, BinOp::Or, BinOp::Xor,
            BinOp::Shl, BinOp::LShr, BinOp::AShr,
        ],
        (false, false) => &[
            BinOp::Add, BinOp::Sub, BinOp::Mul,
            BinOp::And, BinOp::Or, BinOp::Xor,
        ],
    }
}

/// The generated module's function always has this name so the driver can
/// look it up from the interpreter by string.
pub const FUZZ_FN_NAME: &str = "fuzz_fn";

/// Build a random module from a seed. The returned module has exactly one
/// function named [`FUZZ_FN_NAME`] with `cfg.num_params` i64 parameters and
/// a single i64 return.
pub fn gen_module(seed: u64, cfg: &GenConfig) -> TmirModule {
    let mut rng = Prng::new(seed);
    let mut mb = ModuleBuilder::new(format!("fuzz_{}", seed));

    let mut param_tys = Vec::with_capacity(cfg.num_params as usize);
    for _ in 0..cfg.num_params {
        param_tys.push(Ty::I64);
    }
    let ty = mb.add_func_type(param_tys, vec![Ty::I64]);
    let mut fb = mb.function(FUZZ_FN_NAME, ty);

    let entry = fb.create_block();
    // Bind block params.
    let mut values: Vec<_> = Vec::with_capacity((cfg.num_params + cfg.num_ops) as usize);
    for _ in 0..cfg.num_params {
        let v = fb.add_block_param(entry, Ty::I64);
        values.push(v);
    }
    fb.switch_to_block(entry);

    let ops = op_pool(cfg);
    for _ in 0..cfg.num_ops {
        // Pick two existing values as operands (may be the same one twice;
        // that's fine).
        let lhs = values[rng.gen_range_usize(values.len())];
        let rhs = values[rng.gen_range_usize(values.len())];
        let op = ops[rng.gen_range_usize(ops.len())];
        let result = fb.binop(op, Ty::I64, lhs, rhs);
        values.push(result);
    }

    // Return the final computed value.
    let last = *values.last().expect("at least one value exists (params)");
    fb.ret(vec![last]);
    fb.build();

    mb.build()
}

/// Sample test inputs for the generated function. We deliberately include
/// zeros, ones, and a few signed/unsigned extrema in addition to PRNG
/// values — these are the classes most likely to expose
/// corner-case miscompiles (divide-by-zero, INT64_MIN / -1, shift count
/// >= width, etc.).
pub fn sample_inputs(seed: u64, num_params: u32, num_samples: u32) -> Vec<Vec<i64>> {
    let mut rng = Prng::new(seed.wrapping_add(0xDEADBEEF));
    let mut out = Vec::with_capacity(num_samples as usize);

    // Fixed well-known inputs first — deterministic and useful for minimization.
    let well_known: &[i64] = &[0, 1, -1, 2, -2, i64::MAX, i64::MIN, 0xFFFF_FFFF, -0x8000_0000];
    let mut slot = 0usize;
    for _ in 0..num_samples {
        let mut row = Vec::with_capacity(num_params as usize);
        for _ in 0..num_params {
            if slot < well_known.len() {
                // For the very first rows, use well-known values cycling
                // through the table.
                row.push(well_known[slot % well_known.len()]);
                slot += 1;
            } else {
                // After the well-known prefix, use the PRNG.
                row.push(rng.signed_i64(1_000_000_000));
            }
        }
        out.push(row);
    }
    out
}

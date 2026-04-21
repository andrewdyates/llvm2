// llvm2-fuzz/src/prng.rs - Deterministic PRNG for reproducible campaigns
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// SplitMix64: tiny, well-tested, no_std-friendly PRNG. Seed one, get bits.
// Reference: https://xorshift.di.unimi.it/splitmix64.c

/// Simple deterministic PRNG (SplitMix64). One seed, reproducible output.
#[derive(Debug, Clone)]
pub struct Prng {
    state: u64,
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform u32 in `[0, bound)`. Panics if bound == 0.
    pub fn gen_range_u32(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "gen_range_u32: bound must be > 0");
        // Cheap modulo — acceptable bias for small bounds in fuzzing use.
        (self.next_u64() as u32) % bound
    }

    /// Uniform usize in `[0, bound)`. Panics if bound == 0.
    pub fn gen_range_usize(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "gen_range_usize: bound must be > 0");
        (self.next_u64() as usize) % bound
    }

    /// Biased coin flip with probability `p/denom`.
    pub fn chance(&mut self, p: u32, denom: u32) -> bool {
        self.gen_range_u32(denom) < p
    }

    /// i64 in `[-range, range]`.
    pub fn signed_i64(&mut self, range: i64) -> i64 {
        let r = range.unsigned_abs() as u64;
        if r == 0 {
            return 0;
        }
        let v = self.next_u64() % (2 * r + 1);
        (v as i64) - (r as i64)
    }
}

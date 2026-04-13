# LLVM2 Codebase Audit Report

**Date:** 2026-04-13
**Auditor:** Researcher agent
**Scope:** Silent data loss, encoder coverage, test coverage, compilation gaps, correctness risks

## Summary

Filed 14 new issues (including 3 already filed by prior agents). Found 11 net-new issues across 6 categories. The most critical finding is a **duplicate instruction encoder** where two 1700+ line encoders in the same crate produce different machine code for the same opcodes.

## Issues Filed

| Issue | Priority | Category | Title |
|-------|----------|----------|-------|
| #92 | P1 | Duplicate code / miscompilation | Duplicate AArch64 instruction encoder: lower.rs and aarch64/encode.rs diverge |
| #93 | P2 | Miscompilation | lower.rs FP instructions hardcode double precision, ignore single-precision |
| #94 | P2 | Silent data loss | Pipeline operand adapter silently maps FrameIndex/MemOp/Special to Imm(0) |
| #95 | P2 | Missing feature | Symbol operand stub: extract_symbol_name() always returns None |
| #96 | P3 | Code quality | produces_value() duplicated 4x across opt crates |
| #97 | P3 | Silent data loss | Compact unwind silently drops unknown callee-saved register pairs |
| #98 | P2 | Silent data loss | lower.rs preg_hw() silently defaults non-register operands to XZR |
| #99 | P2 | Miscompilation | LdrRI/StrRI in lower.rs always encode 64-bit |
| #100 | P2 | Miscompilation | Immediate shift and TBZ/TBNZ encodings emit NOP in lower.rs |
| #101 | P3 | Test coverage | Low test coverage: greedy.rs, macho/writer.rs, loops.rs |
| #102 | P3 | Silent data loss | adapter.rs type fallbacks silently map unsupported types to I64 |
| #103 | P3 | Scaffolding | Missing RISC-V target definitions |
| #104 | P2 | Silent data loss | expand_pseudos catch-all silently converts unknown opcodes to NOP |
| #105 | P3 | Error masking | FP size helpers default to Double for non-FPR operands |

## Pre-existing Issues (already tracked)

| Issue | Title |
|-------|-------|
| #89 | Pipeline map_isel_opcode catch-all silently drops 20+ ISel opcodes as Nop |
| #90 | Missing Switch and CallIndirect lowering |
| #91 | End-to-end compilation test |
| #73 | Type duplication across crates |
| #71 | Error handling: replace unreachable!/panic with Result types |
| #54 | Audit: unreachable! panics in isel.rs |

## Critical Finding: Duplicate Encoder (#92)

The most impactful finding is that `llvm2-codegen` contains two complete instruction encoders:

1. **aarch64/encode.rs** (2086 lines) - used by `pipeline.rs`, register-class-aware, correct for FP/32-bit
2. **lower.rs** (1719 lines) - used by `lower_function()`, hardcodes 64-bit and double precision

These encoders produce **different machine code for the same opcode**. The lower.rs encoder:
- Hardcodes double precision for all FP operations
- Hardcodes 64-bit for all LDR/STR operations
- Emits NOP for immediate shifts (LSL/LSR/ASR #imm) and TBZ/TBNZ
- Silently maps non-register operands to register 31

The recommended fix is to delete lower.rs::encode_inst() entirely and use aarch64::encode::encode_instruction() everywhere.

## Pattern: Silent Catch-All Cascade

The most pervasive anti-pattern is a cascade of silent catch-alls:

1. `map_isel_opcode` (#89): ISel opcode -> Nop
2. `expand_pseudos` (#104): Unknown pseudo -> Nop
3. `encode_relaxed_instructions`: Nop/pseudo -> skip

An instruction can be silently dropped at any of these three points with no error, warning, or log message. The result is that the instruction simply disappears from the output binary.

## Compilation Gaps

Programs that cannot be compiled correctly today:
- **Switch statements** (#90): No multi-way branch lowering
- **Indirect calls** (#90): No CallIndirect support
- **External functions** (#95): No relocation emission for ADRP+ADD/BL
- **32-bit FP arithmetic** (#93): All FP operations encode as 64-bit through lower.rs
- **Immediate shifts** (#100): LSL/LSR/ASR #imm emit NOP through lower.rs
- **Bit-test branches** (#100): TBZ/TBNZ emit NOP through lower.rs

## Recommendations

1. **P0 urgency**: Fix #92 (duplicate encoder) first -- it is the root cause of #93, #99, #100
2. **Then fix #89**: The silent catch-all in map_isel_opcode is the second-biggest miscompilation risk
3. **Then fix #95**: Without symbol relocations, no external function calls work
4. **Code quality**: Fix #96 and #104 to prevent future silent regressions

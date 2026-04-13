# LLVM2 AArch64 Backend Design

**Date:** 2026-04-12
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Approved
**Reviewed by:** GPT-5.4 (codex, 2 independent reviews)

---

## Overview

Build a production-quality AArch64 compiler backend for macOS (Apple Silicon) that compiles tMIR to optimized machine code. 100% Rust, zero external dependencies for the core backend. Study LLVM source as reference and port the proven algorithms.

**Goals:**
- At least as fast as LLVM in compilation speed and output code quality
- 100% Rust, zero external deps (z4 verification is optional/external)
- AArch64 macOS first, x86-64 and iOS later
- Design for verification from day one (proven correct via z4 SMT)

**Non-goals for MVP:**
- x86-64 or RISC-V targets
- PAC/BTI/arm64e
- Full DWARF debug info (compact unwind only for MVP)

---

## Crate Architecture (6 crates)

```
tmir-* ──> llvm2-lower (instruction selection, ABI lowering)
                |
                v
           llvm2-ir (MachineFunction, MachInst, operands, blocks, stack)
                |
        +-------+-------+-----------+
        v       v       v           v
   llvm2-opt  llvm2-regalloc  llvm2-codegen  llvm2-verify
   (passes)   (liveness+RA)   (encode+Mach-O)  (z4, optional)
```

| Crate | Purpose | Key types |
|-------|---------|-----------|
| `llvm2-ir` | Shared machine model | `MachInst`, `MachFunction`, `MachBlock`, `VReg`, `PReg`, `RegClass`, `Operand`, `StackSlot` |
| `llvm2-lower` | tMIR to MachIR instruction selection | `InstructionSelector`, `AbiClassifier`, `Legalizer` |
| `llvm2-opt` | Optimization passes on MachIR | `PassManager`, DCE, ConstFold, CopyProp, Peephole, AddrMode, CSE |
| `llvm2-regalloc` | Register allocation | `LiveInterval`, `LinearScan`, `SpillGen`, `PhiElim` |
| `llvm2-codegen` | Encoding + Mach-O emission | `AArch64Encoder`, `MachOWriter`, `FixupTable`, `UnwindEmitter` |
| `llvm2-verify` | z4 SMT proofs (optional) | `SmtEncoder`, `LoweringProof`, `Verifier` |

### Why `llvm2-ir` is needed

All crates need the machine IR types but shouldn't depend on lowering logic. Without this shared crate, the 5-crate split creates circular dependency problems. This owns: MachineFunction, blocks, operands, register classes, stack slots, frame indices, memory operands, relocation kinds, and pseudo-instructions.

---

## Machine IR Design

### MachInst: Generic container, target-specific opcodes

```rust
pub struct MachInst {
    pub opcode: AArch64Opcode,
    pub operands: Vec<MachOperand>,
    pub implicit_defs: &'static [PReg],
    pub implicit_uses: &'static [PReg],
    pub flags: InstFlags,
}

bitflags! {
    pub struct InstFlags: u16 {
        const IS_CALL       = 0x01;
        const IS_BRANCH     = 0x02;
        const IS_RETURN     = 0x04;
        const IS_TERMINATOR = 0x08;
        const HAS_SIDE_EFFECTS = 0x10;
        const IS_PSEUDO     = 0x20;
        const READS_MEMORY  = 0x40;
        const WRITES_MEMORY = 0x80;
    }
}
```

### Register Model

```rust
pub enum RegClass {
    Gpr32,   // W0-W30
    Gpr64,   // X0-X30
    Fpr32,   // S0-S31
    Fpr64,   // D0-D31
    Vec128,  // V0-V31 (SIMD)
}

pub struct VReg { pub id: u32, pub class: RegClass }
pub struct PReg(pub u8);  // 0-30: GPR, 32-63: FPR

pub enum MachOperand {
    VReg(VReg),
    PReg(PReg),
    Imm(i64),
    FImm(f64),
    Block(BlockId),
    StackSlot(StackSlotId),
    FrameIndex(FrameIdx),
    MemOp { base: Box<MachOperand>, offset: i64 },
    Special(SpecialReg),
}

pub enum SpecialReg { SP, XZR, WZR }
```

### Storage

- Arena-based `Vec<MachInst>` indexed by `InstId(u32)`
- Blocks: `Vec<BlockId>` (ordered) + `Vec<MachBlock>` by index
- No HashMap for blocks/values -- Vec+index only
- Typed index wrappers: `InstId(u32)`, `BlockId(u32)`, `VRegId(u32)`

---

## Instruction Selection

### Two-phase approach

1. **ISel (SSA tree-pattern matching):** Walk tMIR blocks in RPO, match patterns bottom-up, emit AArch64 MachInst with VRegs. Covers simple addressing (base+imm, base+reg), arithmetic, branches, calls.

2. **Late combines (machine pass):** Address-mode formation for pre/post-index, cmp+branch fusing, csel/cset formation. These patterns depend on one-use analysis and writeback legality that don't fit tree matching.

### Legalization (before/during isel)

| tMIR pattern | Legalization |
|-------------|-------------|
| Large immediate (>12-bit) | `movz`/`movk` sequence |
| Large stack offset | `add` + base register |
| Global address | `adrp` + `add` |
| Unsupported integer width | extension/truncation |
| Aggregate operation | multiple load/stores |

### Apple AArch64 Calling Convention

| Register | Role | Callee-saved? |
|----------|------|---------------|
| X0-X7 | Integer args/returns | No |
| X8 | Indirect result pointer | No |
| X9-X15 | Temporaries | No |
| X16-X17 | IP scratch | No |
| X18 | RESERVED (Apple) | -- |
| X19-X28 | Callee-saved | Yes |
| X29 | Frame pointer (mandatory on Darwin) | Yes |
| X30 | Link register | Yes |
| V0-V7 | FP/SIMD args/returns | No |
| V8-V15 | Lower 64 bits callee-saved | Partial |
| V16-V31 | Temporaries | No |

Additional rules:
- 16-byte stack alignment
- Variadic args on stack (Apple differs from AAPCS)
- Aggregates >16 bytes via indirect (X8)
- HFA (Homogeneous Floating-point Aggregates) in V registers
- Red zone: optional optimization (128 bytes, disabled by default)

---

## Register Allocation

### Phase 1: Linear Scan (bring-up)

Required features (bare linear scan is insufficient):
- Live interval computation
- Interval splitting
- Spill weights with loop-depth scaling
- Call-aware allocation (call-clobber holes)
- Rematerialization for constants and addresses
- Spill-slot reuse

### Phase 2: LLVM-style Greedy Allocator (quality)

### Phi Lowering (critical correctness requirement)

1. Critical-edge splitting (before phi elimination)
2. Parallel-copy resolver (NOT sequential copies -- causes wrong-code)
3. Copy coalescing after phi lowering and after spill rewrite

---

## AArch64 Encoding

Fixed 32-bit, per-format encoder functions. ~15 major encoding formats.

### Known traps (from codex review)

| Trap | Detail |
|------|--------|
| SP vs XZR/WZR | Register 31 = SP or ZR depending on instruction |
| Implicit zero-extension | 32-bit writes zero upper 32 bits |
| Scaled vs unscaled offsets | LDR/STR encode differently |
| LDP/STP offset limits | Signed scaled offsets, restricted range |
| ADRP page semantics | 4KB page alignment, two-relocation pairs |
| Branch reach | B: +/-128MB, B.cond: +/-1MB |

Defer PAC/BTI/arm64e but design for future addition.

---

## Mach-O Object File Emission

### Structure

```
mach_header_64 (magic=0xFEEDFACF, cputype=ARM64, MH_OBJECT)
  MH_SUBSECTIONS_VIA_SYMBOLS flag (required for Apple linker)
LC_SEGMENT_64
  __TEXT,__text (code)
  __TEXT,__unwind_info (compact unwind)
  __DATA,__data (initialized data)
  __DATA,__const (constant data)
LC_SYMTAB
LC_DYSYMTAB
LC_BUILD_VERSION (macOS, arm64)
```

### Relocations

| Type | Usage |
|------|-------|
| BRANCH26 | B, BL instructions |
| PAGE21 | ADRP (page-aligned address) |
| PAGEOFF12 | ADD, LDR (page offset) |
| GOT_LOAD_PAGE21 | External symbol via GOT |
| GOT_LOAD_PAGEOFF12 | GOT offset |
| UNSIGNED | Absolute pointer data |
| TLV (future) | Thread-local variables |

### Fixup Layer

Encode instructions late from fixup records, not directly from selected ops. Allows fixup application after final layout when branch offsets are known.

### Unwind Metadata (REQUIRED)

- Compact unwind: happy path for standard frame layouts (paired callee-saved saves)
- DWARF CFI fallback: when frame shape is not compact-unwind-encodable
- Part of frame lowering, not a later add-on

---

## Optimization Passes

### Pre-Register-Allocation

| Order | Pass | Notes |
|-------|------|-------|
| 1 | Critical-edge splitting | Required before phi lowering |
| 2 | Legalization | Large immediates, unsupported types |
| 3 | DCE | Remove unreachable blocks, dead instructions |
| 4 | Constant folding | Evaluate constant expressions at compile time |
| 5 | Constant materialization | Optimize movz/movk sequences |
| 6 | Copy propagation | Eliminate redundant MOVs |
| 7 | Peephole | Target-specific combines |
| 8 | Address-mode formation | base+offset, pre/post-index |
| 9 | Compare/select combines | cmp+branch, csel/cset |
| 10 | CSE | Requires memory-effects model |
| 11 | LICM | Requires memory-effects model |
| 12 | CFG simplification | Branch folding |
| 13 | Phi elimination | Parallel-copy resolver |

### Post-Register-Allocation

| Order | Pass | Notes |
|-------|------|-------|
| 1 | Copy coalescing | After spill rewrite |
| 2 | Prologue/epilogue | With unwind metadata |
| 3 | Frame index elimination | Resolve frame indices to SP offsets |
| 4 | Post-RA peephole | Target-specific cleanup |
| 5 | Block layout | Minimize taken branches |
| 6 | Branch relaxation | Late range expansion |

### Memory-Effects Model

Required for CSE and LICM safety:
- Pure: no memory access
- Load: reads memory
- Store: writes memory
- Call: clobbers everything unless proved otherwise
- tMIR proof annotations refine aliasing

### Proof-Enabled Optimizations (unique to LLVM2)

| Proof | Optimization |
|-------|-------------|
| NoOverflow | Eliminate overflow checks, unchecked arithmetic |
| InBounds | Eliminate bounds checks |
| NotNull | Eliminate null checks |
| ValidBorrow | Enable load/store reordering |
| PositiveRefCount | Eliminate redundant retain/release |

---

## Frame Lowering

Real slot/layout model:
- **Spill slots**: allocated by regalloc, alignment constraints
- **Local variable slots**: alloca and local aggregates
- **Outgoing argument space**: stack-passed call arguments
- **Callee-saved register area**: paired saves for compact unwind
- **Frame pointer**: required on Darwin arm64 (X29 must be valid)
- **Red zone**: optional optimization, disabled by default
- **Dynamic alloca**: requires frame pointer

---

## Verification Architecture (z4, later phase)

Optional at compile time (Cargo feature flag). Out of hot compile path.

Every lowering rule `tMIR_inst -> Vec<AArch64Inst>` is a verifiable unit:
1. Encode tMIR semantics as SMT formula (from tmir-semantics)
2. Encode AArch64 semantics as SMT formula
3. Prove: for all inputs, tmir_semantics(inputs) = mach_semantics(inputs)

Each peephole optimization is also proven correct.

---

## Build Order

1. Shared machine model (`llvm2-ir`)
2. AArch64 encoder + Mach-O .o writer + relocation tests
3. ABI/call lowering + minimal isel for straight-line integer code
4. Liveness + phi lowering + linear scan + spill/rewrite
5. Prologue/epilogue + frame index elimination + unwind
6. Address-mode formation + peepholes + block layout
7. Branch-range expansion after final layout
8. Proof-aware opts and SMT validation last

---

## LLVM Source Reference

Reference implementation at `~/llvm-project-ref/`:

| Area | LLVM Source | What to study |
|------|------------|---------------|
| AArch64 backend | `llvm/lib/Target/AArch64/` | Instruction defs, encoding, calling convention |
| Machine IR | `llvm/include/llvm/CodeGen/MachineInstr.h` | MachInst data model |
| Register allocation | `llvm/lib/CodeGen/RegAllocGreedy.cpp` | Greedy RA algorithm |
| Liveness | `llvm/lib/CodeGen/LiveIntervals.cpp` | Live interval computation |
| Phi elimination | `llvm/lib/CodeGen/PHIElimination.cpp` | Critical-edge splitting |
| Frame lowering | `llvm/lib/Target/AArch64/AArch64FrameLowering.cpp` | Prologue/epilogue, unwind |
| Encoding | `llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCCodeEmitter.cpp` | Binary encoding |
| Mach-O | `llvm/lib/MC/MachObjectWriter.cpp` | Object file emission |
| Relocations | `llvm/lib/Target/AArch64/MCTargetDesc/AArch64MachObjectWriter.cpp` | ARM64 relocations |

---

## Testing Strategy

| Level | Method |
|-------|--------|
| Encoding | Every encoder vs `llvm-mc --show-encoding` and ARM ARM |
| Mach-O | Every .o vs `otool -l`, `otool -tv`, link with `ld` |
| Correctness | Compile + run suite covering each tMIR instruction |
| Performance | Compile-time benchmarks + runtime vs LLVM `-O2` |
| Reference | Compare output against `clang` for identical programs |

---

## Codex Review Summary

Two independent reviews from GPT-5.4 identified 15 critical gaps, all incorporated:

1. Add shared machine-model crate (`llvm2-ir`)
2. Wrap target-specific opcodes in generic MachInst container
3. VReg needs explicit register classes and widths
4. Red zone is optional optimization, not baseline
5. z4 verification is external, make optional
6. SSA tree matching + late AddrModeForm pass
7. Need legalization layer
8. Phi lowering needs critical-edge splitting + parallel copies
9. LICM/CSE unsafe without memory-effects model
10. Mach-O needs GOT/TLV relocs, fixup layer, compact unwind
11. Unwind metadata is REQUIRED on macOS
12. Frame lowering needs real slot/layout model
13. ABI needs FP/SIMD, varargs, aggregates, HFA, sret
14. Encoding traps: SP vs XZR, zero-extension, scaled offsets
15. Build order: machine model -> encoder -> isel -> regalloc -> unwind -> opts -> proofs
